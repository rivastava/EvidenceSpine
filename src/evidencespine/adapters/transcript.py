from __future__ import annotations

from typing import Any, Iterable, List

from evidencespine.adapters.base import (
    AdapterIngestResult,
    NormalizedTranscriptMessage,
    TranscriptAdapterConfig,
)
from evidencespine.protocol import (
    merge_evidence_refs,
    normalize_evidence_items,
    normalize_state_context,
    safe_float,
    safe_text,
)
from evidencespine.runtime import AgentMemoryRuntime

_CONTENT_KEYS = {"content", "text", "evidence_ref", "evidence_refs", "evidence_items", "state_context"}


def _flatten_content(value: Any, limit: int) -> str:
    """Flatten a message body into text.

    Handles plain strings, LangChain/OpenAI content-part lists such as
    [{"type": "text", "text": "hello"}], nested dicts, and arbitrary objects
    instead of stringifying the Python repr.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return safe_text(value, "", limit)
    if isinstance(value, dict):
        for key in ("text", "content", "value", "output"):
            inner = value.get(key)
            if isinstance(inner, str):
                return safe_text(inner, "", limit)
            if isinstance(inner, (list, tuple, dict)):
                flattened = _flatten_content(inner, limit)
                if flattened:
                    return flattened
        return ""
    if isinstance(value, (list, tuple)):
        parts: List[str] = []
        total = 0
        for part in value:
            text = _flatten_content(part, limit)
            if not text:
                continue
            parts.append(text)
            total += len(text)
            if total >= limit:
                break
        return safe_text(" ".join(parts), "", limit)
    return safe_text(value, "", limit)


def _message_pair(value: Any) -> tuple[str, Any, dict[str, Any]] | None:
    """Recognize (role, content[, metadata]) tuple/list messages."""
    if isinstance(value, (tuple, list)) and len(value) >= 2 and isinstance(value[0], str):
        meta = value[2] if len(value) > 2 and isinstance(value[2], dict) else {}
        return value[0], value[1], dict(meta)
    return None


def _iter_messages(messages_or_state: Any) -> Iterable[Any]:
    """Yield raw messages from a message list or a state carrying messages.

    Raises ValueError for unrecognized shapes so ingest_* surfaces an explicit
    invalid/unsupported result instead of silently ingesting nothing.
    """
    if isinstance(messages_or_state, dict):
        if "messages" not in messages_or_state:
            raise ValueError("unsupported transcript state: expected a 'messages' list")
        messages = messages_or_state.get("messages")
        if isinstance(messages, (list, tuple)):
            yield from messages
            return
        raise ValueError("unsupported transcript state: 'messages' must be a list or tuple")
    if isinstance(messages_or_state, (list, tuple)):
        yield from messages_or_state
        return
    messages_attr = getattr(messages_or_state, "messages", None)
    if isinstance(messages_attr, (list, tuple)):
        yield from messages_attr
        return
    raise ValueError(f"unsupported transcript input: {type(messages_or_state).__name__}")


class TranscriptAdapter:
    def __init__(
        self,
        runtime: AgentMemoryRuntime,
        config: TranscriptAdapterConfig | None = None,
        *,
        default_thread_id: str = "transcript_default",
        source_agent_id: str = "transcript",
    ) -> None:
        self.runtime = runtime
        self.config = config or self.default_config()
        self.default_thread_id = safe_text(default_thread_id, "transcript_default", 128)
        self.source_agent_id = safe_text(source_agent_id, "transcript", 128)

    @staticmethod
    def default_config() -> TranscriptAdapterConfig:
        return TranscriptAdapterConfig(
            role_aliases={
                "human": "user",
                "system": "user",
                "ai": "assistant",
                "agent": "assistant",
                "function": "tool",
            },
            event_type_by_role={
                "user": "intent",
                "assistant": "decision",
                "tool": "outcome",
            },
            default_confidence=0.6,
            default_salience=0.5,
            evidence_prefix="transcript",
            content_limit=4096,
            turn_id_prefix="msg",
            skip_empty=True,
        )

    def _canonical_role(self, role: Any) -> str:
        role_text = safe_text(role, "unknown", 64).lower()
        role_text = self.config.role_aliases.get(role_text, role_text)
        if role_text in self.config.user_role_names:
            return "user"
        if role_text in self.config.assistant_role_names:
            return "assistant"
        if role_text in self.config.tool_role_names:
            return "tool"
        return role_text or "unknown"

    def _extract_role(self, msg: Any) -> str:
        if isinstance(msg, dict):
            role = msg.get("role", msg.get("source", msg.get("name", msg.get("type", "unknown"))))
        else:
            role = getattr(
                msg,
                "role",
                getattr(msg, "source", getattr(msg, "name", getattr(msg, "type", "unknown"))),
            )
        return self._canonical_role(role)

    def _extract_content(self, msg: Any) -> str:
        limit = int(max(16, self.config.content_limit))
        if isinstance(msg, dict):
            content = msg.get("content")
            if content is None:
                content = msg.get("text")
            if content is None:
                content = msg.get("message")
        else:
            content = getattr(msg, "content", None)
            if content is None:
                content = getattr(msg, "text", None)
            if content is None:
                content = getattr(msg, "message", None)
        return _flatten_content(content, limit)

    def _extract_turn_id(self, msg: Any, idx: int) -> str:
        fallback = f"{self.config.turn_id_prefix}_{idx}"
        if isinstance(msg, dict):
            turn_id = msg.get("id", msg.get("message_id", msg.get("turn_id", fallback)))
        else:
            turn_id = getattr(msg, "id", getattr(msg, "message_id", getattr(msg, "turn_id", fallback)))
        return safe_text(turn_id, fallback, 128)

    def _extract_metadata(self, msg: Any, *, canonical_role: str) -> dict[str, Any]:
        if isinstance(msg, NormalizedTranscriptMessage):
            raw = dict(msg.metadata or {})
            raw["normalized_role"] = canonical_role
            return raw
        if isinstance(msg, dict):
            raw = {k: v for k, v in msg.items() if k not in _CONTENT_KEYS}
        else:
            raw = {
                "raw_type": type(msg).__name__,
                "name": safe_text(getattr(msg, "name", ""), "", 128),
                "source": safe_text(getattr(msg, "source", ""), "", 128),
            }
        agent_name = raw.get("name") or raw.get("source")
        if isinstance(agent_name, str) and agent_name.strip():
            raw["agent_name"] = safe_text(agent_name, "", 128)
        raw["normalized_role"] = canonical_role
        return raw

    def _event_type_for_role(self, role: str) -> str:
        role_key = safe_text(role, "unknown", 64).lower()
        return self.config.event_type_by_role.get(role_key, "reflection")

    def _extract_evidence_refs(self, msg: Any, *, default_ref: str) -> list[str]:
        if isinstance(msg, dict):
            refs = msg.get("evidence_refs", msg.get("evidence_ref", []))
        else:
            refs = getattr(msg, "evidence_refs", getattr(msg, "evidence_ref", []))
        merged = merge_evidence_refs(refs)
        if merged:
            return merged
        return [default_ref]

    def _extract_evidence_items(self, msg: Any) -> list[dict[str, Any]]:
        if isinstance(msg, dict):
            raw = msg.get("evidence_items", [])
        else:
            raw = getattr(msg, "evidence_items", [])
        return normalize_evidence_items(raw)

    def _extract_state_context(self, msg: Any) -> dict[str, Any] | None:
        if isinstance(msg, dict):
            if "state_context" not in msg:
                return None
            normalized = normalize_state_context(msg.get("state_context"))
        else:
            if not hasattr(msg, "state_context"):
                return None
            normalized = normalize_state_context(getattr(msg, "state_context"))
        return (normalized or None)

    def _normalized_from_pair(
        self,
        idx: int,
        role_raw: Any,
        content_raw: Any,
        extra_meta: dict[str, Any],
    ) -> NormalizedTranscriptMessage | None:
        role = self._canonical_role(role_raw)
        content = _flatten_content(content_raw, int(max(16, self.config.content_limit)))
        if self.config.skip_empty and not content:
            return None
        turn_id = safe_text(
            extra_meta.get("id", extra_meta.get("message_id", f"{self.config.turn_id_prefix}_{idx}")),
            f"{self.config.turn_id_prefix}_{idx}",
            128,
        )
        evidence_items = normalize_evidence_items(extra_meta.get("evidence_items", []))
        evidence_refs = merge_evidence_refs(
            extra_meta.get("evidence_refs", extra_meta.get("evidence_ref", [])),
            evidence_items,
        )
        metadata = {k: v for k, v in extra_meta.items() if k not in _CONTENT_KEYS}
        metadata["normalized_role"] = role
        return NormalizedTranscriptMessage(
            role=role,
            event_type=self._event_type_for_role(role),
            content=content,
            turn_id=turn_id,
            evidence_ref=f"{self.config.evidence_prefix}:{turn_id}",
            confidence=safe_float(self.config.default_confidence, 0.6, 0.0, 1.0),
            salience=safe_float(self.config.default_salience, 0.5, 0.0, 1.0),
            evidence_refs=evidence_refs or [f"{self.config.evidence_prefix}:{turn_id}"],
            evidence_items=evidence_items,
            state_context=(
                normalize_state_context(extra_meta.get("state_context"))
                if extra_meta.get("state_context") is not None
                else None
            ),
            metadata=metadata,
        )

    def normalize_messages(self, messages_or_state: Any) -> List[NormalizedTranscriptMessage]:
        normalized: List[NormalizedTranscriptMessage] = []
        for idx, msg in enumerate(_iter_messages(messages_or_state)):
            if isinstance(msg, NormalizedTranscriptMessage):
                normalized.append(
                    NormalizedTranscriptMessage(
                        role=safe_text(msg.role, "unknown", 64),
                        event_type=safe_text(msg.event_type, "reflection", 64),
                        content=safe_text(msg.content, "", int(max(16, self.config.content_limit))),
                        turn_id=safe_text(msg.turn_id, f"{self.config.turn_id_prefix}_{idx}", 128),
                        evidence_ref=safe_text(msg.evidence_ref, f"{self.config.evidence_prefix}:{idx}", 256),
                        confidence=safe_float(msg.confidence, self.config.default_confidence, 0.0, 1.0),
                        salience=safe_float(msg.salience, self.config.default_salience, 0.0, 1.0),
                        evidence_refs=merge_evidence_refs(msg.evidence_refs or [msg.evidence_ref], msg.evidence_items),
                        evidence_items=normalize_evidence_items(msg.evidence_items),
                        state_context=(normalize_state_context(msg.state_context) if msg.state_context is not None else None),
                        metadata=dict(msg.metadata or {}),
                    )
                )
                continue
            pair = _message_pair(msg)
            if pair is not None:
                row = self._normalized_from_pair(idx, *pair)
                if row is not None:
                    normalized.append(row)
                continue
            role = self._extract_role(msg)
            content = self._extract_content(msg)
            if self.config.skip_empty and not content:
                continue
            turn_id = self._extract_turn_id(msg, idx)
            normalized.append(
                NormalizedTranscriptMessage(
                    role=role,
                    event_type=self._event_type_for_role(role),
                    content=content,
                    turn_id=turn_id,
                    evidence_ref=f"{self.config.evidence_prefix}:{turn_id}",
                    confidence=safe_float(self.config.default_confidence, 0.6, 0.0, 1.0),
                    salience=safe_float(self.config.default_salience, 0.5, 0.0, 1.0),
                    evidence_refs=self._extract_evidence_refs(msg, default_ref=f"{self.config.evidence_prefix}:{turn_id}"),
                    evidence_items=self._extract_evidence_items(msg),
                    state_context=self._extract_state_context(msg),
                    metadata=self._extract_metadata(msg, canonical_role=role),
                )
            )
        return normalized

    def _ingest_normalized(self, normalized: List[NormalizedTranscriptMessage], *, tid: str) -> tuple[int, int, int, int, str]:
        ingested = 0
        deduped = 0
        disabled = 0
        failed = 0
        reason = ""
        for row in normalized:
            payload = {
                "thread_id": tid,
                "event_type": row.event_type,
                "role": row.role,
                "source_agent_id": self.source_agent_id,
                "source_turn_id": row.turn_id,
                "payload": {
                    "claim": row.content,
                    "fact_state": "asserted",
                },
                "evidence_refs": merge_evidence_refs(row.evidence_refs or [row.evidence_ref], row.evidence_items),
                "evidence_items": normalize_evidence_items(row.evidence_items),
                "state_context": (normalize_state_context(row.state_context) if row.state_context is not None else None),
                "confidence": row.confidence,
                "salience": row.salience,
                "metadata": dict(row.metadata),
            }
            try:
                out = self.runtime.ingest_event(payload)
            except Exception as exc:  # pragma: no cover - runtime is fail-open
                failed += 1
                reason = reason or f"ingest_error:{type(exc).__name__}"
                continue
            status = safe_text(out.get("status", ""), "", 32).lower()
            if status == "ok":
                ingested += 1
            elif status == "deduped":
                deduped += 1
            elif status == "disabled":
                disabled += 1
                reason = reason or safe_text(out.get("reason"), "agent_memory_disabled", 128)
            else:
                failed += 1
                reason = reason or f"ingest_status:{status or 'unknown'}"
        return ingested, deduped, disabled, failed, reason

    def ingest_messages(self, messages_or_state: Any, *, thread_id: str | None = None) -> dict[str, Any]:
        tid = safe_text(thread_id or self.default_thread_id, self.default_thread_id, 128)
        try:
            raw_messages = list(_iter_messages(messages_or_state))
        except ValueError as exc:
            result = AdapterIngestResult(
                status="invalid",
                thread_id=tid,
                seen=0,
                normalized=0,
                ingested=0,
                failed=0,
                skipped=0,
                reason=str(exc),
            )
            return result.to_dict()
        seen = len(raw_messages)
        try:
            normalized = self.normalize_messages(raw_messages)
        except ValueError as exc:
            result = AdapterIngestResult(
                status="invalid",
                thread_id=tid,
                seen=seen,
                normalized=0,
                ingested=0,
                failed=0,
                skipped=0,
                reason=str(exc),
            )
            return result.to_dict()
        ingested, deduped, disabled, failed, reason = self._ingest_normalized(normalized, tid=tid)

        if not normalized:
            status = "ok"
        elif disabled and disabled == len(normalized):
            status = "disabled"
        elif failed > 0 and (ingested + deduped) == 0:
            status = "failed"
        elif failed > 0 or disabled > 0:
            status = "partial"
        else:
            status = "ok"
        if status == "partial" and not reason:
            reason = f"{failed or disabled} of {len(normalized)} message(s) not captured"
        result = AdapterIngestResult(
            status=status,
            thread_id=tid,
            seen=seen,
            normalized=len(normalized),
            ingested=ingested,
            failed=failed,
            skipped=max(0, seen - len(normalized)),
            deduped=deduped,
            disabled=disabled,
            reason=reason if status in {"disabled", "failed", "partial"} else "",
        )
        return result.to_dict()

    def brief(self, query: str, *, thread_id: str | None = None) -> dict[str, Any]:
        tid = safe_text(thread_id or self.default_thread_id, self.default_thread_id, 128)
        return self.runtime.build_brief(tid, query).to_dict()

    def handoff(self, role: str, scope: str, *, thread_id: str | None = None) -> dict[str, Any]:
        tid = safe_text(thread_id or self.default_thread_id, self.default_thread_id, 128)
        return self.runtime.emit_handoff(role=role, thread_id=tid, scope=scope).to_dict()
