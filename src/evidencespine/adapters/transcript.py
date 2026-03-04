from __future__ import annotations

from typing import Any, Iterable, List

from evidencespine.adapters.base import (
    AdapterIngestResult,
    NormalizedTranscriptMessage,
    TranscriptAdapterConfig,
)
from evidencespine.protocol import safe_float, safe_text
from evidencespine.runtime import AgentMemoryRuntime


def _iter_messages(messages_or_state: Any) -> Iterable[Any]:
    if isinstance(messages_or_state, dict):
        messages = messages_or_state.get("messages")
        if isinstance(messages, list):
            for item in messages:
                yield item
        return
    if isinstance(messages_or_state, list):
        for item in messages_or_state:
            yield item


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

    def _extract_role(self, msg: Any) -> str:
        if isinstance(msg, dict):
            role = msg.get("role", msg.get("source", msg.get("name", msg.get("type", "unknown"))))
        else:
            role = getattr(
                msg,
                "role",
                getattr(msg, "source", getattr(msg, "name", getattr(msg, "type", "unknown"))),
            )
        role_text = safe_text(role, "unknown", 64).lower()
        role_text = self.config.role_aliases.get(role_text, role_text)
        if role_text in self.config.user_role_names:
            return "user"
        if role_text in self.config.assistant_role_names:
            return "assistant"
        if role_text in self.config.tool_role_names:
            return "tool"
        return role_text or "unknown"

    def _extract_content(self, msg: Any) -> str:
        if isinstance(msg, dict):
            content = msg.get("content", msg.get("text", ""))
        else:
            content = getattr(msg, "content", getattr(msg, "text", ""))
        return safe_text(content, "", int(max(16, self.config.content_limit)))

    def _extract_turn_id(self, msg: Any, idx: int) -> str:
        fallback = f"{self.config.turn_id_prefix}_{idx}"
        if isinstance(msg, dict):
            turn_id = msg.get("id", msg.get("message_id", msg.get("turn_id", fallback)))
        else:
            turn_id = getattr(msg, "id", getattr(msg, "message_id", getattr(msg, "turn_id", fallback)))
        return safe_text(turn_id, fallback, 128)

    def _extract_metadata(self, msg: Any, *, canonical_role: str) -> dict[str, Any]:
        if isinstance(msg, dict):
            raw = {k: v for k, v in msg.items() if k not in {"content", "text"}}
        else:
            raw = {
                "raw_type": type(msg).__name__,
                "name": safe_text(getattr(msg, "name", ""), "", 128),
                "source": safe_text(getattr(msg, "source", ""), "", 128),
            }
        raw["normalized_role"] = canonical_role
        return raw

    def _event_type_for_role(self, role: str) -> str:
        role_key = safe_text(role, "unknown", 64).lower()
        return self.config.event_type_by_role.get(role_key, "reflection")

    def normalize_messages(self, messages_or_state: Any) -> List[NormalizedTranscriptMessage]:
        normalized: List[NormalizedTranscriptMessage] = []
        for idx, msg in enumerate(_iter_messages(messages_or_state)):
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
                    metadata=self._extract_metadata(msg, canonical_role=role),
                )
            )
        return normalized

    def ingest_messages(self, messages_or_state: Any, *, thread_id: str | None = None) -> dict[str, Any]:
        tid = safe_text(thread_id or self.default_thread_id, self.default_thread_id, 128)
        seen = len(list(_iter_messages(messages_or_state)))
        normalized = self.normalize_messages(messages_or_state)
        ingested = 0
        failed = 0
        for row in normalized:
            out = self.runtime.ingest_event(
                {
                    "thread_id": tid,
                    "event_type": row.event_type,
                    "role": row.role,
                    "source_agent_id": self.source_agent_id,
                    "source_turn_id": row.turn_id,
                    "payload": {
                        "claim": row.content,
                        "fact_state": "asserted",
                    },
                    "evidence_refs": [row.evidence_ref],
                    "confidence": row.confidence,
                    "salience": row.salience,
                    "metadata": dict(row.metadata),
                }
            )
            if out.get("status") in {"ok", "deduped"}:
                ingested += 1
            else:
                failed += 1
        result = AdapterIngestResult(
            status="ok",
            thread_id=tid,
            seen=seen,
            normalized=len(normalized),
            ingested=ingested,
            failed=failed,
            skipped=max(0, seen - len(normalized)),
        )
        return result.to_dict()

    def brief(self, query: str, *, thread_id: str | None = None) -> dict[str, Any]:
        tid = safe_text(thread_id or self.default_thread_id, self.default_thread_id, 128)
        return self.runtime.build_brief(tid, query).to_dict()

    def handoff(self, role: str, scope: str, *, thread_id: str | None = None) -> dict[str, Any]:
        tid = safe_text(thread_id or self.default_thread_id, self.default_thread_id, 128)
        return self.runtime.emit_handoff(role=role, thread_id=tid, scope=scope).to_dict()
