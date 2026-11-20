from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from evidencespine.protocol import (
    CITATION_LINE_MAX,
    AgentControlView,
    AgentConversationBrief,
    AgentHandoffPacket,
    AgentMemoryEvent,
    AgentMemoryFact,
    ClaimCitation,
    ControlViewRow,
    control_row_sort_ts,
    evidence_item_excerpt_matches_checksum,
    event_to_fact_candidates,
    freshness_state_for_context,
    has_grounded_span,
    is_grounded_claim,
    lease_state_for_context,
    merge_evidence_refs,
    normalize_evidence_items,
    normalize_refs,
    normalize_state_context,
    normalize_verification,
    parse_ts_value,
    refresh_event_identity,
    safe_int,
    strip_inline_refs,
    safe_text,
    utc_now_iso,
    validate_event_dict,
    validate_handoff_dict,
)
from evidencespine.retriever import AgentMemoryRetriever, AgentMemoryRetrieverConfig
from evidencespine.store import AgentMemoryStore, AgentMemoryStoreConfig
from evidencespine.vector_backends import (
    FastEmbedVectorBackend,
    HashingVectorBackend,
    VectorBackend,
)


CONTROL_VIEWS = ("active_scopes", "my_work", "open_gates", "stale_claims", "contradictions")
_CONTROL_VIEW_ALIASES: Dict[str, str] = {name: name for name in CONTROL_VIEWS}
_CONTROL_VIEW_ALIASES.update({name.replace("_", "-"): name for name in CONTROL_VIEWS})


def normalize_control_view(view: Any) -> str:
    """Map a requested view name (documented or hyphen alias) to canonical form.

    Returns an empty string for unknown views so callers can fail open with a
    structured invalid result instead of silently falling back to active_scopes.
    """
    requested = safe_text(view, "", 64).strip().lower()
    return _CONTROL_VIEW_ALIASES.get(requested, "")


@dataclass
class AgentMemoryRuntimeConfig:
    enabled: bool = True
    fail_open: bool = True
    max_event_tail: int = 4000
    redaction_enable: bool = True
    dedupe_window_sec: float = 7200.0
    storage_format: str = "sqlite"  # sqlite | jsonl
    db_path: str = ".evidencespine/evidencespine.db"
    events_path: str = ".evidencespine/events.jsonl"
    facts_path: str = ".evidencespine/facts.jsonl"
    state_path: str = ".evidencespine/state.json"
    briefs_dir: str = ".evidencespine/briefs"
    handoffs_dir: str = ".evidencespine/handoffs"
    brief_max_tokens: int = 1800
    brief_top_k_events: int = 32
    brief_top_k_facts: int = 24
    brief_recency_half_life_hours: float = 8.0
    brief_lookback_hours: float = 168.0
    state_flush_interval_sec: float = 1.0
    retrieval_mode: str = "lexical"  # lexical | hybrid | vector
    retrieval_lexical_weight: float = 1.0
    retrieval_vector_weight: float = 0.35
    embedding_backend: str = "auto"  # auto | hashing | fastembed
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    control_view_lookback_hours: float = 168.0
    verified_requires_span: bool = True


@dataclass
class RuntimeHooks:
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None
    on_brief: Optional[Callable[[Dict[str, Any]], None]] = None
    on_handoff: Optional[Callable[[Dict[str, Any]], None]] = None
    contradiction_pass: Optional[Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]]] = None
    reconcile_state: Optional[Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]]] = None


def _cap_claim_line(value: Any) -> str:
    """Cap a brief claim line exactly like on-disk brief serialization does."""
    return safe_text(value, "", CITATION_LINE_MAX)


def _brief_sections(brief: AgentConversationBrief | Dict[str, Any]) -> List[List[str]]:
    payload = brief.to_dict() if isinstance(brief, AgentConversationBrief) else dict(brief or {})
    sections: List[List[str]] = []
    for key in (
        "current_goal",
        "locked_decisions",
        "recent_verified_facts",
        "active_risks",
        "open_items",
        "next_actions",
    ):
        rows = payload.get(key, [])
        sections.append([_cap_claim_line(x) for x in rows if _cap_claim_line(x)] if isinstance(rows, list) else [])
    return sections


def _apply_verified_policy(facts: List[Dict[str, Any]], *, verified_requires_span: bool) -> int:
    """Enforce verified-requires-span on derived facts; returns downgrade count.

    Shared by ingest_event (event-derived fact candidates) and append_fact
    (raw fact rows) so both entrypoints apply identical grounding policy.
    """
    if not bool(verified_requires_span):
        return 0
    downgrades = 0
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        if safe_text(fact.get("state"), "asserted", 32).lower() != "verified":
            continue
        metadata = fact.get("metadata", {}) if isinstance(fact.get("metadata", {}), dict) else {}
        verification = metadata.get("verification") or fact.get("verification")
        if is_grounded_claim(fact.get("evidence_items"), verification):
            continue
        fact["state"] = "asserted"
        meta = dict(metadata)
        meta["policy"] = "verified_requires_span"
        fact["metadata"] = meta
        downgrades += 1
    return downgrades


_IMPORT_ERROR_CAP = 20


def _claim_citation(citations: Dict[str, Any], claim: str) -> ClaimCitation:
    return ClaimCitation.from_value((citations or {}).get(claim, []), fallback_ref=safe_text(claim, "claim", 256))


def _excerpt_metrics(citation: ClaimCitation) -> tuple[int, int]:
    primary_item = citation.primary_evidence_item()
    if not primary_item:
        return 0, 0
    if not safe_text(primary_item.get("excerpt"), "", 4096):
        return 0, 0
    return 1, (1 if evidence_item_excerpt_matches_checksum(primary_item) else 0)


def _dedupe_evidence_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in normalize_evidence_items(items):
        key = json.dumps(item, sort_keys=True, ensure_ascii=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _flatten_handoff_evidence_items(packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = list(normalize_evidence_items(packet.get("evidence_items")))
    for key in ("claims", "unresolved_contradictions"):
        for row in packet.get(key, []) if isinstance(packet.get(key, []), list) else []:
            if isinstance(row, dict):
                items.extend(normalize_evidence_items(row.get("evidence_items")))
    return _dedupe_evidence_items(items)


def _handoff_row_span_grounded(row: Dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    return bool(row.get("span_grounded")) or has_grounded_span(row.get("evidence_items"))


def _source_priority(source_record_type: str) -> int:
    return 1 if safe_text(source_record_type, "", 16) == "fact" else 0


def _derive_claim_from_row(row: Dict[str, Any], source_record_type: str) -> str:
    if source_record_type == "fact":
        return safe_text(row.get("claim"), "", 2048)
    payload = row.get("payload", {}) if isinstance(row.get("payload", {}), dict) else {}
    for key in ("claim", "decision", "outcome", "target"):
        text = safe_text(payload.get(key), "", 2048)
        if text:
            return text
    if isinstance(payload.get("claims"), list) and payload.get("claims"):
        return safe_text(payload["claims"][0], "", 2048)
    return ""


def _is_live_status(status: str) -> bool:
    return safe_text(status, "", 32) not in {"closed", "superseded"}


def _control_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_state_kind: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    seen_scope_ids: set[tuple[str, str]] = set()
    for row in rows:
        state_context = normalize_state_context(row.get("state_context")) if isinstance(row, dict) and "state_context" in row else {}
        scope_id = safe_text(row.get("scope_id", state_context.get("scope_id")), "", 256)
        thread_id = safe_text(row.get("thread_id", state_context.get("thread_id")), "", 128)
        key = (thread_id, scope_id)
        if not scope_id or key in seen_scope_ids:
            continue
        seen_scope_ids.add(key)
        state_kind = safe_text(row.get("state_kind", state_context.get("state_kind")), "", 64)
        status = safe_text(row.get("status", state_context.get("status")), "", 64)
        if state_kind:
            by_state_kind[state_kind] = by_state_kind.get(state_kind, 0) + 1
        if status:
            by_status[status] = by_status.get(status, 0) + 1
    return {
        "active_scope_count": int(len(seen_scope_ids)),
        "by_state_kind": by_state_kind,
        "by_status": by_status,
    }


def _package_version() -> str:
    """Report the running process's package version (lazy: avoids a cycle,
    since this module is imported by the package ``__init__``)."""
    try:
        from evidencespine import __version__

        return str(__version__ or "")
    except Exception:
        return ""


class AgentMemoryRuntime:
    def __init__(
        self,
        config: AgentMemoryRuntimeConfig | None = None,
        *,
        hooks: RuntimeHooks | None = None,
        vector_backend: VectorBackend | None = None,
    ) -> None:
        self.config = config or AgentMemoryRuntimeConfig()
        self.hooks = hooks or RuntimeHooks()
        self.vector_backend = vector_backend

        self.store = AgentMemoryStore(
            AgentMemoryStoreConfig(
                storage_format=safe_text(self.config.storage_format, "sqlite", 16).lower(),
                db_path=str(self.config.db_path),
                events_path=str(self.config.events_path),
                facts_path=str(self.config.facts_path),
                state_path=str(self.config.state_path),
                briefs_dir=str(self.config.briefs_dir),
                handoffs_dir=str(self.config.handoffs_dir),
                max_event_tail=int(max(100, int(self.config.max_event_tail))),
                dedupe_window_sec=float(max(60.0, float(self.config.dedupe_window_sec))),
                redaction_enable=bool(self.config.redaction_enable),
                fail_open=bool(self.config.fail_open),
                state_flush_interval_sec=float(max(0.0, float(self.config.state_flush_interval_sec))),
            )
        )
        self.retriever = AgentMemoryRetriever(
            AgentMemoryRetrieverConfig(
                top_k_events=int(max(1, int(self.config.brief_top_k_events))),
                top_k_facts=int(max(1, int(self.config.brief_top_k_facts))),
                recency_half_life_hours=float(max(0.25, float(self.config.brief_recency_half_life_hours))),
                max_tokens=int(max(64, int(self.config.brief_max_tokens))),
                retrieval_mode=safe_text(self.config.retrieval_mode, "lexical", 16).lower(),
                lexical_weight=float(max(0.0, float(self.config.retrieval_lexical_weight))),
                vector_weight=float(max(0.0, float(self.config.retrieval_vector_weight))),
            ),
            vector_backend=(self.vector_backend if self.vector_backend is not None else self._default_vector_backend()),
        )

    def _default_vector_backend(self) -> VectorBackend | None:
        mode = safe_text(self.config.retrieval_mode, "lexical", 16).lower()
        if mode not in {"hybrid", "vector"}:
            return None
        kind = safe_text(self.config.embedding_backend, "auto", 16).lower()
        if kind in {"fastembed", "auto"}:
            try:
                import fastembed  # type: ignore[import-not-found]  # noqa: F401
            except Exception:
                if kind == "fastembed":
                    raise ImportError(
                        "embedding_backend=fastembed requires the [embeddings] extra; "
                        "run `pip install evidencespine[embeddings]`"
                    )
                return HashingVectorBackend()
            return FastEmbedVectorBackend(model=str(self.config.embedding_model))
        return HashingVectorBackend()

    def _record_fail_open(self, scope: str, reason: str) -> None:
        # Store helper increments the cumulative counter, appends the bounded
        # fail_open_ts ring used by the 24h snapshot metric, and force-flushes.
        self.store.record_fail_open(scope, reason)

    def _run_contradiction_pass(self, query: str, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        fn = self.hooks.contradiction_pass
        if not callable(fn):
            return []
        try:
            out = fn(query, facts)
            if isinstance(out, list):
                return [x for x in out if isinstance(x, dict)]
            return []
        except Exception:
            return []

    def _lookback_hours(self, value: float | None) -> float:
        if value is not None:
            return max(1.0, float(value))
        return max(1.0, float(self.config.control_view_lookback_hours))

    def _control_candidate(self, row: Dict[str, Any], source_record_type: str, now_ts: float) -> Dict[str, Any] | None:
        if not isinstance(row, dict) or "state_context" not in row:
            return None
        state_context = normalize_state_context(row.get("state_context"))
        if not state_context:
            return None
        scope_id = safe_text(state_context.get("scope_id"), "", 256)
        if not scope_id:
            return None
        reported_at = safe_text(row.get("ts_utc", row.get("generated_at", "")), "", 64)
        source_record_id = safe_text(row.get("fact_id" if source_record_type == "fact" else "event_id"), "", 128)
        candidate_row = ControlViewRow(
            scope_id=scope_id,
            thread_id=safe_text(row.get("thread_id"), "", 128),
            scope_kind=safe_text(state_context.get("scope_kind"), "", 64),
            state_kind=safe_text(state_context.get("state_kind"), "", 64),
            status=safe_text(state_context.get("status"), "", 64),
            owner_agent_id=safe_text(state_context.get("owner_agent_id"), "", 128),
            state_basis=safe_text(state_context.get("state_basis"), "", 64),
            claim=_derive_claim_from_row(row, source_record_type),
            source_record_id=source_record_id,
            source_record_type=source_record_type,
            reported_at=reported_at,
            validated_at=safe_text(state_context.get("validated_at"), "", 64),
            fresh_until=safe_text(state_context.get("fresh_until"), "", 64),
            freshness_state=freshness_state_for_context(state_context, now_ts=now_ts),
            lease_expires_at=safe_text(state_context.get("lease_expires_at"), "", 64),
            lease_state=lease_state_for_context(state_context, now_ts=now_ts),
            evidence_refs=merge_evidence_refs(row.get("evidence_refs"), row.get("evidence_items")),
            evidence_items=normalize_evidence_items(row.get("evidence_items")),
            metadata=dict(row.get("metadata", {}) or {}),
        )
        return {
            "row": candidate_row,
            "sort_ts": control_row_sort_ts(candidate_row),
            "source_priority": _source_priority(source_record_type),
            "supersedes_ref": safe_text(state_context.get("supersedes"), safe_text(row.get("supersedes_fact_id"), "", 128), 128),
            "fact_state": safe_text(row.get("state"), "", 32).lower() if source_record_type == "fact" else "",
        }

    def _collect_control_candidates(self, *, thread_id: str = "", lookback_hours: float | None = None) -> List[Dict[str, Any]]:
        now_ts = float(time.time())
        window = self._lookback_hours(lookback_hours)
        events = self.store.list_recent_events(
            thread_id=thread_id,
            max_items=max(256, int(self.config.max_event_tail)),
            lookback_hours=window,
        )
        facts = self.store.list_recent_facts(
            thread_id=thread_id,
            max_items=max(256, int(self.config.max_event_tail)),
            lookback_hours=window,
        )
        candidates: List[Dict[str, Any]] = []
        for event in events:
            candidate = self._control_candidate(event, "event", now_ts)
            if candidate is not None:
                candidates.append(candidate)
        for fact in facts:
            candidate = self._control_candidate(fact, "fact", now_ts)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _resolved_control_rows(self, *, thread_id: str = "", include_closed: bool = False, lookback_hours: float | None = None) -> List[ControlViewRow]:
        candidates = self._collect_control_candidates(thread_id=thread_id, lookback_hours=lookback_hours)
        if not candidates:
            return []
        superseded_ids = {safe_text(item.get("supersedes_ref"), "", 128) for item in candidates if safe_text(item.get("supersedes_ref"), "", 128)}
        # Group by (thread_id, scope_id): the same scope id in two threads is
        # two distinct live rows, not one merged row.
        grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for item in candidates:
            row = item["row"]
            grouped.setdefault((safe_text(row.thread_id, "", 128), row.scope_id), []).append(item)

        resolved: List[ControlViewRow] = []
        for _key, rows in grouped.items():
            active_rows = [item for item in rows if item["row"].source_record_id not in superseded_ids]
            if not active_rows:
                active_rows = list(rows)
            active_rows.sort(key=lambda item: (float(item["sort_ts"]), int(item["source_priority"]), safe_text(item["row"].source_record_id, "", 128)))
            head = active_rows[-1]

            conflict = False
            multi_owner = False
            live_rows = [item for item in active_rows if _is_live_status(item["row"].status)]
            if len(live_rows) > 1:
                owners = {item["row"].owner_agent_id for item in live_rows}
                multi_owner = len(owners) > 1
                by_kind: Dict[str, List[Dict[str, Any]]] = {}
                for item in live_rows:
                    kind = safe_text(item["row"].state_kind, "", 64)
                    by_kind.setdefault(kind, []).append(item)
                for kind, kind_rows in by_kind.items():
                    if len(kind_rows) < 2:
                        continue
                    statuses = {item["row"].status for item in kind_rows}
                    bases = {item["row"].state_basis for item in kind_rows}
                    if len(statuses) > 1 or len(bases) > 1:
                        conflict = True
                        break

            has_contradiction = any(item.get("fact_state") == "contradicted" for item in active_rows)
            head_row = head["row"]
            row = ControlViewRow(
                scope_id=head_row.scope_id,
                thread_id=head_row.thread_id,
                scope_kind=head_row.scope_kind,
                state_kind=head_row.state_kind,
                status=head_row.status,
                owner_agent_id=head_row.owner_agent_id,
                state_basis=head_row.state_basis,
                claim=head_row.claim,
                source_record_id=head_row.source_record_id,
                source_record_type=head_row.source_record_type,
                reported_at=head_row.reported_at,
                validated_at=head_row.validated_at,
                fresh_until=head_row.fresh_until,
                freshness_state=head_row.freshness_state,
                lease_expires_at=head_row.lease_expires_at,
                lease_state=head_row.lease_state,
                has_contradiction=has_contradiction,
                conflict=conflict,
                multi_owner=multi_owner,
                evidence_refs=list(head_row.evidence_refs),
                evidence_items=list(head_row.evidence_items),
                metadata={
                    **dict(head_row.metadata or {}),
                    "supporting_record_count": int(len(active_rows)),
                    "live_record_count": int(len(live_rows)),
                },
            )
            if include_closed or _is_live_status(row.status):
                resolved.append(row)
        resolved.sort(key=lambda row: (control_row_sort_ts(row), _source_priority(row.source_record_type), safe_text(row.source_record_id, "", 128)), reverse=True)
        return resolved

    def ingest_event(self, event: AgentMemoryEvent | Dict[str, Any]) -> Dict[str, Any]:
        if not bool(self.config.enabled):
            return {"status": "disabled", "reason": "agent_memory_disabled"}
        try:
            if isinstance(event, AgentMemoryEvent):
                row = event.to_dict()
            else:
                payload = dict(event or {})
                state_context = payload.get("state_context") if "state_context" in payload else None
                row = AgentMemoryEvent(
                    event_id=safe_text(payload.get("event_id"), "", 128),
                    thread_id=safe_text(payload.get("thread_id"), "", 128),
                    event_type=safe_text(payload.get("event_type"), "reflection", 64),
                    role=safe_text(payload.get("role"), "unknown", 64),
                    source_agent_id=safe_text(payload.get("source_agent_id"), "unknown", 128),
                    source_turn_id=safe_text(payload.get("source_turn_id"), "", 128),
                    ts_utc=safe_text(payload.get("ts_utc"), "", 64),
                    payload=dict(payload.get("payload", {}) or {}),
                    evidence_refs=normalize_refs(payload.get("evidence_refs")),
                    evidence_items=normalize_evidence_items(payload.get("evidence_items")),
                    state_context=state_context,
                    confidence=float(payload.get("confidence", 0.5)),
                    salience=float(payload.get("salience", 0.5)),
                    tags=list(payload.get("tags", []) or []),
                    metadata=dict(payload.get("metadata", {}) or {}),
                ).to_dict()

            # Redact BEFORE validation/hash/derivation: the stored event, its
            # hash identity, dedupe matching and derived fact candidates must
            # all agree with what is actually persisted (store redaction then
            # becomes a no-op on an already-redacted row).
            row = self.store.redact_row(row)
            row = refresh_event_identity(row)

            valid, errors = validate_event_dict(row)
            if not valid:
                return {"status": "invalid", "errors": errors}

            out = self.store.ingest_event(row)
            policy_downgrades = 0
            if out.get("status") == "ok":
                candidates = event_to_fact_candidates(row)
                policy_downgrades = _apply_verified_policy(
                    candidates, verified_requires_span=bool(self.config.verified_requires_span)
                )
                if candidates and any(
                    safe_text(fact.get("state"), "asserted", 32).lower() == "verified"
                    and not safe_text(fact.get("supersedes_fact_id"), "", 128)
                    for fact in candidates
                ):
                    existing = self.store.list_recent_facts(
                        thread_id=safe_text(row.get("thread_id"), "", 128),
                        max_items=max(64, int(self.config.brief_top_k_facts) * 6),
                        lookback_hours=max(1.0, (float(self.config.dedupe_window_sec) / 3600.0) * 3.0),
                    )
                    for fact in candidates:
                        state = safe_text(fact.get("state"), "asserted", 32).lower()
                        if state != "verified" or safe_text(fact.get("supersedes_fact_id"), "", 128):
                            continue
                        target = next(
                            (prior for prior in existing if safe_text(prior.get("claim"), "", 2048).strip().lower() == safe_text(fact.get("claim"), "", 2048).strip().lower()),
                            None,
                        )
                        if target:
                            fact["supersedes_fact_id"] = safe_text(target.get("fact_id"), "", 128)
                for fact in candidates:
                    f = AgentMemoryFact(
                        fact_id=safe_text(fact.get("fact_id"), "", 128),
                        thread_id=safe_text(fact.get("thread_id"), row.get("thread_id", ""), 128),
                        claim=safe_text(fact.get("claim"), "", 4096),
                        state=safe_text(fact.get("state"), "asserted", 32),
                        source_agent_id=safe_text(fact.get("source_agent_id"), row.get("source_agent_id", "unknown"), 128),
                        source_turn_id=safe_text(fact.get("source_turn_id"), row.get("source_turn_id", ""), 128),
                        ts_utc=safe_text(fact.get("ts_utc"), "", 64),
                        evidence_refs=normalize_refs(fact.get("evidence_refs")),
                        evidence_items=normalize_evidence_items(fact.get("evidence_items")),
                        state_context=(fact.get("state_context") if "state_context" in fact else None),
                        confidence=float(fact.get("confidence", row.get("confidence", 0.5))),
                        tags=list(fact.get("tags", []) or []),
                        supersedes_fact_id=safe_text(fact.get("supersedes_fact_id"), "", 128),
                        metadata=dict(fact.get("metadata", {}) or {}),
                    ).to_dict()
                    self.store.append_fact(f)

                if callable(self.hooks.on_event):
                    try:
                        self.hooks.on_event(dict(row))
                    except Exception as exc:
                        self._record_fail_open("hook_on_event", str(exc))

            if policy_downgrades:
                out = dict(out or {})
                out["policy_downgrades"] = policy_downgrades
            return out
        except Exception as exc:
            if bool(self.config.fail_open):
                self._record_fail_open("ingest_event", str(exc))
                return {"status": "fail_open", "reason": str(exc)}
            raise

    def build_brief(self, thread_id: str, query: str, token_budget: int | None = None) -> AgentConversationBrief:
        if not bool(self.config.enabled):
            return AgentConversationBrief(thread_id=safe_text(thread_id, "", 128), query=safe_text(query, "", 1024))
        self.store.state["brief_generation_attempts_total"] = int(max(0, int(self.store.state.get("brief_generation_attempts_total", 0)))) + 1
        stale = False
        try:
            # Briefs are bounded by brief_lookback_hours (default 168h), not by
            # the short dedupe window: recency ranking already bounds relevance,
            # and older-but-important context must be eligible.
            lookback_hours = max(1.0, float(self.config.brief_lookback_hours))
            events = self.store.list_recent_events(
                thread_id=thread_id,
                max_items=max(64, int(self.config.brief_top_k_events) * 6),
                lookback_hours=lookback_hours,
            )
            facts = self.store.list_recent_facts(
                thread_id=thread_id,
                max_items=max(64, int(self.config.brief_top_k_facts) * 6),
                lookback_hours=lookback_hours,
            )
            # Same identity (thread, claim, turn) may appear twice when a
            # verified twin shares its source turn with the asserted original.
            # Newest row wins; a self-reference (twin superseding its own id)
            # retires the older state, never the twin itself.
            latest_by_id: Dict[str, Dict[str, Any]] = {}
            for idx, fact in enumerate(facts):
                fid = safe_text(fact.get("fact_id"), "", 128) or f"__noid_{idx}"
                prev = latest_by_id.get(fid)
                if prev is None or (parse_ts_value(fact.get("ts_utc")) or 0.0) >= (
                    parse_ts_value(prev.get("ts_utc")) or 0.0
                ):
                    latest_by_id[fid] = fact
            facts = list(latest_by_id.values())
            superseded_ids = {
                safe_text(fact.get("supersedes_fact_id"), "", 128)
                for fact in facts
                if safe_text(fact.get("supersedes_fact_id"), "", 128)
                and safe_text(fact.get("supersedes_fact_id"), "", 128)
                != safe_text(fact.get("fact_id"), "", 128)
            }
            superseded_claim_keys: set[str] = set()
            if superseded_ids:
                for fact in facts:
                    if safe_text(fact.get("fact_id"), "", 128) in superseded_ids:
                        clean, _ = strip_inline_refs(safe_text(fact.get("claim"), "", 2048))
                        if clean.strip():
                            superseded_claim_keys.add(clean.strip().lower())
                facts = [fact for fact in facts if safe_text(fact.get("fact_id"), "", 128) not in superseded_ids]
            unresolved = self._run_contradiction_pass(query, facts)
            brief = self.retriever.build_brief(
                thread_id=thread_id,
                query=query,
                events=events,
                facts=facts,
                unresolved_contradictions=unresolved,
                token_budget=token_budget,
                superseded_claim_keys=superseded_claim_keys,
            )
            now_ts = float(time.time())
            latest_ts = 0.0
            for row in events:
                latest_ts = max(latest_ts, parse_ts_value(row.get("ts")) or parse_ts_value(row.get("ts_utc")) or 0.0)
            # No events inside the lookback window means the memory is stale even
            # when the store is non-empty (all rows are older than the window).
            if not events:
                stale = True
            elif latest_ts > 0.0:
                stale = bool((now_ts - latest_ts) > float(max(300.0, self.config.dedupe_window_sec)))
            brief.metadata["stale"] = bool(stale)
            brief.metadata["lookback_hours"] = float(lookback_hours)
            brief.metadata["unresolved_contradictions"] = int(len(unresolved))
            self.store.write_brief(thread_id, brief.to_dict())

            claim_total = 0
            ref_covered = 0
            span_covered = 0
            excerpt_total = 0
            excerpt_covered = 0
            for section in _brief_sections(brief):
                for claim in section:
                    claim_total += 1
                    citation = _claim_citation(brief.citations if isinstance(brief.citations, dict) else {}, claim)
                    if len(citation) > 0:
                        ref_covered += 1
                    if citation.span_grounded:
                        span_covered += 1
                    add_total, add_covered = _excerpt_metrics(citation)
                    excerpt_total += add_total
                    excerpt_covered += add_covered
            self.store.record_brief_stats(
                attempt=False,
                success=True,
                stale=bool(stale),
                citation_ref_total=claim_total,
                citation_ref_covered=ref_covered,
                citation_span_total=claim_total,
                citation_span_covered=span_covered,
                citation_excerpt_total=excerpt_total,
                citation_excerpt_covered=excerpt_covered,
            )

            if callable(self.hooks.on_brief):
                try:
                    self.hooks.on_brief(brief.to_dict())
                except Exception as exc:
                    self._record_fail_open("hook_on_brief", str(exc))

            return brief
        except Exception as exc:
            if bool(self.config.fail_open):
                self._record_fail_open("build_brief", str(exc))
                fallback = AgentConversationBrief(
                    thread_id=safe_text(thread_id, "", 128),
                    query=safe_text(query, "", 1024),
                    token_budget=int(token_budget if token_budget is not None else self.config.brief_max_tokens),
                )
                fallback.metadata["fail_open_reason"] = safe_text(str(exc), "", 512)
                fallback.metadata["stale"] = True
                fallback.metadata["active_scope_count"] = 0
                fallback.metadata["open_gate_count"] = 0
                fallback.metadata["stale_scope_count"] = 0
                failure_ts = list(self.store.state.get("brief_failure_ts", []) or [])
                failure_ts.append(float(time.time()))
                self.store.state["brief_failure_ts"] = failure_ts[-64:]
                self.store.write_brief(thread_id, fallback.to_dict())
                self.store.record_brief_stats(
                    attempt=False,
                    success=False,
                    stale=True,
                    citation_ref_total=0,
                    citation_ref_covered=0,
                    citation_span_total=0,
                    citation_span_covered=0,
                    citation_excerpt_total=0,
                    citation_excerpt_covered=0,
                )
                return fallback
            raise

    def emit_handoff(self, role: str, thread_id: str, scope: str = "cross-agent coordination") -> AgentHandoffPacket:
        brief = self.build_brief(thread_id=thread_id, query=f"handoff for {role}")
        citations = brief.citations if isinstance(brief.citations, dict) else {}
        provenance_raw = (brief.metadata or {}).get("provenance")
        provenance: Dict[str, Any] = provenance_raw if isinstance(provenance_raw, dict) else {}

        def _provenance_fields(key: str) -> Dict[str, Any]:
            candidate = provenance.get(key)
            prov: Dict[str, Any] = candidate if isinstance(candidate, dict) else {}
            verification = normalize_verification(prov.get("verification"))
            metadata_raw = prov.get("metadata")
            metadata: Dict[str, Any] = metadata_raw if isinstance(metadata_raw, dict) else {}
            return {
                "verification": dict(verification) if verification else None,
                "evidence_stale": bool(prov.get("evidence_stale")),
                "metadata": dict(metadata),
            }

        claims: List[Dict[str, Any]] = []
        seen_claim_keys: set[str] = set()

        def _add_claim(claim: str, status: str, *, provenance_key: str = "") -> Dict[str, Any] | None:
            clean = safe_text(claim, "", CITATION_LINE_MAX)
            if not clean:
                return None
            key = clean.strip().lower()
            if key in seen_claim_keys:
                return None
            seen_claim_keys.add(key)
            lookup = safe_text(provenance_key, "", CITATION_LINE_MAX) or clean
            citation = _claim_citation(citations, lookup)
            row: Dict[str, Any] = {
                "claim": clean,
                "evidence_refs": list(citation),
                "evidence_items": list(citation.evidence_items),
                "span_grounded": bool(citation.span_grounded),
                "status": status,
                **_provenance_fields(lookup),
            }
            if citation.state_context is not None:
                row["state_context"] = normalize_state_context(citation.state_context)
            claims.append(row)
            return row

        for verified_claim in list(brief.recent_verified_facts or [])[:24]:
            _add_claim(verified_claim, "verified")
        if len(claims) == 0:
            for fallback_claim in (list(brief.locked_decisions or []) + list(brief.open_items or []))[:24]:
                _add_claim(fallback_claim, "asserted")
        # Stale evidence travels as asserted + evidence_stale=true, never as a
        # verified claim (it must not re-import as verified).
        for risk in list(brief.active_risks or [])[:24]:
            risk_key = safe_text(risk, "", CITATION_LINE_MAX)
            if not risk_key.startswith("STALE EVIDENCE"):
                continue
            inner = risk_key
            if "): " in risk_key:
                inner = risk_key.split("): ", 1)[1]
            stale_row = _add_claim(inner, "asserted", provenance_key=risk_key)
            if stale_row is not None:
                stale_row["evidence_stale"] = True
        unresolved: List[Dict[str, Any]] = []
        for risk in list(brief.active_risks or [])[:24]:
            risk_key = safe_text(risk, "", CITATION_LINE_MAX)
            if "CONTRADICTION" not in risk_key:
                continue
            citation = _claim_citation(citations, risk_key)
            row: Dict[str, Any] = {
                "claim": risk_key,
                "reason": "unresolved_contradiction",
                "evidence_refs": list(citation),
                "evidence_items": list(citation.evidence_items),
                "span_grounded": bool(citation.span_grounded),
                **_provenance_fields(risk_key),
            }
            if citation.state_context is not None:
                row["state_context"] = normalize_state_context(citation.state_context)
            unresolved.append(row)

        evidence_refs: List[str] = []
        evidence_items: List[Dict[str, Any]] = []
        for row in claims:
            evidence_refs.extend(merge_evidence_refs(row.get("evidence_refs"), row.get("evidence_items")))
            evidence_items.extend(normalize_evidence_items(row.get("evidence_items")))
        for row in unresolved:
            evidence_refs.extend(merge_evidence_refs(row.get("evidence_refs"), row.get("evidence_items")))
            evidence_items.extend(normalize_evidence_items(row.get("evidence_items")))
        evidence_refs = list(dict.fromkeys([x for x in evidence_refs if x]))
        evidence_items = _dedupe_evidence_items(evidence_items)
        required_validations = list(brief.open_items or [])
        if len(required_validations) == 0:
            required_validations = [f"Validate scope: {safe_text(scope, 'cross-agent coordination', 256)}"]

        snapshot = {
            "events_total": int(max(0, int(self.store.state.get("events_total", 0)))),
            "facts_total": int(max(0, int(self.store.state.get("facts_total", 0)))),
            "brief_generated_at": safe_text(brief.generated_at, "", 64),
            "last_update_ts": safe_text(self.store.state.get("last_update_ts"), "", 64),
        }
        active_scope_rows = [row for row in claims + unresolved if normalize_state_context(row.get("state_context"))]

        packet = AgentHandoffPacket(
            packet_id="",
            role=safe_text(role, "unknown", 64).lower(),
            thread_id=safe_text(thread_id, "", 128),
            scope=safe_text(scope, "cross-agent coordination", 2048),
            locked_decisions=list(brief.locked_decisions or []),
            claims=claims,
            unresolved_contradictions=unresolved,
            required_validations=list(required_validations),
            evidence_refs=evidence_refs,
            evidence_items=evidence_items,
            source_snapshot=snapshot,
            metadata={
                "brief_token_budget": int(brief.token_budget),
                "brief_token_used_estimate": int((brief.metadata or {}).get("token_used_estimate", 0)),
                "citation_ref_claim_total": int(max(0, int(self.store.state.get("citation_ref_claim_total", 0)))),
                "citation_ref_claim_covered_total": int(max(0, int(self.store.state.get("citation_ref_claim_covered_total", 0)))),
                "citation_span_claim_total": int(max(0, int(self.store.state.get("citation_span_claim_total", 0)))),
                "citation_span_claim_covered_total": int(max(0, int(self.store.state.get("citation_span_claim_covered_total", 0)))),
                "active_scope_summary": _control_summary(active_scope_rows),
            },
        )
        payload = packet.to_dict()
        path = self.store.write_handoff(thread_id, role, payload)
        payload["file_path"] = path
        self.store.record_handoff_packet()

        try:
            self.ingest_event(
                {
                    "thread_id": thread_id,
                    "event_type": "reflection",
                    "role": role,
                    "source_agent_id": "evidencespine_runtime",
                    "source_turn_id": payload.get("packet_id", ""),
                    "payload": {
                        "claim": f"handoff packet emitted for role={role}",
                        "next_actions": payload.get("required_validations", []),
                        "objective_id": f"agent_handoff::{safe_text(thread_id, 'thread', 64)}",
                    },
                    "evidence_refs": normalize_refs(payload.get("evidence_refs")),
                    "evidence_items": normalize_evidence_items(payload.get("evidence_items")),
                    "confidence": 0.75,
                    "salience": 0.6,
                    "metadata": {"packet_id": payload.get("packet_id"), "file_path": path},
                }
            )
        except Exception as exc:
            # Self-ingest is best-effort, but a failure must still be visible
            # (and force-flushed) rather than silently swallowed.
            self._record_fail_open("emit_handoff_self_ingest", str(exc))

        if callable(self.hooks.on_handoff):
            try:
                self.hooks.on_handoff(dict(payload))
            except Exception as exc:
                self._record_fail_open("hook_on_handoff", str(exc))

        return packet

    def import_handoff(
        self,
        payload_or_path: str | Dict[str, Any],
        *,
        source_agent_id: str = "external_agent",
        thread_id: str = "",
    ) -> Dict[str, Any]:
        packet: Dict[str, Any]
        if isinstance(payload_or_path, dict):
            packet = dict(payload_or_path)
        else:
            path = str(payload_or_path or "").strip()
            if not path or not os.path.exists(path):
                return {"status": "missing", "path": path}
            with open(path, "r", encoding="utf-8") as handle:
                packet = json.load(handle)
        valid, errors = validate_handoff_dict(packet)
        if not valid:
            return {"status": "invalid", "errors": errors}

        source_thread = safe_text(packet.get("thread_id"), "", 128)
        target_thread = safe_text(thread_id, "", 128) or source_thread
        packet_id = safe_text(packet.get("packet_id"), "", 128)
        scope = safe_text(packet.get("scope"), "", 1024)
        role = safe_text(packet.get("role"), "unknown", 64)
        source_agent = safe_text(source_agent_id, "external_agent", 128)
        evidence_items = _flatten_handoff_evidence_items(packet)
        base_metadata = {"imported_packet_id": packet_id, "imported_from_thread": source_thread}
        counts: Dict[str, int] = {
            "decisions_imported": 0,
            "claims_imported": 0,
            "contradictions_imported": 0,
            "validations_imported": 0,
            "state_rows_imported": 0,
            "duplicates_skipped": 0,
            "failed_ingests": 0,
        }
        import_errors: List[Dict[str, Any]] = []
        # Thread-filtered scan: only the importer thread is scanned for
        # duplicate claims instead of the whole store.
        existing_fact_keys = {
            (
                safe_text(fact.get("claim"), "", 2048).strip().lower(),
                safe_text(fact.get("state"), "", 32).lower(),
            )
            for fact in self.store.iter_facts(thread_id=target_thread)
        }
        existing_live_claims = {
            claim
            for claim, state in existing_fact_keys
            if state in {"asserted", "verified"}
        }

        def _counted(result: Dict[str, Any], key: str, row_type: str = "", index: int = 0) -> None:
            if result.get("status") in {"ok", "deduped"}:
                counts[key] = int(counts.get(key, 0)) + 1
                return
            counts["failed_ingests"] = int(counts.get("failed_ingests", 0)) + 1
            if len(import_errors) < _IMPORT_ERROR_CAP:
                import_errors.append({"row_type": row_type or key, "index": int(index), "result": dict(result or {})})

        decisions = packet.get("locked_decisions", []) if isinstance(packet.get("locked_decisions", []), list) else []
        for idx, decision in enumerate(decisions):
            if not isinstance(decision, str) or not decision.strip():
                continue
            clean_decision, inline_refs = strip_inline_refs(decision)
            if not clean_decision:
                continue
            if clean_decision.strip().lower() in existing_live_claims:
                counts["duplicates_skipped"] = int(counts.get("duplicates_skipped", 0)) + 1
                continue
            _counted(
                self.ingest_event(
                    {
                        "thread_id": target_thread,
                        "event_type": "decision",
                        "role": role,
                        "source_agent_id": source_agent,
                        "source_turn_id": f"{packet_id}:locked_decisions:{idx}",
                        "payload": {"decision": clean_decision.strip(), "scope": scope},
                        "evidence_refs": merge_evidence_refs([*list(packet.get("evidence_refs", []) or []), *inline_refs], None),
                        "evidence_items": evidence_items,
                        "confidence": 0.7,
                        "salience": 0.55,
                        "metadata": {**base_metadata, "imported_row_type": "locked_decisions", "imported_row_index": idx},
                    }
                ),
                "decisions_imported",
                "locked_decisions",
                idx,
            )

        for row_type in ("claims", "unresolved_contradictions"):
            rows = packet.get(row_type, []) if isinstance(packet.get(row_type, []), list) else []
            for idx, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                status = safe_text(row.get("status"), "", 32).lower()
                if status not in {"verified", "asserted", "contradicted", "superseded"}:
                    status = "contradicted" if row_type == "unresolved_contradictions" else "asserted"
                row_verification = normalize_verification(row.get("verification"))
                row_metadata = row.get("metadata", {}) if isinstance(row.get("metadata", {}), dict) else {}
                evidence_stale = bool(row.get("evidence_stale")) or bool(row_metadata.get("evidence_stale"))
                if status == "verified" and bool(self.config.verified_requires_span) and not is_grounded_claim(
                    normalize_evidence_items(row.get("evidence_items")),
                    row_verification,
                ):
                    status = "asserted"
                if evidence_stale and status == "verified":
                    # Stale evidence may never re-import as verified.
                    status = "asserted"
                clean_claim, inline_refs = strip_inline_refs(safe_text(row.get("claim"), "", 2048))
                if not clean_claim:
                    continue
                if (clean_claim.strip().lower(), status) in existing_fact_keys:
                    counts["duplicates_skipped"] = int(counts.get("duplicates_skipped", 0)) + 1
                    continue
                event: Dict[str, Any] = {
                    "thread_id": target_thread,
                    "event_type": "outcome" if status == "verified" else "reflection",
                    "role": role,
                    "source_agent_id": source_agent,
                    "source_turn_id": f"{packet_id}:{row_type}:{idx}",
                    "payload": {
                        "claim": clean_claim.strip(),
                        "fact_state": status,
                        "scope": scope,
                        **({"verification": dict(row_verification)} if row_verification else {}),
                        **({"evidence_stale": True} if evidence_stale else {}),
                    },
                    "evidence_refs": merge_evidence_refs(
                        merge_evidence_refs([*list(row.get("evidence_refs", []) or []), *inline_refs], None),
                        row.get("evidence_items"),
                    ),
                    "evidence_items": normalize_evidence_items(row.get("evidence_items")),
                    "confidence": 0.6,
                    "salience": 0.5,
                    "metadata": {
                        **base_metadata,
                        "imported_row_type": row_type,
                        "imported_row_index": idx,
                        **({"evidence_stale": True} if evidence_stale else {}),
                    },
                }
                state_context = normalize_state_context(row.get("state_context")) if "state_context" in row else {}
                if state_context:
                    event["state_context"] = state_context
                result = self.ingest_event(event)
                _counted(
                    result,
                    "claims_imported" if row_type == "claims" else "contradictions_imported",
                    row_type,
                    idx,
                )
                if state_context:
                    _counted(result, "state_rows_imported")

        validations = [v for v in (packet.get("required_validations") or []) if isinstance(v, str) and v.strip()]
        out = self.ingest_event(
            {
                "thread_id": target_thread,
                "event_type": "reflection",
                "role": role,
                "source_agent_id": source_agent,
                "source_turn_id": packet_id,
                "payload": {
                    "claim": f"imported handoff packet {packet_id}",
                    "scope": scope,
                    "next_actions": list(validations),
                    "objective_id": f"agent_handoff::{safe_text(target_thread, 'thread', 64)}",
                },
                "evidence_refs": merge_evidence_refs(packet.get("evidence_refs", []), evidence_items),
                "evidence_items": evidence_items,
                "confidence": 0.65,
                "salience": 0.55,
                "metadata": dict(base_metadata),
            }
        )
        if out.get("status") in {"ok", "deduped"}:
            counts["validations_imported"] = len(validations)
        else:
            _counted(out, "validations_imported", "required_validations", 0)
        return {"status": "ok", "ingest": out, "packet_id": packet_id, "errors": import_errors, **counts}

    def _evidence_stale_rows(self, *, thread_id: str = "") -> List[ControlViewRow]:
        """Facts whose grounded evidence no longer matches the live file."""
        out: List[ControlViewRow] = []
        for fact in self.store.iter_facts(thread_id=safe_text(thread_id, "", 128)):
            meta = fact.get("metadata", {}) if isinstance(fact.get("metadata", {}), dict) else {}
            if not bool(meta.get("evidence_stale")):
                continue
            items = normalize_evidence_items(fact.get("evidence_items"))
            source_id = safe_text((items[0].get("source_id") if items else None), "unknown", 512)
            out.append(
                ControlViewRow(
                    scope_id=source_id,
                    thread_id=safe_text(fact.get("thread_id"), "", 128),
                    scope_kind="evidence",
                    state_kind="evidence",
                    status="stale",
                    owner_agent_id=safe_text(fact.get("source_agent_id"), "", 128),
                    state_basis="derived",
                    claim=safe_text(fact.get("claim"), "", 2048),
                    source_record_id=safe_text(fact.get("fact_id"), "", 128),
                    source_record_type="fact",
                    reported_at=safe_text(meta.get("evidence_stale_at"), "", 64),
                    freshness_state="stale",
                    has_contradiction=False,
                    conflict=False,
                    multi_owner=False,
                    evidence_refs=normalize_refs(fact.get("evidence_refs")),
                    evidence_items=items,
                    metadata={"evidence_stale": True, "evidence_stale_reason": safe_text(meta.get("evidence_stale_reason"), "", 32)},
                )
            )
        return out

    def query_view(
        self,
        view: str,
        *,
        thread_id: str = "",
        owner_agent_id: str = "",
        include_closed: bool = False,
        limit: int = 50,
        lookback_hours: float | None = None,
    ) -> AgentControlView:
        requested_view = safe_text(view, "", 64)
        owner = safe_text(owner_agent_id, "", 128)
        view_name = normalize_control_view(requested_view)
        if not bool(self.config.enabled):
            return AgentControlView(
                view=view_name or requested_view,
                thread_id=safe_text(thread_id, "", 128),
                owner_agent_id=owner,
            )
        if not view_name:
            # Durable, fail-open validation shared by CLI/MCP/A2A/async surfaces.
            return AgentControlView(
                view=requested_view,
                thread_id=safe_text(thread_id, "", 128),
                owner_agent_id=owner,
                rows=[],
                metadata={
                    "status": "invalid",
                    "reason": "unknown_view",
                    "valid_views": list(CONTROL_VIEWS),
                },
            )

        rows = self._resolved_control_rows(
            thread_id=safe_text(thread_id, "", 128),
            include_closed=bool(include_closed),
            lookback_hours=lookback_hours,
        )
        if view_name == "my_work":
            if owner:
                rows = [row for row in rows if row.owner_agent_id == owner]
            else:
                rows = [row for row in rows if row.owner_agent_id]
        elif view_name == "open_gates":
            rows = [row for row in rows if row.state_kind == "pending_gate" and row.status != "closed"]
        elif view_name == "stale_claims":
            rows = [row for row in rows if row.freshness_state in {"stale", "unknown"}]
            rows = [*rows, *self._evidence_stale_rows(thread_id=safe_text(thread_id, "", 128))]
        elif view_name == "contradictions":
            rows = [row for row in rows if row.has_contradiction or row.conflict]
        else:
            rows = list(rows)

        rows = rows[: max(1, int(limit))]
        return AgentControlView(
            view=view_name,
            thread_id=safe_text(thread_id, "", 128),
            owner_agent_id=owner,
            rows=rows,
            metadata={
                "status": "ok",
                "include_closed": bool(include_closed),
                "limit": int(max(1, int(limit))),
                "lookback_hours": float(self._lookback_hours(lookback_hours)),
                "total_rows": int(len(rows)),
            },
        )

    def reconcile(self, thread_id: str, *, limit: int = 50) -> Dict[str, Any]:
        fn = self.hooks.reconcile_state
        if not callable(fn):
            return {"status": "unsupported", "reason": "no_reconcile_hook"}
        active = self.query_view(
            "active_scopes",
            thread_id=safe_text(thread_id, "", 128),
            include_closed=False,
            limit=max(1, int(limit)),
            lookback_hours=self._lookback_hours(None),
        ).to_dict()
        try:
            rows = fn(safe_text(thread_id, "", 128), list(active.get("rows", [])))
        except Exception as exc:
            if bool(self.config.fail_open):
                self._record_fail_open("reconcile", str(exc))
                return {"status": "fail_open", "reason": str(exc)}
            raise
        if not isinstance(rows, list):
            return {"status": "invalid", "reason": "reconcile_hook_must_return_list"}

        ingested = 0
        invalid = 0
        errors: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                invalid += 1
                errors.append({"index": idx, "reason": "not_a_dict"})
                continue
            payload = dict(row)
            state_context = payload.get("state_context") if "state_context" in payload else None
            payload.setdefault("thread_id", safe_text(thread_id, "", 128))
            payload.setdefault("event_type", "reflection")
            payload.setdefault("role", "operator")
            payload.setdefault("source_agent_id", "runtime_reconcile")
            payload.setdefault("source_turn_id", f"reconcile_{idx}")
            raw_payload = payload.get("payload", {}) if isinstance(payload.get("payload", {}), dict) else {}
            if not any(safe_text(raw_payload.get(key), "", 2048) for key in ("claim", "decision", "outcome")):
                scope_id = safe_text((state_context or {}).get("scope_id"), "scope", 256)
                raw_payload["claim"] = f"reconciled state for {scope_id}"
            payload["payload"] = raw_payload
            if state_context is not None:
                payload["state_context"] = state_context
            result = self.ingest_event(payload)
            if result.get("status") in {"ok", "deduped"}:
                ingested += 1
            else:
                invalid += 1
                errors.append({"index": idx, "result": dict(result)})

        return {
            "status": "ok",
            "thread_id": safe_text(thread_id, "", 128),
            "seen": int(len(rows)),
            "ingested": int(ingested),
            "invalid": int(invalid),
            "errors": errors,
        }

    def snapshot(self) -> Dict[str, Any]:
        now_ts = float(time.time())
        cutoff_ts = now_ts - 24.0 * 3600.0

        events = self.store.list_recent_events(max_items=max(256, int(self.config.max_event_tail)), lookback_hours=24.0)
        facts = self.store.list_recent_facts(max_items=max(256, int(self.config.max_event_tail)), lookback_hours=24.0)

        superseded_twin_ids = {
            safe_text(f.get("supersedes_fact_id"), "", 128)
            for f in facts
            if safe_text(f.get("supersedes_fact_id"), "", 128)
            and safe_text(f.get("supersedes_fact_id"), "", 128) != safe_text(f.get("fact_id"), "", 128)
        }
        verified_facts = [
            f
            for f in facts
            if str(f.get("state", "")).lower() == "verified"
            and safe_text(f.get("fact_id"), "", 128) not in superseded_twin_ids
        ]
        verified_with_provenance = [
            f for f in verified_facts if isinstance(f.get("metadata"), dict) and f["metadata"].get("verification")
        ]
        contradicted = [f for f in facts if str(f.get("state", "")).lower() == "contradicted"]
        unresolved = [f for f in contradicted if not str(f.get("supersedes_fact_id", "")).strip()]

        brief_files = self.store.iter_brief_files()
        handoff_files = self.store.iter_handoff_files()

        brief_recent = 0
        brief_stale = 0
        brief_failures_24h = 0
        now_ts = float(time.time())
        for ts in list(self.store.state.get("brief_failure_ts", []) or []):
            try:
                if (now_ts - float(ts)) <= 86400.0:
                    brief_failures_24h += 1
            except (TypeError, ValueError):
                pass
        citation_ref_total = 0
        citation_ref_covered = 0
        citation_span_total = 0
        citation_span_covered = 0
        citation_excerpt_total = 0
        citation_excerpt_covered = 0
        brief_artifacts_unreadable = 0
        for path in brief_files[-1024:]:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    row = json.load(handle)
                ts = parse_ts_value(row.get("generated_at")) or 0.0
                if ts < cutoff_ts:
                    continue
                brief_recent += 1
                meta = row.get("metadata", {}) if isinstance(row.get("metadata", {}), dict) else {}
                if bool(meta.get("stale", False)):
                    brief_stale += 1
                citations = row.get("citations", {}) if isinstance(row.get("citations", {}), dict) else {}
                for section in _brief_sections(row):
                    for claim in section if isinstance(section, list) else []:
                        citation_ref_total += 1
                        citation_span_total += 1
                        citation = _claim_citation(citations, claim)
                        if len(citation) > 0:
                            citation_ref_covered += 1
                        if citation.span_grounded:
                            citation_span_covered += 1
                        add_total, add_covered = _excerpt_metrics(citation)
                        citation_excerpt_total += add_total
                        citation_excerpt_covered += add_covered
            except Exception:
                brief_artifacts_unreadable += 1
                continue

        handoff_recent = 0
        handoff_complete = 0
        handoff_span_total = 0
        handoff_span_covered = 0
        handoff_artifacts_unreadable = 0
        for path in handoff_files[-1024:]:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    row = json.load(handle)
                ts = parse_ts_value(row.get("generated_at")) or 0.0
                if ts < cutoff_ts:
                    continue
                handoff_recent += 1
                claims = row.get("claims", []) if isinstance(row.get("claims", []), list) else []
                locked_decisions = row.get("locked_decisions", []) if isinstance(row.get("locked_decisions", []), list) else []
                required_validations = row.get("required_validations", []) if isinstance(row.get("required_validations", []), list) else []
                unresolved_rows = row.get("unresolved_contradictions", []) if isinstance(row.get("unresolved_contradictions", []), list) else []
                has_scope = bool(safe_text(row.get("scope"), "", 64))
                has_role = bool(safe_text(row.get("role"), "", 64))
                has_thread = bool(safe_text(row.get("thread_id"), "", 64))
                has_substance = bool(len(claims) > 0 or len(locked_decisions) > 0 or len(required_validations) > 0 or len(unresolved_rows) > 0)
                if has_scope and has_role and has_thread and has_substance:
                    handoff_complete += 1
                for row_item in claims + unresolved_rows:
                    if not isinstance(row_item, dict):
                        continue
                    handoff_span_total += 1
                    if _handoff_row_span_grounded(row_item):
                        handoff_span_covered += 1
            except Exception:
                handoff_artifacts_unreadable += 1
                continue

        active_view = self.query_view("active_scopes", include_closed=False, limit=max(1, int(self.config.max_event_tail)), lookback_hours=24.0).to_dict()
        open_gates_view = self.query_view("open_gates", include_closed=False, limit=max(1, int(self.config.max_event_tail)), lookback_hours=24.0).to_dict()
        active_rows = active_view.get("rows", []) if isinstance(active_view.get("rows", []), list) else []
        open_gate_rows = open_gates_view.get("rows", []) if isinstance(open_gates_view.get("rows", []), list) else []
        owner_covered = sum(1 for row in active_rows if safe_text((row or {}).get("owner_agent_id"), "", 128))
        freshness_covered = sum(1 for row in active_rows if safe_text((row or {}).get("freshness_state"), "", 16) != "unknown")
        stale_rows = sum(1 for row in active_rows if safe_text((row or {}).get("freshness_state"), "", 16) == "stale")
        conflict_rows = sum(1 for row in active_rows if bool((row or {}).get("conflict")))

        fail_open_24h = 0
        fail_open_ts_ring = self.store.state.get("fail_open_ts", [])
        if isinstance(fail_open_ts_ring, list):
            for raw_ts in fail_open_ts_ring:
                try:
                    if 0.0 <= (now_ts - float(raw_ts)) <= 86400.0:
                        fail_open_24h += 1
                except (TypeError, ValueError):
                    continue
        fail_open_total = int(max(0, int(self.store.state.get("fail_open_events_total", 0))))

        last_update_ts = safe_text(self.store.state.get("last_update_ts"), "", 64)
        state_age = None
        if last_update_ts:
            try:
                state_age = now_ts - (parse_ts_value(last_update_ts) or now_ts)
            except Exception:
                state_age = None

        return {
            "enabled": bool(self.config.enabled),
            "evidencespine_version": _package_version(),
            "events_total": int(max(0, int(self.store.state.get("events_total", 0)))),
            "facts_total": int(max(0, int(self.store.state.get("facts_total", 0)))),
            "agent_memory_events_24h": int(len(events)),
            "agent_memory_verified_facts_24h": int(len(verified_facts)),
            "agent_fact_provenance_rate_24h": (float(len(verified_with_provenance) / max(1, len(verified_facts))) if verified_facts else 0.0),
            "agent_memory_contradiction_count_24h": int(len(contradicted)),
            "agent_memory_unresolved_contradiction_ratio_24h": (float(len(unresolved) / max(1, len(contradicted))) if len(contradicted) > 0 else 0.0),
            "agent_brief_generation_success_rate_24h": (float(brief_recent / max(1, brief_recent + brief_failures_24h)) if (brief_recent + brief_failures_24h) > 0 else 0.0),
            "agent_brief_stale_rate_24h": (float(brief_stale / max(1, brief_recent)) if brief_recent > 0 else 0.0),
            "agent_handoff_packets_emitted_24h": int(handoff_recent),
            "agent_handoff_packet_completeness_24h": (float(handoff_complete / max(1, handoff_recent)) if handoff_recent > 0 else 0.0),
            "agent_claim_citation_coverage_24h": (float(citation_ref_covered / max(1, citation_ref_total)) if citation_ref_total > 0 else 0.0),
            "agent_claim_ref_citation_coverage_24h": (float(citation_ref_covered / max(1, citation_ref_total)) if citation_ref_total > 0 else 0.0),
            "agent_claim_span_citation_coverage_24h": (float(citation_span_covered / max(1, citation_span_total)) if citation_span_total > 0 else 0.0),
            "agent_claim_excerpt_fidelity_24h": (float(citation_excerpt_covered / max(1, citation_excerpt_total)) if citation_excerpt_total > 0 else 0.0),
            "agent_handoff_span_grounding_rate_24h": (float(handoff_span_covered / max(1, handoff_span_total)) if handoff_span_total > 0 else 0.0),
            "agent_active_scope_count_24h": int(len(active_rows)),
            "agent_open_gate_count_24h": int(len(open_gate_rows)),
            "agent_scope_owner_coverage_24h": (float(owner_covered / max(1, len(active_rows))) if active_rows else 0.0),
            "agent_active_scope_freshness_coverage_24h": (float(freshness_covered / max(1, len(active_rows))) if active_rows else 0.0),
            "agent_active_scope_stale_rate_24h": (float(stale_rows / max(1, len(active_rows))) if active_rows else 0.0),
            "agent_scope_conflict_rate_24h": (float(conflict_rows / max(1, len(active_rows))) if active_rows else 0.0),
            "agent_evidence_stale_count_24h": int(len(self._evidence_stale_rows())),
            "agent_memory_fail_open_events_24h": int(fail_open_24h),
            "agent_memory_fail_open_events_total": int(fail_open_total),
            "brief_artifacts_unreadable": int(brief_artifacts_unreadable),
            "handoff_artifacts_unreadable": int(handoff_artifacts_unreadable),
            "agent_memory_last_update_ts": last_update_ts or None,
            "agent_memory_state_age_sec": (float(max(0.0, state_age)) if state_age is not None else None),
        }

    def prune(
        self,
        *,
        thread_id: str = "",
        ttl_hours: float | None = None,
        ttl_hours_facts: float | None = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """TTL archival: delete rows older than ``ttl_hours`` (default 720h/30d)."""
        if not bool(self.config.enabled):
            return {"status": "disabled", "reason": "agent_memory_disabled"}
        try:
            result = self.store.prune(
                thread_id=safe_text(thread_id, "", 128),
                ttl_hours=max(0.1, float(ttl_hours if ttl_hours is not None else 720.0)),
                ttl_hours_facts=ttl_hours_facts,
                dry_run=bool(dry_run),
            )
            result["status"] = "ok"
            return result
        except Exception as exc:
            if bool(self.config.fail_open):
                return {"status": "fail_open", "reason": str(exc)}
            raise

    def append_fact(self, fact_row: Dict[str, Any]) -> Dict[str, Any]:
        """Append a raw fact row (async runtime / A2A entrypoints)."""
        if not bool(self.config.enabled):
            return {"status": "disabled", "reason": "agent_memory_disabled"}
        try:
            row = dict(fact_row or {})
            if not safe_text(row.get("claim"), "", 4096):
                return {"status": "invalid", "errors": ["missing:claim"]}
            metadata = dict(row.get("metadata", {}) or {})
            verification = normalize_verification(metadata.get("verification") or row.get("verification"))
            if verification:
                metadata["verification"] = dict(verification)
            fact = AgentMemoryFact(
                fact_id=safe_text(row.get("fact_id"), "", 128),
                thread_id=safe_text(row.get("thread_id"), "", 128),
                claim=safe_text(row.get("claim"), "", 4096),
                state=safe_text(row.get("state"), "asserted", 32),
                source_agent_id=safe_text(row.get("source_agent_id"), "unknown", 128),
                source_turn_id=safe_text(row.get("source_turn_id"), "", 128),
                evidence_refs=normalize_refs(row.get("evidence_refs")),
                evidence_items=normalize_evidence_items(row.get("evidence_items")),
                state_context=(row.get("state_context") if "state_context" in row else None),
                confidence=float(row.get("confidence", 0.5)),
                tags=list(row.get("tags", []) or []),
                supersedes_fact_id=safe_text(row.get("supersedes_fact_id"), "", 128),
                metadata=metadata,
            ).to_dict()
            # Same verified-requires-span policy as ingest_event.
            policy_downgrades = _apply_verified_policy(
                [fact], verified_requires_span=bool(self.config.verified_requires_span)
            )
            out = self.store.append_fact(fact)
            if isinstance(out, dict):
                out = dict(out)
                out["policy_downgrades"] = int(policy_downgrades)
            return out
        except Exception as exc:
            if bool(self.config.fail_open):
                self._record_fail_open("append_fact", str(exc))
                return {"status": "fail_open", "reason": str(exc)}
            raise

    def verify_fact(
        self,
        fact_id: str,
        *,
        method: str,
        reference: str,
        verified_by: str = "external_agent",
        thread_id: str = "",
    ) -> Dict[str, Any]:
        """Record verification provenance for a fact.

        Ingests a verified copy of the fact (claim, grounded evidence items and
        provenance preserved) that supersedes the original, so the brief shows
        the verified-with-provenance row instead of the asserted one.
        """
        if not bool(self.config.enabled):
            return {"status": "disabled", "reason": "agent_memory_disabled"}
        target_id = safe_text(fact_id, "", 128)
        target = next(
            (f for f in self.store.iter_facts() if safe_text(f.get("fact_id"), "", 128) == target_id),
            None,
        )
        if target is None:
            return {"status": "missing", "fact_id": target_id}
        claim = safe_text(target.get("claim"), "", 2048)
        if not claim:
            return {"status": "invalid", "reason": "fact_has_no_claim"}
        verification = normalize_verification(
            {"method": method, "reference": reference, "verified_by": verified_by}
        )
        if not verification:
            return {"status": "invalid", "reason": "invalid_verification"}
        return self.ingest_event(
            {
                "thread_id": safe_text(thread_id or target.get("thread_id"), "", 128),
                "event_type": "outcome",
                "role": safe_text(target.get("source_agent_id"), "auditor", 64),
                "source_agent_id": safe_text(verified_by, "external_agent", 128),
                "source_turn_id": f"verify:{target_id}",
                "payload": {
                    "claim": claim,
                    "fact_state": "verified",
                    "verification": dict(verification),
                    "supersedes_ref": target_id,
                    "scope": "verification",
                },
                "evidence_refs": normalize_refs(target.get("evidence_refs")),
                "evidence_items": normalize_evidence_items(target.get("evidence_items")),
                "confidence": 0.9,
                "salience": 0.6,
            }
        )

    def check_evidence_stale(
        self,
        *,
        thread_id: str = "",
        source_root: str = ".",
        dry_run: bool = True,
        allow_absolute_source_id: bool = True,
    ) -> Dict[str, Any]:
        """Re-verify grounded evidence against live files.

        Supports line anchors, character anchors (0-based start, inclusive end
        over decoded UTF-8 text) and locator-only items parsed with
        grounding.parse_ref. allow_absolute_source_id=False is the server-side
        mode: absolute source ids are unverifiable and relative ids must
        realpath-confine under source_root.

        Verdicts: changed (file drifted), missing (file unreadable) and
        unverifiable (unsupported/ambiguous anchor, out-of-range char span, or
        a source id outside the allowed root). Unverifiable evidence never
        marks a fact stale and never clears an existing stale flag.
        dry_run reports; dry_run=False writes metadata.evidence_stale.
        """
        from evidencespine.grounding import excerpt_checksum, parse_ref, read_lines

        if not bool(self.config.enabled):
            return {"status": "disabled", "reason": "agent_memory_disabled"}

        root_real = os.path.realpath(str(source_root or "."))

        def _resolve_source(value: Any) -> str | None:
            sid = safe_text(value, "", 512)
            if not sid:
                return None
            if os.path.isabs(sid):
                return sid if bool(allow_absolute_source_id) else None
            full = os.path.realpath(os.path.join(root_real, sid))
            if full != root_real and not full.startswith(root_real + os.sep):
                return None
            return full

        def _read_text(path: str) -> str | None:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    return handle.read()
            except OSError:
                return None

        def _matches(checksum: str, text: str) -> bool:
            want = checksum.lower()
            if want.startswith("sha256:"):
                want = want[len("sha256:") :]
            return excerpt_checksum(text) == f"sha256:{want}"

        def _item_verdict(item: Dict[str, Any]) -> str | None:
            excerpt = safe_text(item.get("excerpt"), "", 4096)
            checksum = safe_text(item.get("checksum"), "", 256)
            if not excerpt or not checksum:
                return None
            char_start = safe_int(item.get("char_start"), None, 0, None)
            line_start = safe_int(item.get("line_start"), None, 1, None)
            locator = safe_text(item.get("locator"), "", 512)
            if char_start is not None and line_start is not None:
                return "unverifiable"
            if char_start is not None:
                char_end = safe_int(item.get("char_end"), char_start, 0, None)
                char_end = char_start if char_end is None or char_end < char_start else char_end
                full = _resolve_source(item.get("source_id"))
                if full is None:
                    return "unverifiable"
                body = _read_text(full)
                if body is None:
                    return "missing"
                if char_start >= len(body) or char_end >= len(body):
                    return "unverifiable"
                return None if _matches(checksum, body[char_start : char_end + 1]) else "changed"
            if line_start is not None:
                line_end = safe_int(item.get("line_end"), line_start, 1, None)
                line_end = line_start if line_end is None else max(line_start, line_end)
                full = _resolve_source(item.get("source_id"))
                if full is None:
                    return "unverifiable"
                lines = read_lines(full, line_start, line_end)
                if lines is None:
                    return "missing"
                return None if _matches(checksum, "\n".join(lines)) else "changed"
            if locator:
                parsed = parse_ref(locator)
                if parsed is None:
                    return "unverifiable"
                loc_path, loc_start, loc_end = parsed
                full = _resolve_source(loc_path)
                if full is None:
                    return "unverifiable"
                lines = read_lines(full, loc_start, loc_end)
                if lines is None:
                    return "missing"
                return None if _matches(checksum, "\n".join(lines)) else "changed"
            return "unverifiable"

        try:
            results: List[Dict[str, Any]] = []
            checked_items = 0
            unverifiable_items = 0
            stale_facts = 0
            cleared_facts = 0
            target_thread = safe_text(thread_id, "", 128)
            for fact in self.store.iter_facts(thread_id=target_thread):
                fact_meta = fact.get("metadata", {}) if isinstance(fact.get("metadata", {}), dict) else {}
                items = normalize_evidence_items(fact.get("evidence_items"))
                verdict: str | None = None
                fact_unverifiable = 0
                source_id = ""
                for item in items:
                    excerpt = safe_text(item.get("excerpt"), "", 4096)
                    checksum = safe_text(item.get("checksum"), "", 256)
                    if not excerpt or not checksum:
                        continue
                    source_id = safe_text(item.get("source_id"), "", 512) or safe_text(item.get("locator"), "", 512)
                    item_verdict = _item_verdict(item)
                    if item_verdict == "unverifiable":
                        fact_unverifiable += 1
                        continue
                    checked_items += 1
                    if item_verdict is not None:
                        verdict = item_verdict
                        break
                unverifiable_items += fact_unverifiable
                fact_id = safe_text(fact.get("fact_id"), "", 128)
                row_base = {
                    "fact_id": fact_id,
                    "thread_id": safe_text(fact.get("thread_id"), "", 128),
                    "claim": safe_text(fact.get("claim"), "", 2048),
                    "state": safe_text(fact.get("state"), "asserted", 32).lower(),
                }
                if verdict is None:
                    if fact_unverifiable:
                        # Cannot re-verify and cannot clear: fail closed.
                        results.append({**row_base, "source_id": source_id, "reason": "unverifiable"})
                        continue
                    if bool(fact_meta.get("evidence_stale")):
                        cleared_facts += 1
                        cleared_meta = {
                            k: v
                            for k, v in dict(fact_meta).items()
                            if k not in {"evidence_stale", "evidence_stale_reason", "evidence_stale_at"}
                        }
                        if not dry_run:
                            self.store.update_fact(
                                fact_id,
                                {
                                    "metadata": {
                                        **cleared_meta,
                                        "evidence_stale": None,
                                        "evidence_stale_reason": None,
                                        "evidence_stale_at": None,
                                    }
                                },
                            )
                        results.append({**row_base, "source_id": source_id, "reason": "cleared"})
                    continue
                stale_facts += 1
                meta = dict(fact.get("metadata", {}) or {})
                meta["evidence_stale"] = True
                meta["evidence_stale_reason"] = verdict
                meta["evidence_stale_at"] = utc_now_iso()
                if not dry_run:
                    self.store.update_fact(fact_id, {"metadata": meta})
                results.append({**row_base, "source_id": source_id, "reason": verdict})
            return {
                "status": "ok",
                "checked_items": int(checked_items),
                "unverifiable_items": int(unverifiable_items),
                "stale_facts": int(stale_facts),
                "cleared_facts": int(cleared_facts),
                "dry_run": bool(dry_run),
                "results": results,
            }
        except Exception as exc:
            if bool(self.config.fail_open):
                return {"status": "fail_open", "reason": str(exc)}
            raise

    def flush(self) -> Dict[str, Any]:
        return self.store.flush()
