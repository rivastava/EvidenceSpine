from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from evidencespine.protocol import (
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
    lease_state_for_context,
    merge_evidence_refs,
    normalize_evidence_items,
    normalize_refs,
    normalize_state_context,
    parse_ts_value,
    safe_text,
    utc_now_iso,
    validate_event_dict,
    validate_handoff_dict,
)
from evidencespine.retriever import AgentMemoryRetriever, AgentMemoryRetrieverConfig
from evidencespine.store import AgentMemoryStore, AgentMemoryStoreConfig
from evidencespine.vector_backends import HashingVectorBackend, VectorBackend


@dataclass
class AgentMemoryRuntimeConfig:
    enabled: bool = True
    fail_open: bool = True
    max_event_tail: int = 4000
    redaction_enable: bool = True
    dedupe_window_sec: float = 7200.0
    events_path: str = ".evidencespine/events.jsonl"
    facts_path: str = ".evidencespine/facts.jsonl"
    state_path: str = ".evidencespine/state.json"
    briefs_dir: str = ".evidencespine/briefs"
    handoffs_dir: str = ".evidencespine/handoffs"
    brief_max_tokens: int = 1800
    brief_top_k_events: int = 32
    brief_top_k_facts: int = 24
    brief_recency_half_life_hours: float = 8.0
    retrieval_mode: str = "lexical"  # lexical | hybrid | vector
    retrieval_lexical_weight: float = 1.0
    retrieval_vector_weight: float = 0.35
    control_view_lookback_hours: float = 168.0


@dataclass
class RuntimeHooks:
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None
    on_brief: Optional[Callable[[Dict[str, Any]], None]] = None
    on_handoff: Optional[Callable[[Dict[str, Any]], None]] = None
    contradiction_pass: Optional[Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]]] = None
    reconcile_state: Optional[Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]]] = None


def _brief_sections(brief: AgentConversationBrief | Dict[str, Any]) -> List[List[str]]:
    payload = brief.to_dict() if isinstance(brief, AgentConversationBrief) else dict(brief or {})
    return [
        payload.get("current_goal", []) if isinstance(payload.get("current_goal", []), list) else [],
        payload.get("locked_decisions", []) if isinstance(payload.get("locked_decisions", []), list) else [],
        payload.get("recent_verified_facts", []) if isinstance(payload.get("recent_verified_facts", []), list) else [],
        payload.get("active_risks", []) if isinstance(payload.get("active_risks", []), list) else [],
        payload.get("open_items", []) if isinstance(payload.get("open_items", []), list) else [],
        payload.get("next_actions", []) if isinstance(payload.get("next_actions", []), list) else [],
    ]


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
    seen_scope_ids: set[str] = set()
    for row in rows:
        state_context = normalize_state_context(row.get("state_context")) if isinstance(row, dict) and "state_context" in row else {}
        scope_id = safe_text(row.get("scope_id", state_context.get("scope_id")), "", 256)
        if not scope_id or scope_id in seen_scope_ids:
            continue
        seen_scope_ids.add(scope_id)
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
                events_path=str(self.config.events_path),
                facts_path=str(self.config.facts_path),
                state_path=str(self.config.state_path),
                briefs_dir=str(self.config.briefs_dir),
                handoffs_dir=str(self.config.handoffs_dir),
                max_event_tail=int(max(100, int(self.config.max_event_tail))),
                dedupe_window_sec=float(max(60.0, float(self.config.dedupe_window_sec))),
                redaction_enable=bool(self.config.redaction_enable),
                fail_open=bool(self.config.fail_open),
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
            vector_backend=(
                self.vector_backend
                if self.vector_backend is not None
                else (
                    HashingVectorBackend()
                    if safe_text(self.config.retrieval_mode, "lexical", 16).lower() in {"hybrid", "vector"}
                    else None
                )
            ),
        )

    def _record_fail_open(self, scope: str, reason: str) -> None:
        self.store.state["fail_open_events_total"] = int(max(0, int(self.store.state.get("fail_open_events_total", 0)))) + 1
        self.store.state["last_fail_open"] = {
            "scope": safe_text(scope, "unknown", 128),
            "reason": safe_text(reason, "", 512),
            "ts": utc_now_iso(),
        }
        self.store.flush()

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
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in candidates:
            row = item["row"]
            grouped.setdefault(row.scope_id, []).append(item)

        resolved: List[ControlViewRow] = []
        for scope_id, rows in grouped.items():
            del scope_id
            active_rows = [item for item in rows if item["row"].source_record_id not in superseded_ids]
            if not active_rows:
                active_rows = list(rows)
            active_rows.sort(key=lambda item: (float(item["sort_ts"]), int(item["source_priority"]), safe_text(item["row"].source_record_id, "", 128)))
            head = active_rows[-1]
            live_rows = [item for item in active_rows if _is_live_status(item["row"].status)]

            conflict = False
            if len(live_rows) > 1:
                statuses = {item["row"].status for item in live_rows}
                owners = {item["row"].owner_agent_id for item in live_rows}
                bases = {item["row"].state_basis for item in live_rows}
                conflict = bool(len(statuses) > 1 or len(owners) > 1 or len(bases) > 1)

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

            valid, errors = validate_event_dict(row)
            if not valid:
                return {"status": "invalid", "errors": errors}

            out = self.store.ingest_event(row)
            if out.get("status") == "ok":
                for fact in event_to_fact_candidates(row):
                    f = AgentMemoryFact(
                        fact_id=safe_text(fact.get("fact_id"), "", 128),
                        thread_id=safe_text(fact.get("thread_id"), row.get("thread_id", ""), 128),
                        claim=safe_text(fact.get("claim"), "", 4096),
                        state=safe_text(fact.get("state"), "asserted", 32),
                        source_agent_id=safe_text(fact.get("source_agent_id"), row.get("source_agent_id", "unknown"), 128),
                        source_turn_id=safe_text(fact.get("source_turn_id"), row.get("source_turn_id", ""), 128),
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
            lookback_hours = max(1.0, (float(self.config.dedupe_window_sec) / 3600.0) * 3.0)
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
            unresolved = self._run_contradiction_pass(query, facts)
            brief = self.retriever.build_brief(
                thread_id=thread_id,
                query=query,
                events=events,
                facts=facts,
                unresolved_contradictions=unresolved,
                token_budget=token_budget,
            )
            now_ts = float(time.time())
            latest_ts = 0.0
            for row in events:
                latest_ts = max(latest_ts, parse_ts_value(row.get("ts")) or parse_ts_value(row.get("ts_utc")) or 0.0)
            if latest_ts > 0.0:
                stale = bool((now_ts - latest_ts) > float(max(300.0, self.config.dedupe_window_sec)))
            brief.metadata["stale"] = bool(stale)
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
                self.store.write_brief(thread_id, fallback.to_dict())
                self.store.record_brief_stats(
                    attempt=False,
                    success=True,
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
        claims: List[Dict[str, Any]] = []
        for claim in list(brief.recent_verified_facts or [])[:24]:
            citation = _claim_citation(brief.citations if isinstance(brief.citations, dict) else {}, claim)
            row: Dict[str, Any] = {
                "claim": claim,
                "evidence_refs": list(citation),
                "evidence_items": list(citation.evidence_items),
                "span_grounded": bool(citation.span_grounded),
                "status": "verified",
            }
            if citation.state_context is not None:
                row["state_context"] = normalize_state_context(citation.state_context)
            claims.append(row)
        if len(claims) == 0:
            fallback_claims = list(brief.locked_decisions or []) + list(brief.open_items or [])
            for claim in fallback_claims[:24]:
                citation = _claim_citation(brief.citations if isinstance(brief.citations, dict) else {}, claim)
                row = {
                    "claim": claim,
                    "evidence_refs": list(citation),
                    "evidence_items": list(citation.evidence_items),
                    "span_grounded": bool(citation.span_grounded),
                    "status": "asserted",
                }
                if citation.state_context is not None:
                    row["state_context"] = normalize_state_context(citation.state_context)
                claims.append(row)
        unresolved: List[Dict[str, Any]] = []
        for risk in list(brief.active_risks or [])[:24]:
            if "CONTRADICTION" not in str(risk):
                continue
            citation = _claim_citation(brief.citations if isinstance(brief.citations, dict) else {}, str(risk))
            row = {
                "claim": str(risk),
                "reason": "unresolved_contradiction",
                "evidence_refs": list(citation),
                "evidence_items": list(citation.evidence_items),
                "span_grounded": bool(citation.span_grounded),
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
        except Exception:
            pass

        if callable(self.hooks.on_handoff):
            try:
                self.hooks.on_handoff(dict(payload))
            except Exception as exc:
                self._record_fail_open("hook_on_handoff", str(exc))

        return packet

    def import_handoff(self, payload_or_path: str | Dict[str, Any], *, source_agent_id: str = "external_agent") -> Dict[str, Any]:
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

        thread_id = safe_text(packet.get("thread_id"), "", 128)
        packet_id = safe_text(packet.get("packet_id"), "", 128)
        scope = safe_text(packet.get("scope"), "", 1024)
        evidence_items = _flatten_handoff_evidence_items(packet)
        out = self.ingest_event(
            {
                "thread_id": thread_id,
                "event_type": "reflection",
                "role": safe_text(packet.get("role"), "unknown", 64),
                "source_agent_id": safe_text(source_agent_id, "external_agent", 128),
                "source_turn_id": packet_id,
                "payload": {
                    "claim": f"imported handoff packet {packet_id}",
                    "scope": scope,
                    "claims": [safe_text(x.get("claim"), "", 2048) for x in packet.get("claims", []) if isinstance(x, dict)],
                    "next_actions": list(packet.get("required_validations", []) or []),
                    "objective_id": f"agent_handoff::{safe_text(thread_id, 'thread', 64)}",
                },
                "evidence_refs": merge_evidence_refs(packet.get("evidence_refs", []), evidence_items),
                "evidence_items": evidence_items,
                "confidence": 0.65,
                "salience": 0.55,
                "metadata": {"imported_packet_id": packet_id},
            }
        )

        imported_state_rows = 0
        for row_type in ("claims", "unresolved_contradictions"):
            for idx, row in enumerate(packet.get(row_type, []) if isinstance(packet.get(row_type, []), list) else []):
                if not isinstance(row, dict):
                    continue
                state_context = normalize_state_context(row.get("state_context")) if "state_context" in row else {}
                if not state_context:
                    continue
                result = self.ingest_event(
                    {
                        "thread_id": thread_id,
                        "event_type": "reflection",
                        "role": safe_text(packet.get("role"), "unknown", 64),
                        "source_agent_id": safe_text(source_agent_id, "external_agent", 128),
                        "source_turn_id": f"{packet_id}:{row_type}:{idx}",
                        "payload": {
                            "claim": safe_text(row.get("claim"), f"imported {row_type} row", 2048),
                            "fact_state": (safe_text(row.get("status"), "asserted", 32).lower() if safe_text(row.get("status"), "", 32).lower() in {"asserted", "verified", "contradicted", "superseded"} else "asserted"),
                        },
                        "evidence_refs": merge_evidence_refs(row.get("evidence_refs", []), row.get("evidence_items")),
                        "evidence_items": normalize_evidence_items(row.get("evidence_items")),
                        "state_context": state_context,
                        "confidence": 0.6,
                        "salience": 0.5,
                        "metadata": {
                            "imported_packet_id": packet_id,
                            "imported_row_index": idx,
                            "imported_row_type": row_type,
                        },
                    }
                )
                if result.get("status") in {"ok", "deduped"}:
                    imported_state_rows += 1
        return {"status": "ok", "ingest": out, "packet_id": packet_id, "state_rows_imported": imported_state_rows}

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
        view_name = safe_text(view, "", 64)
        owner = safe_text(owner_agent_id, "", 128)
        if not bool(self.config.enabled):
            return AgentControlView(view=view_name, thread_id=safe_text(thread_id, "", 128), owner_agent_id=owner)

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

        verified_facts = [f for f in facts if str(f.get("state", "")).lower() == "verified"]
        contradicted = [f for f in facts if str(f.get("state", "")).lower() == "contradicted"]
        unresolved = [f for f in contradicted if not str(f.get("supersedes_fact_id", "")).strip()]

        brief_files = self.store.iter_brief_files()
        handoff_files = self.store.iter_handoff_files()

        brief_recent = 0
        brief_stale = 0
        citation_ref_total = 0
        citation_ref_covered = 0
        citation_span_total = 0
        citation_span_covered = 0
        citation_excerpt_total = 0
        citation_excerpt_covered = 0
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
                continue

        handoff_recent = 0
        handoff_complete = 0
        handoff_span_total = 0
        handoff_span_covered = 0
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
                continue

        active_view = self.query_view("active_scopes", include_closed=False, limit=max(1, int(self.config.max_event_tail)), lookback_hours=24.0).to_dict()
        open_gates_view = self.query_view("open_gates", include_closed=False, limit=max(1, int(self.config.max_event_tail)), lookback_hours=24.0).to_dict()
        active_rows = active_view.get("rows", []) if isinstance(active_view.get("rows", []), list) else []
        open_gate_rows = open_gates_view.get("rows", []) if isinstance(open_gates_view.get("rows", []), list) else []
        owner_covered = sum(1 for row in active_rows if safe_text((row or {}).get("owner_agent_id"), "", 128))
        freshness_covered = sum(1 for row in active_rows if safe_text((row or {}).get("freshness_state"), "", 16) != "unknown")
        stale_rows = sum(1 for row in active_rows if safe_text((row or {}).get("freshness_state"), "", 16) == "stale")
        conflict_rows = sum(1 for row in active_rows if bool((row or {}).get("conflict")))

        last_update_ts = safe_text(self.store.state.get("last_update_ts"), "", 64)
        state_age = None
        if last_update_ts:
            try:
                state_age = now_ts - (parse_ts_value(last_update_ts) or now_ts)
            except Exception:
                state_age = None

        return {
            "enabled": bool(self.config.enabled),
            "events_total": int(max(0, int(self.store.state.get("events_total", 0)))),
            "facts_total": int(max(0, int(self.store.state.get("facts_total", 0)))),
            "agent_memory_events_24h": int(len(events)),
            "agent_memory_verified_facts_24h": int(len(verified_facts)),
            "agent_memory_contradiction_count_24h": int(len(contradicted)),
            "agent_memory_unresolved_contradiction_ratio_24h": (float(len(unresolved) / max(1, len(contradicted))) if len(contradicted) > 0 else 0.0),
            "agent_brief_generation_success_rate_24h": (float(brief_recent / max(1, brief_recent)) if brief_recent > 0 else 0.0),
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
            "agent_memory_fail_open_events_24h": int(max(0, int(self.store.state.get("fail_open_events_total", 0)))),
            "agent_memory_last_update_ts": last_update_ts or None,
            "agent_memory_state_age_sec": (float(max(0.0, state_age)) if state_age is not None else None),
        }

    def flush(self) -> Dict[str, Any]:
        return self.store.flush()
