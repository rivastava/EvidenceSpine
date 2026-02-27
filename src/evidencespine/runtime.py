from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from evidencespine.protocol import (
    AgentConversationBrief,
    AgentHandoffPacket,
    AgentMemoryEvent,
    AgentMemoryFact,
    event_to_fact_candidates,
    normalize_refs,
    safe_text,
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


@dataclass
class RuntimeHooks:
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None
    on_brief: Optional[Callable[[Dict[str, Any]], None]] = None
    on_handoff: Optional[Callable[[Dict[str, Any]], None]] = None
    contradiction_pass: Optional[Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]]] = None


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
            "ts": datetime.utcnow().isoformat() + "Z",
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

    def ingest_event(self, event: AgentMemoryEvent | Dict[str, Any]) -> Dict[str, Any]:
        if not bool(self.config.enabled):
            return {"status": "disabled", "reason": "agent_memory_disabled"}
        try:
            if isinstance(event, AgentMemoryEvent):
                row = event.to_dict()
            else:
                row = AgentMemoryEvent(
                    event_id=safe_text((event or {}).get("event_id"), "", 128),
                    thread_id=safe_text((event or {}).get("thread_id"), "", 128),
                    event_type=safe_text((event or {}).get("event_type"), "reflection", 64),
                    role=safe_text((event or {}).get("role"), "unknown", 64),
                    source_agent_id=safe_text((event or {}).get("source_agent_id"), "unknown", 128),
                    source_turn_id=safe_text((event or {}).get("source_turn_id"), "", 128),
                    ts_utc=safe_text((event or {}).get("ts_utc"), "", 64),
                    payload=dict((event or {}).get("payload", {}) or {}),
                    evidence_refs=normalize_refs((event or {}).get("evidence_refs")),
                    confidence=float((event or {}).get("confidence", 0.5)),
                    salience=float((event or {}).get("salience", 0.5)),
                    tags=list((event or {}).get("tags", []) or []),
                    metadata=dict((event or {}).get("metadata", {}) or {}),
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
                        confidence=float(fact.get("confidence", row.get("confidence", 0.5))),
                        tags=list(fact.get("tags", []) or []),
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
                v = row.get("ts")
                if v is None:
                    v = row.get("ts_utc")
                try:
                    ts = float(v)
                except Exception:
                    try:
                        ts = datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
                    except Exception:
                        ts = 0.0
                latest_ts = max(latest_ts, ts)
            if latest_ts > 0.0:
                stale = bool((now_ts - latest_ts) > float(max(300.0, self.config.dedupe_window_sec)))
            brief.metadata["stale"] = bool(stale)
            brief.metadata["unresolved_contradictions"] = int(len(unresolved))
            self.store.write_brief(thread_id, brief.to_dict())

            claim_total = 0
            claim_covered = 0
            for section in [
                brief.current_goal,
                brief.locked_decisions,
                brief.recent_verified_facts,
                brief.active_risks,
                brief.open_items,
                brief.next_actions,
            ]:
                for claim in section:
                    claim_total += 1
                    refs = brief.citations.get(claim, []) if isinstance(brief.citations, dict) else []
                    if isinstance(refs, list) and len(refs) > 0:
                        claim_covered += 1
            self.store.record_brief_stats(
                attempt=False,
                success=True,
                stale=bool(stale),
                citation_total=claim_total,
                citation_covered=claim_covered,
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
                self.store.write_brief(thread_id, fallback.to_dict())
                self.store.record_brief_stats(
                    attempt=False,
                    success=True,
                    stale=True,
                    citation_total=0,
                    citation_covered=0,
                )
                return fallback
            raise

    def emit_handoff(self, role: str, thread_id: str, scope: str = "cross-agent coordination") -> AgentHandoffPacket:
        brief = self.build_brief(thread_id=thread_id, query=f"handoff for {role}")
        claims: List[Dict[str, Any]] = []
        for claim in list(brief.recent_verified_facts or [])[:24]:
            claims.append(
                {
                    "claim": claim,
                    "evidence_refs": list((brief.citations or {}).get(claim, []) or []),
                    "status": "verified",
                }
            )
        if len(claims) == 0:
            fallback_claims = list(brief.locked_decisions or []) + list(brief.open_items or [])
            for claim in fallback_claims[:24]:
                claims.append(
                    {
                        "claim": claim,
                        "evidence_refs": list((brief.citations or {}).get(claim, []) or []),
                        "status": "asserted",
                    }
                )
        unresolved: List[Dict[str, Any]] = []
        for risk in list(brief.active_risks or [])[:24]:
            if "CONTRADICTION" not in str(risk):
                continue
            unresolved.append(
                {
                    "claim": str(risk),
                    "reason": "unresolved_contradiction",
                    "evidence_refs": list((brief.citations or {}).get(risk, []) or []),
                }
            )

        evidence_refs: List[str] = []
        for row in claims:
            evidence_refs.extend(normalize_refs(row.get("evidence_refs")))
        for row in unresolved:
            evidence_refs.extend(normalize_refs(row.get("evidence_refs")))
        evidence_refs = list(dict.fromkeys([x for x in evidence_refs if x]))
        required_validations = list(brief.open_items or [])
        if len(required_validations) == 0:
            required_validations = [f"Validate scope: {safe_text(scope, 'cross-agent coordination', 256)}"]

        snapshot = {
            "events_total": int(max(0, int(self.store.state.get("events_total", 0)))),
            "facts_total": int(max(0, int(self.store.state.get("facts_total", 0)))),
            "brief_generated_at": safe_text(brief.generated_at, "", 64),
            "last_update_ts": safe_text(self.store.state.get("last_update_ts"), "", 64),
        }

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
            source_snapshot=snapshot,
            metadata={
                "brief_token_budget": int(brief.token_budget),
                "brief_token_used_estimate": int((brief.metadata or {}).get("token_used_estimate", 0)),
                "citation_claim_total": int(max(0, int(self.store.state.get("citation_claim_total", 0)))),
                "citation_claim_covered_total": int(max(0, int(self.store.state.get("citation_claim_covered_total", 0)))),
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
                "evidence_refs": normalize_refs(packet.get("evidence_refs", [])),
                "confidence": 0.65,
                "salience": 0.55,
                "metadata": {"imported_packet_id": packet_id},
            }
        )
        return {"status": "ok", "ingest": out, "packet_id": packet_id}

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
        citation_claim_total = 0
        citation_claim_covered = 0
        for path in brief_files[-1024:]:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    row = json.load(handle)
                ts = datetime.fromisoformat(str(row.get("generated_at", "")).replace("Z", "+00:00")).timestamp()
                if ts < cutoff_ts:
                    continue
                brief_recent += 1
                meta = row.get("metadata", {}) if isinstance(row.get("metadata", {}), dict) else {}
                if bool(meta.get("stale", False)):
                    brief_stale += 1
                citations = row.get("citations", {}) if isinstance(row.get("citations", {}), dict) else {}
                for section in [
                    row.get("current_goal", []),
                    row.get("locked_decisions", []),
                    row.get("recent_verified_facts", []),
                    row.get("active_risks", []),
                    row.get("open_items", []),
                    row.get("next_actions", []),
                ]:
                    for claim in section if isinstance(section, list) else []:
                        citation_claim_total += 1
                        refs = citations.get(claim, []) if isinstance(citations, dict) else []
                        if isinstance(refs, list) and len(refs) > 0:
                            citation_claim_covered += 1
            except Exception:
                continue

        handoff_recent = 0
        handoff_complete = 0
        for path in handoff_files[-1024:]:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    row = json.load(handle)
                ts = datetime.fromisoformat(str(row.get("generated_at", "")).replace("Z", "+00:00")).timestamp()
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
            except Exception:
                continue

        last_update_ts = safe_text(self.store.state.get("last_update_ts"), "", 64)
        state_age = None
        if last_update_ts:
            try:
                state_age = now_ts - datetime.fromisoformat(last_update_ts.replace("Z", "+00:00")).timestamp()
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
            "agent_claim_citation_coverage_24h": (float(citation_claim_covered / max(1, citation_claim_total)) if citation_claim_total > 0 else 0.0),
            "agent_memory_fail_open_events_24h": int(max(0, int(self.store.state.get("fail_open_events_total", 0)))),
            "agent_memory_last_update_ts": last_update_ts or None,
            "agent_memory_state_age_sec": (float(max(0.0, state_age)) if state_age is not None else None),
        }

    def flush(self) -> Dict[str, Any]:
        return self.store.flush()
