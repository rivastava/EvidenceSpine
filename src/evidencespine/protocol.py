from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


EVENT_LIFECYCLE = ("intent", "decision", "action", "outcome", "reflection")
FACT_STATES = ("asserted", "verified", "contradicted", "superseded")
AGENT_ROLES = ("implementer", "auditor", "researcher", "operator", "unknown")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_text(value: Any, default: str = "", limit: int = 2048) -> str:
    text = str(value if value is not None else default).strip()
    if not text:
        text = default
    return text[:limit]


def safe_float(value: Any, default: float = 0.0, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out != out:
        out = float(default)
    if out < lo:
        out = lo
    if out > hi:
        out = hi
    return float(out)


def normalize_refs(refs: Any) -> List[str]:
    if isinstance(refs, (list, tuple, set)):
        out = [safe_text(x, "", 512) for x in refs]
    elif refs is None:
        out = []
    else:
        out = [safe_text(refs, "", 512)]
    return [x for x in out if x]


def _canonical_hash(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8", errors="ignore")).hexdigest()


@dataclass
class AgentMemoryEvent:
    event_id: str
    thread_id: str
    event_type: str
    role: str = "unknown"
    source_agent_id: str = "unknown"
    source_turn_id: str = ""
    ts_utc: str = field(default_factory=utc_now_iso)
    payload: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: List[str] = field(default_factory=list)
    confidence: float = 0.5
    salience: float = 0.5
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized_event_type(self) -> str:
        et = safe_text(self.event_type, "reflection", 64).lower()
        return et if et in EVENT_LIFECYCLE else "reflection"

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "schema_version": "v1",
            "event_id": safe_text(self.event_id, "", 128),
            "thread_id": safe_text(self.thread_id, "", 128),
            "event_type": self.normalized_event_type(),
            "role": safe_text(self.role, "unknown", 64).lower(),
            "source_agent_id": safe_text(self.source_agent_id, "unknown", 128),
            "source_turn_id": safe_text(self.source_turn_id, "", 128),
            "ts_utc": safe_text(self.ts_utc, utc_now_iso(), 64),
            "payload": dict(self.payload or {}),
            "evidence_refs": normalize_refs(self.evidence_refs),
            "confidence": safe_float(self.confidence, 0.5, 0.0, 1.0),
            "salience": safe_float(self.salience, 0.5, 0.0, 1.0),
            "tags": [safe_text(x, "", 64) for x in list(self.tags or []) if safe_text(x, "", 64)],
            "metadata": dict(self.metadata or {}),
        }
        if payload["role"] not in AGENT_ROLES:
            payload["role"] = "unknown"
        hash_payload = {
            "thread_id": payload["thread_id"],
            "event_type": payload["event_type"],
            "source_agent_id": payload["source_agent_id"],
            "source_turn_id": payload["source_turn_id"],
            "payload": payload["payload"],
            "evidence_refs": payload["evidence_refs"],
        }
        payload["event_hash"] = _canonical_hash(hash_payload)
        if not payload["event_id"]:
            payload["event_id"] = f"ame_{payload['event_hash'][:16]}"
        return payload


@dataclass
class AgentMemoryFact:
    fact_id: str
    thread_id: str
    claim: str
    state: str = "asserted"
    source_agent_id: str = "unknown"
    source_turn_id: str = ""
    ts_utc: str = field(default_factory=utc_now_iso)
    evidence_refs: List[str] = field(default_factory=list)
    confidence: float = 0.5
    tags: List[str] = field(default_factory=list)
    contradiction_refs: List[str] = field(default_factory=list)
    supersedes_fact_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        state = safe_text(self.state, "asserted", 32).lower()
        if state not in FACT_STATES:
            state = "asserted"
        payload = {
            "schema_version": "v1",
            "fact_id": safe_text(self.fact_id, "", 128),
            "thread_id": safe_text(self.thread_id, "", 128),
            "claim": safe_text(self.claim, "", 4096),
            "state": state,
            "source_agent_id": safe_text(self.source_agent_id, "unknown", 128),
            "source_turn_id": safe_text(self.source_turn_id, "", 128),
            "ts_utc": safe_text(self.ts_utc, utc_now_iso(), 64),
            "evidence_refs": normalize_refs(self.evidence_refs),
            "confidence": safe_float(self.confidence, 0.5, 0.0, 1.0),
            "tags": [safe_text(x, "", 64) for x in list(self.tags or []) if safe_text(x, "", 64)],
            "contradiction_refs": normalize_refs(self.contradiction_refs),
            "supersedes_fact_id": safe_text(self.supersedes_fact_id, "", 128),
            "metadata": dict(self.metadata or {}),
        }
        if not payload["fact_id"]:
            hash_payload = {
                "thread_id": payload["thread_id"],
                "claim": payload["claim"],
                "source_turn_id": payload["source_turn_id"],
            }
            payload["fact_id"] = f"amf_{_canonical_hash(hash_payload)[:16]}"
        return payload


@dataclass
class AgentConversationBrief:
    thread_id: str
    query: str
    generated_at: str = field(default_factory=utc_now_iso)
    token_budget: int = 1800
    current_goal: List[str] = field(default_factory=list)
    locked_decisions: List[str] = field(default_factory=list)
    recent_verified_facts: List[str] = field(default_factory=list)
    active_risks: List[str] = field(default_factory=list)
    open_items: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)
    citations: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "v1",
            "thread_id": safe_text(self.thread_id, "", 128),
            "query": safe_text(self.query, "", 1024),
            "generated_at": safe_text(self.generated_at, utc_now_iso(), 64),
            "token_budget": int(max(64, int(self.token_budget or 1800))),
            "current_goal": [safe_text(x, "", 1024) for x in list(self.current_goal or []) if safe_text(x, "", 1024)],
            "locked_decisions": [safe_text(x, "", 1024) for x in list(self.locked_decisions or []) if safe_text(x, "", 1024)],
            "recent_verified_facts": [safe_text(x, "", 1024) for x in list(self.recent_verified_facts or []) if safe_text(x, "", 1024)],
            "active_risks": [safe_text(x, "", 1024) for x in list(self.active_risks or []) if safe_text(x, "", 1024)],
            "open_items": [safe_text(x, "", 1024) for x in list(self.open_items or []) if safe_text(x, "", 1024)],
            "next_actions": [safe_text(x, "", 1024) for x in list(self.next_actions or []) if safe_text(x, "", 1024)],
            "citations": {
                safe_text(k, "", 1024): normalize_refs(v)
                for k, v in dict(self.citations or {}).items()
                if safe_text(k, "", 1024)
            },
            "metadata": dict(self.metadata or {}),
        }


@dataclass
class AgentHandoffPacket:
    packet_id: str
    role: str
    thread_id: str
    scope: str
    generated_at: str = field(default_factory=utc_now_iso)
    locked_decisions: List[str] = field(default_factory=list)
    claims: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_contradictions: List[Dict[str, Any]] = field(default_factory=list)
    required_validations: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)
    source_snapshot: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        role = safe_text(self.role, "unknown", 64).lower()
        if role not in AGENT_ROLES:
            role = "unknown"
        payload = {
            "schema_version": "v1",
            "packet_id": safe_text(self.packet_id, "", 128),
            "role": role,
            "thread_id": safe_text(self.thread_id, "", 128),
            "scope": safe_text(self.scope, "", 2048),
            "generated_at": safe_text(self.generated_at, utc_now_iso(), 64),
            "locked_decisions": [safe_text(x, "", 1024) for x in list(self.locked_decisions or []) if safe_text(x, "", 1024)],
            "claims": [dict(x) for x in list(self.claims or []) if isinstance(x, dict)],
            "unresolved_contradictions": [dict(x) for x in list(self.unresolved_contradictions or []) if isinstance(x, dict)],
            "required_validations": [safe_text(x, "", 1024) for x in list(self.required_validations or []) if safe_text(x, "", 1024)],
            "evidence_refs": normalize_refs(self.evidence_refs),
            "source_snapshot": dict(self.source_snapshot or {}),
            "metadata": dict(self.metadata or {}),
        }
        if not payload["packet_id"]:
            hash_payload = {
                "thread_id": payload["thread_id"],
                "role": payload["role"],
                "scope": payload["scope"],
                "claims": payload["claims"],
            }
            payload["packet_id"] = f"ahp_{_canonical_hash(hash_payload)[:16]}"
        payload["checksum"] = _canonical_hash(payload)
        return payload


def _validate_required(payload: Dict[str, Any], required: Tuple[str, ...]) -> List[str]:
    errors: List[str] = []
    for key in required:
        if key not in payload:
            errors.append(f"missing:{key}")
            continue
        if key in {"thread_id", "event_type", "claim", "role", "scope"} and not safe_text(payload.get(key), "", 1024):
            errors.append(f"empty:{key}")
    return errors


def validate_event_dict(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    if not isinstance(payload, dict):
        return False, ["not_a_dict"]
    errors = _validate_required(payload, ("thread_id", "event_type", "source_agent_id", "source_turn_id"))
    et = safe_text(payload.get("event_type"), "", 64).lower()
    if et not in EVENT_LIFECYCLE:
        errors.append("invalid:event_type")
    return len(errors) == 0, errors


def validate_fact_dict(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    if not isinstance(payload, dict):
        return False, ["not_a_dict"]
    errors = _validate_required(payload, ("thread_id", "claim", "state"))
    state = safe_text(payload.get("state"), "", 32).lower()
    if state not in FACT_STATES:
        errors.append("invalid:state")
    return len(errors) == 0, errors


def validate_brief_dict(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    if not isinstance(payload, dict):
        return False, ["not_a_dict"]
    errors = _validate_required(payload, ("thread_id", "query", "generated_at"))
    citations = payload.get("citations", {})
    if not isinstance(citations, dict):
        errors.append("invalid:citations")
    return len(errors) == 0, errors


def validate_handoff_dict(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    if not isinstance(payload, dict):
        return False, ["not_a_dict"]
    errors = _validate_required(payload, ("role", "thread_id", "scope", "claims"))
    if safe_text(payload.get("role"), "", 64).lower() not in AGENT_ROLES:
        errors.append("invalid:role")
    if not isinstance(payload.get("claims"), list):
        errors.append("invalid:claims")
    return len(errors) == 0, errors


def event_to_fact_candidates(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(event, dict):
        return []
    payload = event.get("payload", {}) if isinstance(event.get("payload", {}), dict) else {}
    thread_id = safe_text(event.get("thread_id"), "", 128)
    source_agent_id = safe_text(event.get("source_agent_id"), "unknown", 128)
    source_turn_id = safe_text(event.get("source_turn_id"), "", 128)
    evidence_refs = normalize_refs(event.get("evidence_refs"))

    claims: List[str] = []
    if isinstance(payload.get("claims"), list):
        claims.extend([safe_text(x, "", 2048) for x in payload.get("claims", [])])
    if safe_text(payload.get("claim"), "", 2048):
        claims.append(safe_text(payload.get("claim"), "", 2048))
    if safe_text(payload.get("decision"), "", 2048):
        claims.append(safe_text(payload.get("decision"), "", 2048))
    if safe_text(payload.get("outcome"), "", 2048):
        claims.append(safe_text(payload.get("outcome"), "", 2048))

    state = safe_text(payload.get("fact_state"), "asserted", 32).lower()
    if state not in FACT_STATES:
        state = "asserted"

    dedup: Dict[str, Dict[str, Any]] = {}
    for claim in claims:
        c = safe_text(claim, "", 2048)
        if not c:
            continue
        key = _canonical_hash({"thread_id": thread_id, "claim": c})
        dedup[key] = {
            "thread_id": thread_id,
            "claim": c,
            "state": state,
            "source_agent_id": source_agent_id,
            "source_turn_id": source_turn_id,
            "evidence_refs": list(evidence_refs),
            "confidence": safe_float(payload.get("confidence", event.get("confidence", 0.5)), 0.5, 0.0, 1.0),
            "tags": [safe_text(x, "", 64) for x in list(payload.get("tags", []) or []) if safe_text(x, "", 64)],
            "metadata": {
                "event_id": safe_text(event.get("event_id"), "", 128),
                "event_type": safe_text(event.get("event_type"), "reflection", 64).lower(),
            },
        }
    return list(dedup.values())
