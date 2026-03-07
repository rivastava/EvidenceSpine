from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
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


def safe_int(value: Any, default: int | None = None, minimum: int | None = None, maximum: int | None = None) -> int | None:
    if value is None or value == "":
        out = default
    else:
        try:
            out = int(value)
        except Exception:
            out = default
    if out is None:
        return None
    if minimum is not None and out < minimum:
        out = minimum
    if maximum is not None and out > maximum:
        out = maximum
    return int(out)


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


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    return list(dict.fromkeys([safe_text(x, "", 512) for x in values if safe_text(x, "", 512)]))


@dataclass(frozen=True)
class EvidenceItem:
    source_id: str
    locator: str = ""
    char_start: int | None = None
    char_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    excerpt: str = ""
    checksum: str = ""
    confidence: float | None = None
    verification_state: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "source_id": safe_text(self.source_id, "", 512),
        }
        locator = safe_text(self.locator, "", 512)
        if locator:
            payload["locator"] = locator

        line_start = safe_int(self.line_start, None, 1, None)
        line_end = safe_int(self.line_end, line_start, 1, None) if line_start is not None else None
        if line_start is not None:
            payload["line_start"] = line_start
            payload["line_end"] = max(line_start, line_end if line_end is not None else line_start)

        char_start = safe_int(self.char_start, None, 0, None)
        char_end = safe_int(self.char_end, char_start, 0, None) if char_start is not None else None
        if char_start is not None:
            payload["char_start"] = char_start
            payload["char_end"] = max(char_start, char_end if char_end is not None else char_start)

        excerpt = safe_text(self.excerpt, "", 4096)
        if excerpt:
            payload["excerpt"] = excerpt

        checksum = safe_text(self.checksum, "", 256)
        if checksum:
            payload["checksum"] = checksum

        if self.confidence is not None:
            payload["confidence"] = safe_float(self.confidence, 0.0, 0.0, 1.0)

        verification_state = safe_text(self.verification_state, "", 32).lower()
        if verification_state in FACT_STATES:
            payload["verification_state"] = verification_state

        metadata = dict(self.metadata or {})
        if metadata:
            payload["metadata"] = metadata
        return payload


def _normalize_evidence_item(value: Any) -> Dict[str, Any]:
    if isinstance(value, EvidenceItem):
        return value.to_dict()
    if not isinstance(value, dict):
        return {}
    item = EvidenceItem(
        source_id=safe_text(value.get("source_id"), "", 512),
        locator=safe_text(value.get("locator"), "", 512),
        char_start=safe_int(value.get("char_start"), None, 0, None),
        char_end=safe_int(value.get("char_end"), None, 0, None),
        line_start=safe_int(value.get("line_start"), None, 1, None),
        line_end=safe_int(value.get("line_end"), None, 1, None),
        excerpt=safe_text(value.get("excerpt"), "", 4096),
        checksum=safe_text(value.get("checksum"), "", 256),
        confidence=(value.get("confidence") if value.get("confidence") is not None else None),
        verification_state=safe_text(value.get("verification_state"), "", 32).lower(),
        metadata=dict(value.get("metadata", {}) or {}),
    )
    return item.to_dict()


def validate_evidence_item_dict(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    if not isinstance(payload, dict):
        return False, ["not_a_dict"]
    errors: List[str] = []
    source_id = safe_text(payload.get("source_id"), "", 512)
    if not source_id:
        errors.append("missing:source_id")

    locator = safe_text(payload.get("locator"), "", 512)
    line_start = safe_int(payload.get("line_start"), None, 1, None)
    line_end = safe_int(payload.get("line_end"), None, 1, None)
    char_start = safe_int(payload.get("char_start"), None, 0, None)
    char_end = safe_int(payload.get("char_end"), None, 0, None)

    if not locator and line_start is None and char_start is None:
        errors.append("missing:anchor")
    if line_end is not None and line_start is None:
        errors.append("missing:line_start")
    if char_end is not None and char_start is None:
        errors.append("missing:char_start")
    if line_start is not None and line_end is not None and line_end < line_start:
        errors.append("invalid:line_range")
    if char_start is not None and char_end is not None and char_end < char_start:
        errors.append("invalid:char_range")

    verification_state = safe_text(payload.get("verification_state"), "", 32).lower()
    if verification_state and verification_state not in FACT_STATES:
        errors.append("invalid:verification_state")
    return len(errors) == 0, errors


def normalize_evidence_items(items: Any) -> List[Dict[str, Any]]:
    if items is None:
        return []
    if isinstance(items, (list, tuple, set)):
        raw_items = list(items)
    else:
        raw_items = [items]
    out: List[Dict[str, Any]] = []
    for raw in raw_items:
        item = _normalize_evidence_item(raw)
        if not item:
            continue
        out.append(item)
    return out


def evidence_item_to_ref(item: Any) -> str:
    payload = _normalize_evidence_item(item)
    source_id = safe_text(payload.get("source_id"), "", 512)
    if not source_id:
        return ""
    locator = safe_text(payload.get("locator"), "", 512)
    if locator:
        return locator
    line_start = safe_int(payload.get("line_start"), None, 1, None)
    line_end = safe_int(payload.get("line_end"), line_start, 1, None) if line_start is not None else None
    if line_start is not None:
        if line_end is not None and line_end != line_start:
            return f"{source_id}#L{line_start}-L{line_end}"
        return f"{source_id}#L{line_start}"
    char_start = safe_int(payload.get("char_start"), None, 0, None)
    char_end = safe_int(payload.get("char_end"), char_start, 0, None) if char_start is not None else None
    if char_start is not None:
        if char_end is not None and char_end != char_start:
            return f"{source_id}#C{char_start}-C{char_end}"
        return f"{source_id}#C{char_start}"
    return source_id


def merge_evidence_refs(refs: Any, evidence_items: Any = None) -> List[str]:
    merged = list(normalize_refs(refs))
    for item in normalize_evidence_items(evidence_items):
        ref = evidence_item_to_ref(item)
        if ref:
            merged.append(ref)
    return _dedupe_preserve_order(merged)


def has_grounded_span(evidence_items: Any) -> bool:
    for item in normalize_evidence_items(evidence_items):
        ok, _ = validate_evidence_item_dict(item)
        if ok:
            return True
    return False


def evidence_item_excerpt_matches_checksum(item: Any) -> bool:
    payload = _normalize_evidence_item(item)
    excerpt = safe_text(payload.get("excerpt"), "", 4096)
    checksum = safe_text(payload.get("checksum"), "", 256).lower()
    if not excerpt or not checksum:
        return False
    digest = hashlib.sha256(excerpt.encode("utf-8", errors="ignore")).hexdigest()
    return checksum in {digest, f"sha256:{digest}"}


@dataclass
class ClaimCitation(Sequence[str]):
    primary_ref: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    evidence_items: List[Dict[str, Any]] = field(default_factory=list)
    span_grounded: bool = False

    @classmethod
    def from_value(cls, value: Any, *, fallback_ref: str = "") -> "ClaimCitation":
        if isinstance(value, ClaimCitation):
            refs = merge_evidence_refs(value.evidence_refs, value.evidence_items)
            primary_ref = safe_text(value.primary_ref, "", 512) or (refs[0] if refs else safe_text(fallback_ref, "", 512))
            return cls(
                primary_ref=primary_ref,
                evidence_refs=refs,
                evidence_items=normalize_evidence_items(value.evidence_items),
                span_grounded=bool(value.span_grounded or has_grounded_span(value.evidence_items)),
            )
        if isinstance(value, dict):
            evidence_items = normalize_evidence_items(value.get("evidence_items"))
            refs = merge_evidence_refs(value.get("evidence_refs"), evidence_items)
            primary_ref = safe_text(value.get("primary_ref"), "", 512) or (refs[0] if refs else safe_text(fallback_ref, "", 512))
            return cls(
                primary_ref=primary_ref,
                evidence_refs=refs,
                evidence_items=evidence_items,
                span_grounded=bool(value.get("span_grounded")) or has_grounded_span(evidence_items),
            )
        refs = normalize_refs(value)
        return cls(
            primary_ref=(refs[0] if refs else safe_text(fallback_ref, "", 512)),
            evidence_refs=refs,
            evidence_items=[],
            span_grounded=False,
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self.evidence_refs)

    def __len__(self) -> int:
        return len(self.evidence_refs)

    def __getitem__(self, index: int) -> str:
        return self.evidence_refs[index]

    def primary_evidence_item(self) -> Dict[str, Any] | None:
        if not self.evidence_items:
            return None
        return dict(self.evidence_items[0])

    def to_dict(self) -> Dict[str, Any]:
        evidence_items = normalize_evidence_items(self.evidence_items)
        evidence_refs = merge_evidence_refs(self.evidence_refs, evidence_items)
        primary_ref = safe_text(self.primary_ref, "", 512) or (evidence_refs[0] if evidence_refs else "")
        return {
            "primary_ref": primary_ref,
            "evidence_refs": evidence_refs,
            "evidence_items": evidence_items,
            "span_grounded": bool(self.span_grounded or has_grounded_span(evidence_items)),
        }


def _validate_evidence_items_field(payload: Dict[str, Any], key: str) -> List[str]:
    if key not in payload:
        return []
    raw_items = payload.get(key)
    if raw_items is None:
        return []
    if not isinstance(raw_items, (list, tuple, set)):
        raw_items = [raw_items]
    errors: List[str] = []
    for idx, raw in enumerate(list(raw_items)):
        ok, item_errors = validate_evidence_item_dict(_normalize_evidence_item(raw))
        if not ok:
            errors.extend([f"invalid:{key}[{idx}]:{err}" for err in item_errors])
    return errors


def _validate_claim_citation_value(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set, str)) or value is None:
        return []
    if isinstance(value, ClaimCitation):
        return []
    if not isinstance(value, dict):
        return ["invalid:citation"]
    errors: List[str] = []
    if "evidence_refs" in value and not isinstance(value.get("evidence_refs"), (list, tuple, set)):
        errors.append("invalid:citation:evidence_refs")
    errors.extend(_validate_evidence_items_field(value, "evidence_items"))
    return errors


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
    evidence_items: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    salience: float = 0.5
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized_event_type(self) -> str:
        et = safe_text(self.event_type, "reflection", 64).lower()
        return et if et in EVENT_LIFECYCLE else "reflection"

    def to_dict(self) -> Dict[str, Any]:
        evidence_items = normalize_evidence_items(self.evidence_items)
        evidence_refs = merge_evidence_refs(self.evidence_refs, evidence_items)
        payload = {
            "schema_version": "v2",
            "event_id": safe_text(self.event_id, "", 128),
            "thread_id": safe_text(self.thread_id, "", 128),
            "event_type": self.normalized_event_type(),
            "role": safe_text(self.role, "unknown", 64).lower(),
            "source_agent_id": safe_text(self.source_agent_id, "unknown", 128),
            "source_turn_id": safe_text(self.source_turn_id, "", 128),
            "ts_utc": safe_text(self.ts_utc, utc_now_iso(), 64),
            "payload": dict(self.payload or {}),
            "evidence_refs": evidence_refs,
            "evidence_items": evidence_items,
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
            "evidence_items": payload["evidence_items"],
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
    evidence_items: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    tags: List[str] = field(default_factory=list)
    contradiction_refs: List[str] = field(default_factory=list)
    supersedes_fact_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        state = safe_text(self.state, "asserted", 32).lower()
        if state not in FACT_STATES:
            state = "asserted"
        evidence_items = normalize_evidence_items(self.evidence_items)
        payload = {
            "schema_version": "v2",
            "fact_id": safe_text(self.fact_id, "", 128),
            "thread_id": safe_text(self.thread_id, "", 128),
            "claim": safe_text(self.claim, "", 4096),
            "state": state,
            "source_agent_id": safe_text(self.source_agent_id, "unknown", 128),
            "source_turn_id": safe_text(self.source_turn_id, "", 128),
            "ts_utc": safe_text(self.ts_utc, utc_now_iso(), 64),
            "evidence_refs": merge_evidence_refs(self.evidence_refs, evidence_items),
            "evidence_items": evidence_items,
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
    citations: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        structured_citations: Dict[str, Dict[str, Any]] = {}
        citation_refs: Dict[str, List[str]] = {}
        for key, value in dict(self.citations or {}).items():
            claim = safe_text(key, "", 1024)
            if not claim:
                continue
            citation = ClaimCitation.from_value(value)
            structured_citations[claim] = citation.to_dict()
            citation_refs[claim] = list(citation)
        return {
            "schema_version": "v2",
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
            "citations": structured_citations,
            "citation_refs": citation_refs,
            "metadata": dict(self.metadata or {}),
        }


def _normalize_handoff_row(row: Dict[str, Any], *, require_claim: bool) -> Dict[str, Any]:
    payload = dict(row or {})
    evidence_items = normalize_evidence_items(payload.get("evidence_items"))
    payload["evidence_refs"] = merge_evidence_refs(payload.get("evidence_refs"), evidence_items)
    payload["evidence_items"] = evidence_items
    payload["span_grounded"] = bool(payload.get("span_grounded")) or has_grounded_span(evidence_items)

    if require_claim:
        payload["claim"] = safe_text(payload.get("claim"), "", 2048)
    else:
        claim = safe_text(payload.get("claim"), "", 2048)
        if claim:
            payload["claim"] = claim

    reason = safe_text(payload.get("reason"), "", 512)
    if reason:
        payload["reason"] = reason

    status = safe_text(payload.get("status"), "", 32).lower()
    if status:
        payload["status"] = status
    return payload


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
    evidence_items: List[Dict[str, Any]] = field(default_factory=list)
    source_snapshot: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        role = safe_text(self.role, "unknown", 64).lower()
        if role not in AGENT_ROLES:
            role = "unknown"
        claims = [_normalize_handoff_row(x, require_claim=True) for x in list(self.claims or []) if isinstance(x, dict)]
        claims = [x for x in claims if safe_text(x.get("claim"), "", 2048)]
        unresolved = [_normalize_handoff_row(x, require_claim=False) for x in list(self.unresolved_contradictions or []) if isinstance(x, dict)]
        evidence_items = normalize_evidence_items(self.evidence_items)
        payload = {
            "schema_version": "v2",
            "packet_id": safe_text(self.packet_id, "", 128),
            "role": role,
            "thread_id": safe_text(self.thread_id, "", 128),
            "scope": safe_text(self.scope, "", 2048),
            "generated_at": safe_text(self.generated_at, utc_now_iso(), 64),
            "locked_decisions": [safe_text(x, "", 1024) for x in list(self.locked_decisions or []) if safe_text(x, "", 1024)],
            "claims": claims,
            "unresolved_contradictions": unresolved,
            "required_validations": [safe_text(x, "", 1024) for x in list(self.required_validations or []) if safe_text(x, "", 1024)],
            "evidence_refs": merge_evidence_refs(self.evidence_refs, evidence_items),
            "evidence_items": evidence_items,
            "source_snapshot": dict(self.source_snapshot or {}),
            "metadata": dict(self.metadata or {}),
        }
        if not payload["packet_id"]:
            hash_payload = {
                "thread_id": payload["thread_id"],
                "role": payload["role"],
                "scope": payload["scope"],
                "claims": payload["claims"],
                "evidence_items": payload["evidence_items"],
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
    errors.extend(_validate_evidence_items_field(payload, "evidence_items"))
    return len(errors) == 0, errors


def validate_fact_dict(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    if not isinstance(payload, dict):
        return False, ["not_a_dict"]
    errors = _validate_required(payload, ("thread_id", "claim", "state"))
    state = safe_text(payload.get("state"), "", 32).lower()
    if state not in FACT_STATES:
        errors.append("invalid:state")
    errors.extend(_validate_evidence_items_field(payload, "evidence_items"))
    return len(errors) == 0, errors


def validate_brief_dict(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    if not isinstance(payload, dict):
        return False, ["not_a_dict"]
    errors = _validate_required(payload, ("thread_id", "query", "generated_at"))
    citations = payload.get("citations", {})
    if not isinstance(citations, dict):
        errors.append("invalid:citations")
    else:
        for key, value in citations.items():
            if not safe_text(key, "", 1024):
                errors.append("invalid:citation:key")
                continue
            errors.extend(_validate_claim_citation_value(value))
    citation_refs = payload.get("citation_refs", {})
    if citation_refs and not isinstance(citation_refs, dict):
        errors.append("invalid:citation_refs")
    return len(errors) == 0, errors


def validate_handoff_dict(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    if not isinstance(payload, dict):
        return False, ["not_a_dict"]
    errors = _validate_required(payload, ("role", "thread_id", "scope", "claims"))
    if safe_text(payload.get("role"), "", 64).lower() not in AGENT_ROLES:
        errors.append("invalid:role")
    if not isinstance(payload.get("claims"), list):
        errors.append("invalid:claims")
    else:
        for idx, row in enumerate(payload.get("claims", [])):
            if not isinstance(row, dict):
                errors.append(f"invalid:claims[{idx}]")
                continue
            if not safe_text(row.get("claim"), "", 2048):
                errors.append(f"invalid:claims[{idx}]:claim")
            errors.extend(_validate_evidence_items_field(row, "evidence_items"))
    unresolved = payload.get("unresolved_contradictions", [])
    if unresolved and not isinstance(unresolved, list):
        errors.append("invalid:unresolved_contradictions")
    elif isinstance(unresolved, list):
        for idx, row in enumerate(unresolved):
            if not isinstance(row, dict):
                errors.append(f"invalid:unresolved_contradictions[{idx}]")
                continue
            errors.extend(_validate_evidence_items_field(row, "evidence_items"))
    errors.extend(_validate_evidence_items_field(payload, "evidence_items"))
    return len(errors) == 0, errors


def event_to_fact_candidates(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(event, dict):
        return []
    payload = event.get("payload", {}) if isinstance(event.get("payload", {}), dict) else {}
    thread_id = safe_text(event.get("thread_id"), "", 128)
    source_agent_id = safe_text(event.get("source_agent_id"), "unknown", 128)
    source_turn_id = safe_text(event.get("source_turn_id"), "", 128)
    evidence_items = normalize_evidence_items(event.get("evidence_items"))
    evidence_refs = merge_evidence_refs(event.get("evidence_refs"), evidence_items)

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
            "evidence_items": list(evidence_items),
            "confidence": safe_float(payload.get("confidence", event.get("confidence", 0.5)), 0.5, 0.0, 1.0),
            "tags": [safe_text(x, "", 64) for x in list(payload.get("tags", []) or []) if safe_text(x, "", 64)],
            "metadata": {
                "event_id": safe_text(event.get("event_id"), "", 128),
                "event_type": safe_text(event.get("event_type"), "reflection", 64).lower(),
            },
        }
    return list(dedup.values())
