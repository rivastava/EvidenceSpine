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
STATE_SCOPE_KINDS = ("task", "gate", "blocker", "runtime_state", "thread")
STATE_KINDS = ("agent_local_work", "global_blocker", "pending_gate", "runtime_validated_state")
STATE_STATUSES = ("active", "blocked", "ready", "closed", "superseded")
STATE_BASES = ("reported", "runtime_validated", "derived", "imported")
FRESHNESS_STATES = ("fresh", "stale", "unknown")
LEASE_STATES = ("active", "expired", "none")

_STATE_SCOPE_KIND_DEFAULTS = {
    "agent_local_work": "task",
    "global_blocker": "blocker",
    "pending_gate": "gate",
    "runtime_validated_state": "runtime_state",
}


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


def parse_ts_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = safe_text(value, "", 64)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return float(parsed.timestamp())
    except Exception:
        return None


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


@dataclass(frozen=True)
class StateContext:
    scope_id: str
    scope_kind: str = ""
    state_kind: str = ""
    status: str = ""
    owner_agent_id: str = ""
    state_basis: str = ""
    validated_at: str = ""
    validated_by: str = ""
    fresh_until: str = ""
    lease_expires_at: str = ""
    supersedes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "scope_id": safe_text(self.scope_id, "", 256),
        }
        state_kind = safe_text(self.state_kind, "", 64)
        if state_kind:
            payload["state_kind"] = state_kind
        scope_kind = safe_text(self.scope_kind, "", 64) or _STATE_SCOPE_KIND_DEFAULTS.get(state_kind, "")
        if scope_kind:
            payload["scope_kind"] = scope_kind
        status = safe_text(self.status, "", 64)
        if status:
            payload["status"] = status
        owner_agent_id = safe_text(self.owner_agent_id, "", 128)
        if owner_agent_id:
            payload["owner_agent_id"] = owner_agent_id
        state_basis = safe_text(self.state_basis, "", 64)
        if not state_basis and state_kind:
            state_basis = "runtime_validated" if state_kind == "runtime_validated_state" else "reported"
        if state_basis:
            payload["state_basis"] = state_basis
        validated_at = safe_text(self.validated_at, "", 64)
        if validated_at:
            payload["validated_at"] = validated_at
        validated_by = safe_text(self.validated_by, "", 128)
        if validated_by:
            payload["validated_by"] = validated_by
        fresh_until = safe_text(self.fresh_until, "", 64)
        if fresh_until:
            payload["fresh_until"] = fresh_until
        lease_expires_at = safe_text(self.lease_expires_at, "", 64)
        if lease_expires_at:
            payload["lease_expires_at"] = lease_expires_at
        supersedes = safe_text(self.supersedes, "", 128)
        if supersedes:
            payload["supersedes"] = supersedes
        metadata = dict(self.metadata or {})
        if metadata:
            payload["metadata"] = metadata
        return payload


def normalize_state_context(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, StateContext):
        return value.to_dict()
    if not isinstance(value, dict):
        return {}
    if not any(
        [
            safe_text(value.get("scope_id"), "", 256),
            safe_text(value.get("scope_kind"), "", 64),
            safe_text(value.get("state_kind"), "", 64),
            safe_text(value.get("status"), "", 64),
            safe_text(value.get("owner_agent_id"), "", 128),
            safe_text(value.get("state_basis"), "", 64),
            safe_text(value.get("validated_at"), "", 64),
            safe_text(value.get("validated_by"), "", 128),
            safe_text(value.get("fresh_until"), "", 64),
            safe_text(value.get("lease_expires_at"), "", 64),
            safe_text(value.get("supersedes"), "", 128),
            bool(value.get("metadata")),
        ]
    ):
        return {}
    payload = StateContext(
        scope_id=safe_text(value.get("scope_id"), "", 256),
        scope_kind=safe_text(value.get("scope_kind"), "", 64),
        state_kind=safe_text(value.get("state_kind"), "", 64),
        status=safe_text(value.get("status"), "", 64),
        owner_agent_id=safe_text(value.get("owner_agent_id"), "", 128),
        state_basis=safe_text(value.get("state_basis"), "", 64),
        validated_at=safe_text(value.get("validated_at"), "", 64),
        validated_by=safe_text(value.get("validated_by"), "", 128),
        fresh_until=safe_text(value.get("fresh_until"), "", 64),
        lease_expires_at=safe_text(value.get("lease_expires_at"), "", 64),
        supersedes=safe_text(value.get("supersedes"), "", 128),
        metadata=dict(value.get("metadata", {}) or {}),
    ).to_dict()
    return payload


def validate_state_context_dict(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    if not isinstance(payload, dict):
        return False, ["not_a_dict"]
    errors: List[str] = []
    scope_id = safe_text(payload.get("scope_id"), "", 256)
    state_kind = safe_text(payload.get("state_kind"), "", 64)
    status = safe_text(payload.get("status"), "", 64)
    scope_kind = safe_text(payload.get("scope_kind"), "", 64) or _STATE_SCOPE_KIND_DEFAULTS.get(state_kind, "")
    state_basis = safe_text(payload.get("state_basis"), "", 64)
    if not state_basis and state_kind:
        state_basis = "runtime_validated" if state_kind == "runtime_validated_state" else "reported"

    if not scope_id:
        errors.append("missing:scope_id")
    if not state_kind:
        errors.append("missing:state_kind")
    elif state_kind not in STATE_KINDS:
        errors.append("invalid:state_kind")
    if not status:
        errors.append("missing:status")
    elif status not in STATE_STATUSES:
        errors.append("invalid:status")
    if scope_kind and scope_kind not in STATE_SCOPE_KINDS:
        errors.append("invalid:scope_kind")
    if state_basis and state_basis not in STATE_BASES:
        errors.append("invalid:state_basis")

    for key in ("validated_at", "fresh_until", "lease_expires_at"):
        text = safe_text(payload.get(key), "", 64)
        if text and parse_ts_value(text) is None:
            errors.append(f"invalid:{key}")

    if state_basis == "runtime_validated":
        if not safe_text(payload.get("validated_at"), "", 64):
            errors.append("missing:validated_at")
        if not safe_text(payload.get("validated_by"), "", 128):
            errors.append("missing:validated_by")

    if (
        status in {"active", "blocked", "ready"}
        and state_kind in {"global_blocker", "pending_gate", "runtime_validated_state"}
        and not safe_text(payload.get("fresh_until"), "", 64)
    ):
        errors.append("missing:fresh_until")

    if safe_text(payload.get("lease_expires_at"), "", 64) and not safe_text(payload.get("owner_agent_id"), "", 128):
        errors.append("missing:owner_agent_id")

    if status == "superseded" and not safe_text(payload.get("supersedes"), "", 128):
        errors.append("missing:supersedes")

    return len(errors) == 0, errors


def freshness_state_for_context(value: Any, *, now_ts: float | None = None) -> str:
    payload = normalize_state_context(value)
    fresh_until = parse_ts_value(payload.get("fresh_until"))
    if fresh_until is None:
        return "unknown"
    if now_ts is None:
        now_ts = parse_ts_value(utc_now_iso()) or 0.0
    return "fresh" if float(now_ts) <= float(fresh_until) else "stale"


def lease_state_for_context(value: Any, *, now_ts: float | None = None) -> str:
    payload = normalize_state_context(value)
    lease_until = parse_ts_value(payload.get("lease_expires_at"))
    if lease_until is None:
        return "none"
    if now_ts is None:
        now_ts = parse_ts_value(utc_now_iso()) or 0.0
    return "active" if float(now_ts) <= float(lease_until) else "expired"


@dataclass
class ControlViewRow:
    scope_id: str
    thread_id: str
    scope_kind: str
    state_kind: str
    status: str
    owner_agent_id: str = ""
    state_basis: str = ""
    claim: str = ""
    source_record_id: str = ""
    source_record_type: str = ""
    reported_at: str = ""
    validated_at: str = ""
    fresh_until: str = ""
    freshness_state: str = "unknown"
    lease_expires_at: str = ""
    lease_state: str = "none"
    has_contradiction: bool = False
    conflict: bool = False
    evidence_refs: List[str] = field(default_factory=list)
    evidence_items: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "scope_id": safe_text(self.scope_id, "", 256),
            "thread_id": safe_text(self.thread_id, "", 128),
            "scope_kind": safe_text(self.scope_kind, "", 64),
            "state_kind": safe_text(self.state_kind, "", 64),
            "status": safe_text(self.status, "", 64),
            "owner_agent_id": safe_text(self.owner_agent_id, "", 128),
            "state_basis": safe_text(self.state_basis, "", 64),
            "claim": safe_text(self.claim, "", 2048),
            "source_record_id": safe_text(self.source_record_id, "", 128),
            "source_record_type": safe_text(self.source_record_type, "", 32),
            "reported_at": safe_text(self.reported_at, "", 64),
            "validated_at": safe_text(self.validated_at, "", 64),
            "fresh_until": safe_text(self.fresh_until, "", 64),
            "freshness_state": safe_text(self.freshness_state, "unknown", 16),
            "lease_expires_at": safe_text(self.lease_expires_at, "", 64),
            "lease_state": safe_text(self.lease_state, "none", 16),
            "has_contradiction": bool(self.has_contradiction),
            "conflict": bool(self.conflict),
            "evidence_refs": merge_evidence_refs(self.evidence_refs, self.evidence_items),
            "evidence_items": normalize_evidence_items(self.evidence_items),
            "metadata": dict(self.metadata or {}),
        }
        return payload


def control_row_sort_ts(row: ControlViewRow | Dict[str, Any]) -> float:
    payload = row.to_dict() if isinstance(row, ControlViewRow) else dict(row or {})
    for key in ("validated_at", "reported_at", "generated_at", "ts_utc"):
        ts = parse_ts_value(payload.get(key))
        if ts is not None:
            return float(ts)
    return 0.0


@dataclass
class AgentControlView:
    view: str
    generated_at: str = field(default_factory=utc_now_iso)
    thread_id: str = ""
    owner_agent_id: str = ""
    rows: List[ControlViewRow] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "v2",
            "view": safe_text(self.view, "", 64),
            "generated_at": safe_text(self.generated_at, utc_now_iso(), 64),
            "thread_id": safe_text(self.thread_id, "", 128),
            "owner_agent_id": safe_text(self.owner_agent_id, "", 128),
            "rows": [row.to_dict() if isinstance(row, ControlViewRow) else ControlViewRow(**dict(row)).to_dict() for row in list(self.rows or [])],
            "metadata": dict(self.metadata or {}),
        }


@dataclass
class ClaimCitation(Sequence[str]):
    primary_ref: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    evidence_items: List[Dict[str, Any]] = field(default_factory=list)
    span_grounded: bool = False
    state_context: Dict[str, Any] | None = None

    @classmethod
    def from_value(cls, value: Any, *, fallback_ref: str = "") -> "ClaimCitation":
        if isinstance(value, ClaimCitation):
            refs = merge_evidence_refs(value.evidence_refs, value.evidence_items)
            primary_ref = safe_text(value.primary_ref, "", 512) or (refs[0] if refs else safe_text(fallback_ref, "", 512))
            state_context = normalize_state_context(value.state_context) if value.state_context is not None else {}
            return cls(
                primary_ref=primary_ref,
                evidence_refs=refs,
                evidence_items=normalize_evidence_items(value.evidence_items),
                span_grounded=bool(value.span_grounded or has_grounded_span(value.evidence_items)),
                state_context=(state_context or None),
            )
        if isinstance(value, dict):
            evidence_items = normalize_evidence_items(value.get("evidence_items"))
            refs = merge_evidence_refs(value.get("evidence_refs"), evidence_items)
            primary_ref = safe_text(value.get("primary_ref"), "", 512) or (refs[0] if refs else safe_text(fallback_ref, "", 512))
            state_context = normalize_state_context(value.get("state_context")) if "state_context" in value else {}
            return cls(
                primary_ref=primary_ref,
                evidence_refs=refs,
                evidence_items=evidence_items,
                span_grounded=bool(value.get("span_grounded")) or has_grounded_span(evidence_items),
                state_context=(state_context or None),
            )
        refs = normalize_refs(value)
        return cls(
            primary_ref=(refs[0] if refs else safe_text(fallback_ref, "", 512)),
            evidence_refs=refs,
            evidence_items=[],
            span_grounded=False,
            state_context=None,
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
        payload = {
            "primary_ref": primary_ref,
            "evidence_refs": evidence_refs,
            "evidence_items": evidence_items,
            "span_grounded": bool(self.span_grounded or has_grounded_span(evidence_items)),
        }
        state_context = normalize_state_context(self.state_context) if self.state_context is not None else {}
        if state_context:
            payload["state_context"] = state_context
        return payload


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


def _validate_state_context_field(payload: Dict[str, Any], key: str) -> List[str]:
    if key not in payload:
        return []
    raw = payload.get(key)
    if raw is None:
        return []
    if not isinstance(raw, dict):
        return [f"invalid:{key}:not_a_dict"]
    ok, item_errors = validate_state_context_dict(normalize_state_context(raw))
    if ok:
        return []
    return [f"invalid:{key}:{err}" for err in item_errors]


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
    errors.extend(_validate_state_context_field(value, "state_context"))
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
    state_context: Dict[str, Any] | None = None
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
        state_context = normalize_state_context(self.state_context) if self.state_context is not None else {}
        if state_context:
            payload["state_context"] = state_context
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
            "state_context": payload.get("state_context", {}),
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
    state_context: Dict[str, Any] | None = None
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
        state_context = normalize_state_context(self.state_context) if self.state_context is not None else {}
        if state_context:
            payload["state_context"] = state_context
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

    if "state_context" in payload:
        payload["state_context"] = normalize_state_context(payload.get("state_context"))
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
    errors.extend(_validate_state_context_field(payload, "state_context"))
    return len(errors) == 0, errors


def validate_fact_dict(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    if not isinstance(payload, dict):
        return False, ["not_a_dict"]
    errors = _validate_required(payload, ("thread_id", "claim", "state"))
    state = safe_text(payload.get("state"), "", 32).lower()
    if state not in FACT_STATES:
        errors.append("invalid:state")
    errors.extend(_validate_evidence_items_field(payload, "evidence_items"))
    errors.extend(_validate_state_context_field(payload, "state_context"))
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
            errors.extend(_validate_state_context_field(row, "state_context"))
    unresolved = payload.get("unresolved_contradictions", [])
    if unresolved and not isinstance(unresolved, list):
        errors.append("invalid:unresolved_contradictions")
    elif isinstance(unresolved, list):
        for idx, row in enumerate(unresolved):
            if not isinstance(row, dict):
                errors.append(f"invalid:unresolved_contradictions[{idx}]")
                continue
            errors.extend(_validate_evidence_items_field(row, "evidence_items"))
            errors.extend(_validate_state_context_field(row, "state_context"))
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
    state_context = normalize_state_context(event.get("state_context")) if "state_context" in event else None

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
        clean_claim = safe_text(claim, "", 2048)
        if not clean_claim:
            continue
        key = _canonical_hash({"thread_id": thread_id, "claim": clean_claim})
        row: Dict[str, Any] = {
            "thread_id": thread_id,
            "claim": clean_claim,
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
        if state_context is not None:
            row["state_context"] = dict(state_context)
        dedup[key] = row
    return list(dedup.values())
