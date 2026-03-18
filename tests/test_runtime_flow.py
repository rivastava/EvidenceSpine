from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evidencespine.runtime import AgentMemoryRuntime, RuntimeHooks
from evidencespine.settings import EvidenceSpineSettings


def _runtime(tmp_path: Path, *, hooks: RuntimeHooks | None = None) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    return AgentMemoryRuntime(config=settings.to_runtime_config(), hooks=hooks)


def _jsonl_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines()]


def test_runtime_ingest_brief_handoff_roundtrip_with_evidence_items(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    excerpt = "deploy patch"
    out = rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "t1",
            "payload": {
                "claim": excerpt,
                "fact_state": "verified",
                "next_actions": ["auditor verify"],
            },
            "evidence_items": [
                {
                    "source_id": "patch.diff",
                    "line_start": 7,
                    "line_end": 9,
                    "excerpt": excerpt,
                    "checksum": f"sha256:{hashlib.sha256(excerpt.encode('utf-8')).hexdigest()}",
                }
            ],
            "confidence": 0.8,
            "salience": 0.7,
        }
    )
    assert out["status"] == "ok"

    facts = _jsonl_rows(tmp_path / ".es" / "facts.jsonl")
    assert facts[-1]["evidence_items"][0]["source_id"] == "patch.diff"

    brief = rt.build_brief("demo", "status")
    b = brief.to_dict()
    assert b["thread_id"] == "demo"
    assert b["schema_version"] == "v2"
    claim = b["recent_verified_facts"][0]
    assert b["citations"][claim]["span_grounded"] is True
    assert b["citation_refs"][claim] == ["patch.diff#L7-L9"]

    packet = rt.emit_handoff("auditor", "demo", "verify")
    p = packet.to_dict()
    assert p["thread_id"] == "demo"
    assert p["checksum"]
    assert p["evidence_items"][0]["source_id"] == "patch.diff"
    assert p["claims"][0]["span_grounded"] is True

    imported = rt.import_handoff(p, source_agent_id="auditor")
    assert imported["status"] == "ok"

    snap = rt.snapshot()
    assert snap["agent_memory_events_24h"] >= 1
    assert snap["agent_handoff_packets_emitted_24h"] >= 1
    assert snap["agent_claim_span_citation_coverage_24h"] >= 1.0


def test_runtime_import_handoff_supports_v1_ref_only_packets(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    out = rt.import_handoff(
        {
            "schema_version": "v1",
            "packet_id": "legacy_packet",
            "role": "auditor",
            "thread_id": "demo",
            "scope": "verify",
            "claims": [{"claim": "legacy claim", "evidence_refs": ["legacy.md#L1"], "status": "asserted"}],
            "locked_decisions": [],
            "required_validations": ["check legacy flow"],
            "evidence_refs": ["legacy.md#L1"],
            "checksum": "legacy",
        },
        source_agent_id="auditor",
    )
    assert out["status"] == "ok"


def test_runtime_import_handoff_v2_flattens_claim_and_packet_evidence_items(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    packet = {
        "schema_version": "v2",
        "packet_id": "packet_v2",
        "role": "auditor",
        "thread_id": "demo",
        "scope": "verify",
        "claims": [
            {
                "claim": "verify patch",
                "status": "verified",
                "evidence_items": [{"source_id": "src/file.py", "line_start": 10, "line_end": 12}],
            }
        ],
        "unresolved_contradictions": [
            {
                "claim": "CONTRADICTION: drift",
                "reason": "unresolved_contradiction",
                "evidence_items": [{"source_id": "notes.md", "line_start": 3, "line_end": 4}],
            }
        ],
        "required_validations": ["validate v2"],
        "evidence_items": [{"source_id": "packet.md", "line_start": 1, "line_end": 1}],
        "checksum": "placeholder",
    }
    out = rt.import_handoff(packet, source_agent_id="auditor")
    assert out["status"] == "ok"

    events = _jsonl_rows(tmp_path / ".es" / "events.jsonl")
    latest = events[-1]
    assert {item["source_id"] for item in latest["evidence_items"]} == {"packet.md", "src/file.py", "notes.md"}


def test_snapshot_distinguishes_ref_and_span_coverage(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "t1",
            "payload": {"claim": "ref only claim", "fact_state": "verified"},
            "evidence_refs": ["ref_only.md#L1"],
        }
    )
    excerpt = "span grounded claim"
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "t2",
            "payload": {"claim": excerpt, "fact_state": "verified"},
            "evidence_items": [
                {
                    "source_id": "span.md",
                    "line_start": 4,
                    "line_end": 5,
                    "excerpt": excerpt,
                    "checksum": f"sha256:{hashlib.sha256(excerpt.encode('utf-8')).hexdigest()}",
                }
            ],
        }
    )

    rt.build_brief("demo", "status")
    snap = rt.snapshot()
    assert snap["agent_claim_ref_citation_coverage_24h"] >= snap["agent_claim_span_citation_coverage_24h"]
    assert snap["agent_claim_span_citation_coverage_24h"] < snap["agent_claim_ref_citation_coverage_24h"]
    assert snap["agent_claim_excerpt_fidelity_24h"] == 1.0


def test_runtime_dedupe_hits(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    event = {
        "thread_id": "demo",
        "event_type": "intent",
        "source_agent_id": "impl",
        "source_turn_id": "t1",
        "payload": {"claim": "same"},
    }
    first = rt.ingest_event(event)
    second = rt.ingest_event(event)
    assert first["status"] == "ok"
    assert second["status"] in {"ok", "deduped"}


def test_runtime_state_context_propagates_through_brief_and_handoff(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    out = rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "t1",
            "payload": {"claim": "Retry guard patch is under validation", "fact_state": "verified"},
            "state_context": {
                "scope_id": "auth-timeout-fix",
                "state_kind": "agent_local_work",
                "status": "active",
                "owner_agent_id": "implementer",
            },
            "evidence_refs": ["src/auth.py#L42-L57"],
        }
    )
    assert out["status"] == "ok"

    facts = _jsonl_rows(tmp_path / ".es" / "facts.jsonl")
    assert facts[-1]["state_context"]["scope_id"] == "auth-timeout-fix"

    brief = rt.build_brief("demo", "what matters")
    b = brief.to_dict()
    claim = b["recent_verified_facts"][0]
    assert b["citations"][claim]["state_context"]["owner_agent_id"] == "implementer"
    assert b["metadata"]["active_scope_count"] >= 1

    packet = rt.emit_handoff("auditor", "demo", "verify")
    p = packet.to_dict()
    assert p["claims"][0]["state_context"]["scope_id"] == "auth-timeout-fix"
    assert p["metadata"]["active_scope_summary"]["active_scope_count"] >= 1


def test_runtime_import_handoff_preserves_state_context_via_synthetic_rows(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    out = rt.import_handoff(
        {
            "schema_version": "v2",
            "packet_id": "packet_with_state",
            "role": "auditor",
            "thread_id": "demo",
            "scope": "verify",
            "claims": [
                {
                    "claim": "Gate is ready",
                    "status": "verified",
                    "state_context": {
                        "scope_id": "release-gate",
                        "state_kind": "pending_gate",
                        "status": "ready",
                        "fresh_until": "2099-01-01T00:00:00Z",
                    },
                }
            ],
            "required_validations": ["validate release gate"],
            "checksum": "placeholder",
        },
        source_agent_id="auditor",
    )
    assert out["status"] == "ok"
    assert out["state_rows_imported"] == 1

    events = _jsonl_rows(tmp_path / ".es" / "events.jsonl")
    imported_rows = [row for row in events if row.get("metadata", {}).get("imported_packet_id") == "packet_with_state"]
    assert any(row.get("state_context", {}).get("scope_id") == "release-gate" for row in imported_rows)


def test_query_view_resolves_latest_scope_state_and_filters(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "t1",
            "payload": {"claim": "Work started", "fact_state": "asserted"},
            "state_context": {
                "scope_id": "auth-timeout-fix",
                "state_kind": "agent_local_work",
                "status": "active",
                "owner_agent_id": "implementer",
            },
        }
    )
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "reflection",
            "role": "auditor",
            "source_agent_id": "auditor",
            "source_turn_id": "t2",
            "payload": {"claim": "Release gate ready", "fact_state": "verified"},
            "state_context": {
                "scope_id": "release-gate",
                "state_kind": "pending_gate",
                "status": "ready",
                "fresh_until": "2099-01-01T00:00:00Z",
            },
        }
    )
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "reflection",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "t3",
            "payload": {"claim": "Work complete", "fact_state": "verified"},
            "state_context": {
                "scope_id": "auth-timeout-fix",
                "state_kind": "agent_local_work",
                "status": "closed",
                "owner_agent_id": "implementer",
            },
        }
    )

    active = rt.query_view("active_scopes", thread_id="demo").to_dict()
    assert [row["scope_id"] for row in active["rows"]] == ["release-gate"]

    my_work = rt.query_view("my_work", thread_id="demo", owner_agent_id="implementer", include_closed=True).to_dict()
    assert any(row["scope_id"] == "auth-timeout-fix" for row in my_work["rows"])

    open_gates = rt.query_view("open_gates", thread_id="demo").to_dict()
    assert [row["scope_id"] for row in open_gates["rows"]] == ["release-gate"]


def test_query_view_detects_stale_and_conflicting_rows(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    stale_ts = "2000-01-01T00:00:00Z"
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "reflection",
            "role": "operator",
            "source_agent_id": "operator",
            "source_turn_id": "s1",
            "payload": {"claim": "Runtime degraded", "fact_state": "verified"},
            "state_context": {
                "scope_id": "runtime-health",
                "state_kind": "runtime_validated_state",
                "status": "active",
                "state_basis": "runtime_validated",
                "validated_at": stale_ts,
                "validated_by": "smoke",
                "fresh_until": stale_ts,
            },
        }
    )
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "c1",
            "payload": {"claim": "Owner one", "fact_state": "asserted"},
            "state_context": {
                "scope_id": "shared-scope",
                "state_kind": "agent_local_work",
                "status": "active",
                "owner_agent_id": "implementer",
            },
        }
    )
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "role": "auditor",
            "source_agent_id": "auditor",
            "source_turn_id": "c2",
            "payload": {"claim": "Owner two", "fact_state": "asserted"},
            "state_context": {
                "scope_id": "shared-scope",
                "state_kind": "agent_local_work",
                "status": "active",
                "owner_agent_id": "auditor",
            },
        }
    )

    stale = rt.query_view("stale_claims", thread_id="demo", include_closed=True).to_dict()
    assert any(row["scope_id"] == "runtime-health" and row["freshness_state"] == "stale" for row in stale["rows"])

    contradictions = rt.query_view("contradictions", thread_id="demo", include_closed=True).to_dict()
    assert any(row["scope_id"] == "shared-scope" and row["conflict"] is True for row in contradictions["rows"])

    snap = rt.snapshot()
    assert snap["agent_active_scope_count_24h"] >= 1
    assert snap["agent_active_scope_stale_rate_24h"] >= 0.0
    assert snap["agent_scope_conflict_rate_24h"] >= 0.0


def test_reconcile_returns_unsupported_without_hook(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    out = rt.reconcile("demo")
    assert out == {"status": "unsupported", "reason": "no_reconcile_hook"}


def test_reconcile_ingests_rows_from_hook(tmp_path: Path) -> None:
    def reconcile_state(thread_id: str, rows: list[dict]) -> list[dict]:
        assert thread_id == "demo"
        assert rows == []
        return [
            {
                "payload": {"claim": "Reconciled blocker state", "fact_state": "verified"},
                "state_context": {
                    "scope_id": "runtime-health",
                    "state_kind": "global_blocker",
                    "status": "blocked",
                    "fresh_until": "2099-01-01T00:00:00Z",
                },
                "evidence_refs": ["runtime.json#L1"],
            }
        ]

    rt = _runtime(tmp_path, hooks=RuntimeHooks(reconcile_state=reconcile_state))
    out = rt.reconcile("demo")
    assert out["status"] == "ok"
    assert out["ingested"] == 1

    rows = rt.query_view("active_scopes", thread_id="demo").to_dict()["rows"]
    assert rows[0]["scope_id"] == "runtime-health"
