from __future__ import annotations

import hashlib
from pathlib import Path

from evidencespine.runtime import AgentMemoryRuntime, RuntimeHooks
from evidencespine.settings import EvidenceSpineSettings
from grounding_utils import grounded_item


def _runtime(tmp_path: Path, *, hooks: RuntimeHooks | None = None, storage_format: str = "sqlite") -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"), storage_format=storage_format)
    return AgentMemoryRuntime(config=settings.to_runtime_config(), hooks=hooks)


def _events(rt: AgentMemoryRuntime) -> list[dict]:
    return list(rt.store.iter_events())


def _facts(rt: AgentMemoryRuntime) -> list[dict]:
    return list(rt.store.iter_facts())


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

    facts = _facts(rt)
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

    events = _events(rt)
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
            "evidence_items": [grounded_item(tmp_path)],
        }
    )
    assert out["status"] == "ok"

    facts = _facts(rt)
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

    events = _events(rt)
    imported_rows = [row for row in events if row.get("metadata", {}).get("imported_packet_id") == "packet_with_state"]
    assert any(row.get("state_context", {}).get("scope_id") == "release-gate" for row in imported_rows)


def test_runtime_import_handoff_preserves_claim_classification(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    out = rt.import_handoff(
        {
            "schema_version": "v2",
            "packet_id": "classified_packet",
            "role": "auditor",
            "thread_id": "demo",
            "scope": "verify",
            "locked_decisions": ["Locked: ship v0.5 first"],
            "claims": [
                {
                    "claim": "Verified: tests green",
                    "status": "verified",
                    "evidence_items": [grounded_item(tmp_path, name="tests/evidence.py")],
                },
                {"claim": "Open: concurrency audit", "status": "asserted"},
            ],
            "unresolved_contradictions": [
                {"claim": "CONTRADICTION: timing mismatch", "status": "contradicted"},
            ],
            "required_validations": ["re-run suite", "verify packaging"],
            "checksum": "placeholder",
        },
        source_agent_id="auditor",
    )
    assert out["status"] == "ok"
    assert out["decisions_imported"] == 1
    assert out["claims_imported"] == 2
    assert out["contradictions_imported"] == 1
    assert out["validations_imported"] == 2

    b = rt.build_brief("demo", "status").to_dict()
    assert any(item.startswith("Locked: ship v0.5 first") for item in b["locked_decisions"])
    assert any(item.startswith("Verified: tests green") for item in b["recent_verified_facts"])
    assert any(item.startswith("Open: concurrency audit") for item in b["open_items"])
    assert any("timing mismatch" in item for item in b["active_risks"])
    assert all(
        action.startswith(expected)
        for action, expected in zip(b["next_actions"], ["re-run suite", "verify packaging"])
    )


def test_runtime_import_handoff_binds_to_importer_thread(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    out = rt.import_handoff(
        {
            "schema_version": "v2",
            "packet_id": "handoff_from_demo",
            "role": "auditor",
            "thread_id": "demo",
            "scope": "verify",
            "locked_decisions": ["Locked: successor owns cleanup"],
            "claims": [
                {
                    "claim": "Verified: handoff received",
                    "status": "verified",
                    "evidence_items": [grounded_item(tmp_path, name="packets/evidence.py")],
                }
            ],
            "required_validations": ["continue from demo"],
            "checksum": "placeholder",
        },
        source_agent_id="auditor",
        thread_id="successor",
    )
    assert out["status"] == "ok"

    successor = rt.build_brief("successor", "status").to_dict()
    assert any(item.startswith("Locked: successor owns cleanup") for item in successor["locked_decisions"])
    assert any(item.startswith("Verified: handoff received") for item in successor["recent_verified_facts"])
    assert successor["next_actions"][0].startswith("continue from demo")

    demo = rt.build_brief("demo", "status").to_dict()
    assert not any(item.startswith("Verified: handoff received") for item in demo["recent_verified_facts"])

    events = _events(rt)
    imported = [e for e in events if e.get("metadata", {}).get("imported_packet_id") == "handoff_from_demo"]
    assert imported and all(e["metadata"]["imported_from_thread"] == "demo" for e in imported)


def test_runtime_import_handoff_skips_content_duplicates_on_reimport(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    packet = {
        "schema_version": "v2",
        "packet_id": "reimported_packet",
        "role": "researcher",
        "thread_id": "demo",
        "scope": "verify",
        "locked_decisions": ["Locked: ship v0.5 first"],
        "claims": [
            {"claim": "Verified: tests green", "status": "verified", "evidence_refs": ["tests/#L1"]},
            {"claim": "Open: concurrency audit", "status": "asserted"},
        ],
        "required_validations": ["re-run suite"],
        "checksum": "placeholder",
    }
    first = rt.import_handoff(packet, source_agent_id="researcher")
    assert first["status"] == "ok"
    assert first["decisions_imported"] == 1
    assert first["claims_imported"] == 2

    second = rt.import_handoff(packet, source_agent_id="researcher")
    assert second["status"] == "ok"
    assert second["duplicates_skipped"] == 3
    assert second["decisions_imported"] == 0
    assert second["claims_imported"] == 0

    before = len(_facts(rt))
    rt.import_handoff(packet, source_agent_id="researcher")
    assert len(_facts(rt)) == before


def test_runtime_import_handoff_strips_inline_refs_and_dedupes_against_clean_facts(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "outcome",
            "role": "researcher",
            "source_agent_id": "researcher",
            "source_turn_id": "t1",
            "payload": {"claim": "import fidelity confirmed live", "fact_state": "verified"},
            "evidence_refs": ["src/evidencespine/runtime.py"],
        }
    )
    packet = {
        "schema_version": "v2",
        "packet_id": "suffixed_packet",
        "role": "auditor",
        "thread_id": "demo",
        "scope": "verify",
        "locked_decisions": ["Locked: ship v0.5 [ref:src/evidencespine/runtime.py]"],
        "claims": [
            {"claim": "import fidelity confirmed live [ref:src/evidencespine/runtime.py]", "status": "verified"},
            {"claim": "Open: concurrency audit [ref:src/evidencespine/runtime.py] [ref:src/evidencespine/store.py]", "status": "asserted"},
        ],
        "required_validations": ["re-run suite"],
        "checksum": "placeholder",
    }
    out = rt.import_handoff(packet, source_agent_id="auditor")
    assert out["status"] == "ok"
    assert out["duplicates_skipped"] == 1
    assert out["claims_imported"] == 1

    facts = _facts(rt)
    stored = [f["claim"] for f in facts]
    assert "import fidelity confirmed live" in stored
    assert not any("[ref:" in claim for claim in stored)
    locked = [f for f in facts if f["claim"] == "Locked: ship v0.5"]
    assert locked and locked[0]["state"] == "asserted"
    assert "src/evidencespine/store.py" in locked[0]["evidence_refs"] or "src/evidencespine/store.py" in {
        ref for f in facts if f["claim"] == "Open: concurrency audit" for ref in f["evidence_refs"]
    }


def test_runtime_import_handoff_skips_decision_when_verified_fact_exists(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "t1",
            "payload": {"claim": "Locked: ship v0.5 first", "fact_state": "verified"},
        }
    )
    out = rt.import_handoff(
        {
            "schema_version": "v2",
            "packet_id": "p",
            "role": "auditor",
            "thread_id": "demo",
            "scope": "s",
            "locked_decisions": ["Locked: ship v0.5 first [ref:src/evidencespine/runtime.py]"],
            "claims": [],
            "required_validations": [],
            "checksum": "c",
        },
        source_agent_id="auditor",
    )
    assert out["duplicates_skipped"] == 1
    assert out["decisions_imported"] == 0
    assert sum(1 for f in _facts(rt) if f["claim"] == "Locked: ship v0.5 first") == 1


def test_verified_claim_supersedes_asserted_twin(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    first = rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "t1",
            "payload": {"claim": "Open: concurrency audit", "fact_state": "asserted"},
        }
    )
    assert first["status"] == "ok"

    second = rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "outcome",
            "role": "auditor",
            "source_agent_id": "auditor",
            "source_turn_id": "t2",
            "payload": {"claim": "Open: concurrency audit", "fact_state": "verified"},
            "evidence_items": [grounded_item(tmp_path, name="audit/evidence.py")],
        }
    )
    assert second["status"] == "ok"

    facts = _facts(rt)
    asserted = [f for f in facts if f["state"] == "asserted" and f["claim"] == "Open: concurrency audit"]
    verified = [f for f in facts if f["state"] == "verified" and f["claim"] == "Open: concurrency audit"]
    assert asserted and verified, "both rows exist; the verified twin supersedes the asserted one"
    assert verified[0]["supersedes_fact_id"] == asserted[0]["fact_id"], "verified fact must supersede the asserted twin"

    b = rt.build_brief("demo", "status").to_dict()
    assert any(item.startswith("Open: concurrency audit") for item in b["recent_verified_facts"])
    assert not any(item.startswith("Open: concurrency audit") for item in b["open_items"])


def test_brief_excludes_superseded_facts_and_dedupes_sections(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "t1",
            "payload": {"claim": "Deferred: revisit later", "fact_state": "asserted"},
        }
    )
    old = [f for f in _facts(rt) if f["claim"] == "Deferred: revisit later"][0]
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "outcome",
            "role": "auditor",
            "source_agent_id": "auditor",
            "source_turn_id": "t3",
            "payload": {
                "claim": "Deferred: revisit later",
                "fact_state": "verified",
                "supersedes_ref": old["fact_id"],
            },
            "evidence_items": [grounded_item(tmp_path, name="deferred/evidence.py")],
        }
    )

    b = rt.build_brief("demo", "status").to_dict()
    assert not any(item.startswith("Deferred: revisit later") for item in b["open_items"])
    assert not any(item.startswith("Deferred: revisit later") for item in b["locked_decisions"])

    dup_a = rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "reflection",
            "role": "auditor",
            "source_agent_id": "auditor",
            "source_turn_id": "t4",
            "payload": {"claim": "Dup note", "fact_state": "asserted"},
            "evidence_refs": ["a.py"],
        }
    )
    assert dup_a["status"] == "ok"
    dup_b = rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "reflection",
            "role": "auditor",
            "source_agent_id": "auditor",
            "source_turn_id": "t5",
            "payload": {"claim": "Dup note", "fact_state": "asserted"},
            "evidence_refs": ["b.py"],
        }
    )
    assert dup_b["status"] == "ok"

    b2 = rt.build_brief("demo", "status").to_dict()
    assert sum(1 for item in b2["open_items"] if item.startswith("Dup note")) == 1, (
        "identical claims must be deduped in the brief"
    )


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

    scopes = rt.query_view("active_scopes", thread_id="demo", include_closed=True).to_dict()
    shared = [row for row in scopes["rows"] if row["scope_id"] == "shared-scope"]
    assert shared, "shared-scope must resolve"
    assert shared[0]["multi_owner"] is True, "different owners alone is coordination, not conflict"
    assert shared[0]["conflict"] is False, "same status/kind with different owners is not conflict"

    contradictions = rt.query_view("contradictions", thread_id="demo", include_closed=True).to_dict()
    assert not any(row["scope_id"] == "shared-scope" for row in contradictions["rows"]), (
        "multi-owner coordination must not surface in the contradictions view"
    )

    snap = rt.snapshot()
    assert snap["agent_active_scope_count_24h"] >= 1
    assert snap["agent_active_scope_stale_rate_24h"] >= 0.0
    assert snap["agent_scope_conflict_rate_24h"] < 1.0, "multi-owner scope must not count as conflict"


def test_query_view_flags_status_disagreement_within_state_kind_as_conflict(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    for status, owner, turn in (("ready", "auditor", "g1"), ("blocked", "operator", "g2")):
        rt.ingest_event(
            {
                "thread_id": "demo",
                "event_type": "reflection",
                "role": owner,
                "source_agent_id": owner,
                "source_turn_id": turn,
                "payload": {"claim": f"Gate says {status}", "fact_state": "asserted"},
                "state_context": {
                    "scope_id": "release-gate",
                    "state_kind": "pending_gate",
                    "status": status,
                    "owner_agent_id": owner,
                    "fresh_until": "2099-01-01T00:00:00Z",
                },
            }
        )

    view = rt.query_view("contradictions", thread_id="demo", include_closed=True).to_dict()
    gate = [row for row in view["rows"] if row["scope_id"] == "release-gate"]
    assert gate and gate[0]["conflict"] is True, "ready vs blocked on the same gate is a real conflict"
    assert gate[0]["multi_owner"] is True


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
