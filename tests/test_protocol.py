import hashlib

from evidencespine.protocol import (
    AgentMemoryEvent,
    event_to_fact_candidates,
    normalize_evidence_items,
    normalize_state_context,
    validate_event_dict,
    validate_evidence_item_dict,
    validate_state_context_dict,
)


def test_event_validation_and_hash() -> None:
    event = AgentMemoryEvent(
        event_id="",
        thread_id="demo",
        event_type="decision",
        source_agent_id="agent",
        source_turn_id="t1",
        payload={"claim": "x"},
        evidence_refs=["a.md#L1"],
    ).to_dict()
    ok, errors = validate_event_dict(event)
    assert ok
    assert errors == []
    assert event["schema_version"] == "v2"
    assert event["event_hash"]
    assert event["event_id"].startswith("ame_")


def test_evidence_item_validation_accepts_anchored_minimal() -> None:
    ok, errors = validate_evidence_item_dict(
        {
            "source_id": "src/file.py",
            "line_start": 10,
        }
    )
    assert ok
    assert errors == []


def test_evidence_item_validation_rejects_missing_source_id() -> None:
    ok, errors = validate_evidence_item_dict({"line_start": 10})
    assert not ok
    assert "missing:source_id" in errors


def test_evidence_item_validation_rejects_missing_anchor() -> None:
    ok, errors = validate_evidence_item_dict({"source_id": "src/file.py"})
    assert not ok
    assert "missing:anchor" in errors


def test_normalize_evidence_items_fills_missing_end_positions() -> None:
    item = normalize_evidence_items(
        [
            {
                "source_id": "src/file.py",
                "line_start": 14,
                "char_start": 120,
            }
        ]
    )[0]
    assert item["line_end"] == 14
    assert item["char_end"] == 120


def test_event_hash_changes_when_evidence_items_change() -> None:
    base = {
        "event_id": "",
        "thread_id": "demo",
        "event_type": "decision",
        "source_agent_id": "agent",
        "source_turn_id": "t1",
        "payload": {"claim": "deploy patch"},
    }
    first = AgentMemoryEvent(
        **base,
        evidence_items=[{"source_id": "src/file.py", "line_start": 10, "excerpt": "deploy patch"}],
    ).to_dict()
    second = AgentMemoryEvent(
        **base,
        evidence_items=[{"source_id": "src/file.py", "line_start": 11, "excerpt": "deploy patch"}],
    ).to_dict()
    assert first["event_hash"] != second["event_hash"]


def test_event_to_fact_candidates_extracts_claims_and_inherits_evidence_items() -> None:
    excerpt = "primary"
    event = {
        "event_id": "e1",
        "thread_id": "demo",
        "event_type": "outcome",
        "source_agent_id": "agent",
        "source_turn_id": "t2",
        "payload": {
            "claim": "primary",
            "decision": "dec",
            "outcome": "ok",
            "fact_state": "verified",
        },
        "evidence_refs": ["file.py:1"],
        "evidence_items": [
            {
                "source_id": "file.py",
                "line_start": 1,
                "line_end": 2,
                "excerpt": excerpt,
                "checksum": f"sha256:{hashlib.sha256(excerpt.encode('utf-8')).hexdigest()}",
            }
        ],
    }
    facts = event_to_fact_candidates(event)
    assert len(facts) >= 1
    assert all(f["state"] == "verified" for f in facts)
    assert all(f["evidence_items"][0]["source_id"] == "file.py" for f in facts)


def test_state_context_validation_accepts_reported_agent_work() -> None:
    ok, errors = validate_state_context_dict(
        {
            "scope_id": "auth-timeout-fix",
            "state_kind": "agent_local_work",
            "status": "active",
            "owner_agent_id": "implementer",
        }
    )
    assert ok
    assert errors == []


def test_state_context_validation_rejects_missing_scope_id() -> None:
    ok, errors = validate_state_context_dict(
        {
            "state_kind": "agent_local_work",
            "status": "active",
        }
    )
    assert not ok
    assert "missing:scope_id" in errors


def test_state_context_validation_rejects_runtime_validated_without_validation_fields() -> None:
    ok, errors = validate_state_context_dict(
        {
            "scope_id": "runtime-health",
            "state_kind": "runtime_validated_state",
            "status": "active",
        }
    )
    assert not ok
    assert "missing:validated_at" in errors
    assert "missing:validated_by" in errors


def test_state_context_validation_requires_fresh_until_for_live_blocker_or_gate() -> None:
    ok, errors = validate_state_context_dict(
        {
            "scope_id": "ci-gate",
            "state_kind": "pending_gate",
            "status": "ready",
        }
    )
    assert not ok
    assert "missing:fresh_until" in errors


def test_state_context_validation_requires_supersedes_for_superseded_status() -> None:
    ok, errors = validate_state_context_dict(
        {
            "scope_id": "scope-a",
            "state_kind": "agent_local_work",
            "status": "superseded",
        }
    )
    assert not ok
    assert "missing:supersedes" in errors


def test_normalize_state_context_applies_defaults() -> None:
    row = normalize_state_context(
        {
            "scope_id": "ci-gate",
            "state_kind": "pending_gate",
            "status": "active",
            "fresh_until": "2026-03-18T13:30:00Z",
        }
    )
    assert row["scope_kind"] == "gate"
    assert row["state_basis"] == "reported"
