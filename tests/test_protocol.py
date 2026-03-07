import hashlib

from evidencespine.protocol import (
    AgentMemoryEvent,
    event_to_fact_candidates,
    normalize_evidence_items,
    validate_event_dict,
    validate_evidence_item_dict,
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
