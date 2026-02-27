from evidencespine.protocol import AgentMemoryEvent, event_to_fact_candidates, validate_event_dict


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
    assert event["event_hash"]
    assert event["event_id"].startswith("ame_")


def test_event_to_fact_candidates_extracts_claims() -> None:
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
    }
    facts = event_to_fact_candidates(event)
    assert len(facts) >= 1
    assert all(f["state"] == "verified" for f in facts)
