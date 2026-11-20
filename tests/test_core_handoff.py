"""Core hardening: handoff provenance round-trip and import/emit error accounting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evidencespine.protocol import AgentHandoffPacket, validate_handoff_dict
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings
from grounding_utils import grounded_item


def _runtime(tmp_path: Path, **overrides) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    for key, value in overrides.items():
        setattr(settings, key, value)
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def _event(claim: str, *, thread: str = "demo", turn: str = "", **extra) -> dict:
    row: dict[str, Any] = {
        "thread_id": thread,
        "event_type": "outcome",
        "role": "auditor",
        "source_agent_id": "auditor",
        "source_turn_id": turn or f"turn-{claim}",
        "payload": {"claim": claim},
    }
    row.update(extra)
    return row


def _provenanced_event(claim: str, *, reference: str) -> dict:
    return _event(
        claim,
        payload={"claim": claim, "fact_state": "verified", "verification": {"method": "test", "reference": reference}},
    )


# --- item 7: handoff provenance schema + round trip -----------------------


def test_handoff_row_schema_carries_provenance_keys() -> None:
    packet = AgentHandoffPacket(
        packet_id="p",
        role="auditor",
        thread_id="demo",
        scope="s",
        claims=[{"claim": "c", "status": "verified", "verification": {"method": "test", "reference": "pytest t"}, "evidence_stale": False}],
        unresolved_contradictions=[{"claim": "CONTRADICTION: x", "reason": "unresolved_contradiction"}],
    ).to_dict()
    for row in packet["claims"] + packet["unresolved_contradictions"]:
        assert "verification" in row
        assert "evidence_stale" in row
        assert isinstance(row["metadata"], dict)
    assert packet["claims"][0]["verification"]["reference"] == "pytest t"
    assert packet["claims"][0]["evidence_stale"] is False
    assert packet["unresolved_contradictions"][0]["verification"] is None
    ok, errors = validate_handoff_dict(packet)
    assert ok, errors


def test_handoff_validator_rejects_invalid_verification_shape() -> None:
    base = {"role": "auditor", "thread_id": "demo", "scope": "s", "claims": [], "checksum": "c"}
    bad = dict(base)
    bad["claims"] = [{"claim": "c", "verification": "not-a-dict"}]
    ok, errors = validate_handoff_dict(bad)
    assert not ok
    assert any("verification" in error for error in errors)

    bad2 = dict(base)
    bad2["claims"] = [{"claim": "c", "verification": {"method": "bogus", "reference": ""}}]
    ok, errors = validate_handoff_dict(bad2)
    assert not ok
    assert any("verification" in error for error in errors)


def test_verification_provenance_survives_emit_import_roundtrip(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    reference = "pytest tests/test_core_handoff.py::test_verification_provenance_survives_emit_import_roundtrip"
    rt.ingest_event(_provenanced_event("provenanced claim", reference=reference))
    brief = rt.build_brief("demo", "status").to_dict()
    assert any("provenanced claim" in item for item in brief["recent_verified_facts"])

    packet = rt.emit_handoff("auditor", "demo", "verify").to_dict()
    row = next(row for row in packet["claims"] if "provenanced claim" in row["claim"])
    assert row["status"] == "verified"
    assert row["verification"]["method"] == "test"
    assert row["verification"]["reference"] == reference
    assert row["metadata"].get("verification", {}).get("reference") == reference

    imported = rt.import_handoff(packet, source_agent_id="auditor", thread_id="successor")
    assert imported["status"] == "ok"
    assert imported["failed_ingests"] == 0
    successor_facts = [f for f in rt.store.iter_facts(thread_id="successor") if f["claim"] == "provenanced claim"]
    assert successor_facts, "imported claim must become a successor-thread fact"
    assert successor_facts[0]["state"] == "verified", "provenance-only verified claims must stay verified"
    assert successor_facts[0]["metadata"]["verification"]["reference"] == reference
    rt.store.close()


def test_stale_evidence_emitted_as_asserted_with_flag(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    item = grounded_item(tmp_path, name="src/evidence.py", text="alpha\nbeta\n", start=1, end=2)
    rt.ingest_event(_event("stale handoff claim", payload={"claim": "stale handoff claim", "fact_state": "verified"}, evidence_items=[item]))
    (tmp_path / "src" / "evidence.py").write_text("alpha\nCHANGED\n", encoding="utf-8")
    rt.check_evidence_stale(source_root=str(tmp_path), dry_run=False)

    packet = rt.emit_handoff("auditor", "demo", "verify").to_dict()
    stale_rows = [row for row in packet["claims"] if "stale handoff claim" in row["claim"]]
    assert stale_rows, "stale evidence must be emitted in the handoff"
    assert stale_rows[0]["status"] == "asserted"
    assert stale_rows[0]["evidence_stale"] is True
    assert all(row["status"] != "verified" for row in stale_rows)

    imported = rt.import_handoff(packet, source_agent_id="auditor", thread_id="successor")
    assert imported["status"] == "ok"
    imported_facts = [f for f in rt.store.iter_facts(thread_id="successor") if "stale handoff claim" in f["claim"]]
    assert imported_facts
    assert imported_facts[0]["state"] == "asserted"
    assert imported_facts[0]["metadata"].get("evidence_stale") is True
    rt.store.close()


# --- item 14: import/emit error accounting --------------------------------


def test_import_handoff_counts_failed_ingests_and_caps_errors(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event = lambda event: {"status": "invalid", "errors": ["forced"]}  # type: ignore[method-assign]
    packet = {
        "schema_version": "v2",
        "packet_id": "failing-packet",
        "role": "auditor",
        "thread_id": "demo",
        "scope": "s",
        "locked_decisions": [f"decision {idx}" for idx in range(30)],
        "claims": [{"claim": "a claim", "status": "asserted"}],
        "required_validations": [],
        "checksum": "c",
    }
    out = rt.import_handoff(packet, thread_id="demo")
    assert out["status"] == "ok"
    assert out["failed_ingests"] == 32
    assert out["decisions_imported"] == 0
    assert out["claims_imported"] == 0
    assert len(out["errors"]) == 20, "errors list must be capped"
    assert all(isinstance(row, dict) for row in out["errors"])
    rt.store.close()


def test_emit_self_ingest_failure_records_fail_open(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(_event("a fact worth handing off", payload={"claim": "a fact worth handing off", "fact_state": "verified"}, evidence_items=[grounded_item(tmp_path)]))

    original = rt.ingest_event

    def _self_ingest_boom(event):
        raise RuntimeError("self ingest unavailable")

    rt.ingest_event = _self_ingest_boom  # type: ignore[method-assign]
    packet = rt.emit_handoff("auditor", "demo", "verify")
    assert packet.to_dict()["packet_id"]
    snap = rt.snapshot()
    assert snap["agent_memory_fail_open_events_24h"] >= 1
    assert snap["agent_memory_fail_open_events_total"] >= 1
    rt.ingest_event = original  # type: ignore[method-assign]
    rt.store.close()
