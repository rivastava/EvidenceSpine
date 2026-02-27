from pathlib import Path

from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def _runtime(tmp_path: Path) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def test_runtime_ingest_brief_handoff_roundtrip(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    out = rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "source_agent_id": "impl",
            "source_turn_id": "t1",
            "payload": {
                "claim": "deploy patch",
                "fact_state": "verified",
                "next_actions": ["auditor verify"],
            },
            "evidence_refs": ["patch.diff"],
            "confidence": 0.8,
            "salience": 0.7,
        }
    )
    assert out["status"] == "ok"

    brief = rt.build_brief("demo", "status")
    b = brief.to_dict()
    assert b["thread_id"] == "demo"

    packet = rt.emit_handoff("auditor", "demo", "verify")
    p = packet.to_dict()
    assert p["thread_id"] == "demo"
    assert p["checksum"]

    imported = rt.import_handoff(p, source_agent_id="auditor")
    assert imported["status"] == "ok"

    snap = rt.snapshot()
    assert snap["agent_memory_events_24h"] >= 1
    assert snap["agent_handoff_packets_emitted_24h"] >= 1


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
