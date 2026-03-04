from __future__ import annotations

import json
from pathlib import Path

from evidencespine import AgentMemoryRuntime, EvidenceSpineSettings
from evidencespine.adapters import TranscriptAdapter


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "examples" / "replay_fixtures"


def _runtime(tmp_path: Path) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def _load_fixture(name: str):
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_replay_fixture_generates_nonempty_brief(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    adapter = TranscriptAdapter(runtime, default_thread_id="fixture_demo")
    adapter.ingest_messages(_load_fixture("tool_heavy_trace.json"))
    brief = adapter.brief("what matters now")
    assert any(brief[key] for key in ["locked_decisions", "open_items", "next_actions", "recent_verified_facts"]) 


def test_replay_fixture_preserves_locked_decisions(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    adapter = TranscriptAdapter(runtime, default_thread_id="fixture_demo")
    adapter.ingest_messages(_load_fixture("implementer_auditor_trace.json"))
    brief = adapter.brief("current status and next actions")
    combined = "\n".join(brief.get("locked_decisions", []))
    assert "retry middleware path" in combined


def test_replay_fixture_surfaces_risks_or_open_items(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    adapter = TranscriptAdapter(runtime, default_thread_id="fixture_demo")
    adapter.ingest_messages(_load_fixture("contradiction_trace.json"))
    brief = adapter.brief("what is risky")
    assert brief.get("open_items") or brief.get("active_risks")


def test_replay_fixture_handoff_contains_checksum_and_claims(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    adapter = TranscriptAdapter(runtime, default_thread_id="fixture_demo")
    adapter.ingest_messages(_load_fixture("implementer_auditor_trace.json"))
    packet = adapter.handoff("auditor", "verify replay trace")
    assert packet.get("checksum")
    assert isinstance(packet.get("claims"), list)
    assert packet.get("claims")
