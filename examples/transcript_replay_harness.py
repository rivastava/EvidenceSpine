"""Replay a transcript fixture through the transcript adapter.

Run from the repo root:

    PYTHONPATH=src python examples/transcript_replay_harness.py
    PYTHONPATH=src python examples/transcript_replay_harness.py examples/replay_fixtures/contradiction_trace.json

Prints ingest accounting (new events vs deduped vs disabled), the derived
brief, the handoff, and the snapshot, all against a throwaway store.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from evidencespine import AgentMemoryRuntime, EvidenceSpineSettings
from evidencespine.adapters import TranscriptAdapter


def main() -> int:
    fixture_root = Path(__file__).resolve().parent / "replay_fixtures"
    fixture_path = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else fixture_root / "implementer_auditor_trace.json"
    )
    if not fixture_path.exists():
        print(f"error: fixture not found: {fixture_path}")
        return 2
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="evidencespine_replay_") as tmp:
        settings = EvidenceSpineSettings.from_env(base_dir=str(Path(tmp) / ".evidencespine"))
        runtime = AgentMemoryRuntime(config=settings.to_runtime_config())
        adapter = TranscriptAdapter(
            runtime, default_thread_id="replay_demo", source_agent_id="replay"
        )
        result = adapter.ingest_messages(payload)
        brief = adapter.brief("current status and next actions")
        handoff = adapter.handoff("auditor", "review replay trace")
        snapshot = runtime.snapshot()

    print(
        json.dumps(
            {
                "ingest": {
                    key: result[key]
                    for key in (
                        "status",
                        "seen",
                        "normalized",
                        "ingested",
                        "deduped",
                        "disabled",
                        "failed",
                        "new_events",
                    )
                },
                "brief_sections": {
                    key: len(brief.get(key, []))
                    for key in (
                        "current_goal",
                        "locked_decisions",
                        "recent_verified_facts",
                        "active_risks",
                        "open_items",
                        "next_actions",
                    )
                },
                "handoff_packet_id": handoff.get("packet_id"),
                "snapshot": {
                    key: snapshot[key]
                    for key in (
                        "events_total",
                        "agent_claim_span_citation_coverage_24h",
                        "agent_handoff_span_grounding_rate_24h",
                    )
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
