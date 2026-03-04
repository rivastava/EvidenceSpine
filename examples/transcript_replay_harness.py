from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from evidencespine import AgentMemoryRuntime, EvidenceSpineSettings
from evidencespine.adapters import TranscriptAdapter


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python examples/transcript_replay_harness.py <fixture.json>")
        return 2

    fixture_path = Path(sys.argv[1]).resolve()
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="evidencespine_replay_") as tmp:
        settings = EvidenceSpineSettings.from_env(base_dir=str(Path(tmp) / ".evidencespine"))
        runtime = AgentMemoryRuntime(config=settings.to_runtime_config())
        adapter = TranscriptAdapter(runtime, default_thread_id="replay_demo", source_agent_id="replay")
        result = adapter.ingest_messages(payload)
        brief = adapter.brief("current status and next actions")
        handoff = adapter.handoff("auditor", "review replay trace")
        snapshot = runtime.snapshot()

    print(json.dumps({"ingest": result, "brief": brief, "handoff": handoff, "snapshot": snapshot}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
