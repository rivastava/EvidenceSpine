from __future__ import annotations

import json
import tempfile
from pathlib import Path

from evidencespine import AgentMemoryRuntime, EvidenceSpineSettings
from evidencespine.adapters import LangGraphAdapter


def main() -> int:
    fixture_path = Path(__file__).resolve().parent / "replay_fixtures" / "implementer_auditor_trace.json"
    state = {"messages": json.loads(fixture_path.read_text(encoding="utf-8"))}
    with tempfile.TemporaryDirectory(prefix="evidencespine_langgraph_") as tmp:
        settings = EvidenceSpineSettings.from_env(base_dir=str(Path(tmp) / ".evidencespine"))
        runtime = AgentMemoryRuntime(config=settings.to_runtime_config())
        adapter = LangGraphAdapter(runtime, default_thread_id="langgraph_replay")
        result = adapter.ingest_state(state)
        brief = adapter.brief("what matters now")
        handoff = adapter.handoff("auditor", "verify replay decisions")

    print(json.dumps({"ingest": result, "brief": brief, "handoff": handoff}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
