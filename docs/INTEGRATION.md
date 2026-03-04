# Integration Guide

## 1) Add to any project

```bash
pip install evidencespine
# or from source
pip install -e /path/to/evidencespine
```

## 2) Minimal runtime integration

```python
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings

settings = EvidenceSpineSettings.from_env(base_dir=".evidencespine")
runtime = AgentMemoryRuntime(config=settings.to_runtime_config())

runtime.ingest_event({
    "thread_id": "session_1",
    "event_type": "decision",
    "source_agent_id": "agent_a",
    "source_turn_id": "t-12",
    "payload": {"claim": "Use strategy B", "fact_state": "verified"},
    "evidence_refs": ["decision_log.md#L10"],
    "confidence": 0.82,
    "salience": 0.66,
})

brief = runtime.build_brief("session_1", "what matters now")
print(brief.to_dict())
```

## 3) Multi-agent handoff

```python
packet = runtime.emit_handoff(role="auditor", thread_id="session_1", scope="verify claims")
runtime.import_handoff(packet.to_dict(), source_agent_id="auditor_agent")
```

## 4) Optional contradiction hook

```python
from evidencespine.runtime import AgentMemoryRuntime, RuntimeHooks


def contradiction_pass(query, facts):
    # Return rows like: {"reason": "refute:fact_id"}
    return []

runtime = AgentMemoryRuntime(hooks=RuntimeHooks(contradiction_pass=contradiction_pass))
```

## 5) Hybrid retrieval mode

Use env:

```bash
export EVIDENCESPINE_RETRIEVAL_MODE=hybrid
export EVIDENCESPINE_RETRIEVAL_LEXICAL_WEIGHT=1.0
export EVIDENCESPINE_RETRIEVAL_VECTOR_WEIGHT=0.35
```

Or inject your own backend:

```python
from evidencespine.runtime import AgentMemoryRuntime

class MyVectorBackend:
    def score_texts(self, query, texts):
        return [0.0 for _ in texts]

runtime = AgentMemoryRuntime(vector_backend=MyVectorBackend())
```

## 6) Transcript-first integration (recommended)

```python
from evidencespine import AgentMemoryRuntime, EvidenceSpineSettings
from evidencespine.adapters import TranscriptAdapter

rt = AgentMemoryRuntime(config=EvidenceSpineSettings.from_env().to_runtime_config())
adapter = TranscriptAdapter(rt, default_thread_id="thread_default")

normalized = adapter.normalize_messages(
    [
        {"role": "user", "content": "Check drift"},
        {"role": "assistant", "content": "Patch complete"},
        {"role": "tool", "content": "pytest passed"},
    ]
)

result = adapter.ingest_messages(normalized)
brief = adapter.brief("what matters now")
handoff = adapter.handoff("auditor", "verify latest claims")
```

Use this when your runtime already exposes transcript-like `messages[]` and you want the smallest dependency-free integration surface.

## 7) Framework wrappers (drop-in convenience)

```python
from evidencespine import AgentMemoryRuntime, EvidenceSpineSettings
from evidencespine.adapters import LangGraphAdapter, AutoGenAdapter

rt = AgentMemoryRuntime(config=EvidenceSpineSettings.from_env().to_runtime_config())

lg = LangGraphAdapter(rt, default_thread_id="thread_lg")
lg.ingest_state({"messages": [{"role": "user", "content": "Check drift"}]})
normalized_lg = lg.normalize_state({"messages": [{"role": "tool", "content": "pytest ok"}]})

ag = AutoGenAdapter(rt, default_thread_id="thread_ag")
ag.ingest_messages([{"source": "assistant", "content": "Patch complete"}])
normalized_ag = ag.normalize_messages([{"source": "function", "content": "tool output"}])
```

These wrappers stay dependency-free. They are schema-level adapters, not hard integrations with the framework packages.

## 8) Replay validation

Run the bundled replay examples to inspect how raw traces become briefs and handoff packets:

```bash
PYTHONPATH=src python examples/transcript_replay_harness.py \
  examples/replay_fixtures/implementer_auditor_trace.json

PYTHONPATH=src python examples/langgraph_replay_demo.py
```

## 9) Operational checklist
- Persist events with evidence refs.
- Mark fact state correctly (asserted vs verified).
- Generate brief before each major agent action.
- Emit handoff packet when switching role/agent.
- Watch `snapshot()` metrics for stale or low citation coverage.

## Claude Code

For a Claude Code specific workflow, see:
- `docs/CLAUDE_CODE.md`
- `examples/claude_code_usage.py`
- `docs/ADAPTERS.md`
