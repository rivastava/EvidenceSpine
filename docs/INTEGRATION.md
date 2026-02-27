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

## 6) Framework adapters (drop-in)

```python
from evidencespine import AgentMemoryRuntime, EvidenceSpineSettings
from evidencespine.adapters import LangGraphAdapter, AutoGenAdapter

rt = AgentMemoryRuntime(config=EvidenceSpineSettings.from_env().to_runtime_config())

lg = LangGraphAdapter(rt, default_thread_id="thread_lg")
lg.ingest_state({"messages": [{"role": "user", "content": "Check drift"}]})

ag = AutoGenAdapter(rt, default_thread_id="thread_ag")
ag.ingest_messages([{"source": "assistant", "content": "Patch complete"}])
```

## 7) Operational checklist
- Persist events with evidence refs.
- Mark fact state correctly (asserted vs verified).
- Generate brief before each major agent action.
- Emit handoff packet when switching role/agent.
- Watch `snapshot()` metrics for stale or low citation coverage.

## Claude Code

For a Claude Code specific workflow, see:
- `docs/CLAUDE_CODE.md`
- `examples/claude_code_usage.py`
