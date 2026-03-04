# Adapters

EvidenceSpine v0.2 exposes a generic transcript adapter and thin framework wrappers.

## Recommended default: `TranscriptAdapter`

Use `TranscriptAdapter` when you already have a `messages[]` or transcript-like payload and want a dependency-free import path.

```python
from evidencespine import AgentMemoryRuntime, EvidenceSpineSettings
from evidencespine.adapters import TranscriptAdapter

runtime = AgentMemoryRuntime(config=EvidenceSpineSettings.from_env().to_runtime_config())
adapter = TranscriptAdapter(runtime, default_thread_id="demo")

result = adapter.ingest_messages(
    [
        {"role": "user", "content": "Check drift"},
        {"role": "assistant", "content": "Patch applied"},
    ]
)
```

## Supported input shapes

`TranscriptAdapter` accepts:
- dict state containing `messages`
- plain `list[dict]`
- plain object message lists (with attributes like `role`, `content`, `id`)

## Normalization rules

Default role mapping:
- `user|human|system -> intent`
- `assistant|ai|agent -> decision`
- `tool|function -> outcome`
- everything else -> `reflection`

Normalization outputs `NormalizedTranscriptMessage` rows with:
- `role`
- `event_type`
- `content`
- `turn_id`
- `evidence_ref`
- `confidence`
- `salience`
- `metadata`

## Wrapper adapters

Framework wrappers remain available and are built on top of the transcript adapter:
- `LangGraphAdapter`
- `AutoGenAdapter`

These wrappers keep the same ingest/brief/handoff behavior while also exposing pure normalization helpers:
- `LangGraphAdapter.normalize_state(...)`
- `AutoGenAdapter.normalize_messages(...)`

## Replay harnesses

Use the replay harness examples to validate transformations on realistic traces:

```bash
python examples/transcript_replay_harness.py examples/replay_fixtures/implementer_auditor_trace.json
python examples/langgraph_replay_demo.py
```

## Limitations

- Adapters are schema-level wrappers only; they do not import framework packages.
- EvidenceSpine does not preserve every provider-specific field by default; adapter normalization keeps a bounded subset plus metadata.
- The adapters are designed to improve handoff quality and continuity, not to replace full orchestration runtimes.
