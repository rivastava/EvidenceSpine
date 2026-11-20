# Adapters

Adapters are the transcript-first integration path: normalize a provider's
`messages[]` into EvidenceSpine events without importing the provider's
framework package. Three adapters ship:

- `TranscriptAdapter` — the generic default;
- `LangGraphAdapter` — `ingest_state` / `normalize_state`;
- `AutoGenAdapter` — `ingest_messages` / `normalize_messages`.

The framework wrappers are thin schema layers over the transcript adapter;
they do not import LangGraph or AutoGen.

## Recommended default: TranscriptAdapter

```python
from evidencespine import AgentMemoryRuntime, EvidenceSpineSettings
from evidencespine.adapters import TranscriptAdapter

runtime = AgentMemoryRuntime(config=EvidenceSpineSettings.from_env().to_runtime_config())
adapter = TranscriptAdapter(runtime, default_thread_id="demo", source_agent_id="demo_agent")

result = adapter.ingest_messages([
    {"role": "user", "content": "Check drift"},
    {"role": "assistant", "content": "Patch applied"},
])

print(result["status"])        # ok | partial | failed | disabled | invalid
print(result["ingested"])      # newly persisted events
print(result["deduped"])       # replayed events (same hash inside the dedupe window)
print(result["new_events"])    # alias of ingested
```

`brief(...)` and `handoff(...)` build on the same runtime.

## Ingest result accounting (0.6.0)

`AdapterIngestResult` distinguishes what happened to every normalized message:

| Field | Meaning |
|---|---|
| `status` | `ok` (all captured or nothing to do), `partial` (some captured, some failed/disabled), `failed` (nothing captured and >0 failed), `disabled` (memory disabled for all rows), `invalid` (unrecognized input shape / invalid thread) |
| `seen` | raw messages received |
| `normalized` | rows produced by normalization |
| `ingested` | **newly persisted** events only |
| `deduped` | events the runtime accepted as duplicates (replays) |
| `disabled` | events not persisted because memory is disabled |
| `failed` | events the runtime rejected/failed |
| `skipped` | `seen - normalized` (empty/unparseable messages) |
| `new_events` | alias of `ingested` (backward compatibility) |
| `reason` | present for invalid/partial/failed/disabled |

Earlier releases reported `status: ok, ingested: N` while half the N were
deduped or disabled; callers could not tell "captured" from "replayed". The
status and counters above are the honest accounting.

## Supported input shapes

`TranscriptAdapter` accepts:

- a `list` of message dicts;
- a state dict containing a `messages` list (`{"messages": [...]}`);
- a list of plain objects with attributes (`role`, `content`, `id`, ...).

Message content may be:

- a plain string;
- a content-parts list (`[{"type": "text", "text": "hello"}]`) — text parts
  are joined, so OpenAI/LangChain-style content lists no longer stringify into
  a Python repr;
- tuples of `(role, content)` in a messages list.

Unrecognized shapes (for example `ingest_state({"foo": 1})`) return
`status: "invalid"` with a reason instead of a silent empty success.

## Normalization rules

Default role mapping:

- `user|human|system -> intent`
- `assistant|ai|agent -> decision`
- `tool|function -> outcome`
- everything else -> `reflection`

Each normalized `NormalizedTranscriptMessage` carries: `role`,
`event_type`, `content`, `turn_id`, `evidence_ref`, `evidence_refs`,
`evidence_items`, `state_context`, `confidence`, `salience`,
`metadata`.

Notes:

- `source_agent_id` is per adapter (`source_agent_id` ctor arg); per-message
  agent names are kept in `metadata`.
- `turn_id` comes from the message id when present, else a stable
  `<prefix>:<index>`.
- Caller-supplied `evidence_items` and `state_context` are preserved
  through normalization and ingestion; the adapters never manufacture spans or
  control-state.
- Messages are ingested as `fact_state: "asserted"` with the content as the
  claim: adapters capture transcript state, they do not assert verification.

## Wrapper adapters

```python
from evidencespine.adapters import LangGraphAdapter, AutoGenAdapter

lg = LangGraphAdapter(runtime, default_thread_id="thread_lg")
lg.ingest_state({"messages": [{"role": "user", "content": "Check drift"}]})
normalized_lg = lg.normalize_state({"messages": [{"role": "tool", "content": "pytest ok"}]})

ag = AutoGenAdapter(runtime, default_thread_id="thread_ag")
ag.ingest_messages([{"source": "assistant", "content": "Patch complete"}])
normalized_ag = ag.normalize_messages([{"source": "function", "content": "tool output"}])
```

Pure normalization helpers are public so you can inspect/transform rows before
ingesting them.

## Replay harnesses

```sh
PYTHONPATH=src python examples/transcript_replay_harness.py \
  examples/replay_fixtures/implementer_auditor_trace.json

PYTHONPATH=src python examples/langgraph_replay_demo.py
```

The replay fixtures are also used by the test suite to keep adapter
transformations stable.

## Limitations

- Schema-level only: adapters do not install or import framework packages and
  do not replicate provider-specific fields beyond the bounded metadata subset.
- Evidence is not synthesized: if the transcript carries no refs/items, the
  resulting facts are asserted (not verified).
- Adapters improve continuity and handoff quality; they are not a replacement
  for an orchestration runtime.
