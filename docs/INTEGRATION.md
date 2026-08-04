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
    "evidence_items": [
        {
            "source_id": "decision_log.md",
            "line_start": 10,
            "line_end": 12,
            "excerpt": "Use strategy B",
        }
    ],
    "state_context": {
        "scope_id": "strategy-b-rollout",
        "state_kind": "agent_local_work",
        "status": "active",
        "owner_agent_id": "agent_a",
    },
    "confidence": 0.82,
    "salience": 0.66,
})

brief = runtime.build_brief("session_1", "what matters now")
print(brief.to_dict())
print(runtime.query_view("active_scopes", thread_id="session_1").to_dict())
```

`build_brief(...).to_dict()` now emits structured `citations` plus a legacy `citation_refs` alias.
`query_view(...)` gives agents a compact current-state surface without tailing raw JSONL.

## 2a) Minimal CLI integration

```bash
evidencespine ingest \
  --thread-id session_1 \
  --event-type decision \
  --source-agent-id agent_a \
  --source-turn-id t-12 \
  --claim "Use strategy B" \
  --fact-state verified \
  --scope-id strategy-b-rollout \
  --state-kind agent_local_work \
  --status active \
  --owner-agent-id agent_a \
  --evidence-ref "decision_log.md#L10" \
  --evidence-item-json '{"source_id":"decision_log.md","line_start":10,"line_end":12,"excerpt":"Use strategy B"}' \
  --json

evidencespine view active-scopes --thread-id session_1 --json
```

## 3) Multi-agent handoff

```python
packet = runtime.emit_handoff(role="auditor", thread_id="session_1", scope="verify claims")
runtime.import_handoff(packet.to_dict(), source_agent_id="auditor_agent")
```

Handoff claim rows preserve `state_context` when the brief claim carried it, so ownership and gate freshness survive role changes.

## 4) Optional contradiction hook

```python
from evidencespine.runtime import AgentMemoryRuntime, RuntimeHooks


def contradiction_pass(query, facts):
    # Return rows like: {"reason": "refute:fact_id"}
    return []

runtime = AgentMemoryRuntime(hooks=RuntimeHooks(contradiction_pass=contradiction_pass))
```

Optional reconciliation hook:

```python
def reconcile_state(thread_id, active_scope_rows):
    return [
        {
            "thread_id": thread_id,
            "event_type": "reflection",
            "source_agent_id": "runtime_probe",
            "payload": {"claim": "release gate still blocked", "fact_state": "verified"},
            "state_context": {
                "scope_id": "release-gate",
                "state_kind": "global_blocker",
                "status": "blocked",
                "state_basis": "runtime_validated",
                "validated_at": "2026-03-18T09:30:00Z",
                "validated_by": "smoke",
                "fresh_until": "2026-03-18T10:30:00Z",
            },
        }
    ]

runtime = AgentMemoryRuntime(hooks=RuntimeHooks(reconcile_state=reconcile_state))
runtime.reconcile("session_1")
```

## 5) Hybrid retrieval mode

Retrieval uses BM25 lexical scoring (fallback: jaccard). Enable hybrid or
vector mode and pick a vector backend:

```bash
export EVIDENCESPINE_RETRIEVAL_MODE=hybrid   # or vector
export EVIDENCESPINE_RETRIEVAL_LEXICAL_WEIGHT=1.0
export EVIDENCESPINE_RETRIEVAL_VECTOR_WEIGHT=0.35
export EVIDENCESPINE_EMBEDDING_BACKEND=auto  # auto | fastembed | hashing
export EVIDENCESPINE_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

- `auto` (default) uses fastembed when the `[embeddings]` extra is installed,
  otherwise falls back to a dependency-free hashing backend.
- `fastembed` requires the extra: `pip install evidencespine[embeddings]`.
- `hashing` forces the dependency-free backend.

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
        {
            "role": "assistant",
            "content": "Patch complete",
            "evidence_items": [{"source_id": "patch.diff", "line_start": 7, "line_end": 9}],
        },
        {"role": "tool", "content": "pytest passed"},
    ]
)

result = adapter.ingest_messages(normalized)
brief = adapter.brief("what matters now")
handoff = adapter.handoff("auditor", "verify latest claims")
```

Use this when your runtime already exposes transcript-like `messages[]` and you want the smallest dependency-free integration surface.
Caller-supplied `evidence_items` and `state_context` are preserved through normalization and ingestion; the adapter does not invent span or control-state data on its own.

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

## 8a) MCP server (optional extra)

```bash
pip install "evidencespine[mcp]"
evidencespine mcp                       # stdio transport
evidencespine mcp --transport streamable-http --port 8000
```

Tools: `ingest_event`, `build_brief`, `query_view`, `emit_handoff`,
`import_handoff`, `reconcile`, `snapshot`, `prune`.
Resources: `evidencespine://brief/{thread_id}`, `evidencespine://view/{view}`,
`evidencespine://state/{scope_id}`, `evidencespine://snapshot`.
Prompts: `session_start`, `handoff_receive`, `handoff_send`.

## 8b) Harness delivery layer (auto-recall / auto-retain)

```bash
evidencespine harness install --harness all
evidencespine harness debug
```

Writes a Claude Code plugin manifest, an opencode plugin (brief injection at
session start, retention through compaction, handoff on session end), and a
Cursor MCP config. Session hooks fail open: without a store, `session-start`
returns a one-line notice instead of failing the session.

### Supported harness matrix

| Harness | Install target | What is wired |
| --- | --- | --- |
| opencode | `.opencode/plugins/evidencespine.ts` (or global) | plugin + MCP server, session-start/stop/compaction |
| claude-code | `.claude-plugin/plugin.json` | session-start/stop/precompact hooks |
| cursor | `.cursor/mcp.json` | MCP server registration |
| git | `.git/hooks/` post-commit/post-merge | commit spans + test-record |
| codex | — (manual, see `docs/CODEX.md`) | AGENTS.md guidance + manual MCP registration; full `install_codex` harness wiring is planned, not yet shipped |

The MCP server itself is harness-agnostic: any MCP-capable client (opencode,
Cursor, Claude Code, Codex CLI/IDE/desktop, generic MCP apps) gets the same
tools and resources, so agents reach the spine even without a harness install.
A2A, the async runtime, the CLI, and the framework adapters
(`TranscriptAdapter`, LangGraph, AutoGen) are additional harness-agnostic
channels.

### How agents discover EvidenceSpine

Guidance reaches agents through three complementary layers:

1. **`AGENTS.md` (committed at the repo root)** — the opencode, Codex, Cursor,
   and Claude Code convention for agent instructions: a concise primer on what
   the spine is and decision rules for when to consult it (consult a brief
   before asserting project state; ground + mark verified when claiming
   something is fixed; re-verify evidence after edits; import/emit handoffs at
   role boundaries). The same file is read by all four harnesses — one source,
   no per-harness instruction sets. It is generated from the canonical
   `usage_guide_markdown()` constant (`src/evidencespine/usage.py`) and a sync
   test fails if the two drift.
2. **MCP tool and resource descriptions** — every tool teaches its own use
   case; resources (`evidencespine://brief/{thread}`, `//view/{view}`,
   `//state/{scope}`, `//snapshot`) are progressive-disclosure surfaces. The
   `evidencespine://guide` resource mirrors the AGENTS.md rules for any MCP
   agent at runtime.
3. **The session-start injection** — the auto-recall brief opens with the
   operating rules (locked decisions are binding unless superseded; act
   against a verified fact only by recording the contradiction; open items and
   next actions are the queue) plus the decision rules (consult the spine
   before asserting project state; ground what you mark verified; re-verify
   after edits; emit a handoff at role change), injected automatically by the
   opencode plugin and claude-code hooks.

## 8c) A2A protocol server (optional extra)

```bash
pip install "evidencespine[a2a]"
evidencespine a2a --host 127.0.0.1 --port 8765
curl http://127.0.0.1:8765/.well-known/agent-card.json
```

The agent card advertises `memory.read`, `memory.write`, `memory.handoff`,
and `memory.health` skills. Send an A2A message containing a JSON body
`{"action": "build_brief", "thread_id": "...", "params": {...}}` (or plain
text, treated as a brief query) to the JSON-RPC endpoint.

## 8d) Async wrapper

```python
from evidencespine import AsyncAgentMemoryRuntime

rt = AsyncAgentMemoryRuntime(base_dir=".evidencespine")
await rt.ingest_event({...})
brief = await rt.build_brief("session_1", "what matters now")
```

All blocking I/O runs in worker threads; methods mirror the sync runtime and
return plain dicts.

## 8e) TTL archival

```bash
evidencespine prune --ttl-hours 720 --dry-run --json   # preview
evidencespine prune --ttl-hours 720 --json            # delete old rows
```

Rows without a parseable timestamp are always kept. The runtime method
`runtime.prune(...)` and the MCP `prune` tool expose the same behavior.

## 8f) Realtime debate chat

`evidencespine chat` turns the spine into a live message bus: one polling
loop per role, each reading the room from the store, replying through an LLM
backend, and publishing back. Agents read each other's messages before every
reply; the debate ends on consensus (`AGREE:` from every role), a quiet
period, a message budget, or a duration cap.

```bash
# quick debate with the default roles
evidencespine chat --topic "Is this feature worth building?"

# long, self-sustaining session (see docs/DEBATE.md for the full runbook)
evidencespine chat --topic "<topic>" --minutes 45 --facilitate \
  --max-messages 200 --window-size 12 --max-reply-words 40
```

Chat messages are stored as room events (monotonic `chat_seq`), never as
facts, so they do not pollute briefs. See `docs/DEBATE.md` for all controls
and the opencode verification notes.

## 8g) Grounded evidence: spans, policy, drift, provenance

Grounded evidence binds claims to exact, checksummed file excerpts.

```bash
# build a grounded evidence item from a file:line ref
evidencespine ground "src/mod.py#L10-L20"

# ingest a verified claim that is actually grounded (survives the policy)
evidencespine ingest --claim "fix shipped" --fact-state verified \
  --ground-ref "src/mod.py#L10-L20" --thread-id demo --event-type outcome \
  --source-agent-id me --source-turn-id t1

# policy: fact_state=verified without a grounded item or provenance is
# stored as asserted (metadata.policy = verified_requires_span).
# Disable with EVIDENCESPINE_VERIFIED_REQUIRES_SPAN=0.

# drift-check: re-verify grounded evidence against live files
evidencespine drift-check --source-root .            # preview
evidencespine drift-check --source-root . --apply    # write evidence_stale flags
# stale facts surface in `view stale_claims`, briefs (STALE EVIDENCE risks),
# and the snapshot metric agent_evidence_stale_count_24h.

# provenance: record how a fact was verified (test/gate/tool/manual)
evidencespine verify --fact-id amf_... --method test \
  --reference "pytest tests/test_mod.py" --verified-by qa
```

The same operations are MCP tools: `ground`, `check_drift`, `verify_fact`.

## 8h) Thin git/test hooks

```bash
# install post-commit + post-merge hooks (span ingestion per commit)
evidencespine harness git install-hook --target-dir . --executable evidencespine

# manual commit record with grounded spans from diff hunks
evidencespine harness git git-hook --sha <sha> --repo-dir .

# record a test run (green -> verified fact with provenance)
evidencespine harness git test-record --status passed --command "pytest tests/"
```

## 9) Operational checklist
- Persist events with evidence refs.
- Mark fact state correctly (asserted vs verified).
- Use `state_context` for gates, blockers, ownership, and freshness when those semantics matter.
- Generate brief before each major agent action.
- Use `query_view("active_scopes")` or the CLI `view` command before role changes.
- Emit handoff packet when switching role/agent.
- Watch `snapshot()` metrics for stale or low citation coverage.

## Claude Code

For a Claude Code specific workflow, see:
- `docs/CLAUDE_CODE.md`
- `examples/claude_code_usage.py`
- `docs/ADAPTERS.md`
