# Claude Code Integration

This guide shows how to use EvidenceSpine as a sidecar memory layer for Claude Code workflows.

The point is not to replace Claude. The point is to stop passing large raw chat histories between sessions or roles and instead pass a bounded, evidence-backed brief.

It is still a side-car, not the final runtime authority. If you have live runtime signals, reconcile them into EvidenceSpine rather than pretending the memory layer is the organism itself.

## What to log

Log only high-value events:
1. locked decisions
2. verified facts
3. contradictions
4. open risks
5. handoffs between roles such as `implementer` and `auditor`

Do not dump full transcripts into EvidenceSpine.

## Minimal pattern

1. After an important step, ingest an event.
2. Before the next prompt, build a brief.
3. When switching roles or sessions, emit a handoff packet.

## Install

```bash
pip install evidencespine
```

Or from source:

```bash
pip install -e /path/to/evidencespine
```

## Basic runtime

```python
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings

settings = EvidenceSpineSettings.from_env(base_dir=".evidencespine")
runtime = AgentMemoryRuntime(config=settings.to_runtime_config())
```

## Ingest a Claude Code event

```python
runtime.ingest_event(
    {
        "thread_id": "bugfix_auth_timeout",
        "event_type": "decision",
        "role": "implementer",
        "source_agent_id": "claude_code",
        "source_turn_id": "turn_17",
        "payload": {
            "claim": "Use additive patch only. Do not touch auth token rotation.",
            "decision": "Patch retry path in request middleware",
            "fact_state": "verified",
            "next_actions": ["auditor should verify timeout edge cases"],
        },
        "evidence_refs": ["src/middleware/request_timeout.py:42"],
        "state_context": {
            "scope_id": "auth-timeout-fix",
            "state_kind": "agent_local_work",
            "status": "active",
            "owner_agent_id": "implementer",
        },
        "confidence": 0.87,
        "salience": 0.74,
    }
)
```

## Build a bounded brief before the next prompt

```python
brief = runtime.build_brief(
    "bugfix_auth_timeout",
    "current goal, locked decisions, verified facts, active risks"
)

print(brief.to_dict())
print(runtime.query_view("active_scopes", thread_id="bugfix_auth_timeout").to_dict())
```

Use that brief in the next Claude Code prompt instead of pasting a long raw thread.

If you want markdown-style output, render the dictionary into your preferred prompt format before sending it to Claude.

## Optional transcript import path

If you already have a raw transcript-like `messages[]` payload and want to import it into EvidenceSpine before building the brief, use `TranscriptAdapter`:

```python
from evidencespine.adapters import TranscriptAdapter

adapter = TranscriptAdapter(runtime, default_thread_id="bugfix_auth_timeout")
adapter.ingest_messages(
    [
        {"role": "user", "content": "Audit the timeout regression"},
        {"role": "assistant", "content": "Patched retry middleware only"},
        {"role": "tool", "content": "pytest passed"},
    ]
)
```

This is the recommended default when integrating from transcript-oriented tools because it keeps the transform explicit and replayable.
Caller-provided `state_context` survives transcript normalization if you want to mark ownership, blockers, or pending gates from the calling workflow.

## Emit a handoff packet

```python
packet = runtime.emit_handoff(
    role="auditor",
    thread_id="bugfix_auth_timeout",
    scope="verify patch and regression risk"
)

print(packet.to_dict())
```

## Suggested Claude Code workflow

### Implementer loop

After a significant patch or decision:

```python
runtime.ingest_event(...)
```

Before asking Claude Code for the next step:

```python
brief = runtime.build_brief(thread_id, "what matters now")
```

Prompt shape:

```text
Use this EvidenceSpine brief as the working state.

<brief markdown here>

Task:
- implement the next safe step
- preserve locked decisions
- call out contradictions if any
```

### Auditor loop

When switching from implementer to auditor:

```python
packet = runtime.emit_handoff("auditor", thread_id, scope="verify latest claims")
```

Then give the auditor agent:
1. the handoff packet
2. the file diff or repo state
3. the narrow validation task

## Recommended conventions

Use stable role names:
1. `implementer`
2. `auditor`
3. `researcher`
4. `operator`

Use stable fact states:
1. `asserted`
2. `verified`
3. `contradicted`
4. `superseded`

Use `state_context` for operational semantics:
1. `agent_local_work`
2. `global_blocker`
3. `pending_gate`
4. `runtime_validated_state`

Use real evidence refs whenever possible:
1. file paths with lines
2. test names
3. report paths
4. benchmark artifact paths

## Good use cases for Claude Code

1. long bugfix threads
2. implementer to reviewer handoffs
3. parallel agent workflows
4. architecture changes with many locked constraints
5. overnight or resumed sessions where context drift usually happens

## Anti-patterns

Do not use EvidenceSpine as:
1. a raw transcript archive
2. a replacement for source control
3. a place for unverified claims without evidence refs

## Example file

See:
- `examples/claude_code_usage.py`
- `examples/transcript_replay_harness.py`
- `docs/ADAPTERS.md`

## Operational checklist

1. Ingest only high-signal events.
2. Mark `fact_state` correctly.
3. Build the brief before major prompt transitions.
4. Check `query_view("active_scopes")` or `query_view("open_gates")` before role switches.
5. Emit handoff packets when switching roles.
6. Watch `snapshot()` for citation, freshness, and conflict metrics.
