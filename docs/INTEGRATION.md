# Integration guide

EvidenceSpine is an evidence-bound memory side-car for coding agents. Anything
that can run a process (MCP stdio), call HTTP, or import Python can use it.
This guide starts with the portable path and then gives per-client quick
starts; the complete per-client matrix is in
[HARNESS_MATRIX.md](HARNESS_MATRIX.md).

## 1) Install

```sh
pip install evidencespine
# or from source
pip install -e /path/to/evidencespine

# optional extras
pip install "evidencespine[mcp]"          # MCP server
pip install "evidencespine[a2a]"          # A2A server
pip install "evidencespine[embeddings]"   # fastembed vector backend
```

## 2) Portable delivery in one command

```sh
cd /path/to/repo
evidencespine harness install --harness all --target-dir .   # every supported target
evidencespine harness doctor --target-dir . --json           # validate what was written
evidencespine harness debug  --base-dir .evidencespine --json
```

Or install only what you use:

```sh
evidencespine harness install --harness claude-code      # skills-dir plugin + rules
evidencespine harness install --harness codex            # .codex/config.toml + hooks
evidencespine harness install --harness opencode         # .opencode/plugins/evidencespine.ts
evidencespine harness install --harness cursor           # .cursor/mcp.json + rules
evidencespine harness install --harness vscode-copilot   # portable .mcp.json
evidencespine harness install --harness dsh              # $DSH_HOME/cordis.patch.yml
evidencespine harness install --harness generic          # canonical + projections
evidencespine harness install --harness agents-md        # AGENTS.md managed block
evidencespine harness install --harness git              # post-commit/post-merge hooks
```

Installs are merge-preserving and idempotent: unrelated JSON/YAML/TOML is kept,
hooks are appended and deduped, and an unparseable user file is never
overwritten (a backup path is reported). Use `--dry-run` to preview. For
clients with no installer (Windsurf/Devin, Zed, Gemini CLI, Antigravity,
Cline, Roo, Continue, Junie, Warp, Amp, JetBrains, Claude Desktop) use:

```sh
evidencespine harness print-config --harness windsurf
evidencespine harness print-config --harness zed --scope global
evidencespine harness print-config --harness all --json
```

## 3) Minimal runtime integration (Python)

```python
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings

settings = EvidenceSpineSettings.from_env(base_dir=".evidencespine")
runtime = AgentMemoryRuntime(config=settings.to_runtime_config())

runtime.ingest_event({
    "thread_id": "session_1",              # or let the CLI/harness resolve the workspace thread
    "event_type": "decision",
    "source_agent_id": "agent_a",
    "source_turn_id": "t-12",
    "payload": {"claim": "Use strategy B", "fact_state": "verified"},
    "evidence_refs": ["decision_log.md#L10-L12"],
    "evidence_items": [{
        "source_id": "decision_log.md",
        "line_start": 10,
        "line_end": 12,
        "excerpt": "Use strategy B",
        "checksum": "sha256:<sha256 of the excerpt>",
    }],
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

Grounding is mandatory for `"verified"`: without a checksummed excerpt (or
verification provenance) the claim is stored as `asserted` and the result
reports `policy_downgrades`. Prefer letting the CLI ground the span for you
(see below) instead of hand-computing checksums.

## 4) Minimal CLI integration

```sh
evidencespine ingest \
  --thread-id session_1 --event-type decision --source-agent-id agent_a \
  --source-turn-id t-12 --claim "Use strategy B" --fact-state verified \
  --ground-ref "decision_log.md#L10-L12" \
  --scope-id strategy-b-rollout --state-kind agent_local_work \
  --status active --owner-agent-id agent_a --json

evidencespine brief --thread-id session_1 --query "what matters now" --json
evidencespine view active-scopes --thread-id session_1 --json
```

Exit codes are machine-consumable: `0` ok/deduped/no-op, `2`
invalid/missing/ungroundable, `3` fail-open/disabled/partial, and
`migrate` verification mismatch exits `1`. Add `--json` for the payload.

## 5) MCP clients (stdio first, HTTP when remote)

Register this command as a stdio MCP server:

```sh
evidencespine mcp --base-dir /absolute/path/to/repo/.evidencespine
```

An absolute `--base-dir` avoids surprises when the client's cwd differs.
Tools: `ingest_event, build_brief, query_view, emit_handoff, import_handoff,
ground, check_drift, verify_fact, reconcile, snapshot, prune`.
Resources: `evidencespine://guide`, `//brief/{thread}`, `//view/{view}`,
`//state/{scope}`, `//snapshot`. The server also sends the usage guide in
`initialize.instructions`, so agents get the operating rules with no repo
changes.

For a shared/remote surface, run streamable-HTTP and authenticate:

```sh
export EVIDENCESPINE_MCP_AUTH_TOKEN="$(openssl rand -hex 32)"
evidencespine mcp --transport streamable-http --host 127.0.0.1 --port 8000 --path /mcp
# clients: {"type":"http","url":"http://127.0.0.1:8000/mcp",
#           "headers":{"Authorization":"Bearer $EVIDENCESPINE_MCP_AUTH_TOKEN"}}
```

The server refuses unauthenticated non-loopback binds; TLS is not terminated by
EvidenceSpine (put it behind a proxy). See [SECURITY.md](SECURITY.md).

## 6) Per-client quick starts

### Claude Code

```sh
evidencespine harness install --harness claude-code --target-dir .
# launch Claude Code from the repo root, accept the workspace-trust dialog
claude plugin validate --strict .claude/skills/evidencespine
claude mcp list
```

Delivers SessionStart recall, PreCompact retention, SessionEnd ingest, a skill
and `.claude/rules/evidencespine.md`. Settings hooks are opt-in
(`--with-settings-hooks`; both copies would run). Details:
[CLAUDE_CODE.md](CLAUDE_CODE.md).

### Codex

```sh
evidencespine harness install --harness codex --target-dir .   # or --scope global
codex mcp list
# project hooks need a one-time trust review: run /hooks in Codex
```

Writes `.codex/config.toml` (safe TOML merge, stdio or `--mcp-url` HTTP) and
`.codex/hooks.json` (SessionStart recall, SessionEnd retain, PreCompact
ingest). Details: [CODEX.md](CODEX.md).

### opencode

```sh
evidencespine harness install --harness opencode --target-dir .
opencode plugin list
opencode mcp list
```

Writes the V2 plugin `.opencode/plugins/evidencespine.ts`: MCP registration,
`context` + `compaction` recall, tool-after capture, lifecycle stop.
Requires opencode >= 2.x. Details: [OPENCODE.md](OPENCODE.md).

### Cursor

```sh
evidencespine harness install --harness cursor --target-dir .
```

Writes `.cursor/mcp.json` (merge), `.cursor/rules/evidencespine.mdc`,
`.cursor/permissions.json` and `.cursor/hooks.json`.

### VS Code / Copilot

```sh
evidencespine harness install --harness vscode-copilot --target-dir .   # portable .mcp.json
# legacy Local harness only:
evidencespine harness install --harness vscode --target-dir .
```

Prefer `vscode-copilot`; the `vscode` target writes the deprecated
`.vscode/mcp.json` `servers` shape. See the migration note in
[HARNESS_MATRIX.md](HARNESS_MATRIX.md).

### DeepSeek Harness

```sh
evidencespine harness install --harness dsh --target-dir /path/to/repo
CLI="/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh"
DSH_HOME="$HOME/.dsh" "$CLI" --dump-config --profile headless | grep -A12 evidencespine-mcp
```

Writes the machine-wide MCP client row with absolute paths and materializes the
skill. Details: [DSH.md](DSH.md).

### Generic MCP / anything else

```sh
evidencespine harness install --harness generic --target-dir .
# canonical: evidencespine.mcp.json; projections: .mcp.json, .cursor/mcp.json,
# .agents/mcp_config.json, .agents/.mcp.json, .continue/mcpServers/,
# .junie/mcp/mcp.json, .roo/mcp.json
```

Any client that accepts `{"mcpServers": {"evidencespine": {"command": ...,
"args": [...]}}}` can import the canonical file unchanged.

### Git hooks

```sh
evidencespine harness install --harness git --target-dir .
git commit -m "..."        # post-commit records the commit with grounded diff spans
evidencespine harness git test-record --status passed --command "pytest tests/"
```

## 7) Thread identity

One workspace shares one thread:

```text
explicit --thread-id > EVIDENCESPINE_THREAD_ID > proj_<sha1(realpath(git toplevel))[:12]>
```

Harness hooks, CLI, MCP and A2A resolve this server-side. Provider session ids
are stored as `source_turn_id`/`metadata.session_id`, never as thread ids.
Set `EVIDENCESPINE_THREAD_ID` only when you deliberately want isolation (for
example one thread per incident instead of per repo).

## 8) Agent workflow snippets

### Implementer -> auditor handoff (Python)

```python
# implementer side
runtime.ingest_event({
    "thread_id": "bugfix_auth_timeout", "event_type": "outcome",
    "source_agent_id": "implementer", "source_turn_id": "turn_9",
    "payload": {"claim": "retry timeout fixed in middleware",
                "fact_state": "verified",
                "verification": {"method": "test",
                                 "reference": "pytest tests/test_middleware.py"}},
})
packet = runtime.emit_handoff("auditor", "bugfix_auth_timeout",
                              scope="verify timeout edge cases")
print(packet.to_dict()["file_path"])

# auditor side (new session)
runtime.import_handoff(packet.to_dict(), source_agent_id="auditor")
brief = runtime.build_brief("bugfix_auth_timeout", "what must I validate first?")
```

### Golden loop (CLI)

```sh
# 1. orient
evidencespine brief --thread-id proj_... --query "current goal" --json

# 2. ground + record a decision/outcome
evidencespine ingest --thread-id proj_... --event-type outcome \
  --source-agent-id implementer --source-turn-id t1 \
  --claim "fix shipped" --fact-state verified --ground-ref "src/mod.py#L10-L20"

# 3. after editing cited files, re-verify
evidencespine drift-check --thread-id proj_... --source-root .          # preview
evidencespine drift-check --thread-id proj_... --source-root . --apply  # persist flags

# 4. hand off
evidencespine handoff --role auditor --thread-id proj_... --json
evidencespine handoff --import .evidencespine/handoffs/<packet>.json --thread-id proj_...
```

### Golden loop (MCP tool names)

1. `build_brief` / resource `evidencespine://brief/{thread}` at session start.
2. `ingest_event` for decisions/outcomes; `ground` then attach items for
   claims that matter, with `fact_state: "verified"`.
3. `check_drift` after edits (default `dry_run=true`; pass
   `dry_run=false` to write flags).
4. `verify_fact` when a test/gate/tool verified the claim.
5. `emit_handoff` / `import_handoff` at role boundaries.
6. `query_view` (`contradictions`, `open_gates`, `stale_claims`) when
   claims conflict or state matters.
7. `snapshot` to check health; `prune` previews by default.

## 9) Adapters (transcript-first integration)

```python
from evidencespine.adapters import TranscriptAdapter

adapter = TranscriptAdapter(runtime, default_thread_id="thread_default")
result = adapter.ingest_messages([
    {"role": "user", "content": "Check drift"},
    {"role": "assistant", "content": "Patch complete",
     "evidence_items": [{"source_id": "patch.diff", "line_start": 7, "line_end": 9}]},
    {"role": "tool", "content": "pytest passed"},
])
print(result["status"])   # ok | partial | failed | disabled | invalid
brief = adapter.brief("what matters now")
packet = adapter.handoff("auditor", "verify latest claims")
```

Ingest accounting is explicit: newly persisted events are `ingested`,
replayed ones are `deduped` (`new_events` is an alias for `ingested`),
and `partial`/`failed`/`disabled` are distinct statuses. Caller-provided
`evidence_items` and `state_context` survive normalization; the adapter
does not invent spans. Framework wrappers:
`LangGraphAdapter` (`ingest_state`, `normalize_state`) and
`AutoGenAdapter` (`ingest_messages`, `normalize_messages`). See
[ADAPTERS.md](ADAPTERS.md).

## 10) A2A, async runtime, retrieval

```sh
pip install "evidencespine[a2a]"
export EVIDENCESPINE_A2A_AUTH_TOKEN="$(openssl rand -hex 32)"
evidencespine a2a --host 127.0.0.1 --port 8765
curl http://127.0.0.1:8765/.well-known/agent-card.json
```

A2A operations: `ingest_event, append_fact, build_brief, query_view,
emit_handoff, import_handoff, reconcile, snapshot, prune` (no ground/drift/
verify). The agent card is public; every other path requires the bearer token
when configured, and non-loopback binds without a token are refused. A2A
`prune` deletes by default — pass `dry_run=true` explicitly.

```python
from evidencespine import AsyncAgentMemoryRuntime
rt = AsyncAgentMemoryRuntime(base_dir=".evidencespine")
await rt.ingest_event({...})
brief = await rt.build_brief("session_1", "what matters now")
```

Retrieval modes: `EVIDENCESPINE_RETRIEVAL_MODE=lexical|hybrid|vector`
(lexical default), `EVIDENCESPINE_EMBEDDING_BACKEND=auto|hashing|fastembed`.

## 11) HTTP + auth summary

- MCP: `EVIDENCESPINE_MCP_AUTH_TOKEN` -> static bearer on every HTTP request,
  constant-time compare, non-loopback refusal, no OAuth, no TLS termination.
- A2A: `EVIDENCESPINE_A2A_AUTH_TOKEN` -> same model, public agent card.
- Per-client header/env syntax differs; see the env-dialect table in
  [HARNESS_MATRIX.md](HARNESS_MATRIX.md).

## 12) Troubleshooting

| Symptom | Check / fix |
|---|---|
| Client sees no tools | `evidencespine harness doctor --target-dir . --json`; is the executable on the client's PATH/absolute? For Claude Code, launch from the repo root and accept workspace trust. For Codex, run `/hooks` trust review. |
| Hooks never fire | Hooks fail open silently; check the client's hook log/debug output and that the installed file is the one the client reads (see matrix). |
| Brief is empty or `stale: true` | Expected on a fresh store or when no events fall inside the 168 h lookback; rely on the repo and ingest state as you work. |
| "verified" claim shows as asserted | It was ungrounded: attach a checksummed excerpt (`--ground-ref` / `ground`) or verification provenance. The ingest result's `policy_downgrades` says how many. |
| Handoff import fails | Packet must validate (`role`, `thread_id`, `scope`, `claims`); path imports must be relative to `EVIDENCESPINE_SOURCE_ROOT`. Stale rows import as asserted. |
| Store in the wrong place | Pass absolute `--base-dir`; harness installers emit absolute paths where the client cannot resolve relative ones. |
| MCP server returns old behavior after upgrade | Long-lived servers keep start-time code: compare `snapshot.evidencespine_version` with `harness debug` and restart. |
| HTTP client gets 401 | Token not set/rotated, or header missing. Restart the server after changing `EVIDENCESPINE_MCP_AUTH_TOKEN`. |

## 13) Operational checklist

- Persist events with evidence refs/items for claims that matter.
- Mark fact state correctly; never claim verified without grounding.
- Build a brief before major actions and role changes.
- Query `active_scopes` / `open_gates` before switching roles.
- Emit a handoff when switching roles or ending a session.
- Drift-check after editing cited files; trust the flag over the claim.
- Watch `snapshot()` for citation coverage, stale evidence and fail-open
  counts.

## See also

- [HARNESS_MATRIX.md](HARNESS_MATRIX.md) — all clients, paths, shapes, grades
- [CLAUDE_CODE.md](CLAUDE_CODE.md), [CODEX.md](CODEX.md),
  [OPENCODE.md](OPENCODE.md), [DSH.md](DSH.md)
- [PROTOCOL.md](PROTOCOL.md), [ARCHITECTURE.md](ARCHITECTURE.md),
  [SECURITY.md](SECURITY.md), [ADAPTERS.md](ADAPTERS.md),
  [BENCHMARKS.md](BENCHMARKS.md)
- Canonical agent playbook: `AGENTS.md` / `evidencespine://guide`
