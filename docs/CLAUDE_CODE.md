# Claude Code integration

EvidenceSpine delivers to **Claude Code 2.1.x** (CLI, IDE extension, and the
Desktop Code tab) as a **skills-directory plugin** with lifecycle hooks, an MCP
server, a skill, and a rules file. Delivery is automatic: a bounded context
brief is injected at session start, working state is retained through
compaction, and session end is recorded. No voluntary tool call is required for
recall; the full MCP surface is there for deliberate queries.

## Install

```sh
evidencespine harness install --harness claude-code --target-dir . --scope project

# settings-only setups (double-run caveat below)
evidencespine harness install --harness claude-code --with-settings-hooks

# global plugin under the user's skills directory
evidencespine harness install --harness claude-code --scope global --target-dir ~

# preview
evidencespine harness install --harness claude-code --target-dir . --dry-run
```

Project-scope output (all merge-preserving, idempotent):

```text
<repo>/.claude/skills/evidencespine/
    .claude-plugin/plugin.json      manifest (only file inside .claude-plugin/)
    hooks/hooks.json                wrapped {"description", "hooks": {...}}
    .mcp.json                       plugin MCP registration (stdio)
    skills/evidencespine/SKILL.md   auto-surfacing skill
    README.md                       directory-profile documentation
<repo>/.claude/rules/evidencespine.md   standing policy (loads every session)
```

The installer runs `claude plugin validate --strict <plugin_dir>` when the
`claude` CLI is on PATH and reports `validation.status`
(`passed` / `failed` / `skipped`) plus a `next_step` line in its JSON
result.

## Why a skills-directory plugin

Claude Code does **not** scan a project's `.claude-plugin/` or
`.claude/plugins/` directories for plugins. The only project-local plugin
form that auto-loads with no install flag and no marketplace is a
**skills-directory plugin**:

```text
<project>/.claude/skills/<name>/...
```

It loads as `evidencespine@skills-dir`. Consequences:

- The first interactive session shows a **workspace-trust** dialog; the plugin
  is scanned after trust is accepted. `-p` / SDK / cloud sessions treat the
  folder as trusted and load without asking.
- **Launch from the repository root.** Parent directories are not searched for
  plugins, and a plugin placed anywhere else in the tree (including a
  repo-root `.claude-plugin/plugin.json`) is inert unless loaded with
  `--plugin-dir`, a marketplace install, or
  `CLAUDE_CODE_PLUGIN_DIRS`.
- Admins can block skills-dir plugins; use
  `claude --plugin-dir <dir> plugin list` to confirm a recognized layout.
- `/reload-plugins` rescans after edits.

## Hook wiring

`hooks/hooks.json` must be wrapped under a top-level `hooks` key (the bare
event map fails `claude plugin validate` with
`hooks.json must have "hooks"`). The installer writes:

```json
{
  "description": "EvidenceSpine delivery: auto-recall at session start, retain through pre-compaction, fast auto-retain at session end.",
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume|clear|compact|fork",
        "hooks": [
          {
            "type": "command",
            "command": "<absolute-path-to-evidencespine>",
            "args": ["harness", "claude-code", "session-start", "--base-dir", "${CLAUDE_PROJECT_DIR:-.}/.evidencespine"],
            "timeout": 30
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "<absolute-path-to-evidencespine>",
            "args": ["harness", "claude-code", "session-stop", "--base-dir", "${CLAUDE_PROJECT_DIR:-.}/.evidencespine"],
            "timeout": 5
          }
        ]
      }
    ],
    "PreCompact": [
      {
        "matcher": "manual|auto",
        "hooks": [
          {
            "type": "command",
            "command": "<absolute-path-to-evidencespine>",
            "args": ["harness", "claude-code", "precompact", "--base-dir", "${CLAUDE_PROJECT_DIR:-.}/.evidencespine"],
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

Rules that matter:

- **Exec form** (`command` + `args`) avoids shell quoting entirely and is
  the documented form when a path placeholder is used. Shell-form
  (`command` only) runs through `sh` / Git Bash / PowerShell and is a
  portability trap, especially on Windows.
- `${CLAUDE_PROJECT_DIR}` is set for hook commands and stdio MCP servers;
  project `.mcp.json` must write `${CLAUDE_PROJECT_DIR:-.}` because an
  unset variable without a default is used literally. Plugin-provided MCP
  substitutes `${CLAUDE_PROJECT_DIR}` directly.
- `timeout` is a handler field, in seconds.
- Hooks **fail open**: a missing store or a broken binary returns a one-line
  notice instead of failing the session. Injected text is capped at 10,000
  characters (`CONTEXT_CHAR_LIMIT`); the brief budget (1800 tokens by
  default) stays well under it.

### Plugin hooks vs settings hooks (do not run both)

Claude Code merges hook sources and **does not de-duplicate** a plugin handler
against an identical settings handler: if the same handler exists in both
places, it runs twice. The project install therefore writes plugin hooks only.
`--with-settings-hooks` appends the same groups to `.claude/settings.json`
as an explicit opt-in for settings-only setups; do not combine it with a plugin
that also loads.

**Old-install cleanup:** earlier EvidenceSpine versions wrote a repo-root
plugin (`<repo>/.claude-plugin/plugin.json` plus `<repo>/hooks/hooks.json`)
and merged hooks into `.claude/settings.json`. That layout never auto-loaded,
but it now duplicates delivery if you also run `--plugin-dir`. After
upgrading, delete the stale repo-root `.claude-plugin/` and `hooks/`
folders and remove the EvidenceSpine hook groups from
`.claude/settings.json` (keep user hooks; back the file up first). Keep
`.claude/rules/evidencespine.md`; it is the intended policy surface.

## Output channels (event semantics)

| Event | Channel | What EvidenceSpine does |
|---|---|---|
| `SessionStart` | plain-text stdout is added to Claude's context | prints the bounded brief (plus a one-line operating preamble) for the workspace thread |
| `SessionEnd` | side effects only; no decision control | records one fast reflection event; the stdin `reason` (clear/resume/logout/prompt_input_exit/other) is stored as metadata, never as summary text |
| `PreCompact` | **plain-text stdout is the documented channel**: with exit 0 and no block, Claude uses it as instructions for the compaction summary | persists the custom instructions/summary, then prints the working-state note. `hookSpecificOutput.additionalContext` is **not** a supported PreCompact output and is not emitted (the old envelope was wrong) |
| `PostCompact` (not installed) | stdout discarded | the next `SessionStart` with source `compact` re-anchors recall |

`SessionEnd` has a **~1.5 s default terminate budget**. A plugin hook's
`timeout` does not raise that budget (only settings-file timeouts or
`CLAUDE_CODE_SESSIONEND_HOOKS_TIMEOUT_MS` do), which is why the handler does
one fast ingest and nothing slow. Set
`CLAUDE_CODE_SESSIONEND_HOOKS_TIMEOUT_MS` if you add heavy user hooks.

The provider control fields (`trigger`, `reason`, `source`) are never
stored as the session outcome; only genuine summary content
(`compact_summary`, `custom_instructions`, `conversation_summary`,
`summary`, `note`) is.

## Thread identity

Claude Code sends a fresh `session_id` per session. Using it as the thread id
would create one empty thread per session, so the CLI resolves a
**workspace-scoped** thread:

`text
explicit --thread-id > EVIDENCESPINE_THREAD_ID > proj_<sha1(realpath(git toplevel))[:12]>
`

Session and turn identity travel in `source_turn_id` / `metadata.session_id`.
If you want a deliberate isolation per conversation, set
`EVIDENCESPINE_THREAD_ID` in the environment Claude Code launches with.

## MCP registration

The plugin ships its own `.mcp.json` (stdio, command + args) and registers as
`plugin:evidencespine:evidencespine`; tools are named
`mcp__plugin_evidencespine_evidencespine__<tool>`. A project-scope
`.mcp.json` is also supported by Claude Code directly (one approval prompt):

```json
{
  "mcpServers": {
    "evidencespine": {
      "command": "evidencespine",
      "args": ["mcp", "--base-dir", "${CLAUDE_PROJECT_DIR:-.}/.evidencespine"]
    }
  }
}
```

CLI equivalents:

```sh
claude mcp add --scope user    --transport stdio evidencespine -- evidencespine mcp --base-dir "$PWD/.evidencespine"
claude mcp add --scope project --transport stdio evidencespine -- evidencespine mcp --base-dir .evidencespine
claude mcp list
```

HTTP transport uses `type: "http"` (`streamable-http` is an accepted
alias) with `url` and `headers`:

```json
{
  "mcpServers": {
    "evidencespine": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp",
      "headers": { "Authorization": "Bearer ${EVIDENCESPINE_MCP_AUTH_TOKEN}" }
    }
  }
}
```

## Instructions: rules, skill, AGENTS.md

- `.claude/rules/evidencespine.md` loads every session alongside CLAUDE.md
  (same priority as `.claude/CLAUDE.md`) and carries the standing policy:
  consult the spine before asserting state, ground before claiming verified,
  drift-check after edits, import/emit handoffs. It does not touch the user's
  CLAUDE.md or its recommended 200-line budget.
- The plugin skill `skills/evidencespine/SKILL.md` auto-surfaces when the
  agent is about to assert state, mark something verified, or hand off. It is
  invoked as `/evidencespine:evidencespine` and by description matching.
- `AGENTS.md` is only read by Claude Code when there is **no**
  CLAUDE.md/`.claude/CLAUDE.md`/`CLAUDE.local.md` in the working directory
  or above. The committed AGENTS.md guide is therefore a secondary channel
  here; `--with-agents-md` exists for repos that have no CLAUDE.md, the other
  harnesses, and Codex/DSH. A `CLAUDE.md` at a plugin root is **never**
  loaded (write instructions as a skill instead).

## Validate and verify

```sh
# installer already runs this; run it yourself after edits
claude plugin validate --strict .claude/skills/evidencespine
echo $?          # 0 = passed, 1 = failed, 2 = validator crash

claude plugin list
claude --plugin-dir .claude/skills/evidencespine plugin details evidencespine
# expect: Hooks (3) SessionStart, SessionEnd, PreCompact; MCP servers (1); Skills (1)

claude mcp list
claude --debug-file /tmp/claude-debug.log      # hook execution log
```

Directory submissions additionally need a README (the generated plugin ships
one); the `--profile directory` validator mode can still report
`ICON_MISSING` as a non-blocking warning.

## Limits and troubleshooting

| Symptom | Cause / fix |
|---|---|
| `claude plugin list` shows nothing | Not launched from the repo root, workspace trust not accepted, or an admin blocked skills-dir plugins. Launch from the root and accept the dialog; use `--plugin-dir` to test. |
| Hooks appear to do nothing | Check workspace trust and `claude --debug-file`. Hooks fail open, so a broken side-car is silent except for the fail-open notice line. |
| Brief injected twice | Plugin hooks and settings hooks both active (often from an old install). Remove the settings copy or the plugin; keep one. |
| Session end is slow or cuts off | `SessionEnd` has a ~1.5 s budget; keep handlers fast, or raise `CLAUDE_CODE_SESSIONEND_HOOKS_TIMEOUT_MS`. |
| PreCompact state missing | Confirm the hook fired (`/hooks` or debug log) and that PreCompact is installed (`--with-precompact` is off only if you customized the installer). |
| Different store than expected | A relative `--base-dir` resolves against `${CLAUDE_PROJECT_DIR}` in hooks but per-CWD in a project `.mcp.json` without a default; use the emitted placeholder. |
| MCP tools missing | Approve the project `.mcp.json` prompt, and restart after upgrading: a long-lived MCP server keeps serving its start-time code. Compare `snapshot.evidencespine_version` with `evidencespine harness debug` and restart on mismatch. |

## Reference

- Docs: [HARNESS_MATRIX.md](HARNESS_MATRIX.md), [INTEGRATION.md](INTEGRATION.md),
  [PROTOCOL.md](PROTOCOL.md), [SECURITY.md](SECURITY.md)
- Example: `examples/claude_code_usage.py`
- Canonical agent playbook: the repository `AGENTS.md` /
  `evidencespine://guide` resource.
