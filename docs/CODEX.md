# Codex support (status: guidance + manual MCP; harness install planned)

EvidenceSpine works with OpenAI Codex today through the standard channels;
a full `install_codex` harness wiring (config + lifecycle hooks) is planned
but not yet shipped. This page documents the verified integration surface so
users can wire Codex manually now and contributors can implement the
installer against an accurate spec.

## What works today

1. **AGENTS.md guidance** — Codex reads `AGENTS.md` at the project root
   (precedence: `AGENTS.override.md` then `AGENTS.md`, walked down from the
   git root; a global `~/.codex/AGENTS.md` is loaded first). The committed
   `AGENTS.md` therefore teaches Codex the same decision rules as opencode,
   Cursor, and Claude Code. Combined instructions are capped at 32 KiB
   (`project_doc_max_bytes`) — the guide is well under that.
2. **MCP tools and resources** — Codex (CLI, IDE extension, ChatGPT desktop
   app) connects to the EvidenceSpine MCP server and shares its configuration
   across surfaces. All tools and the `evidencespine://brief|view|state|snapshot`
   resources are available, plus the `evidencespine://guide` usage resource.

## Manual MCP registration

Project-scoped (requires a trusted project):

```toml
# .codex/config.toml
[mcp_servers.evidencespine]
command = "/path/to/evidencespine"
args = ["mcp", "--base-dir", ".evidencespine"]
```

Global (`~/.codex/config.toml`, same shape) applies to every repository.
Alternatively, register via the CLI:

```bash
codex mcp add evidencespine -- /path/to/evidencespine mcp --base-dir .evidencespine
codex mcp list
```

## Lifecycle hooks (available, not yet installed by a script)

Codex supports lifecycle hooks via `.codex/hooks.json` or `[hooks]` tables in
`config.toml`. Verified behavior that a future `install_codex` must respect:

- `SessionStart` — plain text on stdout is added as extra developer context
  (the auto-recall brief can be injected directly).
- `SessionEnd` — advisory; runs with a default 1s timeout (max 3s); `matcher`
  currently only matches reason `other`.
- `PreCompact` — plain text on stdout is **ignored**; must return the JSON
  envelope `{"hookSpecificOutput": {"hookEventName": "PreCompact",
  "additionalContext": "..."}}`.
- Non-managed hooks require per-hash **user trust review** via `/hooks`, and
  project-local hooks load only when the project's `.codex/` layer is trusted.
  Hook commands run with the session cwd; prefer resolving paths from the git
  root.

## Server `instructions` field

Codex reads the MCP `instructions` field returned during server initialization
as server-wide guidance alongside the tools. If the MCP SDK supports it, the
EvidenceSpine server should advertise the usage-guide primer there (first 512
characters self-contained); otherwise the `evidencespine://guide` resource
covers the same need.

## Planned work

- `evidencespine harness install --harness codex`: write project `.codex/`
  (default) or `~/.codex/` (`--scope global`) with `config.toml` MCP table,
  `hooks.json` (SessionStart/SessionEnd/PreCompact), and shaped hook scripts
  (plain text vs JSON envelope).
- `evidencespine harness codex session-start|session-stop|precompact` provider
  actions reusing the shared handlers.
- Tests: parseable config.toml, hooks.json shape, hook output envelopes.
