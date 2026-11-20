# Codex integration

EvidenceSpine works with **OpenAI Codex** across its surfaces (CLI, IDE
extension, ChatGPT desktop Codex app) -- they share `~/.codex/config.toml` and
the lifecycle-hook format. Two things are installed:

1. an MCP server entry so Codex gets the EvidenceSpine tools/resources and the
   server `instructions` primer;
2. lifecycle hooks for auto-recall at session start and auto-retain at session
   end / before compaction.

```sh
# project scope (default): <repo>/.codex/config.toml + <repo>/.codex/hooks.json
evidencespine harness install --harness codex --target-dir .

# user scope: ~/.codex/config.toml + ~/.codex/hooks.json
evidencespine harness install --harness codex --scope global --target-dir ~

# HTTP MCP instead of stdio
evidencespine harness install --harness codex --mcp-url http://127.0.0.1:8000/mcp

# preview
evidencespine harness install --harness codex --target-dir . --dry-run
```

Sources: `learn.chatgpt.com/codex/mcp.md`, `config-basic.md`,
`config-reference.md`, `hooks.md` and `guides/agents-md.md` (the docs
host moved from `developers.openai.com/codex` in 2026; the markdown twins are
authoritative).

## config.toml: MCP servers

Codex config precedence (highest first): CLI flags and `--config` overrides >
project `.codex/config.toml` files from the project root down to the cwd
(closest wins; **trusted projects only**) > the `--profile` file
(`$CODEX_HOME/<name>.config.toml`) > user `~/.codex/config.toml` > cloud
defaults > `/etc/codex/config.toml` > built-ins. An untrusted project makes
Codex skip its project `.codex/` layer entirely, including project hooks.

stdio (what the installer writes by default):

```toml
[mcp_servers.evidencespine]
command = "/absolute/path/to/evidencespine"
args = ["mcp", "--base-dir", ".evidencespine"]
# optional: env = { KEY = "VALUE" }, env_vars = ["SOMENAME"], cwd = ".",
# startup_timeout_sec = 10, tool_timeout_sec = 60, enabled = true
```

Streamable HTTP with auth (supported by Codex; also reachable through
`--mcp-url`):

```toml
[mcp_servers.evidencespine]
url = "http://127.0.0.1:8000/mcp"
bearer_token_env_var = "EVIDENCESPINE_MCP_AUTH_TOKEN"
# alternatives/additions: http_headers = { "x-org" = "acme" },
# env_http_headers = { "x-env" = "ENV_VAR" }, http_headers_helper = "<cmd>",
# auth = "oauth" | "chatgpt" (fallback after bearer/headers)
```

Codex interpolates no env strings in TOML; bearer auth is **field-based**
(`bearer_token_env_var`, `http_headers`, `env_http_headers`).

### [features] merge safety

`hooks` is a **stable, default-on** Codex feature (`codex features list`
shows a hooks row with stability stable and enabled true); declaring
`[features] hooks = true` is harmless but redundant. The dangerous part is
duplication: a **second `[features]` table is a TOML duplicate-key error that
invalidates the whole config** (Codex then reports no MCP servers or fails to
load bootstrap configuration). The installer therefore:

1. parses the existing file with `tomllib` and **refuses to write** when it is
   not valid TOML (leaves it untouched, reports a warning/backup);
2. merges `hooks = true` **inside** an existing `[features]` table (and
   flips `hooks = false` to true) instead of appending a second header;
3. re-parses the merged text and refuses the write if it would produce
   duplicate `[features]` or `[mcp_servers.evidencespine]` tables.

On Python < 3.11 there is no `tomllib`, so an existing non-empty
`config.toml` cannot be validated. The installer then fails safe: it leaves
the file untouched (with a backup) and reports `partial`. Fresh installs work
normally; to update an existing config on 3.10, run the installer on Python
3.11+ or merge the table manually.

## hooks.json

Locations: system `/etc/codex/hooks.json` (Windows
`%ProgramData%/OpenAI/Codex/hooks.json`), user `$CODEX_HOME/hooks.json`
(`~/.codex/hooks.json`), and trusted-project `<repo>/.codex/hooks.json`. An
inline `[hooks]` table in `config.toml` is equivalent; declaring both in one
layer loads both **and warns**, so use one format per layer. A standalone
`hooks.json` is discovered without a sibling `config.toml`.

Shape: event name -> array of matcher groups -> array of handlers.

```json
{
  "description": "EvidenceSpine delivery: auto-recall at session start, retain through pre-compaction, fast auto-retain at session end.",
  "hooks": {
    "SessionStart": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "evidencespine harness codex session-start --base-dir .evidencespine",
            "timeout": 30,
            "additionalContextLimit": 10000
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "matcher": "",
        "hooks": [
          { "type": "command", "command": "evidencespine harness codex session-stop --base-dir .evidencespine", "timeout": 3 }
        ]
      }
    ],
    "PreCompact": [
      {
        "matcher": "",
        "hooks": [
          { "type": "command", "command": "evidencespine harness codex precompact --base-dir .evidencespine", "timeout": 30 }
        ]
      }
    ]
  }
}
```

Handler fields: `type` (`command` or `mcp_tool`; `prompt`/`agent` are
parsed but skipped), `command`, `timeout` (seconds; default 600, but
`SessionEnd`/`Interrupt` default to 1 s and support at most 3 s),
`statusMessage`, `additionalContextLimit` (tokens, default 2500; 0 = no
limit), `commandWindows` / TOML `command_windows`, and `async`
(`SessionEnd` is always synchronous).

Events: `SessionStart, SessionEnd, SubagentStart, PreToolUse,
PermissionRequest, PostToolUse, PreModelRequest, PostModelRequest, PreCompact,
PostCompact, UserPromptSubmit, SubagentStop, Stop, Interrupt`. Matchers are
regexes; for `SessionStart` the matcher filters `source`
(`startup|resume|clear|compact|fork`), for `PreCompact`/`PostCompact` it
filters `trigger` (`manual|auto`), and for tool events it filters the tool
name. `UserPromptSubmit`, `Stop`, `Interrupt` and the model-request events
ignore matchers.

### Output contract (important)

| Event | Plain-text stdout | hookSpecificOutput.additionalContext |
|---|---|---|
| `SessionStart` | **added as developer context** | yes |
| `UserPromptSubmit` / `SubagentStart` | added as context | yes |
| `PreToolUse` / `PostToolUse` | ignored | yes |
| `PreCompact` / `PostCompact` | **ignored** | **not supported** (only common fields: continue, stopReason, systemMessage, suppressOutput) |
| `SessionEnd` | advisory; cannot steer | -- |

So the Codex install is deliberately shaped this way:

- `SessionStart` uses an **empty matcher**, which also matches the
  `compact` source. Codex runs matching `SessionStart` hooks after a root
  compaction, before the next model request (including mid-turn automatic
  compaction), so **post-compaction recall rides SessionStart**, not
  PreCompact.
- `PreCompact` exists for its **ingest side effect only**. It cannot inject
  context, and EvidenceSpine does not pretend otherwise (the old JSON
  envelope is gone; compatibility helpers return the plain note).
- `additionalContextLimit` is raised to 10000 tokens on SessionStart because
  the Codex default (2500) would truncate a long brief into a file preview.
  Hooks also cap generated context at 10,000 characters.

## AGENTS.md

Codex reads instructions in two layers:

- **Global**: `$CODEX_HOME/AGENTS.override.md` else `$CODEX_HOME/AGENTS.md`
  (first non-empty file only).
- **Project**: from the project root (normally the git root) **down to the
  cwd**; per directory, try `AGENTS.override.md`, then `AGENTS.md`, then
  `project_doc_fallback_filenames`; at most one file per directory. Files
  are concatenated root->cwd and stop once the combined size reaches
  `project_doc_max_bytes` (default 32 KiB, configurable).

The repository `AGENTS.md` therefore teaches Codex the EvidenceSpine decision
rules, and `--with-agents-md` / `--harness agents-md` merge that guide into
a managed block when a repo has none. The MCP server also returns the usage
guide in `initialize.instructions`; Codex shows it as server guidance.

## Installer behavior

- Project scope writes `<target>/.codex/config.toml` and
  `<target>/.codex/hooks.json`; global scope writes into
  `<target>/.codex`-style dirs (e.g. `~/.codex/`).
- Both files are **merge-preserving**: unrelated tables/servers/events are kept;
  EvidenceSpine hook groups are appended to existing event arrays (both run)
  and deduped by command/action so repeated installs do not double-deliver.
- An unparseable existing `config.toml` or `hooks.json` is never
  overwritten: the installer leaves it untouched, records a backup path and
  reports `status: "partial"`.
- The result includes `transport`, `warnings`, `wrote` and a `note`
  reminding you that project hooks need per-hash trust review.
- `--dry-run` reports planned writes without touching anything.

## Validate

```sh
codex mcp list                            # should list evidencespine ... enabled
codex features list | grep -E '^hooks '   # hooks row: stable, enabled
python3 -c "import tomllib; tomllib.load(open('.codex/config.toml','rb')); print('config.toml parses')"
python3 -m json.tool .codex/hooks.json >/dev/null && echo "hooks.json parses"
evidencespine harness doctor --target-dir . --json   # codex_config: valid TOML; codex_hooks: valid JSON
```

## Limits

- **Trust gates.** Project `.codex/` config and hooks load only for a
  trusted project (`[projects."/path"] trust_level = "trusted"` in the user
  config). Non-managed hooks additionally require a one-time **per-hash trust
  review** (`/hooks` in the CLI); new or changed hooks are skipped until
  approved. `--dangerously-bypass-hook-trust` skips persisted trust for one
  invocation. An installer cannot make project hooks fire.
- **No PreCompact injection.** Codex ignores PreCompact/PostCompact stdout and
  does not support `additionalContext` there; recall after compaction depends
  on the SessionStart `compact` source.
- **SessionEnd is fast.** 1 s default, 3 s max; the EvidenceSpine handler does
  one fast ingest (`EVIDENCESPINE_AUTO_HANDOFF=1` is opt-in and may not fit).
- **Windows command strings.** Codex's `hooks.json` handler takes a single
  command string (no exec form). The installer builds it with POSIX
  `shlex.join` and does not emit `commandWindows`, so native Windows
  PowerShell hook execution is untested -- verify on the target host.
- **Context truncation.** `additionalContextLimit` is measured in tokens;
  oversized context is saved to disk and replaced by a shortened preview. The
  installer raises it to 10000 and hooks cap text at 10,000 characters, but a
  very large brief can still be truncated.
- **No OAuth server.** Codex can do OAuth (`auth = "oauth"`, `codex mcp
  login`) but EvidenceSpine is a static-bearer server. Use
  `bearer_token_env_var` with a loopback/HTTP deployment behind TLS.
- A long-lived MCP server keeps serving its start-time code after an upgrade:
  compare `snapshot.evidencespine_version` with
  `evidencespine harness debug` and restart on mismatch.
