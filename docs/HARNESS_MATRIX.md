# Client Compatibility Matrix

EvidenceSpine 0.6.0 delivers the same side-car to the major MCP-capable coding
harnesses; the evidence grades below distinguish verified installs from
documented or uncertain configurations. There is no single common config
file: the portable substrate is
**stdio plus the `{"mcpServers": {...}}` object**, and everything else
(paths, key names, env-var dialect, hook systems) is per client. This page is
the complete map: what the installer writes for each registered target, which
clients are print-only, and how to verify the result.

Evidence grades used below (same convention as the research notes this matrix
was built from, fetched 2026-11-20):

- **[C] confirmed**: vendor docs fetched live, or reproduced on this machine /
  covered by a test.
- **[I] inferred**: logically derived (e.g. XDG path convention), not exercised.
- **[U] uncertain**: UI-managed, undocumented, or installation-dependent;
  never trust an [U] path without checking the client.

Status legend: **installed** = `evidencespine harness install --harness <name>`
writes and merge-preserves the files; **print-only** =
`evidencespine harness print-config --harness <name>` prints a copy-paste
snippet and exact path but never writes; **UI-managed** = the client has no
portable file we write.

## Server command (all clients)

Every stdio entry ultimately spawns:

```sh
evidencespine mcp --base-dir .evidencespine
```

For a remote/HTTP surface, start the server yourself and give clients the URL
plus an Authorization header:

```sh
export EVIDENCESPINE_MCP_AUTH_TOKEN="$(openssl rand -hex 32)"
evidencespine mcp --transport streamable-http --host 127.0.0.1 --port 8000 --path /mcp
```

The server refuses an unauthenticated non-loopback bind unless
`EVIDENCESPINE_MCP_ALLOW_UNAUTHENTICATED=1` is set. It is a static bearer
guard (`hmac.compare_digest`), not an OAuth authorization server; clients that
require OAuth must be fronted by a gateway. See [SECURITY.md](SECURITY.md).

## Registered install targets

These are the exact names accepted by `harness install --harness` (the
registry is `evidencespine.harness.install.KNOWN_HARNESSES`).
`all` is an alias that runs every installed target and is not itself a client.

| Target | Surface | What install writes (scope) | Shape / keys | stdio | HTTP + auth syntax | Instruction channel | Status | Verify |
|---|---|---|---|---|---|---|---|---|
| `claude-code` | Claude Code CLI 2.1.x, IDE extension, Desktop Code tab | project: `.claude/skills/evidencespine/` (skills-directory plugin: `.claude-plugin/plugin.json`, `hooks/hooks.json`, `.mcp.json`, `skills/evidencespine/SKILL.md`, `README.md`) + `.claude/rules/evidencespine.md`; global: same plugin under `<target>/.claude/skills/` | `.mcp.json` -> `mcpServers`; hook argv uses `${CLAUDE_PROJECT_DIR:-.}` | yes | `{"type":"http","url":".../mcp","headers":{"Authorization":"Bearer ${EVIDENCESPINE_MCP_AUTH_TOKEN}"}}` [C] | skill + `.claude/rules/` (+ `AGENTS.md` only when no CLAUDE.md exists; opt-in block) | installed | `claude plugin validate --strict .claude/skills/evidencespine`, `claude mcp list` |
| `opencode` | opencode CLI/desktop/web v2.3.x | project: `.opencode/plugins/evidencespine.ts`; global: plugins dir passed with `--target-dir` | V2 plugin (bare default export `{id, setup(ctx)}`); registers MCP through `ctx.mcp.transform` | yes (plugin registers a `type:"local"` argv command) | HTTP is not wired by the plugin; use print-config for `opencode.json` `mcp.servers` remote entries | `AGENTS.md` discovery + optional `.agents/skills` | installed | `opencode plugin list`; grep `loading plugin` in opencode.log; `opencode mcp list` |
| `cursor` | Cursor IDE / CLI | project: `.cursor/mcp.json`, `.cursor/rules/evidencespine.mdc`, `.cursor/permissions.json`, `.cursor/hooks.json`; global: pass `--target-dir ~/.cursor` | `mcpServers` (merge-preserving); hooks `{"version":1,"hooks":{...}}` | yes | `{"url":".../mcp","headers":{"Authorization":"Bearer ${env:EVIDENCESPINE_MCP_AUTH_TOKEN}"}}` -- env dialect `${env:NAME}` [C] | `.cursor/rules/*.mdc` (alwaysApply) + `AGENTS.md` | installed | Cursor Settings > MCP, or `cursor mcp list` |
| `vscode` | VS Code **legacy Local-harness** editor-managed MCP (deprecated) | project: `.vscode/mcp.json`; global: pass a custom dir | top-level `servers` key (+ optional `inputs`, `sandbox`) | yes | `url` + `headers` (editor-managed) [C] | `AGENTS.md` / `.github/copilot-instructions.md` | installed (deprecated -- prefer `vscode-copilot`) | Command Palette: `MCP: List Servers` |
| `vscode-copilot` | VS Code + GitHub Copilot harness (native portable) | project: `.mcp.json`; global: `<target>/.copilot/mcp-config.json` (default `~/.copilot/mcp-config.json` via `$COPILOT_HOME`) | `mcpServers`; entry adds `type:"stdio"` + `tools:["*"]`; HTTP `type:"http"` | yes | `{"type":"http","url":".../mcp","headers":{"Authorization":"Bearer ${NAME}"}}` -- braced env form required [C] | `AGENTS.md` / `.github/copilot-instructions.md` | installed | `MCP: List Servers` from the Command Palette |
| `codex` | Codex CLI 0.169.x, IDE extension, ChatGPT/Codex desktop | project: `.codex/config.toml` + `.codex/hooks.json`; global (`--scope global`): `~/.codex/config.toml` + `~/.codex/hooks.json` | `[mcp_servers.evidencespine]` `command`/`args`; HTTP `url`+`bearer_token_env_var`/`http_headers`; hooks `{"hooks":{Event:[{matcher,hooks:[...]}]}}` | yes | `url = "..."`, `bearer_token_env_var = "EVIDENCESPINE_MCP_AUTH_TOKEN"` (also `http_headers`, `env_http_headers`) [C] | `AGENTS.md` (global `$CODEX_HOME/AGENTS.md`, then project root->cwd, 32 KiB cap) | installed (project hooks need per-hash trust review in `/hooks`) | `codex mcp list`; `python3 -c "import tomllib;tomllib.load(open('.codex/config.toml','rb'))"` |
| `git` | repository git hooks | `.git/hooks/post-commit` + `post-merge` (existing hooks chained; backed up once to `<name>.pre-evidencespine`) | shell scripts with the `evidencespine-git-hook` sentinel; invocation inserted before a trailing `exit` | n/a (no MCP) | n/a | n/a | installed | `git commit` then `evidencespine harness debug --json` (`git_post_commit.present`) |
| `agents-md` | any client that reads AGENTS.md | project: managed block inside `AGENTS.md` (`<!-- evidencespine:begin -->` ... `<!-- evidencespine:end -->`); never clobbers user text | instruction file only | n/a | n/a | `AGENTS.md` | installed | `grep -n "evidencespine:begin" AGENTS.md`; `harness doctor` reports `agents_md` |
| `dsh` | DeepSeek Harness 0.2.2-rc.5 (Electron app, `dsh web/headless`) | machine-wide `$DSH_HOME/cordis.patch.yml` (default `~/.dsh/cordis.patch.yml`), sentinel managed block; optional skill at `.dsh/skills/evidencespine/SKILL.md`; backups + mode 0600 | DSH patch YAML: `- insert: - id: evidencespine-mcp / name: '@deepseek-ai/dsh-mcp-client' / config: {serverName, transport, command, args, cwd, failOnStartupError, startupTimeoutMs}` | yes (`transport: stdio`) | `transport: streamable-http` + `url` + `headers` (static header; env resolution needs the DSH Loader `!!js` tag -- see [DSH.md](DSH.md)) [C] | MCP `initialize.instructions` is injected automatically as `mcp:evidencespine`; optional `AGENTS.md` block / skill | installed | `dsh --dump-config --profile headless | grep -A12 evidencespine-mcp` (and `--profile web`) |
| `generic` | any MCP client (lowest common denominator) | project: canonical `evidencespine.mcp.json` + merge-preserving projections + `evidencespine.mcp.md` notes; plus `.agents/skills/evidencespine/SKILL.md` | `mcpServers` + `command`/`args` (explicit empty `env`) | yes | HTTP snippets only via `print-config --transport http`; the installer writes stdio | `AGENTS.md` block opt-in (`--with-agents-md`) / skill | installed | `python3 -m json.tool evidencespine.mcp.json`; `evidencespine harness debug --json` |
| `instructions` | any instruction-file consumer | project: managed `AGENTS.md` block (same merge as `agents-md`, backed by `harness.skills.install_instructions`) | instruction file only | n/a | n/a | `AGENTS.md` | installed | `grep -n "evidencespine:begin" AGENTS.md` |

Notes:

- Claude Code plugin hooks and settings hooks are **not** de-duplicated by
  Claude Code, so the installer writes plugin hooks only; `--with-settings-hooks`
  is an explicit opt-in for settings-only setups (see [CLAUDE_CODE.md](CLAUDE_CODE.md)).
- Codex project config/hooks only load for a **trusted** project; non-managed
  hooks require a one-time per-hash trust review (`/hooks`).
- The DSH home patch applies to **every** profile; `--scope project` and
  `--scope global` both target that home patch because DSH never auto-discovers
  repo MCP files.

## Generic projection table

`evidencespine harness install --harness generic` writes the canonical file
plus these project-scope projections (each merge-preserving; unrelated servers
and top-level keys are kept verbatim):

| File | Content | Consumed by |
|---|---|---|
| `evidencespine.mcp.json` | canonical `mcpServers` object | humans, CI, any MCP client |
| `evidencespine.mcp.md` | at most 40-line generated setup note (safe to delete) | humans |
| `.mcp.json` | canonical `mcpServers` | Claude Code project scope, VS Code Copilot portable, Warp |
| `.cursor/mcp.json` | canonical `mcpServers` | Cursor |
| `.agents/mcp_config.json` | canonical `mcpServers` | Antigravity workspace |
| `.agents/.mcp.json` | canonical `mcpServers` | Warp (other agents) |
| `.continue/mcpServers/evidencespine.json` | canonical `mcpServers` JSON | Continue auto-import |
| `.junie/mcp/mcp.json` | canonical `mcpServers` | Junie project scope |
| `.roo/mcp.json` | canonical `mcpServers` | Roo Code project scope |
| `.agents/skills/evidencespine/SKILL.md` | skill (symlink to `skills/evidencespine/SKILL.md`) | Cursor, Cline, generic skill loaders |

Windsurf/Devin is deliberately **not** written: its MCP config is user-global
(`~/.config/devin/mcp_config.json`), so use a print-config snippet. VS Code
legacy `.vscode/mcp.json` is only written by the deprecated
`--harness vscode` target.

## Print-only and UI-managed clients

These clients have no `--harness` target (nothing is written); print the
snippet and paste/merge it yourself with
`evidencespine harness print-config --harness <client>`.

| Client | Project path | User/global path | Shape / keys | stdio | HTTP + auth | Grade | Verify |
|---|---|---|---|---|---|---|---|
| Windsurf / Devin Desktop (`windsurf`) | -- (no documented project file) | `~/.config/devin/mcp_config.json`; legacy discovery `~/.codeium/windsurf/mcp_config.json` | `mcpServers`; `serverUrl` (or `url`) + `headers` for HTTP | yes | `headers` Bearer; Cascade OAuth available [C] | [C] | Cascade MCP panel (reload window) |
| Zed (`zed`) | `.zed/settings.json` | `~/.config/zed/settings.json` | `context_servers` inside settings.json | yes | `url` + `headers` Authorization; Zed otherwise prompts OAuth [C] path [I] | [C]/[I] | Agent panel settings (`zed: open settings`) |
| Gemini CLI (`gemini`) | `.gemini/settings.json` | `~/.gemini/settings.json` | `mcpServers` | yes | `httpUrl` + `headers` Bearer (`url` = legacy SSE) [C] | [C] | `gemini mcp list` |
| Antigravity (`antigravity`) | `.agents/mcp_config.json` | `~/.gemini/config/mcp_config.json` | `mcpServers` | yes | `serverUrl` + `headers`; SSE/`httpUrl` unsupported [C] | [C] | Antigravity MCP panel |
| Cline (`cline`) | -- | `~/.cline/data/settings/cline_mcp_settings.json` (`CLINE_DIR` / `CLINE_MCP_SETTINGS_PATH` overrides) | `mcpServers`; `disabled`, `autoApprove` | yes | `type:"streamableHttp"` + `url` + `headers` [C] | [C] | `cline mcp list` |
| Roo Code (`roo`) | `.roo/mcp.json` | VS Code globalStorage `mcp_settings.json` (path installation-dependent) | `mcpServers`; `cwd` optional | yes | `type:"streamable-http"` + `url` + `headers` [C] project / [U] global | [C]/[U] | Roo MCP Servers tab |
| Continue (`continue`) | `.continue/mcpServers/*.yaml` (JSON dropped there is auto-imported) | `~/.continue/config.yaml` | `mcpServers:` list of `{name,type,command,args,env,cwd}` | yes | `type:"streamable-http"` + `url`; header passthrough via `requestOptions` [C] | [C] | Continue MCP tools panel |
| Junie (`junie`) | `.junie/mcp/mcp.json` | `~/.junie/mcp/mcp.json` | `mcpServers` | yes | `url` + `headers` Bearer [C] | [C] | `junie mcp list` |
| Warp (`warp`) | `.warp/.mcp.json` | `~/.warp/.mcp.json` | `mcpServers` | yes | Streamable HTTP + SSE + headers [C] | [C] | Warp MCP settings / `/mcp` |
| Amp (`amp`) | `.amp/settings.json` | `~/.config/amp/settings.json` | JSON key literally `amp.mcpServers` | yes | `url` + `headers` Bearer [C] | [C] | `amp mcp list` |
| JetBrains AI Assistant (`jetbrains`) | -- | UI-managed Settings only | paste `mcpServers` JSON | yes | `url` snippet in UI [U] | [U] | Settings > Tools > AI Assistant > MCP |
| Claude Desktop (`claude-desktop`) | -- | macOS `~/Library/Application Support/Claude/claude_desktop_config.json`; Windows `%APPDATA%/Claude/claude_desktop_config.json`; Linux `~/.config/Claude/` | `mcpServers` | yes | **no remote entries in the file** -- use Settings > Connectors UI [C]/[I] | [C] | restart app; Settings > Connectors |

## Commands

```sh
# write the delivery layer for one registered target (merge-preserving, idempotent)
evidencespine harness install --harness cursor --target-dir . --scope project
evidencespine harness install --harness vscode-copilot --target-dir .
evidencespine harness install --harness codex --target-dir . --scope project
evidencespine harness install --harness codex --mcp-url http://127.0.0.1:8000/mcp   # HTTP MCP (Codex)
evidencespine harness install --harness claude-code --with-settings-hooks          # opt-in only
evidencespine harness install --harness dsh --dry-run                             # preview, no writes
evidencespine harness install --harness all                                       # every installed target

# print a snippet + target path; NEVER writes
evidencespine harness print-config --harness windsurf
evidencespine harness print-config --harness zed --scope global
evidencespine harness print-config --harness codex --transport http
evidencespine harness print-config --harness all --json

# diagnostics
evidencespine harness debug   --base-dir .evidencespine --json
evidencespine harness doctor  --target-dir . --json    # nonzero when a check is "broken"
```

`print-config --json` returns `{"status","client","scope","transport","path",
"snippet","verify","note","grade","writes":false}`; `--harness all` returns
`{"status":"ok","clients":[...],"writes":false}`. `format` is `json`,
`toml` or `yaml`, and per-format convenience keys (`toml`, `yaml`) are
additive. Client names follow the install registry: `--harness vscode` prints
the legacy `.vscode/mcp.json` `servers` shape (matching `install --harness
vscode`), `--harness vscode-copilot` prints the portable `.mcp.json`
`mcpServers` shape, and `--harness vscode-legacy` is an alias of `vscode`.

### Environment-variable dialects

The token placeholder in printed HTTP snippets is a literal
`Bearer ${EVIDENCESPINE_MCP_AUTH_TOKEN}` by default; substitution is not
uniform across clients:

| Client | Env syntax | Notes |
|---|---|---|
| Claude Code | `${VAR}` / `${VAR:-default}` | expanded in command/args/env/url/headers |
| VS Code Copilot portable | `${NAME}` / `${NAME:-default}` | braced form required; bare `$NAME` is literal |
| Cursor | `${env:NAME}`, `${userHome}`, `${workspaceFolder}` | interpolation in command/args/env/url/headers |
| Gemini CLI | `$VAR` / `${VAR}` in env values | process-env expansion |
| opencode | `{env:NAME}` | no shell-style dollar expansion in JSON strings |
| Amp | `${VAR_NAME}` | in config values |
| Codex | field-based: `bearer_token_env_var`, `http_headers`, `env_http_headers` | no string interpolation |
| DSH | Loader `!!js` expressions | deliberately not plain interpolation |
| Windsurf/Devin, Antigravity | not captured in fetched docs | [U] -- prefer stdio |

## Per-client verification commands

```sh
claude mcp list                          # claude-code
claude plugin validate --strict .claude/skills/evidencespine
codex mcp list                           # codex
opencode mcp list                        # opencode (plugin-registered server)
cursor mcp list                          # cursor (or Settings > MCP)
cline mcp list; cline config mcp --json  # cline
junie mcp list                           # junie
devin mcp list                           # windsurf/devin
gemini mcp list                          # gemini
amp mcp list                             # amp
dsh --dump-config --profile headless     # dsh (grep evidencespine-mcp)
python3 -m json.tool .cursor/mcp.json    # any JSON config
python3 -c "import tomllib;tomllib.load(open('.codex/config.toml','rb'))"
evidencespine harness doctor --target-dir . --json
```

A framework-agnostic stdio smoke test:

```sh
npx -y @modelcontextprotocol/inspector evidencespine mcp --base-dir .evidencespine
```

An HTTP smoke test (requires the server and token from the section above):

```sh
curl -s -H "Authorization: Bearer $EVIDENCESPINE_MCP_AUTH_TOKEN" -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2026-07-28","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}' http://127.0.0.1:8000/mcp
```

## VS Code: legacy vs Copilot (migration note)

VS Code changed the MCP file shape in 2026:

- **Legacy Local-harness / editor-managed (deprecated)**: `.vscode/mcp.json`
  with top-level `servers` (plus optional `inputs`, `sandbox`). Only the
  Local harness reads it; current VS Code docs mark it deprecated for new
  configs.
- **Copilot native (current)**: workspace `.mcp.json` (or
  `$COPILOT_HOME/mcp-config.json`, default `~/.copilot/mcp-config.json`)
  with top-level `mcpServers`; entries carry `type: "stdio" | "http"`, and
  env references use `${NAME}`.

`evidencespine harness install --harness vscode` still writes the legacy file
for compatibility (and reports `legacy_deprecated_vscode_local_harness`);
`--harness vscode-copilot` writes the portable `.mcp.json` shape. Prefer the
latter; if you have an old `.vscode/mcp.json` and a new Copilot harness, keep
only one source of truth.

## Uncertainty and caveats

- Roo Code's user-global `mcp_settings.json` lives in VS Code globalStorage
  and its exact path depends on the installation: configure global Roo servers
  through the UI ([U]).
- JetBrains AI Assistant's config is UI-managed; no portable file is
  documented ([U]).
- Zed's user settings path is the XDG convention, not quoted verbatim by the
  vendor docs ([I]).
- Roo/Devin/Antigravity env-interpolation token syntax was not captured ([U]).
- Not installed on the research machine, so untested end-to-end: Cursor, VS
  Code, Windsurf/Devin, Zed, Cline, Roo, Continue, Amp, Junie, JetBrains.
  Version-specific breakage must be validated on a machine that has them.
- `all` runs the installed targets plus `dsh`/`generic` when those modules
  are present; a missing optional module is reported as `{"status":"skipped",
  "reason":"not_installed"}` rather than failing the install.
