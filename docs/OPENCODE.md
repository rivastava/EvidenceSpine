# opencode (v2 plugin)

EvidenceSpine integrates with opencode through its **V2 plugin API**. The
installer writes one TypeScript file:

```sh
evidencespine harness install --harness opencode --target-dir .
# project: <repo>/.opencode/plugins/evidencespine.ts
# global:  --scope global --target-dir ~/.config/opencode/plugins
```

The plugin is a bare file at `.opencode/plugins/evidencespine.ts` (no
package.json, no build step). opencode discovers direct `.ts` and `.js`
files there on startup.

## Supported versions

- **opencode >= 2.x**, verified against **opencode v2.3.0** with the
  `@opencode/plugin` 2.3.2 type definitions.
- **opencode v1 is not supported.** A V2 plugin exports a bare
  `{id, setup}` object; the v1 loader required a `server()` method, so a v1
  binary registers nothing. If `opencode --version` reports 1.x (for example
  a stale Homebrew formula shadowing the newer install), uninstall or reorder
  PATH before installing. The plugin file itself loads cleanly under v2.3.0
  (verified in `opencode.log`: `msg="loading plugin"`, no LoadError).

## V2 plugin contract

A V2 plugin is a module whose default export has a stable `id` and a
`setup(ctx)` function; `setup` may return a cleanup function (it replaces
v1 `dispose`). The typed identity helper `Plugin.define({...})` is optional:
it is an identity function for type-checking, not a runtime requirement.

```ts
import type { Plugin } from "@opencode/plugin"

export default {
  id: "evidencespine",
  async setup(ctx: Plugin.Context) {
    // register hooks/resources here
  },
} satisfies Plugin.Plugin
```

The type-only import is erased at runtime. This is deliberate: a **value**
import of `@opencode/plugin` would require the package to be resolvable from
a bare plugin file (there is no `package.json` in `.opencode/plugins/`),
which is not guaranteed. Keep `import type` and the zero-resolution loading
that is verified working.

The surfaces EvidenceSpine uses (all present in
`@opencode/plugin` 2.3.x):

| Surface | Purpose in the plugin |
|---|---|
| `ctx.mcp.transform((editor) => ...)` | registers the EvidenceSpine MCP server (`type:"local"`, argv command) only when absent |
| `ctx.session.hook("context", ...)` | auto-recall: appends the bounded brief to every assembled agent-loop model context |
| `ctx.session.hook("compaction", ...)` | auto-recall for checkpoint summaries — **a separate hook from `context`** |
| `ctx.tool.hook("execute.after", ...)` | auto-capture: records completed `edit`/`write` tool calls |
| `ctx.event.subscribe({signal})` | lifecycle: warm recall on `session.created`, stop on `session.deleted`, drop the cached brief on `session.compaction.ended` |

`context` runs only for the agent loop (prompts and tool-driven
continuations); `compaction` runs for checkpoint summaries. Registering only
`context` silently loses recall exactly when a session is compacted, so the
installer registers both. The event names are v2 names: `session.compacted`
is **not** a v2 event (the real ones are `session.compaction.started /
.delta / .ended / .failed`), and `session.idle` is a per-turn busy->idle
transition, not a session end — using it as session end would ingest one
"session end" per turn.

## Installer template (what is generated)

The installed file is the template in
`src/evidencespine/harness/opencode.py` rendered with three substitutions:
the base dir string, the executable argv array, and the MCP command argv
array. Behavior:

- **MCP registration** — `editor.set("evidencespine", {type:"local",
  command:[...]})` with
  `[<executable>, "mcp", "--base-dir", <base-dir>]`. If a server with that
  name already exists, it is left untouched.
- **Brief cache** — one `python -m evidencespine.cli harness opencode
  session-start` spawn per session, memoized in a `Map<string,
  Promise<string>>`. Failures are evicted so a later call retries instead of
  reusing a poisoned empty brief. The default session is pre-warmed at plugin
  load (fire-and-forget) so the first model call does not block on a cold
  spawn.
- **Auto-capture** — after a completed `edit`/`write`, the plugin shells
  out to `evidencespine ingest` with `--event-type action`,
  `--source-agent-id opencode-hook` and the edited `filePath` as the
  target.
- **Session end** — `session.deleted` runs
  `harness opencode session-stop --session-id <id>` (non-blocking, warning
  suppressed) exactly once per session.
- **Shell-outs** — `node:child_process` `execFile` with **argv arrays**
  (no shell, so no quoting class of bug) and a 60s timeout.

## Thread identity

The plugin passes the provider session id as `--session-id` only. The CLI
resolves the workspace thread server-side with
`evidencespine.threads.resolve_thread_id`: explicit value ->
`EVIDENCESPINE_THREAD_ID` -> stable
`proj_<sha1(realpath(git toplevel))[:12]>`. The session id travels in
`source_turn_id` / `metadata.session_id`.

Do **not** pass the per-session id as `--thread-id`: that fragments recall
into one empty thread per session, which defeats the side-car. Override the
thread only when you deliberately want isolation.

## Fail-open

Every hook is fail-open: a missing/broken binary, non-zero exit, timeout, or
exception resolves to an empty string and is never cached as a brief. The
`execFile` callback resolves (does not reject), so a dead side-car can never
break a model call, a tool call, a compaction, or session teardown. The one
visible signal is a `console.warn("[EvidenceSpine fail-open: ...]")` line.

## Verification

```sh
# the plugin must be discovered and load without a LoadError
opencode plugin list
grep 'evidencespine' ~/.local/share/opencode/log/opencode.log | head
grep -iE 'LoadError|failed to load' ~/.local/share/opencode/log/opencode.log | tail

# the MCP server registered by the plugin
opencode mcp list

# the file on disk
head -40 .opencode/plugins/evidencespine.ts
evidencespine harness doctor --target-dir . --json   # opencode_plugin check: V2 export + compaction hook
```

## Tooling caveat

The repo's CI has no TypeScript toolchain and no in-repo `tsc` harness: there
is no `package.json`, no `node_modules`, and the plugin file is generated
from a Python template (covered by Python tests that assert the V2 export,
`ctx.session.hook("compaction"`, `session.compaction.ended` and the absence
of `session.compacted`; `harness doctor` re-checks the installed artifact).
Type errors would therefore not be caught in this repo. If you want compile
checking, install `@opencode/plugin` 2.3.x as a devDependency in a
scratch package and run `tsc --noEmit` against a copy of the generated file;
the type-only import and the `satisfies Plugin.Plugin` annotation are what
make that check meaningful.

## Manual registration (no plugin)

If you only want the MCP server (no auto-recall/retain), opencode's own config
works too — print the snippet:

```sh
evidencespine harness print-config --harness opencode
```

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "servers": {
      "evidencespine": {
        "type": "local",
        "command": ["evidencespine", "mcp", "--base-dir", ".evidencespine"]
      }
    }
  }
}
```

V2 puts server names under `mcp.servers` (not directly under `mcp`) and
`command` is an argv array. Remote/HTTP entries use `type:"remote"`,
`url` and `headers`.
