"""opencode plugin delivery (.opencode/plugins/evidencespine.ts).

Targets the opencode V2 plugin API (bare default export
{ id, setup(ctx) } with ctx.session.hook("context"|"compaction"),
ctx.tool.hook("execute.after"), ctx.mcp.transform and ctx.event.subscribe()),
verified against @opencode/plugin@2.3.2 types (opencode v2.3.0 runtime):

* ctx.mcp.transform             registers the EvidenceSpine MCP server
  (only when absent) so tools and resources are available to the agent.
* ctx.session.hook("context")   auto-recall: appends the bounded context brief
  to every assembled agent-loop model context, memoized per session so one
  python spawn serves the whole session.
* ctx.session.hook("compaction") auto-recall for checkpoint summaries.
  "context" runs only for the agent loop; "compaction" is a SEPARATE hook in
  V2 and must be registered explicitly or the brief is missing when a
  session is compacted (opencode.ai/docs/build/plugins/server.md).
* ctx.tool.hook("execute.after") auto-capture: best-effort record of edited
  files so claims stay grounded without manual ingest.
* ctx.event.subscribe()          retention lifecycle: warms recall on
  session.created, records session end on session.deleted only (session.idle
  fires after every turn - using it as session end over-ingests), and drops
  the cached brief on session.compaction.ended (session.compacted is not a V2
  event).

Thread identity: the plugin passes the provider session id as --session-id
(metadata/source_turn_id only) and lets the CLI resolve the workspace thread
with evidencespine.threads.resolve_thread_id. Passing the per-session id as
--thread-id would fragment recall into one empty thread per session.

The type-only import is erased at runtime; keeping it (instead of a value
import of the plugin package or Plugin.define) preserves the zero-resolution
loading that is verified working for a bare .ts plugin file.

Shell-outs use node:child_process argv arrays (no shell quoting class of
bugs) with a 60s timeout. Every hook fails open (never breaks the session;
failures resolve to empty strings and are never cached as briefs).
Requires opencode >= 2.x.
"""

from __future__ import annotations

import json
import shlex
from typing import Dict, List

BACKTICK = chr(96)


def _js(v: str) -> str:
    return json.dumps(v, ensure_ascii=True)


def _js_argv(argv: List[str]) -> str:
    return "[" + ", ".join(json.dumps(t, ensure_ascii=True) for t in argv) + "]"


def split_executable(executable: str) -> List[str]:
    """Split an executable into argv tokens (spaced paths stay one token)."""
    import os

    text = str(executable or "").strip()
    if not text:
        return ["evidencespine"]
    if os.path.exists(text):
        return [text]
    try:
        argv = shlex.split(text)
    except Exception:
        return [text]
    if len(argv) <= 1:
        return argv or ["evidencespine"]
    import shutil

    if shutil.which(argv[0]) is not None or os.path.exists(argv[0]):
        return argv
    return [text]


def mcp_command_argv(executable: str, base_dir: str) -> List[str]:
    return [*split_executable(executable), "mcp", "--base-dir", base_dir]


_TEMPLATE = """\
import { execFile } from "node:child_process"
import type { Plugin } from "@opencode/plugin"

const baseDir = __BASE__
const exeArgv: string[] = __EXE_ARGV__
const mcpCommand: string[] = __MCP_COMMAND__

const briefCache = new Map<string, Promise<string>>()

function runExe(args: string[], opts?: { warn?: boolean }): Promise<string> {
  return new Promise((resolve) => {
    execFile(exeArgv[0], [...exeArgv.slice(1), ...args], { timeout: 60000 }, (error, stdout) => {
      if (error) {
        if (opts?.warn !== false) {
          console.warn(__BT__[EvidenceSpine fail-open: evidencespine ${args.slice(0, 2).join(" ")} unavailable: ${error.message}]__BT__)
        }
        resolve("")
        return
      }
      resolve(String(stdout ?? "").trim())
    })
  })
}

// The provider session id is metadata only: the CLI resolves the workspace
// thread id server-side (EVIDENCESPINE_THREAD_ID or proj_<hash>). Using a
// per-session thread id would create one empty thread per session.
function runSessionStart(sessionID: string): Promise<string> {
  return runExe(["harness", "opencode", "session-start", "--session-id", sessionID, "--base-dir", baseDir])
}

function runSessionStop(sessionID: string): Promise<void> {
  return runExe(["harness", "opencode", "session-stop", "--session-id", sessionID, "--base-dir", baseDir], {
    warn: false,
  }).then(() => undefined)
}

function recordEdit(sessionID: string, file: string): Promise<void> {
  return runExe(
    [
      "ingest",
      "--event-type",
      "action",
      "--source-agent-id",
      "opencode-hook",
      "--source-turn-id",
      "opencode-edit",
      "--session-id",
      sessionID,
      "--claim",
      "edited file",
      "--target",
      file,
      "--base-dir",
      baseDir,
    ],
    { warn: false },
  ).then(() => undefined)
}

function sessionIdOf(event: any): string {
  if (typeof event?.sessionID === "string" && event.sessionID) return event.sessionID
  const props = event?.properties
  if (typeof props?.sessionID === "string" && props.sessionID) return props.sessionID
  if (typeof event?.data?.sessionID === "string" && event.data.sessionID) return event.data.sessionID
  const id = props?.info?.id ?? props?.id ?? event?.data?.sessionID
  if (typeof id === "string" && id) return id
  return "default"
}

const plugin = {
  id: "evidencespine",
  async setup(ctx: Plugin.Context) {
    await ctx.mcp.transform((editor) => {
      if (!editor.get("evidencespine")) {
        editor.set("evidencespine", { type: "local", command: mcpCommand })
      }
    })

    // Pre-warm the default session at plugin load (fire-and-forget) so the
    // first model call never blocks on a cold python spawn.
    const warmupDefault = runSessionStart("default")

    function briefForSession(sessionID: string): Promise<string> {
      const cached = briefCache.get(sessionID)
      if (cached) return cached
      const fresh = sessionID === "default" ? warmupDefault : runSessionStart(sessionID)
      briefCache.set(sessionID, fresh)
      // Evict failures so a later call retries instead of reusing poison.
      fresh.then((text) => {
        if (!text) briefCache.delete(sessionID)
      })
      return fresh
    }

    // Auto-recall for the agent loop (primary calls and tool continuations).
    await ctx.session.hook("context", async (event) => {
      try {
        const brief = await briefForSession(event.sessionID || "default")
        if (brief) event.system.push({ type: "text", text: "\\n\\n" + brief })
      } catch {
        // fail open: never break the model call
      }
    })

    // Auto-recall for checkpoint summaries. "context" is NOT called for
    // compaction turns; V2 exposes a separate "compaction" session hook.
    await ctx.session.hook("compaction", async (event) => {
      try {
        const brief = await briefForSession(event.sessionID || "default")
        if (brief) event.system.push({ type: "text", text: "\\n\\n" + brief })
      } catch {
        // fail open: never break compaction
      }
    })

    // Auto-capture: record edited files (completed edit/write calls only).
    await ctx.tool.hook("execute.after", async (event) => {
      try {
        if (event.status !== "completed") return
        if (event.tool !== "edit" && event.tool !== "write") return
        const input = event.input as any
        const file = String(input?.filePath ?? input?.path ?? input?.file ?? "").trim()
        if (file) await recordEdit(event.sessionID || "default", file)
      } catch {
        // fail open
      }
    })

    const controller = new AbortController()
    void (async () => {
      try {
        for await (const event of ctx.event.subscribe({ signal: controller.signal })) {
          try {
            const type = (event as any)?.type
            if (type === "session.created") {
              const id = sessionIdOf(event)
              if (!briefCache.has(id)) briefCache.set(id, runSessionStart(id))
            } else if (type === "session.deleted") {
              // Session end only. session.idle is a per-turn busy->idle
              // transition; ingesting it would record one "session end" per
              // turn instead of once per session.
              const id = sessionIdOf(event)
              try {
                await runSessionStop(id)
              } finally {
                briefCache.delete(id)
              }
            } else if (type === "session.compaction.ended") {
              // Injection rides the compaction hook; drop the cached brief so
              // the next call re-anchors fresh.
              briefCache.delete(sessionIdOf(event))
            }
          } catch {
            // fail open per-event
          }
        }
      } catch {
        // subscribe ended or aborted
      }
    })()
    return () => controller.abort()
  },
} satisfies Plugin.Plugin

export default plugin
"""


def build_plugin_ts(
    *,
    executable: str = "evidencespine",
    base_dir: str = ".evidencespine",
    mcp_command: str | None = None,
) -> str:
    """Generate the opencode plugin TypeScript source.

    mcp_command is the JS expression for the MCP server argv array. When
    omitted it is derived from executable (split via shlex) so the multi-token
    "python -m evidencespine.cli" fallback launches correctly.
    """
    if mcp_command is None:
        mcp_command = _js_argv(mcp_command_argv(executable, base_dir))
    template = _TEMPLATE.replace("__MCP_COMMAND__", mcp_command)
    template = template.replace("__EXE_ARGV__", _js_argv(split_executable(executable)))
    template = template.replace("__BASE__", _js(base_dir))
    return template.replace("__BT__", BACKTICK)


def build_cursor_mcp_json(
    *,
    executable: str = "evidencespine",
    base_dir: str = ".evidencespine",
) -> Dict[str, object]:
    """Deprecated: build the .cursor/mcp.json payload for Cursor via MCP.

    This helper lives here for historical reasons only (it has nothing to do
    with the opencode plugin). Use evidencespine.harness.cursor for Cursor
    installs; kept for backward compatibility.
    """
    args = [*split_executable(executable), "mcp", "--base-dir", base_dir]
    return {"mcpServers": {"evidencespine": {"command": args[0], "args": args[1:]}}}
