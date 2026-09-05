"""opencode plugin delivery (`.opencode/plugins/evidencespine.ts`).

Targets the opencode v2 plugin API (``export default { id, setup }`` with
``ctx.session.hook("context")``, ``ctx.tool.hook("execute.after")``,
``ctx.mcp.transform`` and ``ctx.event.subscribe()``), verified against the
``@opencode-ai/plugin`` dev types and the opencode.ai/v2 plugin docs:

* ``ctx.mcp.transform``            registers the EvidenceSpine MCP server
  (only when absent) so tools and resources are available to the agent.
* ``ctx.session.hook("context")``  auto-recall: appends the bounded context
  brief to every assembled model context (primary calls and compaction alike),
  memoized per session so one python spawn serves the whole session.
* ``ctx.tool.hook("execute.after")`` auto-capture: best-effort record of
  edited files so claims stay grounded without manual ingest.
* ``ctx.event.subscribe()``        auto-retain: warms recall on
  ``session.created``, records session end on ``session.deleted``/``idle``,
  and re-anchors recall on ``session.compacted``.

Shell-outs use ``node:child_process`` argv arrays (no shell quoting class of
bugs) with a 60s timeout. Every hook fails open (never breaks the session;
failures resolve to empty strings and are never cached as briefs).
"""

from __future__ import annotations

import json
import shlex
from typing import Dict, List


def _js(v: str) -> str:
    return json.dumps(v, ensure_ascii=True)


def _js_argv(argv: List[str]) -> str:
    return "[" + ", ".join(json.dumps(t, ensure_ascii=True) for t in argv) + "]"


def split_executable(executable: str) -> List[str]:
    try:
        argv = shlex.split(executable or "")
    except Exception:
        argv = []
    return argv or ["evidencespine"]


def mcp_command_argv(executable: str, base_dir: str) -> List[str]:
    return [*split_executable(executable), "mcp", "--base-dir", base_dir]


_TEMPLATE = """\
import { execFile } from "node:child_process"
import type { Plugin } from "@opencode-ai/plugin"

const baseDir = __BASE__
const exeArgv: string[] = __EXE_ARGV__
const mcpCommand: string[] = __MCP_COMMAND__

const briefCache = new Map<string, Promise<string>>()

function runExe(args: string[], opts?: { warn?: boolean }): Promise<string> {
  return new Promise((resolve) => {
    execFile(exeArgv[0], [...exeArgv.slice(1), ...args], { timeout: 60000 }, (error, stdout) => {
      if (error) {
        if (opts?.warn !== false) {
          console.warn(`[EvidenceSpine fail-open: evidencespine ${args.slice(0, 2).join(" ")} unavailable: ${error.message}]`)
        }
        resolve("")
        return
      }
      resolve(String(stdout ?? "").trim())
    })
  })
}

function runSessionStart(sessionID: string): Promise<string> {
  return runExe(["harness", "opencode", "session-start", "--thread-id", sessionID, "--base-dir", baseDir])
}

function runSessionStop(sessionID: string): Promise<void> {
  return runExe(["harness", "opencode", "session-stop", "--thread-id", sessionID, "--base-dir", baseDir], {
    warn: false,
  }).then(() => undefined)
}

function recordEdit(sessionID: string, file: string): Promise<void> {
  return runExe(
    [
      "ingest",
      "--thread-id",
      sessionID,
      "--event-type",
      "action",
      "--source-agent-id",
      "opencode-hook",
      "--source-turn-id",
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
  const id = props?.info?.id ?? props?.id
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

    // Auto-recall: fires for primary calls and compaction alike.
    await ctx.session.hook("context", async (event) => {
      try {
        const brief = await briefForSession(event.sessionID || "default")
        if (brief) event.system.push({ type: "text", text: "\\n\\n" + brief })
      } catch {
        // fail open: never break the model call
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
            } else if (type === "session.deleted" || type === "session.idle") {
              const id = sessionIdOf(event)
              try {
                await runSessionStop(id)
              } finally {
                briefCache.delete(id)
              }
            } else if (type === "session.compacted") {
              // Injection already rode the context hook through compaction;
              // drop the cached brief so the next call re-anchors fresh.
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

    ``mcp_command`` is the JS expression for the MCP server argv array. When
    omitted it is derived from ``executable`` (split via shlex) so the
    multi-token ``python -m evidencespine.cli`` fallback launches correctly.
    """
    if mcp_command is None:
        mcp_command = _js_argv(mcp_command_argv(executable, base_dir))
    template = _TEMPLATE.replace("__MCP_COMMAND__", mcp_command)
    template = template.replace("__EXE_ARGV__", _js_argv(split_executable(executable)))
    return template.replace("__BASE__", _js(base_dir))


def build_cursor_mcp_json(
    *,
    executable: str = "evidencespine",
    base_dir: str = ".evidencespine",
) -> Dict[str, object]:
    """Build the `.cursor/mcp.json` payload for Cursor/Codex via MCP.

    .. deprecated:: Use :mod:`evidencespine.harness.cursor` instead; kept for
        backward compatibility. Correctly splits multi-token executables.
    """
    from evidencespine.harness.cursor import build_mcp_json as _build

    return _build(executable=executable, base_dir=base_dir)  # type: ignore[return-value]
