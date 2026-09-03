"""opencode plugin delivery (`.opencode/plugins/evidencespine.ts`).

The generated TypeScript plugin hooks opencode's plugin surface
(opencode.ai/docs/plugins, v1 lineage):

* ``config``                 registers the EvidenceSpine MCP server so tools and
  resources are available to the agent.
* ``experimental.chat.system.transform``  auto-recall: appends the bounded
  context brief to the system prompt once per session (memoized by session id,
  so it runs once instead of on every message).
* ``experimental.session.compacting``     retain-through-compaction: injects the
  current working-state brief into the compaction continuation context.
* ``event`` (session.deleted)             auto-retain: records session end and
  optionally emits a handoff packet.
* ``tool.execute.after`` (edit/write)     auto-capture: best-effort record of
  edited files so claims stay grounded without manual ingest.

Every hook fails open (never breaks the session; failures resolve to empty
strings and are never cached as briefs).
"""

from __future__ import annotations

import json
import shlex
from typing import Dict, List


def _js(v: str) -> str:
    return json.dumps(v, ensure_ascii=True)


def split_executable(executable: str) -> List[str]:
    try:
        argv = shlex.split(executable or "")
    except Exception:
        argv = []
    return argv or ["evidencespine"]


def mcp_command_argv(executable: str, base_dir: str) -> List[str]:
    return [*split_executable(executable), "mcp", "--base-dir", base_dir]


_TEMPLATE = """\
import type { Plugin } from "@opencode-ai/plugin"

const baseDir = __BASE__
const executable = __EXE__

const briefCache = new Map<string, Promise<string>>()

export const EvidenceSpine: Plugin = async ({ $ }) => {
  async function runSessionStart(sessionID: string): Promise<string> {
    try {
      const proc = await $\u0060${executable} harness opencode session-start --thread-id "${sessionID}" --base-dir "${baseDir}"\u0060
      return (await proc.text()).trim()
    } catch (err) {
      console.warn(`[EvidenceSpine fail-open: session-start unavailable: ${err}]`)
      return ""
    }
  }

  async function runSessionStop(sessionID: string): Promise<void> {
    try {
      await $\u0060${executable} harness opencode session-stop --thread-id "${sessionID}" --base-dir "${baseDir}"\u0060.quiet()
    } catch {
      // fail open
    }
  }

  async function runCompaction(sessionID: string): Promise<string> {
    try {
      const proc = await $\u0060${executable} harness opencode compaction --thread-id "${sessionID}" --base-dir "${baseDir}"\u0060
      return (await proc.text()).trim()
    } catch (err) {
      console.warn(`[EvidenceSpine fail-open: compaction unavailable: ${err}]`)
      return ""
    }
  }

  async function recordEdit(sessionID: string, file: string): Promise<void> {
    if (!file) return
    try {
      await $\u0060${executable} ingest --thread-id "${sessionID}" --event-type action --source-agent-id opencode-hook --source-turn-id "${sessionID}" --claim "edited file" --target "${file}" --base-dir "${baseDir}"\u0060.quiet()
    } catch {
      // fail open: edits must never break the session
    }
  }

  // Pre-warm default session at plugin load (fire-and-forget) so the first
  // message never blocks on a cold python spawn inside the message pipeline.
  const warmupDefault = runSessionStart("default")

  function briefFor(sessionID: string): Promise<string> {
    const cached = briefCache.get(sessionID)
    if (cached) return cached
    const fresh =
      sessionID === "default" ? warmupDefault : runSessionStart(sessionID)
    briefCache.set(sessionID, fresh)
    // Evict failures so a later message retries instead of reusing poison.
    fresh.then((text) => {
      if (!text) briefCache.delete(sessionID)
    })
    return fresh
  }

  return {
    config: (cfg) => {
      if (!cfg.mcp || !cfg.mcp["evidencespine"]) {
        cfg.mcp = cfg.mcp || {}
        cfg.mcp["evidencespine"] = {
          type: "local",
          command: __MCP_COMMAND__,
          enabled: true,
        }
      }
    },
    "experimental.chat.system.transform": async (input, output) => {
      const sessionID = input.sessionID || "default"
      const brief = await briefFor(sessionID)
      if (brief) output.system.push("\\n\\n" + brief)
    },
    "experimental.session.compacting": async (input, output) => {
      const sessionID = (input as any)?.sessionID || "default"
      const note = await runCompaction(sessionID)
      if (note) output.context.push(note)
    },
    "tool.execute.after": async (input, output) => {
      try {
        const tool = String((input as any)?.tool || "")
        if (tool !== "edit" && tool !== "write") return
        const sessionID = String((input as any)?.sessionID || "default")
        const args = (input as any)?.args as any
        const file =
          String(args?.filePath || args?.path || args?.file || "").trim()
        if (file) await recordEdit(sessionID, file)
      } catch {
        // fail open
      }
    },
    event: async ({ event }) => {
      const type = (event as any)?.type
      const props = (event as any)?.properties as any
      if (type === "session.created") {
        const id = String(props?.info?.id || props?.id || "default")
        if (!briefCache.has(id)) briefCache.set(id, runSessionStart(id))
      } else if (type === "session.deleted") {
        const id = String(props?.info?.id || props?.id || "default")
        try {
          await runSessionStop(id)
        } finally {
          briefCache.delete(id)
        }
      } else if (type === "session.idle") {
        const id = String(props?.info?.id || props?.id || "default")
        try {
          await runSessionStop(id)
        } catch {
          // fail open
        }
      }
    },
  }
}
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
        argv = mcp_command_argv(executable, base_dir)
        mcp_command = "[" + ", ".join(json.dumps(t, ensure_ascii=True) for t in argv) + "]"
    template = _TEMPLATE.replace("__MCP_COMMAND__", mcp_command)
    return template.replace("__BASE__", _js(base_dir)).replace("__EXE__", _js(executable))


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
