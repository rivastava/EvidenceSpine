"""opencode plugin delivery (`.opencode/plugins/evidencespine.ts`).

The generated TypeScript plugin hooks opencode's plugin surface:

* ``config``                 registers the EvidenceSpine MCP server so tools and
  resources are available to the agent.
* ``experimental.chat.system.transform``  auto-recall: appends the bounded
  context brief to the system prompt once per session (memoized by session id,
  so it runs once instead of on every message).
* ``experimental.session.compacting``     retain-through-compaction: injects the
  current working-state brief into the compaction continuation context.
* ``event`` (session.deleted)             auto-retain: records session end and
  optionally emits a handoff packet.

Every hook fails open (never breaks the session).
"""

from __future__ import annotations

import json
from typing import Dict


def _js(v: str) -> str:
    return json.dumps(v, ensure_ascii=True)


_TEMPLATE = """\
import type { Plugin } from "@opencode-ai/plugin"

const baseDir = __BASE__
const executable = __EXE__

const briefCache = new Map<string, Promise<string>>()

export const EvidenceSpine: Plugin = async ({ $ }) => {
  async function runSessionStart(): Promise<string> {
    try {
      const proc = await $\u0060${executable} harness opencode session-start --base-dir ${baseDir}\u0060
      return (await proc.text()).trim()
    } catch (err) {
      return `[EvidenceSpine fail-open: session-start unavailable: ${err}]`
    }
  }

  async function runSessionStop(): Promise<void> {
    try {
      await $\u0060${executable} harness opencode session-stop --base-dir ${baseDir}\u0060.quiet()
    } catch {
      // fail open
    }
  }

  async function runCompaction(): Promise<string> {
    try {
      const proc = await $\u0060${executable} harness opencode compaction --base-dir ${baseDir}\u0060
      return (await proc.text()).trim()
    } catch (err) {
      return `[EvidenceSpine fail-open: compaction unavailable: ${err}]`
    }
  }

  // Pre-warm session-start at plugin load (fire-and-forget) so the first
  // message never blocks on a cold python spawn inside the message pipeline.
  const warmup = runSessionStart()

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
      let cached = briefCache.get(sessionID)
      if (!cached) {
        cached = warmup
        briefCache.set(sessionID, cached)
      }
      output.system.push("\\n\\n" + (await cached))
    },
    "experimental.session.compacting": async (input, output) => {
      output.context.push(await runCompaction())
    },
    event: async ({ event }) => {
      if (event.type === "session.deleted") {
        await runSessionStop()
      }
    },
  }
}
"""


def build_plugin_ts(
    *,
    executable: str = "evidencespine",
    base_dir: str = ".evidencespine",
    mcp_command: str = "executable",
) -> str:
    """Generate the opencode plugin TypeScript source.

    ``mcp_command`` is the JS expression for the MCP server argv array. It
    defaults to ``executable`` (a single-token console script). When the
    executable is the multi-token ``python -m evidencespine.cli`` fallback,
    pass a split argv expression instead so the MCP config launches correctly.
    """
    template = _TEMPLATE.replace("__MCP_COMMAND__", mcp_command)
    return template.replace("__BASE__", _js(base_dir)).replace("__EXE__", _js(executable))


def build_cursor_mcp_json(
    *,
    executable: str = "evidencespine",
    base_dir: str = ".evidencespine",
) -> Dict[str, object]:
    """Build the `.cursor/mcp.json` payload for Cursor/Codex via MCP."""
    return {
        "mcpServers": {
            "evidencespine": {
                "command": executable,
                "args": ["mcp", "--base-dir", base_dir],
            }
        }
    }
