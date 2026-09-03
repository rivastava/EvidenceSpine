"""Cursor harness delivery (`.cursor/mcp.json` + `.cursor/rules/` + hooks).

Spec (cursor.com/docs/{mcp,rules,hooks,reference/permissions}):

* MCP: project ``.cursor/mcp.json`` vs global ``~/.cursor/mcp.json``,
  shape ``{"mcpServers": {"evidencespine": {"command": ..., "args": [...]}}}``.
  ``command`` MUST be a single binary; multi-token executables
  (``python -m evidencespine.cli`` fallback) must split into command+args.
* Rules: project ``.cursor/rules/**/*.mdc`` with frontmatter
  ``description, globs, alwaysApply:true``. ``AGENTS.md`` is auto-ingested.
* Hooks: ``<proj>/.cursor/hooks.json`` (cwd proj root) vs
  ``~/.cursor/hooks.json``; events include sessionStart/sessionEnd,
  preCompact, stop, before/afterShellExecution, before/afterMCPExecution,
  pre/postToolUse, beforeSubmitPrompt, afterFileEdit.
* Permissions: ``.cursor/permissions.json`` ``mcpAllowlist`` + Run Modes
  Auto-review/Allowlist/yolo. Installer ships an allowlist for the spine
  so tools are integral, not opt-in per call.
"""

from __future__ import annotations

import shlex
from typing import Any, Dict, List

from evidencespine.usage import usage_guide_markdown


def split_executable(executable: str) -> List[str]:
    try:
        argv = shlex.split(executable or "")
    except Exception:
        argv = []
    return argv or ["evidencespine"]


def build_mcp_server_entry(executable: str = "evidencespine", base_dir: str = ".evidencespine") -> Dict[str, Any]:
    argv = split_executable(executable)
    return {
        "command": argv[0],
        "args": [*argv[1:], "mcp", "--base-dir", base_dir],
    }


def build_mcp_json(
    *,
    executable: str = "evidencespine",
    base_dir: str = ".evidencespine",
) -> Dict[str, Any]:
    return {"mcpServers": {"evidencespine": build_mcp_server_entry(executable, base_dir)}}


def merge_mcp_json(existing: Any, *, executable: str = "evidencespine", base_dir: str = ".evidencespine") -> Dict[str, Any]:
    """Merge the spine entry into an existing mcp.json payload (non-destructive)."""
    payload: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    else:
        servers = dict(servers)
    servers["evidencespine"] = build_mcp_server_entry(executable, base_dir)
    payload["mcpServers"] = servers
    return payload


def build_rules_mdc() -> str:
    """Render `.cursor/rules/evidencespine.mdc` (alwaysApply) from the canonical guide."""
    guide = usage_guide_markdown()
    return (
        "---\n"
        "description: EvidenceSpine evidence-bound memory — consult briefs, ground verified claims, retain handoffs\n"
        "globs:\n"
        "alwaysApply: true\n"
        "---\n\n"
        f"{guide}\n"
    )


def build_permissions_json() -> Dict[str, Any]:
    return {"mcpAllowlist": ["evidencespine:*"]}


def merge_permissions_json(existing: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    allow = payload.get("mcpAllowlist")
    merged: List[str] = list(allow) if isinstance(allow, list) else []
    for entry in ("evidencespine:*",):
        if entry not in merged:
            merged.append(entry)
    payload["mcpAllowlist"] = merged
    return payload


def build_hooks_json(*, executable: str = "evidencespine", base_dir: str = ".evidencespine") -> Dict[str, Any]:
    """Minimal Cursor hooks wiring sessionStart/preCompact/stop to shared handlers.

    Shape per cursor.com/docs/hooks: top-level ``{"version": 1, "hooks": ...}``
    with camelCase event names; project hooks run from the project root.
    """

    def _cmd(action: str) -> str:
        argv = split_executable(executable)
        return shlex.join([*argv, "harness", "cursor", action, "--base-dir", base_dir])

    return {
        "version": 1,
        "hooks": {
            "sessionStart": [{"command": _cmd("session-start")}],
            "preCompact": [{"command": _cmd("precompact")}],
            "stop": [{"command": _cmd("session-stop")}],
        },
    }
