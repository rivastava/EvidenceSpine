"""VSCode-family MCP delivery.

Two shapes exist in 2026 (code.visualstudio.com/docs/agents/reference/mcp-configuration):

* LEGACY / deprecated editor-managed "Local harness" shape:
  workspace .vscode/mcp.json (or a profile mcp.json) with top-level key
  "servers" and entries {"type": "stdio"|"http"|"sse", command, args, ...}.
  build_mcp_json/merge_mcp_json keep writing this for existing users, but the
  vendor docs now label it deprecated.
* PORTABLE VS Code Copilot shape: workspace .mcp.json or user
  ~/.copilot/mcp-config.json (COPILOT_HOME) with top-level key "mcpServers"
  and entries {"type": "stdio"|"http", command, args, env, cwd}.
  build_copilot_mcp_json/merge_copilot_mcp_json implement this; the
  "vscode-copilot" installer target uses it.
"""

from __future__ import annotations

import shlex
from typing import Any, Dict, List


def split_executable(executable: str) -> List[str]:
    try:
        argv = shlex.split(executable or "")
    except Exception:
        argv = []
    return argv or ["evidencespine"]


def build_mcp_json(
    *,
    executable: str = "evidencespine",
    base_dir: str = ".evidencespine",
) -> Dict[str, Any]:
    """Deprecated Local-harness shape (.vscode/mcp.json, key "servers")."""
    argv = split_executable(executable)
    return {
        "servers": {
            "evidencespine": {
                "type": "stdio",
                "command": argv[0],
                "args": [*argv[1:], "mcp", "--base-dir", base_dir],
            }
        }
    }


def merge_mcp_json(
    existing: Any,
    *,
    executable: str = "evidencespine",
    base_dir: str = ".evidencespine",
) -> Dict[str, Any]:
    """Merge the legacy spine entry into an existing .vscode/mcp.json."""
    payload: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    servers = payload.get("servers")
    if not isinstance(servers, dict):
        servers = {}
    else:
        servers = dict(servers)
    fresh = build_mcp_json(executable=executable, base_dir=base_dir)["servers"]["evidencespine"]
    servers["evidencespine"] = fresh
    payload["servers"] = servers
    return payload


def build_copilot_mcp_json(
    *,
    executable: str = "evidencespine",
    base_dir: str = ".evidencespine",
) -> Dict[str, Any]:
    """Portable VS Code Copilot shape (.mcp.json / ~/.copilot/mcp-config.json)."""
    argv = split_executable(executable)
    return {
        "mcpServers": {
            "evidencespine": {
                "type": "stdio",
                "command": argv[0],
                "args": [*argv[1:], "mcp", "--base-dir", base_dir],
                "tools": ["*"],
            }
        }
    }


def merge_copilot_mcp_json(
    existing: Any,
    *,
    executable: str = "evidencespine",
    base_dir: str = ".evidencespine",
) -> Dict[str, Any]:
    """Merge the portable spine entry into an existing Copilot MCP config."""
    payload: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    else:
        servers = dict(servers)
    fresh = build_copilot_mcp_json(executable=executable, base_dir=base_dir)["mcpServers"]["evidencespine"]
    servers["evidencespine"] = fresh
    payload["mcpServers"] = servers
    return payload
