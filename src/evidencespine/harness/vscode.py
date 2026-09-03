"""VSCode-family MCP delivery (`.vscode/mcp.json`).

Spec (code.visualstudio.com/docs/agents/reference/mcp-configuration):

* Workspace: ``.vscode/mcp.json`` (commit) vs user profile
  (``MCP: Open User Configuration``).
* Key is ``servers`` (NOT ``mcpServers``); ``type: stdio|http|sse`` required.
* Fields: ``command/args/cwd/env/envFile/inputs`` + ``sandbox``.
* Portable Agent Host also reads ``.mcp.json`` / ``~/.copilot/mcp-config.json``.
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


def merge_mcp_json(existing: Any, *, executable: str = "evidencespine", base_dir: str = ".evidencespine") -> Dict[str, Any]:
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
