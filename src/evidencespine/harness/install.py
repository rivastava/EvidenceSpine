"""Harness delivery installation and debugging.

Installs the EvidenceSpine delivery layer into a coding agent's harness:

* ``claude-code``  writes ``.claude-plugin/plugin.json`` with SessionStart /
  PreCompact / Stop hooks.
* ``opencode``     writes ``.opencode/plugins/evidencespine.ts`` with MCP
  registration, system-transform auto-recall, compaction retention, and
  session-deleted auto-retain.
* ``cursor``       writes ``.cursor/mcp.json`` so Cursor/Codex attach to the MCP
  server (tools + resources; delivery hooks are handled by the harness MCP).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from evidencespine.harness.claude_code import build_manifest
from evidencespine.harness.opencode import build_cursor_mcp_json, build_plugin_ts
from evidencespine.harness.hooks import build_runtime


def resolve_executable() -> str:
    """Resolve the ``evidencespine`` executable used by harness hooks.

    Prefers a console script on PATH; falls back to the running interpreter so
    plugins work in development without a global install.
    """
    found = shutil.which("evidencespine")
    if found:
        return found
    return f"{sys.executable} -m evidencespine.cli"


def install_claude_code(
    *,
    target_dir: str,
    base_dir: str = ".evidencespine",
    executable: Optional[str] = None,
    include_precompact: bool = True,
) -> Dict[str, Any]:
    exe = executable or resolve_executable()
    manifest = build_manifest(executable=exe, base_dir=base_dir, include_precompact=include_precompact)
    plugin_dir = Path(target_dir) / ".claude-plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    path = plugin_dir / "plugin.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {"status": "ok", "harness": "claude-code", "path": str(path), "wrote": [str(path)]}


def install_opencode(
    *,
    target_dir: str,
    base_dir: str = ".evidencespine",
    executable: Optional[str] = None,
    scope: str = "project",
) -> Dict[str, Any]:
    exe = executable or resolve_executable()
    tokens = exe.split()
    mcp_command = "[" + ", ".join(json.dumps(t, ensure_ascii=True) for t in tokens + ["mcp", "--base-dir", base_dir]) + "]"
    source = build_plugin_ts(executable=exe, base_dir=base_dir, mcp_command=mcp_command)
    if str(scope).lower() == "global":
        plugins_dir = Path(target_dir)
    else:
        plugins_dir = Path(target_dir) / ".opencode" / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    path = plugins_dir / "evidencespine.ts"
    path.write_text(source, encoding="utf-8")
    return {"status": "ok", "harness": "opencode", "scope": scope, "path": str(path), "wrote": [str(path)]}


def install_cursor(
    *,
    target_dir: str,
    base_dir: str = ".evidencespine",
    executable: Optional[str] = None,
) -> Dict[str, Any]:
    exe = executable or resolve_executable()
    payload = build_cursor_mcp_json(executable=exe, base_dir=base_dir)
    cursor_dir = Path(target_dir) / ".cursor"
    cursor_dir.mkdir(parents=True, exist_ok=True)
    path = cursor_dir / "mcp.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {"status": "ok", "harness": "cursor", "path": str(path), "wrote": [str(path)]}


def install_harness(
    *,
    harness: str,
    target_dir: str,
    base_dir: str = ".evidencespine",
    executable: Optional[str] = None,
    scope: str = "project",
) -> Dict[str, Any]:
    name = harness.strip().lower()
    if name == "claude-code":
        return install_claude_code(target_dir=target_dir, base_dir=base_dir, executable=executable)
    if name == "opencode":
        return install_opencode(target_dir=target_dir, base_dir=base_dir, executable=executable, scope=scope)
    if name == "cursor":
        return install_cursor(target_dir=target_dir, base_dir=base_dir, executable=executable)
    if name == "all":
        results: List[Dict[str, Any]] = [
            install_claude_code(target_dir=target_dir, base_dir=base_dir, executable=executable),
            install_opencode(target_dir=target_dir, base_dir=base_dir, executable=executable, scope=scope),
            install_cursor(target_dir=target_dir, base_dir=base_dir, executable=executable),
        ]
        return {"status": "ok", "harness": "all", "scope": scope, "wrote": [p for r in results for p in r["wrote"]]}
    raise ValueError(f"unknown harness: {harness!r} (expected claude-code|opencode|cursor|all)")


def debug_harness(
    *,
    base_dir: str = ".evidencespine",
    storage_format: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Report harness delivery health: store connectivity, counts, brief build."""
    try:
        runtime = build_runtime(base_dir, storage_format)
        thread = thread_id or os.path.basename(os.getcwd()) or "default"
        snapshot = runtime.snapshot()
        brief = runtime.build_brief(thread_id=thread, query="harness debug")
        return {
            "status": "ok",
            "base_dir": base_dir,
            "storage_format": runtime.config.storage_format,
            "events_total": int(snapshot.get("events_total", 0)),
            "facts_total": int(snapshot.get("facts_total", 0)),
            "brief_ok": True,
            "brief_token_budget": int(brief.token_budget),
            "brief_sections": sum(
                1
                for key in ("current_goal", "locked_decisions", "recent_verified_facts", "active_risks", "open_items", "next_actions")
                if brief.to_dict().get(key)
            ),
        }
    except Exception as exc:
        return {"status": "fail_open", "reason": str(exc)}
