"""Harness delivery installation and debugging.

Installs the EvidenceSpine delivery layer into a coding agent's harness:

* ``claude-code``  writes ``.claude-plugin/plugin.json`` (pointer) +
  ``.claude-plugin/hooks/hooks.json`` (wrapped ``{"hooks": {...}}`` with
  SessionStart/SessionEnd/PreCompact) and merges ``.claude/settings.json``.
* ``opencode``     writes ``.opencode/plugins/evidencespine.ts`` with MCP
  registration, system-transform auto-recall, compaction retention,
  tool.execute.after auto-capture, and session event auto-retain.
* ``cursor``       merges ``.cursor/mcp.json`` (non-destructive), writes
  ``.cursor/rules/evidencespine.mdc`` (alwaysApply), merges
  ``.cursor/permissions.json`` allowlist, writes ``.cursor/hooks.json``.
* ``vscode``       merges ``.vscode/mcp.json`` (``servers`` key, stdio).
* ``codex``        writes ``.codex/config.toml`` (MCP + ``[features]
  hooks=true``) and ``.codex/hooks.json``
  (SessionStart/SessionEnd/PreCompact with envelope-aware commands).
* ``git``          installs ``post-commit``/``post-merge`` hooks (delegated).
* ``agents-md``    writes ``AGENTS.md`` from the canonical guide when missing.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from evidencespine.harness.hooks import build_runtime
from evidencespine import __version__ as _package_version


def resolve_executable() -> str:
    """Resolve the ``evidencespine`` executable used by harness hooks.

    Prefers a console script on PATH; falls back to the running interpreter so
    plugins work in development without a global install.
    """
    found = shutil.which("evidencespine")
    if found:
        return found
    return f"{sys.executable} -m evidencespine.cli"


def _split_exe(executable: str) -> List[str]:
    try:
        argv = shlex.split(executable or "")
    except Exception:
        argv = []
    return argv or ["evidencespine"]


def _read_json(path: Path) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def install_claude_code(
    *,
    target_dir: str,
    base_dir: str = ".evidencespine",
    executable: Optional[str] = None,
    include_precompact: bool = True,
    scope: str = "project",
) -> Dict[str, Any]:
    from evidencespine.harness.claude_code import build_hooks_json, build_manifest, merge_mcp_json

    exe = executable or resolve_executable()
    wrote: List[str] = []
    # Plugin root = target_dir in both scopes: the manifest lives in
    # <root>/.claude-plugin/plugin.json and components resolve from <root>
    # (only plugin.json belongs inside .claude-plugin/ per the spec).
    plugin_root = Path(target_dir)
    if str(scope).lower() == "global":
        settings_path = Path(os.path.expanduser("~/.claude/settings.json"))
    else:
        settings_path = plugin_root / ".claude" / "settings.json"

    manifest = build_manifest(executable=exe, base_dir=base_dir, include_precompact=include_precompact)
    plugin_dir = plugin_root / ".claude-plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = plugin_dir / "plugin.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    wrote.append(str(manifest_path))

    hooks_payload = build_hooks_json(executable=exe, base_dir=base_dir, include_precompact=include_precompact)
    hooks_path = plugin_root / "hooks" / "hooks.json"
    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    hooks_path.write_text(json.dumps(hooks_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    wrote.append(str(hooks_path))

    # Plugin-root .mcp.json doubles as Claude Code's project MCP scope,
    # so CLI, IDE extension, and Desktop Code tab share one registration.
    mcp_path = plugin_root / ".mcp.json"
    _write_json(mcp_path, merge_mcp_json(_read_json(mcp_path), executable=exe, base_dir=base_dir))
    wrote.append(str(mcp_path))

    # Merge into settings.json (non-destructive): settings hooks are authoritative
    # for CLI/IDE when the plugin is not installed via marketplace.
    try:
        existing = _read_json(settings_path)
        settings: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
        hooks = settings.get("hooks")
        if not isinstance(hooks, dict):
            hooks = {}
        else:
            hooks = dict(hooks)
        for event, groups in (hooks_payload.get("hooks", {}) or {}).items():
            if event not in hooks:
                hooks[event] = groups
        settings["hooks"] = hooks
        _write_json(settings_path, settings)
        if str(settings_path) not in wrote:
            wrote.append(str(settings_path))
    except Exception:
        pass
    return {"status": "ok", "harness": "claude-code", "scope": scope, "path": str(manifest_path), "wrote": wrote}


def install_opencode(
    *,
    target_dir: str,
    base_dir: str = ".evidencespine",
    executable: Optional[str] = None,
    scope: str = "project",
) -> Dict[str, Any]:
    from evidencespine.harness.opencode import build_plugin_ts, mcp_command_argv

    exe = executable or resolve_executable()
    argv = mcp_command_argv(exe, base_dir)
    mcp_command = "[" + ", ".join(json.dumps(t, ensure_ascii=True) for t in argv) + "]"
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
    scope: str = "project",
) -> Dict[str, Any]:
    from evidencespine.harness.cursor import (
        build_hooks_json,
        build_permissions_json,
        build_rules_mdc,
        merge_mcp_json,
        merge_permissions_json,
    )

    exe = executable or resolve_executable()
    wrote: List[str] = []
    if str(scope).lower() == "global":
        cursor_dir = Path(target_dir)
    else:
        cursor_dir = Path(target_dir) / ".cursor"
    cursor_dir.mkdir(parents=True, exist_ok=True)

    mcp_path = cursor_dir / "mcp.json"
    merged = merge_mcp_json(_read_json(mcp_path), executable=exe, base_dir=base_dir)
    _write_json(mcp_path, merged)
    wrote.append(str(mcp_path))

    rules_path = cursor_dir / "rules" / "evidencespine.mdc"
    if not rules_path.exists():
        rules_path.parent.mkdir(parents=True, exist_ok=True)
        rules_path.write_text(build_rules_mdc(), encoding="utf-8")
        wrote.append(str(rules_path))

    perm_path = cursor_dir / "permissions.json"
    merged_perm = merge_permissions_json(_read_json(perm_path))
    # Ensure default allowlist present even when file existed without it.
    if not isinstance(merged_perm, dict) or "mcpAllowlist" not in merged_perm:
        merged_perm = build_permissions_json()
    _write_json(perm_path, merged_perm)
    if str(perm_path) not in wrote:
        wrote.append(str(perm_path))

    hooks_path = cursor_dir / "hooks.json"
    if not hooks_path.exists():
        _write_json(hooks_path, build_hooks_json(executable=exe, base_dir=base_dir))
        wrote.append(str(hooks_path))

    return {"status": "ok", "harness": "cursor", "scope": scope, "path": str(mcp_path), "wrote": wrote}


def install_vscode(
    *,
    target_dir: str,
    base_dir: str = ".evidencespine",
    executable: Optional[str] = None,
    scope: str = "project",
) -> Dict[str, Any]:
    from evidencespine.harness.vscode import merge_mcp_json

    exe = executable or resolve_executable()
    if str(scope).lower() == "global":
        vscode_dir = Path(target_dir)
    else:
        vscode_dir = Path(target_dir) / ".vscode"
    vscode_dir.mkdir(parents=True, exist_ok=True)
    path = vscode_dir / "mcp.json"
    merged = merge_mcp_json(_read_json(path), executable=exe, base_dir=base_dir)
    _write_json(path, merged)
    return {"status": "ok", "harness": "vscode", "scope": scope, "path": str(path), "wrote": [str(path)]}


def install_codex(
    *,
    target_dir: str,
    base_dir: str = ".evidencespine",
    executable: Optional[str] = None,
    scope: str = "project",
) -> Dict[str, Any]:
    from evidencespine.harness.codex import build_hooks_json, render_config_toml

    exe = executable or resolve_executable()
    wrote: List[str] = []
    if str(scope).lower() == "global":
        codex_dir = Path(target_dir)
    else:
        codex_dir = Path(target_dir) / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)

    config_path = codex_dir / "config.toml"
    snippet = render_config_toml(exe, base_dir)
    try:
        existing = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    except Exception:
        existing = ""
    if "[mcp_servers.evidencespine]" not in existing:
        with open(config_path, "a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            handle.write(snippet)
        wrote.append(str(config_path))
    elif "[features]" not in existing or "hooks" not in existing:
        # Ensure hooks feature flag without duplicating the MCP table.
        with open(config_path, "a", encoding="utf-8") as handle:
            handle.write("\n[features]\nhooks = true\n")
        if str(config_path) not in wrote:
            wrote.append(str(config_path))

    hooks_path = codex_dir / "hooks.json"
    payload = build_hooks_json(executable=exe, base_dir=base_dir)
    existing_hooks = _read_json(hooks_path)
    if isinstance(existing_hooks, dict) and isinstance(existing_hooks.get("hooks"), dict):
        merged_hooks: Dict[str, Any] = dict(existing_hooks["hooks"])
        for event, groups in (payload.get("hooks", {}) or {}).items():
            if event not in merged_hooks:
                merged_hooks[event] = groups
        payload = {"hooks": merged_hooks}
    _write_json(hooks_path, payload)
    if str(hooks_path) not in wrote:
        wrote.append(str(hooks_path))

    return {"status": "ok", "harness": "codex", "scope": scope, "path": str(config_path), "wrote": wrote or [str(config_path), str(hooks_path)]}


def install_agents_md(*, target_dir: str) -> Dict[str, Any]:
    """Write AGENTS.md from the canonical guide when missing (GUI discovery)."""
    from evidencespine.usage import usage_guide_markdown

    path = Path(target_dir) / "AGENTS.md"
    if path.exists():
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            text = ""
        if "EvidenceSpine" in text:
            return {"status": "ok", "harness": "agents-md", "path": str(path), "wrote": [], "note": "exists"}
        # Append without clobbering existing agent instructions.
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("\n\n" + usage_guide_markdown())
        return {"status": "ok", "harness": "agents-md", "path": str(path), "wrote": [str(path)], "note": "appended"}
    path.write_text(usage_guide_markdown(), encoding="utf-8")
    return {"status": "ok", "harness": "agents-md", "path": str(path), "wrote": [str(path)]}


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
        return install_claude_code(target_dir=target_dir, base_dir=base_dir, executable=executable, scope=scope)
    if name == "opencode":
        return install_opencode(target_dir=target_dir, base_dir=base_dir, executable=executable, scope=scope)
    if name == "cursor":
        return install_cursor(target_dir=target_dir, base_dir=base_dir, executable=executable, scope=scope)
    if name == "vscode":
        return install_vscode(target_dir=target_dir, base_dir=base_dir, executable=executable, scope=scope)
    if name == "codex":
        return install_codex(target_dir=target_dir, base_dir=base_dir, executable=executable, scope=scope)
    if name == "agents-md":
        return install_agents_md(target_dir=target_dir)
    if name == "git":
        from evidencespine.harness.git import install_git_hooks

        exe = executable or resolve_executable()
        # Route through resolve_executable so dev installs work; absolute base_dir
        # so hooks do not depend on hook-process CWD.
        abs_base = base_dir if os.path.isabs(base_dir) else os.path.abspath(os.path.join(target_dir, base_dir))
        inner = install_git_hooks(repo_dir=target_dir, executable=exe, base_dir=abs_base)
        return {"harness": "git", **inner}
    if name == "all":
        results: List[Dict[str, Any]] = [
            install_claude_code(target_dir=target_dir, base_dir=base_dir, executable=executable, scope=scope),
            install_opencode(target_dir=target_dir, base_dir=base_dir, executable=executable, scope=scope),
            install_cursor(target_dir=target_dir, base_dir=base_dir, executable=executable, scope=scope),
            install_vscode(target_dir=target_dir, base_dir=base_dir, executable=executable, scope=scope),
            install_codex(target_dir=target_dir, base_dir=base_dir, executable=executable, scope=scope),
            install_agents_md(target_dir=target_dir),
        ]
        # Best-effort git hooks (skip outside a git repo).
        try:
            from evidencespine.harness.git import install_git_hooks

            exe = executable or resolve_executable()
            abs_base = base_dir if os.path.isabs(base_dir) else os.path.abspath(os.path.join(target_dir, base_dir))
            git_res = install_git_hooks(repo_dir=target_dir, executable=exe, base_dir=abs_base)
            if git_res.get("status") == "ok":
                results.append({"status": "ok", "harness": "git", **git_res})
        except Exception:
            pass
        wrote: List[str] = []
        for result in results:
            for item in result.get("wrote", []) or []:
                if item not in wrote:
                    wrote.append(item)
        return {"status": "ok", "harness": "all", "scope": scope, "wrote": wrote, "details": results}
    raise ValueError(
        f"unknown harness: {harness!r} (expected claude-code|opencode|cursor|vscode|codex|git|agents-md|all)"
    )


def debug_harness(
    *,
    base_dir: str = ".evidencespine",
    storage_format: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Report harness delivery health: store, executable, MCP extra, grounding."""
    try:
        runtime = build_runtime(base_dir, storage_format)
        thread = thread_id or os.path.basename(os.getcwd()) or "default"
        snapshot = runtime.snapshot()
        brief = runtime.build_brief(thread_id=thread, query="harness debug")
        exe = resolve_executable()
        try:
            argv = shlex.split(exe)
            exe_ok = bool(argv) and (shutil.which(argv[0]) is not None or os.path.exists(argv[0]))
        except Exception:
            exe_ok = False
        try:
            import mcp  # type: ignore[import-not-found]  # noqa: F401

            mcp_available = True
        except Exception:
            mcp_available = False
        try:
            from evidencespine.grounding import ground_file as _ground

            grounding_ok = _ground(__file__, 1, 1, source_root=os.path.dirname(__file__) or ".") is not None
        except Exception:
            grounding_ok = False
        return {
            "status": "ok",
            "base_dir": base_dir,
            "evidencespine_version": str(_package_version or ""),
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
            "executable": exe,
            "executable_ok": bool(exe_ok),
            "mcp_available": bool(mcp_available),
            "grounding_ok": bool(grounding_ok),
        }
    except Exception as exc:
        return {"status": "fail_open", "reason": str(exc)}
