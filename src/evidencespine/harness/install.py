"""Harness delivery installation, verification and debugging.

Installs the EvidenceSpine delivery layer into coding-agent harnesses:

* claude-code   skills-directory plugin at <target>/.claude/skills/evidencespine/
  (manifest, hooks/hooks.json, .mcp.json, skills/evidencespine/SKILL.md,
  README.md) + .claude/rules/evidencespine.md. Settings hooks are opt-in
  (--with-settings-hooks) because plugin + settings copies both run.
* opencode      .opencode/plugins/evidencespine.ts (V2 plugin: MCP registration,
  context + compaction recall, tool-after capture, event lifecycle).
* cursor        .cursor/mcp.json + .cursor/rules/evidencespine.mdc +
  .cursor/permissions.json + .cursor/hooks.json (all merge-preserving).
* vscode        .vscode/mcp.json legacy Local-harness shape (deprecated; kept
  for compatibility), vscode-copilot writes the portable .mcp.json shape.
* codex         .codex/config.toml (safe TOML merge: no duplicate [features],
  HTTP MCP via url/bearer) + .codex/hooks.json (append-merged, deduped).
* agents-md     managed HTML-comment block merge into AGENTS.md.
* dsh/generic/instructions  lazy dispatch to the optional modules
  evidencespine.harness.{dsh,generic,skills}; absent modules are reported as
  {"status": "skipped", "reason": "not_installed"} instead of crashing.
* git           post-commit/post-merge hooks (delegated to harness.git).

Non-destructive by construction: JSON documents are parsed first; an existing
unparseable file is never overwritten (a .bak is left and the result reports
"partial"), hooks are appended to existing event groups (never replaced), and
re-running an install is idempotent.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from evidencespine import __version__ as _package_version
from evidencespine.harness.hooks import build_runtime
from evidencespine.threads import resolve_thread_id

#: Recognized --harness targets (kept in sync with cli.py choices).
KNOWN_HARNESSES = (
    "claude-code",
    "opencode",
    "cursor",
    "vscode",
    "vscode-copilot",
    "codex",
    "git",
    "agents-md",
    "dsh",
    "generic",
    "instructions",
    "all",
)

#: Optional harness modules delivered by parallel work streams. Absence is a
#: skip, never a crash (lazy import at dispatch time).
LAZY_HARNESS_MODULES: Dict[str, Tuple[str, str]] = {
    "dsh": ("evidencespine.harness.dsh", "install_dsh"),
    "generic": ("evidencespine.harness.generic", "install_generic"),
    "instructions": ("evidencespine.harness.skills", "install_instructions"),
}

_SKILL_MODULE = "evidencespine.harness.skills"

_MD_BEGIN = "<!-- evidencespine:begin -->"
_MD_END = "<!-- evidencespine:end -->"


def resolve_executable() -> str:
    """Resolve the evidencespine executable used by harness hooks.

    Prefers a console script on PATH; falls back to the running interpreter so
    plugins work in development without a global install.
    """
    found = shutil.which("evidencespine")
    if found:
        return found
    return f"{sys.executable} -m evidencespine.cli"


def split_executable(executable: str) -> List[str]:
    """Split a possibly multi-token executable into argv tokens.

    A real path containing spaces must stay one argv token; shlex.split alone
    would turn it into three. The heuristic: an existing path is one token, an
    already single token is returned as-is, and multi-token strings are only
    split when the first word resolves to a command/path.
    """
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
    first = argv[0]
    if shutil.which(first) is not None or os.path.exists(first):
        return argv
    return [text]


def executable_available(executable: str) -> bool:
    argv = split_executable(executable)
    if not argv:
        return False
    return shutil.which(argv[0]) is not None or os.path.exists(argv[0])


@dataclass
class JsonDocument:
    """Result of reading a JSON file, distinguishing missing from invalid."""

    path: Path
    exists: bool
    valid: bool
    value: Any = None
    backup: str = ""
    error: str = ""

    @property
    def status(self) -> str:
        if not self.exists:
            return "missing"
        return "ok" if self.valid else "invalid"

    @property
    def reason(self) -> str:
        return f"existing {self.path} is not valid JSON: {self.error}"


def _backup_file(path: Path) -> str:
    """Copy an unparseable file aside once; never clobber a previous backup."""
    backup = path.with_name(path.name + ".bak")
    try:
        if path.exists() and not backup.exists():
            shutil.copy2(path, backup)
            return str(backup)
    except Exception:
        return ""
    return str(backup) if backup.exists() else ""


def read_json_document(path: Path) -> JsonDocument:
    """Read a JSON file without ever destroying unparseable user content."""
    try:
        exists = path.exists()
    except Exception:
        exists = False
    if not exists:
        return JsonDocument(path=path, exists=False, valid=True, value=None)
    try:
        raw = path.read_text(encoding="utf-8")
        return JsonDocument(path=path, exists=True, valid=True, value=json.loads(raw))
    except Exception as exc:
        return JsonDocument(
            path=path,
            exists=True,
            valid=False,
            value=None,
            backup=_backup_file(path),
            error=str(exc),
        )


def _read_json(path: Path) -> Any:
    """Backwards-compatible read: returns the value, or None when absent/invalid.

    Use read_json_document() when the caller must distinguish missing from
    unparseable (installation merges) so invalid user files are never clobbered.
    """
    return read_json_document(path).value


def _write_json(path: Path, payload: Any, *, dry_run: bool = False, wrote: Optional[List[str]] = None) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if wrote is not None:
        wrote.append(str(path))


def _write_text(path: Path, text: str, *, dry_run: bool = False, wrote: Optional[List[str]] = None) -> None:
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if wrote is not None:
        wrote.append(str(path))


def _merge_mapping(existing: Any, key: str, entry_name: str, entry: Any) -> Dict[str, Any]:
    """Merge one entry into a top-level map key, preserving unrelated entries."""
    payload: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    mapping = payload.get(key)
    mapping = dict(mapping) if isinstance(mapping, dict) else {}
    mapping[entry_name] = entry
    payload[key] = mapping
    return payload


def _group_commands(group: Any) -> List[str]:
    commands: List[str] = []
    if not isinstance(group, dict):
        return commands
    handlers = group.get("hooks")
    if not isinstance(handlers, list):
        # Cursor (and other harnesses) use a flat group shape:
        # {"command": "...", "args": [...]} with no nested "hooks" list.
        handlers = [group]
    for handler in handlers if isinstance(handlers, list) else []:
        if isinstance(handler, dict):
            command = str(handler.get("command") or "")
            args = handler.get("args")
            if isinstance(args, list):
                command = " ".join([command, *[str(item) for item in args]])
            if command:
                commands.append(command)
    return commands


def _evidence_action(command: str) -> str:
    """Deliverable action keyword of an EvidenceSpine hook command, if any."""
    text = str(command or "")
    if "evidencespine" not in text:
        return ""
    for action in ("session-start", "session-stop", "precompact", "compaction", "session_stop"):
        if action in text:
            return action
    return "evidencespine"


def _merge_event_hooks(existing: Any, fresh: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Append fresh hook groups to an existing event->groups map.

    Both copies run (harness semantics); a group whose EvidenceSpine command
    already appears under the event is skipped so repeated installs do not
    duplicate deliveries. Unrelated user groups are preserved untouched.
    """
    merged: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    report: Dict[str, Any] = {"appended_events": [], "skipped_events": [], "warnings": []}
    known: Dict[str, set] = {}
    known_actions: Dict[str, set] = {}
    for event, groups in merged.items():
        commands: set = set()
        actions: set = set()
        for group in groups if isinstance(groups, list) else []:
            group_commands = _group_commands(group)
            commands.update(group_commands)
            actions.update(_evidence_action(command) for command in group_commands)
        known[event] = commands
        known_actions[event] = actions
    if not isinstance(fresh, dict):
        return merged, report
    for event, groups in fresh.items():
        current = merged.get(event)
        current_list = list(current) if isinstance(current, list) else []
        commands = known.setdefault(event, set())
        actions = known_actions.setdefault(event, set())
        added = 0
        for group in groups if isinstance(groups, list) else []:
            group_commands = [_cmd for _cmd in _group_commands(group) if "evidencespine" in _cmd]
            group_actions = {_evidence_action(_cmd) for _cmd in group_commands}
            if any(_cmd in commands for _cmd in group_commands) or (group_actions & actions):
                report["skipped_events"].append(event)
                continue
            current_list.append(group)
            commands.update(group_commands)
            actions.update(group_actions)
            added += 1
        if added:
            merged[event] = current_list
            report["appended_events"].append(event)
        elif event not in merged:
            report["warnings"].append(f"{event}: no hook groups appended")
    return merged, report


def _claude_dirs(target_dir: str, scope: str) -> Tuple[Path, Path, Path]:
    """Return (target, claude_dir, plugin_dir) for a Claude Code install."""
    target = Path(str(target_dir)).expanduser()
    if str(scope).lower() == "global":
        claude_dir = target if target.name == ".claude" else target / ".claude"
    else:
        claude_dir = target / ".claude"
    return target, claude_dir, claude_dir / "skills" / "evidencespine"


def _build_skill_markdown() -> str:
    """Use harness.skills when the optional module exists, else the local guide."""
    try:
        module = importlib.import_module(_SKILL_MODULE)
        for name in ("build_skill_markdown", "render_skill_markdown"):
            builder = getattr(module, name, None)
            if callable(builder):
                text = builder()
                if isinstance(text, str) and text.strip():
                    return text if text.endswith("\n") else text + "\n"
    except Exception:
        pass
    from evidencespine.harness.claude_code import build_skill_markdown as _local

    return _local()


def install_claude_code(
    *,
    target_dir: str,
    base_dir: str = ".evidencespine",
    executable: Optional[str] = None,
    include_precompact: bool = True,
    scope: str = "project",
    with_settings_hooks: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Install the EvidenceSpine skill/plugin into a Claude Code project."""
    from evidencespine.harness.claude_code import (
        build_hooks_json,
        build_manifest,
        build_plugin_readme,
        build_rules_markdown,
        merge_mcp_json,
        merge_settings_hooks,
        validate_plugin,
    )

    exe = executable or resolve_executable()
    target, claude_dir, plugin_dir = _claude_dirs(target_dir, scope)
    wrote: List[str] = []
    warnings: List[str] = []
    manifest_path = plugin_dir / ".claude-plugin" / "plugin.json"
    _write_json(manifest_path, build_manifest(executable=exe, base_dir=base_dir, include_precompact=include_precompact), dry_run=dry_run, wrote=wrote)

    hooks_payload = build_hooks_json(executable=exe, base_dir=base_dir, include_precompact=include_precompact)
    hooks_path = plugin_dir / "hooks" / "hooks.json"
    _write_json(hooks_path, hooks_payload, dry_run=dry_run, wrote=wrote)

    mcp_path = plugin_dir / ".mcp.json"
    mcp_doc = read_json_document(mcp_path)
    if mcp_doc.exists and not mcp_doc.valid:
        warnings.append(f"left existing {mcp_path} untouched ({mcp_doc.reason}); backup: {mcp_doc.backup or 'none'}")
    else:
        _write_json(mcp_path, merge_mcp_json(mcp_doc.value, executable=exe, base_dir=base_dir), dry_run=dry_run, wrote=wrote)

    skill_path = plugin_dir / "skills" / "evidencespine" / "SKILL.md"
    _write_text(skill_path, _build_skill_markdown(), dry_run=dry_run, wrote=wrote)
    readme_path = plugin_dir / "README.md"
    _write_text(readme_path, build_plugin_readme(), dry_run=dry_run, wrote=wrote)

    rules_path = claude_dir / "rules" / "evidencespine.md"
    _write_text(rules_path, build_rules_markdown(), dry_run=dry_run, wrote=wrote)

    settings_report: Dict[str, Any] = {}
    settings_path = claude_dir / "settings.json"
    if with_settings_hooks:
        settings_doc = read_json_document(settings_path)
        if settings_doc.exists and not settings_doc.valid:
            warnings.append(f"left existing {settings_path} untouched ({settings_doc.reason}); backup: {settings_doc.backup or 'none'}")
        else:
            merged = merge_settings_hooks(settings_doc.value, hooks_payload)
            settings_report = merged.pop("__report__", {}) if isinstance(merged, dict) else {}
            _write_json(settings_path, merged, dry_run=dry_run, wrote=wrote)
            warnings.extend(list(settings_report.get("warnings") or []))
    elif settings_path.exists():
        settings_report = {
            "skipped": True,
            "reason": "settings_hooks_not_requested",
            "note": "plugin hooks are installed; pass --with-settings-hooks only for settings-only setups",
        }

    validation: Dict[str, Any]
    if dry_run:
        validation = {"status": "skipped", "reason": "dry_run"}
    else:
        validation = validate_plugin(str(plugin_dir))

    status = "partial" if warnings else "ok"
    return {
        "status": status,
        "harness": "claude-code",
        "scope": scope,
        "path": str(plugin_dir),
        "plugin_dir": str(plugin_dir),
        "wrote": wrote,
        "warnings": warnings,
        "settings_hooks": settings_report,
        "validation": validation,
        "dry_run": bool(dry_run),
        "next_step": (
            "Restart Claude Code from the repository root and accept the workspace-trust "
            "dialog; the plugin loads as evidencespine@skills-dir. /reload-plugins rescans "
            "after edits. Project .mcp.json servers may need one approval prompt."
        ),
    }


def install_opencode(
    *,
    target_dir: str,
    base_dir: str = ".evidencespine",
    executable: Optional[str] = None,
    scope: str = "project",
    dry_run: bool = False,
) -> Dict[str, Any]:
    from evidencespine.harness.opencode import build_plugin_ts, mcp_command_argv

    exe = executable or resolve_executable()
    argv = mcp_command_argv(exe, base_dir)
    mcp_command = "[" + ", ".join(json.dumps(token, ensure_ascii=True) for token in argv) + "]"
    source = build_plugin_ts(executable=exe, base_dir=base_dir, mcp_command=mcp_command)
    if str(scope).lower() == "global":
        plugins_dir = Path(str(target_dir)).expanduser()
    else:
        plugins_dir = Path(str(target_dir)).expanduser() / ".opencode" / "plugins"
    path = plugins_dir / "evidencespine.ts"
    wrote: List[str] = []
    _write_text(path, source, dry_run=dry_run, wrote=wrote)
    return {
        "status": "ok",
        "harness": "opencode",
        "scope": scope,
        "path": str(path),
        "wrote": wrote,
        "dry_run": bool(dry_run),
        "next_step": "Restart opencode; the plugin loads from .opencode/plugins on startup.",
    }


def install_cursor(
    *,
    target_dir: str,
    base_dir: str = ".evidencespine",
    executable: Optional[str] = None,
    scope: str = "project",
    dry_run: bool = False,
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
    warnings: List[str] = []
    if str(scope).lower() == "global":
        cursor_dir = Path(str(target_dir)).expanduser()
    else:
        cursor_dir = Path(str(target_dir)).expanduser() / ".cursor"

    mcp_path = cursor_dir / "mcp.json"
    mcp_doc = read_json_document(mcp_path)
    if mcp_doc.exists and not mcp_doc.valid:
        warnings.append(f"left existing {mcp_path} untouched ({mcp_doc.reason}); backup: {mcp_doc.backup or 'none'}")
    else:
        _write_json(mcp_path, merge_mcp_json(mcp_doc.value, executable=exe, base_dir=base_dir), dry_run=dry_run, wrote=wrote)

    rules_path = cursor_dir / "rules" / "evidencespine.mdc"
    if dry_run or not rules_path.exists():
        _write_text(rules_path, build_rules_mdc(), dry_run=dry_run, wrote=wrote)

    perm_path = cursor_dir / "permissions.json"
    perm_doc = read_json_document(perm_path)
    if perm_doc.exists and not perm_doc.valid:
        warnings.append(f"left existing {perm_path} untouched ({perm_doc.reason}); backup: {perm_doc.backup or 'none'}")
    else:
        _write_json(perm_path, merge_permissions_json(perm_doc.value or build_permissions_json()), dry_run=dry_run, wrote=wrote)

    hooks_path = cursor_dir / "hooks.json"
    hooks_doc = read_json_document(hooks_path)
    if hooks_doc.exists and not hooks_doc.valid:
        warnings.append(f"left existing {hooks_path} untouched ({hooks_doc.reason}); backup: {hooks_doc.backup or 'none'}")
    else:
        existing_hooks = hooks_doc.value if isinstance(hooks_doc.value, dict) else None
        fresh_hooks = build_hooks_json(executable=exe, base_dir=base_dir)
        if existing_hooks is None:
            payload = fresh_hooks
        else:
            merged_hooks, _report = _merge_event_hooks(existing_hooks.get("hooks"), fresh_hooks.get("hooks"))
            payload = dict(existing_hooks)
            payload["hooks"] = merged_hooks
            payload.setdefault("version", 1)
        _write_json(hooks_path, payload, dry_run=dry_run, wrote=wrote)

    return {
        "status": "partial" if warnings else "ok",
        "harness": "cursor",
        "scope": scope,
        "path": str(mcp_path),
        "wrote": wrote,
        "warnings": warnings,
        "dry_run": bool(dry_run),
    }


def install_vscode(
    *,
    target_dir: str,
    base_dir: str = ".evidencespine",
    executable: Optional[str] = None,
    scope: str = "project",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Legacy VS Code Local-harness install (.vscode/mcp.json, key "servers")."""
    from evidencespine.harness.vscode import merge_mcp_json

    exe = executable or resolve_executable()
    if str(scope).lower() == "global":
        vscode_dir = Path(str(target_dir)).expanduser()
    else:
        vscode_dir = Path(str(target_dir)).expanduser() / ".vscode"
    path = vscode_dir / "mcp.json"
    wrote: List[str] = []
    warnings: List[str] = []
    doc = read_json_document(path)
    if doc.exists and not doc.valid:
        warnings.append(f"left existing {path} untouched ({doc.reason}); backup: {doc.backup or 'none'}")
    else:
        _write_json(path, merge_mcp_json(doc.value, executable=exe, base_dir=base_dir), dry_run=dry_run, wrote=wrote)
    return {
        "status": "partial" if warnings else "ok",
        "harness": "vscode",
        "scope": scope,
        "path": str(path),
        "wrote": wrote,
        "warnings": warnings,
        "dry_run": bool(dry_run),
        "note": "legacy_deprecated_vscode_local_harness",
        "next_step": "New Copilot configs use --harness vscode-copilot (.mcp.json / ~/.copilot/mcp-config.json).",
    }


def install_vscode_copilot(
    *,
    target_dir: str,
    base_dir: str = ".evidencespine",
    executable: Optional[str] = None,
    scope: str = "project",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Portable VS Code/Copilot workspace MCP config (key mcpServers, .mcp.json)."""
    exe = executable or resolve_executable()
    argv = split_executable(exe)
    entry = {
        "type": "stdio",
        "command": argv[0],
        "args": [*argv[1:], "mcp", "--base-dir", base_dir],
        "tools": ["*"],
    }
    if str(scope).lower() == "global":
        copilot_home = Path(str(target_dir)).expanduser()
        if copilot_home.name != ".copilot":
            copilot_home = copilot_home / ".copilot"
        path = copilot_home / "mcp-config.json"
    else:
        path = Path(str(target_dir)).expanduser() / ".mcp.json"
    wrote: List[str] = []
    warnings: List[str] = []
    doc = read_json_document(path)
    if doc.exists and not doc.valid:
        warnings.append(f"left existing {path} untouched ({doc.reason}); backup: {doc.backup or 'none'}")
    else:
        payload = _merge_mapping(doc.value, "mcpServers", "evidencespine", entry)
        _write_json(path, payload, dry_run=dry_run, wrote=wrote)
    return {
        "status": "partial" if warnings else "ok",
        "harness": "vscode-copilot",
        "scope": scope,
        "path": str(path),
        "wrote": wrote,
        "warnings": warnings,
        "dry_run": bool(dry_run),
        "next_step": "VS Code Copilot: open the workspace config (.mcp.json) and start the server from the MCP view.",
    }


def install_codex(
    *,
    target_dir: str,
    base_dir: str = ".evidencespine",
    executable: Optional[str] = None,
    scope: str = "project",
    mcp_url: Optional[str] = None,
    bearer_token_env_var: Optional[str] = None,
    http_headers: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Install Codex MCP config (safe TOML merge) and lifecycle hooks."""
    from evidencespine.harness.codex import build_hooks_json, merge_config_toml, merge_hooks_json, render_config_toml

    exe = executable or resolve_executable()
    if mcp_url and not bearer_token_env_var:
        bearer_token_env_var = "EVIDENCESPINE_MCP_AUTH_TOKEN"
    wrote: List[str] = []
    warnings: List[str] = []
    if str(scope).lower() == "global":
        codex_dir = Path(str(target_dir)).expanduser()
    else:
        codex_dir = Path(str(target_dir)).expanduser() / ".codex"

    config_path = codex_dir / "config.toml"
    existing_text = ""
    if config_path.exists():
        try:
            existing_text = config_path.read_text(encoding="utf-8")
        except Exception as exc:
            warnings.append(f"could not read {config_path}: {exc}")
    if str(existing_text).strip():
        merged = merge_config_toml(
            existing_text,
            executable=exe,
            base_dir=base_dir,
            mcp_url=mcp_url,
            bearer_token_env_var=bearer_token_env_var,
            http_headers=http_headers,
        )
        warnings.extend(list(merged.get("warnings") or []))
        if merged.get("status") == "ok":
            if merged.get("changed"):
                _write_text(config_path, str(merged.get("text") or ""), dry_run=dry_run, wrote=wrote)
            else:
                wrote.append(str(config_path))
        else:
            backup = _backup_file(config_path)
            warnings.append(f"left existing {config_path} untouched; backup: {backup or 'none'}")
    else:
        text = render_config_toml(
            exe,
            base_dir,
            mcp_url=mcp_url,
            bearer_token_env_var=bearer_token_env_var,
            http_headers=http_headers,
        )
        _write_text(config_path, text, dry_run=dry_run, wrote=wrote)

    hooks_path = codex_dir / "hooks.json"
    fresh_hooks = build_hooks_json(executable=exe, base_dir=base_dir)
    hooks_doc = read_json_document(hooks_path)
    if hooks_doc.exists and not hooks_doc.valid:
        warnings.append(f"left existing {hooks_path} untouched ({hooks_doc.reason}); backup: {hooks_doc.backup or 'none'}")
    else:
        if isinstance(hooks_doc.value, dict):
            merged_hooks = merge_hooks_json(hooks_doc.value, fresh_hooks)
            payload = merged_hooks["payload"]
            warnings.extend(list(merged_hooks["report"].get("warnings") or []))
        else:
            payload = fresh_hooks
        _write_json(hooks_path, payload, dry_run=dry_run, wrote=wrote)

    return {
        "status": "partial" if warnings else "ok",
        "harness": "codex",
        "scope": scope,
        "path": str(config_path),
        "wrote": wrote,
        "warnings": warnings,
        "transport": "http" if mcp_url else "stdio",
        "dry_run": bool(dry_run),
        "note": "project hooks require per-hash trust review: run /hooks in Codex (or --dangerously-bypass-hook-trust).",
    }


def build_managed_agents_block() -> str:
    """AGENTS.md managed block (markers make the merge idempotent)."""
    from evidencespine.usage import usage_guide_markdown

    return f"{_MD_BEGIN}\n{usage_guide_markdown().strip()}\n{_MD_END}\n"


def merge_agents_md(existing: str) -> str:
    """Replace or append the managed EvidenceSpine block, preserving user text."""
    text = str(existing or "")
    block = build_managed_agents_block()
    start = text.find(_MD_BEGIN)
    end = text.find(_MD_END, start + len(_MD_BEGIN)) if start >= 0 else -1
    if start >= 0 and end >= 0:
        return text[:start] + block + text[end + len(_MD_END):].lstrip("\n")
    if not text.strip():
        return block
    if not text.endswith("\n"):
        text += "\n"
    return text + "\n" + block


def _canonical_agents_merge(existing: str) -> str:
    """Prefer harness.skills.merge_agents_text when the module is available."""
    try:
        module = importlib.import_module(_SKILL_MODULE)
        merger = getattr(module, "merge_agents_text", None)
        if callable(merger):
            text = merger(existing)
            if isinstance(text, str) and _MD_BEGIN in text and _MD_END in text:
                return text
    except Exception:
        pass
    return merge_agents_md(existing)


def install_agents_md(*, target_dir: str, scope: str = "project", dry_run: bool = False, **_: Any) -> Dict[str, Any]:
    """Write/merge the AGENTS.md managed block (never clobbers user content)."""
    path = Path(str(target_dir)).expanduser() / "AGENTS.md"
    wrote: List[str] = []
    try:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
    except Exception:
        existing = ""
    merged = _canonical_agents_merge(existing)
    changed = merged != existing or not path.exists()
    if changed:
        _write_text(path, merged, dry_run=dry_run, wrote=wrote)
    return {
        "status": "ok",
        "harness": "agents-md",
        "path": str(path),
        "wrote": wrote,
        "changed": changed,
        "note": "managed block merged" if changed else "already current",
        "dry_run": bool(dry_run),
    }


# ---------------------------------------------------------------------------
# Client MCP configuration matrix (print-config + doctor, never writes)
# ---------------------------------------------------------------------------

#: Per-client MCP discovery paths, shapes and verification commands.
#: Sourced from vendor documentation fetched 2026-11-20.
CLIENT_MATRIX: Dict[str, Dict[str, str]] = {
    "claude-code": {
        "project": ".mcp.json",
        "global": "~/.claude.json",
        "shape": "json-mcp",
        "verify": "claude mcp list",
        "grade": "C",
    },
    "claude-desktop": {
        "global": "~/Library/Application Support/Claude/claude_desktop_config.json",
        "shape": "json-mcp",
        "verify": "restart Claude Desktop; Settings > Connectors",
        "grade": "C (macOS path)",
        "note": "Claude Desktop remote servers are UI-only; this file is stdio-only.",
    },
    "codex": {
        "project": ".codex/config.toml",
        "global": "~/.codex/config.toml",
        "shape": "toml",
        "verify": "codex mcp list",
        "grade": "C",
    },
    "cursor": {
        "project": ".cursor/mcp.json",
        "global": "~/.cursor/mcp.json",
        "shape": "json-mcp",
        "verify": "Cursor: Settings > MCP",
        "grade": "C",
    },
    "vscode": {
        "project": ".vscode/mcp.json",
        "shape": "json-servers",
        "verify": "VS Code: MCP view",
        "grade": "C (deprecated Local-harness shape)",
        "note": "Deprecated editor-managed format; prefer vscode-copilot.",
    },
    "vscode-copilot": {
        "project": ".mcp.json",
        "global": "~/.copilot/mcp-config.json",
        "shape": "json-copilot",
        "verify": "VS Code Copilot: MCP view",
        "grade": "C",
    },
    "windsurf": {
        "global": "~/.config/devin/mcp_config.json",
        "shape": "json-mcp",
        "verify": "Windsurf/Devin: Cascade MCP settings",
        "grade": "C",
        "note": "Legacy discovery file: ~/.codeium/windsurf/mcp_config.json.",
    },
    "zed": {
        "project": ".zed/settings.json",
        "global": "~/.config/zed/settings.json",
        "shape": "json-context-servers",
        "verify": "Zed: agent panel > settings",
        "grade": "C path [I]",
    },
    "gemini": {
        "project": ".gemini/settings.json",
        "global": "~/.gemini/settings.json",
        "shape": "json-mcp",
        "verify": "gemini mcp list",
        "grade": "C",
    },
    "antigravity": {
        "project": ".agents/mcp_config.json",
        "global": "~/.gemini/config/mcp_config.json",
        "shape": "json-mcp",
        "verify": "Antigravity: MCP panel",
        "grade": "C",
    },
    "cline": {
        "global": "~/.cline/data/settings/cline_mcp_settings.json",
        "shape": "json-mcp",
        "verify": "cline mcp list",
        "grade": "C",
    },
    "roo": {
        "project": ".roo/mcp.json",
        "shape": "json-mcp",
        "verify": "Roo Code: MCP servers tab",
        "grade": "C project path",
    },
    "continue": {
        "project": ".continue/mcpServers/evidencespine.json",
        "global": "~/.continue/config.yaml",
        "shape": "json-mcp",
        "verify": "Continue: MCP tools panel",
        "grade": "C",
    },
    "junie": {
        "project": ".junie/mcp/mcp.json",
        "global": "~/.junie/mcp/mcp.json",
        "shape": "json-mcp",
        "verify": "junie mcp list",
        "grade": "C",
    },
    "warp": {
        "project": ".warp/.mcp.json",
        "global": "~/.warp/.mcp.json",
        "shape": "json-mcp",
        "verify": "Warp: MCP settings",
        "grade": "C",
    },
    "amp": {
        "project": ".amp/settings.json",
        "global": "~/.config/amp/settings.json",
        "shape": "json-amp",
        "verify": "amp mcp list",
        "grade": "C",
    },
    "jetbrains": {
        "path": "UI: Settings | Tools | AI Assistant | Model Context Protocol (MCP)",
        "shape": "json-mcp",
        "verify": "JetBrains AI Assistant MCP settings",
        "grade": "C (UI-managed, no documented file)",
    },
    "opencode": {
        "project": "opencode.json",
        "global": "~/.config/opencode/opencode.json",
        "shape": "json-opencode",
        "verify": "opencode mcp list",
        "grade": "C",
    },
    "dsh": {
        "global": "~/.dsh/cordis.patch.yml",
        "project": "~/.dsh/profiles/<name>/cordis.patch.yml",
        "shape": "yaml-dsh",
        "verify": "dsh --dump-config --profile headless",
        "grade": "C",
    },
    "generic": {
        "project": "evidencespine.mcp.json",
        "global": "~/.agents/.mcp.json",
        "shape": "json-mcp",
        "verify": "python3 -m json.tool evidencespine.mcp.json",
        "grade": "I",
    },
}

#: Print-only clients surfaced by harness debug/doctor.
PRINT_ONLY_CLIENTS = tuple(
    name
    for name in CLIENT_MATRIX
    if name
    in {
        "windsurf",
        "zed",
        "gemini",
        "antigravity",
        "cline",
        "roo",
        "continue",
        "junie",
        "warp",
        "amp",
        "jetbrains",
        "claude-desktop",
    }
)

_CLIENT_ALIASES = {
    "claude": "claude-code",
    "claude_code": "claude-code",
    "copilot": "vscode-copilot",
    "vscode_copilot": "vscode-copilot",
    "devin": "windsurf",
    "gemini-cli": "gemini",
    "gemini_cli": "gemini",
}


def normalize_client(client: str) -> str:
    name = str(client or "").strip().lower().replace(" ", "-")
    return _CLIENT_ALIASES.get(name, name)


def _expand_path(raw: str, project_dir: str) -> str:
    if not raw:
        return ""
    if raw.startswith("~"):
        return os.path.expanduser(raw)
    if raw.startswith("UI:"):
        return raw
    return raw if os.path.isabs(raw) else os.path.join(os.path.abspath(project_dir), raw)


def client_paths(client: str, *, project_dir: str = ".") -> Dict[str, str]:
    """Project/global discovery paths for a client (unexpanded placeholders kept)."""
    entry = CLIENT_MATRIX.get(client)
    if not entry:
        return {}
    paths: Dict[str, str] = {}
    if entry.get("project"):
        paths["project"] = _expand_path(entry["project"], project_dir)
    if entry.get("global"):
        paths["global"] = _expand_path(entry["global"], project_dir)
    if entry.get("path"):
        paths["path"] = entry["path"]
    return paths


def _stdio_server_entry(executable: str, base_dir: str) -> Dict[str, Any]:
    argv = split_executable(executable)
    return {"command": argv[0], "args": [*argv[1:], "mcp", "--base-dir", base_dir]}


def _http_server_entry(mcp_url: str) -> Dict[str, Any]:
    return {
        "url": str(mcp_url),
        "headers": {"Authorization": "Bearer ${EVIDENCESPINE_MCP_AUTH_TOKEN}"},
    }


def _snippet_for(client: str, *, executable: str, base_dir: str, transport: str, mcp_url: str) -> Any:
    shape = CLIENT_MATRIX[client]["shape"]
    if shape == "toml":
        from evidencespine.harness.codex import render_http_mcp_table_toml, render_mcp_table_toml

        if transport == "http":
            return render_http_mcp_table_toml(
                mcp_url,
                bearer_token_env_var="EVIDENCESPINE_MCP_AUTH_TOKEN",
            )
        return render_mcp_table_toml(executable, base_dir)
    if shape == "yaml-dsh":
        entry = _stdio_server_entry(executable, base_dir)
        if transport == "http":
            config_lines = [
                "        transport: streamable-http",
                f"        url: {json.dumps(mcp_url)}",
                "        headers:",
                "          Authorization: \'Bearer \' + process.env.EVIDENCESPINE_MCP_AUTH_TOKEN",
            ]
        else:
            args = ", ".join(json.dumps(str(token)) for token in entry["args"])
            config_lines = [
                "        transport: stdio",
                f"        command: {json.dumps(entry['command'])}",
                f"        args: [{args}]",
            ]
        return "\n".join(
            [
                "- insert:",
                "    - id: evidencespine-mcp",
                "      name: '@deepseek-ai/dsh-mcp-client'",
                "      config:",
                "        serverName: evidencespine",
                *config_lines,
                "        failOnStartupError: true",
            ]
        )
    server = _http_server_entry(mcp_url) if transport == "http" else _stdio_server_entry(executable, base_dir)
    if shape == "json-servers":
        return {"servers": {"evidencespine": {"type": "stdio" if transport != "http" else "http", **server}}}
    if shape == "json-copilot":
        payload: Dict[str, Any] = {
            "mcpServers": {
                "evidencespine": {"type": "stdio" if transport != "http" else "http", **server}
            }
        }
        if transport != "http":
            payload["mcpServers"]["evidencespine"]["tools"] = ["*"]
        return payload
    if shape == "json-context-servers":
        return {"context_servers": {"evidencespine": {**server, "env": {}}}}
    if shape == "json-amp":
        return {"amp.mcpServers": {"evidencespine": server}}
    if shape == "json-opencode":
        if transport == "http":
            return {"mcp": {"servers": {"evidencespine": {"type": "remote", **server}}}}
        argv = [*split_executable(executable), "mcp", "--base-dir", base_dir]
        return {"mcp": {"servers": {"evidencespine": {"type": "local", "command": argv}}}}
    return {"mcpServers": {"evidencespine": server}}


def _render_snippet(client: str, *, executable: str, base_dir: str, transport: str, mcp_url: str) -> str:
    payload = _snippet_for(
        client,
        executable=executable,
        base_dir=base_dir,
        transport=transport,
        mcp_url=mcp_url,
    )
    if isinstance(payload, str):
        return payload if payload.endswith("\n") else payload + "\n"
    return json.dumps(payload, indent=2, ensure_ascii=True) + "\n"


def _lazy_generic_render(
    client: str,
    *,
    executable: str,
    base_dir: str,
    project_dir: str,
    scope: str,
    transport: str,
) -> Optional[Dict[str, Any]]:
    """Prefer harness.generic.render_client_config when that module exists."""
    try:
        module = importlib.import_module("evidencespine.harness.generic")
    except Exception:
        return None
    renderer = getattr(module, "render_client_config", None)
    if not callable(renderer):
        return None
    kwargs = {
        "client": client,
        "executable": executable,
        "base_dir": base_dir,
        "project_dir": project_dir,
        "scope": scope,
        "transport": transport,
    }
    try:
        result = renderer(**_filter_kwargs(renderer, kwargs))
    except Exception:
        return None
    if not isinstance(result, dict) or not result.get("snippet"):
        return None
    payload = dict(result)
    payload.setdefault("status", "ok")
    payload.setdefault("client", client)
    payload.setdefault("scope", scope)
    payload.setdefault("transport", transport)
    payload.setdefault("grade", CLIENT_MATRIX.get(client, {}).get("grade", "I"))
    payload["writes"] = False
    return payload


def render_print_config(
    *,
    client: str,
    scope: str = "project",
    transport: str = "stdio",
    executable: Optional[str] = None,
    base_dir: str = ".evidencespine",
    project_dir: str = ".",
) -> Dict[str, Any]:
    """Render a copy-paste MCP config for a client. Never writes files."""
    name = normalize_client(client)
    exe = executable or resolve_executable()
    transport_name = str(transport or "stdio").strip().lower()
    if transport_name not in {"stdio", "http"}:
        return {"status": "invalid", "reason": f"unsupported transport: {transport}", "supported": ["stdio", "http"]}
    if name == "all":
        clients = [
            render_print_config(
                client=known,
                scope=scope,
                transport=transport_name,
                executable=exe,
                base_dir=base_dir,
                project_dir=project_dir,
            )
            for known in sorted(CLIENT_MATRIX)
        ]
        return {"status": "ok", "clients": clients, "writes": False}
    if name not in CLIENT_MATRIX:
        return {
            "status": "invalid",
            "reason": f"unknown client: {client}",
            "known": sorted(CLIENT_MATRIX),
        }
    entry = CLIENT_MATRIX[name]
    mcp_url = "http://127.0.0.1:8000/mcp" if transport_name == "http" else ""
    if transport_name == "http":
        note = (
            "HTTP transport requires EVIDENCESPINE_MCP_AUTH_TOKEN and "
            "evidencespine mcp --transport streamable-http --host 127.0.0.1 --port 8000."
        )
    else:
        note = str(entry.get("note") or "")
    lazy = _lazy_generic_render(
        name,
        executable=exe,
        base_dir=base_dir,
        project_dir=project_dir,
        scope=scope,
        transport=transport_name,
    )
    # A generic renderer may only support stdio snippets; for --transport http
    # fall back to this matrix when the rendered result did not honour HTTP.
    lazy_transport = str(lazy.get("requested_transport") or "").strip().lower() if isinstance(lazy, dict) else ""
    if lazy is not None and (
        transport_name != "http" or lazy_transport == "http" or mcp_url in str(lazy.get("snippet", ""))
    ):
        lazy.setdefault("note", note)
        return lazy
    paths = client_paths(name, project_dir=project_dir)
    target = paths.get(scope) or paths.get("global") or paths.get("project") or paths.get("path") or ""
    return {
        "status": "ok",
        "client": name,
        "scope": scope,
        "transport": transport_name,
        "path": target,
        "snippet": _render_snippet(
            name,
            executable=exe,
            base_dir=base_dir,
            transport=transport_name,
            mcp_url=mcp_url,
        ),
        "verify": entry.get("verify", ""),
        "note": note,
        "grade": entry.get("grade", "C"),
        "writes": False,
    }


def _filter_kwargs(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Pass only kwargs a callable accepts (forward-compatible lazy dispatch)."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return dict(kwargs)
    parameters = signature.parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in parameters}


def _materialize_skill(
    harness_name: str,
    *,
    target_dir: str,
    base_dir: str,
    scope: str,
    dry_run: bool,
) -> Optional[Dict[str, Any]]:
    """Call harness.skills.materialize for dsh/generic when the module exists."""
    try:
        module = importlib.import_module(_SKILL_MODULE)
    except Exception:
        return None
    materialize = getattr(module, "materialize", None)
    if not callable(materialize):
        return None
    kwargs = {
        "target_dir": target_dir,
        "base_dir": base_dir,
        "scope": scope,
        "harness": harness_name,
        "harnesses": [harness_name],
        "dry_run": dry_run,
    }
    try:
        result = materialize(**_filter_kwargs(materialize, kwargs))
    except Exception as exc:
        return {"status": "skipped", "reason": str(exc)}
    if isinstance(result, dict):
        return result
    return {"status": "ok"}


def _call_lazy_install(
    name: str,
    *,
    target_dir: str,
    base_dir: str,
    executable: Optional[str],
    scope: str,
    dry_run: bool,
    mcp_url: Optional[str] = None,
    with_settings_hooks: bool = False,
    with_agents_md: bool = False,
) -> Dict[str, Any]:
    """Install through an optional harness module; absence is a clean skip."""
    module_name, func_name = LAZY_HARNESS_MODULES[name]
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": "not_installed",
            "harness": name,
            "module": module_name,
            "detail": str(exc),
        }
    installer = getattr(module, func_name, None)
    if not callable(installer):
        return {
            "status": "skipped",
            "reason": "not_installed",
            "harness": name,
            "module": module_name,
            "detail": f"{module_name}.{func_name} is missing",
        }
    exe = executable or resolve_executable()
    kwargs = {
        "target_dir": target_dir,
        "base_dir": base_dir,
        "scope": scope,
        "executable": exe,
        "dry_run": dry_run,
        "mcp_url": mcp_url,
        "with_settings_hooks": with_settings_hooks,
        "with_agents_md": with_agents_md,
    }
    try:
        result = installer(**_filter_kwargs(installer, kwargs))
    except Exception as exc:
        return {"status": "fail_open", "harness": name, "reason": str(exc)}
    payload: Dict[str, Any] = dict(result) if isinstance(result, dict) else {"status": "ok", "result": result}
    payload.setdefault("harness", name)
    if name in {"dsh", "generic"}:
        skill = _materialize_skill(
            name,
            target_dir=target_dir,
            base_dir=base_dir,
            scope=scope,
            dry_run=dry_run,
        )
        if skill is not None:
            payload["skill"] = skill
    return payload


def install_harness(
    *,
    harness: str,
    target_dir: str,
    base_dir: str = ".evidencespine",
    executable: Optional[str] = None,
    scope: str = "project",
    dry_run: bool = False,
    mcp_url: Optional[str] = None,
    with_settings_hooks: bool = False,
    with_agents_md: bool = False,
) -> Dict[str, Any]:
    """Dispatch an install to the requested harness (or every supported one)."""
    name = str(harness or "all").strip().lower().replace("_", "-")
    name = {"claude": "claude-code", "copilot": "vscode-copilot"}.get(name, name)
    common: Dict[str, Any] = {
        "target_dir": target_dir,
        "base_dir": base_dir,
        "executable": executable,
        "scope": scope,
        "dry_run": dry_run,
    }
    if name == "claude-code":
        return install_claude_code(**common, with_settings_hooks=with_settings_hooks)
    if name == "opencode":
        return install_opencode(**common)
    if name == "cursor":
        return install_cursor(**common)
    if name == "vscode":
        return install_vscode(**common)
    if name == "vscode-copilot":
        return install_vscode_copilot(**common)
    if name == "codex":
        return install_codex(**common, mcp_url=mcp_url)
    if name == "agents-md":
        return install_agents_md(**common)
    if name in LAZY_HARNESS_MODULES:
        return _call_lazy_install(
            name,
            base_dir=base_dir,
            target_dir=target_dir,
            executable=executable,
            scope=scope,
            dry_run=dry_run,
            mcp_url=mcp_url,
            with_settings_hooks=with_settings_hooks,
            with_agents_md=with_agents_md,
        )
    if name == "git":
        if dry_run:
            return {
                "status": "ok",
                "harness": "git",
                "dry_run": True,
                "hooks": ["post-commit", "post-merge"],
                "note": "dry-run: no hook files written",
            }
        from evidencespine.harness.git import install_git_hooks

        exe = executable or resolve_executable()
        abs_base = base_dir if os.path.isabs(base_dir) else os.path.abspath(os.path.join(target_dir, base_dir))
        inner = install_git_hooks(repo_dir=target_dir, executable=exe, base_dir=abs_base)
        return {"harness": "git", **inner}
    if name == "all":
        results: List[Dict[str, Any]] = [
            install_claude_code(**common, with_settings_hooks=with_settings_hooks),
            install_opencode(**common),
            install_cursor(**common),
            install_vscode(**common),
            install_vscode_copilot(**common),
            install_codex(**common, mcp_url=mcp_url),
            install_agents_md(**common),
        ]
        try:
            git_result = install_harness(harness="git", **common)
            if git_result.get("status") == "ok":
                results.append(git_result)
        except Exception:
            pass
        for lazy_name in ("dsh", "generic"):
            lazy_result = _call_lazy_install(
                lazy_name,
                base_dir=base_dir,
                target_dir=target_dir,
                executable=executable,
                scope=scope,
                dry_run=dry_run,
                mcp_url=mcp_url,
                with_settings_hooks=with_settings_hooks,
                with_agents_md=with_agents_md,
            )
            if lazy_result.get("status") != "skipped":
                results.append(lazy_result)
        wrote: List[str] = []
        warnings: List[str] = []
        for result in results:
            for item in result.get("wrote", []) or []:
                if item not in wrote:
                    wrote.append(item)
            for warning in result.get("warnings", []) or []:
                warnings.append(f"{result.get('harness')}: {warning}")
        failed = [item for item in results if item.get("status") not in {"ok", "partial", "skipped"}]
        status = "partial" if (warnings or failed) else "ok"
        return {
            "status": status,
            "harness": "all",
            "scope": scope,
            "wrote": wrote,
            "warnings": warnings,
            "details": results,
            "dry_run": bool(dry_run),
        }
    return {
        "status": "invalid",
        "reason": f"unknown harness: {harness!r}",
        "known": list(KNOWN_HARNESSES),
    }


# ---------------------------------------------------------------------------
# Debugging and doctor
# ---------------------------------------------------------------------------


def harness_files(target_dir: str = ".") -> Dict[str, Dict[str, Any]]:
    """Known delivery artifacts relative to a project directory."""
    target = Path(str(target_dir)).expanduser()
    claude_plugin = target / ".claude" / "skills" / "evidencespine"
    candidates: Dict[str, Path] = {
        "claude_code_manifest": claude_plugin / ".claude-plugin" / "plugin.json",
        "claude_code_hooks": claude_plugin / "hooks" / "hooks.json",
        "claude_code_mcp": claude_plugin / ".mcp.json",
        "claude_code_skill": claude_plugin / "skills" / "evidencespine" / "SKILL.md",
        "claude_code_rules": target / ".claude" / "rules" / "evidencespine.md",
        "claude_code_settings": target / ".claude" / "settings.json",
        "codex_config": target / ".codex" / "config.toml",
        "codex_hooks": target / ".codex" / "hooks.json",
        "cursor_mcp": target / ".cursor" / "mcp.json",
        "cursor_rules": target / ".cursor" / "rules" / "evidencespine.mdc",
        "cursor_hooks": target / ".cursor" / "hooks.json",
        "vscode_mcp": target / ".vscode" / "mcp.json",
        "vscode_copilot_mcp": target / ".mcp.json",
        "opencode_plugin": target / ".opencode" / "plugins" / "evidencespine.ts",
        "agents_md": target / "AGENTS.md",
        "git_post_commit": target / ".git" / "hooks" / "post-commit",
        "git_post_merge": target / ".git" / "hooks" / "post-merge",
        "generic_canonical": target / "evidencespine.mcp.json",
        "generic_agents_mcp": target / ".agents" / "mcp_config.json",
        "dsh_patch": Path(os.path.expanduser("~/.dsh/cordis.patch.yml")),
    }
    return {
        name: {"path": str(path), "present": path.exists(), "executable": os.access(path, os.X_OK)}
        for name, path in candidates.items()
    }


def _client_presence(target_dir: str) -> Dict[str, Dict[str, Any]]:
    """Print-only client config paths + local presence (no writes)."""
    presence: Dict[str, Dict[str, Any]] = {}
    for name in sorted(CLIENT_MATRIX):
        paths = client_paths(name, project_dir=target_dir)
        present = {
            scope: os.path.exists(path)
            for scope, path in paths.items()
            if path and not path.startswith("UI:")
        }
        presence[name] = {
            "paths": paths,
            "present": present,
            "shape": CLIENT_MATRIX[name]["shape"],
            "verify": CLIENT_MATRIX[name]["verify"],
            "grade": CLIENT_MATRIX[name]["grade"],
            "note": CLIENT_MATRIX[name].get("note", ""),
        }
    return presence


def debug_harness(
    *,
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    thread_id: Optional[str] = None,
    target_dir: str = ".",
) -> Dict[str, Any]:
    """Report harness delivery health: store, executable, MCP extra, grounding."""
    try:
        runtime = build_runtime(base_dir, storage_format)
        thread = resolve_thread_id(str(thread_id or ""), cwd=os.getcwd())
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
        files = harness_files(target_dir)
        return {
            "status": "ok",
            "base_dir": str(getattr(runtime.config, "base_dir", base_dir or ".evidencespine")),
            "target_dir": os.path.abspath(str(target_dir)),
            "thread_id": thread,
            "evidencespine_version": str(_package_version or ""),
            "storage_format": runtime.config.storage_format,
            "events_total": int(snapshot.get("events_total", 0)),
            "facts_total": int(snapshot.get("facts_total", 0)),
            "brief_ok": True,
            "brief_token_budget": int(brief.token_budget),
            "brief_sections": sum(
                1
                for key in (
                    "current_goal",
                    "locked_decisions",
                    "recent_verified_facts",
                    "active_risks",
                    "open_items",
                    "next_actions",
                )
                if brief.to_dict().get(key)
            ),
            "executable": exe,
            "executable_ok": bool(exe_ok),
            "mcp_available": bool(mcp_available),
            "grounding_ok": bool(grounding_ok),
            "harness_files": files,
            "installed": [name for name, row in files.items() if row.get("present")],
            "client_configs": _client_presence(target_dir),
            "print_config_hint": "evidencespine harness print-config --client <name> --json",
        }
    except Exception as exc:
        return {"status": "fail_open", "reason": str(exc)}


def _optional_module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _read_text_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8") if path.exists() else ""
    except Exception:
        return ""


def _detect_http_transport(target_dir: str) -> bool:
    """Best-effort detection of HTTP MCP config (for the auth reminder)."""
    target = Path(str(target_dir)).expanduser()
    for path in (
        target / ".mcp.json",
        target / ".cursor" / "mcp.json",
        target / ".vscode" / "mcp.json",
        target / ".codex" / "config.toml",
        target / ".claude" / "skills" / "evidencespine" / ".mcp.json",
        Path(os.path.expanduser("~/.codex/config.toml")),
        Path(os.path.expanduser("~/.cursor/mcp.json")),
    ):
        text = _read_text_or_empty(path)
        if not text:
            continue
        if '"url"' in text or "\nurl =" in text or "\nurl=" in text:
            return True
    return False


def doctor_harness(
    *,
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    thread_id: Optional[str] = None,
    target_dir: str = ".",
) -> Dict[str, Any]:
    """Validate installed harness files, JSON/TOML validity and the executable.

    Exit-code contract (CLI): "broken" -> nonzero, otherwise 0.
    """
    from evidencespine.harness.codex import parse_toml

    target = os.path.abspath(str(target_dir))
    checks: List[Dict[str, Any]] = []
    warnings: List[str] = []

    def add(name: str, status: str, detail: str = "", path: str = "") -> None:
        checks.append({"name": name, "status": status, "detail": detail, "path": path})
        if status == "broken":
            warnings.append(f"{name}: {detail}")

    exe = resolve_executable()
    if executable_available(exe):
        add("executable", "ok", exe)
    else:
        add("executable", "broken", f"not executable/on PATH: {exe}", exe)

    try:
        runtime = build_runtime(base_dir, storage_format)
        snapshot = runtime.snapshot()
        add("store", "ok", f"events={int(snapshot.get('events_total', 0))} facts={int(snapshot.get('facts_total', 0))}")
    except Exception as exc:
        add("store", "broken", str(exc))

    files = harness_files(target_dir)
    json_names = (
        "claude_code_manifest",
        "claude_code_hooks",
        "claude_code_mcp",
        "claude_code_settings",
        "codex_hooks",
        "cursor_mcp",
        "cursor_hooks",
        "vscode_mcp",
        "vscode_copilot_mcp",
        "generic_canonical",
        "generic_agents_mcp",
    )
    for name in json_names:
        row = files[name]
        path = Path(row["path"])
        if not row.get("present"):
            add(name, "absent", "not installed")
            continue
        doc = read_json_document(path)
        if doc.valid:
            add(name, "ok", "valid JSON", str(path))
        else:
            add(name, "broken", f"invalid JSON: {doc.error}", str(path))
            _backup_file(path)

    row = files["codex_config"]
    if not row.get("present"):
        add("codex_config", "absent", "not installed")
    else:
        text = _read_text_or_empty(Path(row["path"]))
        valid, error = parse_toml(text)
        if valid is False:
            add("codex_config", "broken", f"invalid TOML: {error}", row["path"])
        else:
            add("codex_config", "ok", "valid TOML" if valid else "unvalidated (tomllib unavailable)", row["path"])

    opencode_path = Path(files["opencode_plugin"]["path"])
    if files["opencode_plugin"].get("present"):
        source = _read_text_or_empty(opencode_path)
        if "export default plugin" in source and 'ctx.session.hook("compaction"' in source:
            add("opencode_plugin", "ok", "V2 plugin present", str(opencode_path))
        else:
            add("opencode_plugin", "broken", "plugin missing V2 export/compaction hook", str(opencode_path))
    else:
        add("opencode_plugin", "absent", "not installed")

    git_row = files["git_post_commit"]
    if git_row.get("present"):
        text = _read_text_or_empty(Path(git_row["path"]))
        if "evidencespine-git-hook" not in text:
            add("git_post_commit", "broken", "hook exists but is not EvidenceSpine-managed", git_row["path"])
        elif not git_row.get("executable"):
            add("git_post_commit", "broken", "hook is not executable", git_row["path"])
        else:
            add("git_post_commit", "ok", "managed + executable", git_row["path"])
    else:
        add("git_post_commit", "absent", "not installed")

    agents_row = files["agents_md"]
    if agents_row.get("present"):
        text = _read_text_or_empty(Path(agents_row["path"]))
        if _MD_BEGIN in text and _MD_END in text:
            add("agents_md", "ok", "managed block present", agents_row["path"])
        else:
            add("agents_md", "absent", "managed block not present", agents_row["path"])
    else:
        add("agents_md", "absent", "not installed")

    http_detected = _detect_http_transport(target_dir)
    auth_reminder = ""
    if http_detected:
        auth_reminder = (
            "HTTP MCP transport detected: set EVIDENCESPINE_MCP_AUTH_TOKEN before starting "
            "clients. evidencespine mcp refuses non-loopback binds without auth "
            "(escape hatch: EVIDENCESPINE_MCP_ALLOW_UNAUTHENTICATED=1)."
        )
    status = "broken" if any(check["status"] == "broken" for check in checks) else "ok"
    return {
        "status": status,
        "target_dir": target,
        "base_dir": str(base_dir or ".evidencespine"),
        "thread_id": resolve_thread_id(str(thread_id or ""), cwd=os.getcwd()),
        "executable": exe,
        "checks": checks,
        "warnings": warnings,
        "http_transport_detected": http_detected,
        "auth_reminder": auth_reminder,
        "harness_files": files,
        "client_configs": _client_presence(target_dir),
        "optional_modules": {
            name: _optional_module_available(module)
            for name, module in (
                ("dsh", "evidencespine.harness.dsh"),
                ("generic", "evidencespine.harness.generic"),
                ("skills", "evidencespine.harness.skills"),
            )
        },
        "next_step": "Run evidencespine harness install --harness <name> to repair missing artifacts.",
    }
