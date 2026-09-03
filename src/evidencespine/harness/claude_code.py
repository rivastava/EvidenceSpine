"""Claude Code plugin delivery (`.claude-plugin/` + `.claude/settings.json`).

Claude Code runs hook commands from the project directory.

Spec (code.claude.com/docs/en/{hooks,plugins-reference}):

* Settings: ``~/.claude/settings.json`` (user) | ``.claude/settings.json``
  (project) | ``.claude/settings.local.json`` (gitignored) | plugin
  ``hooks/hooks.json``. Shape is event -> LIST of matcher groups:
  ``{"hooks": {"PostToolUse": [{"matcher": "Edit|Write", "hooks": [...] }]}}``.
* Plugin ``hooks/hooks.json`` MUST be wrapped ``{"hooks": {...}}``; bare
  ``{"PostToolUse": [...]}`` passes ``plugin validate`` but fails at runtime
  (``expected record at path "hooks"``).
* Events: ``SessionStart`` (plaintext stdout -> context) with matchers
  ``startup|resume|clear|compact|fork``; ``SessionEnd`` is session terminate
  (non-blockable, shared 1.5s budget) — distinct from per-turn ``Stop``;
  ``PreCompact`` MUST return
  ``{"hookSpecificOutput": {"hookEventName": "PreCompact",
  "additionalContext": "..."}}`` or stdout is ignored.
* Vars: ``${CLAUDE_PLUGIN_ROOT} ${CLAUDE_PROJECT_DIR} ${CLAUDE_ENV_FILE}``.
* ``timeout`` seconds; default 600 for command hooks.

``SessionStart`` stdout is injected into context (auto-recall); ``PreCompact``
envelope is injected into compaction context (retain-through-compaction);
``SessionEnd`` records session end (auto-retain).
"""

from __future__ import annotations

import json
import shlex
from typing import Any, Dict, List, Optional

from evidencespine import __version__

_HOOK_EVENTS = ("SessionStart", "SessionEnd", "PreCompact")

HOOKS_PATH = "./hooks/hooks.json"
MCP_PATH = "./.mcp.json"


def hook_command(
    executable: str,
    action: str,
    *,
    base_dir: Optional[str] = None,
) -> str:
    try:
        parts = shlex.split(executable or "")
    except Exception:
        parts = []
    if not parts:
        parts = ["evidencespine"]
    parts += ["harness", "claude-code", action]
    if base_dir:
        parts += ["--base-dir", base_dir]
    return shlex.join(parts)


def _hook_entry(
    executable: str,
    action: str,
    *,
    base_dir: Optional[str] = None,
    matcher: str = "",
    timeout: int = 30,
) -> Dict[str, Any]:
    return {
        "matcher": matcher,
        "hooks": [
            {
                "type": "command",
                "command": hook_command(executable, action, base_dir=base_dir),
                "timeout": timeout,
            }
        ],
    }


def build_hooks_config(
    *,
    executable: str = "evidencespine",
    base_dir: Optional[str] = None,
    include_precompact: bool = True,
) -> Dict[str, Any]:
    """Build the ``{"hooks": {...}}`` mapping (settings.json / hooks/hooks.json)."""
    hooks: Dict[str, List[Dict[str, Any]]] = {
        "SessionStart": [_hook_entry(executable, "session-start", base_dir=base_dir, matcher="startup|resume|clear|compact|fork")],
        "SessionEnd": [_hook_entry(executable, "session-stop", base_dir=base_dir, timeout=10)],
    }
    if include_precompact:
        hooks["PreCompact"] = [_hook_entry(executable, "precompact", base_dir=base_dir, matcher="manual|auto")]
    return {"hooks": hooks}


def build_manifest(
    *,
    executable: str = "evidencespine",
    base_dir: Optional[str] = None,
    include_precompact: bool = True,
) -> Dict[str, Any]:
    """Build a Claude Code plugin manifest wiring the delivery hooks.

    Per code.claude.com/docs/en/plugins-reference: ``author`` is an object,
    component paths are relative to the plugin root (the directory containing
    ``.claude-plugin/``), and ``.mcp.json`` at the plugin root is the default
    MCP location (also Claude Code's project MCP scope, so GUI + CLI share it).
    """
    return {
        "name": "evidencespine",
        "description": (
            "Evidence-bound memory delivery: auto-recall a bounded context brief at "
            "session start, retain state through compaction, and record session end."
        ),
        "version": __version__,
        "author": {"name": "EvidenceSpine Contributors"},
        "license": "Apache-2.0",
        "hooks": HOOKS_PATH,
        "mcpServers": MCP_PATH,
    }


def build_hooks_json(
    *,
    executable: str = "evidencespine",
    base_dir: Optional[str] = None,
    include_precompact: bool = True,
) -> Dict[str, Any]:
    """Build the ``hooks/hooks.json`` payload (wrapped ``{"hooks": {...}}``).

    Written at ``<plugin-root>/hooks/hooks.json`` (the spec default); only
    ``plugin.json`` lives inside ``.claude-plugin/``.
    """
    return build_hooks_config(executable=executable, base_dir=base_dir, include_precompact=include_precompact)


def _split_exe(executable: str) -> List[str]:
    import shlex

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
    """Build the plugin-root ``.mcp.json`` payload (Claude project MCP scope)."""
    argv = _split_exe(executable)
    return {
        "mcpServers": {
            "evidencespine": {
                "command": argv[0],
                "args": [*argv[1:], "mcp", "--base-dir", base_dir],
            }
        }
    }


def merge_mcp_json(existing: Any, *, executable: str = "evidencespine", base_dir: str = ".evidencespine") -> Dict[str, Any]:
    """Merge the spine entry into an existing ``.mcp.json`` (non-destructive)."""
    payload: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    servers = payload.get("mcpServers")
    servers = dict(servers) if isinstance(servers, dict) else {}
    fresh = build_mcp_json(executable=executable, base_dir=base_dir)["mcpServers"]["evidencespine"]
    servers["evidencespine"] = fresh
    payload["mcpServers"] = servers
    return payload


def precompact_envelope(note: str) -> str:
    """Wrap a retain-through-compaction note in the PreCompact envelope."""
    return json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreCompact",
                "additionalContext": str(note or ""),
            }
        },
        ensure_ascii=True,
    )
