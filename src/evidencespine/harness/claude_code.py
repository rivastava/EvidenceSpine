"""Claude Code plugin delivery (`.claude-plugin/plugin.json`).

Claude Code runs hook commands from the project directory. ``SessionStart``
stdout is injected into the conversation context (auto-recall); ``PreCompact``
stdout is injected into the compaction context and receives session JSON on
stdin (retain-through-compaction); ``Stop`` records the session end
(auto-retain).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from evidencespine import __version__

_HOOK_EVENTS = ("SessionStart", "PreCompact", "Stop")


def hook_command(
    executable: str,
    action: str,
    *,
    base_dir: Optional[str] = None,
) -> str:
    parts = [executable, "harness", "claude-code", action]
    if base_dir:
        parts += ["--base-dir", base_dir]
    return " ".join(parts)


def build_manifest(
    *,
    executable: str = "evidencespine",
    base_dir: Optional[str] = None,
    include_precompact: bool = True,
) -> Dict[str, Any]:
    """Build a Claude Code plugin manifest wiring the delivery hooks."""
    hooks: Dict[str, Any] = {
        "SessionStart": {
            "hooks": [
                {"type": "command", "command": hook_command(executable, "session-start", base_dir=base_dir)}
            ]
        },
        "Stop": {
            "hooks": [
                {
                    "type": "command",
                    "command": hook_command(executable, "session-stop", base_dir=base_dir),
                }
            ]
        },
    }
    if include_precompact:
        hooks["PreCompact"] = {
            "hooks": [
                {
                    "type": "command",
                    "command": hook_command(executable, "precompact", base_dir=base_dir),
                }
            ]
        }
    return {
        "name": "evidencespine",
        "description": (
            "Evidence-bound memory delivery: auto-recall a bounded context brief at "
            "session start, retain state through compaction, and record session end."
        ),
        "version": __version__,
        "author": "EvidenceSpine Contributors",
        "license": "Apache-2.0",
        "hooks": hooks,
    }
