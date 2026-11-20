"""Claude Code plugin delivery (skills-directory plugin + project MCP).

Claude Code (2.1.x) does NOT scan a project's .claude-plugin/ or
.claude/plugins/ directories. The only project-local plugin form that
auto-loads with no install flag is a *skills-directory plugin*:

    <project>/.claude/skills/evidencespine/
        .claude-plugin/plugin.json     # manifest (only file inside .claude-plugin/)
        hooks/hooks.json               # wrapped {"hooks": {...}} (spec default)
        .mcp.json                      # plugin MCP server (spec default)
        skills/evidencespine/SKILL.md  # auto-surfacing skill
        README.md

It loads as evidencespine@skills-dir after the workspace-trust dialog
(code.claude.com/docs/en/plugins/loading). Launch from the repo root: parent
directories are not searched.

Spec facts this module implements (code.claude.com/docs/en/{hooks,plugins,
manifest-reference,mcp,memory}):

* hooks/hooks.json MUST be wrapped {"hooks": {...}}; the bare event map fails
  "claude plugin validate" (exit 1: hooks.json must have "hooks").
* Handler schema: {"type": "command", "command": ..., "args": [...],
  "timeout": seconds}. Exec form (command + args) avoids shell quoting and is
  the documented form when path placeholders are used; shell-form "command"
  alone runs through sh/Git Bash/PowerShell.
* ${CLAUDE_PROJECT_DIR} is set for hook commands and stdio MCP servers;
  project MCP config must write ${CLAUDE_PROJECT_DIR:-.} (unset without a
  default is used literally).
* SessionStart stdout (plain text) is injected into context. PreCompact stdout
  (exit 0, non-blocking) is used as instructions for the compaction summary;
  hookSpecificOutput.additionalContext is NOT in PreCompact's supported output
  list. SessionEnd is terminate-only (no decision control) with a ~1.5s default
  budget; a plugin hook's timeout does not raise that budget.
* Plugin hooks and settings hooks are NOT de-duplicated: the same handler in
  both places runs twice. Project install therefore writes plugin hooks only;
  --with-settings-hooks is an explicit opt-in for settings-only setups.
* .claude/rules/*.md load every session alongside CLAUDE.md; a plugin-root
  CLAUDE.md is never loaded. Standing policy belongs in .claude/rules/.

precompact_envelope is retained for backwards compatibility only: it now
returns the plain-text note (the envelope is the unsupported channel).
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Set

from evidencespine import __version__
from evidencespine.usage import usage_guide_markdown

PLUGIN_NAME = "evidencespine"

#: Legacy path constants (kept for API compatibility; defaults are implicit
#: in the plugin manifest, so the manifest no longer points at them).
HOOKS_PATH = "./hooks/hooks.json"
MCP_PATH = "./.mcp.json"

#: Placeholder Claude Code expands in hook/MCP configuration. The ":-." default
#: matters because project .mcp.json does not substitute an unset variable.
PROJECT_DIR_PLACEHOLDER = "${CLAUDE_PROJECT_DIR:-.}"

_HOOK_EVENTS = ("SessionStart", "SessionEnd", "PreCompact")


def split_executable(executable: str) -> List[str]:
    """Split an executable into argv tokens (spaced paths stay one token)."""
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
    if shutil.which(argv[0]) is not None or os.path.exists(argv[0]):
        return argv
    return [text]


def project_scoped_base_dir(base_dir: str) -> str:
    """Render --base-dir relative to the Claude project directory.

    Project-scope hooks run from the session cwd (which can move after /cd), so
    a relative store path must resolve against ${CLAUDE_PROJECT_DIR}.
    Absolute paths and already-placeholder values are preserved.
    """
    text = str(base_dir or ".evidencespine").strip() or ".evidencespine"
    if os.path.isabs(text) or text.startswith("$"):
        return text
    rel = text[2:] if text.startswith("./") else text
    return f"{PROJECT_DIR_PLACEHOLDER}/{rel}"


def hook_argv(executable: str, action: str, *, base_dir: Optional[str] = None) -> List[str]:
    """Exec-form argv for one Claude Code hook action (no shell quoting)."""
    argv = list(split_executable(executable))
    argv += ["harness", "claude-code", action]
    if base_dir:
        argv += ["--base-dir", project_scoped_base_dir(base_dir)]
    return argv


def hook_command(executable: str, action: str, *, base_dir: Optional[str] = None) -> str:
    """Display form of the hook command (compatibility helper)."""
    return shlex.join(hook_argv(executable, action, base_dir=base_dir))


def hook_entry(
    executable: str,
    action: str,
    *,
    base_dir: Optional[str] = None,
    matcher: str = "",
    timeout: int = 30,
) -> Dict[str, Any]:
    """One matcher group with a single exec-form command handler."""
    argv = hook_argv(executable, action, base_dir=base_dir)
    return {
        "matcher": matcher,
        "hooks": [
            {
                "type": "command",
                "command": argv[0],
                "args": argv[1:],
                "timeout": int(timeout),
            }
        ],
    }


def build_hooks_config(
    *,
    executable: str = "evidencespine",
    base_dir: Optional[str] = None,
    include_precompact: bool = True,
) -> Dict[str, Any]:
    """Build the {"hooks": {...}} mapping (plugin hooks/hooks.json)."""
    hooks: Dict[str, List[Dict[str, Any]]] = {
        "SessionStart": [
            hook_entry(
                executable,
                "session-start",
                base_dir=base_dir,
                matcher="startup|resume|clear|compact|fork",
                timeout=30,
            )
        ],
        # SessionEnd finishes inside Claude Code's ~1.5s terminate budget:
        # one fast ingest; a plugin timeout cannot raise the budget.
        "SessionEnd": [hook_entry(executable, "session-stop", base_dir=base_dir, timeout=5)],
    }
    if include_precompact:
        hooks["PreCompact"] = [
            hook_entry(executable, "precompact", base_dir=base_dir, matcher="manual|auto", timeout=30)
        ]
    return {"hooks": hooks}


def build_hooks_json(
    *,
    executable: str = "evidencespine",
    base_dir: Optional[str] = None,
    include_precompact: bool = True,
) -> Dict[str, Any]:
    """Build hooks/hooks.json: wrapped, with an optional description."""
    payload = build_hooks_config(
        executable=executable,
        base_dir=base_dir,
        include_precompact=include_precompact,
    )
    description = (
        "EvidenceSpine delivery: auto-recall at session start, retain through "
        "pre-compaction, fast auto-retain at session end."
    )
    return {"description": description, **payload}


def build_manifest(
    *,
    executable: str = "evidencespine",
    base_dir: Optional[str] = None,
    include_precompact: bool = True,
) -> Dict[str, Any]:
    """Build the plugin manifest.

    Components use their default locations (hooks/hooks.json, .mcp.json), so no
    pointer fields are needed. executable/base_dir are accepted for backwards
    compatibility (hooks carry the invocation).
    """
    return {
        "name": PLUGIN_NAME,
        "version": str(__version__),
        "description": (
            "Evidence-bound memory delivery: auto-recall a bounded context brief at "
            "session start, retain state through compaction, and record session end."
        ),
        "author": {"name": "EvidenceSpine Contributors"},
        "license": "Apache-2.0",
    }


def build_mcp_json(
    *,
    executable: str = "evidencespine",
    base_dir: str = ".evidencespine",
) -> Dict[str, Any]:
    """Build the plugin-root .mcp.json payload (stdio server)."""
    argv = split_executable(executable)
    return {
        "mcpServers": {
            "evidencespine": {
                "command": argv[0],
                "args": [*argv[1:], "mcp", "--base-dir", project_scoped_base_dir(base_dir)],
            }
        }
    }


def merge_mcp_json(
    existing: Any,
    *,
    executable: str = "evidencespine",
    base_dir: str = ".evidencespine",
) -> Dict[str, Any]:
    """Merge the spine entry into an existing .mcp.json (non-destructive)."""
    payload: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    servers = payload.get("mcpServers")
    servers = dict(servers) if isinstance(servers, dict) else {}
    fresh = build_mcp_json(executable=executable, base_dir=base_dir)["mcpServers"]["evidencespine"]
    servers["evidencespine"] = fresh
    payload["mcpServers"] = servers
    return payload


def build_skill_markdown() -> str:
    """Render skills/evidencespine/SKILL.md (frontmatter + usage guide)."""
    frontmatter = (
        "---\n"
        "name: evidencespine\n"
        "description: Use before asserting project state, marking a claim verified, "
        "importing/emitting a handoff, or re-checking cited evidence - consult and "
        "ground the EvidenceSpine side-car.\n"
        "---\n\n"
    )
    header = (
        "<!-- Generated by evidencespine harness install --harness claude-code. -->\n\n"
        "# EvidenceSpine\n\n"
        "Use the EvidenceSpine MCP tools (or the CLI) to recall the bounded project "
        "brief before asserting state, ground claims with checksummed excerpts, and "
        "hand off verified state at session end.\n\n"
    )
    return frontmatter + header + usage_guide_markdown() + "\n"


def build_rules_markdown() -> str:
    """Render .claude/rules/evidencespine.md (standing policy, short)."""
    return (
        "# EvidenceSpine\n\n"
        "This repository has an EvidenceSpine side-car: an evidence-bound memory "
        "store at .evidencespine/. The spine is a map - the repo is ground truth.\n\n"
        "## When to use it\n\n"
        "- Before asserting project state (tests green, feature shipped, this is "
        "fixed): build a brief first (build_brief MCP tool or evidencespine brief).\n"
        "- When claiming something is fixed/done/verified: ground it with a "
        "checksummed excerpt (ground, or evidencespine ingest --ground-ref "
        "path#L1-L2) and mark fact_state=verified. Ungrounded verified claims are "
        "stored as asserted.\n"
        "- After editing a file cited as evidence: run check_drift (or "
        "evidencespine drift-check) so stale claims are flagged.\n"
        "- When a test or gate verifies a claim: record provenance with verify_fact "
        "(or evidencespine verify).\n"
        "- On receiving a handoff packet: import_handoff it first; locked_decisions "
        "are binding and required_validations come first.\n"
        "- On role change or session end: emit_handoff so the successor resumes "
        "with verified state.\n"
        "- When claims conflict: query the contradictions view.\n\n"
        "## Hygiene\n\n"
        "- Ingest meaningful state, not trivia; attach evidence to facts that matter.\n"
        "- Locked decisions are binding unless superseded by new evidence; record "
        "the contradiction instead of acting against a verified fact.\n"
    )


def build_plugin_readme() -> str:
    """Render the plugin README.md (directory-profile validation needs one)."""
    return (
        "# EvidenceSpine Claude Code plugin\n\n"
        "EvidenceSpine is an evidence-bound memory side-car for coding agents. This "
        "plugin delivers memory without voluntary tool calls: a bounded project "
        "brief is injected at session start (SessionStart), working state is "
        "retained before context compaction (PreCompact), and session end is "
        "recorded as a reflection event (SessionEnd). The bundled MCP server "
        "exposes the full EvidenceSpine tool surface (ingest_event, build_brief, "
        "query_view, ground, check_drift, verify_fact, emit_handoff, "
        "import_handoff, reconcile, snapshot, prune) plus brief/view/state "
        "resources.\n\n"
        "## Layout\n\n"
        "- .claude-plugin/plugin.json - manifest.\n"
        "- hooks/hooks.json - lifecycle hook wiring.\n"
        "- .mcp.json - stdio MCP server registration.\n"
        "- skills/evidencespine/SKILL.md - auto-surfacing skill.\n\n"
        "## Verify\n\n"
        "Run claude plugin validate --strict . from this directory, accept the "
        "workspace trust dialog, and launch Claude Code from the repository root. "
        "Hooks fail open: a memory error never breaks the session.\n"
    )


def validate_plugin(plugin_dir: str, *, timeout: int = 60) -> Dict[str, Any]:
    """Run claude plugin validate --strict when the Claude CLI is present."""
    claude = shutil.which("claude")
    if not claude:
        return {"status": "skipped", "reason": "claude_cli_not_found"}
    try:
        proc = subprocess.run(
            [claude, "plugin", "validate", "--strict", str(plugin_dir)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}
    output = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    return {
        "status": "passed" if proc.returncode == 0 else "failed",
        "returncode": int(proc.returncode),
        "output": output[-2000:],
        "command": f"claude plugin validate --strict {plugin_dir}",
    }


def precompact_envelope(note: str) -> str:
    """Deprecated: plain-text stdout is the PreCompact channel.

    Claude Code uses PreCompact stdout as instructions for the compaction
    summary; hookSpecificOutput.additionalContext is not supported for that
    event. Kept as a compatibility shim returning the plain note.
    """
    return str(note or "")


def _group_signatures(group: Any) -> Set[str]:
    """Signature set for dedupe: every EvidenceSpine handler command/argv."""
    if not isinstance(group, dict):
        return set()
    signatures: Set[str] = set()
    handlers = group.get("hooks")
    for handler in handlers if isinstance(handlers, list) else []:
        if not isinstance(handler, dict):
            continue
        command = str(handler.get("command") or "")
        args = handler.get("args")
        if isinstance(args, list):
            command = " ".join([command, *[str(a) for a in args]])
        if "evidencespine" in command or "evidencespine.cli" in command:
            signatures.add(command.strip())
    return signatures


def merge_settings_hooks(
    existing: Any,
    hooks_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Append EvidenceSpine hook groups to a settings.json payload.

    Settings hooks merge with plugin hooks by running both, and settings entries
    merge across levels by appending. This appends one matcher group per event
    to any existing event list and never inserts a duplicate EvidenceSpine
    handler (dedupe by command signature). Returns the merged payload plus
    per-event reporting under the "__report__" key (callers strip it).
    """
    settings: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    hooks = settings.get("hooks")
    hooks = dict(hooks) if isinstance(hooks, dict) else {}
    source = hooks_payload.get("hooks") if isinstance(hooks_payload, dict) else {}
    source = source if isinstance(source, dict) else {}
    report: Dict[str, Any] = {"merged_events": [], "skipped_events": [], "warnings": []}
    for event, groups in source.items():
        current = hooks.get(event)
        current_list = list(current) if isinstance(current, list) else []
        added = 0
        for group in groups if isinstance(groups, list) else []:
            signatures = _group_signatures(group)
            already = any(
                signatures & _group_signatures(existing_group)
                for existing_group in current_list
                if isinstance(existing_group, dict)
            )
            if already:
                report["skipped_events"].append(event)
                continue
            current_list.append(group)
            added += 1
        if added:
            hooks[event] = current_list
            report["merged_events"].append(event)
        elif event not in hooks:
            report["warnings"].append(f"{event}: no groups appended")
    settings["hooks"] = hooks
    payload: Dict[str, Any] = dict(settings)
    payload["__report__"] = report
    return payload
