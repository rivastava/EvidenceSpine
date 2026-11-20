"""Concrete handlers backing the 'evidencespine harness' CLI commands."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from evidencespine.harness.hooks import (
    handle_precompact,
    handle_session_start,
    handle_session_stop,
    read_stdin_json,
)
from evidencespine.harness.install import debug_harness, doctor_harness, install_harness, render_print_config

_FAIL_OPEN_PREFIX = "[EvidenceSpine fail-open:"


def _text_status(text: str) -> str:
    return "fail_open" if str(text or "").startswith(_FAIL_OPEN_PREFIX) else "ok"


def cmd_session_start(
    *,
    thread_id: Optional[str] = None,
    objective: str = "",
    token_budget: Optional[int] = None,
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    json_out: bool = False,
    session_id: Optional[str] = None,
) -> str:
    """Render the auto-recall brief for injection into session context."""
    stdin_payload = read_stdin_json()
    text = handle_session_start(
        thread_id=thread_id,
        objective=objective,
        token_budget=token_budget,
        base_dir=base_dir,
        storage_format=storage_format,
        input_json=stdin_payload,
        session_id=session_id,
    )
    if json_out:
        return json.dumps({"status": _text_status(text), "brief": text}, indent=2, ensure_ascii=True)
    return text


def cmd_session_stop(
    *,
    thread_id: Optional[str] = None,
    summary: Optional[str] = None,
    auto_handoff: Optional[bool] = None,
    reason: str = "session_stop",
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    json_out: bool = False,
    session_id: Optional[str] = None,
) -> str:
    """Record session end (auto-retain), optionally emitting a handoff."""
    payload = handle_session_stop(
        thread_id=thread_id,
        summary=summary,
        auto_handoff=auto_handoff,
        reason=reason,
        input_json=read_stdin_json(),
        base_dir=base_dir,
        storage_format=storage_format,
        session_id=session_id,
    )
    if json_out:
        return json.dumps(payload, indent=2, ensure_ascii=True)
    return json.dumps(payload, ensure_ascii=True)


def cmd_precompact(
    *,
    thread_id: Optional[str] = None,
    summary: Optional[str] = None,
    token_budget: Optional[int] = None,
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    json_out: bool = False,
    session_id: Optional[str] = None,
) -> str:
    """Persist the compaction summary and render retain-through-compaction state.

    Claude Code consumes plain-text stdout as instructions for the compaction
    summary (hookSpecificOutput.additionalContext is NOT supported there);
    Codex ignores PreCompact stdout and re-anchors via SessionStart(compact).
    """
    text = handle_precompact(
        thread_id=thread_id,
        summary=summary,
        token_budget=token_budget,
        base_dir=base_dir,
        storage_format=storage_format,
        input_json=read_stdin_json(),
        session_id=session_id,
    )
    if json_out:
        return json.dumps({"status": _text_status(text), "note": text}, indent=2, ensure_ascii=True)
    return text


def cmd_install(
    *,
    harness: str,
    target_dir: str,
    base_dir: str,
    executable: Optional[str] = None,
    scope: str = "project",
    dry_run: bool = False,
    mcp_url: Optional[str] = None,
    with_settings_hooks: bool = False,
    with_agents_md: bool = False,
) -> Dict[str, Any]:
    return install_harness(
        harness=harness,
        target_dir=target_dir,
        base_dir=base_dir,
        executable=executable,
        scope=scope,
        dry_run=dry_run,
        mcp_url=mcp_url,
        with_settings_hooks=with_settings_hooks,
        with_agents_md=with_agents_md,
    )


def cmd_debug(
    *,
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    thread_id: Optional[str] = None,
    target_dir: str = ".",
) -> Dict[str, Any]:
    return debug_harness(
        base_dir=base_dir,
        storage_format=storage_format,
        thread_id=thread_id,
        target_dir=target_dir,
    )


def cmd_doctor(
    *,
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    thread_id: Optional[str] = None,
    target_dir: str = ".",
) -> Dict[str, Any]:
    return doctor_harness(
        base_dir=base_dir,
        storage_format=storage_format,
        thread_id=thread_id,
        target_dir=target_dir,
    )


def cmd_print_config(
    *,
    client: str,
    scope: str = "project",
    transport: str = "stdio",
    executable: Optional[str] = None,
    base_dir: str = ".evidencespine",
    project_dir: str = ".",
) -> Dict[str, Any]:
    return render_print_config(
        client=client,
        scope=scope,
        transport=transport,
        executable=executable,
        base_dir=base_dir,
        project_dir=project_dir,
    )
