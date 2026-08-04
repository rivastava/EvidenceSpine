"""Concrete handlers backing the ``evidencespine harness`` CLI commands."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from evidencespine.harness.hooks import (
    handle_precompact,
    handle_session_start,
    handle_session_stop,
    read_stdin_json,
)
from evidencespine.harness.install import debug_harness, install_harness


def cmd_session_start(
    *,
    thread_id: Optional[str] = None,
    objective: str = "",
    token_budget: Optional[int] = None,
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    json_out: bool = False,
) -> str:
    """Render the auto-recall brief for injection into session context."""
    text = handle_session_start(
        thread_id=thread_id,
        objective=objective,
        token_budget=token_budget,
        base_dir=base_dir,
        storage_format=storage_format,
    )
    if json_out:
        return json.dumps({"status": "ok", "brief": text}, indent=2, ensure_ascii=True)
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
) -> str:
    """Persist the compaction summary and render retain-through-compaction state."""
    text = handle_precompact(
        thread_id=thread_id,
        summary=summary,
        token_budget=token_budget,
        input_json=read_stdin_json(),
        base_dir=base_dir,
        storage_format=storage_format,
    )
    if json_out:
        return json.dumps({"status": "ok", "note": text}, indent=2, ensure_ascii=True)
    return text


def cmd_install(
    *,
    harness: str,
    target_dir: str,
    base_dir: str,
    executable: Optional[str] = None,
    scope: str = "project",
) -> Dict[str, Any]:
    return install_harness(
        harness=harness,
        target_dir=target_dir,
        base_dir=base_dir,
        executable=executable,
        scope=scope,
    )


def cmd_debug(
    *,
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    return debug_harness(base_dir=str(base_dir or ".evidencespine"), storage_format=storage_format, thread_id=thread_id)
