"""MCP prompt definitions for the EvidenceSpine server.

These prompts give coding agents drop-in session-orientation and handoff
workflows, matching the harness integration points in the docs: session start
recall, handoff receive, handoff send.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from evidencespine.render import render_brief_markdown, render_snapshot_markdown


def _clean_arg(value: str, default: str) -> str:
    """Treat MCP placeholder args (``$1``/``$2``/``$3``) as unset.

    Some hosts pre-render prompts with auto-filled positional placeholders
    instead of omitting arguments; a bare ``$1`` thread id would otherwise
    create ``$1_*.json`` briefs and polluted rows.
    """
    text = str(value or "").strip()
    if not text or text.startswith("$"):
        return default
    return text


def _brief_markdown(get_runtime: Callable[[], Any], thread_id: str, query: str) -> str:
    return render_brief_markdown(get_runtime().build_brief(thread_id=thread_id, query=query).to_dict())


def register_prompts(server: Any, get_runtime: Callable[[], Any]) -> None:
    """Register EvidenceSpine prompts on an MCPServer instance."""

    @server.prompt(
        name="session_start",
        title="Session start orientation",
        description=(
            "Orientation prompt for a fresh agent session: load the bounded context "
            "brief, internalize locked decisions and verified facts, and surface "
            "risks/contradictions before working. Use at SessionStart."
        ),
    )
    def session_start(thread_id: str = "default", objective: str = "") -> str:
        """Return a session-start orientation prompt with the context brief embedded."""
        thread_id = _clean_arg(thread_id, "default")
        objective = _clean_arg(objective, "")
        brief = _brief_markdown(get_runtime, thread_id, objective or "session start")
        return (
            "You are resuming work on an EvidenceSpine thread. Load the context below "
            "as your working state. Locked decisions are binding unless new evidence "
            "supersedes them. Verified facts are evidence-bound; if you must act "
            "against one, record the contradiction. Open items and next actions are "
            "your queue. If any context looks stale or conflicts, say so explicitly.\n\n"
            f"{brief}"
        )

    @server.prompt(
        name="handoff_receive",
        title="Receive an agent handoff",
        description=(
            "Prompt to import an inbound handoff packet from another agent/run and "
            "orient on its claims, unresolved contradictions, and required "
            "validations. Use when a handoff packet arrives."
        ),
    )
    def handoff_receive(thread_id: str = "default", source_agent_id: str = "external_agent", packet_path: str = "") -> str:
        """Return a handoff-receive prompt instructing import and orientation."""
        thread_id = _clean_arg(thread_id, "default")
        source_agent_id = _clean_arg(source_agent_id, "external_agent")
        packet_path = _clean_arg(packet_path, "")
        runtime = get_runtime()
        if packet_path:
            result = runtime.import_handoff(packet_path, source_agent_id=source_agent_id)
            imported = json.dumps(result, indent=2, ensure_ascii=True)
        else:
            imported = (
                "No packet path was provided. Call the import_handoff tool with the "
                "incoming packet (or its file path) before orienting."
            )
        return (
            "An agent handoff packet has arrived. Import it (see below), then orient: "
            "treat locked_decisions as binding, claims as evidence-bound working "
            "state, unresolved_contradictions as open risks, and required_validations "
            "as your first tasks. Verify the packet's checksums and spans where cited.\n\n"
            f"## Import result\n{imported}\n\n"
            f"## Current thread brief\n{_brief_markdown(get_runtime, thread_id, 'handoff receive')}"
        )
    @server.prompt(
        name="handoff_send",
        title="Emit an agent handoff",
        description=(
            "Prompt to emit an evidence-bound handoff packet for the current thread "
            "so another agent/run can pick up with verified state and required "
            "validations. Use before stopping."
        ),
    )
    def handoff_send(thread_id: str = "default", role: str = "auditor", scope: str = "cross-agent coordination") -> str:
        """Return a handoff-send prompt instructing packet emission."""
        thread_id = _clean_arg(thread_id, "default")
        role = _clean_arg(role, "auditor")
        scope = _clean_arg(scope, "cross-agent coordination")
        runtime = get_runtime()
        return (
            "Emit an evidence-bound handoff packet so the next agent/run can resume "
            "with verified state. Call the emit_handoff tool with "
            f"role={role!r}, thread_id={thread_id!r}, scope={scope!r} to persist the "
            "packet. Confirm its required_validations become your successor's first "
            "tasks, and note any unresolved_contradictions explicitly.\n\n"
            f"## Snapshot\n{render_snapshot_markdown(runtime.snapshot())}"
        )
