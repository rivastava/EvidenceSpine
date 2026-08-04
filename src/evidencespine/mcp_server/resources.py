"""MCP resource definitions for the EvidenceSpine server.

Resources are progressive-disclosure surfaces: bounded briefs are primary, raw
state rows secondary. Each template resource renders markdown so clients can
attach the result directly to context.
"""

from __future__ import annotations

from typing import Any, Callable

from evidencespine.render import (
    render_brief_markdown,
    render_snapshot_markdown,
    render_state_markdown,
    render_view_markdown,
)
from evidencespine.usage import usage_guide_markdown


def register_resources(server: Any, get_runtime: Callable[[], Any]) -> None:
    """Register EvidenceSpine resources on an MCPServer instance."""

    @server.resource(
        "evidencespine://guide",
        name="usage guide",
        title="Agent usage guide",
        description="Canonical usage guide: what the spine is, the evidence model, and decision rules for when to use each tool.",
        mime_type="text/markdown",
    )
    def usage_guide() -> str:
        """Render the canonical agent usage guide as markdown."""
        return usage_guide_markdown()

    @server.resource(
        "evidencespine://brief/{thread_id}",
        name="context brief",
        title="Bounded agent context brief",
        description="Markdown context brief for a thread: goal, decisions, verified facts, risks, next actions.",
        mime_type="text/markdown",
    )
    def context_brief(thread_id: str) -> str:
        """Render a bounded context brief for a thread as markdown."""
        brief = get_runtime().build_brief(thread_id=thread_id, query="latest context")
        return render_brief_markdown(brief.to_dict())

    @server.resource(
        "evidencespine://view/{view}",
        name="control view",
        title="Derived control view",
        description="Markdown rows for a control view: active_scopes, my_work, open_gates, stale_claims, contradictions.",
        mime_type="text/markdown",
    )
    def control_view(view: str) -> str:
        """Render a derived control view as markdown."""
        payload = get_runtime().query_view(view, limit=50).to_dict()
        return render_view_markdown(payload)

    @server.resource(
        "evidencespine://state/{scope_id}",
        name="scope state",
        title="Resolved state for a scope",
        description="Markdown summary of the resolved agent state records for a scope id.",
        mime_type="text/markdown",
    )
    def scope_state(scope_id: str) -> str:
        """Render resolved state rows for a scope as markdown."""
        payload = get_runtime().query_view("active_scopes", limit=512).to_dict()
        rows = [r for r in payload.get("rows", []) if isinstance(r, dict) and r.get("scope_id") == scope_id]
        if not rows:
            closed = get_runtime().query_view("active_scopes", include_closed=True, limit=512).to_dict()
            rows = [r for r in closed.get("rows", []) if isinstance(r, dict) and r.get("scope_id") == scope_id]
        return render_state_markdown({"scope_id": scope_id, "rows": rows})

    @server.resource(
        "evidencespine://snapshot",
        name="memory snapshot",
        title="24h memory health snapshot",
        description="Markdown summary of the 24h memory health snapshot metrics.",
        mime_type="text/markdown",
    )
    def snapshot_resource() -> str:
        """Render the 24h memory health snapshot as markdown."""
        return render_snapshot_markdown(get_runtime().snapshot())
