"""Markdown renderers for EvidenceSpine payloads.

Used by the MCP server resources and the harness delivery layer so bounded
briefs, control views, scope state, and snapshots render identically
everywhere an agent might attach them to context.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def render_brief_markdown(payload: Dict[str, Any]) -> str:
    """Render a bounded context brief as markdown."""
    lines: List[str] = [f"# Agent Context Brief: {payload.get('thread_id', '')}"]
    query = payload.get("query", "")
    if query:
        lines.append(f"\n**query:** {query}")
    lines.append(f"\n**generated_at:** {payload.get('generated_at', '')}")
    lines.append(f"**token_budget:** {payload.get('token_budget', 0)}")
    meta = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
    if "stale" in meta:
        lines.append(f"**stale:** {str(bool(meta.get('stale', False))).lower()}")
    for key in [
        "current_goal",
        "locked_decisions",
        "recent_verified_facts",
        "active_risks",
        "open_items",
        "next_actions",
    ]:
        lines.append(f"\n## {key.replace('_', ' ')}")
        rows = payload.get(key, []) if isinstance(payload.get(key, []), list) else []
        if not rows:
            lines.append("- none")
        for row in rows:
            lines.append(f"- {row}")
    return "\n".join(lines)


def render_view_markdown(payload: Dict[str, Any]) -> str:
    """Render a derived control view as a markdown table."""
    lines: List[str] = [f"# Control view: {payload.get('view', '')}"]
    thread_id = payload.get("thread_id", "")
    if thread_id:
        lines.append(f"\n**thread_id:** {thread_id}")
    header = ["scope_id", "state_kind", "status", "owner", "freshness", "conflict", "claim"]
    lines.append("\n| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in payload.get("rows", []) if isinstance(payload.get("rows", []), list) else []:
        if not isinstance(row, dict):
            continue
        values = [
            str(row.get("scope_id", "")),
            str(row.get("state_kind", "")),
            str(row.get("status", "")),
            str(row.get("owner_agent_id", "")),
            str(row.get("freshness_state", "")),
            str(bool(row.get("conflict", False))).lower(),
            str(row.get("claim", "")),
        ]
        lines.append("| " + " | ".join(v.replace("|", "\\|") for v in values) + " |")
    meta = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
    if meta:
        lines.append(f"\n**total_rows:** {meta.get('total_rows', '')}")
        lines.append(f"**lookback_hours:** {meta.get('lookback_hours', '')}")
    return "\n".join(lines)


def render_state_markdown(payload: Dict[str, Any]) -> str:
    """Render resolved state rows for a scope as markdown."""
    lines: List[str] = [f"# State: {payload.get('scope_id', '')}"]
    rows = payload.get("rows", []) if isinstance(payload.get("rows", []), list) else []
    if not rows:
        lines.append("\n_no state records found for this scope._")
        return "\n".join(lines)
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append("")
        lines.append(f"## {row.get('scope_id', '')} ({row.get('scope_kind', '')} / {row.get('state_kind', '')})")
        lines.append(f"- **status:** {row.get('status', '')}")
        lines.append(f"- **owner:** {row.get('owner_agent_id', '')}")
        lines.append(f"- **state_basis:** {row.get('state_basis', '')}")
        lines.append(
            f"- **freshness:** {row.get('freshness_state', '')} (validated {row.get('validated_at', '')}, "
            f"fresh_until {row.get('fresh_until', '')})"
        )
        lines.append(f"- **lease:** {row.get('lease_state', '')} (expires {row.get('lease_expires_at', '')})")
        lines.append(f"- **claim:** {row.get('claim', '')}")
        lines.append(
            f"- **source:** {row.get('source_record_type', '')} {row.get('source_record_id', '')} at "
            f"{row.get('reported_at', '')}"
        )
        lines.append(
            f"- **contradiction:** {str(bool(row.get('has_contradiction', False))).lower()} / "
            f"**conflict:** {str(bool(row.get('conflict', False))).lower()}"
        )
        evidence_items = row.get("evidence_items", []) if isinstance(row.get("evidence_items", []), list) else []
        if evidence_items:
            lines.append("- **evidence:**")
            for item in evidence_items:
                if not isinstance(item, dict):
                    continue
                lines.append(f"  - ref={item.get('ref', '')} checksum={item.get('checksum', '')}")
    return "\n".join(lines)


def render_snapshot_markdown(payload: Dict[str, Any]) -> str:
    """Render the 24h memory health snapshot as markdown."""
    lines: List[str] = ["# EvidenceSpine memory snapshot (24h)", ""]
    if isinstance(payload, dict):
        for key in sorted(payload.keys()):
            value = payload[key]
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True, ensure_ascii=True)
            lines.append(f"- **{key}:** {value}")
    return "\n".join(lines)
