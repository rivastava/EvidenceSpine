"""MCP tool definitions for the EvidenceSpine server.

Tool functions are registered with ``@server.tool()`` by ``register_tools``.
Each function runs in a worker thread (the SDK wraps sync tools with
``anyio.to_thread``) and mutates the shared EvidenceSpine store, which is
guarded by a re-entrant lock per backend.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

from evidencespine.grounding import ground_claim_refs


def register_tools(server: Any, get_runtime: Callable[[], Any], *, source_root: str = "") -> None:
    """Register EvidenceSpine tools on an MCPServer instance.

    ``source_root`` is the server-configured grounding root (defaults to the
    server working directory). Server-side grounding rejects absolute paths and
    confines relative paths to this root so a caller cannot read arbitrary host
    files.
    """
    grounding_root = os.path.realpath(source_root or os.getcwd())

    @server.tool(
        name="ingest_event",
        title="Ingest a memory event",
        description=(
            "Ingest one structured memory event. Pass a JSON object following the "
            "AgentMemoryEvent schema: thread_id, event_type (intent|decision|action|"
            "outcome|reflection), role, source_agent_id, source_turn_id, payload "
            "{claim, decision, outcome, target, next_actions, fact_state}, "
            "evidence_refs, evidence_items [{ref, excerpt, checksum}], state_context "
            "{scope_id, state_kind, status, owner_agent_id, fresh_until, ...}, "
            "ground_refs (file:line refs grounded server-side into checksummed "
            "evidence items; relative to the server source root, absolute paths "
            "rejected), confidence, salience. Returns the ingest result with "
            "status ok|invalid|deduped|disabled|fail_open."
        ),
    )
    def ingest_event(event: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest one structured memory event."""
        event = dict(event or {})
        ground_refs = [
            str(ref).strip()
            for ref in list(event.pop("ground_refs", None) or [])
            if isinstance(ref, str) and str(ref).strip()
        ]
        if ground_refs:
            grounded = ground_claim_refs(ground_refs, source_root=grounding_root, allow_absolute=False)
            existing = [dict(x) for x in list(event.get("evidence_items") or [])]
            event["evidence_items"] = [*existing, *grounded]
        return get_runtime().ingest_event(event)

    @server.tool(
        name="build_brief",
        title="Build a bounded context brief",
        description=(
            "Build a bounded context brief for a thread: current goal, locked "
            "decisions, recent verified facts, active risks, open items, next actions. "
            "Claims carry citations back to evidence refs/items. Pass an optional "
            "query to shape focus and token_budget to bound output size."
        ),
    )
    def build_brief(thread_id: str, query: str = "", token_budget: Optional[int] = None) -> Dict[str, Any]:
        """Build a bounded context brief for a thread."""
        budget = int(token_budget) if token_budget is not None and int(token_budget) > 0 else None
        return get_runtime().build_brief(thread_id=thread_id, query=query, token_budget=budget).to_dict()

    @server.tool(
        name="query_view",
        title="Query a derived control view",
        description=(
            "Query a derived agent-state control view. Views: active_scopes, my_work, "
            "open_gates, stale_claims, contradictions. Rows resolve competing state "
            "records per scope (supersedes/freshness/conflict aware)."
        ),
    )
    def query_view(
        view: str,
        thread_id: str = "",
        owner_agent_id: str = "",
        include_closed: bool = False,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Query a derived agent-state control view."""
        return get_runtime().query_view(
            view,
            thread_id=thread_id,
            owner_agent_id=owner_agent_id,
            include_closed=bool(include_closed),
            limit=int(max(1, int(limit))),
        ).to_dict()

    @server.tool(
        name="emit_handoff",
        title="Emit a handoff packet",
        description=(
            "Emit an evidence-bound handoff packet from the current thread state: "
            "locked decisions, verified claims with citation grounding, unresolved "
            "contradictions, required validations. The packet is persisted and an "
            "ingest event is recorded."
        ),
    )
    def emit_handoff(
        role: str = "auditor",
        thread_id: str = "default",
        scope: str = "cross-agent coordination",
    ) -> Dict[str, Any]:
        """Emit an evidence-bound handoff packet."""
        return get_runtime().emit_handoff(role=role, thread_id=thread_id, scope=scope).to_dict()

    @server.tool(
        name="import_handoff",
        title="Import a handoff packet",
        description=(
            "Import a handoff packet produced by another agent/run. Provide either "
            "packet (the packet JSON object) or path (a file path). Locked decisions, "
            "claims, contradictions and validations are ingested with their fact_state/"
            "classification preserved. Pass thread_id to bind imported rows to the "
            "importer's thread; otherwise they inherit the packet's thread_id."
        ),
    )
    def import_handoff(
        packet: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None,
        source_agent_id: str = "external_agent",
        thread_id: str = "",
    ) -> Dict[str, Any]:
        """Import a handoff packet by payload or file path."""
        if packet is None and path is None:
            raise ValueError("provide either packet (dict) or path (str)")
        payload: Any = packet if packet is not None else path
        return get_runtime().import_handoff(payload, source_agent_id=source_agent_id, thread_id=thread_id)

    @server.tool(
        name="ground",
        title="Ground a file reference",
        description=(
            "Build a grounded evidence item from a file:line reference "
            "(path#L10-L20 or path:10-20): reads the excerpt and computes its "
            "sha256 checksum. The path is resolved relative to the server "
            "source root; absolute paths and paths escaping the root are "
            "rejected (status ungroundable)."
        ),
    )
    def ground(ref: str) -> Dict[str, Any]:
        """Ground a file reference into a checksummed evidence item."""
        from evidencespine.grounding import ground_ref

        item = ground_ref(ref, source_root=grounding_root, allow_absolute=False)
        if item is None:
            return {"status": "ungroundable", "ref": ref}
        return {"status": "ok", "item": item}

    @server.tool(
        name="verify_fact",
        title="Record verification provenance",
        description=(
            "Record how a fact was verified: ingests a verified copy of the fact "
            "(claim and grounded evidence preserved) carrying verification "
            "provenance {method: test|gate|tool|manual, reference} that supersedes "
            "the original. The brief then shows the verified-with-provenance row."
        ),
    )
    def verify_fact(
        fact_id: str,
        method: str,
        reference: str,
        verified_by: str = "external_agent",
        thread_id: str = "",
    ) -> Dict[str, Any]:
        """Record verification provenance for a fact."""
        return get_runtime().verify_fact(
            fact_id,
            method=method,
            reference=reference,
            verified_by=verified_by,
            thread_id=thread_id,
        )

    @server.tool(
        name="check_drift",
        title="Re-verify grounded evidence",
        description=(
            "Re-verify facts' checksummed evidence items against the live files: "
            "re-reads each source at the stored line range and compares the "
            "recomputed checksum. Returns changed/missing verdicts. Use dry_run "
            "to preview; dry_run=false writes evidence_stale flags onto facts, "
            "which surface in the stale_claims view, briefs (STALE EVIDENCE) and "
            "the snapshot metric."
        ),
    )
    def check_drift(thread_id: str = "", source_root: str = "", dry_run: bool = True) -> Dict[str, Any]:
        """Re-verify grounded evidence against live files."""
        return get_runtime().check_evidence_stale(
            thread_id=thread_id,
            source_root=source_root or grounding_root,
            dry_run=bool(dry_run),
        )

    @server.tool(
        name="reconcile",
        title="Reconcile agent state",
        description=(
            "Run the optional state reconciliation hook over active scopes. If no "
            "reconcile hook is configured, returns status unsupported. Returns counts "
            "of seen/ingested/invalid rows."
        ),
    )
    def reconcile(thread_id: str, limit: int = 50) -> Dict[str, Any]:
        """Run the optional state reconciliation hook."""
        return get_runtime().reconcile(thread_id, limit=int(max(1, int(limit))))

    @server.tool(
        name="snapshot",
        title="Show memory health snapshot",
        description=(
            "Return a 24h health snapshot: event/fact volumes, verified facts, "
            "contradiction ratio, brief success/stale rates, handoff completeness, "
            "citation coverage, active scope and gate counts, fail-open events."
        ),
    )
    def snapshot() -> Dict[str, Any]:
        """Return a 24h memory health snapshot."""
        return get_runtime().snapshot()

    @server.tool(
        name="prune",
        title="Prune rows older than TTL",
        description=(
            "TTL archival: delete memory rows older than ttl_hours (default 720h / "
            "30 days; rows without a parseable timestamp are always kept). Use "
            "dry_run=true to preview counts without deleting."
        ),
    )
    def prune(
        thread_id: str = "",
        ttl_hours: float = 720.0,
        ttl_hours_facts: Optional[float] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Prune memory rows older than the TTL."""
        return get_runtime().prune(
            thread_id=thread_id,
            ttl_hours=float(ttl_hours),
            ttl_hours_facts=ttl_hours_facts,
            dry_run=bool(dry_run),
        )
