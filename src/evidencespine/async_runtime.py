"""Async wrapper around the synchronous :class:`AgentMemoryRuntime`.

All blocking operations (SQLite/JSONL I/O) are offloaded to a worker thread
via ``asyncio.to_thread`` so the runtime is safe to call from async event
loops (A2A handlers, FastAPI routes, MCP transports).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from evidencespine.runtime import AgentMemoryRuntime


class AsyncAgentMemoryRuntime:
    """Thread-offloaded async facade over an ``AgentMemoryRuntime``.

    Mirror the sync runtime's public surface for the memory operations that
    agents call most: ingest, briefs, handoffs, control views, health, TTL.
    The underlying runtime is built lazily from the same settings path.
    """

    def __init__(self, runtime: AgentMemoryRuntime | None = None, **settings_kwargs: Any) -> None:
        if runtime is not None:
            self._runtime = runtime
        else:
            from evidencespine.settings import EvidenceSpineSettings

            config = EvidenceSpineSettings.from_env(**settings_kwargs).to_runtime_config()
            self._runtime = AgentMemoryRuntime(config=config)

    @property
    def runtime(self) -> AgentMemoryRuntime:
        return self._runtime

    async def ingest_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        return await asyncio.to_thread(self._runtime.ingest_event, dict(event or {}))

    async def append_fact(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        return await asyncio.to_thread(self._runtime.append_fact, dict(fact or {}))

    async def build_brief(self, thread_id: str, query: str = "", token_budget: Optional[int] = None) -> Dict[str, Any]:
        brief = await asyncio.to_thread(self._runtime.build_brief, thread_id, query, token_budget)
        return dict(brief.to_dict())

    async def emit_handoff(self, role: str, thread_id: str, scope: str = "cross-agent coordination") -> Dict[str, Any]:
        packet = await asyncio.to_thread(self._runtime.emit_handoff, role, thread_id, scope)
        return dict(packet.to_dict())

    async def import_handoff(
        self, payload_or_path: str | Dict[str, Any], *, source_agent_id: str = "external_agent", thread_id: str = ""
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self._runtime.import_handoff,
            payload_or_path,
            source_agent_id=source_agent_id,
            thread_id=thread_id,
        )

    async def query_view(
        self,
        view: str,
        *,
        thread_id: str = "",
        owner_agent_id: str = "",
        include_closed: bool = False,
        limit: int = 50,
    ) -> Dict[str, Any]:
        control_view = await asyncio.to_thread(
            self._runtime.query_view,
            view,
            thread_id=thread_id,
            owner_agent_id=owner_agent_id,
            include_closed=include_closed,
            limit=limit,
        )
        return dict(control_view.to_dict())

    async def reconcile(self, thread_id: str, *, limit: int = 50) -> Dict[str, Any]:
        return await asyncio.to_thread(self._runtime.reconcile, thread_id, limit=limit)

    async def snapshot(self) -> Dict[str, Any]:
        return await asyncio.to_thread(self._runtime.snapshot)

    async def prune(
        self,
        *,
        thread_id: str = "",
        ttl_hours: Optional[float] = None,
        ttl_hours_facts: Optional[float] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self._runtime.prune,
            thread_id=thread_id,
            ttl_hours=ttl_hours,
            ttl_hours_facts=ttl_hours_facts,
            dry_run=dry_run,
        )

    async def check_evidence_stale(self, *, thread_id: str = "", source_root: str = ".", dry_run: bool = True) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self._runtime.check_evidence_stale,
            thread_id=thread_id,
            source_root=source_root,
            dry_run=dry_run,
        )

    async def flush(self) -> Dict[str, Any]:
        return await asyncio.to_thread(self._runtime.flush)
