"""A2A executor mapping messages to EvidenceSpine memory operations.

The executor implements the task-based flow of the a2a-sdk 1.1 protocol: it
enqueues an initial ``Task`` (working state) and finishes with a
``TaskStatusUpdateEvent`` in a terminal state. The user message is either a
JSON object ``{"action": ..., "params": {...}, "thread_id": ...}`` (actions
mirror the MCP tools) or plain text, which is treated as a brief query.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import a2a_pb2

from evidencespine.protocol import safe_text


class EvidenceSpineExecutor(AgentExecutor):
    """A2A executor that serves EvidenceSpine memory operations."""

    _OPS = {
        "ingest_event",
        "append_fact",
        "build_brief",
        "query_view",
        "emit_handoff",
        "import_handoff",
        "reconcile",
        "snapshot",
        "prune",
    }

    def __init__(self, get_runtime: Callable[[], Any]) -> None:
        """``get_runtime`` builds an ``AgentMemoryRuntime`` (lazily, once)."""
        self._get_runtime = get_runtime
        self._runtime = None

    @property
    def runtime(self) -> Any:
        if self._runtime is None:
            self._runtime = self._get_runtime()
        return self._runtime

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute the memory operation for the incoming message."""
        task_id = str(context.task_id or "task")
        context_id = str(context.context_id or task_id)
        text = context.get_user_input(delimiter="\n")
        thread = self._thread_id(context_id, task_id, text)

        interim = json.dumps({"status": "working", "thread_id": thread}, ensure_ascii=True)

        task = a2a_pb2.Task(
            id=task_id,
            context_id=context_id,
            status=a2a_pb2.TaskStatus(
                state=a2a_pb2.TASK_STATE_WORKING,
                message=a2a_pb2.Message(
                    role=a2a_pb2.ROLE_AGENT,
                    parts=[a2a_pb2.Part(text=interim)],
                ),
            ),
            history=[
                a2a_pb2.Message(
                    role=a2a_pb2.ROLE_USER,
                    parts=[a2a_pb2.Part(text=text)],
                )
            ],
        )
        await event_queue.enqueue_event(task)

        result = await asyncio.to_thread(self._dispatch, text, thread)

        final = json.dumps(result, indent=2, ensure_ascii=True)
        await event_queue.enqueue_event(
            a2a_pb2.TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=a2a_pb2.TaskStatus(
                    state=a2a_pb2.TASK_STATE_COMPLETED,
                    message=a2a_pb2.Message(
                        role=a2a_pb2.ROLE_AGENT,
                        parts=[a2a_pb2.Part(text=final)],
                    ),
                ),
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the active task (best effort; memory ops are short-lived)."""
        await event_queue.enqueue_event(
            a2a_pb2.TaskStatusUpdateEvent(
                task_id=str(context.task_id or "task"),
                context_id=str(context.context_id or ""),
                status=a2a_pb2.TaskStatus(
                    state=a2a_pb2.TASK_STATE_CANCELED,
                    message=a2a_pb2.Message(
                        role=a2a_pb2.ROLE_AGENT,
                        parts=[a2a_pb2.Part(text=json.dumps({"status": "canceled"}))],
                    ),
                ),
            )
        )

    def _thread_id(self, context_id: str, task_id: str, text: str) -> str:
        if context_id and context_id != "task":
            return context_id
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                thread = safe_text(parsed.get("thread_id"), "", 128)
                if thread:
                    return thread
        except Exception:
            pass
        return safe_text(task_id, "default", 128)

    def _dispatch(self, text: str, thread: str) -> Dict[str, Any]:
        """Run one memory operation; fail-open: errors return a JSON payload."""
        try:
            action, params = self._parse(text)
            return self._run(action, params, thread)
        except Exception as exc:
            return {"status": "error", "reason": str(exc)}

    def _parse(self, text: str) -> tuple[str, Dict[str, Any]]:
        text = str(text or "").strip()
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and isinstance(parsed.get("action"), str):
                    action = safe_text(parsed.get("action"), "", 64).lower()
                    params = parsed.get("params")
                    if not isinstance(params, dict):
                        params = {}
                    params = dict(params)
                    for key in ("thread_id", "source_agent_id", "objective_id"):
                        if key in parsed and key not in params:
                            params[key] = parsed[key]
                    return action, params
            except Exception:
                pass
        return "build_brief", {"query": text}

    def _run(self, action: str, params: Dict[str, Any], thread: str) -> Dict[str, Any]:
        rt = self.runtime
        if action not in self._OPS:
            return {"status": "error", "reason": f"unknown action: {action}"}
        if action == "ingest_event":
            params = dict(params or {})
            params.setdefault("thread_id", thread)
            return rt.ingest_event(params)
        if action == "append_fact":
            params = dict(params or {})
            params.setdefault("thread_id", thread)
            return rt.append_fact(params)
        if action == "build_brief":
            query = safe_text(params.get("query"), params.get("query") or "", 512)
            if not isinstance(query, str):
                query = ""
            budget = params.get("token_budget")
            return rt.build_brief(
                thread_id=safe_text(params.get("thread_id"), thread, 128),
                query=query,
                token_budget=int(budget) if isinstance(budget, int) else None,
            ).to_dict()
        if action == "query_view":
            view = safe_text(params.get("view"), "", 64)
            if not view:
                return {"status": "error", "reason": "view is required"}
            return rt.query_view(
                view,
                thread_id=safe_text(params.get("thread_id"), thread, 128),
                owner_agent_id=safe_text(params.get("owner_agent_id"), "", 128),
                include_closed=bool(params.get("include_closed", False)),
                limit=int(params.get("limit", 50)),
            ).to_dict()
        if action == "emit_handoff":
            return rt.emit_handoff(
                role=safe_text(params.get("role"), "auditor", 64),
                thread_id=safe_text(params.get("thread_id"), thread, 128),
                scope=safe_text(params.get("scope"), "cross-agent coordination", 256),
            ).to_dict()
        if action == "import_handoff":
            packet = params.get("packet")
            return rt.import_handoff(
                packet if packet is not None else str(params.get("path", "")),
                source_agent_id=safe_text(params.get("source_agent_id"), "external_agent", 128),
                thread_id=safe_text(params.get("thread_id"), "", 128),
            )
        if action == "reconcile":
            return rt.reconcile(
                safe_text(params.get("thread_id"), thread, 128),
                limit=int(params.get("limit", 50)),
            )
        if action == "snapshot":
            return rt.snapshot()
        if action == "prune":
            return rt.prune(
                thread_id=safe_text(params.get("thread_id"), "", 128),
                ttl_hours=float(params.get("ttl_hours", 720.0)),
                ttl_hours_facts=params.get("ttl_hours_facts"),
                dry_run=bool(params.get("dry_run", False)),
            )
        return {"status": "error", "reason": f"unknown action: {action}"}
