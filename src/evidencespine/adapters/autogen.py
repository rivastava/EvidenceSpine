from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from evidencespine.runtime import AgentMemoryRuntime


def _safe_text(value: Any, default: str = "", limit: int = 4096) -> str:
    text = str(value if value is not None else default).strip()
    if not text:
        text = default
    return text[:limit]


def _iter_messages(messages: Any) -> Iterable[Any]:
    if isinstance(messages, list):
        for m in messages:
            yield m


@dataclass
class AutoGenAdapter:
    """Drop-in adapter for AutoGen-style message lists.

    Supports dict/object message shapes with fields such as:
    - role / source / name
    - content / text
    - id / message_id
    """

    runtime: AgentMemoryRuntime
    default_thread_id: str = "autogen_default"
    source_agent_id: str = "autogen"

    @staticmethod
    def _event_type_for_role(role: str) -> str:
        role_l = str(role or "").lower()
        if role_l in {"user", "human", "system"}:
            return "intent"
        if role_l in {"assistant", "ai", "agent"}:
            return "decision"
        if role_l in {"tool", "function"}:
            return "outcome"
        return "reflection"

    def ingest_messages(self, messages: Any, *, thread_id: str | None = None) -> Dict[str, Any]:
        tid = str(thread_id or self.default_thread_id)
        ingested = 0
        failed = 0

        for idx, msg in enumerate(_iter_messages(messages)):
            role = "unknown"
            content = ""
            turn_id = f"ag_{idx}"

            if isinstance(msg, dict):
                role = _safe_text(msg.get("role", msg.get("source", msg.get("name", "unknown"))), "unknown", 64)
                content = _safe_text(msg.get("content", msg.get("text", "")), "", 4096)
                turn_id = _safe_text(msg.get("id", msg.get("message_id", turn_id)), turn_id, 128)
            else:
                role = _safe_text(
                    getattr(msg, "role", getattr(msg, "source", getattr(msg, "name", "unknown"))),
                    "unknown",
                    64,
                )
                content = _safe_text(getattr(msg, "content", getattr(msg, "text", "")), "", 4096)
                turn_id = _safe_text(getattr(msg, "id", getattr(msg, "message_id", turn_id)), turn_id, 128)

            if not content:
                continue

            out = self.runtime.ingest_event(
                {
                    "thread_id": tid,
                    "event_type": self._event_type_for_role(role),
                    "role": role,
                    "source_agent_id": self.source_agent_id,
                    "source_turn_id": turn_id,
                    "payload": {
                        "claim": content,
                        "fact_state": "asserted",
                    },
                    "evidence_refs": [f"autogen:{turn_id}"],
                    "confidence": 0.6,
                    "salience": 0.5,
                }
            )
            if out.get("status") in {"ok", "deduped"}:
                ingested += 1
            else:
                failed += 1

        return {
            "status": "ok",
            "thread_id": tid,
            "ingested": ingested,
            "failed": failed,
        }

    def brief(self, query: str, *, thread_id: str | None = None) -> Dict[str, Any]:
        tid = str(thread_id or self.default_thread_id)
        return self.runtime.build_brief(tid, query).to_dict()

    def handoff(self, role: str, scope: str, *, thread_id: str | None = None) -> Dict[str, Any]:
        tid = str(thread_id or self.default_thread_id)
        return self.runtime.emit_handoff(role=role, thread_id=tid, scope=scope).to_dict()
