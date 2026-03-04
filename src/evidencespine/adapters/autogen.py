from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from evidencespine.adapters.transcript import TranscriptAdapter
from evidencespine.runtime import AgentMemoryRuntime


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
    _core: TranscriptAdapter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        config = TranscriptAdapter.default_config()
        config.role_aliases = {
            **config.role_aliases,
            "assistant": "assistant",
            "user": "user",
        }
        config.evidence_prefix = "autogen"
        config.turn_id_prefix = "ag"
        self._core = TranscriptAdapter(
            self.runtime,
            config=config,
            default_thread_id=self.default_thread_id,
            source_agent_id=self.source_agent_id,
        )

    def normalize_messages(self, messages: Any) -> List[Any]:
        return self._core.normalize_messages(messages)

    def ingest_messages(self, messages: Any, *, thread_id: str | None = None) -> Dict[str, Any]:
        return self._core.ingest_messages(messages, thread_id=thread_id)

    def brief(self, query: str, *, thread_id: str | None = None) -> Dict[str, Any]:
        return self._core.brief(query, thread_id=thread_id)

    def handoff(self, role: str, scope: str, *, thread_id: str | None = None) -> Dict[str, Any]:
        return self._core.handoff(role, scope, thread_id=thread_id)
