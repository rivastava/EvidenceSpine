from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from evidencespine.adapters.base import TranscriptAdapterConfig
from evidencespine.adapters.transcript import TranscriptAdapter
from evidencespine.runtime import AgentMemoryRuntime


@dataclass
class LangGraphAdapter:
    """Drop-in adapter for LangGraph-style state dicts.

    Expected shape:
      {"messages": [{"role": "user|assistant|tool", "content": "...", "id": "..."}, ...]}
    """

    runtime: AgentMemoryRuntime
    default_thread_id: str = "langgraph_default"
    source_agent_id: str = "langgraph"
    _core: TranscriptAdapter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        config = TranscriptAdapter.default_config()
        config.evidence_prefix = "langgraph"
        config.turn_id_prefix = "lg"
        self._core = TranscriptAdapter(
            self.runtime,
            config=config,
            default_thread_id=self.default_thread_id,
            source_agent_id=self.source_agent_id,
        )

    def normalize_state(self, state: Any) -> List[Any]:
        return self._core.normalize_messages(state)

    def ingest_state(self, state: Any, *, thread_id: str | None = None) -> Dict[str, Any]:
        return self._core.ingest_messages(state, thread_id=thread_id)

    def brief(self, query: str, *, thread_id: str | None = None) -> Dict[str, Any]:
        return self._core.brief(query, thread_id=thread_id)

    def handoff(self, role: str, scope: str, *, thread_id: str | None = None) -> Dict[str, Any]:
        return self._core.handoff(role, scope, thread_id=thread_id)
