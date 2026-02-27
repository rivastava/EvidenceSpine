"""EvidenceSpine: evidence-bound multi-agent conversation memory fabric."""

from evidencespine.protocol import (
    AgentConversationBrief,
    AgentHandoffPacket,
    AgentMemoryEvent,
    AgentMemoryFact,
)
from evidencespine.runtime import AgentMemoryRuntime, AgentMemoryRuntimeConfig, RuntimeHooks
from evidencespine.settings import EvidenceSpineSettings
from evidencespine.vector_backends import HashingVectorBackend, VectorBackend
from evidencespine.adapters import AutoGenAdapter, LangGraphAdapter

__all__ = [
    "AgentMemoryEvent",
    "AgentMemoryFact",
    "AgentConversationBrief",
    "AgentHandoffPacket",
    "AgentMemoryRuntime",
    "AgentMemoryRuntimeConfig",
    "RuntimeHooks",
    "EvidenceSpineSettings",
    "VectorBackend",
    "HashingVectorBackend",
    "LangGraphAdapter",
    "AutoGenAdapter",
]
