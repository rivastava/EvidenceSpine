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
from evidencespine.adapters import (
    AdapterIngestResult,
    AutoGenAdapter,
    LangGraphAdapter,
    NormalizedTranscriptMessage,
    TranscriptAdapter,
    TranscriptAdapterConfig,
)

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
    "TranscriptAdapter",
    "NormalizedTranscriptMessage",
    "TranscriptAdapterConfig",
    "AdapterIngestResult",
]
