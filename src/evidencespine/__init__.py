"""EvidenceSpine: evidence-bound multi-agent conversation memory fabric."""

from evidencespine.protocol import (
    AgentConversationBrief,
    AgentControlView,
    AgentHandoffPacket,
    AgentMemoryEvent,
    AgentMemoryFact,
    ClaimCitation,
    ControlViewRow,
    EvidenceItem,
    StateContext,
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
    "AgentControlView",
    "AgentHandoffPacket",
    "EvidenceItem",
    "ClaimCitation",
    "StateContext",
    "ControlViewRow",
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
