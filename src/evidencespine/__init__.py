"""EvidenceSpine: evidence-bound multi-agent conversation memory fabric."""

__version__ = "0.5.0"

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
from evidencespine.async_runtime import AsyncAgentMemoryRuntime

__all__ = [
    "__version__",
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
    "AsyncAgentMemoryRuntime",
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
