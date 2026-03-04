from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class NormalizedTranscriptMessage:
    role: str
    event_type: str
    content: str
    turn_id: str
    evidence_ref: str
    confidence: float
    salience: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptAdapterConfig:
    role_aliases: Dict[str, str] = field(default_factory=dict)
    event_type_by_role: Dict[str, str] = field(default_factory=dict)
    default_confidence: float = 0.6
    default_salience: float = 0.5
    evidence_prefix: str = "transcript"
    content_limit: int = 4096
    turn_id_prefix: str = "msg"
    skip_empty: bool = True
    tool_role_names: set[str] = field(default_factory=lambda: {"tool", "function"})
    assistant_role_names: set[str] = field(default_factory=lambda: {"assistant", "ai", "agent"})
    user_role_names: set[str] = field(default_factory=lambda: {"user", "human", "system"})


@dataclass
class AdapterIngestResult:
    status: str
    thread_id: str
    seen: int
    normalized: int
    ingested: int
    failed: int
    skipped: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "thread_id": self.thread_id,
            "seen": int(self.seen),
            "normalized": int(self.normalized),
            "ingested": int(self.ingested),
            "failed": int(self.failed),
            "skipped": int(self.skipped),
        }
