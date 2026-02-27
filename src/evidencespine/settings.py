from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from evidencespine.runtime import AgentMemoryRuntimeConfig


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return max(minimum, int(default))
    try:
        value = int(raw)
    except Exception:
        value = int(default)
    return max(minimum, value)


def _env_float(name: str, default: float, minimum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return max(minimum, float(default))
    try:
        value = float(raw)
    except Exception:
        value = float(default)
    return max(minimum, value)


@dataclass
class EvidenceSpineSettings:
    base_dir: str = ".evidencespine"
    enabled: bool = True
    fail_open: bool = True
    max_event_tail: int = 4000
    redaction_enable: bool = True
    dedupe_window_sec: float = 7200.0
    brief_max_tokens: int = 1800
    brief_top_k_events: int = 32
    brief_top_k_facts: int = 24
    brief_recency_half_life_hours: float = 8.0
    retrieval_mode: str = "lexical"
    retrieval_lexical_weight: float = 1.0
    retrieval_vector_weight: float = 0.35

    @classmethod
    def from_env(cls, *, base_dir: str | None = None) -> "EvidenceSpineSettings":
        effective_base = base_dir or os.getenv("EVIDENCESPINE_BASE_DIR", ".evidencespine")
        retrieval_mode = str(os.getenv("EVIDENCESPINE_RETRIEVAL_MODE", "lexical")).strip().lower()
        if retrieval_mode not in {"lexical", "hybrid", "vector"}:
            retrieval_mode = "lexical"
        return cls(
            base_dir=str(effective_base),
            enabled=_env_bool("EVIDENCESPINE_ENABLE", True),
            fail_open=_env_bool("EVIDENCESPINE_FAIL_OPEN", True),
            max_event_tail=_env_int("EVIDENCESPINE_MAX_EVENT_TAIL", 4000, 100),
            redaction_enable=_env_bool("EVIDENCESPINE_REDACTION_ENABLE", True),
            dedupe_window_sec=_env_float("EVIDENCESPINE_DEDUPE_WINDOW_SEC", 7200.0, 60.0),
            brief_max_tokens=_env_int("EVIDENCESPINE_BRIEF_MAX_TOKENS", 1800, 64),
            brief_top_k_events=_env_int("EVIDENCESPINE_BRIEF_TOP_K_EVENTS", 32, 1),
            brief_top_k_facts=_env_int("EVIDENCESPINE_BRIEF_TOP_K_FACTS", 24, 1),
            brief_recency_half_life_hours=_env_float("EVIDENCESPINE_BRIEF_RECENCY_HALF_LIFE_HOURS", 8.0, 0.25),
            retrieval_mode=retrieval_mode,
            retrieval_lexical_weight=_env_float("EVIDENCESPINE_RETRIEVAL_LEXICAL_WEIGHT", 1.0, 0.0),
            retrieval_vector_weight=_env_float("EVIDENCESPINE_RETRIEVAL_VECTOR_WEIGHT", 0.35, 0.0),
        )

    def to_runtime_config(self) -> AgentMemoryRuntimeConfig:
        base = Path(self.base_dir)
        return AgentMemoryRuntimeConfig(
            enabled=bool(self.enabled),
            fail_open=bool(self.fail_open),
            max_event_tail=int(self.max_event_tail),
            redaction_enable=bool(self.redaction_enable),
            dedupe_window_sec=float(self.dedupe_window_sec),
            events_path=str(base / "events.jsonl"),
            facts_path=str(base / "facts.jsonl"),
            state_path=str(base / "state.json"),
            briefs_dir=str(base / "briefs"),
            handoffs_dir=str(base / "handoffs"),
            brief_max_tokens=int(self.brief_max_tokens),
            brief_top_k_events=int(self.brief_top_k_events),
            brief_top_k_facts=int(self.brief_top_k_facts),
            brief_recency_half_life_hours=float(self.brief_recency_half_life_hours),
            retrieval_mode=str(self.retrieval_mode),
            retrieval_lexical_weight=float(self.retrieval_lexical_weight),
            retrieval_vector_weight=float(self.retrieval_vector_weight),
        )
