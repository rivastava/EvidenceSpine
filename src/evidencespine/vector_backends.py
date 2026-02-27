from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Protocol, Sequence


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower())


def _l2_norm(values: List[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


class VectorBackend(Protocol):
    """Interface for pluggable vector scoring backends.

    Returned scores should be within [0, 1] when possible.
    """

    def score_texts(self, query: str, texts: Sequence[str]) -> Sequence[float]:
        ...


@dataclass
class HashingVectorBackend:
    """Dependency-free baseline vector backend.

    This is not SOTA embedding quality; it provides a deterministic vector signal
    so hybrid retrieval works out of the box without extra dependencies.
    """

    dim: int = 512

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * int(max(64, self.dim))
        tokens = _tokenize(text)
        if not tokens:
            return vec
        for tok in tokens:
            idx = hash(tok) % len(vec)
            vec[idx] += 1.0
        norm = _l2_norm(vec)
        if norm <= 1e-12:
            return vec
        return [x / norm for x in vec]

    def score_texts(self, query: str, texts: Sequence[str]) -> Sequence[float]:
        q = self._embed(query)
        out: List[float] = []
        for text in texts:
            v = self._embed(text)
            # Cosine is dot product because vectors are normalized.
            score = sum(a * b for a, b in zip(q, v))
            if score < 0.0:
                score = 0.0
            if score > 1.0:
                score = 1.0
            out.append(float(score))
        return out
