from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Protocol, Sequence


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


def _cosine(query_vec: Sequence[float], doc_vec: Sequence[float]) -> float:
    qn = _l2_norm(list(query_vec))
    dn = _l2_norm(list(doc_vec))
    if qn <= 1e-12 or dn <= 1e-12:
        return 0.0
    dot = sum(a * b for a, b in zip(query_vec, doc_vec))
    return max(0.0, min(1.0, float(dot / (qn * dn))))


@dataclass
class FastEmbedVectorBackend:
    """Real embedding backend backed by the optional ``[embeddings]`` extra.

    Uses ``fastembed.TextEmbedding`` (ONNX, no GPU required). The model is
    loaded lazily on first use; document embeddings are cached by content hash
    so repeated retrieval calls do not re-embed unchanged rows. Instantiating
    this class never imports fastembed; only ``score_texts`` does.
    """

    model: str = "BAAI/bge-small-en-v1.5"
    cache_size: int = 8192

    def __post_init__(self) -> None:
        self._embedder = None
        self._cache: Dict[str, List[float]] = {}

    def _load_embedder(self):
        if self._embedder is None:
            try:
                from fastembed import TextEmbedding  # type: ignore[import-not-found]
            except Exception as exc:
                raise ImportError(
                    "fastembed is not installed; run `pip install evidencespine[embeddings]`"
                ) from exc
            self._embedder = TextEmbedding(model_name=str(self.model))
        return self._embedder

    def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        embedder = self._load_embedder()
        out: List[List[float]] = []
        missing: List[int] = []
        for idx, text in enumerate(texts):
            cached = self._cache.get(str(text))
            if cached is not None:
                out.append(cached)
            else:
                out.append([])
                missing.append(idx)
        if missing:
            batch = [str(texts[idx]) for idx in missing]
            vectors = list(embedder.embed(batch))
            for pos, idx in enumerate(missing):
                vec = list(vectors[pos])
                out[idx] = vec
                if len(self._cache) >= max(64, int(self.cache_size)):
                    self._cache.clear()
                self._cache[str(texts[idx])] = vec
        return out

    def score_texts(self, query: str, texts: Sequence[str]) -> Sequence[float]:
        if not texts:
            return []
        query_vec = self._embed_texts([query])[0]
        doc_vecs = self._embed_texts(list(texts))
        return [_cosine(query_vec, vec) for vec in doc_vecs]
