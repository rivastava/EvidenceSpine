"""Dependency-free BM25 scoring.

Implements Robertson/Walker BM25 (k1=1.5, b=0.75 by default) over a bounded
candidate set. Built fresh per retrieval call; candidate sets are already
bounded by ``list_recent_*`` before reaching the retriever, so corpus
statistics are cheap to compute.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower())


class Bm25Scorer:
    """BM25 scorer over a document collection.

    Documents are added with an opaque key; scores are returned for query
    tokens. Raw scores are unbounded; use ``normalize()`` to scale to [0, 1].
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 0.5) -> None:
        self.k1 = max(0.1, float(k1))
        self.b = max(0.0, min(1.0, float(b)))
        self.delta = max(0.0, float(delta))
        self._docs: List[tuple[str, List[str]]] = []
        self._term_freqs: List[Dict[str, int]] = []
        self._doc_len: List[int] = []
        self._doc_freq: Dict[str, int] = {}
        self._total_len = 0

    def add_document(self, key: str, text: str) -> None:
        terms = tokenize(text)
        self._docs.append((key, terms))
        freqs: Dict[str, int] = {}
        for term in terms:
            freqs[term] = freqs.get(term, 0) + 1
        self._term_freqs.append(freqs)
        self._doc_len.append(max(1, len(terms)))
        self._total_len += max(1, len(terms))
        for term in freqs:
            self._doc_freq[term] = self._doc_freq.get(term, 0) + 1

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    def _idf(self, term: str) -> float:
        n = max(1, self._doc_freq.get(term, 0))
        n_docs = max(1, self.doc_count)
        # Robertson-Sparck-Jones IDF with smoothing; never negative.
        return math.log(1.0 + (n_docs - n + 0.5) / (n + 0.5))

    def score(self, query_terms: Sequence[str], key: str) -> float:
        """Return the raw BM25 score of ``key`` for the query terms."""
        for idx, (doc_key, _terms) in enumerate(self._docs):
            if doc_key != key:
                continue
            return self.score_index(idx, query_terms)
        return 0.0

    def score_index(self, idx: int, query_terms: Sequence[str]) -> float:
        freqs = self._term_freqs[idx]
        doc_len = self._doc_len[idx]
        avg_dl = float(self._total_len / max(1, self.doc_count))
        total = 0.0
        for term in set(query_terms):
            tf = freqs.get(term, 0)
            if tf <= 0:
                continue
            idf = self._idf(term)
            denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1e-9, avg_dl)))
            total += idf * (tf * (self.k1 + 1.0) / max(1e-9, denom))
        return total

    def scores(self, query_terms: Sequence[str]) -> List[float]:
        return [self.score_index(idx, query_terms) for idx in range(len(self._docs))]

    def keys(self) -> List[str]:
        return [key for key, _terms in self._docs]

    def normalize(self, raw: Sequence[float]) -> List[float]:
        """Scale raw scores to [0, 1] via max normalization (0 when all zero)."""
        out = [max(0.0, float(v)) for v in raw]
        peak = max(out, default=0.0)
        if peak <= 1e-9:
            return out
        return [v / peak for v in out]
