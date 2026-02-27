from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

from evidencespine.protocol import AgentConversationBrief
from evidencespine.vector_backends import VectorBackend


def _safe_text(value: Any, default: str = "", limit: int = 2048) -> str:
    text = str(value if value is not None else default).strip()
    if not text:
        text = default
    return text[:limit]


def _safe_float(value: Any, default: float = 0.0, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out != out:
        out = float(default)
    return float(min(max(out, lo), hi))


def _parse_ts(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = _safe_text(value, "", 64)
    if not text:
        return float(time.time())
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return float(parsed.timestamp())
    except Exception:
        return float(time.time())


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower())


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return float(len(sa.intersection(sb)) / max(1, len(sa.union(sb))))


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    words = max(1, len(text.split()))
    rough_chars = max(1, int(len(text) / 4))
    return max(words, rough_chars)


@dataclass
class AgentMemoryRetrieverConfig:
    top_k_events: int = 32
    top_k_facts: int = 24
    recency_half_life_hours: float = 8.0
    max_tokens: int = 1800
    retrieval_mode: str = "lexical"  # lexical | hybrid | vector
    lexical_weight: float = 1.0
    vector_weight: float = 0.35


class AgentMemoryRetriever:
    def __init__(
        self,
        config: AgentMemoryRetrieverConfig | None = None,
        *,
        vector_backend: VectorBackend | None = None,
    ) -> None:
        self.config = config or AgentMemoryRetrieverConfig()
        self.vector_backend = vector_backend

    def _recency_score(self, ts: Any, now_ts: float) -> float:
        half_life = max(0.25, float(self.config.recency_half_life_hours))
        age_h = max(0.0, (now_ts - _parse_ts(ts)) / 3600.0)
        return float(0.5 ** (age_h / half_life))

    def score_event(self, row: Dict[str, Any], query_tokens: Sequence[str], now_ts: float) -> float:
        payload = row.get("payload", {}) if isinstance(row.get("payload", {}), dict) else {}
        text = " ".join(
            [
                _safe_text(row.get("event_type"), "", 64),
                _safe_text(payload.get("claim"), "", 1024),
                " ".join([_safe_text(x, "", 128) for x in payload.get("claims", [])]) if isinstance(payload.get("claims"), list) else "",
                _safe_text(payload.get("decision"), "", 1024),
                _safe_text(payload.get("outcome"), "", 1024),
            ]
        )
        rel = _jaccard(query_tokens, _tokenize(text))
        rec = self._recency_score(row.get("ts_utc", row.get("ts")), now_ts)
        sal = _safe_float(row.get("salience", payload.get("salience", 0.5)), 0.5, 0.0, 1.0)
        evq = 1.0 if len(row.get("evidence_refs", []) or []) >= 1 else 0.2
        return float(0.45 * rel + 0.25 * rec + 0.20 * sal + 0.10 * evq)

    def score_fact(self, row: Dict[str, Any], query_tokens: Sequence[str], now_ts: float) -> float:
        claim = _safe_text(row.get("claim"), "", 2048)
        rel = _jaccard(query_tokens, _tokenize(claim))
        rec = self._recency_score(row.get("ts_utc", row.get("ts")), now_ts)
        conf = _safe_float(row.get("confidence", 0.5), 0.5, 0.0, 1.0)
        evq = 1.0 if len(row.get("evidence_refs", []) or []) >= 1 else 0.2
        return float(0.50 * rel + 0.20 * rec + 0.20 * conf + 0.10 * evq)

    def _event_text(self, row: Dict[str, Any]) -> str:
        payload = row.get("payload", {}) if isinstance(row.get("payload", {}), dict) else {}
        return " ".join(
            [
                _safe_text(row.get("event_type"), "", 64),
                _safe_text(payload.get("claim"), "", 1024),
                " ".join([_safe_text(x, "", 128) for x in payload.get("claims", [])])
                if isinstance(payload.get("claims"), list)
                else "",
                _safe_text(payload.get("decision"), "", 1024),
                _safe_text(payload.get("outcome"), "", 1024),
                _safe_text(payload.get("target"), "", 256),
            ]
        )

    def _fact_text(self, row: Dict[str, Any]) -> str:
        return " ".join(
            [
                _safe_text(row.get("claim"), "", 2048),
                _safe_text(row.get("state"), "", 32),
            ]
        )

    def _combine_scores(self, lexical: float, vector: float) -> float:
        mode = _safe_text(self.config.retrieval_mode, "lexical", 16).lower()
        lw = max(0.0, float(self.config.lexical_weight))
        vw = max(0.0, float(self.config.vector_weight))
        if mode == "vector":
            return float(vector)
        if mode == "hybrid":
            denom = max(1e-9, lw + vw)
            return float((lw * lexical + vw * vector) / denom)
        return float(lexical)

    def _vector_scores(self, query: str, texts: Sequence[str]) -> List[float]:
        if self.vector_backend is None:
            return [0.0 for _ in texts]
        try:
            scores = list(self.vector_backend.score_texts(query, texts))
        except Exception:
            return [0.0 for _ in texts]
        if len(scores) < len(texts):
            scores.extend([0.0] * (len(texts) - len(scores)))
        out: List[float] = []
        for raw in scores[: len(texts)]:
            out.append(_safe_float(raw, 0.0, 0.0, 1.0))
        return out

    def retrieve(
        self,
        *,
        query: str,
        events: Sequence[Dict[str, Any]],
        facts: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        now_ts = float(time.time())
        query_tokens = _tokenize(query)
        mode = _safe_text(self.config.retrieval_mode, "lexical", 16).lower()
        use_vector = mode in {"hybrid", "vector"} and self.vector_backend is not None

        scored_events: List[Tuple[float, Dict[str, Any]]] = []
        event_rows: List[Dict[str, Any]] = [row for row in events if isinstance(row, dict)]
        event_texts = [self._event_text(row) for row in event_rows]
        event_vector_scores = self._vector_scores(query, event_texts) if use_vector else [0.0] * len(event_rows)
        for idx, row in enumerate(event_rows):
            lexical = self.score_event(row, query_tokens, now_ts)
            vector = event_vector_scores[idx]
            score = self._combine_scores(lexical, vector)
            scored_events.append((score, row))
        scored_events.sort(key=lambda it: (it[0], _safe_text(it[1].get("event_id"), "", 128)), reverse=True)

        scored_facts: List[Tuple[float, Dict[str, Any]]] = []
        fact_rows: List[Dict[str, Any]] = [row for row in facts if isinstance(row, dict)]
        fact_texts = [self._fact_text(row) for row in fact_rows]
        fact_vector_scores = self._vector_scores(query, fact_texts) if use_vector else [0.0] * len(fact_rows)
        for idx, row in enumerate(fact_rows):
            lexical = self.score_fact(row, query_tokens, now_ts)
            vector = fact_vector_scores[idx]
            score = self._combine_scores(lexical, vector)
            scored_facts.append((score, row))
        scored_facts.sort(key=lambda it: (it[0], _safe_text(it[1].get("fact_id"), "", 128)), reverse=True)

        top_events = [dict(row) for _, row in scored_events[: max(1, int(self.config.top_k_events))]]
        top_facts = [dict(row) for _, row in scored_facts[: max(1, int(self.config.top_k_facts))]]
        return top_events, top_facts

    def _claim_with_citation(self, claim: str, refs: Sequence[str], fallback_ref: str) -> Tuple[str, List[str]]:
        clean_claim = _safe_text(claim, "", 2048)
        citations = [_safe_text(x, "", 256) for x in list(refs or []) if _safe_text(x, "", 256)]
        if not citations:
            citations = [_safe_text(fallback_ref, "unknown_ref", 256)]
        return f"{clean_claim} [ref:{citations[0]}]", citations

    def build_brief(
        self,
        *,
        thread_id: str,
        query: str,
        events: Sequence[Dict[str, Any]],
        facts: Sequence[Dict[str, Any]],
        unresolved_contradictions: Sequence[Dict[str, Any]] | None = None,
        token_budget: int | None = None,
    ) -> AgentConversationBrief:
        budget = max(64, int(token_budget if token_budget is not None else self.config.max_tokens))
        top_events, top_facts = self.retrieve(query=query, events=events, facts=facts)

        current_goal: List[str] = []
        locked_decisions: List[str] = []
        recent_verified_facts: List[str] = []
        active_risks: List[str] = []
        open_items: List[str] = []
        next_actions: List[str] = []
        citations: Dict[str, List[str]] = {}

        for event in top_events:
            payload = event.get("payload", {}) if isinstance(event.get("payload", {}), dict) else {}
            et = _safe_text(event.get("event_type"), "", 64).lower()
            if not current_goal:
                goal = _safe_text(payload.get("current_goal", payload.get("goal", "")), "", 512)
                if goal:
                    line, refs = self._claim_with_citation(goal, event.get("evidence_refs", []), _safe_text(event.get("event_id"), "evt", 128))
                    current_goal.append(line)
                    citations[line] = refs
            if et == "decision":
                decision = _safe_text(payload.get("decision", payload.get("claim", "")), "", 512)
                if decision:
                    line, refs = self._claim_with_citation(decision, event.get("evidence_refs", []), _safe_text(event.get("event_id"), "evt", 128))
                    locked_decisions.append(line)
                    citations[line] = refs
            if isinstance(payload.get("next_actions"), list):
                for action in payload.get("next_actions", [])[:3]:
                    txt = _safe_text(action, "", 512)
                    if not txt:
                        continue
                    line, refs = self._claim_with_citation(txt, event.get("evidence_refs", []), _safe_text(event.get("event_id"), "evt", 128))
                    next_actions.append(line)
                    citations[line] = refs

        for fact in top_facts:
            claim = _safe_text(fact.get("claim"), "", 1024)
            if not claim:
                continue
            state = _safe_text(fact.get("state"), "asserted", 32).lower()
            line, refs = self._claim_with_citation(claim, fact.get("evidence_refs", []), _safe_text(fact.get("fact_id"), "fact", 128))
            citations[line] = refs
            if state == "verified":
                recent_verified_facts.append(line)
            elif state == "contradicted":
                active_risks.append(f"CONTRADICTION: {line}")
            elif state == "asserted":
                open_items.append(line)

        for row in list(unresolved_contradictions or [])[:8]:
            if not isinstance(row, dict):
                continue
            note = _safe_text(row.get("reason", row.get("query", "unresolved contradiction")), "unresolved contradiction", 512)
            if note:
                active_risks.append(f"CONTRADICTION: {note}")

        ordered_sections = [
            ("current_goal", current_goal),
            ("locked_decisions", locked_decisions),
            ("recent_verified_facts", recent_verified_facts),
            ("active_risks", active_risks),
            ("open_items", open_items),
            ("next_actions", next_actions),
        ]

        used = 0
        trimmed: Dict[str, List[str]] = {name: [] for name, _ in ordered_sections}
        for name, rows in ordered_sections:
            for row in rows:
                cost = _estimate_tokens(row)
                if used + cost > budget:
                    break
                trimmed[name].append(row)
                used += cost

        return AgentConversationBrief(
            thread_id=_safe_text(thread_id, "", 128),
            query=_safe_text(query, "", 1024),
            token_budget=int(budget),
            current_goal=trimmed["current_goal"],
            locked_decisions=trimmed["locked_decisions"],
            recent_verified_facts=trimmed["recent_verified_facts"],
            active_risks=trimmed["active_risks"],
            open_items=trimmed["open_items"],
            next_actions=trimmed["next_actions"],
            citations={k: v for k, v in citations.items() if k in set(sum(trimmed.values(), []))},
            metadata={
                "token_budget": int(budget),
                "token_used_estimate": int(used),
                "top_events_used": int(len(top_events)),
                "top_facts_used": int(len(top_facts)),
            },
        )
