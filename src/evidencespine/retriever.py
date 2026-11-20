from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

from evidencespine.bm25 import Bm25Scorer
from evidencespine.protocol import (
    CITATION_LINE_MAX,
    AgentConversationBrief,
    ClaimCitation,
    freshness_state_for_context,
    has_grounded_span,
    merge_evidence_refs,
    normalize_evidence_items,
    normalize_state_context,
    normalize_verification,
    strip_inline_refs,
)
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


def _is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x3000 <= cp <= 0x30FF  # CJK punctuation + kana
        or 0x3400 <= cp <= 0x4DBF  # CJK ext A
        or 0x4E00 <= cp <= 0x9FFF  # CJK unified ideographs
        or 0xAC00 <= cp <= 0xD7AF  # Hangul syllables
        or 0xF900 <= cp <= 0xFAFF  # CJK compatibility ideographs
        or 0xFF00 <= cp <= 0xFFEF  # fullwidth forms
    )


def _estimate_tokens(text: str) -> int:
    """Script-aware, deliberately conservative token estimate.

    CJK characters are ~1 token each (English heuristics of chars/4 badly
    undercount them); latin text uses max(words, chars/4), the higher of the
    two common heuristics. Trimming must never overflow a real token budget.
    """
    if not text:
        return 0
    cjk = 0
    ascii_chars = 0
    ascii_words = 0
    in_word = False
    for ch in text:
        if _is_cjk_char(ch):
            cjk += 1
            in_word = False
            continue
        if ch.isspace():
            in_word = False
            continue
        ascii_chars += 1
        if not in_word:
            ascii_words += 1
            in_word = True
    latin = max(ascii_words, (ascii_chars + 3) // 4)
    return max(1, cjk + latin)


def _cap_citation_line(line: str) -> str:
    return str(line)[:CITATION_LINE_MAX]


def _event_provenance(event: Dict[str, Any]) -> Dict[str, Any]:
    payload = event.get("payload", {}) if isinstance(event.get("payload", {}), dict) else {}
    verification = normalize_verification(payload.get("verification") or event.get("verification"))
    metadata = event.get("metadata", {}) if isinstance(event.get("metadata", {}), dict) else {}
    return {
        "event_id": _safe_text(event.get("event_id"), "", 128),
        "verification": dict(verification) if verification else None,
        "evidence_stale": bool(payload.get("evidence_stale")) or bool(metadata.get("evidence_stale")),
        "metadata": dict(metadata),
    }


@dataclass
class AgentMemoryRetrieverConfig:
    top_k_events: int = 32
    top_k_facts: int = 24
    recency_half_life_hours: float = 8.0
    max_tokens: int = 1800
    retrieval_mode: str = "lexical"  # lexical | hybrid | vector
    lexical_weight: float = 1.0
    vector_weight: float = 0.35
    bm25_enabled: bool = True
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


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

    def _evidence_quality_score(self, row: Dict[str, Any]) -> float:
        evidence_items = normalize_evidence_items(row.get("evidence_items"))
        if has_grounded_span(evidence_items):
            return 1.0
        if len(merge_evidence_refs(row.get("evidence_refs"), evidence_items)) >= 1:
            return 0.7
        return 0.2

    def score_event(
        self,
        row: Dict[str, Any],
        query_tokens: Sequence[str],
        now_ts: float,
        *,
        rel: float | None = None,
    ) -> float:
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
        rel = rel if rel is not None else _jaccard(query_tokens, _tokenize(text))
        rec = self._recency_score(row.get("ts_utc", row.get("ts")), now_ts)
        sal = _safe_float(row.get("salience", payload.get("salience", 0.5)), 0.5, 0.0, 1.0)
        evq = self._evidence_quality_score(row)
        return float(0.45 * rel + 0.25 * rec + 0.20 * sal + 0.10 * evq)

    def score_fact(
        self,
        row: Dict[str, Any],
        query_tokens: Sequence[str],
        now_ts: float,
        *,
        rel: float | None = None,
    ) -> float:
        claim = _safe_text(row.get("claim"), "", 2048)
        rel = rel if rel is not None else _jaccard(query_tokens, _tokenize(claim))
        rec = self._recency_score(row.get("ts_utc", row.get("ts")), now_ts)
        conf = _safe_float(row.get("confidence", 0.5), 0.5, 0.0, 1.0)
        evq = self._evidence_quality_score(row)
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

        event_rows: List[Dict[str, Any]] = [row for row in events if isinstance(row, dict)]
        fact_rows: List[Dict[str, Any]] = [row for row in facts if isinstance(row, dict)]
        event_texts = [self._event_text(row) for row in event_rows]
        fact_texts = [self._fact_text(row) for row in fact_rows]

        bm25_events = None
        bm25_facts = None
        if bool(self.config.bm25_enabled):
            bm25_events = Bm25Scorer(k1=float(self.config.bm25_k1), b=float(self.config.bm25_b))
            bm25_facts = Bm25Scorer(k1=float(self.config.bm25_k1), b=float(self.config.bm25_b))
            for idx, row in enumerate(event_rows):
                bm25_events.add_document(_safe_text(row.get("event_id"), f"evt{idx}", 128), event_texts[idx])
            for idx, row in enumerate(fact_rows):
                bm25_facts.add_document(_safe_text(row.get("fact_id"), f"fact{idx}", 128), fact_texts[idx])
        event_bm25 = bm25_events.normalize(bm25_events.scores(query_tokens)) if bm25_events is not None else None
        fact_bm25 = bm25_facts.normalize(bm25_facts.scores(query_tokens)) if bm25_facts is not None else None

        event_vector_scores = self._vector_scores(query, event_texts) if use_vector else [0.0] * len(event_rows)
        scored_events: List[Tuple[float, Dict[str, Any]]] = []
        for idx, row in enumerate(event_rows):
            rel = event_bm25[idx] if event_bm25 is not None else _jaccard(query_tokens, _tokenize(event_texts[idx]))
            lexical = self.score_event(row, query_tokens, now_ts, rel=rel)
            vector = event_vector_scores[idx]
            score = self._combine_scores(lexical, vector)
            scored_events.append((score, row))
        scored_events.sort(key=lambda it: (it[0], _safe_text(it[1].get("event_id"), "", 128)), reverse=True)

        fact_vector_scores = self._vector_scores(query, fact_texts) if use_vector else [0.0] * len(fact_rows)
        scored_facts: List[Tuple[float, Dict[str, Any]]] = []
        for idx, row in enumerate(fact_rows):
            rel = fact_bm25[idx] if fact_bm25 is not None else _jaccard(query_tokens, _tokenize(fact_texts[idx]))
            lexical = self.score_fact(row, query_tokens, now_ts, rel=rel)
            vector = fact_vector_scores[idx]
            score = self._combine_scores(lexical, vector)
            scored_facts.append((score, row))
        scored_facts.sort(key=lambda it: (it[0], _safe_text(it[1].get("fact_id"), "", 128)), reverse=True)

        top_events = [dict(row) for _, row in scored_events[: max(1, int(self.config.top_k_events))]]
        top_facts = [dict(row) for _, row in scored_facts[: max(1, int(self.config.top_k_facts))]]
        return top_events, top_facts

    def _claim_with_citation(
        self,
        claim: str,
        refs: Sequence[str],
        evidence_items: Sequence[Dict[str, Any]] | None,
        fallback_ref: str,
        state_context: Dict[str, Any] | None,
    ) -> Tuple[str, ClaimCitation]:
        clean_claim = _safe_text(claim, "", 2048)
        merged_refs = merge_evidence_refs(refs, evidence_items)
        citation = ClaimCitation.from_value(
            {
                "primary_ref": (merged_refs[0] if merged_refs else _safe_text(fallback_ref, "unknown_ref", 256)),
                "evidence_refs": merged_refs,
                "evidence_items": list(evidence_items or []),
                "state_context": state_context,
            },
            fallback_ref=_safe_text(fallback_ref, "unknown_ref", 256),
        )
        return _cap_citation_line(f"{clean_claim} [ref:{citation.primary_ref}]"), citation

    def build_brief(
        self,
        *,
        thread_id: str,
        query: str,
        events: Sequence[Dict[str, Any]],
        facts: Sequence[Dict[str, Any]],
        unresolved_contradictions: Sequence[Dict[str, Any]] | None = None,
        token_budget: int | None = None,
        superseded_claim_keys: set[str] | None = None,
    ) -> AgentConversationBrief:
        budget = max(64, int(token_budget if token_budget is not None else self.config.max_tokens))
        top_events, top_facts = self.retrieve(query=query, events=events, facts=facts)
        superseded = {str(key).strip().lower() for key in (superseded_claim_keys or set())}

        def _is_superseded(text: str) -> bool:
            clean, _ = strip_inline_refs(_safe_text(text, "", 2048))
            return clean.strip().lower() in superseded

        current_goal: List[str] = []
        locked_decisions: List[str] = []
        recent_verified_facts: List[str] = []
        active_risks: List[str] = []
        open_items: List[str] = []
        next_actions: List[str] = []
        citations: Dict[str, ClaimCitation] = {}
        provenance: Dict[str, Dict[str, Any]] = {}

        for event in top_events:
            payload = event.get("payload", {}) if isinstance(event.get("payload", {}), dict) else {}
            et = _safe_text(event.get("event_type"), "", 64).lower()
            state_context = normalize_state_context(event.get("state_context")) if "state_context" in event else None
            if not current_goal:
                goal = _safe_text(payload.get("current_goal", payload.get("goal", "")), "", 512)
                if goal and not _is_superseded(goal):
                    line, citation = self._claim_with_citation(
                        goal,
                        event.get("evidence_refs", []),
                        event.get("evidence_items", []),
                        _safe_text(event.get("event_id"), "evt", 128),
                        state_context,
                    )
                    current_goal.append(line)
                    citations[line] = citation
                    provenance[line] = _event_provenance(event)
            if et == "decision":
                decision = _safe_text(payload.get("decision", payload.get("claim", "")), "", 512)
                if decision and not _is_superseded(decision):
                    line, citation = self._claim_with_citation(
                        decision,
                        event.get("evidence_refs", []),
                        event.get("evidence_items", []),
                        _safe_text(event.get("event_id"), "evt", 128),
                        state_context,
                    )
                    locked_decisions.append(line)
                    citations[line] = citation
                    provenance[line] = _event_provenance(event)
            if isinstance(payload.get("next_actions"), list):
                for action in payload.get("next_actions", [])[:3]:
                    txt = _safe_text(action, "", 512)
                    if not txt or _is_superseded(txt):
                        continue
                    line, citation = self._claim_with_citation(
                        txt,
                        event.get("evidence_refs", []),
                        event.get("evidence_items", []),
                        _safe_text(event.get("event_id"), "evt", 128),
                        state_context,
                    )
                    next_actions.append(line)
                    citations[line] = citation
                    provenance[line] = _event_provenance(event)

        for fact in top_facts:
            claim = _safe_text(fact.get("claim"), "", 1024)
            if not claim:
                continue
            state = _safe_text(fact.get("state"), "asserted", 32).lower()
            state_context = normalize_state_context(fact.get("state_context")) if "state_context" in fact else None
            line, citation = self._claim_with_citation(
                claim,
                fact.get("evidence_refs", []),
                fact.get("evidence_items", []),
                _safe_text(fact.get("fact_id"), "fact", 128),
                state_context,
            )
            citations[line] = citation
            fact_meta = fact.get("metadata", {}) if isinstance(fact.get("metadata", {}), dict) else {}
            fact_verification = normalize_verification(fact_meta.get("verification") or fact.get("verification"))
            provenance[line] = {
                "fact_id": _safe_text(fact.get("fact_id"), "", 128),
                "verification": dict(fact_verification) if fact_verification else None,
                "evidence_stale": bool(fact_meta.get("evidence_stale")),
                "metadata": dict(fact_meta),
            }
            if bool(fact_meta.get("evidence_stale")):
                # Stale evidence is a risk, never a verified fact: it must not
                # sit in recent_verified_facts or silently pass as open work.
                reason = _safe_text(fact_meta.get("evidence_stale_reason"), "changed", 16)
                risk_line = _cap_citation_line(f"STALE EVIDENCE ({reason}): {line}")
                active_risks.append(risk_line)
                citations[risk_line] = citation
                provenance[risk_line] = dict(provenance[line])
                continue
            if state == "verified":
                recent_verified_facts.append(line)
            elif state == "contradicted":
                risk_line = _cap_citation_line(f"CONTRADICTION: {line}")
                active_risks.append(risk_line)
                citations[risk_line] = citation
                provenance[risk_line] = dict(provenance[line])
            elif state == "asserted":
                open_items.append(line)

        for row in list(unresolved_contradictions or [])[:8]:
            if not isinstance(row, dict):
                continue
            note = _safe_text(row.get("reason", row.get("query", "unresolved contradiction")), "unresolved contradiction", 512)
            if note:
                state_context = normalize_state_context(row.get("state_context")) if "state_context" in row else None
                line, citation = self._claim_with_citation(
                    note,
                    row.get("evidence_refs", []),
                    row.get("evidence_items", []),
                    _safe_text(row.get("claim", "contradiction"), "contradiction", 128),
                    state_context,
                )
                risk_line = _cap_citation_line(f"CONTRADICTION: {line}")
                active_risks.append(risk_line)
                citations[risk_line] = citation
                verification = normalize_verification(row.get("verification"))
                provenance[risk_line] = {
                    "verification": dict(verification) if verification else None,
                    "evidence_stale": bool(row.get("evidence_stale")),
                    "metadata": dict(row.get("metadata", {}) or {}) if isinstance(row.get("metadata", {}), dict) else {},
                }

        ordered_sections = [
            ("current_goal", current_goal),
            ("locked_decisions", locked_decisions),
            ("recent_verified_facts", recent_verified_facts),
            ("active_risks", active_risks),
            ("open_items", open_items),
            ("next_actions", next_actions),
        ]

        def _section_key(text: str) -> str:
            clean, _ = strip_inline_refs(text)
            return clean.strip().lower()

        deduped: Dict[str, List[str]] = {name: [] for name, _ in ordered_sections}
        for name, rows in ordered_sections:
            seen: set[str] = set()
            for line in rows:
                key = _section_key(line)
                if not key or key in seen:
                    continue
                seen.add(key)
                deduped[name].append(line)
        if deduped["open_items"]:
            stronger = {
                _section_key(line)
                for section in ("locked_decisions", "recent_verified_facts")
                for line in deduped[section]
            }
            if stronger:
                deduped["open_items"] = [
                    line for line in deduped["open_items"] if _section_key(line) not in stronger
                ]
        ordered_sections = [(name, deduped[name]) for name, _ in ordered_sections]

        # Greedy fill in priority order: when a row does not fit, keep scanning
        # the remaining (possibly cheaper/lower) rows instead of abandoning the
        # section. Processing sections in priority order means high-priority
        # sections (current_goal first) are never displaced by lower ones.
        used = 0
        trimmed: Dict[str, List[str]] = {name: [] for name, _ in ordered_sections}
        for name, rows in ordered_sections:
            for line in rows:
                cost = _estimate_tokens(line)
                if used + cost > budget:
                    continue
                trimmed[name].append(line)
                used += cost

        included_rows = set(sum(trimmed.values(), []))
        included_citations = {key: value for key, value in citations.items() if key in included_rows}
        included_provenance = {key: value for key, value in provenance.items() if key in included_rows}

        now_ts = float(time.time())
        active_scope_ids: set[str] = set()
        open_gate_scope_ids: set[str] = set()
        stale_scope_ids: set[str] = set()
        for citation in included_citations.values():
            state_context = normalize_state_context(citation.state_context) if citation.state_context is not None else {}
            scope_id = _safe_text(state_context.get("scope_id"), "", 256)
            if not scope_id:
                continue
            active_scope_ids.add(scope_id)
            if _safe_text(state_context.get("state_kind"), "", 64) == "pending_gate" and _safe_text(state_context.get("status"), "", 64) != "closed":
                open_gate_scope_ids.add(scope_id)
            if freshness_state_for_context(state_context, now_ts=now_ts) == "stale":
                stale_scope_ids.add(scope_id)

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
            citations=included_citations,
            metadata={
                "token_budget": int(budget),
                "token_used_estimate": int(used),
                "top_events_used": int(len(top_events)),
                "top_facts_used": int(len(top_facts)),
                "active_scope_count": int(len(active_scope_ids)),
                "open_gate_count": int(len(open_gate_scope_ids)),
                "stale_scope_count": int(len(stale_scope_ids)),
                # Provenance per citation line so emit_handoff can carry
                # verification/staleness through the packet schema.
                "provenance": included_provenance,
            },
        )
