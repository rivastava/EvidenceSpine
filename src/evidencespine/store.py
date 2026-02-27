from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_text(value: Any, default: str = "", limit: int = 2048) -> str:
    text = str(value if value is not None else default).strip()
    if not text:
        text = default
    return text[:limit]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out != out:
        out = float(default)
    return float(out)


def _parse_ts(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        pass
    text = _safe_text(value, "", 64)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return float(parsed.timestamp())
    except Exception:
        return None


@dataclass
class AgentMemoryStoreConfig:
    events_path: str = ".evidencespine/events.jsonl"
    facts_path: str = ".evidencespine/facts.jsonl"
    state_path: str = ".evidencespine/state.json"
    briefs_dir: str = ".evidencespine/briefs"
    handoffs_dir: str = ".evidencespine/handoffs"
    max_event_tail: int = 4000
    dedupe_window_sec: float = 7200.0
    redaction_enable: bool = True
    fail_open: bool = True


class AgentMemoryStore:
    _REDACTION_SKIP_KEYS = {
        "event_hash",
        "event_id",
        "fact_id",
        "packet_id",
        "checksum",
        "thread_id",
        "source_turn_id",
        "event_type",
        "state",
        "role",
    }
    _REDACTION_PATTERNS = [
        re.compile(r"\b(sk|api|token|secret)[_-]?[a-z0-9]{8,}\b", re.IGNORECASE),
        re.compile(r"\b[A-Fa-f0-9]{32,}\b"),
        re.compile(r"\b\d{12,19}\b"),
    ]

    def __init__(self, config: AgentMemoryStoreConfig | None = None) -> None:
        self.config = config or AgentMemoryStoreConfig()
        self._lock = threading.RLock()
        self.state: Dict[str, Any] = self._load_state()
        self._ensure_paths()

    def _ensure_paths(self) -> None:
        for path in [self.config.events_path, self.config.facts_path, self.config.state_path]:
            parent = os.path.dirname(str(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            if path and (not os.path.exists(path)) and path.endswith(".jsonl"):
                with open(path, "a", encoding="utf-8"):
                    pass
        os.makedirs(self.config.briefs_dir, exist_ok=True)
        os.makedirs(self.config.handoffs_dir, exist_ok=True)

    def _load_state(self) -> Dict[str, Any]:
        path = str(self.config.state_path)
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    obj = json.load(handle)
                if isinstance(obj, dict):
                    obj.setdefault("schema_version", "v1")
                    obj.setdefault("events_total", 0)
                    obj.setdefault("facts_total", 0)
                    obj.setdefault("dedupe_hits_total", 0)
                    obj.setdefault("redactions_total", 0)
                    obj.setdefault("fail_open_events_total", 0)
                    obj.setdefault("last_update_ts", None)
                    obj.setdefault("event_hash_ring", [])
                    obj.setdefault("brief_generation_attempts_total", 0)
                    obj.setdefault("brief_generation_success_total", 0)
                    obj.setdefault("brief_stale_total", 0)
                    obj.setdefault("handoff_packets_total", 0)
                    obj.setdefault("citation_claim_total", 0)
                    obj.setdefault("citation_claim_covered_total", 0)
                    return obj
            except Exception:
                pass
        return {
            "schema_version": "v1",
            "events_total": 0,
            "facts_total": 0,
            "dedupe_hits_total": 0,
            "redactions_total": 0,
            "fail_open_events_total": 0,
            "last_update_ts": None,
            "event_hash_ring": [],
            "brief_generation_attempts_total": 0,
            "brief_generation_success_total": 0,
            "brief_stale_total": 0,
            "handoff_packets_total": 0,
            "citation_claim_total": 0,
            "citation_claim_covered_total": 0,
        }

    def _save_state(self) -> None:
        payload = dict(self.state or {})
        payload["last_update_ts"] = _utc_now_iso()
        path = str(self.config.state_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        os.replace(tmp_path, path)

    def _append_jsonl(self, path: str, row: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _tail_jsonl(self, path: str, max_lines: int) -> Iterable[Dict[str, Any]]:
        if not path or (not os.path.exists(path)):
            return []
        rows: List[str] = []
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                rows.append(line)
                if len(rows) > max(1, int(max_lines)):
                    rows.pop(0)
        out: List[Dict[str, Any]] = []
        for raw in rows:
            text = str(raw or "").strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out

    def _redact_obj(self, payload: Any, key: str = "") -> Any:
        if not bool(self.config.redaction_enable):
            return payload
        key_u = str(key or "").strip().lower()
        if isinstance(payload, dict):
            return {str(k): self._redact_obj(v, str(k)) for k, v in payload.items()}
        if isinstance(payload, list):
            return [self._redact_obj(x, key) for x in payload]
        if not isinstance(payload, str):
            return payload
        if key_u in self._REDACTION_SKIP_KEYS:
            return payload
        text = payload
        redacted = text
        for pat in self._REDACTION_PATTERNS:
            redacted = pat.sub("[REDACTED]", redacted)
        if redacted != text:
            self.state["redactions_total"] = int(max(0, int(self.state.get("redactions_total", 0)))) + 1
        return redacted

    def _prune_hash_ring(self, now_ts: float) -> None:
        ring = self.state.get("event_hash_ring", [])
        if not isinstance(ring, list):
            ring = []
        window = max(60.0, float(_safe_float(self.config.dedupe_window_sec, 7200.0)))
        kept: List[Dict[str, Any]] = []
        for row in ring:
            if not isinstance(row, dict):
                continue
            ts = _parse_ts(row.get("ts"))
            if ts is None:
                continue
            if now_ts - ts <= window:
                kept.append({"event_hash": _safe_text(row.get("event_hash"), "", 128), "ts": float(ts)})
        self.state["event_hash_ring"] = kept[-max(16, int(self.config.max_event_tail)) :]

    def _is_duplicate_event_hash(self, event_hash: str) -> bool:
        ring = self.state.get("event_hash_ring", [])
        if not isinstance(ring, list):
            return False
        target = _safe_text(event_hash, "", 128)
        if not target:
            return False
        return any(_safe_text(row.get("event_hash"), "", 128) == target for row in ring if isinstance(row, dict))

    def ingest_event(self, event_row: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            try:
                now_ts = float(time.time())
                self._prune_hash_ring(now_ts)
                row = dict(event_row or {})
                row = self._redact_obj(row)
                event_hash = _safe_text(row.get("event_hash"), "", 128)
                if event_hash and self._is_duplicate_event_hash(event_hash):
                    self.state["dedupe_hits_total"] = int(max(0, int(self.state.get("dedupe_hits_total", 0)))) + 1
                    self._save_state()
                    return {
                        "status": "deduped",
                        "event_id": _safe_text(row.get("event_id"), "", 128),
                        "event_hash": event_hash,
                    }
                self._append_jsonl(str(self.config.events_path), row)
                if event_hash:
                    ring = self.state.get("event_hash_ring", [])
                    if not isinstance(ring, list):
                        ring = []
                    ring.append({"event_hash": event_hash, "ts": float(now_ts)})
                    self.state["event_hash_ring"] = ring[-max(16, int(self.config.max_event_tail)) :]
                self.state["events_total"] = int(max(0, int(self.state.get("events_total", 0)))) + 1
                self._save_state()
                return {
                    "status": "ok",
                    "event_id": _safe_text(row.get("event_id"), "", 128),
                    "event_hash": event_hash,
                }
            except Exception as exc:
                self.state["fail_open_events_total"] = int(max(0, int(self.state.get("fail_open_events_total", 0)))) + 1
                if bool(self.config.fail_open):
                    self._save_state()
                    return {"status": "fail_open", "reason": str(exc)}
                raise

    def append_fact(self, fact_row: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            try:
                row = self._redact_obj(dict(fact_row or {}))
                self._append_jsonl(str(self.config.facts_path), row)
                self.state["facts_total"] = int(max(0, int(self.state.get("facts_total", 0)))) + 1
                self._save_state()
                return {"status": "ok", "fact_id": _safe_text(row.get("fact_id"), "", 128)}
            except Exception as exc:
                self.state["fail_open_events_total"] = int(max(0, int(self.state.get("fail_open_events_total", 0)))) + 1
                if bool(self.config.fail_open):
                    self._save_state()
                    return {"status": "fail_open", "reason": str(exc)}
                raise

    def list_recent_events(self, *, thread_id: str = "", max_items: int = 128, lookback_hours: float = 24.0) -> List[Dict[str, Any]]:
        cutoff_ts = float(time.time()) - max(0.0, float(lookback_hours) * 3600.0)
        rows = list(self._tail_jsonl(str(self.config.events_path), max(self.config.max_event_tail, max_items * 4)))
        out: List[Dict[str, Any]] = []
        target_thread = _safe_text(thread_id, "", 128)
        for row in rows:
            ts = _parse_ts(row.get("ts"))
            if ts is None:
                ts = _parse_ts(row.get("ts_utc"))
            if ts is not None and ts < cutoff_ts:
                continue
            if target_thread and _safe_text(row.get("thread_id"), "", 128) != target_thread:
                continue
            out.append(row)
        return out[-max(1, int(max_items)) :]

    def list_recent_facts(
        self,
        *,
        thread_id: str = "",
        states: Optional[List[str]] = None,
        max_items: int = 128,
        lookback_hours: float = 24.0,
    ) -> List[Dict[str, Any]]:
        cutoff_ts = float(time.time()) - max(0.0, float(lookback_hours) * 3600.0)
        rows = list(self._tail_jsonl(str(self.config.facts_path), max(self.config.max_event_tail, max_items * 4)))
        target_thread = _safe_text(thread_id, "", 128)
        wanted = {str(x).strip().lower() for x in (states or []) if str(x).strip()}
        out: List[Dict[str, Any]] = []
        for row in rows:
            ts = _parse_ts(row.get("ts"))
            if ts is None:
                ts = _parse_ts(row.get("ts_utc"))
            if ts is not None and ts < cutoff_ts:
                continue
            if target_thread and _safe_text(row.get("thread_id"), "", 128) != target_thread:
                continue
            state = _safe_text(row.get("state"), "asserted", 32).lower()
            if wanted and state not in wanted:
                continue
            out.append(row)
        return out[-max(1, int(max_items)) :]

    def write_brief(self, thread_id: str, payload: Dict[str, Any]) -> str:
        with self._lock:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            safe_thread = _safe_text(thread_id, "thread", 128)
            path = os.path.join(str(self.config.briefs_dir), f"{safe_thread}_{ts}.json")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self._redact_obj(dict(payload or {})), handle, indent=2, sort_keys=True, ensure_ascii=True)
            return path

    def write_handoff(self, thread_id: str, role: str, payload: Dict[str, Any]) -> str:
        with self._lock:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            safe_thread = _safe_text(thread_id, "thread", 128)
            safe_role = _safe_text(role, "unknown", 64)
            path = os.path.join(str(self.config.handoffs_dir), f"{safe_thread}_{safe_role}_{ts}.json")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self._redact_obj(dict(payload or {})), handle, indent=2, sort_keys=True, ensure_ascii=True)
            return path

    def record_brief_stats(self, *, attempt: bool, success: bool, stale: bool, citation_total: int, citation_covered: int) -> None:
        with self._lock:
            if bool(attempt):
                self.state["brief_generation_attempts_total"] = int(max(0, int(self.state.get("brief_generation_attempts_total", 0)))) + 1
            if bool(success):
                self.state["brief_generation_success_total"] = int(max(0, int(self.state.get("brief_generation_success_total", 0)))) + 1
            if bool(stale):
                self.state["brief_stale_total"] = int(max(0, int(self.state.get("brief_stale_total", 0)))) + 1
            self.state["citation_claim_total"] = int(max(0, int(self.state.get("citation_claim_total", 0)))) + int(max(0, int(citation_total)))
            self.state["citation_claim_covered_total"] = int(max(0, int(self.state.get("citation_claim_covered_total", 0)))) + int(max(0, int(citation_covered)))
            self._save_state()

    def record_handoff_packet(self) -> None:
        with self._lock:
            self.state["handoff_packets_total"] = int(max(0, int(self.state.get("handoff_packets_total", 0)))) + 1
            self._save_state()

    def iter_handoff_files(self) -> List[str]:
        if not os.path.isdir(self.config.handoffs_dir):
            return []
        return [
            os.path.join(self.config.handoffs_dir, name)
            for name in sorted(os.listdir(self.config.handoffs_dir))
            if name.endswith(".json")
        ]

    def iter_brief_files(self) -> List[str]:
        if not os.path.isdir(self.config.briefs_dir):
            return []
        return [
            os.path.join(self.config.briefs_dir, name)
            for name in sorted(os.listdir(self.config.briefs_dir))
            if name.endswith(".json")
        ]

    def flush(self) -> Dict[str, Any]:
        with self._lock:
            self._save_state()
            return {
                "status": "ok",
                "events_total": int(max(0, int(self.state.get("events_total", 0)))),
                "facts_total": int(max(0, int(self.state.get("facts_total", 0)))),
                "last_update_ts": self.state.get("last_update_ts"),
            }
