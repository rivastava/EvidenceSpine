from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Optional
from uuid import uuid4

from evidencespine.protocol import parse_ts_value, safe_text


def _row_ts(row: Dict[str, Any]) -> Optional[float]:
    ts = parse_ts_value(row.get("ts"))
    if ts is None:
        ts = parse_ts_value(row.get("ts_utc"))
    return ts


def _filter_recent_events(
    rows: Iterable[Dict[str, Any]],
    *,
    thread_id: str = "",
    max_items: int = 128,
    lookback_hours: float = 24.0,
) -> List[Dict[str, Any]]:
    cutoff_ts = time.time() - max(0.0, float(lookback_hours)) * 3600.0
    target_thread = safe_text(thread_id, "", 128)
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ts = _row_ts(row)
        if ts is not None and ts < cutoff_ts:
            continue
        if target_thread and safe_text(row.get("thread_id"), "", 128) != target_thread:
            continue
        out.append(row)
    return out[-max(1, int(max_items)) :]


def _filter_recent_facts(
    rows: Iterable[Dict[str, Any]],
    *,
    thread_id: str = "",
    states: Optional[List[str]] = None,
    max_items: int = 128,
    lookback_hours: float = 24.0,
) -> List[Dict[str, Any]]:
    cutoff_ts = time.time() - max(0.0, float(lookback_hours)) * 3600.0
    target_thread = safe_text(thread_id, "", 128)
    wanted = {str(x).strip().lower() for x in (states or []) if str(x).strip()}
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ts = _row_ts(row)
        if ts is not None and ts < cutoff_ts:
            continue
        if target_thread and safe_text(row.get("thread_id"), "", 128) != target_thread:
            continue
        state = safe_text(row.get("state"), "asserted", 32).lower()
        if wanted and state not in wanted:
            continue
        out.append(row)
    return out[-max(1, int(max_items)) :]


class StoreBackend(ABC):
    """Persistence interface for event/fact rows.

    Backends must preserve append order for iter_events/iter_facts so that
    callers can rely on stable ordering regardless of the underlying format.
    """

    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def append_event(self, row: Dict[str, Any]) -> str:
        """Append an event row; return ``ok`` or ``deduped`` (hash already stored)."""
        ...

    @abstractmethod
    def append_fact(self, row: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def update_fact(self, fact_id: str, patch: Dict[str, Any]) -> bool:
        """Merge ``patch`` into the stored fact row; return True when found."""
        ...

    @abstractmethod
    def list_recent_events(
        self,
        *,
        thread_id: str = "",
        max_items: int = 128,
        lookback_hours: float = 24.0,
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def list_recent_facts(
        self,
        *,
        thread_id: str = "",
        states: Optional[List[str]] = None,
        max_items: int = 128,
        lookback_hours: float = 24.0,
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def iter_events(self) -> Iterator[Dict[str, Any]]:
        ...

    @abstractmethod
    def iter_facts(self) -> Iterator[Dict[str, Any]]:
        ...

    @abstractmethod
    def count_events(self) -> int:
        ...

    @abstractmethod
    def count_facts(self) -> int:
        ...

    def count_old(
        self,
        *,
        thread_id: str = "",
        ttl_hours: float = 720.0,
        ttl_hours_facts: float | None = None,
    ) -> Dict[str, Any]:
        """Count rows older than the TTL without deleting anything."""
        cutoff = time.time() - max(0.1, float(ttl_hours)) * 3600.0
        fact_cutoff = time.time() - max(0.1, float(ttl_hours_facts if ttl_hours_facts is not None else ttl_hours)) * 3600.0
        target_thread = safe_text(thread_id, "", 128)
        events_old = 0
        facts_old = 0
        events_total = 0
        facts_total = 0
        for row in self.iter_events():
            events_total += 1
            ts = _row_ts(row)
            if ts is not None and ts < cutoff:
                if not target_thread or safe_text(row.get("thread_id"), "", 128) == target_thread:
                    events_old += 1
        for row in self.iter_facts():
            facts_total += 1
            ts = _row_ts(row)
            if ts is not None and ts < fact_cutoff:
                if not target_thread or safe_text(row.get("thread_id"), "", 128) == target_thread:
                    facts_old += 1
        return {
            "events_removed": int(events_old),
            "facts_removed": int(facts_old),
            "events_kept": int(events_total - events_old),
            "facts_kept": int(facts_total - facts_old),
        }

    def prune(
        self,
        *,
        thread_id: str = "",
        ttl_hours: float = 720.0,
        ttl_hours_facts: float | None = None,
    ) -> Dict[str, Any]:
        """Delete rows older than the TTL (rows without a parseable ts are kept).

        Subclasses may override for backend-specific bulk deletes; the default
        implementation rewrites via ``iter_events``/``iter_facts`` and is
        correct but not optimal for large stores.
        """
        cutoff = time.time() - max(0.1, float(ttl_hours)) * 3600.0
        fact_cutoff = time.time() - max(0.1, float(ttl_hours_facts if ttl_hours_facts is not None else ttl_hours)) * 3600.0
        target_thread = safe_text(thread_id, "", 128)
        events_removed = 0
        kept_events: List[Dict[str, Any]] = []
        for row in self.iter_events():
            ts = _row_ts(row)
            if target_thread and safe_text(row.get("thread_id"), "", 128) != target_thread:
                kept_events.append(row)
                continue
            if ts is not None and ts < cutoff:
                events_removed += 1
                continue
            kept_events.append(row)
        facts_removed = 0
        kept_facts: List[Dict[str, Any]] = []
        for row in self.iter_facts():
            ts = _row_ts(row)
            if target_thread and safe_text(row.get("thread_id"), "", 128) != target_thread:
                kept_facts.append(row)
                continue
            if ts is not None and ts < fact_cutoff:
                facts_removed += 1
                continue
            kept_facts.append(row)
        self._replace_all(kept_events, kept_facts)
        return {
            "events_removed": int(events_removed),
            "facts_removed": int(facts_removed),
            "events_kept": int(len(kept_events)),
            "facts_kept": int(len(kept_facts)),
        }

    def _replace_all(self, events: List[Dict[str, Any]], facts: List[Dict[str, Any]]) -> None:
        """Replace all rows (used by the default ``prune``)."""
        for path, rows in ((getattr(self.config, "events_path", ""), events), (getattr(self.config, "facts_path", ""), facts)):
            if not path:
                continue
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp = f"{path}.{uuid4().hex[:8]}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            os.replace(tmp, path)

    def close(self) -> None:
        pass


def jsonl_rows_reader(path: str) -> Iterator[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = str(line or "").strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``patch`` into ``base``.

    Nested dicts merge in place; a ``None`` patch value deletes the key.
    """
    out = dict(base)
    for key, value in dict(patch or {}).items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def append_jsonl_row(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
