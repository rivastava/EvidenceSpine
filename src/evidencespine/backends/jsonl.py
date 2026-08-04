from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional
from uuid import uuid4

from evidencespine.backends.base import (
    _deep_merge,
    _filter_recent_events,
    _filter_recent_facts,
    StoreBackend,
    append_jsonl_row,
    jsonl_rows_reader,
)
from evidencespine.protocol import safe_text


class JsonlStoreBackend(StoreBackend):
    """Append-only JSONL backend (legacy).

    Preserves the original v0.4 file layout: events.jsonl / facts.jsonl under
    the store directory. Reads are tail-bounded by ``max_event_tail``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._ensure_paths()

    def _ensure_paths(self) -> None:
        for path in [str(self.config.events_path), str(self.config.facts_path)]:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if path and not os.path.exists(path):
                with open(path, "a", encoding="utf-8"):
                    pass

    def _tail_rows(self, path: str, max_lines: int) -> List[Dict[str, Any]]:
        if not path or not os.path.exists(path):
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

    def append_event(self, row: Dict[str, Any]) -> str:
        append_jsonl_row(str(self.config.events_path), dict(row))
        return "ok"

    def update_fact(self, fact_id: str, patch: Dict[str, Any]) -> bool:
        path = str(self.config.facts_path)
        if not path or not os.path.exists(path):
            return False
        rows = list(jsonl_rows_reader(path))
        found = False
        for row in rows:
            if safe_text(row.get("fact_id"), "", 128) == fact_id:
                merged = _deep_merge(row, patch)
                row.clear()
                row.update(merged)
                found = True
        if not found:
            return False
        tmp = f"{path}.{uuid4().hex[:8]}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        os.replace(tmp, path)
        return True

    def append_fact(self, row: Dict[str, Any]) -> None:
        append_jsonl_row(str(self.config.facts_path), dict(row))

    def list_recent_events(
        self,
        *,
        thread_id: str = "",
        max_items: int = 128,
        lookback_hours: float = 24.0,
    ) -> List[Dict[str, Any]]:
        rows = self._tail_rows(
            str(self.config.events_path),
            max(int(self.config.max_event_tail), int(max_items) * 4),
        )
        return _filter_recent_events(
            rows,
            thread_id=thread_id,
            max_items=max_items,
            lookback_hours=lookback_hours,
        )

    def list_recent_facts(
        self,
        *,
        thread_id: str = "",
        states: Optional[List[str]] = None,
        max_items: int = 128,
        lookback_hours: float = 24.0,
    ) -> List[Dict[str, Any]]:
        rows = self._tail_rows(
            str(self.config.facts_path),
            max(int(self.config.max_event_tail), int(max_items) * 4),
        )
        return _filter_recent_facts(
            rows,
            thread_id=thread_id,
            states=states,
            max_items=max_items,
            lookback_hours=lookback_hours,
        )

    def iter_events(self) -> Iterator[Dict[str, Any]]:
        yield from jsonl_rows_reader(str(self.config.events_path))

    def iter_facts(self) -> Iterator[Dict[str, Any]]:
        yield from jsonl_rows_reader(str(self.config.facts_path))

    def count_events(self) -> int:
        if not str(self.config.events_path) or not os.path.exists(str(self.config.events_path)):
            return 0
        return sum(1 for _ in self.iter_events())

    def count_facts(self) -> int:
        if not str(self.config.facts_path) or not os.path.exists(str(self.config.facts_path)):
            return 0
        return sum(1 for _ in self.iter_facts())
