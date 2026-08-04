from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

from evidencespine.backends.base import _deep_merge, StoreBackend
from evidencespine.protocol import parse_ts_value, safe_text

_SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL DEFAULT '',
    thread_id TEXT NOT NULL DEFAULT '',
    ts REAL,
    row_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_thread_ts ON events(thread_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id TEXT NOT NULL DEFAULT '',
    thread_id TEXT NOT NULL DEFAULT '',
    ts REAL,
    state TEXT NOT NULL DEFAULT 'asserted',
    row_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_thread_ts ON facts(thread_id, ts);
CREATE INDEX IF NOT EXISTS idx_facts_ts ON facts(ts);
CREATE INDEX IF NOT EXISTS idx_facts_state ON facts(state);

CREATE TABLE IF NOT EXISTS dedup_hashes (
    hash TEXT PRIMARY KEY,
    ts REAL
);
"""


def _row_ts_float(row: Dict[str, Any]) -> Optional[float]:
    ts = parse_ts_value(row.get("ts"))
    if ts is None:
        ts = parse_ts_value(row.get("ts_utc"))
    return ts


class SqliteStoreBackend(StoreBackend):
    """SQLite-backed store (default).

    Uses WAL mode for concurrent readers + a single writer. Rows are stored as
    JSON blobs with indexed columns (thread_id, ts, state) so recent-row reads
    never rescan the full append log. ``PRAGMA user_version`` records the
    EvidenceSpine store schema version (3).
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _db_path(self) -> str:
        return str(getattr(self.config, "db_path", ".evidencespine/evidencespine.db"))

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            path = self._db_path()
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            conn = sqlite3.connect(path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=10000")
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version={int(_SCHEMA_VERSION)}")
            self._conn = conn
        return self._conn

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except sqlite3.Error:
                    pass
                self._conn.commit()
                self._conn.close()
                self._conn = None

    def append_event(self, row: Dict[str, Any]) -> str:
        with self._lock:
            conn = self._connect()
            event_hash = safe_text(row.get("event_hash"), "", 128)
            row_ts = _row_ts_float(row) or float(time.time())
            window = max(60.0, float(getattr(self.config, "dedupe_window_sec", 7200.0)))
            if event_hash:
                existing = conn.execute("SELECT ts FROM dedup_hashes WHERE hash = ?", (event_hash,)).fetchone()
                if existing is not None:
                    existing_ts = float(existing[0] or 0.0)
                    if time.time() - existing_ts <= window:
                        return "deduped"
                    conn.execute("UPDATE dedup_hashes SET ts = ? WHERE hash = ?", (row_ts, event_hash))
                else:
                    conn.execute("INSERT OR IGNORE INTO dedup_hashes (hash, ts) VALUES (?, ?)", (event_hash, row_ts))
            conn.execute(
                "INSERT INTO events (event_id, thread_id, ts, row_json) VALUES (?, ?, ?, ?)",
                (
                    safe_text(row.get("event_id"), "", 128),
                    safe_text(row.get("thread_id"), "", 128),
                    row_ts,
                    json.dumps(row, ensure_ascii=True),
                ),
            )
            conn.commit()
            return "ok"

    def append_fact(self, row: Dict[str, Any]) -> None:
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT INTO facts (fact_id, thread_id, ts, state, row_json) VALUES (?, ?, ?, ?, ?)",
                (
                    safe_text(row.get("fact_id"), "", 128),
                    safe_text(row.get("thread_id"), "", 128),
                    _row_ts_float(row),
                    safe_text(row.get("state"), "asserted", 32).lower(),
                    json.dumps(row, ensure_ascii=True),
                ),
            )
            conn.commit()

    def update_fact(self, fact_id: str, patch: Dict[str, Any]) -> bool:
        with self._lock:
            conn = self._connect()
            row = conn.execute(
                "SELECT id, row_json FROM facts WHERE fact_id = ? ORDER BY id DESC LIMIT 1", (fact_id,)
            ).fetchone()
            if row is None:
                return False
            merged = _deep_merge(json.loads(row[1]), patch)
            conn.execute(
                "UPDATE facts SET row_json = ?, state = ? WHERE id = ?",
                (json.dumps(merged, ensure_ascii=True), safe_text(merged.get("state"), "asserted", 32).lower(), row[0]),
            )
            conn.commit()
            return True

    def list_recent_events(
        self,
        *,
        thread_id: str = "",
        max_items: int = 128,
        lookback_hours: float = 24.0,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            cutoff = time.time() - max(0.0, float(lookback_hours)) * 3600.0
            target = safe_text(thread_id, "", 128)
            rows = conn.execute(
                "SELECT row_json FROM ("
                "  SELECT row_json, id AS sid FROM events"
                "  WHERE (ts IS NULL OR ts >= ?) AND (? = '' OR thread_id = ?)"
                "  ORDER BY id DESC LIMIT ?"
                ") ORDER BY sid ASC",
                (cutoff, target, target, max(1, int(max_items))),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def list_recent_facts(
        self,
        *,
        thread_id: str = "",
        states: Optional[List[str]] = None,
        max_items: int = 128,
        lookback_hours: float = 24.0,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            cutoff = time.time() - max(0.0, float(lookback_hours)) * 3600.0
            target = safe_text(thread_id, "", 128)
            wanted = {str(x).strip().lower() for x in (states or []) if str(x).strip()}
            state_clause = "AND state IN ({})".format(",".join("?" * len(wanted))) if wanted else ""
            params: List[Any] = [cutoff]
            if wanted:
                params.extend(sorted(wanted))
            params.extend([target, target, max(1, int(max_items))])
            rows = conn.execute(
                f"SELECT row_json FROM ("
                f"  SELECT row_json, id AS sid FROM facts"
                f"  WHERE (ts IS NULL OR ts >= ?) AND (? = '' OR thread_id = ?) {state_clause}"
                f"  ORDER BY id DESC LIMIT ?"
                f") ORDER BY sid ASC",
                params,
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def iter_events(self) -> Iterator[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute("SELECT row_json FROM events ORDER BY id ASC").fetchall()
        for r in rows:
            yield json.loads(r[0])

    def iter_facts(self) -> Iterator[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            rows = conn.execute("SELECT row_json FROM facts ORDER BY id ASC").fetchall()
        for r in rows:
            yield json.loads(r[0])

    def count_events(self) -> int:
        with self._lock:
            conn = self._connect()
            return int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def count_facts(self) -> int:
        with self._lock:
            conn = self._connect()
            return int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])

    def prune(
        self,
        *,
        thread_id: str = "",
        ttl_hours: float = 720.0,
        ttl_hours_facts: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Bulk-delete rows older than the TTL (rows without a ts are kept)."""
        cutoff = time.time() - max(0.1, float(ttl_hours)) * 3600.0
        fact_cutoff = time.time() - max(0.1, float(ttl_hours_facts if ttl_hours_facts is not None else ttl_hours)) * 3600.0
        target_thread = safe_text(thread_id, "", 128)
        with self._lock:
            conn = self._connect()
            if target_thread:
                event_cur = conn.execute(
                    "DELETE FROM events WHERE ts IS NOT NULL AND ts < ? AND thread_id = ?",
                    (cutoff, target_thread),
                )
                fact_cur = conn.execute(
                    "DELETE FROM facts WHERE ts IS NOT NULL AND ts < ? AND thread_id = ?",
                    (fact_cutoff, target_thread),
                )
            else:
                event_cur = conn.execute("DELETE FROM events WHERE ts IS NOT NULL AND ts < ?", (cutoff,))
                fact_cur = conn.execute("DELETE FROM facts WHERE ts IS NOT NULL AND ts < ?", (fact_cutoff,))
            events_removed = int(event_cur.rowcount)
            facts_removed = int(fact_cur.rowcount)
            window = max(60.0, float(getattr(self.config, "dedupe_window_sec", 7200.0)))
            hash_cutoff = time.time() - window
            dedup_removed = int(conn.execute("DELETE FROM dedup_hashes WHERE ts IS NOT NULL AND ts < ?", (hash_cutoff,)).rowcount)
            conn.commit()
            events_kept = int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
            facts_kept = int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
        return {
            "events_removed": events_removed,
            "facts_removed": facts_removed,
            "events_kept": events_kept,
            "facts_kept": facts_kept,
            "dedup_hashes_removed": dedup_removed,
        }
