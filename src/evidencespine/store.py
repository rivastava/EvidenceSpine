from __future__ import annotations

import atexit
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from evidencespine.backends import JsonlStoreBackend, SqliteStoreBackend, StoreBackend
from evidencespine.protocol import parse_ts_value, safe_text, utc_now_iso

_TRUE_ENV = {"1", "true", "yes", "on", "enable", "enabled"}


def _env_flag(name: str) -> bool:
    raw = os.getenv(name)
    return bool(raw is not None and str(raw).strip().lower() in _TRUE_ENV)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out != out:
        out = float(default)
    return float(out)


@dataclass
class AgentMemoryStoreConfig:
    storage_format: str = "sqlite"  # sqlite | jsonl
    db_path: str = ".evidencespine/evidencespine.db"
    events_path: str = ".evidencespine/events.jsonl"
    facts_path: str = ".evidencespine/facts.jsonl"
    state_path: str = ".evidencespine/state.json"
    briefs_dir: str = ".evidencespine/briefs"
    handoffs_dir: str = ".evidencespine/handoffs"
    max_event_tail: int = 4000
    dedupe_window_sec: float = 7200.0
    redaction_enable: bool = True
    fail_open: bool = True
    state_flush_interval_sec: float = 1.0
    reimport_jsonl: bool = False


def _jsonl_has_rows(path: str) -> bool:
    if not path or not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if str(line or "").strip():
                    return True
    except Exception:
        return False
    return False


def auto_migrate_jsonl_to_sqlite(config: AgentMemoryStoreConfig) -> Dict[str, Any]:
    """One-time import of a legacy JSONL store into the SQLite DB.

    Only runs when the SQLite DB is empty/missing and the JSONL files contain
    rows. Best-effort and idempotent: appending the same rows again is a no-op
    because migrate_source_to_target skips rows already present by content.
    """
    if not (_jsonl_has_rows(str(config.events_path)) or _jsonl_has_rows(str(config.facts_path))):
        return {"status": "noop", "reason": "no_jsonl_rows"}
    from evidencespine.migrate import migrate_source_to_target

    return migrate_source_to_target(config, source_format="jsonl", target_format="sqlite")


def build_backend(config: AgentMemoryStoreConfig) -> StoreBackend:
    fmt = str(config.storage_format or "sqlite").strip().lower()
    if fmt == "jsonl":
        return JsonlStoreBackend(config)
    return SqliteStoreBackend(config)


FAIL_OPEN_TS_MAX = 512


class AgentMemoryStore:
    _REDACTION_SKIP_KEYS = {
        "event_hash",
        "event_id",
        "fact_id",
        "packet_id",
        "checksum",
        "source_id",
        "locator",
        "thread_id",
        "source_turn_id",
        "event_type",
        "state",
        "role",
        "line_start",
        "line_end",
        "char_start",
        "char_end",
        "scope_id",
        "scope_kind",
        "state_kind",
        "status",
        "owner_agent_id",
        "state_basis",
        "validated_at",
        "validated_by",
        "fresh_until",
        "lease_expires_at",
        "supersedes",
        "commit",
        "excerpt",
        "reference",
        "written_at",
    }
    _REDACTION_PATTERNS = [
        re.compile(r"\b(sk|api|token|secret)[_-]?[a-z0-9]{8,}\b", re.IGNORECASE),
        re.compile(r"\b[A-Fa-f0-9]{32,}\b"),
        re.compile(r"\b\d{12,19}\b"),
    ]

    def __init__(self, config: AgentMemoryStoreConfig | None = None) -> None:
        self.config = config or AgentMemoryStoreConfig()
        self._lock = threading.RLock()
        self._backend = build_backend(self.config)
        # SQLite keeps dedupe state in the dedup_hashes table (authoritative);
        # the legacy in-memory event ring is only needed/kept for JSONL.
        self._ring_enabled = not isinstance(self._backend, SqliteStoreBackend)
        self._last_state_save_ts = 0.0
        self.state: Dict[str, Any] = self._load_state()
        self._ensure_paths()
        if isinstance(self._backend, SqliteStoreBackend):
            self._auto_migrate_if_empty()
        # Short-lived CLI/hook processes exit before the debounced state flush
        # would fire; persist counters and close the backend at interpreter
        # exit so snapshot()/harness debug report truthful totals next run.
        try:
            atexit.register(self.close)
        except Exception:
            pass

    # -- state persistence ---------------------------------------------------

    def _state_defaults(self) -> Dict[str, Any]:
        return {
            "schema_version": "v3",
            "events_total": 0,
            "facts_total": 0,
            "dedupe_hits_total": 0,
            "redactions_total": 0,
            "fail_open_events_total": 0,
            "fail_open_ts": [],
            "jsonl_migration": None,
            "last_update_ts": None,
            "event_hash_ring": [],
            "brief_generation_attempts_total": 0,
            "brief_generation_success_total": 0,
            "brief_stale_total": 0,
            "handoff_packets_total": 0,
            "citation_ref_claim_total": 0,
            "citation_ref_claim_covered_total": 0,
            "citation_span_claim_total": 0,
            "citation_span_claim_covered_total": 0,
            "citation_excerpt_claim_total": 0,
            "citation_excerpt_claim_covered_total": 0,
        }

    def _load_state(self) -> Dict[str, Any]:
        path = str(self.config.state_path)
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    obj = json.load(handle)
                if isinstance(obj, dict):
                    for key, value in self._state_defaults().items():
                        obj.setdefault(key, value)
                    # Transient diagnostics must not round-trip through disk.
                    obj.pop("auto_migrated_from_jsonl", None)
                    if not self._ring_enabled:
                        obj["event_hash_ring"] = []
                    return obj
            except Exception:
                pass
        return self._state_defaults()

    def _save_state(self) -> None:
        payload = dict(self.state or {})
        now_iso = utc_now_iso()
        payload["last_update_ts"] = now_iso
        self.state["last_update_ts"] = now_iso
        payload.pop("auto_migrated_from_jsonl", None)
        if not self._ring_enabled:
            payload["event_hash_ring"] = []
        path = str(self.config.state_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = f"{path}.{uuid4().hex[:8]}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)

    def _save_state_if_due(self, *, force: bool = False) -> bool:
        """Persist in-memory state, at most once per state_flush_interval_sec.

        AgentMemoryStore.state is authoritative in memory; state.json is a
        crash-recovery snapshot, so hot append paths must not rewrite+fsync it
        on every row (that collapses ingest throughput as the ring grows).
        flush()/close()/prune()/fail-open paths force a save.
        """
        now = float(time.time())
        interval = max(0.0, float(_safe_float(self.config.state_flush_interval_sec, 1.0)))
        if not force and self._last_state_save_ts > 0.0 and (now - self._last_state_save_ts) < interval:
            return False
        self._last_state_save_ts = now
        self._save_state()
        return True

    def _auto_migrate_if_empty(self) -> None:
        """Migrate legacy JSONL rows only once, into an empty SQLite target.

        Guard order matters: a populated target is never rescanned, and after a
        successful migration a persistent jsonl_migration marker prevents
        resurrection of rows from the legacy files (e.g. after a prune empties
        the DB). EVIDENCESPINE_REIMPORT_JSONL=1 is the explicit escape hatch.
        """
        marker = self.state.get("jsonl_migration")
        reimport = bool(getattr(self.config, "reimport_jsonl", False)) or _env_flag("EVIDENCESPINE_REIMPORT_JSONL")
        if isinstance(marker, dict) and marker.get("migrated_at") and not reimport:
            return
        try:
            target_empty = int(self._backend.count_events()) == 0 and int(self._backend.count_facts()) == 0
        except Exception:
            return
        if not target_empty:
            return
        if not (_jsonl_has_rows(str(self.config.events_path)) or _jsonl_has_rows(str(self.config.facts_path))):
            return
        if reimport and isinstance(self._backend, SqliteStoreBackend):
            # Explicit reimport: legacy dedupe hashes would otherwise mark every
            # legacy event as a duplicate of the deleted rows and silently drop
            # the re-imported content.
            try:
                with self._backend._lock:
                    conn = self._backend._connect()
                    conn.execute("DELETE FROM dedup_hashes")
                    conn.commit()
            except Exception:
                pass
        try:
            migrated = auto_migrate_jsonl_to_sqlite(self.config)
        except Exception:
            return
        if migrated.get("status") == "ok":
            self.state["jsonl_migration"] = {
                "migrated_at": utc_now_iso(),
                "events": int(migrated.get("events_copied", 0)),
                "facts": int(migrated.get("facts_copied", 0)),
            }
            self.state["auto_migrated_from_jsonl"] = migrated
            try:
                # Migration writes rows behind the store's back; keep snapshot
                # totals truthful instead of reporting 0 for a populated DB.
                self.state["events_total"] = int(self._backend.count_events())
                self.state["facts_total"] = int(self._backend.count_facts())
            except Exception:
                pass
            self._save_state_if_due(force=True)

    def record_fail_open(self, scope: str = "unknown", reason: str = "") -> None:
        """Count one fail-open event and force-flush the state snapshot."""
        with self._lock:
            self._record_fail_open(scope, reason)

    def _record_fail_open(self, scope: str = "unknown", reason: str = "") -> None:
        self.state["fail_open_events_total"] = int(max(0, int(self.state.get("fail_open_events_total", 0)))) + 1
        ring = self.state.get("fail_open_ts", [])
        if not isinstance(ring, list):
            ring = []
        ring = list(ring) + [float(time.time())]
        self.state["fail_open_ts"] = ring[-FAIL_OPEN_TS_MAX:]
        self.state["last_fail_open"] = {
            "scope": safe_text(scope, "unknown", 128),
            "reason": safe_text(reason, "", 512),
            "ts": utc_now_iso(),
        }
        self._save_state_if_due(force=True)

    # -- rows ----------------------------------------------------------------

    def _ensure_paths(self) -> None:
        if isinstance(self._backend, JsonlStoreBackend):
            for path in [str(self.config.events_path), str(self.config.facts_path)]:
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                if path and (not os.path.exists(path)) and path.endswith(".jsonl"):
                    with open(path, "a", encoding="utf-8"):
                        pass
        else:
            db_path = str(self.config.db_path)
            parent = os.path.dirname(db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        for path in [str(self.config.state_path)]:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        os.makedirs(self.config.briefs_dir, exist_ok=True)
        os.makedirs(self.config.handoffs_dir, exist_ok=True)

    def redact_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Public, idempotent redaction of a full event/fact row.

        Callers that derive further rows (fact candidates, hashes) from a row
        must redact first so derived identity matches what is stored.
        """
        with self._lock:
            redacted = self._redact_obj(dict(row or {}), "")
        return dict(redacted) if isinstance(redacted, dict) else {}

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
            ts = parse_ts_value(row.get("ts"))
            if ts is None:
                continue
            if now_ts - ts <= window:
                kept.append({"event_hash": safe_text(row.get("event_hash"), "", 128), "ts": float(ts)})
        self.state["event_hash_ring"] = kept[-max(16, int(self.config.max_event_tail)) :]

    def _is_duplicate_event_hash(self, event_hash: str) -> bool:
        if not self._ring_enabled:
            return False
        ring = self.state.get("event_hash_ring", [])
        if not isinstance(ring, list):
            return False
        target = safe_text(event_hash, "", 128)
        if not target:
            return False
        return any(safe_text(row.get("event_hash"), "", 128) == target for row in ring if isinstance(row, dict))

    def ingest_event(self, event_row: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            try:
                now_ts = float(time.time())
                row = dict(event_row or {})
                row = self._redact_obj(row)
                event_hash = safe_text(row.get("event_hash"), "", 128)
                if self._ring_enabled:
                    self._prune_hash_ring(now_ts)
                    if event_hash and self._is_duplicate_event_hash(event_hash):
                        self.state["dedupe_hits_total"] = int(max(0, int(self.state.get("dedupe_hits_total", 0)))) + 1
                        self._save_state_if_due()
                        return {
                            "status": "deduped",
                            "event_id": safe_text(row.get("event_id"), "", 128),
                            "event_hash": event_hash,
                        }
                backend_status = self._backend.append_event(row)
                if backend_status == "deduped":
                    self.state["dedupe_hits_total"] = int(max(0, int(self.state.get("dedupe_hits_total", 0)))) + 1
                    self._save_state_if_due()
                    return {
                        "status": "deduped",
                        "event_id": safe_text(row.get("event_id"), "", 128),
                        "event_hash": event_hash,
                    }
                if self._ring_enabled and event_hash:
                    ring = self.state.get("event_hash_ring", [])
                    if not isinstance(ring, list):
                        ring = []
                    ring.append({"event_hash": event_hash, "ts": float(now_ts)})
                    self.state["event_hash_ring"] = ring[-max(16, int(self.config.max_event_tail)) :]
                self.state["events_total"] = int(max(0, int(self.state.get("events_total", 0)))) + 1
                self._save_state_if_due()
                return {
                    "status": "ok",
                    "event_id": safe_text(row.get("event_id"), "", 128),
                    "event_hash": event_hash,
                }
            except Exception as exc:
                if bool(self.config.fail_open):
                    self._record_fail_open("ingest_event", str(exc))
                    return {"status": "fail_open", "reason": str(exc)}
                raise

    def append_fact(self, fact_row: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            try:
                row = self._redact_obj(dict(fact_row or {}))
                self._backend.append_fact(row)
                self.state["facts_total"] = int(max(0, int(self.state.get("facts_total", 0)))) + 1
                self._save_state_if_due()
                return {"status": "ok", "fact_id": safe_text(row.get("fact_id"), "", 128)}
            except Exception as exc:
                if bool(self.config.fail_open):
                    self._record_fail_open("append_fact", str(exc))
                    return {"status": "fail_open", "reason": str(exc)}
                raise

    def update_fact(self, fact_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            try:
                found = self._backend.update_fact(safe_text(fact_id, "", 128), dict(patch or {}))
                if found:
                    self._save_state_if_due()
                    return {"status": "ok", "fact_id": safe_text(fact_id, "", 128)}
                return {"status": "missing", "fact_id": safe_text(fact_id, "", 128)}
            except Exception as exc:
                if bool(self.config.fail_open):
                    return {"status": "fail_open", "reason": str(exc)}
                raise

    def list_recent_events(self, *, thread_id: str = "", max_items: int = 128, lookback_hours: float = 24.0) -> List[Dict[str, Any]]:
        with self._lock:
            return self._backend.list_recent_events(
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
        with self._lock:
            return self._backend.list_recent_facts(
                thread_id=thread_id,
                states=states,
                max_items=max_items,
                lookback_hours=lookback_hours,
            )

    def iter_events(self, thread_id: str = "") -> Iterable[Dict[str, Any]]:
        with self._lock:
            yield from self._backend.iter_events(thread_id=thread_id)

    def iter_facts(self, thread_id: str = "") -> Iterable[Dict[str, Any]]:
        with self._lock:
            yield from self._backend.iter_facts(thread_id=thread_id)

    def count_events(self) -> int:
        with self._lock:
            return int(self._backend.count_events())

    def count_facts(self) -> int:
        with self._lock:
            return int(self._backend.count_facts())

    def prune(
        self,
        *,
        thread_id: str = "",
        ttl_hours: float = 720.0,
        ttl_hours_facts: float | None = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Delete rows older than the TTL (TTL archival).

        Rows without a parseable timestamp are always kept. With dry_run
        the backend runs against a throwaway clone so nothing is deleted.
        """
        with self._lock:
            if bool(dry_run):
                result = self._backend.count_old(
                    thread_id=thread_id,
                    ttl_hours=ttl_hours,
                    ttl_hours_facts=ttl_hours_facts,
                )
                result["dry_run"] = True
                return result
            result = self._backend.prune(
                thread_id=thread_id,
                ttl_hours=ttl_hours,
                ttl_hours_facts=ttl_hours_facts,
            )
            self.state["last_prune"] = {
                "ts": utc_now_iso(),
                "ttl_hours": float(ttl_hours),
                "events_removed": int(result.get("events_removed", 0)),
                "facts_removed": int(result.get("facts_removed", 0)),
            }
            self.state["events_total"] = int(self._backend.count_events())
            self.state["facts_total"] = int(self._backend.count_facts())
            removed_briefs = self._prune_dir_files(self.config.briefs_dir, ttl_hours=ttl_hours)
            removed_handoffs = self._prune_dir_files(self.config.handoffs_dir, ttl_hours=ttl_hours)
            result["briefs_removed"] = int(removed_briefs)
            result["handoffs_removed"] = int(removed_handoffs)
            self.flush()
            return result

    def _prune_dir_files(self, directory: str, *, ttl_hours: float) -> int:
        """Delete evidence artifact files (briefs/handoffs) older than the TTL."""
        directory = str(directory or "")
        if not directory or not os.path.isdir(directory):
            return 0
        cutoff = float(time.time()) - max(0.1, float(ttl_hours)) * 3600.0
        removed = 0
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed += 1
            except OSError:
                continue
        return removed

    def _write_artifact(self, directory: str, payload: Dict[str, Any], *, thread_id: str, role: str) -> str:
        """Persist an artifact under directory with a random filename.

        Filenames are opaque UUIDs so caller-controlled identifiers can never
        traverse out of the configured directory; thread_id/role live in the
        JSON payload only. Writes are atomic and exclusive (no silent
        overwrite within the same timestamp).
        """
        with self._lock:
            directory_real = os.path.realpath(str(directory))
            os.makedirs(directory_real, exist_ok=True)
            written_at = utc_now_iso()
            body = dict(payload or {})
            body.setdefault("thread_id", safe_text(thread_id, "thread", 128))
            body.setdefault("role", safe_text(role, "unknown", 64))
            body.setdefault("written_at", written_at)
            for _ in range(8):
                path = os.path.join(directory_real, f"{uuid4().hex}.json")
                if os.path.realpath(path).startswith(directory_real + os.sep):
                    try:
                        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                    except OSError:
                        continue
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        json.dump(self._redact_obj(body), handle, indent=2, sort_keys=True, ensure_ascii=True)
                    return path
            raise OSError("could not create artifact file under {0}".format(directory_real))

    def write_brief(self, thread_id: str, payload: Dict[str, Any]) -> str:
        return self._write_artifact(str(self.config.briefs_dir), payload, thread_id=thread_id, role="brief")

    def write_handoff(self, thread_id: str, role: str, payload: Dict[str, Any]) -> str:
        return self._write_artifact(str(self.config.handoffs_dir), payload, thread_id=thread_id, role=role)

    def record_brief_stats(
        self,
        *,
        attempt: bool,
        success: bool,
        stale: bool,
        citation_ref_total: int,
        citation_ref_covered: int,
        citation_span_total: int,
        citation_span_covered: int,
        citation_excerpt_total: int,
        citation_excerpt_covered: int,
    ) -> None:
        with self._lock:
            if bool(attempt):
                self.state["brief_generation_attempts_total"] = int(max(0, int(self.state.get("brief_generation_attempts_total", 0)))) + 1
            if bool(success):
                self.state["brief_generation_success_total"] = int(max(0, int(self.state.get("brief_generation_success_total", 0)))) + 1
            if bool(stale):
                self.state["brief_stale_total"] = int(max(0, int(self.state.get("brief_stale_total", 0)))) + 1
            self.state["citation_ref_claim_total"] = int(max(0, int(self.state.get("citation_ref_claim_total", 0)))) + int(max(0, int(citation_ref_total)))
            self.state["citation_ref_claim_covered_total"] = int(max(0, int(self.state.get("citation_ref_claim_covered_total", 0)))) + int(max(0, int(citation_ref_covered)))
            self.state["citation_span_claim_total"] = int(max(0, int(self.state.get("citation_span_claim_total", 0)))) + int(max(0, int(citation_span_total)))
            self.state["citation_span_claim_covered_total"] = int(max(0, int(self.state.get("citation_span_claim_covered_total", 0)))) + int(max(0, int(citation_span_covered)))
            self.state["citation_excerpt_claim_total"] = int(max(0, int(self.state.get("citation_excerpt_claim_total", 0)))) + int(max(0, int(citation_excerpt_total)))
            self.state["citation_excerpt_claim_covered_total"] = int(max(0, int(self.state.get("citation_excerpt_claim_covered_total", 0)))) + int(max(0, int(citation_excerpt_covered)))
            self._save_state_if_due()

    def record_handoff_packet(self) -> None:
        with self._lock:
            self.state["handoff_packets_total"] = int(max(0, int(self.state.get("handoff_packets_total", 0)))) + 1
            self._save_state_if_due()

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
            self._save_state_if_due(force=True)
            return {
                "status": "ok",
                "events_total": int(max(0, int(self.state.get("events_total", 0)))),
                "facts_total": int(max(0, int(self.state.get("facts_total", 0)))),
                "last_update_ts": self.state.get("last_update_ts"),
            }

    def close(self) -> None:
        with self._lock:
            try:
                self._save_state_if_due(force=True)
            except Exception:
                # Interpreter shutdown must never raise from atexit.
                pass
            try:
                self._backend.close()
            except Exception:
                pass
