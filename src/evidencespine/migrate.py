from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from evidencespine.backends import JsonlStoreBackend, SqliteStoreBackend, StoreBackend
from evidencespine.protocol import utc_now_iso
from evidencespine.store import AgentMemoryStoreConfig


def mark_jsonl_migrated(config: AgentMemoryStoreConfig, *, events: int, facts: int) -> Dict[str, Any]:
    """Record that legacy JSONL rows were deliberately migrated to SQLite.

    Explicit migrate must write the same persistent marker the auto-migration
    path uses; otherwise a later prune that empties the SQLite database would
    make the next open re-import (resurrect) the legacy JSONL rows.
    """
    path = str(getattr(config, "state_path", "") or "")
    payload: Dict[str, Any] = {}
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = {}
    payload.setdefault("schema_version", "v3")
    payload["jsonl_migration"] = {
        "migrated_at": utc_now_iso(),
        "events": int(events),
        "facts": int(facts),
        "source": "explicit_migrate",
    }
    try:
        payload["events_total"] = max(int(payload.get("events_total", 0) or 0), int(events))
        payload["facts_total"] = max(int(payload.get("facts_total", 0) or 0), int(facts))
    except (TypeError, ValueError):
        pass
    if path:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = path + "." + uuid4().hex[:8] + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    return payload["jsonl_migration"]


def _row_key(row: Dict[str, Any]) -> str:
    blob = json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8", errors="ignore")).hexdigest()


def _rows_without_dupes(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    counts = {"events_total": 0, "events_skipped": 0, "facts_total": 0, "facts_skipped": 0}
    for row in rows:
        key = _row_key(row)
        if key in seen:
            counts["events_skipped"] = int(counts.get("events_skipped", 0)) + 1
            continue
        seen.add(key)
        out.append(row)
    return out, counts


def migrate_source_to_target(
    config: AgentMemoryStoreConfig,
    *,
    source_format: str,
    target_format: str,
) -> Dict[str, Any]:
    """Copy events and facts from one store format to another, idempotently.

    Rows already present in the target (by canonical content hash) are skipped,
    so re-running migration never duplicates data.
    """
    source = build_backend_for(config, source_format)
    target = build_backend_for(config, target_format)

    existing_events = {_row_key(row) for row in target.iter_events()}
    existing_facts = {_row_key(row) for row in target.iter_facts()}

    events_copied = 0
    events_skipped = 0
    events_duplicate_source = 0
    facts_copied = 0
    facts_skipped = 0
    facts_duplicate_source = 0

    seen_events: set[str] = set()
    for row in source.iter_events():
        key = _row_key(row)
        if key in seen_events:
            events_duplicate_source += 1
            events_skipped += 1
            continue
        seen_events.add(key)
        if key in existing_events:
            events_skipped += 1
            continue
        target.append_event(row)
        existing_events.add(key)
        events_copied += 1

    seen_facts: set[str] = set()
    for row in source.iter_facts():
        key = _row_key(row)
        if key in seen_facts:
            facts_duplicate_source += 1
            facts_skipped += 1
            continue
        seen_facts.add(key)
        if key in existing_facts:
            facts_skipped += 1
            continue
        target.append_fact(row)
        existing_facts.add(key)
        facts_copied += 1

    try:
        target_events = int(target.count_events())
        target_facts = int(target.count_facts())
    except Exception:
        target_events = int(events_copied)
        target_facts = int(facts_copied)
    target.close()
    source.close()

    marker: Dict[str, Any] = {}
    if str(target_format).strip().lower() == "sqlite" and str(source_format).strip().lower() == "jsonl":
        marker = mark_jsonl_migrated(config, events=target_events, facts=target_facts)

    return {
        "status": "ok",
        "source_format": source_format,
        "target_format": target_format,
        "jsonl_migration": marker,
        "jsonl_migration_marked": bool(marker),
        "events_copied": int(events_copied),
        "events_skipped": int(events_skipped),
        "events_duplicate_source": int(events_duplicate_source),
        "facts_copied": int(facts_copied),
        "facts_skipped": int(facts_skipped),
        "facts_duplicate_source": int(facts_duplicate_source),
        "copied": {"events": int(events_copied), "facts": int(facts_copied)},
        "skipped": {"events": int(events_skipped), "facts": int(facts_skipped)},
    }


def build_backend_for(config: AgentMemoryStoreConfig, storage_format: str) -> StoreBackend:
    fmt = str(storage_format or "sqlite").strip().lower()
    if fmt == "jsonl":
        return JsonlStoreBackend(config)
    return SqliteStoreBackend(config)


def verify_migration(config: AgentMemoryStoreConfig, *, source_format: str, target_format: str) -> Dict[str, Any]:
    source = build_backend_for(config, source_format)
    target = build_backend_for(config, target_format)
    try:
        src_events = source.count_events()
        src_facts = source.count_facts()
        tgt_events = target.count_events()
        tgt_facts = target.count_facts()
        events_match = int(tgt_events) >= int(src_events)
        facts_match = int(tgt_facts) >= int(src_facts)
        return {
            "status": "verified" if (events_match and facts_match) else "mismatch",
            "source_format": source_format,
            "target_format": target_format,
            "source_events": int(src_events),
            "source_facts": int(src_facts),
            "target_events": int(tgt_events),
            "target_facts": int(tgt_facts),
            "events_match": events_match,
            "facts_match": facts_match,
        }
    finally:
        source.close()
        target.close()
