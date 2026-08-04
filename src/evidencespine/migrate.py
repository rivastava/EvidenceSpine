from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

from evidencespine.backends import JsonlStoreBackend, SqliteStoreBackend, StoreBackend
from evidencespine.store import AgentMemoryStoreConfig


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
    facts_copied = 0
    facts_skipped = 0

    for row in source.iter_events():
        key = _row_key(row)
        if key in existing_events:
            events_skipped += 1
            continue
        target.append_event(row)
        existing_events.add(key)
        events_copied += 1

    for row in source.iter_facts():
        key = _row_key(row)
        if key in existing_facts:
            facts_skipped += 1
            continue
        target.append_fact(row)
        existing_facts.add(key)
        facts_copied += 1

    target.close()
    source.close()

    return {
        "status": "ok",
        "source_format": source_format,
        "target_format": target_format,
        "events_copied": int(events_copied),
        "events_skipped": int(events_skipped),
        "facts_copied": int(facts_copied),
        "facts_skipped": int(facts_skipped),
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
        return {
            "source_format": source_format,
            "target_format": target_format,
            "source_events": int(src_events),
            "source_facts": int(src_facts),
            "target_events": int(tgt_events),
            "target_facts": int(tgt_facts),
            "events_match": int(tgt_events) >= int(src_events),
            "facts_match": int(tgt_facts) >= int(src_facts),
        }
    finally:
        source.close()
        target.close()
