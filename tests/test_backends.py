from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from evidencespine.migrate import migrate_source_to_target, verify_migration
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings
from grounding_utils import grounded_item
from evidencespine.store import AgentMemoryStoreConfig


def _runtime(tmp_path: Path, *, storage_format: str) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"), storage_format=storage_format)
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def _ingest_sample(rt: AgentMemoryRuntime, tmp_path: Path | None = None) -> None:
    items = [grounded_item(tmp_path)] if tmp_path is not None else []
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "t1",
            "payload": {"claim": "deploy patch", "fact_state": "verified"},
            "evidence_items": items or [{"source_id": "src/file.py", "line_start": 1, "line_end": 3}],
        }
    )
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "reflection",
            "role": "auditor",
            "source_agent_id": "audit",
            "source_turn_id": "t2",
            "payload": {"claim": "review complete"},
            "state_context": {
                "scope_id": "release-gate",
                "state_kind": "pending_gate",
                "status": "ready",
                "fresh_until": "2099-01-01T00:00:00Z",
            },
        }
    )


def test_sqlite_is_default_backend(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, storage_format="sqlite")
    _ingest_sample(rt, tmp_path)
    assert rt.store.count_events() == 2
    assert rt.store.count_facts() >= 1
    assert (tmp_path / ".es" / "evidencespine.db").exists()
    assert not (tmp_path / ".es" / "events.jsonl").exists()


def test_jsonl_backend_writes_legacy_files(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, storage_format="jsonl")
    _ingest_sample(rt, tmp_path)
    assert (tmp_path / ".es" / "events.jsonl").exists()
    assert (tmp_path / ".es" / "facts.jsonl").exists()
    assert not (tmp_path / ".es" / "evidencespine.db").exists()


def test_sqlite_and_jsonl_backends_produce_equivalent_reads(tmp_path: Path) -> None:
    sqlite_dir = tmp_path / "sqlite"
    jsonl_dir = tmp_path / "jsonl"
    sqlite_dir.mkdir()
    jsonl_dir.mkdir()

    sqlite_rt = _runtime(sqlite_dir, storage_format="sqlite")
    jsonl_rt = _runtime(jsonl_dir, storage_format="jsonl")
    _ingest_sample(sqlite_rt, tmp_path)
    _ingest_sample(jsonl_rt, tmp_path)

    sqlite_events = sqlite_rt.store.list_recent_events(max_items=100, lookback_hours=24.0)
    jsonl_events = jsonl_rt.store.list_recent_events(max_items=100, lookback_hours=24.0)
    assert sqlite_events == jsonl_events

    sqlite_facts = sqlite_rt.store.list_recent_facts(max_items=100, lookback_hours=24.0)
    jsonl_facts = jsonl_rt.store.list_recent_facts(max_items=100, lookback_hours=24.0)
    assert sqlite_facts == jsonl_facts

    sqlite_rt.store.close()
    jsonl_rt.store.close()


def test_sqlite_store_roundtrip_brief_and_handoff(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, storage_format="sqlite")
    _ingest_sample(rt, tmp_path)
    brief = rt.build_brief("demo", "what changed").to_dict()
    assert brief["recent_verified_facts"]
    packet = rt.emit_handoff("auditor", "demo", "verify").to_dict()
    assert packet["claims"]
    imported = rt.import_handoff(packet)
    assert imported["status"] == "ok"


def test_migrate_jsonl_to_sqlite(tmp_path: Path) -> None:
    base = tmp_path / ".es"
    rt = _runtime(tmp_path, storage_format="jsonl")
    _ingest_sample(rt, tmp_path)
    rt.store.close()

    settings = EvidenceSpineSettings.from_env(base_dir=str(base))
    rc = settings.to_runtime_config()
    config = AgentMemoryStoreConfig(
        storage_format="jsonl",
        db_path=str(rc.db_path),
        events_path=str(rc.events_path),
        facts_path=str(rc.facts_path),
        state_path=str(rc.state_path),
        briefs_dir=str(rc.briefs_dir),
        handoffs_dir=str(rc.handoffs_dir),
    )

    result = migrate_source_to_target(config, source_format="jsonl", target_format="sqlite")
    assert result["status"] == "ok"
    assert result["events_copied"] == 2
    assert result["facts_copied"] >= 1

    verified = verify_migration(config, source_format="jsonl", target_format="sqlite")
    assert verified["events_match"] is True
    assert verified["facts_match"] is True

    # Idempotent: running again copies nothing new.
    again = migrate_source_to_target(config, source_format="jsonl", target_format="sqlite")
    assert again["events_copied"] == 0
    assert again["facts_copied"] == 0

    # SQLite now serves the migrated rows.
    rt2 = _runtime(tmp_path, storage_format="sqlite")
    assert rt2.store.count_events() == 2


def test_sqlite_auto_migrates_existing_jsonl_store_on_first_open(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, storage_format="jsonl")
    _ingest_sample(rt, tmp_path)
    rt.store.close()

    # Opening with the SQLite default should pull legacy JSONL rows in.
    rt2 = _runtime(tmp_path, storage_format="sqlite")
    assert rt2.store.count_events() == 2
    assert rt2.store.count_facts() >= 1
    assert rt2.store.state.get("auto_migrated_from_jsonl", {}).get("status") == "ok"
    rt2.store.close()

    # Second open is a no-op (idempotent).
    rt3 = _runtime(tmp_path, storage_format="sqlite")
    assert rt3.store.count_events() == 2
    rt3.store.close()


def test_sqlite_schema_version_is_v3(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, storage_format="sqlite")
    _ingest_sample(rt, tmp_path)
    rt.store.close()
    conn = sqlite3.connect(str(tmp_path / ".es" / "evidencespine.db"))
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == 3
    finally:
        conn.close()


def test_state_schema_version_bumped_to_v3(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, storage_format="sqlite")
    assert rt.store.state.get("schema_version") == "v3"


@pytest.mark.parametrize("storage_format", ["sqlite", "jsonl"])
def test_prune_removes_rows_older_than_ttl(tmp_path: Path, storage_format: str) -> None:
    rt = _runtime(tmp_path, storage_format=storage_format)
    _ingest_sample(rt, tmp_path)
    old = time.time() - 30 * 24 * 3600.0
    rt.store._backend.append_event(
        {
            "thread_id": "demo",
            "event_type": "reflection",
            "role": "auditor",
            "source_agent_id": "audit",
            "source_turn_id": "old",
            "ts": old,
            "payload": {"claim": "ancient"},
        }
    )
    rt.store._backend.append_fact(
        {
            "thread_id": "demo",
            "claim": "ancient fact",
            "source_agent_id": "audit",
            "ts": old,
        }
    )
    assert rt.store.count_events() == 3
    assert rt.store.count_facts() >= 2

    dry = rt.prune(ttl_hours=720.0, dry_run=True)
    assert dry["dry_run"] is True
    assert dry["events_removed"] >= 1
    assert dry["facts_removed"] >= 1
    assert rt.store.count_events() == 3
    assert rt.store.count_facts() >= 2

    result = rt.prune(ttl_hours=720.0)
    assert result["status"] == "ok"
    assert result["events_removed"] >= 1
    assert result["facts_removed"] >= 1
    assert rt.store.count_events() == 2
    assert rt.store.count_facts() >= 1
    assert rt.store.state.get("last_prune", {}).get("events_removed", 0) >= 1


@pytest.mark.parametrize("storage_format", ["sqlite", "jsonl"])
def test_prune_respects_thread_id_scope(tmp_path: Path, storage_format: str) -> None:
    rt = _runtime(tmp_path, storage_format=storage_format)
    _ingest_sample(rt, tmp_path)
    old = time.time() - 30 * 24 * 3600.0
    for thread, claim in (("other", "other ancient"), ("demo", "demo ancient")):
        rt.store._backend.append_event(
            {
                "thread_id": thread,
                "event_type": "reflection",
                "role": "auditor",
                "source_agent_id": "audit",
                "source_turn_id": f"old-{thread}",
                "ts": old,
                "payload": {"claim": claim},
            }
        )

    result = rt.prune(thread_id="other", ttl_hours=720.0)
    assert result["events_removed"] == 1
    rows = list(rt.store._backend.iter_events())
    assert sum(1 for row in rows if row["thread_id"] == "other") == 0
    assert sum(1 for row in rows if row["thread_id"] == "demo") == 3


@pytest.mark.parametrize("storage_format", ["sqlite", "jsonl"])
def test_prune_keeps_rows_without_timestamp(tmp_path: Path, storage_format: str) -> None:
    rt = _runtime(tmp_path, storage_format=storage_format)
    _ingest_sample(rt, tmp_path)
    rt.store._backend.append_event(
        {
            "thread_id": "demo",
            "event_type": "reflection",
            "role": "auditor",
            "source_agent_id": "audit",
            "source_turn_id": "no-ts",
            "payload": {"claim": "no timestamp"},
        }
    )
    before = rt.store.count_events()
    result = rt.prune(ttl_hours=0.1)
    assert result["events_removed"] == 0
    assert rt.store.count_events() == before
