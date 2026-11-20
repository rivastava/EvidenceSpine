"""Core hardening: state persistence, migration guard, filters, fail-open ring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings
from evidencespine.store import AgentMemoryStore, AgentMemoryStoreConfig


def _runtime(tmp_path: Path, *, storage_format: str = "sqlite", **overrides) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"), storage_format=storage_format)
    for key, value in overrides.items():
        setattr(settings, key, value)
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def _event(claim: str, *, thread: str = "demo", turn: str = "") -> dict:
    return {
        "thread_id": thread,
        "event_type": "reflection",
        "role": "operator",
        "source_agent_id": "agent",
        "source_turn_id": turn or f"turn-{claim}",
        "payload": {"claim": claim},
    }


# --- item 1: state.json hot-path debounce ---------------------------------


def test_hot_path_ingests_debounce_state_saves(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    saves = {"count": 0}

    def _counting_save() -> None:
        saves["count"] += 1

    rt.store._save_state = _counting_save  # type: ignore[method-assign]
    for idx in range(300):
        out = rt.ingest_event(_event(f"claim {idx}", turn=f"t{idx}"))
        assert out["status"] == "ok"

    assert saves["count"] <= 3, f"state.json must be debounced on hot paths, got {saves['count']} saves"
    rt.store.flush()
    assert saves["count"] <= 4, "flush must force a save"
    rt.store.close()


def test_hot_path_fact_appends_debounce_state_saves(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    saves = {"count": 0}
    rt.store._save_state = lambda: saves.__setitem__("count", saves["count"] + 1)  # type: ignore[method-assign]
    for idx in range(100):
        rt.append_fact({"thread_id": "demo", "claim": f"fact {idx}", "state": "asserted"})
    assert saves["count"] <= 3
    rt.store.close()


def test_sqlite_does_not_maintain_or_persist_event_hash_ring(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, storage_format="sqlite")
    for idx in range(3):
        rt.ingest_event(_event(f"ring claim {idx}", turn=f"r{idx}"))
    assert rt.store.state.get("event_hash_ring") == [], "sqlite dedupe lives in dedup_hashes"
    rt.store.flush()
    state = json.loads((tmp_path / ".es" / "state.json").read_text(encoding="utf-8"))
    assert state.get("event_hash_ring") == []
    rt.store.close()

    # Reopening the sqlite store must not resurrect a ring from state.json.
    rt2 = _runtime(tmp_path, storage_format="sqlite")
    assert rt2.store.state.get("event_hash_ring") == []
    rt2.store.close()


def test_jsonl_keeps_ring_in_memory_and_persists_on_flush(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, storage_format="jsonl")
    rt.ingest_event(_event("jsonl ring one", turn="j1"))
    rt.ingest_event(_event("jsonl ring two", turn="j2"))
    ring = rt.store.state.get("event_hash_ring")
    assert isinstance(ring, list) and len(ring) == 2
    rt.store.flush()
    state = json.loads((tmp_path / ".es" / "state.json").read_text(encoding="utf-8"))
    assert len(state.get("event_hash_ring", [])) == 2
    rt.store.close()


def test_state_is_authoritative_in_memory_between_flushes(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(_event("memory authoritative", turn="m1"))
    assert rt.store.count_events() == 1
    assert rt.store.state["events_total"] == 1
    flushed = rt.store.flush()
    assert flushed["events_total"] == 1
    assert flushed["last_update_ts"]
    rt.store.close()



def test_migration_marker_prevents_jsonl_resurrection_after_prune(tmp_path: Path) -> None:
    jsonl = _runtime(tmp_path, storage_format="jsonl")
    jsonl.ingest_event(_event("resurrection row", turn="res1"))
    jsonl.store.close()

    migrated = _runtime(tmp_path, storage_format="sqlite")
    assert migrated.store.count_events() == 1
    marker = migrated.store.state.get("jsonl_migration")
    assert isinstance(marker, dict) and marker.get("migrated_at")
    assert marker.get("events") == 1
    # Simulate a prune/cleanup that empties the SQLite DB.
    migrated.store._backend.prune(thread_id="", ttl_hours=720.0)
    with migrated.store._backend._lock:  # type: ignore[attr-defined]
        conn = migrated.store._backend._connect()  # type: ignore[attr-defined]
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM facts")
        conn.commit()
    assert migrated.store.count_events() == 0
    migrated.store.close()

    reopened = _runtime(tmp_path, storage_format="sqlite")
    assert reopened.store.count_events() == 0, "legacy JSONL rows must not resurrect after migration"
    assert reopened.store.count_facts() == 0
    reopened.store.close()


def test_reimport_env_allows_explicit_jsonl_reimport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jsonl = _runtime(tmp_path, storage_format="jsonl")
    jsonl.ingest_event(_event("reimport row", turn="re1"))
    jsonl.store.close()

    migrated = _runtime(tmp_path, storage_format="sqlite")
    assert migrated.store.count_events() == 1
    with migrated.store._backend._lock:  # type: ignore[attr-defined]
        conn = migrated.store._backend._connect()  # type: ignore[attr-defined]
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM facts")
        conn.commit()
    migrated.store.close()

    monkeypatch.setenv("EVIDENCESPINE_REIMPORT_JSONL", "1")
    reopened = _runtime(tmp_path, storage_format="sqlite")
    assert reopened.store.count_events() == 1, "explicit reimport env must allow re-reading legacy JSONL"
    reopened.store.close()


def test_store_ingest_fail_open_appends_ring(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, storage_format="jsonl")

    def _boom(row):
        raise RuntimeError("backend down")

    rt.store._backend.append_event = _boom  # type: ignore[method-assign]
    out = rt.store.ingest_event(_event("ingest boom", turn="ib"))
    assert out["status"] == "fail_open"
    assert len(rt.store.state.get("fail_open_ts", [])) == 1
    rt.store.close()


def test_env_bool_unrecognized_value_keeps_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EVIDENCESPINE_FAIL_OPEN", "garbage")
    monkeypatch.setenv("EVIDENCESPINE_MCP_ALLOW_UNAUTHENTICATED", "definitely-not-a-bool")
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    assert settings.fail_open is True, "garbage boolean must not silently disable fail-open"
    assert settings.mcp_allow_unauthenticated is False
    monkeypatch.setenv("EVIDENCESPINE_FAIL_OPEN", "off")
    assert EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es2")).fail_open is False
    monkeypatch.setenv("EVIDENCESPINE_FAIL_OPEN", "YES")
    assert EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es3")).fail_open is True

# --- item 2: auto-migrate guard --------------------------------------------


def test_auto_migrate_only_when_sqlite_target_is_empty(tmp_path: Path) -> None:
    jsonl = _runtime(tmp_path, storage_format="jsonl")
    jsonl.ingest_event(_event("legacy row", turn="legacy"))
    jsonl.store.close()

    first = _runtime(tmp_path, storage_format="sqlite")
    assert first.store.count_events() == 1
    assert first.store.state.get("auto_migrated_from_jsonl", {}).get("status") == "ok"
    first.store.close()

    # Re-opening a populated target must not re-run (or re-report) migration.
    second = _runtime(tmp_path, storage_format="sqlite")
    assert "auto_migrated_from_jsonl" not in second.store.state
    assert second.store.count_events() == 1
    second.store.close()


def test_auto_migrate_is_not_rescanned_when_target_populated(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(_event("seed", turn="seed"))
    rt.store.close()

    import evidencespine.store as store_module

    calls = {"count": 0}

    def _boom(config):
        calls["count"] += 1
        raise AssertionError("must not rescan for migration")

    original = store_module.auto_migrate_jsonl_to_sqlite
    store_module.auto_migrate_jsonl_to_sqlite = _boom
    try:
        again = _runtime(tmp_path)
        assert again.store.count_events() == 1
        again.store.close()
    finally:
        store_module.auto_migrate_jsonl_to_sqlite = original
    assert calls["count"] == 0


# --- item 16: thread-filtered iteration ------------------------------------


@pytest.mark.parametrize("storage_format", ["sqlite", "jsonl"])
def test_iter_events_and_facts_accept_thread_filter(tmp_path: Path, storage_format: str) -> None:
    rt = _runtime(tmp_path, storage_format=storage_format)
    rt.ingest_event(_event("thread a", thread="a", turn="a1"))
    rt.ingest_event(_event("thread b", thread="b", turn="b1"))
    assert [row["thread_id"] for row in rt.store.iter_events(thread_id="a")] == ["a"]
    assert [row["thread_id"] for row in rt.store.iter_facts(thread_id="b")] == ["b"]
    assert len(list(rt.store.iter_events())) == 2
    assert len(list(rt.store.iter_facts())) == 2
    rt.store.close()


def test_import_handoff_uses_target_thread_scan(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(_event("unrelated thread fact", thread="other", turn="o1"))
    calls: list[str] = []
    original = rt.store.iter_facts

    def _tracking_iter_facts(thread_id: str = ""):
        calls.append(thread_id)
        yield from original(thread_id=thread_id)

    rt.store.iter_facts = _tracking_iter_facts  # type: ignore[method-assign]
    out = rt.import_handoff(
        {
            "schema_version": "v2",
            "packet_id": "p",
            "role": "auditor",
            "thread_id": "demo",
            "scope": "s",
            "claims": [{"claim": "imported claim", "status": "asserted"}],
            "required_validations": [],
            "checksum": "c",
        },
        source_agent_id="auditor",
        thread_id="demo",
    )
    assert out["status"] == "ok"
    assert "demo" in calls, "import_handoff must scan only the target thread"
    rt.store.close()


# --- item 10: public redact_row --------------------------------------------


def test_redact_row_is_public_and_idempotent(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    row = {"payload": {"claim": "secret 123456789012 token"}, "metadata": {"note": "api_key12345678"}}
    first = rt.store.redact_row(row)
    assert "[REDACTED]" in first["payload"]["claim"]
    assert "[REDACTED]" in first["metadata"]["note"]
    changes = rt.store.state["redactions_total"]
    second = rt.store.redact_row(first)
    assert second == first
    assert rt.store.state["redactions_total"] == changes, "redaction must be idempotent"
    rt.store.close()


# --- item 13: fail-open ring ------------------------------------------------


def test_store_fail_open_appends_ring_and_forces_flush(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)

    def _boom(row):
        raise RuntimeError("backend down")

    rt.store._backend.append_fact = _boom  # type: ignore[method-assign]
    out = rt.store.append_fact({"fact_id": "f", "thread_id": "demo", "claim": "x"})
    assert out["status"] == "fail_open"
    ring = rt.store.state.get("fail_open_ts", [])
    assert isinstance(ring, list) and len(ring) == 1
    assert rt.store.state["fail_open_events_total"] == 1
    # Fail-open is a forced flush, so the ring is on disk immediately.
    state = json.loads((tmp_path / ".es" / "state.json").read_text(encoding="utf-8"))
    assert len(state.get("fail_open_ts", [])) == 1
    rt.store.close()


# --- item 19 (contract 6): settings additions ------------------------------


def test_settings_add_security_and_lookback_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVIDENCESPINE_BRIEF_LOOKBACK_HOURS", "72.5")
    monkeypatch.setenv("EVIDENCESPINE_STATE_FLUSH_INTERVAL_SEC", "2.5")
    monkeypatch.setenv("EVIDENCESPINE_A2A_AUTH_TOKEN", "a2a-secret")
    monkeypatch.setenv("EVIDENCESPINE_MCP_ALLOW_UNAUTHENTICATED", "true")
    monkeypatch.setenv("EVIDENCESPINE_A2A_ALLOW_UNAUTHENTICATED", "1")
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    assert settings.brief_lookback_hours == 72.5
    assert settings.state_flush_interval_sec == 2.5
    assert settings.a2a_auth_token == "a2a-secret"
    assert settings.mcp_allow_unauthenticated is True
    assert settings.a2a_allow_unauthenticated is True

    config = settings.to_runtime_config()
    assert config.brief_lookback_hours == 72.5
    assert config.state_flush_interval_sec == 2.5
    assert not hasattr(config, "a2a_auth_token"), "auth tokens stay on settings only"
    assert not hasattr(config, "mcp_allow_unauthenticated")


def test_settings_defaults_for_new_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "EVIDENCESPINE_BRIEF_LOOKBACK_HOURS",
        "EVIDENCESPINE_STATE_FLUSH_INTERVAL_SEC",
        "EVIDENCESPINE_A2A_AUTH_TOKEN",
        "EVIDENCESPINE_MCP_ALLOW_UNAUTHENTICATED",
        "EVIDENCESPINE_A2A_ALLOW_UNAUTHENTICATED",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    assert settings.brief_lookback_hours == 168.0
    assert settings.state_flush_interval_sec == 1.0
    assert settings.a2a_auth_token == ""
    assert settings.mcp_allow_unauthenticated is False
    assert settings.a2a_allow_unauthenticated is False
    assert settings.to_runtime_config().brief_lookback_hours == 168.0


def test_store_config_defaults_and_flush_interval_override(tmp_path: Path) -> None:
    config = AgentMemoryStoreConfig(
        storage_format="sqlite",
        db_path=str(tmp_path / "s.db"),
        events_path=str(tmp_path / "events.jsonl"),
        facts_path=str(tmp_path / "facts.jsonl"),
        state_path=str(tmp_path / "state.json"),
        briefs_dir=str(tmp_path / "briefs"),
        handoffs_dir=str(tmp_path / "handoffs"),
        state_flush_interval_sec=0.0,
    )
    store = AgentMemoryStore(config)
    assert store.config.state_flush_interval_sec == 0.0
    store.close()
