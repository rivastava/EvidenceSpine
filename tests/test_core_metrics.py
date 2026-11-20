"""Core hardening: metrics (fail-open ring, verified facts, aliases)."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings
from evidencespine.store import FAIL_OPEN_TS_MAX


def _runtime(tmp_path: Path) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _verified_event(claim: str, *, turn: str, supersedes: str = "") -> dict:
    item = {"source_id": "src/e.py", "line_start": 1, "excerpt": "x", "checksum": _sha("x")}
    payload: dict = {"claim": claim, "fact_state": "verified"}
    if supersedes:
        payload["supersedes_ref"] = supersedes
    return {
        "thread_id": "demo",
        "event_type": "outcome",
        "role": "auditor",
        "source_agent_id": "auditor",
        "source_turn_id": turn,
        "payload": payload,
        "evidence_items": [item],
    }


# --- item 13: fail-open ring / 24h metric --------------------------------


def test_fail_open_24h_metric_uses_ring_window_and_keeps_cumulative_alias(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    now = time.time()
    rt.store.state["fail_open_events_total"] = 5
    rt.store.state["fail_open_ts"] = [now - 100.0, now - 1000.0, now - 90000.0, now - 200000.0]
    snap = rt.snapshot()
    assert snap["agent_memory_fail_open_events_24h"] == 2, "only the last 24h ring entries count"
    assert snap["agent_memory_fail_open_events_total"] == 5, "cumulative alias keeps the total counter"
    rt.store.close()


def test_fail_open_ring_is_capped(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    for _ in range(FAIL_OPEN_TS_MAX + 40):
        rt.store.record_fail_open("test", "boom")
    ring = rt.store.state["fail_open_ts"]
    assert len(ring) == FAIL_OPEN_TS_MAX
    assert rt.store.state["fail_open_events_total"] == FAIL_OPEN_TS_MAX + 40
    rt.store.close()


def test_runtime_fail_open_counter_and_ring_track_together(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt._record_fail_open("unit", "forced")
    assert rt.store.state["fail_open_events_total"] == 1
    assert len(rt.store.state["fail_open_ts"]) == 1
    assert rt.store.state["last_fail_open"]["scope"] == "unit"
    rt.store.close()


# --- item 13: verified facts exclude superseded twins ---------------------


def test_verified_facts_metric_excludes_superseded_twin_ids(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(_verified_event("older verified", turn="t1"))
    older = [f for f in rt.store.iter_facts() if f["claim"] == "older verified"][0]
    rt.ingest_event(_verified_event("newer verified", turn="t2", supersedes=older["fact_id"]))
    facts = list(rt.store.iter_facts())
    assert sum(1 for fact in facts if fact["state"] == "verified") == 2
    snap = rt.snapshot()
    assert snap["agent_memory_verified_facts_24h"] == 1, "superseded verified twins must not be counted"
    rt.store.close()


def test_snapshot_keeps_existing_metric_keys(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    snap = rt.snapshot()
    for key in (
        "events_total",
        "facts_total",
        "agent_memory_events_24h",
        "agent_memory_verified_facts_24h",
        "agent_fact_provenance_rate_24h",
        "agent_memory_contradiction_count_24h",
        "agent_brief_generation_success_rate_24h",
        "agent_handoff_packets_emitted_24h",
        "agent_evidence_stale_count_24h",
        "agent_memory_fail_open_events_24h",
        "agent_memory_fail_open_events_total",
        "brief_artifacts_unreadable",
        "handoff_artifacts_unreadable",
    ):
        assert key in snap, key
    rt.store.close()
