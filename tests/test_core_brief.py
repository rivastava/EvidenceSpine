"""Core hardening: brief lookback/staleness, control-view grouping, token budget, citations."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from evidencespine.protocol import CITATION_LINE_MAX
from evidencespine.retriever import _estimate_tokens
from evidencespine.runtime import AgentMemoryRuntime, _control_summary
from evidencespine.settings import EvidenceSpineSettings
from grounding_utils import grounded_item


def _runtime(tmp_path: Path, *, storage_format: str = "sqlite", **overrides) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"), storage_format=storage_format)
    for key, value in overrides.items():
        setattr(settings, key, value)
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def _event(claim: str, *, thread: str = "demo", turn: str = "", ts_utc: str = "", **extra) -> dict:
    row: dict[str, Any] = {
        "thread_id": thread,
        "event_type": "reflection",
        "role": "operator",
        "source_agent_id": "agent",
        "source_turn_id": turn or f"turn-{claim}",
        "payload": {"claim": claim},
    }
    if ts_utc:
        row["ts_utc"] = ts_utc
    row.update(extra)
    return row


def _hours_ago(hours: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - hours * 3600.0))


def _sha_item(excerpt: str = "grounded excerpt") -> dict:
    return {
        "source_id": "src/evidence.py",
        "line_start": 1,
        "line_end": 1,
        "excerpt": excerpt,
        "checksum": "sha256:" + hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
    }


# --- item 4: control views group by (thread_id, scope_id) -----------------


def test_same_scope_id_in_two_threads_stays_two_rows(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    for thread in ("alpha", "beta"):
        rt.ingest_event(
            _event(
                "working",
                thread=thread,
                state_context={
                    "scope_id": "shared-scope",
                    "state_kind": "agent_local_work",
                    "status": "active",
                    "owner_agent_id": f"owner-{thread}",
                },
            )
        )
    view = rt.query_view("active_scopes", include_closed=True).to_dict()
    rows = [row for row in view["rows"] if row["scope_id"] == "shared-scope"]
    assert len(rows) == 2
    assert {row["thread_id"] for row in rows} == {"alpha", "beta"}
    assert rt.snapshot()["agent_active_scope_count_24h"] == 2, "snapshot counts per-thread rows"
    rt.store.close()


def test_control_summary_dedupes_by_thread_and_scope(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rows = [
        {
            "thread_id": "alpha",
            "scope_id": "shared",
            "state_context": {"scope_id": "shared", "state_kind": "agent_local_work", "status": "active"},
        },
        {
            "thread_id": "beta",
            "scope_id": "shared",
            "state_context": {"scope_id": "shared", "state_kind": "agent_local_work", "status": "active"},
        },
        {
            "thread_id": "alpha",
            "scope_id": "shared",
            "state_context": {"scope_id": "shared", "state_kind": "pending_gate", "status": "ready"},
        },
    ]
    summary = _control_summary(rows)
    assert summary["active_scope_count"] == 2
    assert summary["by_status"]["active"] == 2
    rt.store.close()


# --- item 5: brief lookback and staleness ---------------------------------


def test_brief_lookback_includes_ten_hour_old_events(tmp_path: Path) -> None:
    # A 24h activity window keeps a 10h-old event fresh; the lookback fix makes
    # the brief actually see it (the old 3x dedupe-window fetch missed it).
    rt = _runtime(tmp_path, dedupe_window_sec=86400.0)
    rt.ingest_event(_event("ten hour old decision", ts_utc=_hours_ago(10.0), event_type="decision", payload={"claim": "ten hour old decision", "decision": "ten hour old decision"}))
    brief = rt.build_brief("demo", "status").to_dict()
    assert any("ten hour old decision" in item for item in brief["locked_decisions"] + brief["open_items"])
    assert brief["metadata"]["stale"] is False
    assert brief["metadata"]["lookback_hours"] == 168.0
    rt.store.close()


def test_brief_is_stale_when_store_has_only_out_of_window_events(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(_event("very old claim", ts_utc=_hours_ago(400.0)))
    assert rt.store.count_events() == 1
    brief = rt.build_brief("demo", "status").to_dict()
    assert brief["metadata"]["stale"] is True, "no events in the lookback window means stale memory"
    rt.store.close()


def test_brief_is_not_stale_with_fresh_events(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(_event("fresh claim", ts_utc=_hours_ago(1.0)))
    brief = rt.build_brief("demo", "status").to_dict()
    assert brief["metadata"]["stale"] is False
    rt.store.close()


def test_brief_stale_uses_dedupe_window_floor(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, dedupe_window_sec=7200.0)
    rt.ingest_event(_event("two hour old claim", ts_utc=_hours_ago(10.0)))
    brief = rt.build_brief("demo", "status").to_dict()
    assert brief["metadata"]["stale"] is True
    rt.store.close()


# --- item 8: stale evidence excluded from verified facts ------------------


def test_stale_evidence_is_excluded_from_verified_facts_and_surfaced_as_risk(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "evidence.py").write_text("alpha\nbeta\n", encoding="utf-8")
    item = grounded_item(tmp_path, name="src/evidence.py", text="alpha\nbeta\n", start=1, end=2)
    rt.ingest_event(
        _event(
            "stale claim",
            event_type="outcome",
            payload={"claim": "stale claim", "fact_state": "verified"},
            evidence_items=[item],
        )
    )
    (tmp_path / "src" / "evidence.py").write_text("alpha\nCHANGED\n", encoding="utf-8")
    assert rt.check_evidence_stale(source_root=str(tmp_path), dry_run=False)["stale_facts"] == 1

    brief = rt.build_brief("demo", "status").to_dict()
    assert not any("stale claim" in item for item in brief["recent_verified_facts"])
    assert not any("stale claim" in item for item in brief["open_items"])
    assert any(item.startswith("STALE EVIDENCE") for item in brief["active_risks"])
    assert rt.snapshot()["agent_evidence_stale_count_24h"] == 1
    rt.store.close()


# --- item 9: citation line truncation consistency -------------------------


def test_long_claim_lines_are_capped_and_coverage_agrees_on_disk(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    long_claim = "L" * 1500
    rt.ingest_event(
        _event(
            long_claim,
            event_type="outcome",
            payload={"claim": long_claim, "fact_state": "verified"},
            evidence_items=[_sha_item("L")],
        )
    )
    brief = rt.build_brief("demo", "status").to_dict()
    claims = brief["recent_verified_facts"]
    assert claims
    assert all(len(claim) <= CITATION_LINE_MAX for claim in claims)
    assert all(len(key) <= CITATION_LINE_MAX for key in brief["citations"])
    assert all(key in brief["citations"] for claim in claims for key in [claim])
    assert rt.snapshot()["agent_claim_citation_coverage_24h"] == 1.0

    # On-disk brief artifacts use the same constant, so their coverage matches.
    files = rt.store.iter_brief_files()
    assert files
    on_disk = json.loads(Path(files[-1]).read_text(encoding="utf-8"))
    disk_claims = [claim for section in ("current_goal", "locked_decisions", "recent_verified_facts", "active_risks", "open_items", "next_actions") for claim in on_disk.get(section, [])]
    assert disk_claims
    assert all(len(claim) <= CITATION_LINE_MAX for claim in disk_claims)
    assert all(claim in on_disk["citations"] for claim in disk_claims)
    rt.store.close()


# --- item 6: script-aware token estimate and trimming ----------------------


def test_token_estimate_counts_cjk_conservatively() -> None:
    assert _estimate_tokens("你好世界") >= 4
    assert _estimate_tokens("你好世界") > _estimate_tokens("abcd")
    assert _estimate_tokens("hello world") >= 2
    assert _estimate_tokens("") == 0


def test_trimming_keeps_scanning_lower_rows_in_section(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(
        _event(
            "action event",
            event_type="decision",
            payload={"claim": "action event", "next_actions": ["X" * 800, "tiny action"]},
        )
    )
    brief = rt.build_brief("demo", "status", token_budget=64).to_dict()
    assert not any(item.startswith("XXXX") for item in brief["next_actions"])
    assert any(item.startswith("tiny action") for item in brief["next_actions"]), (
        "trimming must continue scanning cheaper lower rows instead of breaking the section"
    )
    rt.store.close()


def test_goal_is_not_displaced_by_next_actions_when_it_fits(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(
        _event(
            "goal event",
            event_type="intent",
            payload={"claim": "goal event", "current_goal": "KEEP GOAL"},
        )
    )
    rt.ingest_event(
        _event(
            "actions event",
            event_type="decision",
            payload={"claim": "actions event", "next_actions": ["tiny one", "tiny two"]},
        )
    )
    goal_cost = _estimate_tokens("KEEP GOAL [ref:ame_" + "0" * 16 + "]")
    brief = rt.build_brief("demo", "status", token_budget=max(64, goal_cost + 1)).to_dict()
    assert brief["current_goal"], "current_goal must be preserved when it fits"
    assert all(item.startswith("KEEP GOAL") for item in brief["current_goal"])
    rt.store.close()



# --- lead follow-up: durable control-view validation ----------------------


def test_query_view_unknown_view_fails_open_with_structured_metadata(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    view = rt.query_view("bogus-view").to_dict()
    assert view["rows"] == []
    assert view["view"] == "bogus-view"
    assert view["metadata"]["status"] == "invalid"
    assert view["metadata"]["reason"] == "unknown_view"
    assert "active_scopes" in view["metadata"]["valid_views"]
    rt.store.close()


def test_query_view_accepts_documented_hyphen_aliases(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(
        _event(
            "gate event",
            state_context={
                "scope_id": "release-gate",
                "state_kind": "pending_gate",
                "status": "ready",
                "fresh_until": "2099-01-01T00:00:00Z",
            },
        )
    )
    underscored = rt.query_view("open_gates", thread_id="demo").to_dict()
    hyphenated = rt.query_view("open-gates", thread_id="demo").to_dict()
    assert hyphenated["view"] == "open_gates"
    assert hyphenated["rows"] == underscored["rows"]
    assert hyphenated["metadata"]["status"] == "ok"
    rt.store.close()

# --- item 17: unreadable artifact counters --------------------------------


def test_snapshot_counts_unreadable_brief_and_handoff_artifacts(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.build_brief("demo", "status")
    rt.emit_handoff("auditor", "demo", "verify")
    (tmp_path / ".es" / "briefs" / "corrupt.json").write_text("{not json", encoding="utf-8")
    (tmp_path / ".es" / "handoffs" / "corrupt.json").write_text("{not json", encoding="utf-8")
    snap = rt.snapshot()
    assert snap["brief_artifacts_unreadable"] >= 1
    assert snap["handoff_artifacts_unreadable"] >= 1
    rt.store.close()
