from __future__ import annotations

from pathlib import Path

from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings
from grounding_utils import grounded_item


def _runtime(tmp_path: Path) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def _ingest_grounded(rt: AgentMemoryRuntime, tmp_path: Path, *, claim: str = "grounded claim") -> str:
    out = rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "outcome",
            "role": "auditor",
            "source_agent_id": "auditor",
            "source_turn_id": claim.replace(" ", "-")[:40],
            "payload": {"claim": claim, "fact_state": "verified"},
            "evidence_items": [grounded_item(tmp_path, name="src/evidence.py", text="alpha\nbeta\ngamma\n", start=1, end=2)],
        }
    )
    assert out["status"] == "ok"
    return list(rt.store.iter_facts())[-1]["fact_id"]


def test_drift_check_clean_when_files_unchanged(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    _ingest_grounded(rt, tmp_path)
    out = rt.check_evidence_stale(source_root=str(tmp_path))
    assert out["status"] == "ok"
    assert out["stale_facts"] == 0
    assert out["checked_items"] == 1
    rt.store.close()


def test_drift_check_flags_changed_file(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    _ingest_grounded(rt, tmp_path)
    (tmp_path / "src" / "evidence.py").write_text("alpha\nCHANGED\n", encoding="utf-8")
    out = rt.check_evidence_stale(source_root=str(tmp_path))
    assert out["stale_facts"] == 1
    assert out["results"][0]["reason"] == "changed"

    applied = rt.check_evidence_stale(source_root=str(tmp_path), dry_run=False)
    assert applied["stale_facts"] == 1
    facts = list(rt.store.iter_facts())
    assert facts[0]["metadata"]["evidence_stale"] is True
    assert facts[0]["metadata"]["evidence_stale_reason"] == "changed"
    rt.store.close()


def test_drift_check_flags_missing_file(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    _ingest_grounded(rt, tmp_path)
    (tmp_path / "src" / "evidence.py").unlink()
    out = rt.check_evidence_stale(source_root=str(tmp_path))
    assert out["stale_facts"] == 1
    assert out["results"][0]["reason"] == "missing"
    rt.store.close()


def test_drift_check_respects_thread_filter(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    _ingest_grounded(rt, tmp_path)
    out = rt.check_evidence_stale(thread_id="other", source_root=str(tmp_path))
    assert out["stale_facts"] == 0
    assert out["checked_items"] == 0
    rt.store.close()


def test_stale_evidence_surfaces_in_view_brief_and_snapshot(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    _ingest_grounded(rt, tmp_path)
    (tmp_path / "src" / "evidence.py").write_text("alpha\nCHANGED\n", encoding="utf-8")
    rt.check_evidence_stale(source_root=str(tmp_path), dry_run=False)

    view = rt.query_view("stale_claims", thread_id="demo").to_dict()
    stale_rows = [row for row in view["rows"] if row.get("metadata", {}).get("evidence_stale")]
    assert stale_rows, "evidence-stale facts must appear in stale_claims"
    assert stale_rows[0]["state_kind"] == "evidence"

    brief = rt.build_brief("demo", "status").to_dict()
    assert any(item.startswith("STALE EVIDENCE") for item in brief["active_risks"]), (
        "stale evidence must surface in brief risks"
    )

    snap = rt.snapshot()
    assert snap["agent_evidence_stale_count_24h"] >= 1
    rt.store.close()


def test_drift_check_ignores_ungrounded_facts(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "reflection",
            "role": "auditor",
            "source_agent_id": "auditor",
            "source_turn_id": "t-ungrounded",
            "payload": {"claim": "refs only", "fact_state": "asserted"},
            "evidence_refs": ["src/evidence.py#L1-L2"],
        }
    )
    out = rt.check_evidence_stale(source_root=str(tmp_path))
    assert out["checked_items"] == 0
    assert out["stale_facts"] == 0
    rt.store.close()
