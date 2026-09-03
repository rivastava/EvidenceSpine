from __future__ import annotations

from pathlib import Path

from evidencespine.grounding import excerpt_checksum, ground_file, ground_ref, parse_ref
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings
from grounding_utils import grounded_item


def _runtime(tmp_path: Path) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def test_parse_ref_variants() -> None:
    assert parse_ref("src/a.py#L10-L20") == ("src/a.py", 10, 20)
    assert parse_ref("src/a.py#L10") == ("src/a.py", 10, 10)
    assert parse_ref("src/a.py:10-20") == ("src/a.py", 10, 20)
    assert parse_ref("noref") is None


def test_ground_ref_extracts_exact_excerpt_and_checksum(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "a.py").write_text("zero\none\ntwo\nthree\n", encoding="utf-8")
    item = ground_ref("src/a.py#L2-L3", source_root=str(tmp_path))
    assert item is not None
    assert item["source_id"] == "src/a.py"
    assert item["line_start"] == 2
    assert item["line_end"] == 3
    assert item["excerpt"] == "one\ntwo"
    assert item["checksum"] == excerpt_checksum("one\ntwo")


def test_ground_ref_returns_none_for_missing_or_out_of_range(tmp_path: Path) -> None:
    assert ground_ref("missing.py#L1", source_root=str(tmp_path)) is None
    (tmp_path / "b.py").write_text("x\n", encoding="utf-8")
    assert ground_ref("b.py#L10", source_root=str(tmp_path)) is None


def test_verified_requires_span_policy_downgrades_refs_only(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    out = rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "outcome",
            "role": "auditor",
            "source_agent_id": "auditor",
            "source_turn_id": "t1",
            "payload": {"claim": "Verified but ungrounded", "fact_state": "verified"},
            "evidence_refs": ["src/evidence.py#L1-L2"],
        }
    )
    assert out["status"] == "ok"
    assert out.get("policy_downgrades") == 1
    facts = list(rt.store.iter_facts())
    assert facts and facts[0]["state"] == "asserted"
    assert facts[0]["metadata"].get("policy") == "verified_requires_span"

    brief = rt.build_brief("demo", "status").to_dict()
    assert not any(item.startswith("Verified but ungrounded") for item in brief["recent_verified_facts"])
    assert any(item.startswith("Verified but ungrounded") for item in brief["open_items"])
    rt.store.close()


def test_verified_with_grounded_item_stays_verified(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    out = rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "outcome",
            "role": "auditor",
            "source_agent_id": "auditor",
            "source_turn_id": "t1",
            "payload": {"claim": "Grounded verified claim", "fact_state": "verified"},
            "evidence_items": [grounded_item(tmp_path)],
        }
    )
    assert out["status"] == "ok"
    assert not out.get("policy_downgrades")
    facts = list(rt.store.iter_facts())
    assert facts and facts[0]["state"] == "verified"
    rt.store.close()


def test_verified_with_verification_provenance_stays_verified(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    out = rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "outcome",
            "role": "auditor",
            "source_agent_id": "auditor",
            "source_turn_id": "t1",
            "payload": {
                "claim": "Test-proven claim",
                "fact_state": "verified",
                "verification": {"method": "test", "reference": "pytest tests/test_grounding.py"},
            },
        }
    )
    assert out["status"] == "ok"
    assert not out.get("policy_downgrades")
    facts = list(rt.store.iter_facts())
    assert facts and facts[0]["state"] == "verified"
    assert facts[0]["metadata"]["verification"]["method"] == "test"
    rt.store.close()


def test_policy_can_be_disabled(tmp_path: Path) -> None:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    settings = settings.__class__(**{**settings.__dict__, "verified_requires_span": False})
    rt = AgentMemoryRuntime(config=settings.to_runtime_config())
    out = rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "outcome",
            "role": "auditor",
            "source_agent_id": "auditor",
            "source_turn_id": "t1",
            "payload": {"claim": "Refs only verified", "fact_state": "verified"},
            "evidence_refs": ["src/a.py#L1"],
        }
    )
    assert out["status"] == "ok"
    facts = list(rt.store.iter_facts())
    assert facts and facts[0]["state"] == "verified"
    rt.store.close()


def test_ground_cli_module_roundtrip(tmp_path: Path) -> None:
    item = grounded_item(tmp_path, name="cli/proof.py")
    assert item["checksum"].startswith("sha256:")
    assert len(item["excerpt"]) > 0
    assert ground_file(item["source_id"], item["line_start"], item["line_end"], source_root=str(tmp_path)) == item


def test_verify_fact_supersedes_with_provenance(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "t1",
            "payload": {"claim": "Retry guard is correct", "fact_state": "asserted"},
            "evidence_items": [grounded_item(tmp_path)],
        }
    )
    target = list(rt.store.iter_facts())[0]["fact_id"]

    out = rt.verify_fact(target, method="test", reference="pytest tests/test_retry_guard.py", verified_by="qa", thread_id="demo")
    assert out["status"] == "ok"
    assert out.get("policy_downgrades", 0) == 0

    facts = list(rt.store.iter_facts())
    verified = [f for f in facts if f["state"] == "verified"]
    assert len(verified) == 1
    assert verified[0]["supersedes_fact_id"] == target
    assert verified[0]["metadata"]["verification"]["method"] == "test"
    assert verified[0]["metadata"]["verification"]["reference"] == "pytest tests/test_retry_guard.py"

    brief = rt.build_brief("demo", "status").to_dict()
    assert any(item.startswith("Retry guard is correct") for item in brief["recent_verified_facts"])

    snap = rt.snapshot()
    assert snap["agent_fact_provenance_rate_24h"] == 1.0
    rt.store.close()


def test_verify_fact_missing_and_invalid(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    assert rt.verify_fact("nope", method="test", reference="x")["status"] == "missing"
    assert rt.verify_fact("nope", method="bogus", reference="x")["status"] == "missing"
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "role": "impl",
            "source_agent_id": "impl",
            "source_turn_id": "t1",
            "payload": {"claim": "C", "fact_state": "asserted"},
        }
    )
    target = list(rt.store.iter_facts())[0]["fact_id"]
    assert rt.verify_fact(target, method="bogus", reference="x")["status"] == "invalid"
    assert rt.verify_fact(target, method="manual", reference="")["status"] == "invalid"
    rt.store.close()


def test_grounded_span_with_boundary_whitespace_stays_verified(tmp_path: Path) -> None:
    """Spans touching blank/indented boundary lines must verify: excerpts are
    byte-exact artifacts and normalization must not strip them."""
    from evidencespine.protocol import evidence_item_excerpt_matches_checksum

    src = tmp_path / "w.py"
    src.write_text("\n    indented = 1\n    body = 2\n\n", encoding="utf-8")
    for ref in ("w.py#L1-L3", "w.py#L2-L3", "w.py#L2-L4"):
        item = ground_ref(ref, source_root=str(tmp_path))
        assert item is not None, ref
        assert evidence_item_excerpt_matches_checksum(item), ref
    rt = _runtime(tmp_path)
    out = rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "outcome",
            "role": "auditor",
            "source_agent_id": "auditor",
            "source_turn_id": "t1",
            "payload": {"claim": "Boundary whitespace span verified", "fact_state": "verified"},
            "evidence_items": [ground_ref("w.py#L1-L4", source_root=str(tmp_path))],
        }
    )
    assert out["status"] == "ok"
    assert not out.get("policy_downgrades")
    facts = list(rt.store.iter_facts())
    assert facts and facts[0]["state"] == "verified"
    rt.store.close()


def test_ground_ref_rejects_whitespace_only_span(tmp_path: Path) -> None:
    (tmp_path / "blank.py").write_text("real = 1\n\n\n", encoding="utf-8")
    assert ground_ref("blank.py#L2-L3", source_root=str(tmp_path)) is None
    assert ground_ref("blank.py#L1-L1", source_root=str(tmp_path)) is not None
