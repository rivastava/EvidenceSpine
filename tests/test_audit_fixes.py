from __future__ import annotations

import hashlib
from pathlib import Path

from evidencespine.grounding import ground_ref
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings
from grounding_utils import grounded_item


def _runtime(tmp_path: Path) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def _sha_item(tmp_path: Path, token: str = "a" * 40) -> dict:
    path = tmp_path / "src" / "sha_evidence.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"TOKEN = {token!r}\n"
    path.write_text(text, encoding="utf-8")
    item = ground_ref("src/sha_evidence.py#L1-L1", source_root=str(tmp_path))
    assert item is not None
    return item


def test_redaction_does_not_corrupt_excerpt_checksum(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    item = _sha_item(tmp_path)
    out = rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "outcome",
            "role": "auditor",
            "source_agent_id": "auditor",
            "source_turn_id": "t1",
            "payload": {"claim": "sha grounded claim", "fact_state": "verified"},
            "evidence_items": [item],
        }
    )
    assert out["status"] == "ok"
    assert not out.get("policy_downgrades"), "grounded excerpt must survive the policy"
    fact = list(rt.store.iter_facts())[0]
    stored = fact["evidence_items"][0]
    assert stored["excerpt"] == item["excerpt"], "excerpt must not be redacted"
    assert stored["checksum"] == item["checksum"], "checksum must be preserved"
    assert stored["checksum"].endswith(hashlib.sha256(item["excerpt"].encode()).hexdigest())
    rt.store.close()


def test_redaction_preserves_verification_reference(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    reference = f"pytest tests/test_x.py::{'b' * 40}"
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "outcome",
            "role": "auditor",
            "source_agent_id": "auditor",
            "source_turn_id": "t1",
            "payload": {
                "claim": "provenanced claim",
                "fact_state": "verified",
                "verification": {"method": "test", "reference": reference},
            },
            "evidence_items": [grounded_item(tmp_path)],
        }
    )
    fact = list(rt.store.iter_facts())[0]
    assert fact["metadata"]["verification"]["reference"] == reference, "provenance reference must survive"
    rt.store.close()


def test_drift_check_clears_stale_on_reverify(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "outcome",
            "role": "auditor",
            "source_agent_id": "auditor",
            "source_turn_id": "t1",
            "payload": {"claim": "drift claim", "fact_state": "verified"},
            "evidence_items": [grounded_item(tmp_path, name="src/evidence.py", text="one\ntwo\n", start=1, end=2)],
        }
    )
    (tmp_path / "src" / "evidence.py").write_text("one\nCHANGED\n", encoding="utf-8")
    rt.check_evidence_stale(source_root=str(tmp_path), dry_run=False)
    assert list(rt.store.iter_facts())[0]["metadata"]["evidence_stale"] is True

    (tmp_path / "src" / "evidence.py").write_text("one\ntwo\n", encoding="utf-8")
    out = rt.check_evidence_stale(source_root=str(tmp_path), dry_run=False)
    assert out["stale_facts"] == 0
    assert out["cleared_facts"] == 1
    assert out["results"][0]["reason"] == "cleared"
    fact = list(rt.store.iter_facts())[0]
    assert "evidence_stale" not in fact["metadata"], "flag must be cleared on re-verify"
    assert rt.snapshot()["agent_evidence_stale_count_24h"] == 0
    rt.store.close()


def test_update_fact_deep_merges_metadata(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "outcome",
            "role": "auditor",
            "source_agent_id": "auditor",
            "source_turn_id": "t1",
            "payload": {"claim": "merge target", "fact_state": "verified"},
            "evidence_items": [grounded_item(tmp_path)],
        }
    )
    fact_id = list(rt.store.iter_facts())[0]["fact_id"]
    rt.store.update_fact(fact_id, {"metadata": {"verification": {"method": "manual", "reference": "x"}}})
    fact = [f for f in rt.store.iter_facts() if f["fact_id"] == fact_id][0]
    assert fact["metadata"]["verification"]["reference"] == "x"
    assert fact["metadata"]["event_id"], "existing metadata keys must survive the merge"
    rt.store.close()


def test_append_fact_on_runtime(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    out = rt.append_fact(
        {
            "thread_id": "demo",
            "claim": "raw appended fact",
            "state": "asserted",
            "source_agent_id": "a2a",
            "source_turn_id": "t1",
        }
    )
    assert out["status"] == "ok"
    facts = list(rt.store.iter_facts())
    assert any(f["claim"] == "raw appended fact" for f in facts)
    assert rt.append_fact({"thread_id": "demo", "state": "asserted"})["status"] == "invalid"
    rt.store.close()


def test_cross_process_dedup_via_sqlite(tmp_path: Path) -> None:
    base = str(tmp_path / ".es")
    first = _runtime(tmp_path)
    event = {
        "thread_id": "demo",
        "event_type": "reflection",
        "role": "operator",
        "source_agent_id": "p1",
        "source_turn_id": "t1",
        "payload": {"claim": "cross process"},
    }
    assert first.ingest_event(event)["status"] == "ok"
    first.store.close()

    second = AgentMemoryRuntime(config=EvidenceSpineSettings.from_env(base_dir=base).to_runtime_config())
    result = second.ingest_event(event)
    assert result["status"] == "deduped", "second process must dedupe against the DB"
    second.store.close()
    rt = _runtime(tmp_path)
    assert len([e for e in rt.store.iter_events() if e.get("payload", {}).get("claim") == "cross process"]) == 1
    rt.store.close()


def test_prune_recomputes_counters(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "reflection",
            "role": "operator",
            "source_agent_id": "p",
            "source_turn_id": "t-old",
            "payload": {"claim": "old"},
        }
    )
    rt.store.flush()

    # verify prune keeps counters consistent with actual rows after a prune.
    out = rt.prune(ttl_hours=0.0001, dry_run=False)
    assert out["status"] == "ok"
    snap = rt.snapshot()
    remaining = len(list(rt.store.iter_events()))
    assert snap["events_total"] == remaining, "state counter must match the store after prune"
    rt.store.close()


def test_brief_success_rate_reflects_failures(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.build_brief("demo", "status")
    snap = rt.snapshot()
    assert snap["agent_brief_generation_success_rate_24h"] == 1.0

    rt.store.state["brief_failure_ts"] = [float(__import__("time").time())]
    snap = rt.snapshot()
    assert snap["agent_brief_generation_success_rate_24h"] < 1.0
    rt.store.close()


def test_ground_rejects_path_traversal(tmp_path: Path) -> None:
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")
    assert ground_ref("../outside_secret.txt#L1-L1", source_root=str(tmp_path)) is None

    nested_root = tmp_path / "sub"
    nested_root.mkdir()
    (nested_root / "inner.txt").write_text("inner", encoding="utf-8")
    assert ground_ref("../outside_secret.txt#L1-L1", source_root=str(nested_root)) is None
    assert ground_ref("inner.txt#L1-L1", source_root=str(nested_root)) is not None


def test_git_hook_script_quotes_arguments(tmp_path: Path) -> None:
    from evidencespine.harness.git import install_git_hooks

    repo = tmp_path / "repo"
    repo.mkdir()
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    out = install_git_hooks(
        repo_dir=str(repo),
        executable="/path with spaces/evidencespine",
        base_dir=".es with spaces",
    )
    assert out["status"] == "ok"
    script = (repo / ".git" / "hooks" / "post-commit").read_text(encoding="utf-8")
    assert "'/path with spaces/evidencespine'" in script
    # base_dir is absolutized so hooks do not depend on hook-process CWD.
    assert ".es with spaces" in script
    assert "evidencespine-git-hook" in script
