from __future__ import annotations

import subprocess
from pathlib import Path

from evidencespine.harness.git import install_git_hooks, record_commit, record_test_result
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def _runtime(tmp_path: Path) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "mod.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


def test_record_commit_ingests_event_with_grounded_spans(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "mod.py").write_text("def a():\n    return 1\n\ndef b():\n    return 2\n", encoding="utf-8")
    _git(repo, "add", ".")
    sha = _git(repo, "commit", "-q", "-m", "add function b")
    sha = _git(repo, "rev-parse", "HEAD")

    out = record_commit(sha, repo_dir=str(repo), base_dir=str(tmp_path / ".es"))
    assert out["status"] == "ok"
    assert out["items"] >= 1, "commit must produce grounded spans"

    rt = _runtime(tmp_path)
    events = list(rt.store.iter_events())
    commit_event = [e for e in events if e.get("metadata", {}).get("commit") == sha]
    assert commit_event, "commit event must be stored"
    items = commit_event[0].get("evidence_items", [])
    assert items and all(i.get("excerpt") and i.get("checksum") for i in items), "spans must be grounded"
    assert any("add function b" in str(e.get("payload", {}).get("claim", "")) for e in events)
    rt.store.close()


def test_record_test_result_passed_is_verified_with_provenance(tmp_path: Path) -> None:
    out = record_test_result("passed", "pytest tests/test_demo.py", base_dir=str(tmp_path / ".es"))
    assert out["status"] == "ok" and out["passed"] is True

    rt = _runtime(tmp_path)
    facts = list(rt.store.iter_facts())
    assert facts and facts[0]["state"] == "verified", "green test runs must be verified"
    assert facts[0]["metadata"]["verification"]["method"] == "test"
    assert facts[0]["metadata"]["verification"]["reference"] == "pytest tests/test_demo.py"
    assert out["ingest"].get("policy_downgrades", 0) == 0, "provenance must satisfy the verified policy"
    rt.store.close()


def test_record_test_result_failed_is_asserted(tmp_path: Path) -> None:
    out = record_test_result("failed", "pytest tests/test_demo.py", base_dir=str(tmp_path / ".es"))
    assert out["status"] == "ok" and out["passed"] is False
    rt = _runtime(tmp_path)
    facts = list(rt.store.iter_facts())
    assert facts and facts[0]["state"] == "asserted"
    rt.store.close()


def test_install_git_hooks_writes_scripts(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = install_git_hooks(repo_dir=str(repo), executable="evidencespine", base_dir=".evidencespine")
    assert out["status"] == "ok"
    for name in ("post-commit", "post-merge"):
        script = repo / ".git" / "hooks" / name
        assert script.exists()
        assert "evidencespine" in script.read_text(encoding="utf-8")
        assert script.stat().st_mode & 0o111, "hook must be executable"


def test_record_commit_missing_sha_fails_open(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = record_commit("deadbeef" * 5, repo_dir=str(repo), base_dir=str(tmp_path / ".es"))
    assert out["status"] == "ok"  # fail-open: empty subject, no spans, event still recorded
    assert out["items"] == 0
