from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from evidencespine.harness.git import _added_ranges, install_git_hooks, record_commit, record_test_result
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
    _git(repo, "commit", "-q", "-m", "add function b")
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


def test_record_commit_unknown_sha_is_invalid_not_ok(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = record_commit("deadbeef" * 5, repo_dir=str(repo), base_dir=str(tmp_path / ".es"))
    assert out["status"] == "invalid"
    assert out["reason"] == "unknown_commit"


def test_added_ranges_skips_pure_deletion_hunks() -> None:
    assert _added_ranges("@@ -5,3 +5,0 @@") == []
    assert _added_ranges("@@ -1,2 +1,3 @@") == [(1, 3)]
    assert _added_ranges("@@ -1 +7 @@") == [(7, 7)]


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


def test_install_git_hooks_quotes_multitoken_executable(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    out = install_git_hooks(
        repo_dir=str(repo),
        executable=f"{sys.executable} -m evidencespine.cli",
        base_dir=".es",
    )
    assert out["status"] == "ok"
    script = (repo / ".git" / "hooks" / "post-commit").read_text(encoding="utf-8")
    assert shlex.quote(sys.executable) in script
    # Individually quoted argv tokens, not one quoted blob.
    assert f"{shlex.quote(sys.executable)} -m evidencespine.cli harness git git-hook" in script


def test_install_git_hooks_inserts_before_trailing_exit(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    path = repo / ".git" / "hooks" / "post-commit"
    path.write_text("#!/bin/sh\necho user-hook\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    out = install_git_hooks(repo_dir=str(repo), executable="evidencespine", base_dir=".es")
    assert out["status"] == "ok"
    script = path.read_text(encoding="utf-8")
    assert "echo user-hook" in script, "user hook preserved"
    lines = script.splitlines()
    invocation_index = next(i for i, line in enumerate(lines) if "harness git git-hook" in line)
    exit_index = next(i for i, line in enumerate(lines) if line.strip() == "exit 0")
    assert invocation_index < exit_index, "spine invocation must run before an unconditional exit"
    assert (repo / ".git" / "hooks" / "post-commit.pre-evidencespine").exists(), "backup written"


def test_install_git_hooks_is_idempotent(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    install_git_hooks(repo_dir=str(repo), executable="evidencespine", base_dir=".es")
    first = (repo / ".git" / "hooks" / "post-commit").read_text(encoding="utf-8")
    install_git_hooks(repo_dir=str(repo), executable="evidencespine", base_dir=".es")
    second = (repo / ".git" / "hooks" / "post-commit").read_text(encoding="utf-8")
    assert first == second
    assert second.count("harness git git-hook") == 1


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_installed_git_hook_actually_runs_on_commit(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    marker = tmp_path / "marker.txt"
    fake = tmp_path / "fake-exe"
    fake.write_text(f"#!/bin/sh\necho ran >> {marker}\n", encoding="utf-8")
    fake.chmod(0o755)
    out = install_git_hooks(repo_dir=str(repo), executable=str(fake), base_dir=".evidencespine")
    assert out["status"] == "ok"
    (repo / "change.txt").write_text("change\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "trigger hook")
    assert marker.exists(), "post-commit hook must execute a real git commit"
    assert "ran" in marker.read_text(encoding="utf-8")
