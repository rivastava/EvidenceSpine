from __future__ import annotations

import os
import subprocess
from pathlib import Path

from evidencespine.threads import (
    PROJECT_THREAD_PREFIX,
    project_root,
    resolve_thread_id,
    thread_id_for_workspace,
    thread_identity,
)


def _git_init(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)


def test_workspace_thread_id_is_stable_and_prefixed(tmp_path: Path) -> None:
    a = thread_id_for_workspace(str(tmp_path))
    b = thread_id_for_workspace(str(tmp_path) + os.sep)
    assert a == b
    assert a.startswith(PROJECT_THREAD_PREFIX)
    assert len(a) == len(PROJECT_THREAD_PREFIX) + 12


def test_project_root_prefers_git_toplevel(tmp_path: Path) -> None:
    _git_init(tmp_path)
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert project_root(str(nested)) == os.path.realpath(str(tmp_path))


def test_project_root_falls_back_to_cwd(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    assert project_root(str(plain)) == os.path.realpath(str(plain))


def test_explicit_and_env_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("EVIDENCESPINE_THREAD_ID", raising=False)
    assert resolve_thread_id("thread-explicit", cwd=str(tmp_path)) == "thread-explicit"
    monkeypatch.setenv("EVIDENCESPINE_THREAD_ID", "thread-env")
    assert resolve_thread_id("", cwd=str(tmp_path)) == "thread-env"
    identity = thread_identity(cwd=str(tmp_path))
    assert identity["thread_id"] == "thread-env"
    assert identity["source"] == "env"


def test_subdirectories_share_one_thread(tmp_path: Path) -> None:
    _git_init(tmp_path)
    first = tmp_path / "svc-a"
    second = tmp_path / "svc-b"
    first.mkdir()
    second.mkdir()
    assert resolve_thread_id(cwd=str(first)) == resolve_thread_id(cwd=str(second))
