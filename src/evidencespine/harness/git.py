"""Thin git and test harness hooks.

Two low-friction evidence sources:
- ``record_commit`` ingests one action event per commit with span-grounded
  evidence items parsed from the diff hunks (added-line ranges per file).
- ``record_test_result`` ingests test outcomes with verification provenance
  (method=test), so a green run is a grounded ``verified`` fact.

``install_git_hooks`` writes ``post-commit``/``post-merge`` hooks so evidence
flows in with zero agent effort.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from evidencespine.grounding import ground_file
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings

_HUNK_RE = r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,(\d+))? @@"  # noqa: S105


def _run_git(repo_dir: str, args: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _added_ranges(diff_text: str) -> List[Tuple[int, int]]:
    """Extract new-file line ranges from unified diff hunks."""
    import re

    ranges: List[Tuple[int, int]] = []
    for line in diff_text.splitlines():
        match = re.match(_HUNK_RE, line)
        if not match:
            continue
        start = int(match.group(2))
        length = int(match.group(3) or "1")
        ranges.append((start, start + max(0, length - 1)))
    return ranges


def record_commit(
    sha: str,
    *,
    repo_dir: str = ".",
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    limit_files: int = 8,
    limit_items: int = 12,
) -> Dict[str, Any]:
    """Ingest one action event for a commit with span-grounded evidence items."""
    short = sha[:12]
    subject = _run_git(repo_dir, ["log", "-1", "--format=%s", sha]).stdout.strip()
    name_only = _run_git(repo_dir, ["show", "--name-only", "--format=", sha]).stdout
    files = [f.strip() for f in name_only.splitlines() if f.strip()][: max(1, int(limit_files))]

    source_root = repo_dir or "."
    items: List[Dict[str, Any]] = []
    for path in files:
        full = path if os.path.isabs(path) else os.path.join(source_root, path)
        if not os.path.exists(full):
            continue
        diff = _run_git(repo_dir, ["show", "--unified=0", "--format=", sha, "--", path]).stdout
        for start, end in _added_ranges(diff):
            item = ground_file(path, start, end, source_root=source_root)
            if item is None:
                continue
            items.append(item)
            if len(items) >= max(1, int(limit_items)):
                break
        if len(items) >= max(1, int(limit_items)):
            break

    settings = EvidenceSpineSettings.from_env(base_dir=str(base_dir or ".evidencespine"), storage_format=storage_format)
    runtime = AgentMemoryRuntime(config=settings.to_runtime_config())
    try:
        result = runtime.ingest_event(
            {
                "thread_id": "default",
                "event_type": "action",
                "role": "operator",
                "source_agent_id": "git_hook",
                "source_turn_id": f"commit:{sha}",
                "payload": {
                    "claim": f"commit {short}: {subject}" if subject else f"commit {short}",
                    "fact_state": "asserted",
                    "scope": "git_activity",
                },
                "evidence_items": items,
                "confidence": 0.8,
                "salience": 0.5,
                "metadata": {"source": "git_hook", "commit": sha, "files": list(files)},
            }
        )
        return {"status": "ok", "commit": sha, "ingest": result, "files": list(files), "items": len(items)}
    finally:
        runtime.store.close()


def record_test_result(
    status: str,
    command: str,
    *,
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
) -> Dict[str, Any]:
    """Ingest a test run with verification provenance (green = verified)."""
    ok = str(status or "").strip().lower() in {"passed", "ok", "green", "success", "0"}
    settings = EvidenceSpineSettings.from_env(base_dir=str(base_dir or ".evidencespine"), storage_format=storage_format)
    runtime = AgentMemoryRuntime(config=settings.to_runtime_config())
    try:
        if ok:
            result = runtime.ingest_event(
                {
                    "thread_id": "default",
                    "event_type": "outcome",
                    "role": "operator",
                    "source_agent_id": "test_hook",
                    "source_turn_id": f"test:{command[:120]}",
                    "payload": {
                        "claim": f"tests passed: {command[:400]}",
                        "fact_state": "verified",
                        "verification": {
                            "method": "test",
                            "reference": command[:512],
                            "verified_by": "test_hook",
                        },
                        "scope": "test_activity",
                    },
                    "confidence": 0.9,
                    "salience": 0.5,
                    "metadata": {"source": "test_hook", "status": "passed"},
                }
            )
        else:
            result = runtime.ingest_event(
                {
                    "thread_id": "default",
                    "event_type": "reflection",
                    "role": "operator",
                    "source_agent_id": "test_hook",
                    "source_turn_id": f"test-fail:{command[:120]}",
                    "payload": {
                        "claim": f"tests failed: {command[:400]}",
                        "fact_state": "asserted",
                        "scope": "test_activity",
                    },
                    "confidence": 0.8,
                    "salience": 0.6,
                    "metadata": {"source": "test_hook", "status": "failed"},
                }
            )
        return {"status": "ok", "passed": ok, "ingest": result}
    finally:
        runtime.store.close()


_HOOK_SCRIPT = """#!/bin/sh
# EvidenceSpine thin hook (installed by `evidencespine harness git install-hook`)
__EXE__ harness git git-hook --sha "$(git rev-parse HEAD)" --base-dir __BASE__ >/dev/null 2>&1 || true
"""


def _sh_quote(value: str) -> str:
    import shlex

    return shlex.quote(str(value))


def install_git_hooks(repo_dir: str = ".", executable: str = "evidencespine", base_dir: str = ".evidencespine") -> Dict[str, Any]:
    """Install post-commit and post-merge hooks in a git repository."""
    hooks_dir = os.path.join(repo_dir, ".git", "hooks")
    if not os.path.isdir(hooks_dir):
        return {"status": "invalid", "reason": "not_a_git_repo"}
    installed = []
    for name in ("post-commit", "post-merge"):
        script = _HOOK_SCRIPT.replace("__EXE__", _sh_quote(executable)).replace("__BASE__", _sh_quote(base_dir))
        path = os.path.join(hooks_dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(script)
        os.chmod(path, 0o755)
        installed.append(name)
    return {"status": "ok", "hooks": installed, "hooks_dir": hooks_dir}
