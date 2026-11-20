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
import re
import shutil
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
    ranges: List[Tuple[int, int]] = []
    for line in diff_text.splitlines():
        match = re.match(_HUNK_RE, line)
        if not match:
            continue
        start = int(match.group(2))
        length = int(match.group(3) or "1")
        if length <= 0:
            # Pure-deletion hunk ("@@ -5,3 +5,0 @@"): no added lines to ground.
            continue
        ranges.append((start, start + length - 1))
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
    log_proc = _run_git(repo_dir, ["log", "-1", "--format=%s", sha])
    if log_proc.returncode != 0:
        return {"status": "invalid", "reason": "unknown_commit", "commit": sha}
    subject = log_proc.stdout.strip()
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


#: Sentinel comment identifying an EvidenceSpine-managed hook.
_HOOK_SENTINEL = "evidencespine-git-hook"

#: Shell command a managed hook runs. __EXE__ is an argv-prefix of individually
#: quoted tokens (so "python3 -m evidencespine.cli" works) and __BASE__ is a
#: single quoted absolute path.
_HOOK_INVOCATION = (
    '__EXE__ harness git git-hook --sha "$(git rev-parse HEAD)"'
    ' --repo-dir "$(git rev-parse --show-toplevel)" --base-dir __BASE__ >/dev/null 2>&1 || true'
)

_HOOK_SCRIPT = """#!/bin/sh
# EvidenceSpine thin hook (installed by `evidencespine harness git install-hook`)
# Sentinel: evidencespine-git-hook (do not remove this line)
""" + _HOOK_INVOCATION + """
"""


def _sh_quote(value: str) -> str:
    import shlex

    return shlex.quote(str(value))


def _executable_argv(executable: str) -> List[str]:
    """Split an executable into argv tokens without breaking spaced paths.

    An existing path (or a string whose first word does not resolve to a
    command) is treated as one executable token; only genuine multi-token
    launchers like "python3 -m evidencespine.cli" are split.
    """
    text = str(executable or "").strip()
    if not text:
        return ["evidencespine"]
    if os.path.exists(text):
        return [text]
    import shlex

    try:
        argv = shlex.split(text)
    except Exception:
        return [text]
    if len(argv) <= 1:
        return argv or ["evidencespine"]
    first = argv[0]
    if shutil.which(first) is not None or os.path.exists(first):
        return argv
    return [text]


def _quote_argv_prefix(executable: str) -> str:
    """Quote each argv token of the executable separately.

    Quoting the whole multi-token string (shlex.quote("python3 -m mod")) would
    pass it to the shell as one filename and the hook could never run; joining
    individually quoted tokens keeps both the multi-token launcher and a
    single spaced path executable working.
    """
    import shlex

    return " ".join(shlex.quote(token) for token in _executable_argv(executable))


def _render_hook_invocation(executable: str, base_dir: str) -> str:
    invocation = _HOOK_INVOCATION.replace("__EXE__", _quote_argv_prefix(executable))
    return invocation.replace("__BASE__", _sh_quote(str(base_dir)))


def _render_hook_script(executable: str, base_dir: str) -> str:
    return _HOOK_SCRIPT.replace("__EXE__", _quote_argv_prefix(executable)).replace(
        "__BASE__", _sh_quote(str(base_dir))
    )


def _insert_before_trailing_exit(existing: str, invocation: str) -> str:
    """Insert a chained invocation before a trailing shell 'exit' statement.

    Appending after an unconditional "exit 0" would make the spine hook dead
    code in a chained user hook; insert before the last top-level exit instead.
    """
    text = str(existing or "")
    if text and not text.endswith("\n"):
        text += "\n"
    if not text.startswith("#!"):
        text = "#!/bin/sh\n" + text
    lines = text.splitlines()
    insert_at = None
    for index in range(len(lines) - 1, -1, -1):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"^exit(?:\s|;|$)", stripped):
            insert_at = index
        break
    if insert_at is None:
        return text + invocation + "\n"
    merged = lines[:insert_at] + [invocation] + lines[insert_at:]
    return "\n".join(merged) + "\n"


def _git_dir(repo_dir: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    raw = proc.stdout.strip()
    if not raw:
        return None
    if not os.path.isabs(raw):
        raw = os.path.join(repo_dir, raw)
    hooks_dir = os.path.join(raw, "hooks")
    # git worktrees use a gitdir file; --git-dir resolves correctly.
    return hooks_dir


def install_git_hooks(repo_dir: str = ".", executable: str = "evidencespine", base_dir: str = ".evidencespine") -> Dict[str, Any]:
    """Install post-commit and post-merge hooks in a git repository.

    Non-destructive: existing hooks are preserved (backed up once to
    ``<name>.pre-evidencespine``) and chained — the spine invocation is
    appended when the sentinel is absent, making installs idempotent.
    """
    hooks_dir = _git_dir(repo_dir)
    if not hooks_dir or not os.path.isdir(hooks_dir):
        # Fallback for bare .git layout.
        fallback = os.path.join(repo_dir, ".git", "hooks")
        if not os.path.isdir(fallback):
            return {"status": "invalid", "reason": "not_a_git_repo"}
        hooks_dir = fallback
    abs_base = base_dir if os.path.isabs(base_dir) else os.path.abspath(os.path.join(repo_dir, base_dir))
    invocation = _render_hook_invocation(executable, abs_base)
    fresh_script = _render_hook_script(executable, abs_base)
    installed = []
    for name in ("post-commit", "post-merge"):
        path = os.path.join(hooks_dir, name)
        existing = ""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    existing = handle.read()
            except Exception:
                existing = ""
            if "evidencespine-git-hook" in existing or "evidencespine harness git git-hook" in existing:
                installed.append(name)
                continue
            # Backup once, then chain.
            backup = path + ".pre-evidencespine"
            if not os.path.exists(backup):
                try:
                    with open(backup, "w", encoding="utf-8") as handle:
                        handle.write(existing)
                except Exception:
                    pass
            # Preserve the user hook; insert the spine invocation before a
            # trailing 'exit' so chained hooks actually run it.
            script = _insert_before_trailing_exit(existing, invocation)
        else:
            script = fresh_script
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(script)
        os.chmod(path, 0o755)
        installed.append(name)
    return {"status": "ok", "hooks": installed, "hooks_dir": hooks_dir}
