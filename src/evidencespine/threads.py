"""Stable thread identity for project-scoped memory.

Harness hooks run per agent session, but memory should accumulate across
sessions for the same workspace. Deriving a thread id from the session id
means every session starts from an empty brief, which defeats the side-car.
This module resolves a *workspace-scoped* thread id instead:

1. an explicit id always wins;
2. `EVIDENCESPINE_THREAD_ID` overrides discovery;
3. otherwise a stable `proj_<sha1(realpath(project root))[:12]>` id.

The project root is the enclosing git worktree when available, so subdirectory
tools and hooks agree on one thread. Session/turn identity still travels in
`source_turn_id` / metadata.
"""

from __future__ import annotations

import hashlib
import os
import subprocess

PROJECT_THREAD_PREFIX = "proj_"
THREAD_ID_ENV = "EVIDENCESPINE_THREAD_ID"


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def project_root(cwd: str = "") -> str:
    """Return the realpath of the enclosing git worktree, else of *cwd*."""
    start = os.path.realpath(str(cwd or os.getcwd()))
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            capture_output=True,
            text=True,
            timeout=5,
        )
        root = str(proc.stdout or "").strip()
        if proc.returncode == 0 and root:
            return os.path.realpath(root)
    except (OSError, subprocess.SubprocessError):
        pass
    return start


def thread_id_for_workspace(path: str) -> str:
    """Stable thread id for a workspace path (idempotent across sessions)."""
    clean = os.path.realpath(str(path or os.getcwd()))
    return f"{PROJECT_THREAD_PREFIX}{_digest(clean)}"


def resolve_thread_id(explicit: str = "", *, cwd: str = "", env_var: str = THREAD_ID_ENV) -> str:
    """Resolve the thread id for a surface call.

    Explicit values are returned unchanged (after stripping); the env override
    is honored next; otherwise the workspace-scoped id is derived.
    """
    clean = str(explicit or "").strip()
    if clean:
        return clean
    override = str(os.getenv(env_var, "") or "").strip()
    if override:
        return override
    return thread_id_for_workspace(project_root(cwd))


def thread_identity(explicit: str = "", *, cwd: str = "") -> dict[str, str]:
    """Return `{thread_id, source, project_root}` for diagnostics/metadata."""
    clean = str(explicit or "").strip()
    if clean:
        return {"thread_id": clean, "source": "explicit", "project_root": project_root(cwd)}
    override = str(os.getenv(THREAD_ID_ENV, "") or "").strip()
    if override:
        return {"thread_id": override, "source": "env", "project_root": project_root(cwd)}
    root = project_root(cwd)
    return {"thread_id": thread_id_for_workspace(root), "source": "workspace", "project_root": root}
