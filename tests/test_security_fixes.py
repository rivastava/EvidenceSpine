"""Security hardening regression tests.

Covers the external audit findings:
1. MCP grounding is confined to a server source root and rejects absolute paths.
2. Brief/handoff artifact filenames are opaque UUIDs (no caller-controlled
   traversal); thread/role live in the JSON payload; same-second writes are
   exclusive (no overwrite).
3. SQLite dedupe honors dedupe_window_sec (expired hashes are accepted again
   and old hash rows are pruned).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from evidencespine.backends.sqlite import SqliteStoreBackend
from evidencespine.grounding import ground_ref
from evidencespine.mcp_server.server import _static_bearer_middleware
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def _runtime(tmp_path: Path, **env_extra: str) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    for key, value in env_extra.items():
        setattr(settings, key, value)
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def _event(thread_id: str = "t1", *, claim: str = "same claim", extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "thread_id": thread_id,
        "event_type": "decision",
        "source_agent_id": "agent",
        "source_turn_id": "turn",
        "payload": {"claim": claim},
    }
    row.update(extra or {})
    return row


# --- finding 1: grounding confinement ---


def test_ground_ref_rejects_absolute_paths_when_disallowed(tmp_path: Path) -> None:
    target = tmp_path / "secret.txt"
    target.write_text("needle", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "ok.txt"
    inside.write_text("visible", encoding="utf-8")

    # allow_absolute=False rejects every absolute path (even inside the root)
    assert ground_ref(str(target) + "#L1", source_root=str(root), allow_absolute=False) is None
    assert ground_ref(str(inside) + "#L1", source_root=str(root), allow_absolute=False) is None
    # relative refs must resolve inside the root
    assert ground_ref("ok.txt#L1", source_root=str(root), allow_absolute=False) is not None
    assert ground_ref("../root/ok.txt#L1", source_root=str(root), allow_absolute=False) is not None
    assert ground_ref("../secret.txt#L1", source_root=str(root), allow_absolute=False) is None


def test_ground_keeps_allowing_absolute_paths_locally(tmp_path: Path) -> None:
    target = tmp_path / "local.txt"
    target.write_text("needle", encoding="utf-8")
    assert ground_ref(str(target) + "#L1", source_root=".", allow_absolute=True) is not None


def test_mcp_ground_tool_confined_to_source_root(tmp_path: Path) -> None:
    import anyio

    from evidencespine.mcp_server.server import create_server

    outside = tmp_path / "outside_secret.txt"
    outside.write_text("needle", encoding="utf-8")
    server = create_server(base_dir=str(tmp_path / ".es"))

    async def go() -> tuple[Any, Any]:
        r_abs = await server.call_tool("ground", {"ref": str(outside) + "#L1"})
        r_rel = await server.call_tool("ground", {"ref": "ok.txt#L1"})
        return r_abs, r_rel

    r_abs, r_rel = anyio.run(go)
    # Absolute paths are always rejected by the server, and refs that do not
    # resolve inside the server source root are ungroundable.
    assert '"ungroundable"' in r_abs.content[0].text
    assert '"ungroundable"' in r_rel.content[0].text


def test_mcp_ingest_ground_refs_confined(tmp_path: Path) -> None:
    import anyio

    from evidencespine.mcp_server.server import create_server

    server = create_server(base_dir=str(tmp_path / ".es"))
    secret = tmp_path / "secret.txt"
    secret.write_text("needle", encoding="utf-8")

    async def go() -> Any:
        return await server.call_tool(
            "ingest_event",
            {
                "event": {
                    "thread_id": "t1",
                    "event_type": "decision",
                    "source_agent_id": "a",
                    "source_turn_id": "b",
                    "payload": {"claim": "x"},
                    "ground_refs": [str(secret) + "#L1"],
                }
            },
        )

    result = anyio.run(go)
    assert '"status": "ok"' in result.content[0].text or '"status":"ok"' in result.content[0].text
    assert "needle" not in result.content[0].text, "absolute path must not be grounded server-side"


# --- finding 2: artifact filenames ---


def test_artifact_filename_is_uuid_and_payload_keeps_identifiers(tmp_path: Path) -> None:
    import os

    rt = _runtime(tmp_path)
    rt.store.write_brief("../../etc/passwd", {"summary": "s"})
    rt.store.write_handoff("../../tmp/evil", "x/y", {"packet": "p"})

    briefs = list((tmp_path / ".es" / "briefs").glob("*.json"))
    handoffs = list((tmp_path / ".es" / "handoffs").glob("*.json"))
    assert len(briefs) == 1 and len(handoffs) == 1
    for file_set in (briefs, handoffs):
        name = file_set[0].name
        assert Path(name).suffix == ".json"
        assert Path(name).stem.isalnum() and len(Path(name).stem) == 32  # hex uuid

    body = json.loads(briefs[0].read_text(encoding="utf-8"))
    assert body["thread_id"] == "../../etc/passwd"
    assert body["role"] == "brief"
    assert body.get("written_at")
    briefs_real = os.path.realpath(str(tmp_path / ".es" / "briefs"))
    handoffs_real = os.path.realpath(str(tmp_path / ".es" / "handoffs"))
    assert all(os.path.commonpath([os.path.realpath(str(p)), briefs_real]) == briefs_real for p in briefs)
    assert all(os.path.commonpath([os.path.realpath(str(p)), handoffs_real]) == handoffs_real for p in handoffs)


def test_no_file_escapes_artifact_directory(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.store.write_brief("../escape", {"summary": "s"})
    rt.store.write_handoff("../../escape", "role", {"packet": "p"})
    outside = tmp_path / "escape.json"
    assert not outside.exists()
    assert not (tmp_path.parent / "escape.json").exists()


def test_same_second_writes_are_distinct(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    p1 = rt.store.write_brief("t", {"summary": "one"})
    p2 = rt.store.write_brief("t", {"summary": "two"})
    assert p1 != p2
    assert Path(p1).exists() and Path(p2).exists()
    one = json.loads(Path(p1).read_text(encoding="utf-8"))
    two = json.loads(Path(p2).read_text(encoding="utf-8"))
    assert one["summary"] == "one" and two["summary"] == "two"


# --- finding 3: dedupe window ---


def test_sqlite_dedupe_honors_window(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    first = rt.ingest_event(_event())
    assert first["status"] == "ok"

    duplicate = rt.ingest_event(_event())
    assert duplicate["status"] == "deduped"

    # expire both the in-memory ring and the sqlite dedup_hashes row beyond the
    # window, then re-ingest: an identical event must be accepted again
    backend = rt.store._backend
    assert isinstance(backend, SqliteStoreBackend)
    ring = rt.store.state.get("event_hash_ring", [])
    if isinstance(ring, list):
        for row in ring:
            if isinstance(row, dict):
                row["ts"] = time.time() - 99999
    with backend._lock:
        conn = backend._connect()
        conn.execute("UPDATE dedup_hashes SET ts = ? WHERE hash IS NOT NULL", (time.time() - 99999,))
        conn.commit()
        backend._conn = None
    reingest = rt.ingest_event(_event())
    assert reingest["status"] == "ok", "identical event after window must be accepted again"


def test_sqlite_prune_removes_stale_dedup_hashes(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    rt.ingest_event(_event())
    backend = rt.store._backend
    assert isinstance(backend, SqliteStoreBackend)
    with backend._lock:
        conn = backend._connect()
        conn.execute("UPDATE dedup_hashes SET ts = ? WHERE hash IS NOT NULL", (time.time() - 99999,))
        conn.commit()
        backend._conn = None
    out = rt.prune(ttl_hours=720.0)
    assert out["dedup_hashes_removed"] >= 1
    with backend._lock:
        conn = backend._connect()
        remaining = int(conn.execute("SELECT COUNT(*) FROM dedup_hashes").fetchone()[0])
        backend._conn = None
    assert remaining == 0


# --- HTTP auth middleware ---


def test_static_bearer_middleware_requires_token() -> None:
    import anyio

    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def home(_request: Any) -> Any:
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/", home)])
    wrapped = _static_bearer_middleware(app, "s3cret")

    async def go(headers: dict[str, str]) -> tuple[int, str]:
        import httpx

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=wrapped), base_url="http://test") as client:
            response = await client.get("/", headers=headers)
            return response.status_code, response.text

    assert anyio.run(go, {})[0] == 401
    assert anyio.run(go, {"Authorization": "Bearer wrong"})[0] == 401
    status, text = anyio.run(go, {"Authorization": "Bearer s3cret"})
    assert status == 200
    assert "ok" in text
