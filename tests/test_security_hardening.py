"""Security hardening regression tests (auth, confinement, safe defaults).

Covers the network-facing MCP tool confinement fixes:
1. check_drift ignores the caller's source_root and runs in the runtime's
   server-side confinement mode (absolute / escaping source ids are
   unverifiable instead of becoming a blind content oracle on host files).
2. import_handoff(path=...) is resolved under the server grounding root;
   absolute paths, traversal and symlink escapes are rejected.
3. prune previews by default (dry_run=true) and only deletes when the caller
   explicitly passes dry_run=false.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from evidencespine.grounding import excerpt_checksum, ground_file
from evidencespine.mcp_server.server import create_server
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def _call(server: Any, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    async def go() -> Any:
        result = await server.call_tool(name, args)
        return json.loads(result.content[0].text)

    return asyncio.run(go())


def _runtime(tmp_path: Path, *, name: str = ".es") -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(
        base_dir=str(tmp_path / name),
        storage_format="sqlite",
    )
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def _confined_drift_supported(runtime: Any) -> bool:
    try:
        params = inspect.signature(runtime.check_evidence_stale).parameters
    except (TypeError, ValueError):
        return False
    return "allow_absolute_source_id" in params


# --- check_drift confinement -----------------------------------------------------


class _ConfinedDriftRuntime:
    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def check_evidence_stale(
        self,
        *,
        thread_id: str = "",
        source_root: str = ".",
        dry_run: bool = True,
        allow_absolute_source_id: bool = True,
    ) -> Dict[str, Any]:
        self.calls.append(
            {
                "thread_id": thread_id,
                "source_root": source_root,
                "dry_run": dry_run,
                "allow_absolute_source_id": allow_absolute_source_id,
            }
        )
        return {
            "status": "ok",
            "checked_items": 0,
            "stale_facts": 0,
            "cleared_facts": 0,
            "unverifiable_items": 0,
            "dry_run": dry_run,
            "results": [],
        }

    def prune(self, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "ok", "dry_run": bool(kwargs.get("dry_run"))}


class _LegacyDriftRuntime:
    """Pre-hardening runtime without the server confinement mode."""

    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def check_evidence_stale(
        self,
        *,
        thread_id: str = "",
        source_root: str = ".",
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        self.calls.append({"thread_id": thread_id, "source_root": source_root, "dry_run": dry_run})
        return {"status": "ok", "checked_items": 0, "dry_run": dry_run, "results": []}


def test_check_drift_ignores_caller_source_root(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("EVIDENCESPINE_SOURCE_ROOT", str(root))
    runtime = _ConfinedDriftRuntime()
    server = create_server(runtime=runtime)

    out = _call(
        server,
        "check_drift",
        {"thread_id": "t1", "source_root": str(tmp_path / "attacker"), "dry_run": True},
    )
    assert out["status"] == "ok"
    assert runtime.calls == [
        {
            "thread_id": "t1",
            "source_root": os.path.realpath(str(root)),
            "dry_run": True,
            "allow_absolute_source_id": False,
        }
    ]


def test_check_drift_refuses_unconfined_legacy_runtime(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EVIDENCESPINE_SOURCE_ROOT", str(tmp_path / "root"))
    runtime = _LegacyDriftRuntime()
    server = create_server(runtime=runtime)

    out = _call(server, "check_drift", {"source_root": "/etc", "dry_run": True})
    assert out["status"] == "unsupported"
    assert out["reason"] == "runtime_missing_evidence_confinement"
    # The unconfined checker must never run from a network-facing tool.
    assert runtime.calls == []


def test_check_drift_tool_description_documents_confinement(monkeypatch, tmp_path: Path) -> None:
    runtime = _ConfinedDriftRuntime()
    server = create_server(runtime=runtime)

    async def go() -> Any:
        return await server.list_tools()

    tools = asyncio.run(go())
    drift = next(tool for tool in tools if tool.name == "check_drift")
    assert "deprecated" in drift.description
    assert "unverifiable" in drift.description


def test_check_drift_real_runtime_confines_absolute_source_id(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("EVIDENCESPINE_SOURCE_ROOT", str(root))
    runtime = _runtime(tmp_path)
    if not _confined_drift_supported(runtime):
        pytest.skip("core check_evidence_stale confinement contract has not landed yet")

    secret = tmp_path / "outside_secret.txt"
    secret.write_text("TOP_SECRET\n", encoding="utf-8")
    ingest = runtime.ingest_event(
        {
            "thread_id": "drift",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "a",
            "payload": {"claim": "leaky fact", "fact_state": "verified"},
            "evidence_items": [
                {
                    "source_id": str(secret),
                    "line_start": 1,
                    "line_end": 1,
                    "excerpt": "TOP_SECRET",
                    "checksum": excerpt_checksum("TOP_SECRET"),
                }
            ],
        }
    )
    assert ingest["status"] == "ok", ingest
    # A relative source id that escapes the server root must be unverifiable too.
    escaping = runtime.ingest_event(
        {
            "thread_id": "drift",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "b",
            "payload": {"claim": "relative escape fact", "fact_state": "verified"},
            "evidence_items": [
                {
                    "source_id": "../outside_secret.txt",
                    "line_start": 1,
                    "line_end": 1,
                    "excerpt": "TOP_SECRET",
                    "checksum": excerpt_checksum("TOP_SECRET"),
                }
            ],
        }
    )
    assert escaping["status"] == "ok", escaping
    server = create_server(runtime=runtime)

    out = _call(
        server,
        "check_drift",
        {"thread_id": "drift", "source_root": str(secret.parent), "dry_run": True},
    )
    assert out["status"] == "ok"
    assert out.get("unverifiable_items", 0) >= 2
    reasons = [str(row.get("reason")) for row in out.get("results", [])]
    assert reasons.count("unverifiable") >= 2
    # The oracle must not reveal whether the guess matched the host file.
    assert "changed" not in reasons
    assert "missing" not in reasons
    runtime.store.close()


def test_check_drift_real_runtime_still_detects_in_root_drift(monkeypatch, tmp_path: Path) -> None:
    """Confinement must not blind the checker to genuine in-root drift."""
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("EVIDENCESPINE_SOURCE_ROOT", str(root))
    runtime = _runtime(tmp_path)
    if not _confined_drift_supported(runtime):
        pytest.skip("core check_evidence_stale confinement contract has not landed yet")
    target = root / "inside.py"
    target.write_text("original\n", encoding="utf-8")
    item = ground_file("inside.py", 1, 1, source_root=str(root))
    assert item is not None
    ingest = runtime.ingest_event(
        {
            "thread_id": "drift-inside",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "a",
            "payload": {"claim": "in-root fact", "fact_state": "verified"},
            "evidence_items": [item],
        }
    )
    assert ingest["status"] == "ok", ingest
    target.write_text("modified\n", encoding="utf-8")
    server = create_server(runtime=runtime)

    out = _call(server, "check_drift", {"thread_id": "drift-inside", "dry_run": True})
    assert out["status"] == "ok"
    reasons = [str(row.get("reason")) for row in out.get("results", [])]
    assert "changed" in reasons
    assert out.get("unverifiable_items", 0) == 0
    runtime.store.close()


# --- import_handoff path confinement ---------------------------------------------


class _FakeImportRuntime:
    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def import_handoff(
        self,
        payload: Any,
        *,
        source_agent_id: str = "external_agent",
        thread_id: str = "",
    ) -> Dict[str, Any]:
        self.calls.append({"payload": payload, "source_agent_id": source_agent_id, "thread_id": thread_id})
        return {"status": "ok"}


def test_import_handoff_rejects_absolute_and_escaping_paths(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "packet.json").write_text("{}", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "packet.json").write_text("{}", encoding="utf-8")
    (root / "link.json").symlink_to(outside / "packet.json")
    monkeypatch.setenv("EVIDENCESPINE_SOURCE_ROOT", str(root))
    runtime = _FakeImportRuntime()
    server = create_server(runtime=runtime)

    rejected = [
        str(outside / "packet.json"),  # absolute path
        "../outside/packet.json",  # traversal escape
        "link.json",  # symlink escape
        "..",  # the parent directory itself
        "",
    ]
    for path in rejected:
        out = _call(server, "import_handoff", {"path": path})
        assert out["status"] == "invalid", (path, out)
        assert out["reason"] in {"path_outside_source_root", "missing_path"}, (path, out)
    assert runtime.calls == []

    out = _call(server, "import_handoff", {"path": "packet.json"})
    assert out["status"] == "ok"
    assert runtime.calls[0]["payload"] == os.path.realpath(str(root / "packet.json"))


def test_import_handoff_real_runtime_accepts_in_root_packet(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("EVIDENCESPINE_SOURCE_ROOT", str(root))
    runtime = _runtime(tmp_path)
    source = _runtime(tmp_path, name=".source")
    source.ingest_event(
        {
            "thread_id": "handoff-demo",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "a",
            "payload": {"claim": "confinement probe", "fact_state": "verified"},
        }
    )
    packet = source.emit_handoff(role="auditor", thread_id="handoff-demo", scope="path confinement").to_dict()
    (root / "packet.json").write_text(json.dumps(packet), encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "packet.json").write_text(json.dumps(packet), encoding="utf-8")
    server = create_server(runtime=runtime)

    ok = _call(server, "import_handoff", {"path": "packet.json"})
    assert ok["status"] == "ok", ok
    escaped = _call(server, "import_handoff", {"path": "../outside/packet.json"})
    assert escaped == {"status": "invalid", "reason": "path_outside_source_root"}
    absolute = _call(server, "import_handoff", {"path": str(outside / "packet.json")})
    assert absolute == {"status": "invalid", "reason": "path_outside_source_root"}
    source.store.close()
    runtime.store.close()


# --- prune safer default ---------------------------------------------------------


class _FakePruneRuntime:
    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def prune(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {"status": "ok", "dry_run": bool(kwargs.get("dry_run"))}


def test_prune_defaults_to_preview(tmp_path: Path) -> None:
    runtime = _FakePruneRuntime()
    server = create_server(runtime=runtime)

    out = _call(server, "prune", {})
    assert out["dry_run"] is True
    assert runtime.calls[-1]["dry_run"] is True

    explicit = _call(server, "prune", {"dry_run": False})
    assert explicit["dry_run"] is False
    assert runtime.calls[-1]["dry_run"] is False


def test_prune_explicit_false_still_deletes(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    server = create_server(runtime=runtime)
    ingest = _call(
        server,
        "ingest_event",
        {
            "event": {
                "thread_id": "prune-demo",
                "event_type": "decision",
                "role": "implementer",
                "source_agent_id": "impl",
                "source_turn_id": "a",
                "payload": {"claim": "old row", "fact_state": "asserted"},
            }
        },
    )
    assert ingest["status"] == "ok", ingest

    preview = _call(server, "prune", {"ttl_hours": 0.1})
    assert preview["dry_run"] is True
    assert preview["events_removed"] == 0
    assert runtime.store._backend.count_events() >= 1

    # The sqlite backend keeps a numeric ts column for TTL pruning; age it far
    # into the past so the explicit prune is forced to delete.
    backend = runtime.store._backend
    with backend._lock:
        conn = backend._connect()
        conn.execute("UPDATE events SET ts = ?", (0.0,))
        conn.execute("UPDATE facts SET ts = ?", (0.0,))
        conn.commit()
        backend._conn = None

    deleted = _call(server, "prune", {"ttl_hours": 0.1, "dry_run": False})
    assert deleted["status"] == "ok"
    assert deleted["events_removed"] >= 1
    assert runtime.store._backend.count_events() == 0
    runtime.store.close()


def test_handoff_receive_prompt_confines_packet_path(monkeypatch, tmp_path: Path) -> None:
    """The handoff_receive prompt imports packet_path server-side, so it must
    apply the same path confinement as the import_handoff tool."""
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("EVIDENCESPINE_SOURCE_ROOT", str(root))
    runtime = _runtime(tmp_path)
    source = _runtime(tmp_path, name=".source")
    source.ingest_event(
        {
            "thread_id": "prompt-demo",
            "event_type": "decision",
            "role": "implementer",
            "source_agent_id": "impl",
            "source_turn_id": "a",
            "payload": {"claim": "prompt confinement probe", "fact_state": "verified"},
        }
    )
    packet = source.emit_handoff(role="auditor", thread_id="prompt-demo", scope="prompt confinement").to_dict()
    (root / "packet.json").write_text(json.dumps(packet), encoding="utf-8")
    outside = tmp_path / "outside_packet.json"
    outside.write_text(json.dumps(packet), encoding="utf-8")
    server = create_server(runtime=runtime)

    async def prompt_text(args: Dict[str, Any]) -> str:
        result = await server.get_prompt("handoff_receive", args, None)
        return result.messages[0].content.text

    absolute = asyncio.run(prompt_text({"packet_path": str(outside)}))
    assert "path_outside_source_root" in absolute
    traversal = asyncio.run(prompt_text({"packet_path": "../outside_packet.json"}))
    assert "path_outside_source_root" in traversal
    in_root = asyncio.run(prompt_text({"packet_path": "packet.json"}))
    assert '"status": "ok"' in in_root
    source.store.close()
    runtime.store.close()


# --- control view validation -----------------------------------------------------


def test_query_view_rejects_unknown_view(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    server = create_server(runtime=runtime)

    out = _call(server, "query_view", {"view": "bogus"})
    assert out["status"] == "invalid"
    assert out["reason"] == "unknown_view"
    assert "active_scopes" in out["valid_views"]
    assert "rows" not in out
    runtime.store.close()


def test_query_view_accepts_hyphen_alias_and_filters(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    server = create_server(runtime=runtime)
    common = {"thread_id": "views", "event_type": "decision", "role": "implementer", "source_agent_id": "impl"}
    runtime.ingest_event(
        {
            **common,
            "source_turn_id": "g1",
            "payload": {"claim": "gate claim", "fact_state": "asserted"},
            "state_context": {
                "scope_id": "gate-scope",
                "state_kind": "pending_gate",
                "status": "ready",
                "fresh_until": "2099-01-01T00:00:00Z",
            },
        }
    )
    runtime.ingest_event(
        {
            **common,
            "source_turn_id": "w1",
            "payload": {"claim": "work claim", "fact_state": "asserted"},
            "state_context": {
                "scope_id": "work-scope",
                "state_kind": "agent_local_work",
                "status": "active",
                "owner_agent_id": "impl",
            },
        }
    )

    hyphenated = _call(server, "query_view", {"view": "open-gates", "thread_id": "views"})
    assert hyphenated["view"] == "open_gates"
    assert [row["scope_id"] for row in hyphenated["rows"]] == ["gate-scope"]
    raw = _call(server, "query_view", {"view": "active_scopes", "thread_id": "views"})
    assert {"gate-scope", "work-scope"} <= {row["scope_id"] for row in raw["rows"]}
    runtime.store.close()


# --- handoff_receive prompt thread binding ---------------------------------------


def test_handoff_receive_prompt_binds_import_to_requested_thread(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("EVIDENCESPINE_SOURCE_ROOT", str(root))
    runtime = _runtime(tmp_path)
    packet = {
        "packet_id": "pkt-thread-binding",
        "role": "auditor",
        "thread_id": "source-thread",
        "scope": "prompt import thread binding",
        "claims": [{"claim": "imported claim for importer thread", "status": "asserted"}],
        "locked_decisions": [],
        "unresolved_contradictions": [],
        "required_validations": [],
        "evidence_refs": [],
        "evidence_items": [],
    }
    (root / "packet.json").write_text(json.dumps(packet), encoding="utf-8")
    server = create_server(runtime=runtime)

    async def prompt_text(args: Dict[str, Any]) -> str:
        result = await server.get_prompt("handoff_receive", args, None)
        return result.messages[0].content.text

    text = asyncio.run(prompt_text({"thread_id": "importer-thread", "packet_path": "packet.json"}))
    assert '"status": "ok"' in text
    runtime_facts = list(runtime.store.iter_facts())
    importer_claims = [
        fact.get("claim")
        for fact in runtime_facts
        if fact.get("thread_id") == "importer-thread"
    ]
    assert any("imported claim for importer thread" in str(claim) for claim in importer_claims)
    # The import must bind to the requested thread, not the packet's thread.
    source_claims = [
        fact.get("claim")
        for fact in runtime_facts
        if fact.get("thread_id") == "source-thread"
    ]
    assert not any("imported claim for importer thread" in str(claim) for claim in source_claims)
    runtime.store.close()


def test_prune_tool_description_documents_safe_default(tmp_path: Path) -> None:
    runtime = _FakePruneRuntime()
    server = create_server(runtime=runtime)

    async def go() -> Any:
        return await server.list_tools()

    tools = asyncio.run(go())
    prune = next(tool for tool in tools if tool.name == "prune")
    assert "dry_run defaults to true" in prune.description
