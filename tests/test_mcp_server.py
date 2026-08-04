from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")

from evidencespine.mcp_server import create_server  # noqa: E402

pytestmark = pytest.mark.skipif(
    not hasattr(mcp, "server"),
    reason="mcp SDK unavailable",
)


def _run(coro):
    return asyncio.run(coro)


def _server(tmp_path: Path):
    return create_server(base_dir=str(tmp_path / ".es"))


def _valid_event(thread_id: str = "demo", scope_id: str = "release-gate") -> dict:
    return {
        "thread_id": thread_id,
        "event_type": "decision",
        "role": "implementer",
        "source_agent_id": "impl",
        "source_turn_id": "t1",
        "payload": {"claim": "deploy patch", "fact_state": "verified"},
        "evidence_items": [{"source_id": "src/file.py", "line_start": 1, "line_end": 3}],
        "state_context": {
            "scope_id": scope_id,
            "state_kind": "runtime_validated_state",
            "status": "active",
            "state_basis": "runtime_validated",
            "validated_at": "2026-08-03T19:58:18Z",
            "validated_by": "test_agent",
            "fresh_until": "2099-01-01T00:00:00Z",
        },
    }


def test_server_lists_tools_resources_prompts(tmp_path: Path) -> None:
    server = _server(tmp_path)

    async def go() -> None:
        tools = await server.list_tools()
        names = {t.name for t in tools}
        assert {
            "ingest_event",
            "build_brief",
            "query_view",
            "emit_handoff",
            "import_handoff",
            "reconcile",
            "snapshot",
        } <= names

        resources = await server.list_resources()
        assert any(r.uri == "evidencespine://snapshot" for r in resources)

        templates = await server.list_resource_templates()
        template_uris = {t.uri_template for t in templates}
        assert "evidencespine://brief/{thread_id}" in template_uris
        assert "evidencespine://view/{view}" in template_uris
        assert "evidencespine://state/{scope_id}" in template_uris

        prompts = await server.list_prompts()
        assert {"session_start", "handoff_receive", "handoff_send"} <= {p.name for p in prompts}

    _run(go())


def test_tool_ingest_brief_view_handoff_snapshot_flow(tmp_path: Path) -> None:
    server = _server(tmp_path)

    async def go() -> None:
        result = await server.call_tool("ingest_event", {"event": _valid_event()})
        assert '"status": "ok"' in result.content[0].text

        result = await server.call_tool("build_brief", {"thread_id": "demo", "query": "deploy state"})
        assert '"recent_verified_facts"' in result.content[0].text

        result = await server.call_tool("query_view", {"view": "active-scopes"})
        assert '"scope_id": "release-gate"' in result.content[0].text

        result = await server.call_tool("emit_handoff", {"thread_id": "demo", "role": "auditor"})
        assert '"packet_id"' in result.content[0].text

        result = await server.call_tool("snapshot", {})
        assert '"agent_memory_events_24h"' in result.content[0].text

    _run(go())


def test_tool_import_handoff_roundtrip(tmp_path: Path) -> None:
    server = _server(tmp_path)

    async def go() -> None:
        await server.call_tool("ingest_event", {"event": _valid_event()})
        emitted = await server.call_tool("emit_handoff", {"thread_id": "demo", "role": "auditor"})
        packet = emitted.content[0].text

        result = await server.call_tool(
            "import_handoff",
            {"packet": packet, "source_agent_id": "another_agent"},
        )
        assert '"status": "ok"' in result.content[0].text

    _run(go())


def test_resource_reads_return_markdown(tmp_path: Path) -> None:
    server = _server(tmp_path)

    async def go() -> None:
        await server.call_tool("ingest_event", {"event": _valid_event()})

        out = await server.read_resource("evidencespine://brief/demo")
        contents = [item for item in out]
        assert contents
        assert contents[0].mime_type == "text/markdown"
        assert contents[0].content.startswith("# Agent Context Brief: demo")

        out = await server.read_resource("evidencespine://snapshot")
        contents = [item for item in out]
        assert contents[0].content.startswith("# EvidenceSpine memory snapshot")

        out = await server.read_resource("evidencespine://state/release-gate")
        contents = [item for item in out]
        assert "release-gate" in contents[0].content

    _run(go())


def test_prompt_session_start_renders_brief(tmp_path: Path) -> None:
    server = _server(tmp_path)

    async def go() -> None:
        await server.call_tool("ingest_event", {"event": _valid_event()})
        result = await server.get_prompt("session_start", {"thread_id": "demo"}, None)
        assert len(result.messages) >= 1
        text = result.messages[0].content.text
        assert "EvidenceSpine" in text
        assert "deploy patch" in text

    _run(go())


def test_prompt_handoff_send_renders_with_snapshot(tmp_path: Path) -> None:
    server = _server(tmp_path)

    async def go() -> None:
        handoffs_dir = tmp_path / ".es" / "handoffs"
        before = set(handoffs_dir.glob("*.json")) if handoffs_dir.exists() else set()
        result = await server.get_prompt("handoff_send", {"thread_id": "demo"}, None)
        assert len(result.messages) >= 1
        text = result.messages[0].content.text
        assert "handoff" in text
        assert "Snapshot" in text
        assert "emit_handoff tool" in text
        after = set(handoffs_dir.glob("*.json")) if handoffs_dir.exists() else set()
        assert after == before, "prompt render must not emit a packet"

    _run(go())


def test_server_builds_runtime_lazily_and_isolates_dirs(tmp_path: Path) -> None:
    first = create_server(base_dir=str(tmp_path / "a" / ".es"))
    second = create_server(base_dir=str(tmp_path / "b" / ".es"))

    async def go() -> None:
        await first.call_tool("ingest_event", {"event": _valid_event()})
        snap_a = await first.call_tool("snapshot", {})
        snap_b = await second.call_tool("snapshot", {})
        assert '"agent_memory_events_24h": 1' in snap_a.content[0].text
        assert '"agent_memory_events_24h": 0' in snap_b.content[0].text

    _run(go())


def test_concurrent_prompt_renders_build_runtime_once(tmp_path: Path) -> None:
    server = _server(tmp_path)

    async def go() -> None:
        results = await asyncio.gather(
            server.get_prompt("handoff_send", {"thread_id": "demo"}, None),
            server.get_prompt("handoff_receive", {"thread_id": "demo"}, None),
        )
        for result in results:
            assert len(result.messages) >= 1
        assert "handoff" in results[0].messages[0].content.text
        assert "Import result" in results[1].messages[0].content.text

    _run(go())


def test_prompts_render_with_no_arguments_use_default_thread(tmp_path: Path) -> None:
    server = _server(tmp_path)

    async def go() -> None:
        for name in ("session_start", "handoff_receive", "handoff_send"):
            result = await server.get_prompt(name, {}, None)
            assert len(result.messages) >= 1

    _run(go())


def test_prompts_treat_placeholder_args_as_unset(tmp_path: Path) -> None:
    server = _server(tmp_path)

    async def go() -> None:
        for name, args in (
            ("session_start", {"thread_id": "$1", "objective": "$2"}),
            ("handoff_receive", {"thread_id": "$1", "source_agent_id": "$2", "packet_path": "$3"}),
            ("handoff_send", {"thread_id": "$1", "role": "$2", "scope": "$3"}),
        ):
            result = await server.get_prompt(name, args, None)
            assert len(result.messages) >= 1

        briefs_dir = tmp_path / ".es" / "briefs"
        names = [p.name for p in briefs_dir.glob("*.json")] if briefs_dir.exists() else []
        assert names, "brief files should be written"
        assert not any(n.startswith("$") for n in names), "placeholder args must not leak into brief names"

    _run(go())
