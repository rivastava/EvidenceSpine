from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from evidencespine.a2a import build_fastapi_app
from evidencespine.settings import EvidenceSpineSettings


pytest.importorskip("a2a")


def _app(tmp_path: Path, *, storage_format: str = "sqlite"):
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"), storage_format=storage_format)
    from evidencespine.runtime import AgentMemoryRuntime

    rt = AgentMemoryRuntime(config=settings.to_runtime_config())
    return build_fastapi_app(runtime=rt)


async def _send_message(client: httpx.AsyncClient, text: str, *, message_id: str = "msg-1") -> dict:
    resp = await client.post(
        "/",
        headers={"A2A-Version": "1.0"},
        json={
            "jsonrpc": "2.0",
            "id": "1",
            "method": "SendMessage",
            "params": {
                "message": {
                    "message_id": message_id,
                    "role": "ROLE_USER",
                    "parts": [{"text": text}],
                },
            },
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("jsonrpc") == "2.0"
    assert "error" not in body, body
    return body["result"]["task"]


def _result_text(result: dict) -> str:
    status = result.get("status") or {}
    message = status.get("message") or {}
    texts = []
    for part in message.get("parts", []):
        texts.append(part.get("text", ""))
    return "\n".join(texts)


def _run(coro) -> None:
    asyncio.run(coro)


def test_agent_card_served_at_well_known_path(tmp_path: Path) -> None:
    app = _app(tmp_path)

    async def go() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/.well-known/agent-card.json")
        assert resp.status_code == 200
        card = resp.json()
        assert card["name"] == "evidencespine-agent"
        assert card["provider"]["organization"] == "evidencespine"
        skill_ids = {skill["id"] for skill in card["skills"]}
        assert {"memory.read", "memory.write", "memory.handoff", "memory.health"} <= skill_ids

    _run(go())


def test_send_message_builds_brief_for_plain_text(tmp_path: Path) -> None:
    app = _app(tmp_path)

    async def go() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            result = await _send_message(client, "latest context")
        text = _result_text(result)
        payload = json.loads(text)
        assert "thread_id" in payload or "current_goal" in payload
        assert result["status"]["state"] == "TASK_STATE_COMPLETED"

    _run(go())


def test_send_message_ingest_event_roundtrip(tmp_path: Path) -> None:
    app = _app(tmp_path)

    async def go() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            ingest = await _send_message(
                client,
                json.dumps(
                    {
                        "action": "ingest_event",
                        "thread_id": "demo",
                        "params": {
                            "event_type": "decision",
                            "role": "implementer",
                            "source_agent_id": "impl",
                            "source_turn_id": "a",
                            "payload": {"claim": "A2A ingest works", "fact_state": "verified"},
                        },
                    }
                ),
                message_id="msg-t2",
            )
            brief = await _send_message(
                client,
                json.dumps({"action": "build_brief", "thread_id": "demo", "params": {"query": "ingest works"}}),
                message_id="msg-t3",
            )
            snap = await _send_message(
                client,
                json.dumps({"action": "snapshot"}),
                message_id="msg-t4",
            )
        ingest_payload = json.loads(_result_text(ingest))
        assert ingest_payload["status"] == "ok"
        brief_payload = json.loads(_result_text(brief))
        assert brief_payload["recent_verified_facts"] or brief_payload["locked_decisions"]
        snap_payload = json.loads(_result_text(snap))
        assert snap_payload["agent_memory_events_24h"] >= 1

    _run(go())


def test_send_message_handoff_roundtrip(tmp_path: Path) -> None:
    app = _app(tmp_path)

    async def go() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await _send_message(
                client,
                json.dumps(
                    {
                        "action": "ingest_event",
                        "thread_id": "demo",
                        "params": {
                            "event_type": "decision",
                            "role": "implementer",
                            "source_agent_id": "impl",
                            "source_turn_id": "a",
                            "payload": {"claim": "handoff source", "fact_state": "verified"},
                        },
                    }
                ),
                message_id="msg-t5",
            )
            emitted = await _send_message(
                client,
                json.dumps(
                    {"action": "emit_handoff", "thread_id": "demo", "params": {"role": "auditor", "scope": "a2a test"}}
                ),
                message_id="msg-t6",
            )
            imported = await _send_message(
                client,
                json.dumps(
                    {
                        "action": "import_handoff",
                        "params": {"packet": json.loads(_result_text(emitted))},
                    }
                ),
                message_id="msg-t7",
            )
        emit_payload = json.loads(_result_text(emitted))
        assert emit_payload["packet_id"]
        assert emit_payload["claims"]
        import_payload = json.loads(_result_text(imported))
        assert import_payload["status"] == "ok"

    _run(go())


def test_unknown_action_fails_open(tmp_path: Path) -> None:
    app = _app(tmp_path)

    async def go() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            result = await _send_message(
                client,
                json.dumps({"action": "nonsense", "params": {}}),
                message_id="msg-t8",
            )
        payload = json.loads(_result_text(result))
        assert payload["status"] == "error"
        assert "unknown action" in payload["reason"]

    _run(go())


def test_get_task_endpoint(tmp_path: Path) -> None:
    app = _app(tmp_path)

    async def go() -> None:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            result = await _send_message(client, "hello", message_id="msg-t9")
            task_id = result["id"]
            get = await client.post(
                "/",
                headers={"A2A-Version": "1.0"},
                json={
                    "jsonrpc": "2.0",
                    "id": "2",
                    "method": "GetTask",
                    "params": {"id": task_id},
                },
            )
            assert get.status_code == 200
            get_body = get.json()
            assert "error" not in get_body, get_body
            assert get_body["result"]["id"] == task_id

    _run(go())
