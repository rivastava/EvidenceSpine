from __future__ import annotations

import asyncio
import json
from pathlib import Path

from evidencespine import AsyncAgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings
from grounding_utils import grounded_item


def _async_runtime(tmp_path: Path) -> AsyncAgentMemoryRuntime:
    from evidencespine.runtime import AgentMemoryRuntime

    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    return AsyncAgentMemoryRuntime(runtime=AgentMemoryRuntime(config=settings.to_runtime_config()))


def test_async_ingest_brief_snapshot_roundtrip(tmp_path: Path) -> None:
    rt = _async_runtime(tmp_path)

    async def go() -> None:
        ingest = await rt.ingest_event(
            {
                "thread_id": "demo",
                "event_type": "decision",
                "role": "implementer",
                "source_agent_id": "impl",
                "source_turn_id": "a",
                "payload": {"claim": "async ingest works", "fact_state": "verified"},
                "evidence_items": [grounded_item(tmp_path, name="async/evidence.py")],
            }
        )
        assert ingest["status"] == "ok"
        brief = await rt.build_brief("demo", "async ingest works")
        assert brief["recent_verified_facts"]
        snap = await rt.snapshot()
        assert snap["agent_memory_events_24h"] >= 1
        view = await rt.query_view("my_work", thread_id="demo")
        assert view["view"] == "my_work"

    asyncio.run(go())


def test_async_handoff_roundtrip(tmp_path: Path) -> None:
    rt = _async_runtime(tmp_path)

    async def go() -> None:
        await rt.ingest_event(
            {
                "thread_id": "demo",
                "event_type": "decision",
                "role": "implementer",
                "source_agent_id": "impl",
                "source_turn_id": "a",
                "payload": {"claim": "async handoff source", "fact_state": "verified"},
            }
        )
        emitted = await rt.emit_handoff("auditor", "demo", "async test")
        assert emitted["claims"]
        imported = await rt.import_handoff(emitted)
        assert imported["status"] == "ok"

    asyncio.run(go())


def test_async_prune_and_flush(tmp_path: Path) -> None:
    rt = _async_runtime(tmp_path)

    async def go() -> None:
        dry = await rt.prune(dry_run=True)
        assert dry["dry_run"] is True
        result = await rt.prune(ttl_hours=720.0)
        assert result["status"] == "ok"
        flushed = await rt.flush()
        assert flushed["status"] == "ok"

    asyncio.run(go())


def test_async_runtime_builds_from_settings(tmp_path: Path) -> None:
    rt = AsyncAgentMemoryRuntime(base_dir=str(tmp_path / ".es2"))

    async def go() -> None:
        snap = await rt.snapshot()
        assert snap["enabled"] is True

    asyncio.run(go())


def test_async_import_handoff_accepts_json_path(tmp_path: Path) -> None:
    rt = _async_runtime(tmp_path)
    path = tmp_path / "packet.json"

    async def go() -> None:
        await rt.ingest_event(
            {
                "thread_id": "demo",
                "event_type": "decision",
                "role": "implementer",
                "source_agent_id": "impl",
                "source_turn_id": "a",
                "payload": {"claim": "path import", "fact_state": "verified"},
            }
        )
        emitted = await rt.emit_handoff("auditor", "demo", "path test")
        path.write_text(json.dumps(emitted))
        imported = await rt.import_handoff(str(path))
        assert imported["status"] == "ok"

    asyncio.run(go())
