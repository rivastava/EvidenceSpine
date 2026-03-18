from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from evidencespine.adapters import TranscriptAdapter, TranscriptAdapterConfig
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


@dataclass
class _MessageObject:
    role: str
    content: str
    id: str = ""


def _runtime(tmp_path: Path) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def test_transcript_adapter_normalizes_dict_messages(tmp_path: Path) -> None:
    adapter = TranscriptAdapter(_runtime(tmp_path))
    rows = adapter.normalize_messages(
        [
            {"id": "m1", "role": "user", "content": "Need audit"},
            {"id": "m2", "role": "assistant", "content": "Patch applied"},
        ]
    )
    assert [r.role for r in rows] == ["user", "assistant"]
    assert [r.event_type for r in rows] == ["intent", "decision"]


def test_transcript_adapter_normalizes_object_messages(tmp_path: Path) -> None:
    adapter = TranscriptAdapter(_runtime(tmp_path))
    rows = adapter.normalize_messages([
        _MessageObject(role="human", content="Check drift", id="a1"),
        _MessageObject(role="function", content="tests passed", id="a2"),
    ])
    assert [r.role for r in rows] == ["user", "tool"]
    assert [r.event_type for r in rows] == ["intent", "outcome"]


def test_transcript_adapter_skips_empty_messages(tmp_path: Path) -> None:
    adapter = TranscriptAdapter(_runtime(tmp_path))
    rows = adapter.normalize_messages([
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": "Apply patch"},
    ])
    assert len(rows) == 1
    assert rows[0].content == "Apply patch"


def test_transcript_adapter_role_mapping_defaults(tmp_path: Path) -> None:
    adapter = TranscriptAdapter(_runtime(tmp_path))
    rows = adapter.normalize_messages([
        {"role": "system", "content": "Use additive patch"},
        {"role": "agent", "content": "Done"},
        {"role": "tool", "content": "pytest ok"},
        {"role": "observer", "content": "Noted"},
    ])
    assert [r.event_type for r in rows] == ["intent", "decision", "outcome", "reflection"]


def test_transcript_adapter_custom_role_aliases_override_defaults(tmp_path: Path) -> None:
    config = TranscriptAdapter.default_config()
    config.role_aliases["reviewer"] = "assistant"
    adapter = TranscriptAdapter(_runtime(tmp_path), config=config)
    rows = adapter.normalize_messages([{"role": "reviewer", "content": "Looks safe"}])
    assert rows[0].role == "assistant"
    assert rows[0].event_type == "decision"


def test_transcript_adapter_turn_id_generation_is_stable(tmp_path: Path) -> None:
    adapter = TranscriptAdapter(_runtime(tmp_path))
    rows = adapter.normalize_messages([
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ])
    assert [r.turn_id for r in rows] == ["msg_0", "msg_1"]


def test_transcript_adapter_preserves_caller_evidence_items(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    adapter = TranscriptAdapter(runtime, default_thread_id="demo")
    rows = adapter.normalize_messages(
        [
            {
                "id": "m1",
                "role": "assistant",
                "content": "Patch complete",
                "evidence_items": [{"source_id": "src/file.py", "line_start": 22, "line_end": 24}],
            }
        ]
    )
    assert rows[0].evidence_items[0]["source_id"] == "src/file.py"

    adapter.ingest_messages(rows)
    events_path = tmp_path / ".es" / "events.jsonl"
    latest = json.loads(events_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert latest["evidence_items"][0]["source_id"] == "src/file.py"


def test_transcript_adapter_preserves_state_context_from_raw_messages(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    adapter = TranscriptAdapter(runtime, default_thread_id="demo")
    rows = adapter.normalize_messages(
        [
            {
                "id": "m1",
                "role": "assistant",
                "content": "Patch under verification",
                "state_context": {
                    "scope_id": "auth-timeout-fix",
                    "state_kind": "agent_local_work",
                    "status": "active",
                    "owner_agent_id": "implementer",
                },
            }
        ]
    )
    assert rows[0].state_context is not None
    assert rows[0].state_context["scope_id"] == "auth-timeout-fix"

    adapter.ingest_messages(rows)
    events_path = tmp_path / ".es" / "events.jsonl"
    latest = json.loads(events_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert latest["state_context"]["owner_agent_id"] == "implementer"
