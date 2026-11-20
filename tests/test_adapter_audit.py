from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from evidencespine.adapters import AutoGenAdapter, LangGraphAdapter, TranscriptAdapter
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def _runtime(tmp_path: Path, *, enabled: bool = True) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    settings.enabled = enabled
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def _events(rt: AgentMemoryRuntime) -> list[dict]:
    return list(rt.store.iter_events())


@dataclass
class _ObjectMessage:
    role: str
    content: object
    id: str = ""
    name: str = ""


class _StubRuntime:
    """Duck-typed runtime returning scripted ingest statuses."""

    def __init__(self, statuses: list[str]) -> None:
        self.statuses = list(statuses)
        self.calls: list[dict] = []

    def ingest_event(self, payload: dict) -> dict:
        self.calls.append(payload)
        status = self.statuses.pop(0) if self.statuses else "ok"
        out = {"status": status}
        if status == "disabled":
            out["reason"] = "agent_memory_disabled"
        return out


def test_content_parts_list_is_flattened_to_text(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    adapter = TranscriptAdapter(runtime, default_thread_id="shapes")
    out = adapter.ingest_messages(
        [
            {
                "id": "m1",
                "role": "assistant",
                "content": [{"type": "text", "text": "hello world"}],
            }
        ]
    )
    assert out["status"] == "ok"
    assert out["ingested"] == 1
    stored = _events(runtime)[-1]
    assert stored["payload"]["claim"] == "hello world", stored["payload"]["claim"]
    runtime.store.close()


def test_multi_part_content_list_joins_parts(tmp_path: Path) -> None:
    adapter = TranscriptAdapter(_runtime(tmp_path))
    rows = adapter.normalize_messages(
        [{"role": "user", "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]}]
    )
    assert rows[0].content == "first second"


def test_tuple_messages_are_normalized(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    adapter = TranscriptAdapter(runtime, default_thread_id="tuples")
    rows = adapter.normalize_messages(
        [
            ("user", "tuple message"),
            ("assistant", [{"type": "text", "text": "parts message"}]),
        ]
    )
    assert [r.role for r in rows] == ["user", "assistant"]
    assert [r.content for r in rows] == ["tuple message", "parts message"]
    out = adapter.ingest_messages([("user", "tuple message")])
    assert out["ingested"] == 1
    assert _events(runtime)[-1]["payload"]["claim"] == "tuple message"
    runtime.store.close()


def test_tuple_message_metadata_is_preserved(tmp_path: Path) -> None:
    adapter = TranscriptAdapter(_runtime(tmp_path))
    rows = adapter.normalize_messages(
        [
            (
                "assistant",
                "grounded",
                {
                    "id": "tuple-1",
                    "evidence_items": [{"source_id": "src/a.py", "line_start": 3, "line_end": 4}],
                    "state_context": {
                        "scope_id": "scope-1",
                        "state_kind": "agent_local_work",
                        "status": "active",
                    },
                },
            )
        ]
    )
    assert rows[0].turn_id == "tuple-1"
    assert rows[0].evidence_items[0]["source_id"] == "src/a.py"
    assert rows[0].state_context is not None
    assert rows[0].state_context["scope_id"] == "scope-1"


def test_state_object_with_messages_attribute_is_accepted(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    adapter = LangGraphAdapter(runtime, default_thread_id="obj_state")
    state = SimpleNamespace(messages=[{"role": "user", "content": "state object message"}])
    out = adapter.ingest_state(state)
    assert out["status"] == "ok"
    assert out["ingested"] == 1
    runtime.store.close()


def test_non_messages_state_returns_clear_invalid_result(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    adapter = LangGraphAdapter(runtime, default_thread_id="bad_state")
    out = adapter.ingest_state({"foo": 1})
    assert out["status"] == "invalid"
    assert out["normalized"] == 0
    assert "messages" in out["reason"]
    hint = adapter.ingest_state({"messages": "not-a-list"})
    assert hint["status"] == "invalid"
    assert "list" in hint["reason"]
    runtime.store.close()


def test_empty_message_list_is_ok_not_error(tmp_path: Path) -> None:
    adapter = TranscriptAdapter(_runtime(tmp_path))
    out = adapter.ingest_messages([])
    assert out["status"] == "ok"
    assert out["seen"] == 0


def test_replay_dedupe_is_counted_separately_from_ingest(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    adapter = TranscriptAdapter(runtime, default_thread_id="replay")
    messages = [
        {"id": "m1", "role": "user", "content": "Need audit"},
        {"id": "m2", "role": "assistant", "content": "Patch applied"},
    ]
    first = adapter.ingest_messages(messages)
    assert first["status"] == "ok"
    assert first["ingested"] == 2
    assert first["deduped"] == 0
    assert first["new_events"] == 2
    second = adapter.ingest_messages(messages)
    assert second["status"] == "ok", second
    assert second["ingested"] == 0, "replayed events must not be counted as newly ingested"
    assert second["deduped"] == 2
    assert second["new_events"] == 0
    assert len(_events(runtime)) == 2, "replay must not duplicate stored events"
    runtime.store.close()


def test_partial_and_failed_and_disabled_statuses() -> None:
    partial = TranscriptAdapter(_StubRuntime(["ok", "error"]))  # type: ignore[arg-type]
    partial_out = partial.ingest_messages(
        [
            {"id": "a", "role": "user", "content": "one"},
            {"id": "b", "role": "assistant", "content": "two"},
        ]
    )
    assert partial_out["status"] == "partial"
    assert partial_out["ingested"] == 1
    assert partial_out["failed"] == 1

    failed = TranscriptAdapter(_StubRuntime(["error", "error"]))  # type: ignore[arg-type]
    failed_out = failed.ingest_messages(
        [
            {"id": "a", "role": "user", "content": "one"},
            {"id": "b", "role": "assistant", "content": "two"},
        ]
    )
    assert failed_out["status"] == "failed"
    assert failed_out["failed"] == 2

    disabled = TranscriptAdapter(_StubRuntime(["disabled", "disabled"]))  # type: ignore[arg-type]
    disabled_out = disabled.ingest_messages(
        [
            {"id": "a", "role": "user", "content": "one"},
            {"id": "b", "role": "assistant", "content": "two"},
        ]
    )
    assert disabled_out["status"] == "disabled"
    assert disabled_out["disabled"] == 2
    assert disabled_out["reason"]


def test_disabled_runtime_reports_disabled_not_ok(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, enabled=False)
    adapter = TranscriptAdapter(runtime, default_thread_id="disabled")
    out = adapter.ingest_messages([{"id": "m1", "role": "user", "content": "never stored"}])
    assert out["status"] == "disabled"
    assert out["ingested"] == 0
    assert out["failed"] == 0
    runtime.store.close()


def test_evidence_and_state_context_survive_ingest(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    adapter = TranscriptAdapter(runtime, default_thread_id="evidence")
    out = adapter.ingest_messages(
        [
            {
                "id": "m1",
                "role": "assistant",
                "content": "Patch under verification",
                "evidence_refs": ["src/file.py#L22-L24"],
                "evidence_items": [
                    {
                        "source_id": "src/file.py",
                        "locator": "src/file.py#L22-L24",
                        "line_start": 22,
                        "line_end": 24,
                        "excerpt": "additive patch",
                        "checksum": "sha256:" + "0" * 64,
                    }
                ],
                "state_context": {
                    "scope_id": "auth-timeout-fix",
                    "state_kind": "agent_local_work",
                    "status": "active",
                    "owner_agent_id": "implementer",
                },
            }
        ]
    )
    assert out["ingested"] == 1
    stored = _events(runtime)[-1]
    assert stored["evidence_items"][0]["source_id"] == "src/file.py"
    assert "src/file.py#L22-L24" in stored["evidence_refs"]
    assert stored["state_context"]["owner_agent_id"] == "implementer"
    runtime.store.close()


def test_object_message_with_content_parts_and_name(tmp_path: Path) -> None:
    adapter = TranscriptAdapter(_runtime(tmp_path))
    rows = adapter.normalize_messages(
        [
            _ObjectMessage(
                role="assistant",
                content=[{"type": "text", "text": "object parts"}],
                id="o1",
                name="reviewer",
            )
        ]
    )
    assert rows[0].content == "object parts"
    assert rows[0].metadata.get("agent_name") == "reviewer"


def test_autogen_adapter_dedupes_replay(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    adapter = AutoGenAdapter(runtime, default_thread_id="ag_replay")
    messages = [{"id": "a1", "source": "user", "content": "Please audit"}]
    assert adapter.ingest_messages(messages)["ingested"] == 1
    replay = adapter.ingest_messages(messages)
    assert replay["ingested"] == 0
    assert replay["deduped"] == 1
    runtime.store.close()
