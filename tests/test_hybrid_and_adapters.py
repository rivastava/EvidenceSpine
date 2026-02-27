from pathlib import Path

from evidencespine.adapters import AutoGenAdapter, LangGraphAdapter
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


class _MockVectorBackend:
    def score_texts(self, query, texts):
        return [1.0 if "target_vector" in str(t) else 0.0 for t in texts]


def _runtime(tmp_path: Path, *, mode: str = "lexical", topk: int = 24, vector_backend=None) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    settings.retrieval_mode = mode
    settings.retrieval_lexical_weight = 0.0 if mode in {"hybrid", "vector"} else 1.0
    settings.retrieval_vector_weight = 1.0 if mode in {"hybrid", "vector"} else 0.0
    settings.brief_top_k_events = topk
    settings.brief_top_k_facts = topk
    return AgentMemoryRuntime(config=settings.to_runtime_config(), vector_backend=vector_backend)


def test_hybrid_vector_backend_changes_ranking(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, mode="hybrid", topk=1, vector_backend=_MockVectorBackend())
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "source_agent_id": "impl",
            "source_turn_id": "a",
            "payload": {"claim": "ordinary claim"},
            "evidence_refs": ["a.md#L1"],
        }
    )
    rt.ingest_event(
        {
            "thread_id": "demo",
            "event_type": "decision",
            "source_agent_id": "impl",
            "source_turn_id": "b",
            "payload": {"claim": "target_vector prioritized claim"},
            "evidence_refs": ["b.md#L1"],
        }
    )

    brief = rt.build_brief("demo", "unrelated query").to_dict()
    combined = "\n".join(
        brief.get("locked_decisions", [])
        + brief.get("recent_verified_facts", [])
        + brief.get("open_items", [])
    )
    assert "target_vector" in combined


def test_langgraph_adapter_ingests_state(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    adapter = LangGraphAdapter(rt, default_thread_id="lg_thread")
    out = adapter.ingest_state(
        {
            "messages": [
                {"id": "m1", "role": "user", "content": "Investigate drift"},
                {"id": "m2", "role": "assistant", "content": "Applying patch"},
            ]
        }
    )
    assert out["status"] == "ok"
    assert out["ingested"] == 2
    snap = rt.snapshot()
    assert snap["agent_memory_events_24h"] >= 2


def test_autogen_adapter_ingests_messages(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    adapter = AutoGenAdapter(rt, default_thread_id="ag_thread")
    out = adapter.ingest_messages(
        [
            {"id": "a1", "source": "user", "content": "Please audit"},
            {"id": "a2", "source": "assistant", "content": "Audit complete"},
        ]
    )
    assert out["status"] == "ok"
    assert out["ingested"] == 2
    brief = adapter.brief("what changed")
    assert brief["thread_id"] == "ag_thread"
