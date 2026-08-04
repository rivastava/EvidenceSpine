from __future__ import annotations

import time
from pathlib import Path

from evidencespine.chat import ChatCoordinator, render_conversation, scripted_backend
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def _runtime(tmp_path: Path) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def _scripted_room(tmp_path: Path, max_messages: int = 10, quiet_secs: float = 1.5, interval: float = 0.2) -> ChatCoordinator:
    rt = _runtime(tmp_path)
    room = ChatCoordinator(rt, thread_id="chat", room_id="room1", topic="test topic", interval=interval, quiet_secs=quiet_secs, max_messages=max_messages)
    replies = {
        "dev": ["dev: implement it", "dev: AGREE: ship it"],
        "eng": ["eng: it must be robust", "eng: AGREE: with tests"],
        "aud": ["aud: verify it", "aud: AGREE: verified"],
    }
    for role, pool in replies.items():
        room.add_agent(role, f"You are {role}", scripted_backend({role: pool}))
    return room


def test_chat_messages_land_in_spine_without_fact_pollution(tmp_path: Path) -> None:
    room = _scripted_room(tmp_path)
    room.run()

    chat_rows = [
        row
        for row in room.runtime.store.iter_events()
        if row.get("metadata", {}).get("chat_room") == "room1"
    ]
    assert chat_rows, "chat messages must be stored"
    assert all("chat" in (row.get("payload", {}) or {}) for row in chat_rows)
    assert len(list(room.runtime.store.iter_facts())) == 0, "chat messages must not create facts"
    seqs = [int(row["metadata"]["chat_seq"]) for row in chat_rows]
    assert seqs == sorted(seqs), "chat seq must be monotonic"
    room.runtime.store.close()


def test_chat_agents_take_turns_and_reply(tmp_path: Path) -> None:
    room = _scripted_room(tmp_path)
    out = room.run()

    roles = {m["agent"] for m in out["messages"]}
    assert "host" in roles
    assert {"dev", "eng", "aud"} <= roles, f"all agents should speak, got {roles}"
    texts = [m["text"] for m in out["messages"] if m["agent"] != "host"]
    assert any("AGREE:" in t for t in texts), "convergence marker expected"

    agents_seen = [m["agent"] for m in out["messages"] if m["agent"] != "host"]
    assert agents_seen, "agents must have spoken"
    assert all(m["seq"] == i + 1 for i, m in enumerate(out["messages"])), "seq continuity"
    room.runtime.store.close()


def test_chat_stops_at_max_messages(tmp_path: Path) -> None:
    room = _scripted_room(tmp_path, max_messages=7)
    out = room.run()
    assert len(out["messages"]) <= 7 + len(room.agents), "cap may overshoot by in-flight agents"
    room.runtime.store.close()


def test_chat_converges_on_quiet_period(tmp_path: Path) -> None:
    room = _scripted_room(tmp_path, max_messages=200, quiet_secs=1.0, interval=0.2)
    started = time.time()
    out = room.run()
    elapsed = time.time() - started
    assert 1 <= len(out["messages"]) <= 12, f"should converge quickly, got {len(out['messages'])}"
    assert elapsed < 30, "quiet-period convergence must be bounded"
    room.runtime.store.close()


def test_render_conversation_includes_roles() -> None:
    rendered = render_conversation(
        "room1",
        "topic",
        [{"seq": 1, "agent": "host", "text": "kickoff"}, {"seq": 2, "agent": "dev", "text": "reply"}],
    )
    assert "[host] kickoff" in rendered
    assert "[dev] reply" in rendered
    assert "Topic: topic" in rendered


def test_chat_reply_length_cap_is_enforced_and_preserves_agree(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    room = ChatCoordinator(rt, thread_id="chat", room_id="cap", topic="cap test", interval=0.2, quiet_secs=1.0, max_messages=8, max_reply_words=40)

    def verbose_backend(prompt: str) -> str:
        return "word " * 120 + "AGREE: done with this point"

    room.add_agent("dev", "dev persona", verbose_backend)
    room.add_agent("aud", "aud persona", verbose_backend)
    room.run()

    stored = [
        row
        for row in rt.store.iter_events()
        if row.get("metadata", {}).get("chat_room") == "cap"
    ]
    replies = [
        str(row["payload"]["chat"])
        for row in stored
        if row.get("source_agent_id") != "host"
    ]
    assert replies, "agents must reply"
    for reply in replies:
        words = len(reply.split())
        assert words <= 40, f"reply exceeds cap: {words} words: {reply[:80]}"
        if "AGREE:" in reply:
            agree_words = reply.split("AGREE:", 1)[1].split()
            head_words = reply.split("AGREE:", 1)[0].split()
            assert head_words and len(head_words) + len(agree_words) <= 40, "AGREE clause must fit the cap"
    rt.store.close()


def test_chat_rolling_summary_keeps_prompts_bounded(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    seen_prompts: list[str] = []

    def capturing_backend(prompt: str) -> str:
        seen_prompts.append(prompt)
        return "reply"

    def summarizer_backend(prompt: str) -> str:
        return "SUMMARY of the earlier exchange"

    room = ChatCoordinator(
        rt, thread_id="chat", room_id="sum", topic="sum test",
        interval=0.1, quiet_secs=0.6, max_messages=30, window_size=4,
        summarizer_backend=summarizer_backend,
    )
    room.add_agent("dev", "dev persona", capturing_backend)
    room.add_agent("aud", "aud persona", capturing_backend)
    room.run()

    stored = [
        row
        for row in rt.store.iter_events()
        if row.get("metadata", {}).get("chat_room") == "sum"
    ]
    summaries = [row for row in stored if row.get("metadata", {}).get("chat_summary")]
    assert summaries, "a rolling summary must be published for a long room"

    for prompt in seen_prompts:
        if "SUMMARY of the earlier exchange" in prompt:
            lines = [ln for ln in prompt.splitlines() if ln.startswith("[dev]") or ln.startswith("[aud]")]
            assert len(lines) <= 4, f"window must bound rendered messages, got {len(lines)}"
            return
    raise AssertionError("no agent prompt ever included the summary")


def test_chat_facilitator_revives_quiet_room(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)

    def facilitator_backend(prompt: str) -> str:
        return "Facilitator: what is the concrete next step?"

    def stateful_backend(prompt: str) -> str:
        if "Facilitator:" in prompt:
            return "AGREE: answered the facilitator"
        return "point taken"

    room = ChatCoordinator(
        rt, thread_id="chat", room_id="fac", topic="fac test",
        interval=0.2, quiet_secs=0.8, max_messages=12, facilitate=True,
        facilitator_budget=2, facilitator_backend=facilitator_backend,
    )
    room.add_agent("dev", "dev persona", stateful_backend)
    room.run()

    stored = [
        row
        for row in rt.store.iter_events()
        if row.get("metadata", {}).get("chat_room") == "fac"
    ]
    pumps = [row for row in stored if row.get("source_agent_id") == "facilitator"]
    assert pumps, "facilitator must pump when the room stalls"
    texts = [str(row["payload"]["chat"]) for row in stored if row.get("source_agent_id") != "host"]
    assert any("answered the facilitator" in t for t in texts), "agents must reply to the pump"
    rt.store.close()


def test_chat_duration_cap_stops_run(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)

    def endless_backend(prompt: str) -> str:
        return "still talking"

    room = ChatCoordinator(
        rt, thread_id="chat", room_id="dur", topic="dur test",
        interval=0.2, quiet_secs=30.0, max_messages=500, minutes=0.08,
    )
    room.add_agent("dev", "dev persona", endless_backend)
    room.run()
    assert room.stopped_by == "duration"
    rt.store.close()
