from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from evidencespine import chat as chat_module
from evidencespine.chat import ChatCoordinator, render_conversation, run_chat, scripted_backend
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings

ROOT = Path(__file__).resolve().parent.parent


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


def test_chat_stops_exactly_at_max_messages(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    room = ChatCoordinator(
        rt, thread_id="chat", room_id="budget", topic="budget test",
        interval=0.1, quiet_secs=30.0, max_messages=9,
    )

    def endless_backend(prompt: str) -> str:
        return "still talking, no agreement"  # deliberately no AGREE marker

    room.add_agent("dev", "dev persona", endless_backend)
    room.add_agent("eng", "eng persona", endless_backend)
    out = room.run()
    assert out["stopped_by"] == "max_messages"
    assert out["message_count"] == 9, f"must stop exactly at the budget, got {out['message_count']}"
    assert len(out["messages"]) == 9, "in-flight replies must not overshoot the budget"
    rt.store.close()


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


def _fail_backend_factory(**kwargs):
    def _fail(prompt: str) -> str:
        raise RuntimeError("backend exploded")

    return _fail


def test_run_chat_terminates_when_opencode_backend_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a dead LLM must not hang the room (default minutes == 0)."""
    monkeypatch.setattr(chat_module, "opencode_backend", _fail_backend_factory)
    started = time.time()
    out = run_chat(
        topic="failure test",
        roles=["dev", "eng"],
        base_dir=str(tmp_path / ".es"),
        llm="opencode",
        interval=0.1,
        quiet_secs=0.5,
        max_messages=50,
        summarize=False,
        facilitate=False,
    )
    elapsed = time.time() - started
    assert elapsed < 20, f"run_chat must terminate on backend failure, took {elapsed:.1f}s"
    assert out["stopped_by"] == "backend_failure"
    assert out["message_count"] == 1, "only the kickoff should have landed"
    assert out["status"] == "error"


def test_run_chat_wall_clock_cap_applies_when_minutes_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a hanging/slow backend is bounded by the derived wall-clock cap."""
    monkeypatch.setattr(chat_module, "opencode_backend", _slow_backend_factory(delay=1.5))
    room = chat_module.ChatCoordinator(
        _runtime(tmp_path), quiet_secs=30.0, interval=0.1, max_messages=50
    )
    assert room._derived_wall_secs() <= 180.0, "derived backstop must stay bounded"

    started = time.time()
    out = run_chat(
        topic="slow backend",
        roles=["dev"],
        base_dir=str(tmp_path / ".es"),
        llm="opencode",
        interval=0.1,
        quiet_secs=30.0,
        max_messages=50,
        summarize=False,
        facilitate=False,
        max_wall_secs=2.0,
    )
    elapsed = time.time() - started
    assert elapsed < 15, f"explicit wall cap must win, took {elapsed:.1f}s"
    assert out["stopped_by"] in {"max_wall_secs", "duration"}


def _slow_backend_factory(*, delay: float):
    def _factory(**kwargs):
        def _slow(prompt: str) -> str:
            time.sleep(delay)
            return "slow reply"

        return _slow

    return _factory


def test_chat_facilitate_without_backend_does_not_spin(tmp_path: Path) -> None:
    """Regression: facilitate=True with no facilitator backend must terminate."""
    rt = _runtime(tmp_path)
    calls = {"n": 0}

    def slowish_backend(prompt: str) -> str:
        calls["n"] += 1
        time.sleep(0.02)
        return "a point without agreement"

    room = ChatCoordinator(
        rt, thread_id="chat", room_id="nofac", topic="nofac test",
        interval=0.05, quiet_secs=0.5, max_messages=100,
        facilitate=True, facilitator_budget=3,
    )
    room.add_agent("dev", "dev persona", slowish_backend)
    outcome: dict = {}

    def _run() -> None:
        outcome.update(room.run())

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=12.0)
    assert not worker.is_alive(), "facilitate without a backend must not busy-spin forever"
    assert outcome["stopped_by"] == "quiet"
    assert calls["n"] <= 20, f"facilitator-less room must stay bounded, got {calls['n']} calls"
    rt.store.close()


def test_echo_backend_labels_own_role_and_spawns_no_opencode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: echo replies carry their own role and never build opencode."""

    def _forbidden_factory(**kwargs):
        raise AssertionError("--llm echo must not construct an opencode backend")

    monkeypatch.setattr(chat_module, "opencode_backend", _forbidden_factory)
    out = run_chat(
        topic="echo test",
        roles=["alpha", "beta"],
        base_dir=str(tmp_path / ".es"),
        llm="echo",
        interval=0.05,
        quiet_secs=0.6,
        max_messages=12,
        summarize=True,
        facilitate=True,
    )
    replies = [m for m in out["messages"] if m["agent"] in {"alpha", "beta"}]
    assert replies, "echo agents must reply"
    by_role = {m["agent"]: m["text"] for m in replies}
    assert "alpha echoes:" in by_role.get("alpha", "")
    assert "beta echoes:" in by_role.get("beta", "")
    assert "beta echoes:" not in by_role.get("alpha", ""), "role closure must not be late-bound"
    assert "alpha echoes:" not in by_role.get("beta", "")


def test_chat_cli_echo_path_terminates(tmp_path: Path) -> None:
    """The CLI path must terminate without a real opencode binary."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "evidencespine.cli",
            "chat",
            "--llm",
            "echo",
            "--no-summarize",
            "--max-messages",
            "6",
            "--quiet-secs",
            "0.4",
            "--interval",
            "0.05",
            "--json",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert proc.returncode == 0, proc.stderr
    assert '"stopped_by"' in proc.stdout


def test_chat_cli_exits_nonzero_when_backend_fails(tmp_path: Path) -> None:
    """End-to-end F1: a dead opencode binary must terminate the CLI promptly."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "opencode"
    fake.write_text("#!/bin/sh\necho boom >&2\nexit 3\n", encoding="utf-8")
    fake.chmod(0o755)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    started = time.time()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "evidencespine.cli",
            "chat",
            "--llm",
            "opencode",
            "--no-summarize",
            "--max-messages",
            "4",
            "--quiet-secs",
            "1",
            "--interval",
            "0.1",
            "--max-wall-secs",
            "20",
            "--json",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    elapsed = time.time() - started
    assert elapsed < 30, f"CLI chat must not hang on backend failure, took {elapsed:.1f}s"
    assert proc.returncode == 3, f"backend failure must exit nonzero: {proc.returncode} {proc.stderr}"
    assert '"stopped_by": "backend_failure"' in proc.stdout


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
