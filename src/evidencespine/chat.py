"""Realtime multi-agent chat over the EvidenceSpine store.

Turns the spine into a live message bus: one polling loop per agent role.
Each loop watches the room thread for new messages, renders the
conversation (rolling window + auto-summary), calls an LLM backend (default:
headless ``opencode run --pure``), and publishes the reply back to the
store. Messages land within a poll interval plus LLM latency; the run stops
on consensus, a quiet period, a message budget, or a duration cap.

Chat messages are stored as events whose payload carries only ``chat``
text (no ``claim``/``decision``/``outcome`` keys), so they never pollute
fact rows or briefs.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings

DEFAULT_PERSONAS: Dict[str, str] = {
    "developer": (
        "You are the DEVELOPER. You care about practicality: buildability, "
        "developer experience, complexity budget, whether features earn their code. "
        "You distrust gold-plating and demand daily benefit."
    ),
    "engineer": (
        "You are the SYSTEMS ENGINEER. You care about architecture: correctness, "
        "concurrency, failure modes, data integrity, and whether mechanisms hold "
        "under real load. You judge features by robustness, not demo value."
    ),
    "engineer2": (
        "You are the RELIABILITY ENGINEER. You care about durability: fail-open "
        "consistency, state divergence across processes, index usage, unbounded "
        "growth, and whether the health metrics actually measure anything. You "
        "distrust any metric that cannot go wrong."
    ),
    "scientist": (
        "You are the RESEARCH SCIENTIST. You care about epistemics: what is "
        "verified vs asserted, citation fidelity, span grounding, and whether "
        "claims can be checked. You distrust unverified assertions."
    ),
    "skeptic": (
        "You are the PRODUCT SKEPTIC. You question whether the thing is worth it: "
        "what real pain it solves, who adopts it, what complexity debt it adds. "
        "You demand evidence of value over architecture aesthetics."
    ),
}


def _default_persona(role: str) -> str:
    return (
        f"You are {role}. You are a thoughtful, evidence-minded participant in a "
        "multi-agent debate. Make a case, address others directly, and concede "
        "when the evidence warrants it."
    )


def _parse_opencode_json_stream(raw: str) -> str:
    parts: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        part = event.get("part") if isinstance(event, dict) else None
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts).strip()


def opencode_backend(*, directory: str, timeout: int = 180) -> Callable[[str], str]:
    """LLM backend that shells out to headless ``opencode run --pure``."""

    def _call(prompt: str) -> str:
        proc = subprocess.run(
            ["opencode", "run", "--pure", "--format", "json", "--dir", directory, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        text = _parse_opencode_json_stream(proc.stdout)
        if not text:
            raise RuntimeError(f"opencode backend returned no text: {proc.stderr[:200]}")
        return text

    return _call


def scripted_backend(replies: Dict[str, List[str]]) -> Callable[[str], str]:
    """Deterministic backend for tests: pops the next reply per role marker.

    The prompt must contain a line ``ROLE: <name>``; replies are keyed by
    that name. When a role has no replies left, it echoes its last line.
    """
    locks = {name: threading.Lock() for name in replies}

    def _call(prompt: str) -> str:
        role = ""
        for line in prompt.splitlines():
            if line.startswith("ROLE:"):
                role = line.split(":", 1)[1].strip()
                break
        pool = replies.get(role, [])
        with locks.get(role, threading.Lock()):
            if pool:
                return pool.pop(0)
            return "AGREE: acknowledged"

    return _call


def render_conversation(room_id: str, topic: str, messages: List[Dict[str, Any]]) -> str:
    lines = [f"Topic: {topic}", "Conversation so far:"]
    for msg in messages:
        agent = str(msg.get("agent", "host"))
        text = str(msg.get("text", "")).strip()
        lines.append(f"[{agent}] {text}")
    return "\n".join(lines)


def _trim_reply(text: str, max_words: int) -> str:
    """Enforce a hard word cap on a reply, preserving a trailing AGREE clause."""
    raw = (text or "").strip()
    if not raw:
        return "."
    agree = ""
    agree_words = 0
    if "AGREE:" in raw:
        head, tail = raw.split("AGREE:", 1)
        agree = f"AGREE:{tail.strip()}"
        agree_words = len(agree.split())
        raw = head.strip()
    words = raw.split()
    if len(words) + agree_words <= max_words:
        base = " ".join(words)
        return f"{base} {agree}".strip() if agree else base
    head_budget = max(1, max_words - agree_words)
    prefix = " ".join(words[:head_budget])
    idx = -1
    for boundary in (". ", "? ", "! "):
        found = prefix.rfind(boundary)
        if found > idx:
            idx = found
    if idx > 0:
        trimmed = prefix[: idx + 1].rstrip()
    else:
        trimmed = prefix.rstrip()
    trimmed += "…"
    return f"{trimmed} {agree}".strip() if agree else trimmed


@dataclass
class ChatAgent:
    role: str
    persona: str
    backend: Callable[[str], str]
    last_seq: int = 0
    stop: threading.Event = field(default_factory=threading.Event)


class ChatCoordinator:
    """Runs one polling loop per role over a shared spine room."""

    def __init__(
        self,
        runtime: AgentMemoryRuntime,
        *,
        thread_id: str = "chat",
        room_id: str = "chat",
        topic: str = "",
        interval: float = 1.5,
        quiet_secs: float = 12.0,
        max_messages: int = 24,
        max_reply_words: int = 60,
        window_size: int = 12,
        minutes: float = 0.0,
        facilitate: bool = False,
        facilitator_budget: int = 3,
        summarizer_backend: Optional[Callable[[str], str]] = None,
        facilitator_backend: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.runtime = runtime
        self.thread_id = thread_id
        self.room_id = room_id
        self.topic = topic
        self.interval = max(0.2, float(interval))
        self.quiet_secs = max(1.0, float(quiet_secs))
        self.max_messages = max(4, int(max_messages))
        self.max_reply_words = max(10, int(max_reply_words))
        self.window_size = max(4, int(window_size))
        self.minutes = max(0.0, float(minutes))
        self.facilitate = bool(facilitate)
        self.facilitator_budget = max(0, int(facilitator_budget))
        self.summarizer_backend = summarizer_backend
        self.facilitator_backend = facilitator_backend
        self._publish_lock = threading.Lock()
        self._seq = self._max_existing_seq()
        self.messages: List[Dict[str, Any]] = []
        self.agents: List[ChatAgent] = []
        self._stop = threading.Event()
        self._last_publish_ts = 0.0
        self._summarized_through = 0
        self._facilitator_pumps = 0
        self.stopped_by = "unknown"

    def _max_existing_seq(self) -> int:
        seq = 0
        for row in self.runtime.store.iter_events():
            meta = row.get("metadata", {}) if isinstance(row.get("metadata", {}), dict) else {}
            if meta.get("chat_room") == self.room_id:
                try:
                    seq = max(seq, int(meta.get("chat_seq", 0)))
                except (TypeError, ValueError):
                    pass
        return seq

    def _read_messages(self) -> List[Dict[str, Any]]:
        rows = [
            row
            for row in self.runtime.store.iter_events()
            if (row.get("metadata", {}).get("chat_room") if isinstance(row.get("metadata", {}), dict) else None)
            == self.room_id
        ]
        rows.sort(key=lambda r: int(r.get("metadata", {}).get("chat_seq", 0)))
        out: List[Dict[str, Any]] = []
        for row in rows:
            meta = row.get("metadata", {})
            payload = row.get("payload", {}) if isinstance(row.get("payload", {}), dict) else {}
            out.append(
                {
                    "seq": int(meta.get("chat_seq", 0)),
                    "agent": str(row.get("source_agent_id", "host")),
                    "text": str(payload.get("chat", "")),
                    "summary": bool(meta.get("chat_summary", False)),
                }
            )
        return out

    def publish(self, text: str, agent: str = "host", extra_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._publish_lock:
            self._seq += 1
            seq = self._seq
            result = self.runtime.ingest_event(
                {
                    "thread_id": self.thread_id,
                    "event_type": "reflection",
                    "role": agent,
                    "source_agent_id": agent,
                    "source_turn_id": f"chat:{self.room_id}:{seq}",
                    "payload": {"chat": text, "room": self.room_id},
                    "metadata": {
                        "chat_room": self.room_id,
                        "chat_seq": seq,
                        "chat_topic": self.topic,
                        **(extra_meta or {}),
                    },
                    "confidence": 0.7,
                    "salience": 0.3,
                }
            )
            self.messages.append({"seq": seq, "agent": agent, "text": text, "summary": bool((extra_meta or {}).get("chat_summary"))})
            self._last_publish_ts = time.time()
            return result

    def add_agent(self, role: str, persona: str, backend: Callable[[str], str]) -> None:
        self.agents.append(ChatAgent(role=role, persona=persona, backend=backend))

    def _latest_summary(self, messages: List[Dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("summary"):
                return f"[summary of earlier messages]\n{msg['text']}"
        return ""

    def _render_for_agent(self, messages: List[Dict[str, Any]]) -> str:
        """Render summary + the most recent window of non-summary messages."""
        live = [m for m in messages if not m.get("summary")]
        summary = self._latest_summary(messages)
        window = live[-self.window_size :]
        lines = [f"Topic: {self.topic}", "Conversation so far:"]
        if summary:
            lines.append(summary)
        for msg in window:
            lines.append(f"[{msg['agent']}] {msg['text']}")
        return "\n".join(lines)

    def _all_agreed(self) -> bool:
        if not self.agents:
            return False
        latest_by_agent: Dict[str, str] = {}
        for msg in self.messages:
            latest_by_agent[str(msg.get("agent", ""))] = str(msg.get("text", ""))
        return all(
            "AGREE:" in str(latest_by_agent.get(agent.role, "")).upper()
            for agent in self.agents
        )

    def _agent_loop(self, agent: ChatAgent) -> None:
        while not agent.stop.is_set() and not self._stop.is_set():
            try:
                messages = self._read_messages()
                if not messages:
                    time.sleep(self.interval)
                    continue
                latest_live = [m for m in messages if not m.get("summary")]
                if not latest_live:
                    time.sleep(self.interval)
                    continue
                latest = latest_live[-1]
                if latest["seq"] <= agent.last_seq or latest["agent"] == agent.role:
                    agent.last_seq = max(agent.last_seq, messages[-1]["seq"])
                    time.sleep(self.interval)
                    continue
                if self._all_agreed():
                    agent.last_seq = max(agent.last_seq, messages[-1]["seq"])
                    time.sleep(self.interval)
                    continue
                history = self._render_for_agent(messages)
                prompt = (
                    f"{agent.persona}\n\n"
                    f"ROLE: {agent.role}\n"
                    f"You are chatting in realtime about: {self.topic}\n\n"
                    f"{history}\n\n"
                    f"Reply as {agent.role}. HARD LIMIT: at most {self.max_reply_words} words "
                    f"({max(1, self.max_reply_words // 20)} sentences) - never exceed it, be crisp. "
                    f"Address the latest message. Stay in character. When you reach agreement, "
                    f"end with 'AGREE: <points>'."
                )
                reply = agent.backend(prompt)
                reply = _trim_reply(reply, self.max_reply_words)
                reply = reply.strip() or "."
                self.publish(reply, agent=agent.role)
                agent.last_seq = latest["seq"]
                print(f"[{agent.role}] {reply}", flush=True)
            except Exception as exc:
                print(f"[{agent.role}] loop error: {exc}", flush=True)
                time.sleep(self.interval)

    def run(self) -> Dict[str, Any]:
        self.publish(f"Kickoff: {self.topic}", agent="host")
        threads = []
        for agent in self.agents:
            thread = threading.Thread(target=self._agent_loop, args=(agent,), daemon=True)
            threads.append(thread)
            thread.start()

        deadline = time.time() + (self.minutes * 60.0) if self.minutes > 0 else None
        while not self._stop.is_set():
            messages = self._read_messages()
            landed = len(messages)
            if landed >= self.max_messages:
                self.stopped_by = "max_messages"
                break
            if self._all_agreed():
                self.stopped_by = "consensus"
                break
            if deadline is not None and time.time() >= deadline:
                self.stopped_by = "duration"
                break
            if landed >= 2 and time.time() - self._last_publish_ts >= self.quiet_secs:
                if not self.facilitate or self._facilitator_pumps >= self.facilitator_budget:
                    self.stopped_by = "quiet"
                    break
                self._facilitator_pump(messages)
                continue
            if self.summarizer_backend is not None and landed - self._summarized_through > self.window_size + 4:
                self._summarize(messages)
            time.sleep(min(self.interval, 0.5))

        self._stop.set()
        for agent in self.agents:
            agent.stop.set()
        for thread in threads:
            thread.join(timeout=3.0)
        return {
            "status": "ok",
            "room_id": self.room_id,
            "thread_id": self.thread_id,
            "stopped_by": self.stopped_by,
            "messages": list(self.messages),
            "message_count": len(self.messages),
        }

    def _summarize(self, messages: List[Dict[str, Any]]) -> None:
        """Roll earlier messages into a summary event so prompts stay bounded."""
        backend = self.summarizer_backend
        if backend is None:
            return
        live = [m for m in messages if not m.get("summary")]
        if len(live) - self._summarized_through <= self.window_size:
            return
        cut = live[: max(0, len(live) - self.window_size)]
        cut = [m for m in cut if m["seq"] > self._summarized_through]
        if not cut:
            return
        transcript = "\n".join(f"[{m['agent']}] {m['text']}" for m in cut)
        prompt = (
            "You are the summarizer for an ongoing agent debate.\n"
            f"Topic: {self.topic}\n\n"
            f"Messages to summarize:\n{transcript}\n\n"
            "Write a neutral 2-3 sentence summary of the positions and agreements so far. "
            "HARD LIMIT: at most 60 words."
        )
        try:
            text = backend(prompt)
            text = _trim_reply(text, 60)
            self._summarized_through = cut[-1]["seq"]
            self.publish(f"{text}", agent="summarizer", extra_meta={"chat_summary": 1, "chat_summarized_through": cut[-1]["seq"]})
            print(f"[summarizer] {text}", flush=True)
        except Exception as exc:
            print(f"[summarizer] error: {exc}", flush=True)

    def _facilitator_pump(self, messages: List[Dict[str, Any]]) -> None:
        """Revive a quiet room with a facilitator question or challenge."""
        if self.facilitator_backend is None:
            return
        history = self._render_for_agent(messages)
        prompt = (
            "You are the FACILITATOR of a realtime agent debate.\n"
            f"Topic: {self.topic}\n\n{history}\n\n"
            "The debate has stalled. Ask ONE sharp follow-up question or challenge "
            "that pushes the agents toward a concrete decision. "
            "HARD LIMIT: at most 30 words."
        )
        try:
            text = self.facilitator_backend(prompt)
            text = _trim_reply(text, 30)
            self._facilitator_pumps += 1
            self.publish(text, agent="facilitator")
            print(f"[facilitator] {text}", flush=True)
        except Exception as exc:
            print(f"[facilitator] error: {exc}", flush=True)


def build_runtime(base_dir: Optional[str], storage_format: Optional[str]) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(base_dir or ".evidencespine"), storage_format=storage_format)
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def run_chat(
    *,
    topic: str,
    roles: List[str],
    thread_id: str = "chat",
    room_id: str = "chat",
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    llm: str = "opencode",
    directory: str = ".",
    interval: float = 1.5,
    quiet_secs: float = 12.0,
    max_messages: int = 24,
    max_reply_words: int = 60,
    window_size: int = 12,
    minutes: float = 0.0,
    facilitate: bool = False,
    facilitator_budget: int = 3,
    summarize: bool = True,
    timeout: int = 180,
) -> Dict[str, Any]:
    """Run a realtime multi-agent chat on a shared spine room."""
    runtime = build_runtime(base_dir, storage_format)
    try:
        summarizer_backend = None
        facilitator_backend = None
        if summarize and llm != "scripted":
            summarizer_backend = opencode_backend(directory=directory, timeout=timeout)
        if facilitate and llm != "scripted":
            facilitator_backend = opencode_backend(directory=directory, timeout=timeout)
        coordinator = ChatCoordinator(
            runtime,
            thread_id=thread_id,
            room_id=room_id,
            topic=topic,
            interval=interval,
            quiet_secs=quiet_secs,
            max_messages=max_messages,
            max_reply_words=max_reply_words,
            window_size=window_size,
            minutes=minutes,
            facilitate=facilitate,
            facilitator_budget=facilitator_budget,
            summarizer_backend=summarizer_backend,
            facilitator_backend=facilitator_backend,
        )
        for role in roles:
            if llm == "scripted":
                backend = scripted_backend({role: [f"{role} says: agreed in principle"]})
            elif llm == "echo":
                def _echo(prompt: str) -> str:
                    return f"{role} echoes: message received"
                backend = _echo
            else:
                backend = opencode_backend(directory=directory, timeout=timeout)
            coordinator.add_agent(role, DEFAULT_PERSONAS.get(role, _default_persona(role)), backend)
        return coordinator.run()
    finally:
        runtime.store.close()
