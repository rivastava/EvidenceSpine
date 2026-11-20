"""Harness-agnostic delivery hooks.

The research ground for v0.5.0 ("Delivery, Not Storage") is that agents make
almost no voluntary memory calls; recall and retain must be properties of the
harness. These handlers implement the two auto-behaviors shared by every
supported harness:

* session-start   auto-recall: a bounded context brief is injected into the
  conversation/system context before work begins.
* session-stop    auto-retain: the session is recorded as an evidence-bound
  reflection event, and (optionally) a handoff packet is emitted so a successor
  can resume with verified state.
* precompact      retain-through-compaction: memory state is injected into
  the compaction summary so the continuation keeps working state.

Thread identity: a hook payload carries a per-session
identifier (session_id / conversation_id), never a workspace thread. Using it
as the memory thread made every session start from an empty brief. Thread ids
are therefore resolved with evidencespine.threads.resolve_thread_id
(explicit value -> EVIDENCESPINE_THREAD_ID -> stable workspace hash) and the
session id travels in source_turn_id / metadata.session_id only.

Payload hygiene: provider control fields (reason, trigger,
source, prompt_input_exit, manual, auto, clear) are metadata, never summaries.
Only real content fields (compact_summary/custom_instructions/summary/note) map
to a stored outcome.

Injected context is capped at the harness limit (10,000 chars for Claude Code
and Codex); over-cap text is replaced with a file path/preview by the harness,
so the cap is applied explicitly here.

All handlers fail open: a memory error never breaks the agent session.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional

from evidencespine.render import render_brief_markdown
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings
from evidencespine.threads import resolve_thread_id

#: Claude Code and Codex cap injected hook context at 10,000 characters.
CONTEXT_CHAR_LIMIT = 10000

#: Real content fields, in priority order. Provider control fields are NOT here.
_SUMMARY_FIELDS = ("compact_summary", "custom_instructions", "conversation_summary", "summary", "note")

#: Values that are provider control signals, never summary prose.
_CONTROL_VALUES = {
    "manual",
    "auto",
    "clear",
    "resume",
    "logout",
    "startup",
    "other",
    "compact",
    "prompt_input_exit",
    "session_stop",
    "precompact",
}

#: Payload keys that carry the project/working directory for thread resolution.
_CWD_KEYS = ("cwd", "project_dir", "projectDir", "workspace", "working_directory", "workingDirectory", "directory")


def _env_bool(name: str, default: bool = False) -> bool:
    value = str(os.getenv(name, "")).strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def build_runtime(
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=base_dir, storage_format=storage_format)
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def default_thread_id(cwd: str = "") -> str:
    """Workspace-scoped thread id for a hook run (proj_<hash>)."""
    return resolve_thread_id("", cwd=cwd or os.getcwd())


def read_stdin_json() -> Dict[str, Any]:
    """Parse JSON passed to a hook on stdin. Returns {} when none is present."""
    try:
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
            if raw and raw.strip():
                return dict(json.loads(raw))
    except Exception:
        return {}
    return {}


def hook_cwd(input_json: Optional[Dict[str, Any]]) -> str:
    """Project directory a hook payload refers to (falls back to process cwd)."""
    payload = input_json if isinstance(input_json, dict) else {}
    for key in _CWD_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = payload.get("session")
    if isinstance(nested, dict):
        value = nested.get("cwd")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return os.getcwd()


def extract_session_id(input_json: Optional[Dict[str, Any]]) -> str:
    """Per-session identifier from a hook payload (metadata only, not a thread)."""
    payload = input_json if isinstance(input_json, dict) else {}
    for key in ("session_id", "sessionID", "conversation_id", "conversationId"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("session", "properties"):
        nested = payload.get(key)
        if not isinstance(nested, dict):
            continue
        for inner in ("id", "session_id"):
            value = nested.get(inner)
            if isinstance(value, str) and value.strip():
                return value.strip()
        info = nested.get("info")
        if isinstance(info, dict):
            value = info.get("id")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def extract_thread_id(thread_id: Optional[str], input_json: Optional[Dict[str, Any]]) -> str:
    """Resolve the workspace thread id for a hook payload.

    Per-session ids (session_id/conversation_id) are deliberately not used:
    they change every session and would fragment recall. Use
    extract_session_id for metadata/source_turn_id instead.
    """
    payload = input_json if isinstance(input_json, dict) else {}
    explicit = str(thread_id or "").strip()
    if not explicit:
        value = payload.get("thread_id")
        if isinstance(value, str) and value.strip():
            explicit = value.strip()
    return resolve_thread_id(explicit, cwd=hook_cwd(payload))


def extract_hook_metadata(input_json: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Control fields worth remembering as metadata (never as summary text)."""
    payload = input_json if isinstance(input_json, dict) else {}
    meta: Dict[str, str] = {}
    session_id = extract_session_id(payload)
    if session_id:
        meta["session_id"] = session_id
    for key in ("reason", "trigger", "source", "hook_event_name"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            meta[key] = value.strip()
    cwd = hook_cwd(payload)
    if cwd:
        meta["cwd"] = cwd
    return meta


def _coerce_summary(explicit: Optional[str], input_json: Dict[str, Any]) -> str:
    """Return genuine summary content from explicit arg or provider payload.

    Provider control fields (reason/trigger/source/...) and their signal values
    are rejected so e.g. PreCompact {"trigger":"auto"} or SessionEnd
    {"reason":"prompt_input_exit"} can never become the stored session outcome.
    """
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    payload = input_json if isinstance(input_json, dict) else {}
    for key in _SUMMARY_FIELDS:
        value = payload.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text and text.lower() not in _CONTROL_VALUES:
                return text
            continue
        if isinstance(value, dict):
            inner = value.get("summary")
            if isinstance(inner, str) and inner.strip() and inner.strip().lower() not in _CONTROL_VALUES:
                return inner.strip()
    return ""


def _cap_context(text: str, limit: int = CONTEXT_CHAR_LIMIT) -> str:
    """Truncate injected context to the harness limit with an explicit marker."""
    clean = str(text or "")
    budget = max(64, int(limit))
    if len(clean) <= budget:
        return clean
    marker = "\n\n[EvidenceSpine context truncated to fit the harness context cap]"
    return clean[: budget - len(marker)] + marker


def _clean_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in meta.items() if value not in (None, "", [], {})}


def handle_session_start(
    *,
    thread_id: Optional[str] = None,
    objective: str = "",
    token_budget: Optional[int] = None,
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    include_instructions: bool = True,
    input_json: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> str:
    """Auto-recall: render a bounded context brief for injection into context."""
    try:
        payload = dict(input_json or {})
        thread = extract_thread_id(thread_id, payload)
        runtime = build_runtime(base_dir, storage_format)
        budget = int(token_budget) if token_budget is not None and int(token_budget) > 0 else None
        brief = runtime.build_brief(
            thread_id=thread,
            query=str(objective) or "session start",
            token_budget=budget,
        )
        text = render_brief_markdown(brief.to_dict())
        if include_instructions:
            text = (
                "Project memory from EvidenceSpine follows: a bounded brief of "
                "evidence-bound working state for this session. Locked decisions are "
                "binding unless superseded by new evidence. Verified facts carry "
                "checksums; re-run drift-check after editing cited files. Open items "
                "and next actions are the working queue. Decision rules: consult the "
                "spine before asserting project state; ground any claim marked "
                "verified (checksummed excerpt or verification provenance); emit a "
                "handoff at role change or session end.\n\n" + text
            )
        return _cap_context(text)
    except Exception as exc:
        return f"[EvidenceSpine fail-open: session-start unavailable: {exc}]"


def handle_session_stop(
    *,
    thread_id: Optional[str] = None,
    summary: Optional[str] = None,
    auto_handoff: Optional[bool] = None,
    reason: str = "session_stop",
    input_json: Optional[Dict[str, Any]] = None,
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Auto-retain: record the session as an evidence-bound reflection event.

    SessionEnd runs under a ~1.5s budget in Claude Code (plugin hook timeouts do
    not raise it), so this path stays tiny: one event ingest. A handoff packet is
    only emitted when explicitly requested or EVIDENCESPINE_AUTO_HANDOFF=1.

    reason (clear/logout/prompt_input_exit/...) is recorded as metadata, never as
    the stored outcome/claim body.
    """
    try:
        payload = dict(input_json or {})
        thread = extract_thread_id(thread_id, payload)
        sid = str(session_id or "").strip() or extract_session_id(payload)
        hook_reason = str(payload.get("reason") or "").strip() or str(reason or "session_stop")
        body = _coerce_summary(summary, payload)
        runtime = build_runtime(base_dir, storage_format)
        if not body and _env_bool("EVIDENCESPINE_SESSION_SUMMARY", True):
            try:
                snap = runtime.snapshot()
                body = (
                    f"session health: {int(snap.get('agent_memory_events_24h', 0))} events, "
                    f"{int(snap.get('agent_memory_verified_facts_24h', 0))} verified facts, "
                    f"{int(snap.get('agent_handoff_packets_emitted_24h', 0))} handoffs, "
                    f"{float(snap.get('agent_claim_citation_coverage_24h', 0.0)):.2f} citation coverage, "
                    f"{int(snap.get('agent_evidence_stale_count_24h', 0))} stale evidence, "
                    f"{int(snap.get('agent_memory_fail_open_events_24h', 0))} fail-open"
                )
            except Exception:
                body = ""
        wants_handoff = bool(auto_handoff) or _env_bool("EVIDENCESPINE_AUTO_HANDOFF", False)
        metadata = _clean_metadata(
            {
                "harness_hook": "session_stop",
                "reason": hook_reason,
                "session_id": sid,
                "auto_handoff": bool(wants_handoff),
            }
        )
        result = runtime.ingest_event(
            {
                "thread_id": thread,
                "event_type": "reflection",
                "role": "operator",
                "source_agent_id": "evidencespine_harness",
                "source_turn_id": f"session_stop:{hook_reason}",
                "payload": {
                    "claim": "session ended",
                    "outcome": body[:4096],
                    "next_actions": (["emit_handoff"] if wants_handoff else []),
                },
                "confidence": 0.7,
                "salience": 0.5,
                "metadata": metadata,
            }
        )
        handoff = None
        if wants_handoff:
            try:
                handoff = runtime.emit_handoff(role="auditor", thread_id=thread, scope="session handoff").to_dict()
            except Exception as exc:
                handoff = {"status": "fail_open", "reason": str(exc)}
        return {"status": "ok", "ingest": result, "thread_id": thread, "session_id": sid, "handoff": handoff}
    except Exception as exc:
        return {"status": "fail_open", "reason": str(exc)}


def handle_precompact(
    *,
    thread_id: Optional[str] = None,
    summary: Optional[str] = None,
    token_budget: Optional[int] = None,
    input_json: Optional[Dict[str, Any]] = None,
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Retain-through-compaction: persist the summary and inject working state.

    The returned text is plain-text stdout, which Claude Code uses as
    compaction-summary instructions and Codex ignores (Codex re-anchors through
    its SessionStart compact matcher). The pre-compaction ingest is the durable
    side effect.
    """
    try:
        payload = dict(input_json or {})
        thread = extract_thread_id(thread_id, payload)
        sid = str(session_id or "").strip() or extract_session_id(payload)
        body = _coerce_summary(summary, payload)
        runtime = build_runtime(base_dir, storage_format)
        if body:
            metadata = _clean_metadata(
                {
                    "harness_hook": "precompact",
                    "session_id": sid,
                    "trigger": str(payload.get("trigger") or "").strip(),
                }
            )
            runtime.ingest_event(
                {
                    "thread_id": thread,
                    "event_type": "reflection",
                    "role": "operator",
                    "source_agent_id": "evidencespine_harness",
                    "source_turn_id": "precompact",
                    "payload": {
                        "claim": "conversation summary captured before compaction",
                        "outcome": body[:4096],
                    },
                    "confidence": 0.6,
                    "salience": 0.6,
                    "metadata": metadata,
                }
            )
        budget = int(token_budget) if token_budget is not None and int(token_budget) > 0 else None
        brief = runtime.build_brief(thread_id=thread, query="retain through compaction", token_budget=budget)
        return _cap_context(render_brief_markdown(brief.to_dict()))
    except Exception as exc:
        return f"[EvidenceSpine fail-open: precompact unavailable: {exc}]"
