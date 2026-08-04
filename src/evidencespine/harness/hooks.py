"""Harness-agnostic delivery hooks.

The research ground for v0.5.0 ("Delivery, Not Storage") is that agents make
almost no voluntary memory calls; recall and retain must be properties of the
harness. These handlers implement the two auto-behaviors shared by every
supported harness:

* ``session-start``   auto-recall: a bounded context brief is injected into the
  conversation/system context before work begins.
* ``session-stop``    auto-retain: the session is recorded as an evidence-bound
  reflection event, and (optionally) a handoff packet is emitted so a successor
  can resume with verified state.
* ``precompact``      retain-through-compaction: memory state is injected into
  the compaction summary so the continuation keeps working state.

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


def default_thread_id() -> str:
    """Thread id for a hook run: env override, else the current directory name."""
    env = str(os.getenv("EVIDENCESPINE_THREAD_ID", "")).strip()
    if env:
        return env
    return os.path.basename(os.getcwd()) or "default"


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


def _coerce_summary(explicit: Optional[str], input_json: Dict[str, Any]) -> str:
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    for key in ("conversation_summary", "summary"):
        value = input_json.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            inner = value.get("summary")
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    return ""


def _extract_thread_id(thread_id: Optional[str], input_json: Dict[str, Any]) -> str:
    if thread_id and str(thread_id).strip():
        return str(thread_id).strip()
    for key in ("thread_id", "project_id"):
        value = input_json.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default_thread_id()


def handle_session_start(
    *,
    thread_id: Optional[str] = None,
    objective: str = "",
    token_budget: Optional[int] = None,
    base_dir: Optional[str] = None,
    storage_format: Optional[str] = None,
    include_instructions: bool = True,
) -> str:
    """Auto-recall: render a bounded context brief for injection into context."""
    try:
        runtime = build_runtime(base_dir, storage_format)
        budget = int(token_budget) if token_budget is not None and int(token_budget) > 0 else None
        brief = runtime.build_brief(
            thread_id=thread_id or default_thread_id(),
            query=str(objective) or "session start",
            token_budget=budget,
        )
        text = render_brief_markdown(brief.to_dict())
        if include_instructions:
            text = (
                "EvidenceSpine delivery: below is your evidence-bound working state "
                "for this session. Locked decisions are binding unless superseded by "
                "new evidence. Act against a verified fact only if you record the "
                "contradiction. Open items and next actions are your queue. "
                "Decision rules: consult the spine before asserting project state; "
                "ground any claim you mark verified (checksummed excerpt or "
                "verification provenance — ungrounded verified claims are stored as "
                "asserted); re-verify cited evidence after editing files (drift-"
                "check); emit a handoff at role change or session end.\n\n" + text
            )
        return text
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
) -> Dict[str, Any]:
    """Auto-retain: record the session as an evidence-bound reflection event.

    If ``auto_handoff`` is truthy (env ``EVIDENCESPINE_AUTO_HANDOFF=1``) a
    handoff packet is also emitted so a successor can resume with verified state.
    """
    try:
        payload = dict(input_json or {})
        thread = _extract_thread_id(thread_id, payload)
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
        result = runtime.ingest_event(
            {
                "thread_id": thread,
                "event_type": "reflection",
                "role": "operator",
                "source_agent_id": "evidencespine_harness",
                "source_turn_id": reason,
                "payload": {
                    "claim": f"session ended: {reason}",
                    "outcome": body[:4096],
                    "next_actions": ["emit_handoff" if auto_handoff else ""],
                },
                "confidence": 0.7,
                "salience": 0.5,
                "metadata": {"harness_hook": reason, "auto_handoff": bool(auto_handoff)},
            }
        )
        handoff = None
        if bool(auto_handoff):
            try:
                handoff = runtime.emit_handoff(role="auditor", thread_id=thread, scope="session handoff").to_dict()
            except Exception as exc:
                handoff = {"status": "fail_open", "reason": str(exc)}
        return {"status": "ok", "ingest": result, "thread_id": thread, "handoff": handoff}
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
) -> str:
    """Retain-through-compaction: persist the summary and inject working state.

    The summary (when provided) is ingested as an evidence-bound reflection; the
    returned text is a bounded note for the compaction context so the
    continuation prompt keeps memory state.
    """
    try:
        payload = dict(input_json or {})
        thread = _extract_thread_id(thread_id, payload)
        body = _coerce_summary(summary, payload)
        runtime = build_runtime(base_dir, storage_format)
        if body:
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
                    "metadata": {"harness_hook": "precompact"},
                }
            )
        budget = int(token_budget) if token_budget is not None and int(token_budget) > 0 else None
        brief = runtime.build_brief(thread_id=thread, query="retain through compaction", token_budget=budget)
        return render_brief_markdown(brief.to_dict())
    except Exception as exc:
        return f"[EvidenceSpine fail-open: precompact unavailable: {exc}]"
