# Architecture

## Goal
Provide shared memory continuity across agent sessions without stuffing full logs into prompts.

EvidenceSpine is a side-car. Runtime truth can be reconciled into it, but the repo does not claim to replace live system state.

## Core design
1. Append-only event stream (`events.jsonl`)
2. Normalized fact stream (`facts.jsonl`)
3. Top-k retrieval with recency/salience/evidence weighting
4. Optional hybrid/vector retrieval backend adapters
5. Bounded brief assembly (`current_goal`, `locked_decisions`, `recent_verified_facts`, `active_risks`, `open_items`, `next_actions`)
6. Portable handoff packets for agent-to-agent transfer
7. Derived active-state control views over append-only history
8. Fail-open: memory subsystem failure does not block caller

## Claim grounding

Protocol v2 adds claim-to-span grounding without changing the storage engine:

- `evidence_refs` remain the backward-compatible ref string layer.
- `evidence_items` add exact line or character anchors, optional excerpts, checksums, and verification metadata.
- Brief citations and handoff claim rows can now point to exact spans while still exposing compatible ref lists.

Grounding quality is observable through snapshot metrics:
- ref citation coverage
- span-grounded citation coverage
- excerpt fidelity
- handoff span grounding rate

## Agent-state control layer

Protocol v2 now also carries optional `state_context` on events, facts, brief citations, and handoff rows.

That makes three things possible without changing storage:
- ownership visibility: who owns a live scope right now
- freshness visibility: whether a gate or blocker is stale, fresh, or unknown
- local-vs-global separation: task work, blockers, pending gates, and runtime-validated state can be represented explicitly instead of inferred from prose

EvidenceSpine derives five public views from that data:
- `active_scopes`
- `my_work`
- `open_gates`
- `stale_claims`
- `contradictions`

## Data flow
`intent -> decision -> action -> outcome -> reflection`

Each event can generate one or more fact candidates.
Structured evidence items and optional `state_context` propagate from events into derived facts, then into briefs and handoffs.

## Adapter pipeline

Recommended integration path:

`messages[] / state -> TranscriptAdapter normalization -> ingest_event -> fact extraction -> brief / handoff`

Framework wrappers (`LangGraphAdapter`, `AutoGenAdapter`) are thin schema-level layers on top of this transcript adapter pipeline. This keeps the package dependency-free while making adapter transformations testable outside runtime writes.

## Storage
Default base dir: `.evidencespine/`
- `events.jsonl`
- `facts.jsonl`
- `state.json`
- `briefs/*.json`
- `handoffs/*.json`

## Extensibility hooks
`RuntimeHooks` supports:
- `on_event(event_dict)`
- `on_brief(brief_dict)`
- `on_handoff(packet_dict)`
- `contradiction_pass(query, facts) -> list[dict]`
- `reconcile_state(thread_id, active_scope_rows) -> list[dict]`

These hooks allow plugging EvidenceSpine into any orchestrator without hardcoding framework-specific or project-specific truth adapters into the package.

Framework adapters provided:
- `evidencespine.adapters.TranscriptAdapter`
- `evidencespine.adapters.LangGraphAdapter`
- `evidencespine.adapters.AutoGenAdapter`

## Reliability model
- Dedupe by event hash + window
- Sensitive value redaction on persistence
- Fallback brief/handoff behavior if operations fail
- JSON artifacts are deterministic and auditable

Protocol v2 keeps this model intact:
- no storage migration
- no runtime dependency additions
- mixed refs-only and span-grounded artifacts remain readable
- control views are derived, not stored in a secondary database
