# Architecture

## Goal
Provide shared memory continuity across agent sessions without stuffing full logs into prompts.

EvidenceSpine is a side-car. Runtime truth can be reconciled into it, but the repo does not claim to replace live system state.

## Core design
1. Append-only event stream (`events` table in SQLite; legacy `events.jsonl`)
2. Normalized fact stream (`facts` table in SQLite; legacy `facts.jsonl`)
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
- `evidencespine.db` — SQLite (default since v0.5.0), WAL mode, indexed `events`/`facts`/`dedup_hashes` tables
- `events.jsonl` / `facts.jsonl` — legacy JSONL storage (opt-in via `storage_format=jsonl`); existing JSONL stores are auto-migrated to SQLite on open
- `state.json` — counters and in-memory hash ring (JSONL fallback state)
- `briefs/*.json` — bounded brief artifacts (random UUID filenames; thread/role live in the JSON)
- `handoffs/*.json` — handoff packet artifacts (random UUID filenames)

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
- Dedupe by event hash + window (in-memory ring and SQLite `dedup_hashes` both expire against `dedupe_window_sec`)
- Sensitive value redaction on persistence (see SECURITY.md for the trade-off)
- Fallback brief/handoff behavior if operations fail
- JSON artifacts are deterministic and auditable
- Artifact filenames are random UUIDs written atomically (`O_EXCL`), so caller-controlled identifiers can never traverse out of the configured directories

Protocol v2 keeps this model intact:
- no storage migration (SQLite is the default; JSONL legacy rows auto-migrate once)
- no runtime dependency additions
- mixed refs-only and span-grounded artifacts remain readable
- control views are derived, not stored in a secondary database
