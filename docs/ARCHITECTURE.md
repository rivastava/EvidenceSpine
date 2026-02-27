# Architecture

## Goal
Provide shared memory continuity across agent sessions without stuffing full logs into prompts.

## Core design
1. Append-only event stream (`events.jsonl`)
2. Normalized fact stream (`facts.jsonl`)
3. Top-k retrieval with recency/salience/evidence weighting
4. Optional hybrid/vector retrieval backend adapters
5. Bounded brief assembly (`current_goal`, `locked_decisions`, `recent_verified_facts`, `active_risks`, `open_items`, `next_actions`)
6. Portable handoff packets for agent-to-agent transfer
7. Fail-open: memory subsystem failure does not block caller

## Data flow
`intent -> decision -> action -> outcome -> reflection`

Each event can generate one or more fact candidates.

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

These hooks allow plugging EvidenceSpine into any orchestrator.

Framework adapters provided:
- `evidencespine.adapters.LangGraphAdapter`
- `evidencespine.adapters.AutoGenAdapter`

## Reliability model
- Dedupe by event hash + window
- Sensitive value redaction on persistence
- Fallback brief/handoff behavior if operations fail
- JSON artifacts are deterministic and auditable
