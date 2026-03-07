# Protocol v2

EvidenceSpine Protocol v2 adds structured evidence spans while preserving `evidence_refs` for backward compatibility.

## Fact states
- `asserted`
- `verified`
- `contradicted`
- `superseded`

## Roles
- `implementer`
- `auditor`
- `researcher`
- `operator`
- `unknown`

## Evidence item

Structured grounding is represented as `evidence_items: list[EvidenceItem]`.

Required:
- `source_id`
- at least one anchor:
  - `locator`
  - `line_start`
  - `char_start`

Optional:
- `line_end`
- `char_end`
- `excerpt`
- `checksum`
- `confidence`
- `verification_state`
- `metadata`

Normalization rules:
- `line_end` defaults to `line_start` when omitted.
- `char_end` defaults to `char_start` when omitted.
- `verification_state` uses the same vocabulary as fact states.
- Evidence items are projected back into compatible `evidence_refs`.

## Event schema

Required:
- `thread_id`
- `event_type`
- `source_agent_id`
- `source_turn_id`

Optional key fields:
- `payload.claim`, `payload.decision`, `payload.outcome`
- `evidence_refs`
- `evidence_items`
- `confidence`, `salience`

Notes:
- Event artifacts now emit `schema_version: "v2"`.
- Event hashes include normalized `evidence_items`, so distinct spans do not dedupe accidentally.

## Fact schema

Derived facts inherit:
- `evidence_refs`
- `evidence_items`

Fact ids remain backward-compatible and are still derived from thread, claim, and source turn.

## Brief schema

Required:
- `thread_id`
- `query`
- `generated_at`

Sections:
- `current_goal`
- `locked_decisions`
- `recent_verified_facts`
- `active_risks`
- `open_items`
- `next_actions`

Citations:
- `citations` now maps claim strings to structured citation objects:
  - `primary_ref`
  - `evidence_refs`
  - `evidence_items`
  - `span_grounded`
- `citation_refs` is emitted as a legacy alias for refs-only consumers.

## Handoff schema

Required:
- `role`
- `thread_id`
- `scope`
- `claims`

Includes:
- `checksum`
- `locked_decisions`
- `unresolved_contradictions`
- `required_validations`
- `evidence_refs`
- `evidence_items`
- `source_snapshot`

Claim rows and contradiction rows may also carry:
- `evidence_refs`
- `evidence_items`
- `span_grounded`

## Compatibility

- `validate_*` helpers accept both `v1` refs-only payloads and `v2` span-grounded payloads.
- Existing `evidence_refs` callers do not need to change immediately.
- New writers emit `schema_version: "v2"` and preserve `evidence_refs` alongside `evidence_items`.
