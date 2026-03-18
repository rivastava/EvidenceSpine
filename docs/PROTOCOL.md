# Protocol v2

EvidenceSpine Protocol v2 adds structured evidence spans while preserving `evidence_refs` for backward compatibility. In `0.4.0`, Protocol v2 is extended additively with `state_context` and derived control views. `schema_version` remains `v2`.

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
- `state_context`
- `confidence`, `salience`

Notes:
- Event artifacts now emit `schema_version: "v2"`.
- Event hashes include normalized `evidence_items`, so distinct spans do not dedupe accidentally.

## Fact schema

Derived facts inherit:
- `evidence_refs`
- `evidence_items`
- `state_context`

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
  - `state_context` when the underlying claim carries structured control-state
- `citation_refs` is emitted as a legacy alias for refs-only consumers.

Brief metadata may also include:
- `active_scope_count`
- `open_gate_count`
- `stale_scope_count`

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
- `state_context`

## State context

Structured control-state is represented as `state_context: StateContext`.

Required when present:
- `scope_id`
- `state_kind`
- `status`

Allowed values:
- `scope_kind`: `task | gate | blocker | runtime_state | thread`
- `state_kind`: `agent_local_work | global_blocker | pending_gate | runtime_validated_state`
- `status`: `active | blocked | ready | closed | superseded`
- `state_basis`: `reported | runtime_validated | derived | imported`

Normalization rules:
- `scope_kind` defaults from `state_kind`
- `state_basis` defaults from `state_kind`
- `runtime_validated` rows require `validated_at` and `validated_by`
- live `global_blocker`, `pending_gate`, and `runtime_validated_state` rows require `fresh_until`
- `lease_expires_at` requires `owner_agent_id`
- `status = superseded` requires `supersedes`

Example:

```json
{
  "scope_id": "auth-timeout-fix",
  "scope_kind": "task",
  "state_kind": "agent_local_work",
  "status": "active",
  "owner_agent_id": "implementer",
  "state_basis": "reported",
  "lease_expires_at": "2026-03-18T12:00:00Z"
}
```

## Control views

EvidenceSpine derives active control views from append-only history without adding an on-disk index.

Public views:
- `active_scopes`
- `my_work`
- `open_gates`
- `stale_claims`
- `contradictions`

Each row includes:
- `scope_id`
- `thread_id`
- `scope_kind`
- `state_kind`
- `status`
- `owner_agent_id`
- `state_basis`
- `claim`
- `freshness_state`
- `lease_state`
- `has_contradiction`
- `conflict`
- `evidence_refs`
- `evidence_items`

## Compatibility

- `validate_*` helpers accept both `v1` refs-only payloads and `v2` span-grounded payloads.
- Legacy rows without `state_context` remain valid and readable.
- Existing `evidence_refs` callers do not need to change immediately.
- New writers emit `schema_version: "v2"` and preserve `evidence_refs` alongside `evidence_items`.
