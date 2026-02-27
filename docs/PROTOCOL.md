# Protocol v1

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

## Event schema
Required:
- `thread_id`
- `event_type`
- `source_agent_id`
- `source_turn_id`

Optional key fields:
- `payload.claim`, `payload.decision`, `payload.outcome`
- `evidence_refs`
- `confidence`, `salience`

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
- `source_snapshot`
