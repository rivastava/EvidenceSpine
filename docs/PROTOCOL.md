# Protocol v2

EvidenceSpine Protocol v2 adds structured evidence spans while preserving
`evidence_refs` for backward compatibility. 0.6.0 keeps `schema_version:
"v2"` and extends the protocol **additively** (no breaking shape changes):
`append_fact` policy, provenance/staleness keys on handoff rows, thread-id
conventions, drift verdicts, and citation-size constants. Control-state
(`state_context`) is unchanged.

## Fact states

- `asserted` — claimed, not evidence-backed
- `verified` — evidence-backed (grounded span or verification provenance)
- `contradicted`
- `superseded`

## Roles

`implementer`, `auditor`, `researcher`, `operator`, `unknown`.

## Thread identity

Threads identify **workspaces**, not sessions:

- Canonical id: `proj_<sha1(realpath(git toplevel))[:12]>`.
- Resolution order: explicit `thread_id` > `EVIDENCESPINE_THREAD_ID` >
  workspace hash. Harness hooks/CLI/MCP/A2A resolve through
  `evidencespine.threads.resolve_thread_id`.
- Provider session ids are **not** thread ids: they travel in
  `source_turn_id` / `metadata.session_id`. One thread per workspace unless
  isolation is deliberate.

## Evidence item

Structured grounding is `evidence_items: list[EvidenceItem]`.

Required: `source_id` and at least one anchor (`locator`,
`line_start`, or `char_start`).

Optional: `line_end`, `char_end`, `excerpt`, `checksum`,
`confidence`, `verification_state`, `metadata`.

Normalization: `line_end` defaults to `line_start`; `char_end` defaults
to `char_start`; character spans are 0-based with an inclusive end over the
decoded UTF-8 text; `verification_state` uses the fact-state vocabulary.
Evidence items are projected back into compatible `evidence_refs`
(`source_id` + span). Grounding is line-anchor, character-anchor, or
locator-only.

## Verified-requires-span policy

With `verified_requires_span` (default on; env
`EVIDENCESPINE_VERIFIED_REQUIRES_SPAN`):

- A `"verified"` claim must be grounded: a checksummed excerpt
  (`excerpt` + matching sha256 `checksum`) or verification provenance
  (`method` in `test|gate|tool|manual` plus a non-empty `reference`).
- Ungrounded `verified` claims are stored as `asserted` with
  `metadata.policy = "verified_requires_span"` and counted in the ingest
  result's `policy_downgrades`.
- The policy is enforced identically by `ingest_event` (event-derived facts)
  and by `append_fact` (the raw-fact entrypoint used by the async runtime and
  A2A). Stale evidence is downgraded too: `metadata.policy =
  "stale_evidence"` and `state = asserted`.
- The policy is about **self-consistency** at ingest. `check_drift` is what
  proves the live source still matches.

## Event schema

Required: `thread_id`, `event_type`, `source_agent_id`,
`source_turn_id`.

Optional key fields: `payload.claim`, `payload.decision`,
`payload.outcome`, `evidence_refs`, `evidence_items`,
`state_context`, `confidence`, `salience`.

Notes:

- Event artifacts emit `schema_version: "v2"`.
- Event hashes include normalized `evidence_items`, so distinct spans do not
  dedupe accidentally.
- **Redaction runs before hashing**: the stored `event_hash` binds the
  redacted payload, so a redacted claim's dedupe identity differs from its
  unredacted form (a deliberate 0.6.0 behavior change).
- Events derive facts whose `ts_utc` **inherits the source event time**.

## Fact schema

Derived facts inherit `evidence_refs`, `evidence_items` and
`state_context`.

Fact ids are `amf_<hash>` over (thread, case-insensitive claim,
source_turn_id), so "Fix X" and "fix x" are the same fact; the original claim
text is stored verbatim. Facts may carry `supersedes_fact_id` and
`metadata.verification`.

## Brief schema

Required: `thread_id`, `query`, `generated_at`.

Sections: `current_goal`, `locked_decisions`, `recent_verified_facts`,
`active_risks`, `open_items`, `next_actions`.

Citations: `citations` maps claim strings to structured citation objects
(`primary_ref`, `evidence_refs`, `evidence_items`, `span_grounded`,
`state_context` when present); `citation_refs` is the legacy refs-only
alias.

Brief metadata includes `stale` (no events inside the lookback window —
which defaults to `EVIDENCESPINE_BRIEF_LOOKBACK_HOURS` = 168 h — or the
newest event older than `max(300 s, dedupe window)`),
`lookback_hours`, `unresolved_contradictions`, `active_scope_count`,
`open_gate_count`, `stale_scope_count` and `token_used_estimate`.

### Citation size limit

`CITATION_LINE_MAX = 1024` characters is the per-claim citation/render
budget: brief section lines, citation keys, handoff claim text, risk/provenance
lookups and rendered citation lines are clamped with `safe_text(...,
CITATION_LINE_MAX)`. Long claims are truncated at the protocol boundary rather
than producing unbounded artifacts; the stored fact claim itself allows up to
4096 characters.

## Handoff schema

Required: `role`, `thread_id`, `scope`, `claims`.

Includes: `schema_version`, `packet_id` (deterministic hash when omitted),
`checksum` (canonical payload hash), `locked_decisions`,
`unresolved_contradictions`, `required_validations`, `evidence_refs`,
`evidence_items`, `source_snapshot`, `metadata`.

### Additive provenance keys (0.6.0)

Claim rows and contradiction rows may carry:

- `evidence_refs`, `evidence_items`, `span_grounded`, `state_context`
  (as before);
- **`verification`** — normalized `{method, reference, verified_at,
  verified_by}` or `null`;
- **`evidence_stale`** — boolean; true when the evidence drifted;
- **`metadata`** — free-form row metadata.

Import behavior: claim rows are re-ingested with their `fact_state`
(unknown/invalid states fall back to asserted/contradicted); a row whose
stale flag is set is imported as `asserted` even if it said verified; a
verified row that lacks grounding is downgraded by the same
verified-requires-span policy. `locked_decisions`,
`required_validations` and the packet's evidence travel with the imported
events. Re-importing the same packet is idempotent (dedupe), and failures are
counted in `failed_ingests` with a bounded `errors` list.

## State context

Unchanged from 0.4.0. `state_context: StateContext` requires `scope_id`,
`state_kind`, `status` when present.

Allowed values:

- `scope_kind`: `task | gate | blocker | runtime_state | thread`
- `state_kind`: `agent_local_work | global_blocker | pending_gate |
  runtime_validated_state`
- `status`: `active | blocked | ready | closed | superseded`
- `state_basis`: `reported | runtime_validated | derived | imported`

Normalization: `scope_kind` defaults from `state_kind`; `state_basis`
defaults from `state_kind`; `runtime_validated` rows require
`validated_at` + `validated_by`; live `global_blocker`,
`pending_gate` and `runtime_validated_state` rows require
`fresh_until`; `lease_expires_at` requires `owner_agent_id`;
`status = superseded` requires `supersedes`.

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

Five public views are derived from append-only history: `active_scopes`,
`my_work`, `open_gates`, `stale_claims`, `contradictions`. Hyphenated
aliases are accepted; unknown views return an invalid-view error (never
unfiltered rows).

Rows include: `scope_id`, `thread_id`, `scope_kind`, `state_kind`,
`status`, `owner_agent_id`, `state_basis`, `claim`,
`freshness_state` (fresh/stale/unknown), `lease_state`,
`has_contradiction`, `conflict` (live rows of the same `state_kind`
disagree on status/state_basis), `multi_owner`, `evidence_refs`,
`evidence_items`. Views are **thread-scoped**: the caller supplies
`thread_id` and the resolved live row is per (thread, scope).

## Supersession

- A fact referenced by a live fact's `supersedes_fact_id` is excluded from
  briefs and control rows.
- Ingests accept `payload.supersedes_ref` (or event-level
  `supersedes_fact_id`) to supersede explicitly; `state_context.supersedes`
  is promoted onto the derived fact.
- A newer verified claim auto-supersedes an asserted twin with identical
  (case-insensitive) claim text in the same thread.
- Briefs suppress superseded decision/action event claims, so resolved items
  stop surfacing as locked decisions.

## Verification provenance

- `payload.verification = {method, reference, verified_at, verified_by}` is
  stored as `metadata.verification` on facts.
- `verify_fact` (MCP) / `evidencespine verify` ingest a verified copy of a
  fact (claim + grounded items preserved) that supersedes the original, so the
  brief shows the verified-with-provenance row.
- Snapshot reports `agent_fact_provenance_rate_24h`.

## Evidence staleness (drift)

`check_drift` / `evidencespine drift-check` re-reads each grounded source
at its anchor and recomputes the checksum:

- `changed` — the checksum no longer matches the live span;
- `missing` — the source cannot be read;
- `unverifiable` — the anchor cannot be checked (unsupported/ambiguous
  anchor, out-of-range character span, or a source id outside the allowed
  root). Unverifiable never clears an existing stale flag and never promotes a
  fact.

Preview/apply: `dry_run=true` reports; `dry_run=false` writes
`metadata.evidence_stale = true` + `evidence_stale_reason` +
`evidence_stale_at` (and clears them on a clean re-check). Stale facts surface
in `stale_claims`, briefs (`STALE EVIDENCE (reason): ...` in active risks)
and `agent_evidence_stale_count_24h`.

## Chat rooms

`evidencespine chat` stores messages as events with `payload.chat` (no
`claim`/`decision` keys, so no facts are derived), plus
`metadata.chat_room`, `metadata.chat_seq` (monotonic) and
`metadata.chat_topic`. Summary events carry `metadata.chat_summary = 1` and
`metadata.chat_summarized_through`.

## Compatibility

- `validate_*` helpers accept both v1 refs-only and v2 span-grounded payloads.
- Legacy rows without `state_context` / `verification` /
  `evidence_stale` remain valid and readable.
- Existing `evidence_refs` callers do not need to change; new writers emit
  `schema_version: "v2"` and preserve refs alongside items.
- No storage migration is required for Protocol v2; the only migration is the
  one-time JSONL -> SQLite import (see [ARCHITECTURE.md](ARCHITECTURE.md)).
- Citation/handoff text is clamped at `CITATION_LINE_MAX = 1024` characters;
  consumers that need the full claim should read the fact/event row, not the
  rendered brief line.
