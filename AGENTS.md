# EvidenceSpine — agent usage guide

## What this is

EvidenceSpine is an evidence-bound memory side-car. It stores claims with
attached evidence (checksummed file excerpts) so "verified" means *checkable*.
This repository's agent sessions share one store (`.evidencespine/`). The
spine is a map — the repo is ground truth.

## The evidence model

- Facts have states: `asserted` (claimed), `verified` (evidence-backed),
  `contradicted`, `superseded`.
- `verified` REQUIRES grounding: a checksummed excerpt OR verification
  provenance (method + reference). Ungrounded "verified" claims are stored as
  `asserted` — do not fight the policy; ground the claim instead.
- Evidence items bind claims to exact file excerpts with sha256 checksums. The
  drift-checker re-verifies them against live files and flags stale evidence.

## Decision rules — when to use what

- **Before asserting project state** ("tests green", "feature shipped",
  "this is fixed"): consult the spine first — build a brief
  (MCP: `build_brief`; resource: `evidencespine://brief/{thread}`;
  CLI: `evidencespine brief`).
- **When claiming something is fixed/done/verified**: ground it — attach a
  checksummed excerpt (MCP: `ground`; CLI: `ingest --ground-ref path#L1-L2`)
  and mark `fact_state=verified`. No excerpt, no "verified".
- **After editing files a claim cites**: re-verify — run drift-check
  (MCP: `check_drift`; CLI: `evidencespine drift-check`) so stale claims are
  flagged instead of believed.
- **When a test/gate verifies a claim**: record provenance
  (MCP: `verify_fact`; CLI: `evidencespine verify`) with method + reference.
- **On receiving a handoff packet**: import it and orient
  (MCP: `import_handoff`; CLI: `evidencespine handoff --import`) —
  `locked_decisions` are binding, `required_validations` come first.
- **On role change or session end**: emit a handoff
  (MCP: `emit_handoff`; CLI: `evidencespine handoff`) so the successor resumes
  with verified state.
- **When claims conflict**: query the contradictions view
  (MCP: `query_view`; CLI: `evidencespine view contradictions`).
- **When you resolve something deferred/open**: record it as a verified
  outcome; supersede-on-fix removes it from locked decisions automatically.

## Hygiene

- Ingest events for meaningful state, not trivia.
- Cite what you claim: attach `evidence_refs` or grounded items to every fact
  that matters.
- Locked decisions are binding unless superseded by new evidence. Act against
  a verified fact only by recording the contradiction.

## How to reach the spine

- MCP tools: `ingest_event`, `build_brief`, `query_view`, `emit_handoff`,
  `import_handoff`, `ground`, `check_drift`, `verify_fact`, `reconcile`,
  `snapshot`, `prune`.
- MCP resources: `evidencespine://brief/{thread}`, `//view/{view}`,
  `//state/{scope}`, `//snapshot`, `//guide`.
- CLI: `evidencespine <ingest|brief|view|handoff|ground|drift-check|verify|snapshot|prune>`.
- A2A / async runtime / adapters expose the same capabilities.
