---
name: evidencespine
description: Evidence-bound memory side-car for project state, grounding, drift checks and handoffs.
whenToUse: Before claiming something is verified, fixed, or shipped, or when importing or emitting a handoff packet.
user-invocable: true
---
# EvidenceSpine — agent usage guide

Canonical guide. `AGENTS.md`, the `evidencespine://guide` MCP resource, and the
MCP `initialize.instructions` primer derive from this text.

## What this is

EvidenceSpine is an evidence-bound memory side-car for multi-agent work. It
stores structured events and claims ("facts") with checksummed evidence
excerpts, and derives bounded briefs, control views, and portable handoff
packets. The spine is a map — the repository is ground truth.

One workspace shares one store (`.evidencespine/`, SQLite by default; JSONL
is opt-in; `EVIDENCESPINE_BASE_DIR` overrides).

## The evidence model

- Facts have states: `asserted` (claimed), `verified` (evidence-backed),
  `contradicted`, `superseded`.
- `verified` REQUIRES grounding: a checksummed excerpt OR verification
  provenance (method + reference). Ungrounded "verified" claims are stored as
  `asserted` — do not fight the policy; ground the claim instead.
- Ingest-time "verified" is self-consistency only: sha256(excerpt) must equal
  the stored checksum. It does NOT prove the live source still matches;
  `check_drift` re-reads the source and is what proves that.
- Evidence items bind a fact to an exact file span (`source_id`, line or char
  anchors, `excerpt`, `checksum`). Drift re-verifies the checksummed span
  against live files and flags stale facts; stale evidence is never emitted as
  verified in a handoff.
- Supersede-on-fix: a newer verified claim supersedes asserted twins, and
  resolved items stop surfacing as locked decisions.
- Handoff claim rows carry `verification` and `evidence_stale`, so
  provenance and drift state survive emit/import round trips.

## Golden session loop

1. Start: build a brief for the workspace thread (MCP `build_brief`; resource
   `evidencespine://brief/{thread}`; CLI `evidencespine brief`). Treat
   `locked_decisions` as binding and `required_validations` as first work;
   check `metadata.stale` — if true, memory is thin, so rely on the repo.
2. Work: ingest decisions, outcomes, gates and blockers as events as you make
   them (MCP `ingest_event`; CLI `evidencespine ingest`), with grounded
   evidence for claims that matter.
3. Before claiming verified/fixed/shipped: ground the exact span
   (MCP `ground`; CLI `evidencespine ground path#L12-L14`), attach it as
   `evidence_items` (or CLI `--ground-ref path#L1-L2`), and set
   `fact_state=verified`. If a test/gate/tool verified it instead, record
   provenance with `verify_fact` (method + reference). No evidence, no
   "verified".
4. After editing files a claim cites: run drift-check (MCP `check_drift`;
   CLI `evidencespine drift-check`, dry-run by default) so stale evidence is
   flagged instead of believed. Add CLI `--apply` to persist the flags.
5. Hand off: emit a packet on role change or session end (MCP
   `emit_handoff`; CLI `evidencespine handoff`). The successor imports it and
   orients (MCP `import_handoff`; CLI `evidencespine handoff --import
   packet.json`) — `locked_decisions` are binding, `required_validations`
   come first.

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
  (MCP: `query_view`; CLI: `evidencespine view contradictions`) and record the
  contradiction; never silently pick a side.
- **When you resolve something deferred/open**: record it as a verified
  outcome; supersede-on-fix removes it from locked decisions automatically.

## Contradictions

Contradictions surface in the `contradictions` view, as "CONTRADICTION" risks
in briefs, and as `unresolved_contradictions` in handoffs. Resolve them with
new grounded evidence and ingest the outcome; a locked decision is not
permission to ignore contradicting evidence.

## Task-state queries

- MCP `query_view` / CLI `evidencespine view <view>` views: `active_scopes`,
  `my_work`, `open_gates`, `stale_claims`, `contradictions` (hyphen
  aliases accepted; unknown views return `status: invalid`, never unfiltered
  rows).
- State travels with events as `state_context` (`scope_id`, `state_kind`,
  `status`, owner, freshness, validation basis); the views derive the live
  row per (thread, scope).
- MCP `snapshot` / CLI `evidencespine snapshot` for health metrics
  (citation coverage, stale evidence, fail-open count, store totals).
- MCP resource `evidencespine://state/{scope}` for one scope's state.

## Thread identity

- Threads are workspace-scoped: `proj_<sha1(realpath(git toplevel))[:12]>`,
  so sessions and subdirectories of one checkout share one brief.
- Override with `EVIDENCESPINE_THREAD_ID` or an explicit `--thread-id`.
- Provider session ids are metadata (`source_turn_id`/`session_id`), not
  thread ids. Use one thread per workspace unless you want isolation.

## Tool map

- MCP tools: `ingest_event`, `build_brief`, `query_view`, `emit_handoff`,
  `import_handoff`, `ground`, `check_drift`, `verify_fact`, `reconcile`,
  `snapshot`, `prune`.
- MCP resources: `evidencespine://guide`, `//brief/{thread}`,
  `//view/{view}`, `//state/{scope}`, `//snapshot`.
- CLI: `evidencespine <ingest|ground|drift-check|verify|brief|view|handoff|snapshot|reconcile|prune|migrate|mcp|a2a|harness|chat>`.
- A2A (`evidencespine a2a`): `ingest_event`, `append_fact`,
  `build_brief`, `emit_handoff`, `import_handoff`, `query_view`,
  `reconcile`, `snapshot`, `prune` — no `ground`/`check_drift`/`verify_fact`.
- Async runtime: ingest/append_fact/brief/emit/import/query_view/reconcile/
  snapshot/prune/check_evidence_stale.
- Adapters (Transcript/LangGraph/AutoGen): transcript ingest then brief/handoff;
  ingest reports status ok|partial|failed|disabled plus deduped/new_events.
- Key env vars: `EVIDENCESPINE_BASE_DIR`, `EVIDENCESPINE_STORAGE_FORMAT`,
  `EVIDENCESPINE_THREAD_ID`, `EVIDENCESPINE_SOURCE_ROOT` (MCP grounding
  confinement), `EVIDENCESPINE_BRIEF_LOOKBACK_HOURS` (default 168),
  `EVIDENCESPINE_MCP_AUTH_TOKEN`, `EVIDENCESPINE_A2A_AUTH_TOKEN`,
  `EVIDENCESPINE_ENABLE`, `EVIDENCESPINE_FAIL_OPEN`.

## Handoff rules

- Emit before a role change or session end (harness hooks do this
  automatically); work from the packet before editing shared state.
- `locked_decisions` are binding; `required_validations` come first.
- Contradictions travel as `unresolved_contradictions`: resolve them with new
  grounded evidence, not by editing the packet.
- Importing the same packet twice is idempotent (dedupe); failed rows are
  counted in `failed_ingests` with a bounded `errors` list.

## Hygiene & fail-open

- Ingest meaningful state (decisions, outcomes, blockers, gates), not chat
  trivia. Cite every claim that matters.
- Memory failures fail open: hooks and MCP calls return a fail-open brief or
  warning instead of blocking the caller; never treat that as data. Restart a
  stale long-lived server on `snapshot.evidencespine_version` mismatch.
- Locked decisions are binding unless superseded by new evidence. Act against
  a verified fact only by recording the contradiction.
- MCP `prune` previews by default (`dry_run=true`); pass `dry_run=false`
  deliberately.

## How to reach the spine

- MCP tools/resources: see Tool map. Clients get this guide as
  `initialize.instructions`; read `evidencespine://guide` for the full text.
- CLI: `evidencespine <command> --help`; add `--json` for machine output.
- Harness hooks deliver recall/retain automatically and fail open;
  `evidencespine harness doctor --json` validates installed files.
- A2A, the async runtime, and adapters expose subsets only — see Tool map.

## Harness delivery (one line each)

- claude-code: `harness install --harness claude-code` — skills-dir plugin
  (hooks, `.mcp.json`, skill, rules); settings hooks are opt-in.
- opencode: `--harness opencode` — `.opencode/plugins/evidencespine.ts` (V2).
- cursor: `--harness cursor` — `.cursor/mcp.json`, rule, permissions, hooks.
- vscode: `--harness vscode-copilot` — portable `.mcp.json`; `vscode` is the
  legacy `.vscode/mcp.json` Local-harness shape.
- codex: `--harness codex` — `.codex/config.toml` (stdio or HTTP) plus
  `.codex/hooks.json`.
- dsh: `--harness dsh` — machine-wide `$DSH_HOME/cordis.patch.yml` MCP row
  (absolute paths) plus the materialized skill.
- generic: `--harness generic` — canonical `evidencespine.mcp.json` +
  project projections; `harness print-config --harness <client>` prints only.
- instructions: `--harness instructions` — merges this guide into AGENTS.md.
- git: `--harness git` — post-commit/post-merge evidence hooks.
- MCP-only clients: register `evidencespine mcp --base-dir <absolute>` as a
  stdio server; they receive this guide via MCP `instructions`.
- `harness doctor` validates installed files; `harness debug` checks the
  store and reports fail-open.

## Security rules

- Local-first by default: MCP over stdio, streamable-HTTP and A2A bind
  loopback.
- Non-loopback binds are refused unless a bearer token is set
  (`EVIDENCESPINE_MCP_AUTH_TOKEN` / `EVIDENCESPINE_A2A_AUTH_TOKEN`) or the
  explicit `EVIDENCESPINE_{MCP,A2A}_ALLOW_UNAUTHENTICATED=1` dev escape hatch
  is used. Never use the escape hatch on a shared network.
- MCP `ground`, ingest `ground_refs`, and `check_drift` are confined to
  `EVIDENCESPINE_SOURCE_ROOT`; absolute or escaping source ids are rejected or
  reported `unverifiable`. Handoff imports are confined the same way.
- Redaction runs before persistence and hashing, but grounded excerpts are
  persisted verbatim because they are the evidence — never ground secrets.
- No OAuth server and no TLS termination are shipped; terminate TLS at a
  gateway/ssh tunnel and authenticate every remote hop.
