# Architecture

## Goal

Provide shared memory continuity across agent sessions without stuffing full
logs into prompts. EvidenceSpine is a side-car: runtime truth can be reconciled
into it, but it does not claim to replace live system state. The spine is a
map; the repository is ground truth.

0.6.0 does **not** change the architecture: same append-only event/fact model,
same Protocol v2 payloads, same derived artifacts, no new storage engine and no
new mandatory dependency. The release hardens storage behavior, thread
identity, evidence verification and the delivery layer.

## Core design

1. Append-only event stream (`events` table in SQLite; legacy
   `events.jsonl`).
2. Normalized fact stream (`facts` table; legacy `facts.jsonl`), derived
   from events or appended directly (`append_fact`).
3. Top-k retrieval with recency/salience/evidence weighting (lexical default;
   optional hybrid/vector backends).
4. Bounded brief assembly (`current_goal`, `locked_decisions`,
   `recent_verified_facts`, `active_risks`, `open_items`,
   `next_actions`) with structured citations.
5. Portable handoff packets for agent-to-agent transfer.
6. Derived active-state control views over append-only history.
7. Fail-open: a memory-subsystem failure never blocks the caller.

## Claim grounding

Protocol v2 grounds claims without changing the storage engine:

- `evidence_refs` remain the backward-compatible ref-string layer.
- `evidence_items` add exact line or character anchors, excerpts, sha256
  checksums, and verification metadata.
- Brief citations and handoff claim rows point to exact spans while still
  exposing compatible ref lists.
- `"verified"` at ingest means **self-consistency** (sha256(excerpt) equals
  the stored checksum) or verification provenance. It does not mean "true":
  `check_drift` re-reads live files and is what proves the source still
  matches. See [SECURITY.md](SECURITY.md).

Grounding quality is observable through snapshot metrics: ref citation
coverage, span-grounded citation coverage, excerpt fidelity, handoff span
grounding rate, and `agent_evidence_stale_count_24h`.

## Thread identity

Memory is **workspace-scoped**, not session-scoped. A harness session id is a
fresh UUID per session; using it as the thread id yields one empty brief per
session. Resolution order:

```text
explicit --thread-id > EVIDENCESPINE_THREAD_ID > proj_<sha1(realpath(git toplevel))[:12]>
```

The project root is the enclosing git worktree when available, so hooks,
subdirectory tools and MCP calls for the same checkout agree on one thread.
Provider session/turn identity travels in `source_turn_id` /
`metadata.session_id`. `evidencespine.threads` owns this logic; harness
hooks, CLI, MCP tools and A2A all resolve through it (MCP/A2A callers that pass
an explicit thread id keep it).

## Brief window and staleness

- Briefs are bounded by `EVIDENCESPINE_BRIEF_LOOKBACK_HOURS` (**default
  168 h**, raised from 6 h), not by the short dedupe window: old-but-important
  decisions stay eligible while recency ranking bounds relevance.
- `metadata.stale` is true when there are no events inside the lookback
  window — including when the store is non-empty but every row is older than
  the window — and when the newest event in the window is older than
  `max(300 s, dedupe window)`, so a silent/aged memory is never presented as
  fresh.
- Supersede-on-fix: a newer verified claim supersedes an asserted twin with the
  same case-insensitive claim text in the same thread; resolved items stop
  surfacing as locked decisions.
- Stale evidence is excluded from `"verified"` claims: it appears as
  `STALE EVIDENCE` risks, in the `stale_claims` view, and in handoffs as
  asserted rows with `evidence_stale: true`.

## Handoff provenance

Handoff claim and contradiction rows carry additive provenance:

- `verification` — the normalized `{method, reference, verified_at,
  verified_by}` record when the claim was verified;
- `evidence_stale` — true when the row's evidence has drifted.

Emit reads these from brief citations; import preserves them (and refuses to
re-import a stale row as verified). Packet JSON also carries a canonical
`checksum` and a deterministic `packet_id`, plus
`source_snapshot`/`metadata` counts so a receiver can audit the state it
inherited.

## Drift check

`check_drift` re-reads each evidence item's live source at its anchor and
recomputes the checksum. Verdicts are `changed`, `missing`, and
`unverifiable`; `unverifiable` roughly means "cannot prove either way"
(unsupported/ambiguous anchor, out-of-range character span, or a source id
outside the allowed root) and **never** clears a stale flag or promotes a fact
to verified. Server-side drift is confined: `allow_absolute_source_id=False`,
relative ids realpath-confined under `EVIDENCESPINE_SOURCE_ROOT`. The CLI
(trusted shell) may use absolute paths.

## Agent-state control layer

Optional `state_context` travels on events, facts, brief citations and
handoff rows, making ownership, freshness and local-vs-global state explicit.
The derived views are:

- `active_scopes`
- `my_work`
- `open_gates`
- `stale_claims` (including stale-evidence rows)
- `contradictions`

**Thread-scoped:** a view is resolved for one thread at a time (callers pass
`thread_id`; the CLI/brief/handoff surfaces use the workspace thread), and
the live row is derived per (thread, scope) rather than globally. Views are
computed from append-only history — no secondary index/database — so the
architecture is unchanged.

## Data flow

```text
intent -> decision -> action -> outcome -> reflection
```

Each event can generate one or more fact candidates. Structured evidence items
and optional `state_context` propagate events -> facts -> briefs -> handoffs.
Derived facts inherit the source event's `ts_utc` so control-view recency and
the brief lookback see when the claim actually happened.

## Adapter pipeline

```text
messages[] / state -> TranscriptAdapter normalization -> ingest_event -> fact extraction -> brief / handoff
```

Framework wrappers (`LangGraphAdapter`, `AutoGenAdapter`) are thin
schema-level layers over the transcript adapter, keeping the package
dependency-free. Ingest results distinguish `ok` / `partial` / `failed` /
`disabled` and report `deduped` separately from newly persisted events.

## Storage

Default base dir `.evidencespine/`:

- `evidencespine.db` — SQLite (default since 0.5.0), WAL mode, indexed
  `events`/`facts`/`dedup_hashes` tables. The `dedup_hashes` table is
  **authoritative** for dedupe.
- `events.jsonl` / `facts.jsonl` — legacy JSONL storage (opt-in via
  `storage_format=jsonl`). Legacy stores auto-migrate to SQLite **once**:
  only into an empty target, and a persistent `state["jsonl_migration"]`
  marker prevents pruned rows from being resurrected on reopen.
  `EVIDENCESPINE_REIMPORT_JSONL=1` (or `reimport_jsonl`) is the explicit
  reimport escape hatch and clears legacy dedupe hashes first.
- `state.json` — crash-recovery snapshot of counters and the migration
  marker. **In-memory counters are authoritative** and `state.json` is
  written at most once per second
  (`EVIDENCESPINE_STATE_FLUSH_INTERVAL_SEC`); fail-open/prune/close/flush
  force a save. A crash can lose at most about one second of counters — rows
  are already committed in SQLite.
- `briefs/*.json` — bounded brief artifacts (random UUID filenames; thread
  and role live in the JSON).
- `handoffs/*.json` — handoff packet artifacts (random UUID filenames).

## Performance

The 0.6.0 storage hardening removed a per-row `state.json` rewrite+fsync that
made ingest collapse as the hash ring grew. Measured in-process on one
machine (4000-event hot ingest; single-machine numbers, not a cross-host
claim):

| Path | Before | After |
|---|---:|---:|
| `runtime.ingest_event` while the store fills to 4,000 events | 516 -> **34.8** events/s (collapse) | **~5,870** events/s (flat) |
| store-level ingest (`AgentMemoryStore.ingest_event`) | 1,135 -> 68.6 events/s | ~18,200 -> ~20,900 events/s |

The mechanism: hot append paths now update in-memory counters and persist
`state.json` at most once per second, and SQLite dedupe reads the
`dedup_hashes` table instead of depending on the loaded in-memory ring. See
[BENCHMARKS.md](BENCHMARKS.md) for the apples-to-apples workload numbers.

## Harness delivery layer

`evidencespine harness install --harness <target>` writes per-client delivery:

- Claude Code — skills-directory plugin (hooks, `.mcp.json`, skill) plus
  `.claude/rules/` policy.
- opencode — V2 plugin (`context` + `compaction` recall, tool capture,
  lifecycle events).
- Cursor — MCP config, rule, permissions, hooks.
- VS Code/Copilot — portable `.mcp.json` (`vscode` target is the deprecated
  `.vscode/mcp.json` shape).
- Codex — `.codex/config.toml` (stdio/HTTP) + `.codex/hooks.json`.
- DeepSeek Harness — machine-wide `$DSH_HOME/cordis.patch.yml` client row
  with absolute paths.
- generic — canonical `evidencespine.mcp.json` plus merge-preserving
  projections; `instructions`/`agents-md` — managed `AGENTS.md` block;
  `git` — post-commit/post-merge evidence hooks.
- `harness print-config` prints snippets for 20+ clients without writing;
  `harness doctor` validates installed files; `harness debug` checks the
  store, executable and MCP extra.

**Every delivery channel fails open**: a missing/broken side-car produces a
one-line fail-open notice or an empty injection, never a blocked session, and
fail-open events are counted in the snapshot.

## Extensibility hooks

`RuntimeHooks` supports `on_event`, `on_brief`, `on_handoff`,
`contradiction_pass` and `reconcile_state` so an orchestrator can customize
behavior without hardcoding adapters into the package. Framework adapters:
`TranscriptAdapter`, `LangGraphAdapter`, `AutoGenAdapter`.

## Reliability model

- Dedupe by redacted event hash + window (SQLite table authoritative; JSONL
  ring legacy). Redaction runs before hashing so the hash binds what is
  persisted.
- Sensitive-value redaction on persistence (see [SECURITY.md](SECURITY.md)).
- Fail-open fallback brief/handoff when operations fail; fail-open counters in
  `state.json` and the snapshot.
- JSON artifacts are deterministic, checksummed and auditable; artifact
  filenames are random UUIDs written atomically, so caller-controlled ids can
  never traverse out of the configured directories.
- Backups/atomic writes in installers: invalid user files are never
  overwritten, hook groups are appended not replaced.

Protocol v2 compatibility is unchanged: mixed refs-only and span-grounded
artifacts remain readable, control views are derived rather than stored, and
no runtime dependency was added.
