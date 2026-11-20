# Changelog

All notable changes to this project will be documented in this file.

## [0.6.0] - 2026-11-20

Harness-compatibility release: the same evidence-bound memory now reaches the
major coding harnesses — verified on Claude Code, opencode, Codex, and DeepSeek
Harness, with documented configurations for Cursor, VS Code/Copilot and other
MCP clients (per-client evidence grades in `docs/HARNESS_MATRIX.md`). Delivery
paths are non-destructive and the core store/policy is hardened. This release
also fixes several correctness, security and UX defects found in an audit of
0.5.0.

### Added

- Harness delivery layer: one registry (`KNOWN_HARNESSES`) and
  idempotent, non-destructive installers for `claude-code`, `opencode`,
  `cursor`, `vscode`, `vscode-copilot`, `codex`, `dsh`, `generic`,
  `instructions`, `git`, `agents-md`, and `all`.
- `harness print-config`: copy-paste MCP config for 20+ clients
  (amp, antigravity, claude-code, claude-desktop, cline, codex, continue,
  cursor, dsh, gemini, generic, jetbrains, junie, opencode, roo, vscode,
  vscode-copilot, warp, windsurf, zed), stdio and streamable-HTTP variants,
  target paths, verification command, and an evidence-confidence grade. It
  never writes files.
- `harness doctor`: validates installed JSON/TOML, the opencode V2 plugin,
  git hooks, executable availability, and the store; `harness debug` reports
  store, executable, MCP-extra, grounding, harness files, and version.
- DeepSeek Harness (DSH) delivery: machine-wide
  `$DSH_HOME/cordis.patch.yml` `@deepseek-ai/dsh-mcp-client` row with
  absolute `command`/`args`/`cwd`, managed begin/end markers, atomic 0600
  writes and timestamped backups, `dsh --dump-config` composition
  verification, `profile:<name>` manager scope, non-destructive uninstall,
  and `.dsh/skills` skill materialization.
- Generic MCP delivery: canonical `evidencespine.mcp.json` plus
  merge-preserving project projections (`.mcp.json`, `.cursor/mcp.json`,
  `.agents/mcp_config.json`, `.agents/.mcp.json`,
  `.continue/mcpServers/evidencespine.json`, `.junie/mcp/mcp.json`,
  `.roo/mcp.json`), plus `.agents/skills` skill delivery and uninstall.
- Skills: canonical `skills/evidencespine/SKILL.md` rendered from the usage
  guide, materialized as relative symlinks (copy fallback) into
  `.dsh/skills/evidencespine/` and `.agents/skills/evidencespine/`;
  idempotent, never clobbers user-modified copies, supports `--dry-run`.
- `instructions` harness target (and `--with-agents-md`): merges a
  marker-delimited managed guide block into `AGENTS.md` while preserving user
  text; `agents-md` remains as a compatible target.
- MCP delivery hardening: condensed `mcp_instructions_markdown()` primer
  (<= 3800 characters) for `initialize.instructions`, constant-time Bearer
  comparison, and refusal of unauthenticated non-loopback streamable-HTTP
  binds. The primer makes every MCP client aware of the tools with zero
  install-time files.
- A2A delivery hardening: optional bearer auth via
  `EVIDENCESPINE_A2A_AUTH_TOKEN` (constant-time compare), public agent card,
  and refusal of unauthenticated non-loopback binds.
- Project-scoped thread identity: `proj_<sha1(realpath(git toplevel))[:12]>`
  resolved by CLI, MCP tool/resource, and harness-hook surfaces;
  `EVIDENCESPINE_THREAD_ID` and explicit `--thread-id` override; provider
  session ids now travel as `source_turn_id`/metadata instead of fragmenting
  memory into one thread per session.
- `append_fact`: direct fact rows now share the same verified-requires-span
  policy as event ingest.
- Drift-checker: character anchors, `locator` anchors, and an explicit
  `unverifiable` verdict (unsupported/ambiguous anchor, out-of-range span,
  confined absolute/escaping source id); MCP drift is confined server-side.
- Handoff provenance: claim and unresolved-contradiction rows carry
  `verification` (method + reference) and `evidence_stale`; imports report
  `failed_ingests` with a bounded error list and count dedupe separately.
- Metrics: `agent_memory_fail_open_events_24h` (24h capped ring) alongside the
  cumulative total, `brief_artifacts_unreadable`/
  `handoff_artifacts_unreadable`, and evidence-stale counts.
- Adapters: content-part lists, plain strings, tuples, and objects with
  `.messages` are normalized; results report
  `status: ok|partial|failed|disabled` plus `ingested`, `deduped`,
  `disabled`, `failed`, and `new_events`.
- Benchmark harness: `--seeds 42,43,44` multi-seed runs with mean/p50/p95
  aggregates; the verified/contradiction probes now ingest real checksummed
  evidence items instead of bare claims.
- Docs consistency tests that execute README promises: the quickstart
  evidence checksum must match sha256(excerpt), README doc links must exist,
  and `docs/HARNESS_MATRIX.md` must cover every registered harness.
- New core/security/harness regression tests for brief lookback and stale
  semantics, drift anchors/confinement, handoff provenance round trips,
  append_fact policy, state dedupe/migration, A2A/MCP auth, adapter shapes,
  thread identity, DSH/generic/skills installers, and benchmark metrics.

- Repository `SECURITY.md` (private vulnerability reporting and scope) and
  `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1) for GitHub community
  standards.

### Changed

- Default brief lookback is now `EVIDENCESPINE_BRIEF_LOOKBACK_HOURS` (168h)
  instead of a dedupe-window multiple, so week-old decisions remain eligible;
  an empty lookback window sets `metadata.stale: true` instead of pretending
  the thread is fresh.
- `state.json` is written at most once per second
  (`EVIDENCESPINE_STATE_FLUSH_INTERVAL_SEC`): in-memory counters are
  authoritative and the file is a crash-recovery snapshot (<= 1s counter loss,
  force-flushed on fail-open/prune/close). Event/fact rows remain durable on
  write.
- SQLite duplicate detection is authoritative via the `dedup_hashes` table;
  the in-memory event ring is kept only for the JSONL backend.
- Legacy JSONL auto-migration now runs once into an empty SQLite target and
  records `state["jsonl_migration"]`, so pruned rows are not resurrected on
  reopen; `EVIDENCESPINE_REIMPORT_JSONL=1` is the explicit escape hatch.
- `event_hash` is computed after redaction, so the stored hash binds the
  redacted payload; derived facts inherit the source event `ts_utc`; fact ids
  hash the case-insensitive claim.
- Control views group rows per `(thread_id, scope_id)` and count/report per
  thread; view names accept hyphen aliases and unknown views return a
  structured `invalid` payload instead of unfiltered rows.
- Brief claim lines are capped at `CITATION_LINE_MAX` (1024) everywhere
  (in-memory, on-disk, metrics), and token estimation counts CJK text
  conservatively; trimming keeps scanning cheaper lower rows instead of
  abandoning a section.
- Verified-facts metric excludes superseded twins; stale evidence is excluded
  from `recent_verified_facts` and emitted as asserted + `evidence_stale`.
- CLI process status now reflects payload status: `0` ok/deduped, `2`
  invalid/missing/ungroundable/mismatch, `3` fail-open/partial/disabled.
- `evidencespine chat` always has a hard deadline (derived wall-clock backstop
  when `--minutes` is absent), stops after repeated backend failures, never
  busy-spins without a facilitator backend, and reports `stopped_by` and
  `backend_failures`.
- `harness install --harness all` now also runs the DSH/generic lazy installers
  and best-effort git hooks; `harness install` accepts `--mcp-url` for HTTP
  MCP registration.
- Claude Code delivery moved to a skills-directory plugin at
  `.claude/skills/evidencespine/` (the previous project-plugin location was not
  auto-loaded); settings hooks are opt-in because plugin and settings copies
  both run.
- Codex delivery merges TOML safely (no duplicate `[features]` table), merges
  `hooks.json` event groups instead of replacing them, and supports HTTP MCP
  via `--mcp-url` + bearer env var.
- Opencode delivery was rewritten for the V2 plugin API: `ctx.session.hook`
  for context and compaction, `ctx.tool.hook("execute.after")` capture,
  `ctx.mcp.transform` registration, `ctx.event.subscribe` lifecycle, and
  fail-open empty recall (never a poisoned cached brief). `session.idle` is
  treated as a per-turn busy->idle signal (not session end), and the plugin
  passes session ids as metadata while the CLI resolves the workspace thread.
- Git hooks are installed non-destructively (sentinel + backup + chained
  invocation), worktree-aware, and use individually quoted argv tokens so
  `python3 -m evidencespine.cli`-style executables and spaced paths both run.
- `benchmarks/results/latest.json` workflow is joined by published multi-seed
  apples-to-apples artifacts; benchmark claims now list skipped external
  runners honestly.

### Fixed

- Ingest throughput collapse in long-lived runtimes: state persistence is
  debounced (see Changed), removing the per-row `state.json` rewrite/fsync that
  drove runtime ingest from 516 to 34.8 events/s as the store filled; the
  runtime now holds ~5,870 events/s at 4,000 events.
- Briefs missed older-than-6h history; stale detection ignored an empty
  lookback window; superseded twins inflated the verified-fact metric.
- `append_fact` could mint ungrounded `verified` facts, and imported stale
  evidence could re-import as verified.
- Drift-check ignored character anchors and locator-only items, could read
  arbitrary host files through MCP-supplied roots, and silently treated
  unsupported anchors as clean.
- MCP `//view/{view}` returned unfiltered rows for hyphenated/unknown views;
  MCP `handoff_receive` imported into the packet thread but rendered another
  thread's brief and re-imported on every render.
- Harness hooks imported provider control fields (`trigger`, `reason`) as
  session summaries; invalid boolean env values silently flipped fail-open or
  disabled memory.
- CLI: `chat` wrote a `./None/` store (stringified `None` base dir);
  `ingest --ground-ref` aborted for more than five refs or duplicate refs;
  `migrate` reported `mismatch` while exiting 0 and `--no-verify` did not skip
  verification; `migrate` silently collapsed duplicate source rows; invalid
  harness installs/hook SHAs/missing facts exited 0.
- Chat: `--llm echo` replies used the wrong role name, echo/scripted modes
  spawned an unnecessary opencode backend, and a failing backend could hang
  the run.
- Adapters reported `status: ok` for deduped/disabled/replayed rows, garbled
  content-part lists into Python reprs, silently dropped tuple messages, and
  no-op'd unrecognized state objects.
- Claude Code delivery failed `claude plugin validate`: bare event maps,
  wrong SessionEnd/Stop event, plugin-root hook pointer, shell-quoted commands,
  and `hookSpecificOutput` used for PreCompact (plain-text stdout is the
  supported channel) were corrected; SessionStart/compaction hooks now share
  the workspace thread instead of fragmenting it.
- Codex install could create a duplicate `[features]` table (invalid TOML)
  and replaced hook groups instead of appending.
- Python 3.10 (no `tomllib`) could not validate an existing `config.toml`, so
  the Codex installer would append to a possibly-malformed file; it now fails
  safe — the file is left untouched with a backup and a `partial` status, and
  the installer must run on Python 3.11+ (or the table merged manually) to
  update an existing config.
- Git evidence hooks accepted unknown commits and pure-deletion hunks as "ok"
  evidence, and hook invocation quoting broke multi-token executables.
- Cursor hook installation appended duplicate `sessionStart`/`preCompact`/`stop` groups on every re-install (flat Cursor hook groups were not recognised by the merge); installs are now byte-stable and idempotent.
- Short-lived CLI/hook processes exited before the debounced state flush, so `facts_total`/`events_total` could stay stale in `state.json`; the store now flushes and closes at interpreter exit.
- `evidencespine migrate` (explicit JSONL->SQLite) did not write the one-time migration marker, so a later prune that emptied SQLite resurrected the legacy rows on the next open; explicit migration now records `state["jsonl_migration"].source = "explicit_migrate"`.
- Benchmark verified/contradiction probes were invalid under the
  verified-requires-span policy (they ingested bare claims), so the previously
  published 1.000 probe hit was not meaningful; published probe numbers now
  come from real checksummed evidence.
- `harness session-start --json`/`precompact --json` reported `ok` even when
  the handler failed open; the payload now carries `fail_open`.

### Security

- MCP streamable-HTTP: static Bearer auth with `hmac.compare_digest`;
  unauthenticated non-loopback binds are refused unless
  `EVIDENCESPINE_MCP_AUTH_TOKEN` is set (or the explicit
  `EVIDENCESPINE_MCP_ALLOW_UNAUTHENTICATED=1` dev opt-in).
- A2A Agent Cards now advertise `supportedInterfaces` (`url` + `JSONRPC`) so a2a-sdk 1.x clients can resolve a transport; the public card still carries the memory skills.
- A2A `prune` action previews by default (`dry_run=true`), matching MCP.
- A2A: optional static Bearer auth via `EVIDENCESPINE_A2A_AUTH_TOKEN`;
  unauthenticated non-loopback binds are refused, with the explicit
  `EVIDENCESPINE_A2A_ALLOW_UNAUTHENTICATED=1` dev opt-in and a warning.
- Network-facing `check_drift` no longer accepts a caller-supplied root and
  never opens absolute/escaping source ids: they are reported `unverifiable`,
  closing a file-read/checksum oracle. Legacy runtimes without confinement are
  refused (`runtime_missing_evidence_confinement`) instead of downgraded.
- MCP `ground`, ingest `ground_refs`, and handoff imports are confined to the
  server source root; absolute paths, traversal, and symlink escapes are
  rejected.
- MCP `prune` now defaults to `dry_run=true`, so an accidental call previews
  instead of deleting.
- Invalid boolean env values keep their safe defaults instead of disabling
  fail-open/auth on a typo.

### Behavior changes & migration notes

- Thread identity changed from per-session/provider ids to workspace-scoped
  `proj_<hash>`. Existing stores keep old thread ids; to continue a thread,
  set `EVIDENCESPINE_THREAD_ID=<old id>` (hooks/CLI) or pass `--thread-id`.
- Brief lookback is 168h by default; set `EVIDENCESPINE_BRIEF_LOOKBACK_HOURS`
  to preserve tighter windows. Empty-in-window memory now reports
  `metadata.stale: true`.
- MCP `prune` requires `dry_run=false` to delete. JSONL stores auto-migrate
  once and are not reimported after prune; use
  `EVIDENCESPINE_REIMPORT_JSONL=1` deliberately.
- `event_hash` values change for events whose payload was redacted (hash is
  computed after redaction); unredacted event hashes are stable, so duplicate
  windows across an upgrade may see one re-inserted row for redacted claims.
- CLI exit codes are now meaningful (`2` invalid/missing, `3` fail-open) and
  `migrate` exits 1 on verification mismatch.
- `harness install --harness vscode` remains the deprecated
  `.vscode/mcp.json` Local-harness shape; new Copilot configs should use
  `--harness vscode-copilot` (portable `.mcp.json`).
- Claude Code installs move from `.claude-plugin/` to
  `.claude/skills/evidencespine/`; rerun the installer and remove the old
  project plugin. Old settings hooks keep working but plugin hooks are the
  default (settings hooks only with `--with-settings-hooks`).
- Codex hook groups are merged, not replaced; existing user hooks are
  preserved and EvidenceSpine groups are appended once.
- Benchmark methodology changed (real probe evidence, 3 seeds); 0.6.0 numbers
  are not directly comparable to earlier published snapshots.

## [Unreleased]

No unreleased changes — 0.6.0 below was cut from this work.

## [0.5.0] - 2026-08-04

### Added
- Added MCP server (`evidencespine mcp`): 8 tools (`ingest_event`, `build_brief`, `query_view`, `emit_handoff`, `import_handoff`, `reconcile`, `snapshot`, `prune`), 4 resources, and 3 prompts over the `[mcp]` extra (mcp SDK 2.0).
- Added the harness delivery layer (`evidencespine harness`): fail-open session hooks for Claude Code (SessionStart/Stop/PreCompact), an opencode plugin (system-prompt brief injection, compaction retention, session-stop handoff), and a Cursor MCP config; `harness install`/`harness debug` commands.
- Added A2A protocol support (`evidencespine a2a`): Agent Card with `memory.read`/`memory.write`/`memory.handoff`/`memory.health` skills, an executor mapping A2A messages to memory ops, and a FastAPI server under the `[a2a]` extra (a2a-sdk 1.1).
- Added BM25 lexical scoring with pluggable vector backends, plus the `[embeddings]` extra (`fastembed`) with auto/hashing fallback and `EVIDENCESPINE_EMBEDDING_BACKEND`/`_MODEL` env knobs.
- Added TTL archival: `prune` on the store/runtime with dry-run, thread scoping, and a `last_prune` state record.
- Added `AsyncAgentMemoryRuntime` thread-offloaded async wrappers for all core memory operations.
- Added SQLite as the default storage backend with auto-migration from JSONL, the `evidencespine migrate` command, and a JSONL legacy runner in the benchmark harness.
- Added shared markdown renderers for briefs, control views, state, and snapshots.
- Added `[all]` extra wiring all optional integrations.
- Added test coverage for MCP, harness hooks/installers, A2A server flows, prune/TTL, embedding fallbacks, and async wrappers (94 tests).

### Changed
- Default storage format is now SQLite (WAL); JSONL remains opt-in via `--storage-format jsonl`.
- Retrieval now uses BM25 relevance (fallback jaccard) and supports hybrid/vector modes.
- Bumped package version to `0.5.0`.

### Notes
- The core package remains dependency-free; integrations are optional extras (`[mcp]`, `[embeddings]`, `[a2a]`, `[all]`).
- Existing JSONL stores auto-migrate to SQLite on first open; `evidencespine migrate` performs explicit migrations.

## [0.4.0] - 2026-03-18

### Added
- Added agent-state `state_context` support across events, facts, brief citations, handoff rows, and transcript normalization.
- Added derived control views for `active_scopes`, `my_work`, `open_gates`, `stale_claims`, and `contradictions`.
- Added a generic `reconcile_state` runtime hook plus `AgentMemoryRuntime.reconcile(...)`.
- Added CLI support for structured control-state ingest and new `view` / `reconcile` commands.
- Added runtime, protocol, CLI, and adapter tests for freshness, ownership, conflicts, and active-scope derivation.

### Changed
- Extended Protocol v2 without changing `schema_version`, storage layout, or the dependency-free runtime.
- Propagated control-state metadata through brief generation and handoff import/export.
- Added snapshot metrics for active-scope count, open gates, owner coverage, freshness coverage, stale rate, and conflict rate.
- Updated docs and examples to position EvidenceSpine explicitly as a side-car rather than the final runtime authority.

### Notes
- Existing refs-only and free-form callers remain supported.
- No storage migration or framework-specific truth adapter was introduced.

## [0.3.0] - 2026-03-07

### Added
- Added Protocol v2 structured evidence spans via `evidence_items` and public helper types `EvidenceItem` and `ClaimCitation`.
- Added CLI ingest support for structured evidence via `--evidence-item-json` and `--evidence-item-file`.
- Added brief and handoff span-grounding metrics, including ref coverage, span coverage, excerpt fidelity, and handoff span grounding.
- Added protocol, runtime, CLI, adapter, and benchmark result-shape tests for the new evidence span flow.

### Changed
- Upgraded brief `citations` output to structured citation bundles while preserving `citation_refs` as a legacy ref-list alias.
- Propagated structured evidence through runtime ingest, fact derivation, brief assembly, handoff emission, handoff import, and transcript adapter ingestion.
- Updated benchmark harnesses to generate synthetic grounded evidence spans and report grounding-quality metrics.
- Updated docs and examples to describe Protocol v2 claim-to-span grounding.

### Notes
- The package remains dependency-free at runtime.
- Existing refs-only callers remain supported; `evidence_refs` is still accepted and emitted alongside `evidence_items`.

## [0.2.0] - 2026-03-04

### Added
- Added `TranscriptAdapter` as the new generic transcript-first integration path for plain `messages[]` and transcript-like payloads.
- Added public adapter contract types: `NormalizedTranscriptMessage`, `TranscriptAdapterConfig`, and `AdapterIngestResult`.
- Added replay fixtures and replay harness examples for realistic brief and handoff validation.
- Added adapter-focused documentation in `docs/ADAPTERS.md`.

### Changed
- Refactored `LangGraphAdapter` and `AutoGenAdapter` to delegate to the shared transcript adapter while preserving backward compatibility.
- Updated integration and architecture docs to recommend transcript-first integration as the default path.
- Expanded adapter coverage with normalization, replay, and compatibility tests.
- Bumped package version to `0.2.0`.

### Notes
- The package remains dependency-free.
- No protocol, storage, or CLI breaking changes were introduced in this release.
