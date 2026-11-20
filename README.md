<p align="center">
  <img src="docs/evidencespine_logo.png" alt="EvidenceSpine logo" width="320" />
</p>

# EvidenceSpine

EvidenceSpine is an agent-agnostic memory and handoff quality side-car for multi-agent workflows.

It is not a workflow engine or final runtime authority. It gives agents a bounded, evidence-backed state layer they can consult, query, and hand off without stuffing raw transcripts into prompts.

## Why this exists

Most agent-memory systems store context. EvidenceSpine adds strict claim quality controls:

- Every claim can carry citations, and citations can carry exact file spans.
- Claims are bound to checksummed excerpts, so "verified" is checkable in principle.
- Contradictions are explicit rows, not silent overwrites.
- Handoffs are structured, portable JSON with provenance.

The spine is a map — the repository is ground truth. A fact can be internally
consistent and still be wrong about the live tree; `drift-check` is the step
that compares evidence to the current files.

## Features

- Evidence-bound event and fact memory (`asserted|verified|contradicted|superseded`) with Protocol v2 evidence spans.
- Verified-requires-span policy: `verified` facts must carry a checksummed excerpt or verification provenance, else they are stored as `asserted`.
- Span grounding first-class: `evidencespine ground file#line` turns refs into checksummed evidence items.
- Drift-checker (`drift-check` / `check_drift`): re-verifies checksummed evidence against live files, flags stale facts in views/briefs/metrics, and never re-emits stale evidence as verified. MCP drift and grounding are confined to the server source root.
- Verification provenance (`verify` / `verify_fact`): records *how* a claim was verified (test/gate/tool/manual) plus a reference, and carries it through handoffs.
- Bounded context briefs: 1,800-token default budget, sectioned (goal, locked decisions, verified facts, risks, open items, next actions), with a 168h lookback and explicit staleness signal.
- Project-scoped thread identity: `proj_<sha1(workspace)>[:12]` so every session, hook, and subdirectory shares one brief; `EVIDENCESPINE_THREAD_ID` overrides.
- Cross-agent handoff packets with checksum, citations, verification provenance, `evidence_stale` flags, and stale-evidence exclusion.
- Agent-state freshness, ownership, and lease semantics via `state_context`, with derived `active_scopes`, `my_work`, `open_gates`, `stale_claims`, and `contradictions` views (validated names; unknown views fail with a structured error).
- Supersede-on-fix: verified claims auto-supersede asserted twins; resolved items stop surfacing as locked decisions.
- Harness delivery layer: `evidencespine harness install --harness <target>` for Claude Code, opencode, Codex, and DeepSeek Harness (DSH) — verified in this release — plus documented installs for Cursor, VS Code/Copilot, generic MCP clients, instructions/AGENTS.md and git hooks. `harness print-config` covers 20+ MCP clients and `harness doctor` validates what was written. Per-client evidence grades are in the client matrix.
- Skills delivery: one canonical `skills/evidencespine/SKILL.md` materialized into `.dsh/skills` and `.agents/skills` (symlink-first; reinstall refreshes generated copies and leaves user-modified, marker-less files alone).
- MCP server (`evidencespine mcp`) over stdio or streamable-HTTP with static bearer auth; non-loopback binds are refused without auth.
- A2A protocol server (`evidencespine a2a`) with bearer auth, a public agent card, and a documented op subset (no `ground`/`verify`/`check_drift`).
- Fail-open behavior everywhere: a broken store or hook returns a warning/fail-open brief, never a blocked caller.
- Hybrid retrieval (`lexical|hybrid|vector`) with pluggable vector backends; dependency-free core.
- Transcript-first adapters (plain `messages[]`, LangGraph, AutoGen) with honest `ok|partial|failed|disabled` ingest status and dedupe accounting.
- Realtime multi-agent debate chat (`evidencespine chat`): polling loops over the spine with reply caps, rolling windows, auto-summary, wall-clock backstop, and termination on backend failure.

## Install

Core package is dependency-free (Python 3.10+):

```bash
cd evidencespine
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional integrations: `pip install -e ".[mcp]"`, `".[a2a]"`, `".[embeddings]"`, or `".[all]"`.

From a source checkout without installing, prefix commands with
`PYTHONPATH=src python3 -m evidencespine` (or `... -m evidencespine.cli`); the
examples below use the installed `evidencespine` console script.

## Quick start

Run these from the repository root; the store lands in `.evidencespine/`
(override with `--base-dir` or `EVIDENCESPINE_BASE_DIR`). The workflow ingests a
verified decision with a checksummed excerpt, builds a brief, inspects state,
emits and imports a handoff, and reads the health snapshot.

1. Ingest a grounded decision:
```bash
evidencespine ingest \
  --thread-id demo \
  --event-type decision \
  --source-agent-id implementer \
  --source-turn-id t1 \
  --claim "Use additive patch set" \
  --fact-state verified \
  --evidence-ref "reports/decision_log.md" \
  --evidence-item-json '{"source_id":"reports/decision_log.md","line_start":12,"line_end":14,"excerpt":"Use additive patch set","checksum":"sha256:8d335d8ea27fd104a59d597ab7b4ee403cd4fe9e6958a89632e3635d2360b22c"}' \
  --scope-id demo-scope \
  --state-kind agent_local_work \
  --status active \
  --owner-agent-id implementer \
  --json
```

2. Build a context brief:
```bash
evidencespine brief --thread-id demo --query "current status and next actions" --json
```

3. Inspect the derived active-scopes view:
```bash
evidencespine view active-scopes --thread-id demo --json
```

4. Emit a handoff packet:
```bash
evidencespine handoff --role auditor --thread-id demo --scope "verify latest claims" --output /tmp/demo_handoff.json
```

5. Import a handoff packet:
```bash
evidencespine handoff --import /tmp/demo_handoff.json --json
```

6. Read the health snapshot:
```bash
evidencespine snapshot --json
```

`--thread-id` is optional elsewhere: CLI/MCP/hook commands resolve the
workspace thread (`proj_<hash>`) when it is omitted. Commands print JSON with
`--json` and use exit codes `0` ok, `2` invalid/missing, `3` fail-open/partial.

## Agent usage

The canonical, decision-oriented playbook is [`AGENTS.md`](AGENTS.md) (also the
`evidencespine://guide` MCP resource and the MCP `initialize.instructions`
primer). In short:

1. **Start** with `build_brief`; treat `locked_decisions` as binding and
   `required_validations` as first work.
2. **As you work**, ingest decisions/outcomes/gates with grounded evidence.
3. **Before claiming verified/fixed**, ground the span, attach it, and set
   `fact_state=verified` (or record provenance with `verify_fact`).
4. **After edits**, run `drift-check` so stale evidence is flagged instead of
   believed.
5. **On handoff or session end**, emit a packet; the successor imports it before
   editing shared state. Contradictions are recorded, not ignored.

```bash
evidencespine harness install --harness all --target-dir .
```

installs every delivery layer for the current repository (hooks/plugins for
Claude Code, opencode, Cursor, VS Code/Copilot, Codex, DSH, generic MCP
projections, AGENTS.md instructions, and git hooks) and is idempotent.

## Supported harnesses

`evidencespine harness install --harness <target> --target-dir .` writes one
delivery layer, non-destructively and idempotently; every target supports
`--dry-run` and prints JSON.

| target | writes | install |
|---|---|---|
| `claude-code` | `.claude/skills/evidencespine/` skills-dir plugin (manifest, `hooks/hooks.json`, `.mcp.json`, `SKILL.md`) + `.claude/rules/evidencespine.md`. Settings hooks are opt-in (`--with-settings-hooks`). | `evidencespine harness install --harness claude-code` |
| `opencode` | `.opencode/plugins/evidencespine.ts` V2 plugin (MCP registration, context/compaction recall, tool-after capture, lifecycle events). | `--harness opencode` |
| `cursor` | `.cursor/mcp.json`, `.cursor/rules/evidencespine.mdc`, `.cursor/permissions.json`, `.cursor/hooks.json` (merge-preserving). | `--harness cursor` |
| `vscode` | legacy `.vscode/mcp.json` Local-harness shape (`servers`), deprecated; kept for compatibility. | `--harness vscode` |
| `vscode-copilot` | portable workspace `.mcp.json` (key `mcpServers`); global writes `~/.copilot/mcp-config.json`. | `--harness vscode-copilot` |
| `codex` | `.codex/config.toml` `[mcp_servers.evidencespine]` (safe TOML merge; `--mcp-url` for HTTP) + `.codex/hooks.json` (SessionStart/SessionEnd/PreCompact). | `--harness codex` |
| `dsh` | machine-wide `$DSH_HOME/cordis.patch.yml` `@deepseek-ai/dsh-mcp-client` row with absolute paths, plus the skill in `.dsh/skills`. | `--harness dsh` |
| `generic` | canonical `evidencespine.mcp.json` plus project projections: `.mcp.json`, `.cursor/mcp.json`, `.agents/mcp_config.json`, `.agents/.mcp.json`, `.continue/mcpServers/evidencespine.json`, `.junie/mcp/mcp.json`, `.roo/mcp.json` (merge-preserving); `.agents/skills` skill. | `--harness generic` |
| `instructions` | managed, marker-delimited block merged into `AGENTS.md` (never clobbers user text). | `--harness instructions` (or `--with-agents-md`) |
| `agents-md` | managed block merged into `AGENTS.md` (legacy alias of `instructions`). | `--harness agents-md` |
| `git` | executable `post-commit`/`post-merge` hooks that ingest span-grounded commit evidence and test-result provenance. | `--harness git` |
| `all` | every target above (git hooks best-effort, DSH/generic lazy modules skipped if absent). | `--harness all` |

Harness-specific references: [`docs/CLAUDE_CODE.md`](docs/CLAUDE_CODE.md),
[`docs/CODEX.md`](docs/CODEX.md), [`docs/OPENCODE.md`](docs/OPENCODE.md),
[`docs/DSH.md`](docs/DSH.md).

## Portable setup (any MCP client)

Two portable routes cover every MCP-capable client:

1. **Print a config, never write it:** `evidencespine harness print-config --harness <client>`
   supports `all` plus amp, antigravity, claude-code, claude-desktop, cline,
   codex, continue, cursor, dsh, gemini, generic, jetbrains, junie, opencode,
   roo, vscode, vscode-copilot, warp, windsurf, zed. Add `--transport http` for
   the streamable-HTTP variant (sets a Bearer header from
   `EVIDENCESPINE_MCP_AUTH_TOKEN`). Paths, verification commands, and confidence
   grades are included; UI-managed clients (JetBrains) are marked print-only.
2. **Run the server yourself:** register
   `evidencespine mcp --base-dir <absolute path>` as a stdio MCP server (DSH and
   hook children need absolute paths because their working directory is not the
   workspace). The MCP `initialize.instructions` field carries the condensed
   agent primer automatically.

```bash
# stdio (default)
evidencespine mcp --base-dir /absolute/path/to/repo/.evidencespine

# streamable-HTTP: bearer auth is required; bind loopback unless you front it
export EVIDENCESPINE_MCP_AUTH_TOKEN=change-me
evidencespine mcp --transport streamable-http --host 127.0.0.1 --port 8000 --base-dir /absolute/path/to/repo/.evidencespine
```

Validate what is installed with `evidencespine harness doctor --json`;
`evidencespine harness debug --json` additionally checks the store,
executable, MCP extra, and grounding. The full per-client matrix (surfaces,
config paths, transports, auth support, uncertainty grades, migration from the
legacy VS Code shape) is [`docs/HARNESS_MATRIX.md`](docs/HARNESS_MATRIX.md);
DSH specifics (profiles, one-command install, verification, limits) are
[`docs/DSH.md`](docs/DSH.md).

## Skills

`skills/evidencespine/SKILL.md` is the canonical skill (YAML frontmatter +
this agent guide). `harness install --harness dsh` and `--harness generic`
materialize it into `.dsh/skills/evidencespine/SKILL.md` and
`.agents/skills/evidencespine/SKILL.md` as a relative symlink (copy fallback),
idempotently; generated copies are refreshed on reinstall while user-modified,
marker-less copies are preserved. DSH discovers project skills from both roots;
Cursor/Cline read `.agents/skills`.

## Hybrid retrieval (vector + lexical)

Default is BM25 lexical. Set via env:

```bash
export EVIDENCESPINE_RETRIEVAL_MODE=hybrid
export EVIDENCESPINE_RETRIEVAL_LEXICAL_WEIGHT=1.0
export EVIDENCESPINE_RETRIEVAL_VECTOR_WEIGHT=0.35
```

By default hybrid uses an internal hashing vector backend (dependency-free).
You can inject your own backend by implementing `score_texts(query, texts) -> scores`.

## Verified semantics (what the words mean)

- `verified` at ingest means **self-consistency**: the stored excerpt's SHA-256 matches the stored checksum, or verification provenance is present. It does **not** prove the live file still says that.
- `drift-check` re-reads each evidence span from the live tree and compares checksums. It is what proves the source still matches; it flags `changed`, `missing`, or `unverifiable` (unsupported anchor, out-of-range span, or — on the MCP server — a path outside the confined source root).
- Stale evidence is excluded from `recent_verified_facts`, surfaces as `STALE EVIDENCE` risks, appears in `stale_claims`, and is emitted in handoffs as `asserted` with `evidence_stale: true` — it never re-imports as verified.
- Contradictions are explicit: `contradicted` facts, `contradictions` view, brief risks, and `unresolved_contradictions` in handoffs.
- Fail-open responses are diagnostics, not facts: if the spine is unavailable the caller is told, not silently misled.

## Protocol v2 evidence spans and agent state

EvidenceSpine keeps Protocol v2 and extends it additively:

- `evidence_refs`: compatible string refs for existing integrations
- `evidence_items`: structured spans with `source_id`, `locator`, line or char anchors, optional excerpt, checksum, confidence, and verification state
- `citations`: structured brief claim citations with `primary_ref`, `evidence_refs`, `evidence_items`, and `span_grounded`
- `citation_refs`: legacy brief alias for refs-only consumers; claim lines are capped at `CITATION_LINE_MAX` (1024 chars) consistently in memory and on disk
- `state_context`: optional control-state envelope with scope, ownership, freshness, validation basis, and supersession semantics
- Handoff rows carry `verification` (method + reference) and `evidence_stale` for claims and unresolved contradictions; deriving facts preserves the source event timestamp and evidence flags.

New control-state views are derived from the append-only row log:
`active_scopes`, `my_work`, `open_gates`, `stale_claims`, `contradictions`.
Rows resolve per `(thread_id, scope_id)`, so the same scope name in two threads
is two rows, not one merged row.

## Transcript-first integration

```python
from evidencespine import AgentMemoryRuntime, EvidenceSpineSettings
from evidencespine.adapters import TranscriptAdapter, LangGraphAdapter, AutoGenAdapter

rt = AgentMemoryRuntime(config=EvidenceSpineSettings.from_env().to_runtime_config())
ta = TranscriptAdapter(rt)
lg = LangGraphAdapter(rt)
ag = AutoGenAdapter(rt)

ta.ingest_messages(
    [
        {
            "role": "assistant",
            "content": "Retry guard patch is under validation",
            "state_context": {
                "scope_id": "auth-timeout-fix",
                "state_kind": "agent_local_work",
                "status": "active",
                "owner_agent_id": "implementer",
            },
        }
    ]
)

print(rt.query_view("active_scopes", thread_id="thread_default").to_dict())
```

Adapters accept plain strings, content-part lists (LangChain/OpenAI shape),
tuples, and objects with `.messages`; unrecognized shapes return an explicit
invalid result instead of a silent empty ingest. Results report `status`
`ok|partial|failed|disabled` plus `ingested`, `deduped`, `disabled`,
`failed`, and `new_events`, so a replay is distinguishable from a capture.

See [`docs/ADAPTERS.md`](docs/ADAPTERS.md),
[`examples/multi_agent_handoff.py`](examples/multi_agent_handoff.py),
[`examples/reconcile_hook_demo.py`](examples/reconcile_hook_demo.py),
[`examples/claude_code_usage.py`](examples/claude_code_usage.py), and
[`examples/transcript_replay_harness.py`](examples/transcript_replay_harness.py).

## Storage, state, and performance

Default store in `.evidencespine/`:

- `evidencespine.db` — SQLite (WAL) events, facts, and the authoritative `dedup_hashes` table
- `events.jsonl` / `facts.jsonl` — legacy/opt-in JSONL backend (`--storage-format jsonl`)
- `state.json` — crash-recovery snapshot of counters and migration marker
- `briefs/*.json`, `handoffs/*.json` — bounded artifacts used by metrics
- `skill`/`instructions` artifacts live in each harness's own directory, not here

State and durability behavior in 0.6.0:

- **Debounced state:** hot append paths update in-memory counters and persist `state.json` at most once per second (`EVIDENCESPINE_STATE_FLUSH_INTERVAL_SEC`, force-flushed on fail-open/prune/close). A crash can lose at most ~1s of counters; rows are already committed.
- **SQLite dedupe is authoritative:** duplicate detection reads the `dedup_hashes` table, so it does not depend on how many events were loaded into the in-memory ring. JSONL keeps the legacy ring.
- **One-time JSONL migration:** legacy JSONL rows are auto-migrated only into an empty SQLite target, and a persistent `state["jsonl_migration"]` marker prevents pruned rows from being resurrected on reopen. `EVIDENCESPINE_REIMPORT_JSONL=1` is the explicit reimport escape hatch.
- **Brief window:** briefs consider 168h of history by default (`EVIDENCESPINE_BRIEF_LOOKBACK_HOURS`); memory that is empty in-window, or whose newest event predates `max(300 s, dedupe window)`, is reported as `metadata.stale: true` instead of pretending the thread is fresh.
- **Ingest no longer degrades:** earlier builds rewrote and fsynced `state.json` per row, collapsing throughput as the store filled. In this repo's harness, runtime ingest holds ~5,870 events/s as the store fills to 4,000 events (previously 516 → 34.8 events/s), and store-level ingest is ~18,200 → ~20,900 events/s (previously 1,135 → 68.6). Single-machine numbers; see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

Use `--base-dir` or `EVIDENCESPINE_BASE_DIR` to override the store location.
`prune` is preview-by-default over MCP (`dry_run=true`) and deletes only when
explicitly asked; rows without a parseable timestamp are always kept.

## Benchmarks

The benchmark harness and apples-to-apples comparison are reproducible:

```bash
cd evidencespine
PYTHONPATH=src python benchmarks/bench_evidencespine.py \
  --events 2000 --brief-queries 100 --handoffs 40 \
  --out benchmarks/results/latest.json

PYTHONPATH=src python benchmarks/apples_to_apples_compare.py \
  --events 300 --queries 20 --handoffs 8 --seeds 42,43,44 \
  --out-json benchmarks/results/apples_to_apples_multiseed_seeds_42_43_44.json \
  --out-md benchmarks/results/apples_to_apples_multiseed_seeds_42_43_44.md
```

Current published snapshot (300 events, 20 brief queries, 8 handoffs; mean of
seeds 42, 43, 44):

| runner | ingest events/s | brief qps | handoff qps | governance score |
|---|---:|---:|---:|---:|
| evidencespine (SQLite, lexical) | 1,590.5 | 51.8 | 46.9 | 1.000 |
| evidencespine (SQLite, hybrid) | 1,561.5 | 25.1 | 23.9 | 1.000 |
| evidencespine (JSONL, lexical) | 918.7 | 39.1 | 35.7 | 1.000 |
| baseline SQLite (no grounding) | 3,021.3 | 769.7 | 796.4 | 0.375 |
| mem0 / letta | skipped — not installed | — | — | — |

Caveats: numbers are single-machine, the baseline is a DIY SQLite store with no
grounding or citations by design (hence its 0.375 governance score), external
runners (mem0, letta) were skipped rather than faked, and the old verified-probe
result was invalid under the verified-requires-span policy — the probe now
ingests a real checksummed evidence item, so the 1.000 probe hit means
something. Full artifacts: the published snapshot
[`docs/benchmarks/apples_to_apples_2026-11-20.md`](docs/benchmarks/apples_to_apples_2026-11-20.md)
(+ `.json`); the multi-seed run is reproducible with the command above and
writes to the gitignored `benchmarks/results/` directory. See also
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and
[`docs/APPLE_TO_APPLE_COMPARISON.md`](docs/APPLE_TO_APPLE_COMPARISON.md).

## Limits and security

- Local-first defaults: MCP over stdio; streamable-HTTP and A2A bind `127.0.0.1`. HTTP MCP uses a static Bearer token (`EVIDENCESPINE_MCP_AUTH_TOKEN`, constant-time compare); A2A uses `EVIDENCESPINE_A2A_AUTH_TOKEN` and keeps its agent card public.
- Non-loopback binds are refused unless a token is configured, or the explicit dev escape hatches `EVIDENCESPINE_MCP_ALLOW_UNAUTHENTICATED=1` / `EVIDENCESPINE_A2A_ALLOW_UNAUTHENTICATED=1` are set. There is **no OAuth server** and **no TLS termination** in this project: terminate TLS at a gateway/ssh tunnel and authenticate every remote hop.
- Grounding confinement is server-side: MCP `ground`/`ingest ground_refs`/`check_drift` resolve only under `EVIDENCESPINE_SOURCE_ROOT`, reject absolute/escaping paths, and report `unverifiable` instead of reading arbitrary host files. Handoff imports are confined the same way. The CLI is trusted local tooling, not a network boundary.
- Redaction runs before persistence, hashing, and fact derivation, but grounded excerpts are stored verbatim because they are the evidence. Do not ground secrets.
- No encryption at rest: protect the store directory like any other agent scratch state.
- Fail-open: hooks and MCP calls degrade to warnings/briefs so a memory outage cannot block an agent; a fail-open response is never authoritative.
- Chat has reply caps, a confined rolling window, an auto-summary, and a hard wall-clock backstop; it exits nonzero on backend failure.
- DSH 0.2.2-rc.5 has no external hook system and no TUI profile: DSH delivery is MCP instructions + a static MCP row + an optional skill/AGENTS.md block. JetBrains AI is print-only (UI-managed config). Client paths marked "uncertain" in the matrix are unverified by design.
- Benchmark numbers are single-machine and this is a side-car, not a model: governance scores measure grounding/citation/probe behavior of the recorded memory, not agent intelligence.

## Documentation

- [`AGENTS.md`](AGENTS.md): canonical agent playbook (also served via MCP)
- [`docs/HARNESS_MATRIX.md`](docs/HARNESS_MATRIX.md): per-client MCP compatibility matrix
- [`docs/DSH.md`](docs/DSH.md): DeepSeek Harness integration
- [`docs/CLAUDE_CODE.md`](docs/CLAUDE_CODE.md): Claude Code delivery
- [`docs/CODEX.md`](docs/CODEX.md): Codex CLI/IDE delivery
- [`docs/OPENCODE.md`](docs/OPENCODE.md): opencode V2 plugin
- [`docs/INTEGRATION.md`](docs/INTEGRATION.md): integration guide
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): architecture and storage
- [`docs/PROTOCOL.md`](docs/PROTOCOL.md): Protocol v2 contract
- [`docs/ADAPTERS.md`](docs/ADAPTERS.md): transcript adapters
- [`docs/SECURITY.md`](docs/SECURITY.md): threat model and auth
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md): methodology and results
- [`docs/APPLE_TO_APPLE_COMPARISON.md`](docs/APPLE_TO_APPLE_COMPARISON.md): apples-to-apples comparison
- [`docs/DEBATE.md`](docs/DEBATE.md): realtime debate chat semantics
- [`examples/`](examples): runnable examples
- [`skills/evidencespine/SKILL.md`](skills/evidencespine/SKILL.md): canonical skill

## Release checklist

- [x] Installable package (`pyproject.toml`)
- [x] OSI-approved open-source license
- [x] CLI and examples
- [x] Replay-backed adapter examples
- [x] Adapter normalization contract
- [x] Harness delivery + client compatibility matrix
- [x] Documentation

## License

This project is distributed under the **Apache License 2.0**.

See [`LICENSE`](LICENSE) and [`DCO`](DCO).

## Contributing

Contributions are welcome. By submitting a contribution, you agree it is licensed under Apache-2.0 and certify origin via the Developer Certificate of Origin ([`DCO`](DCO)).
Use `Signed-off-by` in commits, for example:

```bash
git commit -s -m "feat: improve adapter replay docs"
```

## Security

See [`docs/SECURITY.md`](docs/SECURITY.md) for the threat model, the
redaction trade-off (grounded excerpts/references are persisted verbatim),
grounding confinement, and the HTTP/A2A authentication options.

## Positioning

EvidenceSpine is not a general orchestration framework. It is a memory + handoff quality layer that can plug into existing stacks (LangGraph, AutoGen, CrewAI, custom runtimes).
