# Benchmark Snapshots

This folder stores publishable benchmark artifacts that are checked into git.

## Current snapshot (2026-11-20, protocol v2)

- `apples_to_apples_2026-11-20.json`
- `apples_to_apples_2026-11-20.md`

Protocol:

1. `events=300`
2. `queries=20`
3. `handoffs=8`
4. seeds `42,43,44` (per-seed runs; report mean and p50/p95 across seeds)

Runners: `evidencespine_lexical`, `evidencespine_jsonl_lexical`,
`evidencespine_hybrid`, `baseline_sqlite`; `mem0` / `letta` are optional
and are reported `skipped` when not installed.

The verified/contradiction probes ingest a real checksummed evidence item (the
pre-0.6 probe was unacceptable under the verified-requires-span policy, so its
published 1.000 was stale). See `docs/APPLE_TO_APPLE_COMPARISON.md` and
`docs/BENCHMARKS.md` for the full protocol, honest caveats and the numbers.

## Historical snapshots

- `apples_to_apples_2026-08-04.json` / `.md` — pre-correction probe
- `apples_to_apples_2026-02-28.json` / `.md` — pre-correction probe,
  isolated benchmark virtualenv (`.bench_venv`); `mem0` used
  `qdrant-client==1.13.3` and `letta` ran in local server mode

Do not mix historical snapshots with the 2026-11-20 numbers: the quality
columns are not comparable after the grounded-probe correction.
