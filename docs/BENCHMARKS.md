# Benchmarks

Benchmarks are reproducible local measurements, not marketing claims. They
cover throughput/latency and evidence-quality metrics on a synthetic workload;
they do not measure real coding-agent task success.

## Harnesses

- `benchmarks/bench_evidencespine.py` — per-mode ingest/brief/handoff
  latency and throughput (lexical, hybrid, vector).
- `benchmarks/apples_to_apples_compare.py` — the same workload across
  EvidenceSpine modes and a DIY baseline (plus optional external runners),
  including evidence-quality probes.

## Workloads

`apples_to_apples` (snapshot protocol): 300 events, 20 brief queries,
8 handoffs, repeated for seeds 42, 43, 44. Per-run throughput is
events/sec (ingest), queries/sec (brief), handoffs/sec; p50/p95 are computed
across the three per-seed runs. Events carry structured evidence so citation
metrics are meaningful.

`bench_evidencespine` (scale): configurable event counts up to thousands,
optional `--transport` stdio/HTTP MCP variants. Use it for regression
tracking of ingest/brief latency, not for client-to-client claims.

## Corrected verified probe (important)

The verified/contradiction probes now ingest a **real checksummed evidence
item** (`_mk_probe_event` in `apples_to_apples_compare.py`): the probe
event attaches `evidence_items` with an excerpt, a span, and
sha256(excerpt) as checksum, then asks the brief whether the verified claim
appears in `recent_verified_facts` and the contradicted claim in
`active_risks`.

Why: since the `verified_requires_span` policy (default on), a bare
`fact_state: "verified"` event is **downgraded to asserted** and can never
appear as a verified fact. The previous probe omitted grounding, so it measured
nothing enforceable and the published `verified probe hit = 1.000` was stale
for any build that enforced the policy. The corrected probe measures the policy
and the brief end-to-end. Runs must therefore not be compared against
pre-correction snapshots.

## Fresh snapshot (2026-11-20, seeds 42/43/44)

Artifacts:

- `docs/benchmarks/apples_to_apples_2026-11-20.md`
- `docs/benchmarks/apples_to_apples_2026-11-20.json`

Mean per-seed values (p50/p95 across seeds in the artifact):

| runner | ingest eps | brief qps | handoff qps | span citation | excerpt fidelity | verified probe | contradiction probe | governance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| evidencespine_lexical | 1590.5 | 51.8 | 46.9 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| evidencespine_jsonl_lexical | 918.7 | 39.1 | 35.7 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| evidencespine_hybrid | 1561.5 | 25.1 | 23.9 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| baseline_sqlite | 3021.3 | 769.7 | 796.4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.375 |
| mem0 | skipped (not installed) | -- | -- | -- | -- | -- | -- | -- |
| letta | skipped (not installed) | -- | -- | -- | -- | -- | -- | -- |

p50/p95 examples (lexical): ingest 1590.8 / 1597.6 eps, brief 51.5 / 53.1
qps, handoff 46.5 / 48.3 qps. The full per-metric grid is in the artifact.

How to read it:

- `governance_score` is the mean of the eight quality terms (ref coverage,
  span coverage, excerpt fidelity, handoff completeness, handoff span
  grounding, checksum rate, verified probe, contradiction probe). The baseline
  scores 0.375 because it has **no grounding/citation implementation by
  design** — its 0.000 span/excerpt/probe terms are correct, not a bug.
- Throughput and guarantees are different axes. `baseline_sqlite` is faster
  because it does not do grounding, citations, brief assembly, or probes.
  Do not read 3021 eps vs 1590 eps as "slower is worse"; compare
  like-for-like and read the quality columns with the timing columns.
- `hybrid` trades brief/handoff throughput for vector scoring; lexical is the
  default.
- `evidencespine_jsonl_lexical` is the legacy JSONL backend (slower by
  design; SQLite is the default).

## Run it yourself

```sh
cd evidencespine
PYTHONPATH=src python benchmarks/apples_to_apples_compare.py \
  --events 300 --queries 20 --handoffs 8 \
  --seeds 42,43,44 \
  --out-json benchmarks/results/apples_to_apples_multiseed_seeds_42_43_44.json \
  --out-md  benchmarks/results/apples_to_apples_multiseed_seeds_42_43_44.md
```

Use `--runners evidencespine_lexical,baseline_sqlite` to restrict runners;
`mem0` / `letta` are attempted only when their packages are importable and
are reported `skipped` otherwise (not zero-scored runners). Single-seed runs
are supported (`--seed 42`) but should not be published as claims.

Scale/regression run:

```sh
PYTHONPATH=src python benchmarks/bench_evidencespine.py \
  --events 2000 --brief-queries 100 --handoffs 40 \
  --out benchmarks/results/latest.json
```

## Honest caveats

1. **Single machine.** Numbers come from one developer machine in a shared
   session; they are not a cross-host or production benchmark.
2. **DIY baseline.** `baseline_sqlite` is an internal lower-bound
   implementation for fairness, not a competitor. External runners
   (`mem0`, `letta`) are optional and were skipped in this environment
   (not installed).
3. **Synthetic workload.** 300 synthetic events / 20 queries / 8 handoffs do
   not represent real agent sessions; they are stable for regression tracking.
4. **Three seeds is a minimum.** p50/p95 across three runs is indicative, not
   statistically strong. Run more seeds before making comparative claims.
5. **Wall-clock noise.** Throughput varies with filesystem cache, CPU
   frequency, and other processes; treat small deltas as noise.
6. **Do not mix snapshots.** Pre-2026-11-20 snapshots predate the corrected
   grounded probe and the verified-requires-span enforcement; their quality
   columns are not comparable.
7. The core storage fix (state.json debounce) is documented in
   [ARCHITECTURE.md](ARCHITECTURE.md): runtime ingest no longer collapses as
   the store fills; `bench_evidencespine` is the right tool to re-check that
   regression.
