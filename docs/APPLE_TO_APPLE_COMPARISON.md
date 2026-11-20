# Apples-to-Apples Comparison

This benchmark runs runner implementations on the **same synthetic workload**
with the same counts, seeds, and quality probes.

## Script and runners

`benchmarks/apples_to_apples_compare.py`

Default runners:

- `evidencespine_lexical` (SQLite, lexical retrieval) — the default build
- `evidencespine_jsonl_lexical` (legacy JSONL backend)
- `evidencespine_hybrid` (SQLite + hybrid retrieval)
- `baseline_sqlite` — DIY lower bound (stores text rows only)
- `mem0` / `letta` — optional external runners; attempted only when their
  packages are importable, reported `skipped` otherwise

## Protocol (3 seeds)

```sh
cd evidencespine
PYTHONPATH=src python benchmarks/apples_to_apples_compare.py \
  --events 300 --queries 20 --handoffs 8 \
  --seeds 42,43,44 \
  --out-json benchmarks/results/apples_to_apples_multiseed_seeds_42_43_44.json \
  --out-md  benchmarks/results/apples_to_apples_multiseed_seeds_42_43_44.md
```

- Each seed is a full independent run (fresh store under
  `benchmarks/runs/seed_<n>/` when multi-seed).
- Reported ingest eps / brief qps / handoff qps are the **mean of the
  per-seed runs**; p50/p95 are computed **across seeds**.
- The workload is 300 events, 20 brief queries, 8 handoffs, with grounded
  evidence attached to events so citation/span metrics are meaningful.

## Metrics

Throughput/latency: `ingest_eps, ingest_p50_ms, ingest_p95_ms, brief_qps,
brief_p50_ms, brief_p95_ms, handoff_qps, handoff_p50_ms, handoff_p95_ms`.

Quality: `brief_claim_citation_coverage`, `brief_claim_ref_citation_coverage`,
`brief_claim_span_citation_coverage`, `brief_claim_excerpt_fidelity`,
`handoff_completeness_rate`, `handoff_claim_span_grounding_rate`,
`handoff_checksum_rate`, `verified_probe_hit`, `contradiction_probe_hit`,
`governance_score` (mean of the eight quality terms).

## Corrected verified/contradiction probe

The probe now ingests a real checksummed evidence item: a probe event with
`evidence_items` (source span + excerpt + sha256(excerpt) checksum) for a
verified claim and for a contradicted claim. The brief must then show the
verified claim in `recent_verified_facts` and the contradicted claim in
`active_risks`.

The old probe attached `fact_state: "verified"` **without grounding**, which
the verified-requires-span policy has stored as `asserted` since it was
enforced. It therefore measured nothing and the published 1.000 was stale.
Runs from this corrected probe must not be compared with pre-correction
snapshots.

## Latest snapshot (2026-11-20, seeds 42/43/44)

Artifact: `docs/benchmarks/apples_to_apples_2026-11-20.{md,json}`

| runner | ingest eps | brief qps | handoff qps | verified probe | governance |
|---|---:|---:|---:|---:|---:|
| evidencespine_lexical | 1590.5 | 51.8 | 46.9 | 1.000 | 1.000 |
| evidencespine_jsonl_lexical | 918.7 | 39.1 | 35.7 | 1.000 | 1.000 |
| evidencespine_hybrid | 1561.5 | 25.1 | 23.9 | 1.000 | 1.000 |
| baseline_sqlite | 3021.3 | 769.7 | 796.4 | 0.000 | 0.375 |
| mem0 | skipped (not installed) | -- | -- | -- | -- |
| letta | skipped (not installed) | -- | -- | -- | -- |

Span citation, excerpt fidelity, contradiction probe and handoff span grounding
are 1.000 for the three EvidenceSpine runners and 0.000 for `baseline_sqlite`
(it has no grounding/citation implementation by design). Full p50/p95 grid:
see the JSON/Markdown artifact.

## Reading the result honestly

1. **Faster is not better here.** `baseline_sqlite` wins throughput because
   it stores text and does no grounding, citations, brief assembly, or probes;
   its quality row is intentionally 0.000.
2. **Hybrid is a trade.** It keeps 1.000 quality and pays brief/handoff
   throughput for vector scoring; lexical is the default.
3. **JSONL is legacy.** `evidencespine_jsonl_lexical` is slower by design;
   SQLite is the default since 0.5.0.
4. **Three seeds.** Mean with p50/p95 across seeds 42/43/44 is a stability
   check, not statistical significance. Run more seeds or hardware profiles
   before publishing comparative claims.
5. **Single machine, synthetic workload.** 300 synthetic events / 20 queries /
   8 handoffs on one machine; not a production or cross-host benchmark.
6. **External runners are optional.** `mem0` and `letta` were skipped
   because they are not installed in this environment; a skipped runner is
   reported as `status: skipped` and must not be presented as a 0 score.
7. Older snapshots (`2026-02-28`, `2026-08-04`) predate the corrected
   grounded probe and the verified-requires-span enforcement. Do not mix them
   with 2026-11-20 numbers.

## Reproducibility notes

- Stores are created under a temp/benchmark root and removed by the harness;
  `benchmarks/runs/<timestamp>/` keeps per-run artifacts for inspection.
- Vector/hybrid runs use the configured embedding backend (`auto` by default:
  fastembed when installed, otherwise the dependency-free hashing backend), so
  hybrid numbers depend on that environment.
- Use `--runners` to reproduce a subset, and keep the JSON artifact alongside
  any published table.
