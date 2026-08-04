# Apples-to-Apples Comparison

This benchmark compares runners on the **same synthetic workload** with the same counts:
1. Event ingest latency/throughput
2. Brief generation latency/throughput
3. Handoff latency/throughput
4. Brief citation coverage
5. Handoff completeness

## Script
- `benchmarks/apples_to_apples_compare.py`

## Default runners
- `evidencespine_lexical` (SQLite)
- `evidencespine_jsonl_lexical` (legacy JSONL backend)
- `evidencespine_hybrid` (SQLite + BM25 hybrid retrieval)
- `baseline_sqlite`
- `mem0` (optional; skipped when not installed)
- `letta` (optional; skipped when not installed)

## Run

```bash
cd evidencespine
PYTHONPATH=src python benchmarks/apples_to_apples_compare.py \
  --events 1200 \
  --queries 80 \
  --handoffs 30 \
  --out-json benchmarks/results/apples_to_apples.json \
  --out-md benchmarks/results/apples_to_apples.md
```

## Output
- JSON: `benchmarks/results/apples_to_apples.json`
- Markdown table: `benchmarks/results/apples_to_apples.md`

Published repo snapshots:
- `docs/benchmarks/apples_to_apples_2026-02-28.json`
- `docs/benchmarks/apples_to_apples_2026-02-28.md`
- Refresh in-progress snapshot: `benchmarks/results/apples_to_apples.md` (2026-08-04)

## Latest numbers (2026-08-04, 300 events / 20 queries / 8 handoffs, seed 42)

| runner | ingest eps | brief p95 (ms) | verified probe hit | governance score |
|---|---:|---:|---:|---:|
| evidencespine_lexical | 193.0 | 49.2 | 1.000 | 1.000 |
| evidencespine_jsonl_lexical | 190.6 | 53.7 | 1.000 | 1.000 |
| evidencespine_hybrid | 198.7 | 79.7 | 1.000 | 1.000 |
| baseline_sqlite | 1389.7 | 2.7 | 0.000 | 0.375 |

## Notes
1. `baseline_sqlite` is a DIY internal baseline for fairness.
2. External framework runners are optional and can be added incrementally.
3. For publishing claims, run 3+ seeds and report p50/p95 across runs.
4. The published 2026-02-28 snapshot was executed in an isolated benchmark venv:
   - `mem0` required `qdrant-client==1.13.3`
   - `letta` was exercised in local server mode
