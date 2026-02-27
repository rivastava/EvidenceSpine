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
- `evidencespine_lexical`
- `evidencespine_hybrid`
- `baseline_sqlite`
- `mem0` (optional; skipped when not installed)
- `letta` (optional; skipped when not installed)

## Run

```bash
cd oss/evidencespine
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

Published repo snapshot:
- `docs/benchmarks/apples_to_apples_2026-02-28.json`
- `docs/benchmarks/apples_to_apples_2026-02-28.md`

## Notes
1. `baseline_sqlite` is a DIY internal baseline for fairness.
2. External framework runners are optional and can be added incrementally.
3. For publishing claims, run 3+ seeds and report p50/p95 across runs.
4. The published 2026-02-28 snapshot was executed in an isolated benchmark venv:
   - `mem0` required `qdrant-client==1.13.3`
   - `letta` was exercised in local server mode
