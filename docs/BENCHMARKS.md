# Benchmarks

This project includes a reproducible local benchmark harness:
- `benchmarks/bench_evidencespine.py`
- `benchmarks/apples_to_apples_compare.py`

It measures:
1. Ingest latency and throughput (`events/sec`, p50, p95)
2. Brief latency and throughput (`queries/sec`, p50, p95)
3. Handoff latency and throughput (`handoffs/sec`, p50, p95)

Modes:
- `lexical`
- `hybrid`
- `vector`

## Run

```bash
cd oss/evidencespine
PYTHONPATH=src python benchmarks/bench_evidencespine.py \
  --events 2000 \
  --brief-queries 100 \
  --handoffs 40 \
  --out benchmarks/results/latest.json
```

## Expected output

- Console JSON summary for all modes.
- Artifact file:
  - `benchmarks/results/latest.json`
- Per-run state snapshots:
  - `benchmarks/runs/<timestamp>/<mode>/...`

## Interpretation notes

1. `hybrid`/`vector` can be slower than `lexical` because they add vector scoring.
2. Focus on p95 latency and throughput, not single-run p50 only.
3. Run multiple seeds/hardware profiles before making product claims.

See apples-to-apples guide:
- `docs/APPLE_TO_APPLE_COMPARISON.md`

## Published snapshot

Latest checked-in apples-to-apples snapshot:
- `docs/benchmarks/apples_to_apples_2026-02-28.json`
- `docs/benchmarks/apples_to_apples_2026-02-28.md`

Notes for this snapshot:
1. `mem0` run used `qdrant-client==1.13.3` for compatibility with installed `mem0` APIs.
2. `letta` run used a local Letta server process in benchmark mode.
