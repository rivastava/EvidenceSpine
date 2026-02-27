# Benchmarks

This project includes a reproducible local benchmark harness:
- `benchmarks/bench_evidencespine.py`

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
