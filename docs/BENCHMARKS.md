# Benchmarks

This project includes a reproducible local benchmark harness:
- `benchmarks/bench_evidencespine.py`
- `benchmarks/apples_to_apples_compare.py`

It measures:
1. Ingest latency and throughput (`events/sec`, p50, p95)
2. Brief latency and throughput (`queries/sec`, p50, p95)
3. Handoff latency and throughput (`handoffs/sec`, p50, p95)
4. Brief claim ref citation coverage
5. Brief claim span citation coverage
6. Brief excerpt fidelity
7. Handoff claim span grounding rate

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
2. Grounding metrics are meaningful only when benchmark events carry structured `evidence_items`.
3. `brief_claim_ref_citation_coverage` measures whether claims point to any citation ref at all.
4. `brief_claim_span_citation_coverage` measures whether claims point to at least one structured grounded span.
5. `brief_claim_excerpt_fidelity` measures whether the primary evidence excerpt survives persistence and still matches its checksum.
6. `handoff_claim_span_grounding_rate` measures whether handoff claim and contradiction rows retain structured grounded evidence.
7. Focus on p95 latency and grounding quality together, not single-run p50 only.
8. Run multiple seeds or hardware profiles before making product claims.

See apples-to-apples guide:
- `docs/APPLE_TO_APPLE_COMPARISON.md`

## Published snapshot

Latest checked-in apples-to-apples snapshot:
- `docs/benchmarks/apples_to_apples_2026-02-28.json`
- `docs/benchmarks/apples_to_apples_2026-02-28.md`

Notes for this snapshot:
1. `mem0` run used `qdrant-client==1.13.3` for compatibility with installed `mem0` APIs.
2. `letta` run used a local Letta server process in benchmark mode.
