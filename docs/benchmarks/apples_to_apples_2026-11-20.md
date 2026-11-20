# Apples-to-Apples Comparison

- events: 300
- queries: 20
- handoffs: 8
- seed: 42
- seeds: 42, 43, 44 (aggregated: mean of per-seed runs; p50/p95 across seeds)

| runner | status | ingest eps | brief qps | handoff qps | brief ref coverage | brief span coverage | excerpt fidelity | handoff completeness | handoff span grounding |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| evidencespine_lexical | ok | 1590.48 | 51.76 | 46.95 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| evidencespine_jsonl_lexical | ok | 918.71 | 39.05 | 35.73 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| evidencespine_hybrid | ok | 1561.52 | 25.11 | 23.88 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| baseline_sqlite | ok | 3021.26 | 769.74 | 796.35 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| mem0 | skipped | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| letta | skipped | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

| runner | checksum rate | verified probe hit | contradiction probe hit | governance score |
|---|---:|---:|---:|---:|
| evidencespine_lexical | 1.000 | 1.000 | 1.000 | 1.000 |
| evidencespine_jsonl_lexical | 1.000 | 1.000 | 1.000 | 1.000 |
| evidencespine_hybrid | 1.000 | 1.000 | 1.000 | 1.000 |
| baseline_sqlite | 1.000 | 0.000 | 0.000 | 0.375 |
| mem0 | 0.000 | 0.000 | 0.000 | 0.000 |
| letta | 0.000 | 0.000 | 0.000 | 0.000 |

## Multi-seed aggregates (mean / p50 / p95 across seeds)

| runner | metric | mean | p50 | p95 |
|---|---|---:|---:|---:|
| evidencespine_lexical | ingest_eps | 1590.4776 | 1590.8431 | 1597.6234 |
| evidencespine_lexical | brief_qps | 51.7645 | 51.4615 | 53.0883 |
| evidencespine_lexical | handoff_qps | 46.9459 | 46.5261 | 48.3039 |
| evidencespine_lexical | brief_claim_span_citation_coverage | 1.0000 | 1.0000 | 1.0000 |
| evidencespine_lexical | brief_claim_excerpt_fidelity | 1.0000 | 1.0000 | 1.0000 |
| evidencespine_lexical | handoff_claim_span_grounding_rate | 1.0000 | 1.0000 | 1.0000 |
| evidencespine_lexical | verified_probe_hit | 1.0000 | 1.0000 | 1.0000 |
| evidencespine_lexical | contradiction_probe_hit | 1.0000 | 1.0000 | 1.0000 |
| evidencespine_lexical | governance_score | 1.0000 | 1.0000 | 1.0000 |
| evidencespine_jsonl_lexical | ingest_eps | 918.7061 | 905.1615 | 949.3716 |
| evidencespine_jsonl_lexical | brief_qps | 39.0548 | 38.5771 | 40.0610 |
| evidencespine_jsonl_lexical | handoff_qps | 35.7308 | 35.4313 | 37.5212 |
| evidencespine_jsonl_lexical | brief_claim_span_citation_coverage | 1.0000 | 1.0000 | 1.0000 |
| evidencespine_jsonl_lexical | brief_claim_excerpt_fidelity | 1.0000 | 1.0000 | 1.0000 |
| evidencespine_jsonl_lexical | handoff_claim_span_grounding_rate | 1.0000 | 1.0000 | 1.0000 |
| evidencespine_jsonl_lexical | verified_probe_hit | 1.0000 | 1.0000 | 1.0000 |
| evidencespine_jsonl_lexical | contradiction_probe_hit | 1.0000 | 1.0000 | 1.0000 |
| evidencespine_jsonl_lexical | governance_score | 1.0000 | 1.0000 | 1.0000 |
| evidencespine_hybrid | ingest_eps | 1561.5191 | 1608.1268 | 1613.6714 |
| evidencespine_hybrid | brief_qps | 25.1148 | 25.1497 | 25.3702 |
| evidencespine_hybrid | handoff_qps | 23.8779 | 24.0422 | 24.1509 |
| evidencespine_hybrid | brief_claim_span_citation_coverage | 1.0000 | 1.0000 | 1.0000 |
| evidencespine_hybrid | brief_claim_excerpt_fidelity | 1.0000 | 1.0000 | 1.0000 |
| evidencespine_hybrid | handoff_claim_span_grounding_rate | 1.0000 | 1.0000 | 1.0000 |
| evidencespine_hybrid | verified_probe_hit | 1.0000 | 1.0000 | 1.0000 |
| evidencespine_hybrid | contradiction_probe_hit | 1.0000 | 1.0000 | 1.0000 |
| evidencespine_hybrid | governance_score | 1.0000 | 1.0000 | 1.0000 |
| baseline_sqlite | ingest_eps | 3021.2606 | 2899.5102 | 3304.4834 |
| baseline_sqlite | brief_qps | 769.7435 | 769.4256 | 771.8122 |
| baseline_sqlite | handoff_qps | 796.3528 | 798.9980 | 800.6972 |
| baseline_sqlite | brief_claim_span_citation_coverage | 0.0000 | 0.0000 | 0.0000 |
| baseline_sqlite | brief_claim_excerpt_fidelity | 0.0000 | 0.0000 | 0.0000 |
| baseline_sqlite | handoff_claim_span_grounding_rate | 0.0000 | 0.0000 | 0.0000 |
| baseline_sqlite | verified_probe_hit | 0.0000 | 0.0000 | 0.0000 |
| baseline_sqlite | contradiction_probe_hit | 0.0000 | 0.0000 | 0.0000 |
| baseline_sqlite | governance_score | 0.3750 | 0.3750 | 0.3750 |
