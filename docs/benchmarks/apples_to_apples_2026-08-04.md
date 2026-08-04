# Apples-to-Apples Comparison

- events: 300
- queries: 20
- handoffs: 8
- seed: 42

| runner | status | ingest eps | brief qps | handoff qps | brief ref coverage | brief span coverage | excerpt fidelity | handoff completeness | handoff span grounding |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| evidencespine_lexical | ok | 192.99 | 20.63 | 19.44 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| evidencespine_jsonl_lexical | ok | 190.57 | 19.64 | 20.39 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| evidencespine_hybrid | ok | 198.71 | 13.33 | 12.72 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| baseline_sqlite | ok | 1389.67 | 536.35 | 471.18 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 |
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
