# Apples-to-Apples Comparison

- events: 1000
- queries: 50
- handoffs: 25
- seed: 42

| runner | status | ingest eps | brief qps | handoff qps | brief citation coverage | handoff completeness |
|---|---:|---:|---:|---:|---:|---:|
| evidencespine_lexical | ok | 114.79 | 33.00 | 23.58 | 1.000 | 1.000 |
| evidencespine_hybrid | ok | 113.50 | 20.15 | 16.33 | 1.000 | 1.000 |
| baseline_sqlite | ok | 2953.17 | 491.31 | 523.80 | 1.000 | 1.000 |
| mem0 | ok | 678.49 | 154.22 | 97.21 | 1.000 | 1.000 |
| letta | ok | 127.47 | 6.41 | 6.25 | 1.000 | 1.000 |

| runner | checksum rate | verified probe hit | contradiction probe hit | governance score |
|---|---:|---:|---:|---:|
| evidencespine_lexical | 1.000 | 1.000 | 1.000 | 1.000 |
| evidencespine_hybrid | 1.000 | 1.000 | 1.000 | 1.000 |
| baseline_sqlite | 1.000 | 0.000 | 0.000 | 0.600 |
| mem0 | 1.000 | 1.000 | 1.000 | 1.000 |
| letta | 1.000 | 1.000 | 1.000 | 1.000 |
