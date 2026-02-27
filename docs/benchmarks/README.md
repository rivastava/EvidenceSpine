# Benchmark Snapshots

This folder stores publishable benchmark artifacts that are checked into git.

Current snapshot:
- `apples_to_apples_2026-02-28.json`
- `apples_to_apples_2026-02-28.md`

Workload for this snapshot:
1. `events=1000`
2. `queries=50`
3. `handoffs=25`
4. `seed=42`

Runners:
1. `evidencespine_lexical`
2. `evidencespine_hybrid`
3. `baseline_sqlite`
4. `mem0`
5. `letta`

Execution notes:
1. Run executed in isolated benchmark virtualenv (`.bench_venv`).
2. `mem0` compatibility path used `qdrant-client==1.13.3`.
3. `letta` was exercised using local server mode.
