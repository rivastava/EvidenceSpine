<p align="center">
  <img src="docs/evidencespine_logo.png" alt="EvidenceSpine logo" width="320" />
</p>

# EvidenceSpine

EvidenceSpine is an agent-agnostic conversation memory fabric for multi-agent workflows.

It provides:
- Evidence-bound event and fact memory (`asserted|verified|contradicted|superseded`)
- Bounded context brief generation (no raw prompt stuffing)
- Cross-agent handoff packets with checksum, citations, and optional exact evidence spans
- Fail-open behavior (memory failures do not block the caller)
- Optional governance hooks for contradiction checks and external policy systems
- Hybrid retrieval mode (`lexical|hybrid|vector`) with pluggable vector backend
- Transcript-first adapters for plain `messages[]`, with LangGraph and AutoGen wrappers built on top

## Why this exists
Most agent-memory systems store context. EvidenceSpine adds strict claim quality controls:
- Every claim can carry citations
- Claims can carry structured evidence items with exact line or character spans
- Contradictions are explicit
- Handoffs are structured and portable JSON

## Install

```bash
cd oss/evidencespine
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick start

1. Ingest an event:
```bash
evidencespine ingest \
  --thread-id demo \
  --event-type decision \
  --source-agent-id implementer \
  --source-turn-id t1 \
  --claim "Use additive patch set" \
  --fact-state verified \
  --evidence-ref "reports/decision_log.md" \
  --evidence-item-json '{"source_id":"reports/decision_log.md","line_start":12,"line_end":14,"excerpt":"Use additive patch set","checksum":"sha256:3f8f41ad7f0d13005c2d2f5e732c69b5c7f7bd03ecf8d38df4ef0ae50930f550"}' \
  --json
```

2. Build a brief:
```bash
evidencespine brief --thread-id demo --query "current status and next actions" --json
```

3. Emit a handoff packet:
```bash
evidencespine handoff --role auditor --thread-id demo --scope "verify latest claims" --output /tmp/demo_handoff.json
```

4. Import a handoff packet:
```bash
evidencespine handoff --import /tmp/demo_handoff.json --json
```

5. Read health snapshot:
```bash
evidencespine snapshot --json
```

## Hybrid retrieval (vector + lexical)

Set via env:

```bash
export EVIDENCESPINE_RETRIEVAL_MODE=hybrid
export EVIDENCESPINE_RETRIEVAL_LEXICAL_WEIGHT=1.0
export EVIDENCESPINE_RETRIEVAL_VECTOR_WEIGHT=0.35
```

By default hybrid uses an internal hashing vector backend (dependency-free).  
You can inject your own backend by implementing `score_texts(query, texts) -> scores`.

## Protocol v2 evidence spans

EvidenceSpine `0.3.0` adds structured evidence spans without removing legacy refs:

- `evidence_refs`: compatible string refs for existing integrations
- `evidence_items`: structured spans with `source_id`, `locator`, line or char anchors, optional excerpt, checksum, confidence, and verification state
- `citations`: structured brief claim citations with `primary_ref`, `evidence_refs`, `evidence_items`, and `span_grounded`
- `citation_refs`: legacy brief alias for refs-only consumers

## Transcript-first integration

```python
from evidencespine import AgentMemoryRuntime, EvidenceSpineSettings
from evidencespine.adapters import TranscriptAdapter, LangGraphAdapter, AutoGenAdapter

rt = AgentMemoryRuntime(config=EvidenceSpineSettings.from_env().to_runtime_config())
ta = TranscriptAdapter(rt)
lg = LangGraphAdapter(rt)
ag = AutoGenAdapter(rt)
```

See:
- `docs/ADAPTERS.md`
- `examples/multi_agent_handoff.py`
- `examples/claude_code_usage.py`
- `examples/transcript_replay_harness.py`
- `docs/INTEGRATION.md`

## Storage layout
By default data is stored in `.evidencespine/`:
- `.evidencespine/events.jsonl`
- `.evidencespine/facts.jsonl`
- `.evidencespine/state.json`
- `.evidencespine/briefs/*.json`
- `.evidencespine/handoffs/*.json`

Use `--base-dir` or `EVIDENCESPINE_BASE_DIR` to override.

## Protocol contract
See:
- `docs/PROTOCOL.md`
- `docs/ARCHITECTURE.md`
- `docs/ADAPTERS.md`
- `docs/INTEGRATION.md`
- `docs/CLAUDE_CODE.md`
- `docs/BENCHMARKS.md`
- `docs/APPLE_TO_APPLE_COMPARISON.md`

## Benchmarks

Run benchmark harness:

```bash
cd oss/evidencespine
PYTHONPATH=src python benchmarks/bench_evidencespine.py \
  --events 2000 \
  --brief-queries 100 \
  --handoffs 40 \
  --out benchmarks/results/latest.json
```

Apples-to-apples comparison:

```bash
cd oss/evidencespine
PYTHONPATH=src python benchmarks/apples_to_apples_compare.py \
  --events 1200 \
  --queries 80 \
  --handoffs 30 \
  --out-json benchmarks/results/apples_to_apples.json \
  --out-md benchmarks/results/apples_to_apples.md
```

Published benchmark snapshot:
- `docs/benchmarks/apples_to_apples_2026-02-28.json`
- `docs/benchmarks/apples_to_apples_2026-02-28.md`

## Release checklist
- [x] Installable package (`pyproject.toml`)
- [x] OSI-approved open-source license
- [x] CLI and examples
- [x] Replay-backed adapter examples
- [x] Adapter normalization contract
- [x] Documentation

## License
This project is distributed under the **Apache License 2.0**.

See:
- `LICENSE`
- `DCO`

## Contributing
Contributions are welcome. By submitting a contribution, you agree it is licensed under Apache-2.0 and certify origin via the Developer Certificate of Origin (`DCO`).
Use `Signed-off-by` in commits, for example:

```bash
git commit -s -m "feat: improve adapter replay docs"
```

## Positioning
EvidenceSpine is not a general orchestration framework. It is a memory + handoff quality layer that can plug into existing stacks (LangGraph, AutoGen, CrewAI, custom runtimes).
