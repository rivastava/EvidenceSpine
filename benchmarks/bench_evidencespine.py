#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import string
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def _quantile_ms(samples_ms: List[float], q: float) -> float:
    if not samples_ms:
        return 0.0
    ordered = sorted(samples_ms)
    idx = int(round((len(ordered) - 1) * q))
    idx = max(0, min(len(ordered) - 1, idx))
    return float(ordered[idx])


def _rand_token(n: int = 8) -> str:
    return "".join(random.choice(string.ascii_lowercase) for _ in range(max(3, n)))


def _claim(i: int) -> str:
    # Include repeated tokens to simulate retrievable structure.
    segment = f"segment_{i % 11}"
    return f"decision {i} about {_rand_token(6)} {_rand_token(5)} {segment} status {_rand_token(4)}"


@dataclass
class ModeResult:
    mode: str
    events: int
    ingest_total_ms: float
    ingest_eps: float
    ingest_p50_ms: float
    ingest_p95_ms: float
    brief_total_ms: float
    brief_qps: float
    brief_p50_ms: float
    brief_p95_ms: float
    handoff_total_ms: float
    handoff_qps: float
    handoff_p50_ms: float
    handoff_p95_ms: float


def _run_mode(
    *,
    base_dir: Path,
    mode: str,
    events: int,
    brief_queries: int,
    handoffs: int,
) -> ModeResult:
    settings = EvidenceSpineSettings.from_env(base_dir=str(base_dir))
    settings.retrieval_mode = mode
    settings.retrieval_lexical_weight = 1.0
    settings.retrieval_vector_weight = 0.35
    runtime = AgentMemoryRuntime(config=settings.to_runtime_config())

    ingest_lat_ms: List[float] = []
    t0 = time.perf_counter()
    for i in range(events):
        s = time.perf_counter()
        runtime.ingest_event(
            {
                "thread_id": "bench_thread",
                "event_type": "decision" if i % 3 else "outcome",
                "role": "implementer" if i % 2 else "auditor",
                "source_agent_id": "bench_runner",
                "source_turn_id": f"turn_{i}",
                "payload": {
                    "claim": _claim(i),
                    "fact_state": "verified" if i % 7 == 0 else "asserted",
                    "decision": f"apply_patch_{i % 5}",
                    "outcome": "ok" if i % 4 else "needs_review",
                    "next_actions": [f"validate_{i % 9}", f"audit_{i % 6}"],
                },
                "evidence_refs": [f"bench/file_{i % 13}.md#L{i % 100 + 1}"],
                "confidence": 0.7,
                "salience": 0.6,
            }
        )
        ingest_lat_ms.append((time.perf_counter() - s) * 1000.0)
    ingest_total_ms = (time.perf_counter() - t0) * 1000.0

    brief_lat_ms: List[float] = []
    t1 = time.perf_counter()
    for i in range(brief_queries):
        s = time.perf_counter()
        runtime.build_brief("bench_thread", f"what changed in segment_{i % 11}?")
        brief_lat_ms.append((time.perf_counter() - s) * 1000.0)
    brief_total_ms = (time.perf_counter() - t1) * 1000.0

    handoff_lat_ms: List[float] = []
    t2 = time.perf_counter()
    for i in range(handoffs):
        s = time.perf_counter()
        runtime.emit_handoff("auditor", "bench_thread", scope=f"verify wave {i}")
        handoff_lat_ms.append((time.perf_counter() - s) * 1000.0)
    handoff_total_ms = (time.perf_counter() - t2) * 1000.0

    runtime.flush()

    return ModeResult(
        mode=mode,
        events=events,
        ingest_total_ms=ingest_total_ms,
        ingest_eps=float(events / max(0.001, ingest_total_ms / 1000.0)),
        ingest_p50_ms=_quantile_ms(ingest_lat_ms, 0.50),
        ingest_p95_ms=_quantile_ms(ingest_lat_ms, 0.95),
        brief_total_ms=brief_total_ms,
        brief_qps=float(brief_queries / max(0.001, brief_total_ms / 1000.0)),
        brief_p50_ms=_quantile_ms(brief_lat_ms, 0.50),
        brief_p95_ms=_quantile_ms(brief_lat_ms, 0.95),
        handoff_total_ms=handoff_total_ms,
        handoff_qps=float(handoffs / max(0.001, handoff_total_ms / 1000.0)),
        handoff_p50_ms=_quantile_ms(handoff_lat_ms, 0.50),
        handoff_p95_ms=_quantile_ms(handoff_lat_ms, 0.95),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="EvidenceSpine benchmark")
    parser.add_argument("--events", type=int, default=2000)
    parser.add_argument("--brief-queries", type=int, default=100)
    parser.add_argument("--handoffs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="benchmarks/results/latest.json")
    parser.add_argument("--modes", default="lexical,hybrid,vector")
    args = parser.parse_args()

    random.seed(int(args.seed))

    modes = [m.strip().lower() for m in str(args.modes).split(",") if m.strip()]
    modes = [m for m in modes if m in {"lexical", "hybrid", "vector"}]
    if not modes:
        raise SystemExit("No valid modes requested")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_ts = int(time.time())
    root = Path("benchmarks") / "runs" / str(run_ts)
    root.mkdir(parents=True, exist_ok=True)

    results: List[ModeResult] = []
    for mode in modes:
        mode_dir = root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        results.append(
            _run_mode(
                base_dir=mode_dir,
                mode=mode,
                events=int(max(100, args.events)),
                brief_queries=int(max(10, args.brief_queries)),
                handoffs=int(max(5, args.handoffs)),
            )
        )

    payload: Dict[str, Any] = {
        "benchmark": "evidencespine_v1",
        "timestamp": run_ts,
        "params": {
            "events": int(args.events),
            "brief_queries": int(args.brief_queries),
            "handoffs": int(args.handoffs),
            "seed": int(args.seed),
            "modes": modes,
        },
        "results": [asdict(r) for r in results],
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
