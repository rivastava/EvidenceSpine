from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def test_bench_evidencespine_reports_grounding_metrics(tmp_path: Path) -> None:
    out_path = tmp_path / "bench.json"
    subprocess.run(
        [
            sys.executable,
            "benchmarks/bench_evidencespine.py",
            "--events",
            "100",
            "--brief-queries",
            "10",
            "--handoffs",
            "5",
            "--out",
            str(out_path),
            "--modes",
            "lexical",
        ],
        cwd=ROOT,
        env=_env(),
        check=True,
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    row = payload["results"][0]
    assert isinstance(row["brief_claim_ref_citation_coverage"], float)
    assert isinstance(row["brief_claim_span_citation_coverage"], float)
    assert isinstance(row["brief_claim_excerpt_fidelity"], float)
    assert isinstance(row["handoff_claim_span_grounding_rate"], float)


def test_apples_to_apples_reports_grounding_metrics(tmp_path: Path) -> None:
    out_json = tmp_path / "apples.json"
    out_md = tmp_path / "apples.md"
    subprocess.run(
        [
            sys.executable,
            "benchmarks/apples_to_apples_compare.py",
            "--events",
            "100",
            "--queries",
            "10",
            "--handoffs",
            "5",
            "--runners",
            "evidencespine_lexical,baseline_sqlite",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=ROOT,
        env=_env(),
        check=True,
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    row = payload["results"][0]
    assert isinstance(row["brief_claim_ref_citation_coverage"], float)
    assert isinstance(row["brief_claim_span_citation_coverage"], float)
    assert isinstance(row["brief_claim_excerpt_fidelity"], float)
    assert isinstance(row["handoff_claim_span_grounding_rate"], float)
    assert "brief span coverage" in out_md.read_text(encoding="utf-8").lower()
    # The verified probe must be grounded with a checksummed excerpt so the
    # verified-requires-span policy keeps it verified (regression guard).
    assert row["verified_probe_hit"] == 1.0, "verified probe must survive grounding policy"
    assert row["contradiction_probe_hit"] == 1.0
    assert row["governance_score"] == 1.0
    assert "runs" not in payload, "single-seed default output shape must stay unchanged"
    assert "aggregates" not in row


def test_apples_to_apples_multi_seed_aggregates(tmp_path: Path) -> None:
    out_json = tmp_path / "apples_seeds.json"
    out_md = tmp_path / "apples_seeds.md"
    subprocess.run(
        [
            sys.executable,
            "benchmarks/apples_to_apples_compare.py",
            "--events",
            "100",
            "--queries",
            "10",
            "--handoffs",
            "5",
            "--seeds",
            "42,43",
            "--runners",
            "evidencespine_lexical",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=ROOT,
        env=_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["params"]["seeds"] == [42, 43]
    assert len(payload["runs"]) == 2, "each seed must produce a run record"
    row = payload["results"][0]
    assert row["runs"] == 2
    assert row["seed"] in {42, 43} or row.get("seeds") == [42, 43]
    aggregates = row["aggregates"]
    for metric in (
        "governance_score",
        "brief_claim_span_citation_coverage",
        "ingest_eps",
        "brief_qps",
    ):
        agg = aggregates[metric]
        assert {"mean", "p50", "p95"} <= set(agg)
        assert agg["p50"] <= agg["p95"]
    markdown = out_md.read_text(encoding="utf-8")
    assert "Multi-seed aggregates" in markdown
    assert "p50" in markdown and "p95" in markdown
    assert "seeds: 42, 43" in markdown
