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
