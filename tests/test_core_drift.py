"""Core hardening: drift-check anchors, confinement mode, unverifiable verdicts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def _runtime(tmp_path: Path) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ingest(rt: AgentMemoryRuntime, claim: str, item: dict[str, Any], *, thread: str = "demo") -> str:
    out = rt.ingest_event(
        {
            "thread_id": thread,
            "event_type": "outcome",
            "role": "auditor",
            "source_agent_id": "auditor",
            "source_turn_id": f"turn-{claim}",
            "payload": {"claim": claim, "fact_state": "verified"},
            "evidence_items": [item],
        }
    )
    assert out["status"] == "ok", out
    return claim


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- item 15: char anchors / locator anchors / unverifiable ---------------


def test_char_anchor_verifies_and_flags_change(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root, "src/f.txt", "0123456789")
    rt = _runtime(tmp_path)
    _ingest(rt, "char claim", {"source_id": "src/f.txt", "char_start": 2, "char_end": 5, "excerpt": "2345", "checksum": _sha("2345")})
    clean = rt.check_evidence_stale(source_root=str(root))
    assert clean["stale_facts"] == 0
    assert clean["checked_items"] == 1
    assert clean["unverifiable_items"] == 0

    _write(root, "src/f.txt", "01XX456789")
    changed = rt.check_evidence_stale(source_root=str(root))
    assert changed["stale_facts"] == 1
    assert changed["results"][0]["reason"] == "changed"
    rt.store.close()


def test_char_anchor_out_of_range_is_unverifiable(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root, "src/f.txt", "short")
    rt = _runtime(tmp_path)
    _ingest(rt, "range claim", {"source_id": "src/f.txt", "char_start": 10, "char_end": 12, "excerpt": "xyz", "checksum": _sha("xyz")})
    out = rt.check_evidence_stale(source_root=str(root), dry_run=False)
    assert out["stale_facts"] == 0
    assert out["unverifiable_items"] == 1
    assert out["results"][0]["reason"] == "unverifiable"
    assert not list(rt.store.iter_facts())[0]["metadata"].get("evidence_stale")
    rt.store.close()


def test_locator_only_item_is_verified_through_parse_ref(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root, "src/f.txt", "alpha\nbeta\n")
    rt = _runtime(tmp_path)
    _ingest(
        rt,
        "locator claim",
        {"source_id": "src/f.txt", "locator": "src/f.txt#L1-L2", "excerpt": "alpha\nbeta", "checksum": _sha("alpha\nbeta")},
    )
    clean = rt.check_evidence_stale(source_root=str(root))
    assert clean["stale_facts"] == 0
    assert clean["checked_items"] == 1
    _write(root, "src/f.txt", "alpha\nCHANGED\n")
    changed = rt.check_evidence_stale(source_root=str(root))
    assert changed["stale_facts"] == 1
    assert changed["results"][0]["reason"] == "changed"
    rt.store.close()


def test_unsupported_locator_is_unverifiable(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root, "src/f.txt", "alpha")
    rt = _runtime(tmp_path)
    _ingest(rt, "bad locator", {"source_id": "src/f.txt", "locator": "no-anchor-here", "excerpt": "alpha", "checksum": _sha("alpha")})
    out = rt.check_evidence_stale(source_root=str(root))
    assert out["stale_facts"] == 0
    assert out["unverifiable_items"] == 1
    assert out["results"][0]["reason"] == "unverifiable"
    rt.store.close()


def test_ambiguous_line_and_char_anchors_are_unverifiable(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root, "src/f.txt", "0123456789")
    rt = _runtime(tmp_path)
    _ingest(
        rt,
        "ambiguous claim",
        {"source_id": "src/f.txt", "line_start": 1, "char_start": 0, "char_end": 3, "excerpt": "0123", "checksum": _sha("0123")},
    )
    out = rt.check_evidence_stale(source_root=str(root))
    assert out["stale_facts"] == 0
    assert out["unverifiable_items"] == 1
    rt.store.close()


# --- item 15: confinement / absolute source id mode -----------------------


def test_absolute_source_id_is_unverifiable_when_disallowed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = _write(root, "src/f.txt", "alpha\n")
    rt = _runtime(tmp_path)
    _ingest(
        rt,
        "absolute claim",
        {"source_id": str(target), "line_start": 1, "excerpt": "alpha", "checksum": _sha("alpha")},
    )
    confined = rt.check_evidence_stale(source_root=str(root), allow_absolute_source_id=False)
    assert confined["stale_facts"] == 0
    assert confined["unverifiable_items"] == 1
    assert confined["results"][0]["reason"] == "unverifiable"

    local = rt.check_evidence_stale(source_root=str(root), allow_absolute_source_id=True)
    assert local["stale_facts"] == 0
    assert local["unverifiable_items"] == 0
    rt.store.close()


def test_relative_source_escape_is_unverifiable_in_confined_mode(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    _write(tmp_path, "secret.txt", "needle\n")
    rt = _runtime(tmp_path)
    _ingest(
        rt,
        "escape claim",
        {"source_id": "../secret.txt", "line_start": 1, "excerpt": "needle", "checksum": _sha("needle")},
    )
    confined = rt.check_evidence_stale(source_root=str(root), allow_absolute_source_id=False)
    assert confined["stale_facts"] == 0
    assert confined["unverifiable_items"] == 1
    rt.store.close()


def test_unverifiable_does_not_clear_existing_stale_flag(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root, "src/f.txt", "alpha\nbeta\n")
    rt = _runtime(tmp_path)
    _ingest(
        rt,
        "flag claim",
        {"source_id": "src/f.txt", "line_start": 1, "line_end": 2, "excerpt": "alpha\nbeta", "checksum": _sha("alpha\nbeta")},
    )
    _write(root, "src/f.txt", "alpha\nCHANGED\n")
    assert rt.check_evidence_stale(source_root=str(root), dry_run=False)["stale_facts"] == 1
    fact = list(rt.store.iter_facts())[0]
    assert fact["metadata"]["evidence_stale"] is True

    # Swap in an anchor the checker cannot evaluate: stale must stay (fail-closed).
    rt.store.update_fact(
        fact["fact_id"],
        {"evidence_items": [{"source_id": "src/f.txt", "char_start": 99, "char_end": 120, "excerpt": "zz", "checksum": _sha("zz")}]},
    )
    out = rt.check_evidence_stale(source_root=str(root), dry_run=False)
    assert out["unverifiable_items"] == 1
    assert out["cleared_facts"] == 0
    assert list(rt.store.iter_facts())[0]["metadata"]["evidence_stale"] is True
    rt.store.close()
