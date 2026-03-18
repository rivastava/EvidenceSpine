from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from evidencespine.cli import _cmd_ingest, _cmd_reconcile, _cmd_view


def _ingest_args(tmp_path: Path, **overrides) -> Namespace:
    values = {
        "base_dir": str(tmp_path / ".es"),
        "thread_id": "demo",
        "event_type": "decision",
        "role": "implementer",
        "source_agent_id": "tester",
        "source_turn_id": "t1",
        "objective_id": "",
        "claim": "deploy patch",
        "decision": "",
        "outcome": "",
        "target": "",
        "fact_state": "verified",
        "next_action": [],
        "evidence_ref": [],
        "evidence_item_json": [],
        "evidence_item_file": [],
        "scope_id": "",
        "scope_kind": "",
        "state_kind": "",
        "status": "",
        "owner_agent_id": "",
        "state_basis": "",
        "validated_at": "",
        "validated_by": "",
        "fresh_until": "",
        "lease_expires_at": "",
        "supersedes": "",
        "confidence": 0.6,
        "salience": 0.5,
        "json": True,
    }
    values.update(overrides)
    return Namespace(**values)


def _latest_event(tmp_path: Path) -> dict:
    path = tmp_path / ".es" / "events.jsonl"
    return json.loads(path.read_text(encoding="utf-8").strip().splitlines()[-1])


def test_cli_ingest_accepts_evidence_item_json(tmp_path: Path) -> None:
    args = _ingest_args(
        tmp_path,
        evidence_item_json=['{"source_id":"src/file.py","line_start":10,"line_end":12}'],
    )
    assert _cmd_ingest(args) == 0
    assert _latest_event(tmp_path)["evidence_items"][0]["source_id"] == "src/file.py"


def test_cli_ingest_accepts_evidence_item_file_object(tmp_path: Path) -> None:
    item_path = tmp_path / "item.json"
    item_path.write_text('{"source_id":"src/file.py","line_start":20,"line_end":22}', encoding="utf-8")
    args = _ingest_args(tmp_path, evidence_item_file=[str(item_path)])
    assert _cmd_ingest(args) == 0
    assert _latest_event(tmp_path)["evidence_items"][0]["line_start"] == 20


def test_cli_ingest_accepts_evidence_item_file_array(tmp_path: Path) -> None:
    item_path = tmp_path / "items.json"
    item_path.write_text(
        '[{"source_id":"src/file.py","line_start":30,"line_end":31},{"source_id":"notes.md","line_start":2,"line_end":3}]',
        encoding="utf-8",
    )
    args = _ingest_args(tmp_path, evidence_item_file=[str(item_path)])
    assert _cmd_ingest(args) == 0
    assert len(_latest_event(tmp_path)["evidence_items"]) == 2


def test_cli_ingest_merges_evidence_refs_and_evidence_items(tmp_path: Path) -> None:
    args = _ingest_args(
        tmp_path,
        evidence_ref=["manual.md#L1"],
        evidence_item_json=['{"source_id":"src/file.py","line_start":40,"line_end":42}'],
    )
    assert _cmd_ingest(args) == 0
    event = _latest_event(tmp_path)
    assert "manual.md#L1" in event["evidence_refs"]
    assert "src/file.py#L40-L42" in event["evidence_refs"]


def test_cli_ingest_accepts_state_context_fields(tmp_path: Path) -> None:
    args = _ingest_args(
        tmp_path,
        scope_id="auth-timeout-fix",
        state_kind="agent_local_work",
        status="active",
        owner_agent_id="implementer",
    )
    assert _cmd_ingest(args) == 0
    event = _latest_event(tmp_path)
    assert event["state_context"]["scope_id"] == "auth-timeout-fix"


def test_cli_view_returns_expected_json_shape(tmp_path: Path, capsys) -> None:
    ingest_args = _ingest_args(
        tmp_path,
        scope_id="release-gate",
        state_kind="pending_gate",
        status="ready",
        fresh_until="2099-01-01T00:00:00Z",
    )
    assert _cmd_ingest(ingest_args) == 0
    capsys.readouterr()

    args = Namespace(
        base_dir=str(tmp_path / ".es"),
        view_name="active-scopes",
        thread_id="demo",
        owner_agent_id="",
        include_closed=False,
        limit=50,
        json=True,
    )
    assert _cmd_view(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["view"] == "active_scopes"
    assert payload["rows"][0]["scope_id"] == "release-gate"


def test_cli_reconcile_reports_unsupported_without_hook(tmp_path: Path, capsys) -> None:
    args = Namespace(
        base_dir=str(tmp_path / ".es"),
        thread_id="demo",
        limit=50,
        json=True,
    )
    assert _cmd_reconcile(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "unsupported", "reason": "no_reconcile_hook"}
