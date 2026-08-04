from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path


from evidencespine.cli import _cmd_ingest, _cmd_mcp, _cmd_migrate, _cmd_reconcile, _cmd_view, build_parser
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


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
    settings = EvidenceSpineSettings.from_env(base_dir=str(tmp_path / ".es"))
    rt = AgentMemoryRuntime(config=settings.to_runtime_config())
    events = list(rt.store.iter_events())
    rt.store.close()
    return events[-1]


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


def test_cli_ingest_jsonl_then_migrate_to_sqlite(tmp_path: Path, capsys) -> None:
    base_dir = str(tmp_path / ".es")
    args = _ingest_args(tmp_path)
    args.storage_format = "jsonl"
    assert _cmd_ingest(args) == 0
    assert (tmp_path / ".es" / "events.jsonl").exists()

    capsys.readouterr()
    migrate_args = Namespace(
        base_dir=base_dir,
        source_format="jsonl",
        target_format="sqlite",
        verify=True,
    )
    assert _cmd_migrate(migrate_args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["events_copied"] >= 1
    assert payload["verify"]["events_match"] is True


def test_cli_migrate_noop_when_formats_match(tmp_path: Path, capsys) -> None:
    migrate_args = Namespace(
        base_dir=str(tmp_path / ".es"),
        source_format="sqlite",
        target_format="sqlite",
        verify=True,
    )
    assert _cmd_migrate(migrate_args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "noop", "reason": "source_format_equals_target_format"}


def test_cli_build_parser_has_mcp_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["mcp"])
    assert args.command == "mcp"
    assert args.transport == "stdio"
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.path == "/mcp"


def test_cli_mcp_reports_missing_extra(tmp_path: Path, capsys, monkeypatch) -> None:
    def _block_import(name, *args, **kwargs):
        if name == "mcp":
            raise ImportError("No module named 'mcp'")
        return _orig_import(name, *args, **kwargs)

    _orig_import = __import__
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setattr("builtins.__import__", _block_import)

    args = Namespace(
        base_dir=str(tmp_path / ".es"),
        storage_format=None,
        transport="stdio",
        host="127.0.0.1",
        port=8000,
        path="/mcp",
    )
    assert _cmd_mcp(args) == 2
    assert "[mcp] extra is not installed" in capsys.readouterr().out
