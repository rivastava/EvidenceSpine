from __future__ import annotations

import json
import sys
import types
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


# ---------------------------------------------------------------------------
# CLI correctness: exit codes, base-dir resolution, ground refs
# ---------------------------------------------------------------------------


def _cli_main(monkeypatch, argv) -> int:
    monkeypatch.setattr(sys, "argv", ["evidencespine", *argv])
    from evidencespine.cli import main

    return main()


def test_cli_verify_and_handoff_missing_exit_nonzero(tmp_path: Path, monkeypatch, capsys) -> None:
    base = str(tmp_path / ".es")
    assert _cli_main(
        monkeypatch,
        ["verify", "--base-dir", base, "--fact-id", "nope", "--method", "test", "--reference", "x", "--json"],
    ) == 2
    capsys.readouterr()
    assert _cli_main(monkeypatch, ["handoff", "--base-dir", base, "--import", str(tmp_path / "nope.json")]) == 2
    capsys.readouterr()


def test_cli_ingest_invalid_state_context_exits_nonzero(tmp_path: Path, monkeypatch, capsys) -> None:
    code = _cli_main(
        monkeypatch,
        [
            "ingest",
            "--base-dir",
            str(tmp_path / ".es"),
            "--thread-id",
            "t",
            "--event-type",
            "decision",
            "--claim",
            "c",
            "--state-kind",
            "pending_gate",
            "--status",
            "active",
            "--json",
        ],
    )
    assert code == 2
    payload = _json_tail(capsys.readouterr().out)
    assert payload["status"] == "invalid"


def test_cli_harness_install_git_outside_repo_exits_nonzero(tmp_path: Path, monkeypatch, capsys) -> None:
    code = _cli_main(monkeypatch, ["harness", "install", "--harness", "git", "--target-dir", str(tmp_path)])
    assert code == 2
    payload = _json_tail(capsys.readouterr().out)
    assert payload["status"] == "invalid"


def test_cli_git_hook_unknown_sha_exits_nonzero(tmp_path: Path, monkeypatch, capsys) -> None:
    code = _cli_main(
        monkeypatch,
        [
            "harness",
            "git",
            "git-hook",
            "--base-dir",
            str(tmp_path / ".es"),
            "--sha",
            "deadbeef" * 5,
            "--repo-dir",
            str(tmp_path),
        ],
    )
    assert code == 2
    payload = _json_tail(capsys.readouterr().out)
    assert payload["status"] == "invalid"


def test_cli_ingest_ground_refs_accept_more_than_five_and_duplicates(tmp_path: Path, monkeypatch, capsys) -> None:
    ref_file = tmp_path / "refs.txt"
    ref_file.write_text("\n".join(f"line {i}" for i in range(1, 7)) + "\n", encoding="utf-8")
    base = str(tmp_path / ".es")
    argv = ["ingest", "--base-dir", base, "--thread-id", "t", "--event-type", "action", "--claim", "c", "--json"]
    argv += ["--source-root", str(tmp_path)]
    for ref in ["refs.txt#L1-L1", "refs.txt#L1-L1", "refs.txt#L2-L2", "refs.txt#L3-L3", "refs.txt#L4-L4", "refs.txt#L5-L5", "refs.txt#L6-L6"]:
        argv += ["--ground-ref", ref]
    assert _cli_main(monkeypatch, argv) == 0
    capsys.readouterr()
    rt = AgentMemoryRuntime(config=EvidenceSpineSettings.from_env(base_dir=base).to_runtime_config())
    event = list(rt.store.iter_events())[-1]
    rt.store.close()
    assert len(event.get("evidence_items", [])) == 6, "all unique refs grounded, duplicates deduped"


def test_cli_migrate_mismatch_exits_nonzero_and_no_verify_skips(tmp_path: Path, monkeypatch, capsys) -> None:
    base = str(tmp_path / ".es")
    assert _cli_main(
        monkeypatch,
        [
            "ingest",
            "--base-dir",
            base,
            "--storage-format",
            "jsonl",
            "--thread-id",
            "t",
            "--event-type",
            "decision",
            "--claim",
            "migrate me",
        ],
    ) == 0
    capsys.readouterr()
    events_path = tmp_path / ".es" / "events.jsonl"
    line = events_path.read_text(encoding="utf-8").splitlines()[0]
    events_path.write_text(events_path.read_text(encoding="utf-8") + line + "\n", encoding="utf-8")

    code = _cli_main(monkeypatch, ["migrate", "--base-dir", base, "--source-format", "jsonl", "--target-format", "sqlite"])
    payload = _json_tail(capsys.readouterr().out)
    assert code == 1, "verification mismatch must be a nonzero exit"
    assert payload["status"] == "mismatch"
    assert payload["verify"]["status"] == "mismatch"
    assert payload["report"]["copied"]["events"] == 1
    assert payload["report"]["skipped"]["events"] == 1

    code = _cli_main(
        monkeypatch,
        ["migrate", "--base-dir", base, "--source-format", "jsonl", "--target-format", "sqlite", "--no-verify"],
    )
    payload = _json_tail(capsys.readouterr().out)
    assert code == 0
    assert "verify" not in payload, "--no-verify must actually skip verification"
    assert payload["report"]["verified"] == "skipped"
    assert payload["verify_requested"] is False


def test_cli_chat_passes_none_base_dir_and_reports_backend_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    import evidencespine.chat as chat_module

    captured: dict = {}

    def fake_run_chat(**kwargs):
        captured.update(kwargs)
        return {"stopped_by": "backend_failure", "messages": []}

    monkeypatch.setattr(chat_module, "run_chat", fake_run_chat)
    code = _cli_main(
        monkeypatch,
        ["chat", "--llm", "scripted", "--max-messages", "2", "--quiet-secs", "1", "--interval", "0.1", "--max-wall-secs", "12"],
    )
    assert code == 3, "chat backend failure must be a nonzero exit"
    assert captured["base_dir"] is None, "None must reach settings/env, never the string 'None'"
    assert captured["max_wall_secs"] == 12.0
    assert not (Path.cwd() / "None").exists()


def test_cli_harness_doctor_and_print_config_exit_codes(tmp_path: Path, monkeypatch, capsys) -> None:
    assert _cli_main(monkeypatch, ["harness", "print-config", "--client", "codex", "--target-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "[mcp_servers.evidencespine]" in out
    assert not any(tmp_path.iterdir()), "print-config must not write"

    bad = tmp_path / ".cursor"
    bad.mkdir()
    (bad / "mcp.json").write_text("{broken", encoding="utf-8")
    assert _cli_main(monkeypatch, ["harness", "doctor", "--target-dir", str(tmp_path), "--base-dir", str(tmp_path / ".es")]) == 2
    payload = _json_tail(capsys.readouterr().out)
    assert payload["status"] == "broken"


def test_cli_harness_precompact_emits_plaintext(tmp_path: Path, monkeypatch, capsys) -> None:
    code = _cli_main(
        monkeypatch,
        ["harness", "claude-code", "precompact", "--thread-id", "t", "--base-dir", str(tmp_path / ".es")],
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "# Agent Context Brief: t" in out
    assert not out.lstrip().startswith("{"), "PreCompact must use the plain-text channel"


def test_cli_ingest_without_thread_id_resolves_workspace_thread(tmp_path: Path, monkeypatch, capsys) -> None:
    base = str(tmp_path / ".es")
    assert _cli_main(
        monkeypatch,
        ["ingest", "--base-dir", base, "--event-type", "decision", "--claim", "resolved", "--session-id", "sess-9"],
    ) == 0
    capsys.readouterr()
    rt = AgentMemoryRuntime(config=EvidenceSpineSettings.from_env(base_dir=base).to_runtime_config())
    event = list(rt.store.iter_events())[-1]
    rt.store.close()
    assert str(event["thread_id"]).startswith("proj_")
    assert event["metadata"]["session_id"] == "sess-9"
    assert event["source_turn_id"] == "sess-9"


def test_cli_mcp_and_a2a_value_error_is_clean_nonzero(tmp_path: Path, monkeypatch, capsys) -> None:
    fake_mcp = types.ModuleType("mcp")
    fake_server = types.ModuleType("evidencespine.mcp_server")

    def _raise(**kwargs):
        raise ValueError("non-loopback bind without token refused")

    fake_server.run_server = _raise
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "evidencespine.mcp_server", fake_server)
    code = _cli_main(monkeypatch, ["mcp", "--base-dir", str(tmp_path / ".es")])
    out = capsys.readouterr().out
    assert code == 2
    assert "error: non-loopback bind without token refused" in out
    assert "Traceback" not in out

    monkeypatch.setitem(sys.modules, "a2a", types.ModuleType("a2a"))
    fake_a2a = types.ModuleType("evidencespine.a2a")

    def _raise_a2a(**kwargs):
        raise ValueError("a2a auth required")

    fake_a2a.run_server = _raise_a2a
    monkeypatch.setitem(sys.modules, "evidencespine.a2a", fake_a2a)
    code = _cli_main(monkeypatch, ["a2a", "--base-dir", str(tmp_path / ".es")])
    out = capsys.readouterr().out
    assert code == 2
    assert "error: a2a auth required" in out


def test_cli_harness_choices_match_registry() -> None:
    from argparse import _SubParsersAction

    from evidencespine.cli import build_parser
    from evidencespine.harness.install import KNOWN_HARNESSES

    parser = build_parser()
    choices = []
    for action in parser._actions:
        if isinstance(action, _SubParsersAction):
            for subparser in action.choices.values():
                for sub_action in subparser._actions:
                    if isinstance(sub_action, _SubParsersAction):
                        for third in sub_action.choices.values():
                            for leaf in third._actions:
                                if getattr(leaf, "dest", "") == "harness_target":
                                    choices = [str(x) for x in leaf.choices]
    assert choices, "harness install --harness choices not found"
    assert set(choices) == set(KNOWN_HARNESSES)
    for expected in ("dsh", "generic", "instructions", "vscode-copilot"):
        assert expected in choices


def _json_tail(text: str) -> dict:
    start = text.find("{")
    assert start >= 0, f"no JSON in output: {text!r}"
    return json.loads(text[start:])
