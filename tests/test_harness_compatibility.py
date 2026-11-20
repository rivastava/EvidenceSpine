"""Compatibility regression suite for the universal harness layer.

Covers the research-driven contracts: Claude Code skills-directory plugin
layout + settings-hook opt-in + plaintext PreCompact + validate reporting;
Codex safe TOML merge + append-merged hooks + HTTP MCP; opencode V2 compaction
hook + thread identity; non-destructive JSON merges; lazy dsh/generic/skills
dispatch; managed AGENTS.md blocks; git hook argv/execution; universal
print-config / doctor surfaces; and CLI exit-code correctness.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from evidencespine import cli
from evidencespine.harness import claude_code
from evidencespine.harness.install import (
    JsonDocument,
    install_harness,
    merge_agents_md,
    read_json_document,
    render_print_config,
    doctor_harness,
)
from evidencespine.harness.hooks import handle_session_start, handle_session_stop
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def _runtime(base_dir: Path) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(base_dir / ".es"))
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def _main(monkeypatch, argv) -> int:
    monkeypatch.setattr(sys, "argv", ["evidencespine", *argv])
    return cli.main()


def _json_from_stdout(capsys) -> Dict[str, Any]:
    out = capsys.readouterr().out
    start = out.find("{")
    assert start >= 0, f"no JSON in output: {out!r}"
    return json.loads(out[start:])


# ---------------------------------------------------------------------------
# Non-destructive JSON reads / merges
# ---------------------------------------------------------------------------


def test_read_json_document_distinguishes_missing_invalid_and_valid(tmp_path: Path) -> None:
    missing = read_json_document(tmp_path / "absent.json")
    assert isinstance(missing, JsonDocument)
    assert missing.status == "missing" and missing.valid and missing.value is None

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    invalid = read_json_document(bad)
    assert invalid.status == "invalid" and not invalid.valid
    assert invalid.backup and Path(invalid.backup).exists()
    assert bad.read_text(encoding="utf-8") == "{not json", "source must never be overwritten"

    good = tmp_path / "good.json"
    good.write_text('{"a": 1}', encoding="utf-8")
    assert read_json_document(good).value == {"a": 1}


def test_claude_settings_hooks_are_opt_in_and_appended(tmp_path: Path) -> None:
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps({"hooks": {"SessionStart": [{"matcher": "user", "hooks": [{"type": "command", "command": "user-cmd"}]}]}}),
        encoding="utf-8",
    )
    before = settings_path.read_text(encoding="utf-8")
    # Default install must not touch settings.json (plugin hooks are the delivery).
    install_harness(harness="claude-code", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    assert settings_path.read_text(encoding="utf-8") == before
    assert len(json.loads(before)["hooks"]["SessionStart"]) == 1

    result = install_harness(
        harness="claude-code",
        target_dir=str(tmp_path),
        base_dir=".es",
        executable="evidencespine",
        with_settings_hooks=True,
    )
    assert "SessionStart" in result["settings_hooks"]["merged_events"]
    merged = json.loads(settings_path.read_text(encoding="utf-8"))
    assert len(merged["hooks"]["SessionStart"]) == 2
    assert merged["hooks"]["SessionStart"][0]["hooks"][0]["command"] == "user-cmd", "user group preserved"
    assert "evidencespine" in json.dumps(merged["hooks"]["SessionStart"][1])

    # Idempotent: re-running must not duplicate the EvidenceSpine handler.
    install_harness(
        harness="claude-code",
        target_dir=str(tmp_path),
        base_dir=".es",
        executable="evidencespine",
        with_settings_hooks=True,
    )
    again = json.loads(settings_path.read_text(encoding="utf-8"))
    assert len(again["hooks"]["SessionStart"]) == 2


def test_unparseable_settings_are_preserved_with_backup(tmp_path: Path) -> None:
    settings_path = tmp_path / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("{bad json", encoding="utf-8")
    result = install_harness(
        harness="claude-code",
        target_dir=str(tmp_path),
        base_dir=".es",
        executable="evidencespine",
        with_settings_hooks=True,
    )
    assert result["status"] == "partial"
    assert settings_path.read_text(encoding="utf-8") == "{bad json"
    assert (tmp_path / ".claude" / "settings.json.bak").exists()
    assert any("untouched" in warning for warning in result["warnings"])


def test_claude_validation_result_is_reported(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(claude_code.shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    monkeypatch.setattr(
        claude_code.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="Validation passed", stderr=""),
    )
    result = install_harness(harness="claude-code", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    assert result["validation"]["status"] == "passed"
    assert result["validation"]["returncode"] == 0

    monkeypatch.setattr(claude_code.shutil, "which", lambda name: None)
    skipped = install_harness(harness="claude-code", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    assert skipped["validation"] == {"status": "skipped", "reason": "claude_cli_not_found"}


def test_agents_md_managed_block_merge_is_idempotent(tmp_path: Path) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# Project notes\n\nKeep this line.\n", encoding="utf-8")
    first = install_harness(harness="agents-md", target_dir=str(tmp_path))
    text = agents.read_text(encoding="utf-8")
    assert first["changed"] is True
    assert "Keep this line." in text, "user content preserved"
    assert text.count("<!-- evidencespine:begin -->") == 1
    assert text.count("<!-- evidencespine:end -->") == 1
    second = install_harness(harness="agents-md", target_dir=str(tmp_path))
    assert second["changed"] is False
    assert agents.read_text(encoding="utf-8") == text
    # Replacing the managed block (not appending) when it already exists.
    replaced = merge_agents_md(text.replace("Keep this line.", "Keep this line too."))
    assert replaced.count("<!-- evidencespine:begin -->") == 1


def test_vscode_copilot_writes_portable_mcp_shape(tmp_path: Path) -> None:
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"other": {"command": "x", "args": []}}}),
        encoding="utf-8",
    )
    result = install_harness(harness="vscode-copilot", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    assert result["status"] == "ok"
    payload = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    assert "other" in payload["mcpServers"], "existing servers preserved"
    entry = payload["mcpServers"]["evidencespine"]
    assert entry["type"] == "stdio"
    assert entry["args"] == ["mcp", "--base-dir", ".es"]
    # Legacy target still writes the deprecated "servers" shape.
    install_harness(harness="vscode", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    legacy = json.loads((tmp_path / ".vscode" / "mcp.json").read_text(encoding="utf-8"))
    assert "servers" in legacy and "mcpServers" not in legacy


# ---------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------


def test_codex_install_regression_duplicate_features_table(tmp_path: Path) -> None:
    tomllib = pytest.importorskip("tomllib")

    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir(parents=True)
    (codex_dir / "config.toml").write_text(
        "[features]\nmemories = true\n\n[mcp_servers.evidencespine]\ncommand = \"/x\"\nargs = []\n",
        encoding="utf-8",
    )
    result = install_harness(harness="codex", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    assert result["status"] == "ok"
    text = (codex_dir / "config.toml").read_text(encoding="utf-8")
    assert text.count("[features]") == 1, "never append a second [features] table"
    data = tomllib.loads(text), "merged config must remain parseable TOML"
    data = data[0]
    assert data["features"] == {"memories": True, "hooks": True}
    assert data["mcp_servers"]["evidencespine"]["command"] == "/x"

    # Second run is a no-op (byte-stable file).
    install_harness(harness="codex", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    assert (codex_dir / "config.toml").read_text(encoding="utf-8") == text


def test_codex_install_updates_hooks_false_and_keeps_user_toml(tmp_path: Path) -> None:
    tomllib = pytest.importorskip("tomllib")

    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir(parents=True)
    (codex_dir / "config.toml").write_text(
        "model = \"gpt-5\"\n\n[features]\nhooks = false\n\n[tui]\nnotifications = true\n",
        encoding="utf-8",
    )
    result = install_harness(harness="codex", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    assert result["status"] == "ok"
    text = (codex_dir / "config.toml").read_text(encoding="utf-8")
    data = tomllib.loads(text)
    assert data["features"]["hooks"] is True
    assert data["model"] == "gpt-5"
    assert data["tui"]["notifications"] is True


def test_codex_install_invalid_toml_is_never_overwritten(tmp_path: Path) -> None:
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir(parents=True)
    broken = "[mcp_servers\nbroken = "
    (codex_dir / "config.toml").write_text(broken, encoding="utf-8")
    result = install_harness(harness="codex", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    assert result["status"] == "partial"
    assert (codex_dir / "config.toml").read_text(encoding="utf-8") == broken
    assert any("untouched" in warning for warning in result["warnings"])


def test_codex_merge_is_fail_safe_when_toml_cannot_be_validated(monkeypatch, tmp_path: Path) -> None:
    """Python < 3.11 has no tomllib: never modify an existing config.toml we cannot validate."""
    from evidencespine.harness import codex

    monkeypatch.setattr(codex, "parse_toml", lambda text: (None, ""))
    original = "[features]\nhooks = true\n"
    merged = codex.merge_config_toml(original, executable="evidencespine", base_dir=".es")
    assert merged["status"] == "unvalidated"
    assert merged["text"] == original

    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir(parents=True)
    broken = "[mcp_servers\nbroken = "
    (codex_dir / "config.toml").write_text(broken, encoding="utf-8")
    result = install_harness(harness="codex", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    assert result["status"] == "partial"
    assert (codex_dir / "config.toml").read_text(encoding="utf-8") == broken

def test_codex_hooks_merge_appends_to_existing_groups(tmp_path: Path) -> None:
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir(parents=True)
    existing = {
        "hooks": {
            "SessionStart": [{"matcher": "", "hooks": [{"type": "command", "command": "user-hook"}]}],
            "Stop": [{"hooks": [{"type": "command", "command": "user-stop"}]}],
        }
    }
    (codex_dir / "hooks.json").write_text(json.dumps(existing), encoding="utf-8")
    install_harness(harness="codex", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    payload = json.loads((codex_dir / "hooks.json").read_text(encoding="utf-8"))
    assert "user-hook" in json.dumps(payload["hooks"]["SessionStart"]), "user group preserved"
    assert "user-stop" in json.dumps(payload["hooks"]["Stop"]), "unrelated events preserved"
    assert "evidencespine" in json.dumps(payload["hooks"]["SessionStart"])
    # Re-run: append-merged, never duplicated.
    install_harness(harness="codex", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    again = json.loads((codex_dir / "hooks.json").read_text(encoding="utf-8"))
    assert len(again["hooks"]["SessionStart"]) == 2
    assert len(again["hooks"]["Stop"]) == 1


def test_codex_http_mcp_transport(tmp_path: Path) -> None:
    tomllib = pytest.importorskip("tomllib")

    result = install_harness(
        harness="codex",
        target_dir=str(tmp_path),
        base_dir=".es",
        executable="evidencespine",
        mcp_url="http://127.0.0.1:8000/mcp",
    )
    assert result["status"] == "ok"
    assert result["transport"] == "http"
    data = tomllib.loads((tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8"))
    server = data["mcp_servers"]["evidencespine"]
    assert server["url"] == "http://127.0.0.1:8000/mcp"
    assert server["bearer_token_env_var"] == "EVIDENCESPINE_MCP_AUTH_TOKEN"
    assert "command" not in server


# ---------------------------------------------------------------------------
# Thread identity across sessions
# ---------------------------------------------------------------------------


def test_hooks_accumulate_one_thread_across_sessions(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    es = str(tmp_path / ".es")
    first = handle_session_start(input_json={"session_id": "uuid-1", "cwd": str(project)}, base_dir=es)
    second = handle_session_start(input_json={"session_id": "uuid-2", "cwd": str(project)}, base_dir=es)
    assert first == second or isinstance(first, str)
    handle_session_stop(
        input_json={"session_id": "uuid-1", "cwd": str(project), "reason": "prompt_input_exit"},
        base_dir=es,
    )
    handle_session_stop(
        input_json={"session_id": "uuid-2", "cwd": str(project), "reason": "clear"},
        base_dir=es,
    )
    rt = AgentMemoryRuntime(config=EvidenceSpineSettings.from_env(base_dir=es).to_runtime_config())
    events = list(rt.store.iter_events())
    thread_ids = {str(event.get("thread_id")) for event in events}
    assert len(thread_ids) == 1, f"all sessions must share one workspace thread: {thread_ids}"
    thread = thread_ids.pop()
    assert thread != "uuid-1" and thread != "uuid-2"
    assert thread.startswith("proj_")
    session_ids = {str(event.get("metadata", {}).get("session_id")) for event in events}
    assert session_ids == {"uuid-1", "uuid-2"}
    rt.store.close()


# ---------------------------------------------------------------------------
# Lazy dispatch (dsh / generic / instructions / skills.materialize)
# ---------------------------------------------------------------------------


def _fake_harness_modules(monkeypatch, calls) -> None:
    dsh = types.ModuleType("evidencespine.harness.dsh")
    generic = types.ModuleType("evidencespine.harness.generic")
    skills = types.ModuleType("evidencespine.harness.skills")

    def install_dsh(**kwargs):
        calls.append(("dsh", kwargs))
        return {"status": "ok", "harness": "dsh", "wrote": []}

    def install_generic(**kwargs):
        calls.append(("generic", kwargs))
        return {"status": "ok", "harness": "generic", "wrote": []}

    def install_instructions(**kwargs):
        calls.append(("instructions", kwargs))
        return {"status": "ok", "harness": "instructions", "wrote": []}

    def materialize(**kwargs):
        calls.append(("materialize", kwargs))
        return {"status": "ok", "harnesses": kwargs.get("harnesses")}

    dsh.install_dsh = install_dsh
    generic.install_generic = install_generic
    skills.install_instructions = install_instructions
    skills.materialize = materialize
    monkeypatch.setitem(sys.modules, "evidencespine.harness.dsh", dsh)
    monkeypatch.setitem(sys.modules, "evidencespine.harness.generic", generic)
    monkeypatch.setitem(sys.modules, "evidencespine.harness.skills", skills)


def test_lazy_harness_dispatch_calls_modules_and_materializes_skills(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    _fake_harness_modules(monkeypatch, calls)
    result = install_harness(harness="dsh", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    assert result["status"] == "ok" and result["harness"] == "dsh"
    assert result["skill"]["status"] == "ok"
    dispatched = [name for name, _ in calls]
    assert "dsh" in dispatched and "materialize" in dispatched
    materialize_kwargs = next(kwargs for name, kwargs in calls if name == "materialize")
    assert materialize_kwargs["harnesses"] == ["dsh"]

    generic_result = install_harness(harness="generic", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    assert generic_result["status"] == "ok"
    instructions = install_harness(harness="instructions", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    assert instructions["status"] == "ok"


def test_lazy_harness_missing_module_is_clean_skip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "evidencespine.harness.dsh", None)
    result = install_harness(harness="dsh", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    assert result["status"] == "skipped"
    assert result["reason"] == "not_installed"
    assert not any(tmp_path.iterdir()), "skipped dispatch must not write"


def test_install_all_is_tolerant_of_optional_modules(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    _fake_harness_modules(monkeypatch, calls)
    result = install_harness(harness="all", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    assert result["status"] in {"ok", "partial"}
    details = result["details"]
    harnesses = {item.get("harness") for item in details}
    assert {"claude-code", "codex", "opencode", "cursor", "vscode", "vscode-copilot", "agents-md", "dsh", "generic"} <= harnesses


def test_install_dry_run_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    _fake_harness_modules(monkeypatch, calls)
    result = install_harness(
        harness="claude-code",
        target_dir=str(tmp_path),
        base_dir=".es",
        executable="evidencespine",
        dry_run=True,
    )
    assert result["dry_run"] is True
    assert result["validation"] == {"status": "skipped", "reason": "dry_run"}
    assert not (tmp_path / ".claude").exists()
    assert not any(tmp_path.iterdir()), "dry run must not create files"


# ---------------------------------------------------------------------------
# print-config / doctor
# ---------------------------------------------------------------------------


def test_print_config_never_writes_and_reports_contract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "evidencespine.harness.generic", None)
    payload = render_print_config(client="claude-code", project_dir=str(tmp_path))
    assert payload["status"] == "ok"
    assert payload["path"].endswith(".mcp.json")
    assert "mcpServers" in payload["snippet"] and "evidencespine" in payload["snippet"]
    assert payload["verify"] and payload["grade"]
    assert payload["writes"] is False
    assert not any(tmp_path.iterdir()), "print-config must never write"

    http_payload = render_print_config(client="cursor", transport="http", project_dir=str(tmp_path))
    assert "http://127.0.0.1:8000/mcp" in http_payload["snippet"]
    assert "EVIDENCESPINE_MCP_AUTH_TOKEN" in http_payload["snippet"]

    codex_payload = render_print_config(client="codex", project_dir=str(tmp_path))
    assert "[mcp_servers.evidencespine]" in codex_payload["snippet"]

    all_payload = render_print_config(client="all", project_dir=str(tmp_path))
    assert all_payload["status"] == "ok" and len(all_payload["clients"]) > 10


def test_print_config_unknown_client_is_invalid(tmp_path: Path) -> None:
    payload = render_print_config(client="not-a-client", project_dir=str(tmp_path))
    assert payload["status"] == "invalid"
    assert "known" in payload


def test_doctor_reports_broken_json_and_auth_reminder(tmp_path: Path) -> None:
    cursor = tmp_path / ".cursor"
    cursor.mkdir()
    (cursor / "mcp.json").write_text("{bad json", encoding="utf-8")
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"evidencespine": {"url": "http://127.0.0.1:8000/mcp"}}}),
        encoding="utf-8",
    )
    payload = doctor_harness(target_dir=str(tmp_path))
    assert payload["status"] == "broken"
    assert any(check["status"] == "broken" for check in payload["checks"])
    assert payload["http_transport_detected"] is True
    assert "EVIDENCESPINE_MCP_AUTH_TOKEN" in payload["auth_reminder"]
    # Backups are left, source files untouched.
    assert (cursor / "mcp.json").read_text(encoding="utf-8") == "{bad json"

    clean = tmp_path / "clean"
    clean.mkdir()
    assert doctor_harness(target_dir=str(clean))["status"] == "ok"
