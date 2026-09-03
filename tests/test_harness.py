from __future__ import annotations

import json
from pathlib import Path

from evidencespine.harness.claude_code import build_manifest
from evidencespine.harness.commands import cmd_debug, cmd_install, cmd_session_start, cmd_session_stop
from evidencespine.harness.hooks import handle_precompact, handle_session_stop
from evidencespine.harness.opencode import build_cursor_mcp_json, build_plugin_ts
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def _runtime(base_dir: Path) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(base_dir / ".es"))
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def test_claude_code_manifest_hooks_shape() -> None:
    from evidencespine.harness.claude_code import build_hooks_config, build_hooks_json, precompact_envelope

    manifest = build_manifest(executable="evidencespine", base_dir=".es")
    assert manifest["name"] == "evidencespine"
    # Manifest holds pointers; components resolve from the plugin root
    # (only plugin.json lives inside .claude-plugin/ per the spec).
    assert manifest["hooks"] == "./hooks/hooks.json"
    assert manifest["mcpServers"] == "./.mcp.json"
    assert manifest["author"] == {"name": "EvidenceSpine Contributors"}
    hooks_payload = build_hooks_json(executable="evidencespine", base_dir=".es")
    hooks = hooks_payload["hooks"]
    assert set(hooks) == {"SessionStart", "SessionEnd", "PreCompact"}
    assert isinstance(hooks["SessionStart"], list)
    assert hooks["SessionStart"][0]["hooks"][0]["type"] == "command"
    assert "session-start" in hooks["SessionStart"][0]["hooks"][0]["command"]
    assert "precompact" in hooks["PreCompact"][0]["hooks"][0]["command"]
    assert "session-stop" in hooks["SessionEnd"][0]["hooks"][0]["command"]
    # List-shape with matcher/timeout per code.claude.com/docs/en/hooks.
    assert hooks["SessionStart"][0]["matcher"] == "startup|resume|clear|compact|fork"
    assert hooks["SessionStart"][0]["hooks"][0]["timeout"] == 30
    config = build_hooks_config(executable="evidencespine", base_dir=".es")
    assert set(config["hooks"]) == {"SessionStart", "SessionEnd", "PreCompact"}
    envelope = json.loads(precompact_envelope("note"))
    assert envelope["hookSpecificOutput"]["hookEventName"] == "PreCompact"
    assert envelope["hookSpecificOutput"]["additionalContext"] == "note"


def test_opencode_plugin_ts_registers_delivery_hooks() -> None:
    source = build_plugin_ts(executable="/usr/bin/evidencespine", base_dir=".es")
    assert "type: \"local\"" in source
    assert '"evidencespine"' in source
    assert '"experimental.chat.system.transform"' in source
    assert '"experimental.session.compacting"' in source
    assert '"tool.execute.after"' in source
    assert "session.created" in source
    assert "session.deleted" in source
    assert "session-start" in source
    assert "session-stop" in source
    assert "compaction" in source
    assert "--thread-id" in source
    assert 'Plugin = async ({ $ })' in source
    # Fail-open resolves to empty (never poisons system context).
    assert "console.warn" in source
    assert 'return ""' in source
    assert 'output.system.push("\\n\\n" + (await cached))' not in source


def test_opencode_plugin_template_prewarms_session_start() -> None:
    """Cold-start regression guard: the render path must never spawn the CLI.

    The TUI glitch was caused by the plugin spawning the python session-start
    inside the first-message pipeline (chat.system.transform). The template
    must pre-warm at plugin load and the transform must await a memoized
    per-session promise — never spawn `$` itself.
    """
    source = build_plugin_ts(executable="/usr/bin/evidencespine", base_dir=".es")
    assert 'runSessionStart("default")' in source, "default session must be pre-warmed at plugin load"
    assert "function briefFor(sessionID" in source, "transform must use per-session memoization"
    assert source.count('runSessionStart("default")') == 1

    transform = source.split('"experimental.chat.system.transform"', 1)[1].split('"experimental.session.compacting"', 1)[0]
    assert "briefFor(sessionID)" in transform, "transform must await the memoized brief"
    assert "$\u0060" not in transform, "transform must not shell out at all"


def test_opencode_plugin_template_has_no_render_side_effects() -> None:
    source = build_plugin_ts(executable="/usr/bin/evidencespine", base_dir=".es")
    session_start = source.split("async function runSessionStart(sessionID", 1)[1].split("async function runSessionStop(", 1)[0]
    assert "ingest" not in session_start and "emit" not in session_start, (
        "session-start command must be render-safe (reads briefs, no writes)"
    )


def test_cursor_mcp_json_shape() -> None:
    payload = build_cursor_mcp_json(executable="evidencespine", base_dir=".es")
    server = payload["mcpServers"]["evidencespine"]
    assert server["command"] == "evidencespine"
    assert server["args"] == ["mcp", "--base-dir", ".es"]


def test_claude_mcp_json_merge_preserves_existing(tmp_path: Path) -> None:
    from evidencespine.harness.claude_code import build_mcp_json, merge_mcp_json

    payload = build_mcp_json(executable="/usr/bin/python -m evidencespine.cli", base_dir=".es")
    server = payload["mcpServers"]["evidencespine"]
    assert server["command"] == "/usr/bin/python"
    assert server["args"] == ["-m", "evidencespine.cli", "mcp", "--base-dir", ".es"]
    merged = merge_mcp_json({"mcpServers": {"other": {"command": "x"}}}, executable="evidencespine", base_dir=".es")
    assert "other" in merged["mcpServers"] and "evidencespine" in merged["mcpServers"]
    result = cmd_install(harness="claude-code", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    assert (tmp_path / ".mcp.json").exists()
    assert (tmp_path / "hooks" / "hooks.json").exists()
    assert str(tmp_path / ".mcp.json") in result["wrote"]


def test_cursor_hooks_json_has_version_and_thread_keys() -> None:
    from evidencespine.harness.cursor import build_hooks_json
    from evidencespine.harness.hooks import extract_thread_id

    payload = build_hooks_json(executable="evidencespine", base_dir=".es")
    assert payload["version"] == 1
    assert set(payload["hooks"]) == {"sessionStart", "preCompact", "stop"}
    # Cursor hook stdin carries conversation_id (cursor.com/docs/hooks).
    assert extract_thread_id(None, {"conversation_id": "conv-1"}) == "conv-1"
    assert extract_thread_id("", {"session_id": "sess-1"}) == "sess-1"


def test_cursor_mcp_json_splits_multitoken_executable() -> None:
    from evidencespine.harness.cursor import build_mcp_json, merge_mcp_json

    payload = build_mcp_json(executable="/usr/bin/python -m evidencespine.cli", base_dir=".es")
    server = payload["mcpServers"]["evidencespine"]
    assert server["command"] == "/usr/bin/python"
    assert server["args"] == ["-m", "evidencespine.cli", "mcp", "--base-dir", ".es"]
    merged = merge_mcp_json({"mcpServers": {"other": {"command": "x", "args": []}}}, executable="evidencespine", base_dir=".es")
    assert "other" in merged["mcpServers"] and "evidencespine" in merged["mcpServers"]


def test_opencode_plugin_ts_splits_multitoken_executable_for_mcp() -> None:
    mcp = '["/usr/bin/python", "-m", "evidencespine.cli", "mcp", "--base-dir", ".es"]'
    source = build_plugin_ts(
        executable="/usr/bin/python -m evidencespine.cli",
        base_dir=".es",
        mcp_command=mcp,
    )
    assert 'command: ["/usr/bin/python", "-m", "evidencespine.cli", "mcp", "--base-dir", ".es"]' in source
    assert '${executable} harness opencode session-start' in source


def test_session_start_returns_bounded_brief_markdown(tmp_path: Path) -> None:
    text = cmd_session_start(thread_id="demo", objective="deploy", base_dir=str(tmp_path / ".es"))
    assert "# Agent Context Brief: demo" in text
    assert "EvidenceSpine delivery" in text
    assert "deploy" in text


def test_session_stop_records_event_and_handoff(tmp_path: Path) -> None:
    text = cmd_session_stop(
        thread_id="demo",
        summary="wrapped up the deploy; sqlite migration done",
        auto_handoff=True,
        reason="user_stop",
        base_dir=str(tmp_path / ".es"),
        json_out=True,
    )
    payload = json.loads(text)
    assert payload["status"] == "ok"
    assert payload["ingest"]["status"] == "ok"
    assert payload["thread_id"] == "demo"
    assert payload["handoff"]["role"] == "auditor"

    rt = _runtime(tmp_path)
    events = list(rt.store.iter_events())
    assert any("session ended" in str(e.get("payload", {}).get("claim", "")) for e in events)
    rt.store.close()


def test_precompact_persists_summary_and_returns_state_note(tmp_path: Path) -> None:
    result = handle_precompact(
        thread_id="demo",
        summary="decided on sqlite backend",
        input_json={"conversation_summary": "ignored because explicit wins"},
        base_dir=str(tmp_path / ".es"),
    )
    assert "# Agent Context Brief: demo" in result

    rt = _runtime(tmp_path)
    events = list(rt.store.iter_events())
    assert any("compaction" in str(e.get("payload", {}).get("claim", "")) for e in events)
    rt.store.close()


def test_precompact_reads_summary_from_stdin_json(tmp_path: Path) -> None:
    result = handle_precompact(
        thread_id="demo",
        input_json={"conversation_summary": "from stdin summary"},
        base_dir=str(tmp_path / ".es"),
    )
    assert "# Agent Context Brief: demo" in result
    rt = _runtime(tmp_path)
    events = list(rt.store.iter_events())
    stored = [e for e in events if "compaction" in str(e.get("payload", {}).get("claim", ""))]
    assert stored
    assert stored[-1]["payload"]["outcome"] == "from stdin summary"
    rt.store.close()


def test_session_stop_fails_open_on_bad_base_dir(tmp_path: Path) -> None:
    payload = handle_session_stop(
        thread_id="demo",
        summary="x",
        base_dir=str(tmp_path / "does" / "not" / "exist"),
    )
    assert payload["status"] in {"ok", "fail_open"}


def test_session_start_fails_open_without_raising(tmp_path: Path) -> None:
    text = cmd_session_start(thread_id="demo", base_dir=str(tmp_path / "does" / "not" / "exist"))
    assert "EvidenceSpine" in text


def test_install_writes_harness_files(tmp_path: Path) -> None:
    result = cmd_install(harness="all", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    assert result["status"] == "ok"
    written = set(result["wrote"])
    # Core artifacts must exist (superset: install is merge-preserving + multi-harness).
    for expected in (
        str(tmp_path / ".claude-plugin" / "plugin.json"),
        str(tmp_path / "hooks" / "hooks.json"),
        str(tmp_path / ".mcp.json"),
        str(tmp_path / ".opencode" / "plugins" / "evidencespine.ts"),
        str(tmp_path / ".cursor" / "mcp.json"),
        str(tmp_path / ".vscode" / "mcp.json"),
        str(tmp_path / ".codex" / "config.toml"),
        str(tmp_path / ".codex" / "hooks.json"),
        str(tmp_path / "AGENTS.md"),
    ):
        assert expected in written, f"missing {expected} in {sorted(written)}"


def test_install_codex_writes_config_and_hooks(tmp_path: Path) -> None:
    from evidencespine.harness.codex import build_hooks_json, precompact_envelope

    result = cmd_install(harness="codex", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    assert result["status"] == "ok"
    config = (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")
    assert "[mcp_servers.evidencespine]" in config
    assert "[features]" in config and "hooks = true" in config
    hooks = json.loads((tmp_path / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    assert set(hooks["hooks"]) == {"SessionStart", "SessionEnd", "PreCompact"}
    assert hooks["hooks"]["SessionStart"][0]["hooks"][0]["timeout"] == 30
    envelope = json.loads(precompact_envelope("x"))
    assert envelope["hookSpecificOutput"]["hookEventName"] == "PreCompact"
    _ = build_hooks_json(executable="evidencespine", base_dir=".es")


def test_install_opencode_global_scope_writes_directly(tmp_path: Path) -> None:
    result = cmd_install(
        harness="opencode",
        target_dir=str(tmp_path),
        base_dir=".es",
        executable="evidencespine",
        scope="global",
    )
    assert result["status"] == "ok"
    assert result["scope"] == "global"
    assert result["wrote"] == [str(tmp_path / "evidencespine.ts")]
    assert (tmp_path / "evidencespine.ts").exists()
    assert not (tmp_path / ".opencode").exists()


def test_debug_harness_reports_health(tmp_path: Path) -> None:
    cmd_session_stop(thread_id="demo", summary="debug seed", base_dir=str(tmp_path / ".es"))
    payload = cmd_debug(base_dir=str(tmp_path / ".es"), thread_id="demo")
    assert payload["status"] == "ok"
    assert payload["events_total"] >= 1
    assert payload["brief_ok"] is True
    from evidencespine import __version__

    assert payload["evidencespine_version"] == __version__
