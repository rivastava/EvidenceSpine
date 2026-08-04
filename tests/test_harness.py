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
    manifest = build_manifest(executable="evidencespine", base_dir=".es")
    assert manifest["name"] == "evidencespine"
    hooks = manifest["hooks"]
    assert set(hooks) == {"SessionStart", "PreCompact", "Stop"}
    assert hooks["SessionStart"]["hooks"][0]["type"] == "command"
    assert "session-start" in hooks["SessionStart"]["hooks"][0]["command"]
    assert "precompact" in hooks["PreCompact"]["hooks"][0]["command"]
    assert "session-stop" in hooks["Stop"]["hooks"][0]["command"]


def test_opencode_plugin_ts_registers_delivery_hooks() -> None:
    source = build_plugin_ts(executable="/usr/bin/evidencespine", base_dir=".es")
    assert "type: \"local\"" in source
    assert '"evidencespine"' in source
    assert '"experimental.chat.system.transform"' in source
    assert '"experimental.session.compacting"' in source
    assert 'event.type === "session.deleted"' in source
    assert "session-start" in source
    assert "session-stop" in source
    assert "compaction" in source
    assert "command: executable" in source
    assert 'Plugin = async ({ $ })' in source
    assert source.count("${executable}") == 3


def test_opencode_plugin_template_prewarms_session_start() -> None:
    """Cold-start regression guard: the render path must never spawn the CLI.

    The TUI glitch was caused by the plugin spawning the python session-start
    inside the first-message pipeline (chat.system.transform). The template
    must pre-warm at plugin load and the transform must await the warmup
    promise — never call runSessionStart() or spawn `$` itself.
    """
    source = build_plugin_ts(executable="/usr/bin/evidencespine", base_dir=".es")
    assert "const warmup = runSessionStart()" in source, "session-start must be pre-warmed at plugin load"
    assert 'cached = warmup' in source, "transform must await the pre-warmed promise"
    assert source.count("runSessionStart()") == 2, "definition + exactly one call (the warmup)"

    transform = source.split('"experimental.chat.system.transform"', 1)[1].split('"experimental.session.compacting"', 1)[0]
    assert "runSessionStart()" not in transform, "transform must not spawn session-start"
    assert "$\u0060" not in transform, "transform must not shell out at all"


def test_opencode_plugin_template_has_no_render_side_effects() -> None:
    source = build_plugin_ts(executable="/usr/bin/evidencespine", base_dir=".es")
    session_start = source.split("async function runSessionStart()", 1)[1].split("async function runSessionStop()", 1)[0]
    assert "ingest" not in session_start and "emit" not in session_start, (
        "session-start command must be render-safe (reads briefs, no writes)"
    )


def test_cursor_mcp_json_shape() -> None:
    payload = build_cursor_mcp_json(executable="evidencespine", base_dir=".es")
    server = payload["mcpServers"]["evidencespine"]
    assert server["command"] == "evidencespine"
    assert server["args"] == ["mcp", "--base-dir", ".es"]


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
    assert written == {
        str(tmp_path / ".claude-plugin" / "plugin.json"),
        str(tmp_path / ".opencode" / "plugins" / "evidencespine.ts"),
        str(tmp_path / ".cursor" / "mcp.json"),
    }


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
