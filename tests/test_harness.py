from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidencespine.harness.claude_code import (
    build_hooks_config,
    build_hooks_json,
    build_manifest,
    build_mcp_json,
    build_skill_markdown,
    merge_mcp_json,
)
from evidencespine.harness.codex import build_hooks_json as codex_hooks_json
from evidencespine.harness.codex import merge_config_toml, render_config_toml
from evidencespine.harness.commands import cmd_debug, cmd_install, cmd_session_start, cmd_session_stop
from evidencespine.harness.hooks import (
    CONTEXT_CHAR_LIMIT,
    _cap_context,
    _coerce_summary,
    extract_thread_id,
    handle_precompact,
    handle_session_stop,
)
from evidencespine.harness.opencode import build_plugin_ts
from evidencespine.runtime import AgentMemoryRuntime
from evidencespine.settings import EvidenceSpineSettings


def _runtime(base_dir: Path) -> AgentMemoryRuntime:
    settings = EvidenceSpineSettings.from_env(base_dir=str(base_dir / ".es"))
    return AgentMemoryRuntime(config=settings.to_runtime_config())


def test_claude_code_manifest_and_hooks_shape() -> None:
    manifest = build_manifest(executable="evidencespine", base_dir=".es")
    assert manifest["name"] == "evidencespine"
    assert manifest["author"] == {"name": "EvidenceSpine Contributors"}
    assert manifest["version"]
    # Component paths are the spec defaults, so no pointer fields are needed.
    assert "hooks" not in manifest and "mcpServers" not in manifest

    hooks_payload = build_hooks_json(executable="evidencespine", base_dir=".es")
    assert "description" in hooks_payload
    hooks = hooks_payload["hooks"]
    assert set(hooks) == {"SessionStart", "SessionEnd", "PreCompact"}
    start_handler = hooks["SessionStart"][0]["hooks"][0]
    # Exec form: command + args (no POSIX shell quoting; portable on Windows).
    assert start_handler["type"] == "command"
    assert start_handler["command"] == "evidencespine"
    assert start_handler["args"][:3] == ["harness", "claude-code", "session-start"]
    assert "--base-dir" in start_handler["args"]
    assert "${CLAUDE_PROJECT_DIR:-.}/.es" == start_handler["args"][-1]
    assert hooks["SessionStart"][0]["matcher"] == "startup|resume|clear|compact|fork"
    assert hooks["PreCompact"][0]["matcher"] == "manual|auto"

    config = build_hooks_config(executable="evidencespine", base_dir=".es")
    assert set(config["hooks"]) == {"SessionStart", "SessionEnd", "PreCompact"}


def test_claude_mcp_json_uses_project_dir_placeholder() -> None:
    import sys

    payload = build_mcp_json(executable=f"{sys.executable} -m evidencespine.cli", base_dir=".es")
    server = payload["mcpServers"]["evidencespine"]
    assert server["command"] == sys.executable
    assert server["args"] == ["-m", "evidencespine.cli", "mcp", "--base-dir", "${CLAUDE_PROJECT_DIR:-.}/.es"]
    merged = merge_mcp_json({"mcpServers": {"other": {"command": "x"}}}, executable="evidencespine", base_dir=".es")
    assert "other" in merged["mcpServers"] and "evidencespine" in merged["mcpServers"]


def test_claude_skill_and_rules_are_short_and_factual() -> None:
    skill = build_skill_markdown()
    assert skill.startswith("---\nname: evidencespine\n")
    assert "description:" in skill
    assert "EvidenceSpine" in skill
    from evidencespine.harness.claude_code import build_rules_markdown

    rules = build_rules_markdown()
    assert len(rules.splitlines()) < 60, "Claude rules file must stay short"
    assert "CLAUDE.md" not in rules


def test_claude_precompact_is_plaintext_not_envelope() -> None:
    from evidencespine.harness.claude_code import precompact_envelope

    note = "# Agent Context Brief: demo\n\nstate"
    assert precompact_envelope(note) == note


def test_opencode_plugin_ts_registers_v2_delivery_contract() -> None:
    source = build_plugin_ts(executable="/usr/bin/evidencespine", base_dir=".es")
    # v2 loader contract: bare default export with id + setup.
    assert "export default plugin" in source
    assert 'id: "evidencespine"' in source
    assert "async setup(ctx" in source
    assert 'import type { Plugin } from "@opencode/plugin"' in source
    assert "ctx.mcp.transform" in source
    assert 'ctx.session.hook("context",' in source
    assert 'ctx.session.hook("compaction",' in source
    assert 'ctx.tool.hook("execute.after",' in source
    assert "ctx.event.subscribe" in source
    assert "session.deleted" in source
    assert "session.compaction.ended" in source
    assert "session.compacted" not in source
    assert 'type === "session.idle"' not in source
    # No v1 surface and no value import / Plugin.define.
    assert "Plugin.define" not in source
    assert "experimental." not in source
    assert '"tool.execute.after"' not in source
    # Thread identity: session id is metadata; no per-session thread ids.
    assert '"--thread-id"' not in source
    assert '"--session-id"' in source
    # Fail-open resolves to empty (never poisons model context).
    assert "console.warn" in source
    assert 'resolve("")' in source
    assert '{ type: "text", text:' in source


def test_opencode_plugin_prewarm_and_compaction_helpers() -> None:
    source = build_plugin_ts(executable="/usr/bin/evidencespine", base_dir=".es")
    assert 'runSessionStart("default")' in source
    assert source.count('runSessionStart("default")') == 1
    assert "function briefForSession(sessionID" in source

    context_hook = source.split('ctx.session.hook("context"', 1)[1].split('ctx.session.hook("compaction"', 1)[0]
    compaction_hook = source.split('ctx.session.hook("compaction"', 1)[1].split('ctx.tool.hook(', 1)[0]
    for hook in (context_hook, compaction_hook):
        assert "briefForSession(" in hook
        assert "execFile" not in hook, "hooks must await the memoized brief, not spawn processes"


def test_codex_config_toml_renders_parseable_toml() -> None:
    tomllib = pytest.importorskip("tomllib")

    text = render_config_toml("evidencespine", ".es")
    data = tomllib.loads(text)
    assert data["features"]["hooks"] is True
    assert data["mcp_servers"]["evidencespine"]["command"] == "evidencespine"
    assert data["mcp_servers"]["evidencespine"]["args"] == ["mcp", "--base-dir", ".es"]


def test_codex_config_merge_never_duplicates_features_table() -> None:
    tomllib = pytest.importorskip("tomllib")

    existing = '[features]\nmemories = true\n\n[mcp_servers.evidencespine]\ncommand = "/x"\nargs = []\n'
    merged = merge_config_toml(existing, executable="/bin/evidencespine", base_dir=".es")
    assert merged["status"] == "ok"
    text = merged["text"]
    assert text.count("[features]") == 1
    data = tomllib.loads(text)
    assert data["features"] == {"memories": True, "hooks": True}
    assert data["mcp_servers"]["evidencespine"]["command"] == "/x", "existing server table preserved"


def test_codex_config_merge_inserts_features_inside_existing_table() -> None:
    tomllib = pytest.importorskip("tomllib")

    merged = merge_config_toml("[features]\nmemories = true\n", executable="evidencespine", base_dir=".es")
    text = merged["text"]
    assert text.count("[features]") == 1
    assert tomllib.loads(text)["features"]["hooks"] is True
    assert tomllib.loads(text)["mcp_servers"]["evidencespine"]["command"] == "evidencespine"


def test_codex_config_merge_flips_hooks_false() -> None:
    tomllib = pytest.importorskip("tomllib")

    merged = merge_config_toml("[features]\nhooks = false\n", executable="evidencespine", base_dir=".es")
    assert tomllib.loads(merged["text"])["features"]["hooks"] is True
    assert merged["features"]["action"] == "updated_false_to_true"


def test_codex_config_merge_refuses_unparseable_file() -> None:
    broken = "[mcp_servers\nbroken = "
    merged = merge_config_toml(broken, executable="evidencespine", base_dir=".es")
    # "unvalidated" is returned when tomllib is unavailable (Python < 3.11);
    # either way the existing file must never be modified.
    assert merged["status"] in {"invalid", "unvalidated"}
    assert merged["text"] == broken


def test_codex_hooks_json_precompact_has_no_additional_context_envelope() -> None:
    payload = codex_hooks_json(executable="evidencespine", base_dir=".es")
    assert set(payload["hooks"]) == {"SessionStart", "SessionEnd", "PreCompact"}
    start = payload["hooks"]["SessionStart"][0]
    assert start["matcher"] == "", "empty matcher covers compact/fork/resume sources"
    assert start["hooks"][0]["additionalContextLimit"] == 10000
    precompact = payload["hooks"]["PreCompact"][0]["hooks"][0]
    assert "additionalContext" not in json.dumps(precompact)


def test_hook_thread_identity_is_workspace_scoped_not_session_scoped() -> None:
    first = extract_thread_id(None, {"session_id": "uuid-1", "cwd": "/tmp/project-a"})
    second = extract_thread_id(None, {"session_id": "uuid-2", "cwd": "/tmp/project-a"})
    other = extract_thread_id(None, {"session_id": "uuid-1", "cwd": "/tmp/project-b"})
    assert first == second, "same workspace must resolve the same thread across sessions"
    assert first != other
    assert first.startswith("proj_")
    assert extract_thread_id("explicit", {"session_id": "uuid"}) == "explicit"


def test_coerce_summary_ignores_provider_control_fields() -> None:
    assert _coerce_summary(None, {"reason": "prompt_input_exit"}) == ""
    assert _coerce_summary(None, {"trigger": "auto"}) == ""
    assert _coerce_summary(None, {"source": "startup"}) == ""
    assert _coerce_summary(None, {"trigger": "manual", "custom_instructions": None}) == ""
    assert _coerce_summary(None, {"compact_summary": "real summary"}) == "real summary"
    assert _coerce_summary(None, {"summary": "plain summary"}) == "plain summary"
    assert _coerce_summary("explicit wins", {"summary": "ignored"}) == "explicit wins"


def test_context_cap_is_10000_chars() -> None:
    assert CONTEXT_CHAR_LIMIT == 10000
    capped = _cap_context("x" * 20000)
    assert len(capped) <= CONTEXT_CHAR_LIMIT
    assert "truncated" in capped


def test_session_start_returns_bounded_plaintext_brief(tmp_path: Path) -> None:
    text = cmd_session_start(thread_id="demo", objective="deploy", base_dir=str(tmp_path / ".es"))
    assert isinstance(text, str)
    assert not text.lstrip().startswith("{")
    assert "# Agent Context Brief: demo" in text
    assert "Project memory from EvidenceSpine" in text
    assert "deploy" in text
    assert len(text) <= 10000


def test_session_stop_records_event_and_optional_handoff(tmp_path: Path) -> None:
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
    assert any(e.get("payload", {}).get("claim") == "session ended" for e in events)
    rt.store.close()


def test_session_stop_reason_is_metadata_never_outcome(tmp_path: Path) -> None:
    payload = handle_session_stop(
        input_json={"reason": "clear", "session_id": "sess-1", "cwd": str(tmp_path)},
        base_dir=str(tmp_path / ".es"),
    )
    assert payload["status"] == "ok"
    rt = _runtime(tmp_path)
    events = list(rt.store.iter_events())
    stop_event = events[-1]
    assert stop_event["metadata"]["reason"] == "clear"
    assert stop_event["metadata"]["session_id"] == "sess-1"
    outcome = str(stop_event["payload"].get("outcome", ""))
    assert outcome != "clear", "reason must never become the stored summary"
    rt.store.close()


def test_precompact_persists_real_summary_and_returns_plaintext(tmp_path: Path) -> None:
    result = handle_precompact(
        thread_id="demo",
        input_json={"custom_instructions": "keep the sqlite decision", "trigger": "manual"},
        base_dir=str(tmp_path / ".es"),
    )
    assert "# Agent Context Brief: demo" in result
    assert not result.lstrip().startswith("{")

    rt = _runtime(tmp_path)
    events = list(rt.store.iter_events())
    stored = [e for e in events if "compaction" in str(e.get("payload", {}).get("claim", ""))]
    assert stored
    assert stored[-1]["payload"]["outcome"] == "keep the sqlite decision"
    rt.store.close()


def test_precompact_auto_trigger_stores_no_outcome(tmp_path: Path) -> None:
    handle_precompact(
        thread_id="demo",
        input_json={"trigger": "auto", "custom_instructions": None},
        base_dir=str(tmp_path / ".es"),
    )
    rt = _runtime(tmp_path)
    events = list(rt.store.iter_events())
    assert not any(str(e.get("payload", {}).get("outcome", "")) in {"auto", "manual"} for e in events)
    rt.store.close()


def test_session_start_fails_open_without_raising(tmp_path: Path) -> None:
    text = cmd_session_start(thread_id="demo", base_dir=str(tmp_path / "does" / "not" / "exist"))
    assert "EvidenceSpine" in text


def test_install_writes_claude_code_plugin_layout(tmp_path: Path) -> None:
    result = cmd_install(
        harness="claude-code",
        target_dir=str(tmp_path),
        base_dir=".es",
        executable="evidencespine",
    )
    assert result["status"] == "ok"
    plugin = tmp_path / ".claude" / "skills" / "evidencespine"
    for relative in (
        ".claude-plugin/plugin.json",
        "hooks/hooks.json",
        ".mcp.json",
        "skills/evidencespine/SKILL.md",
        "README.md",
    ):
        assert (plugin / relative).exists(), f"missing plugin file {relative}"
    assert (tmp_path / ".claude" / "rules" / "evidencespine.md").exists()
    # No repo-root plugin pollution and no settings hooks by default.
    assert not (tmp_path / ".claude-plugin").exists()
    assert not (tmp_path / "hooks").exists()
    assert not (tmp_path / ".claude" / "settings.json").exists()
    written = set(result["wrote"])
    for expected in (
        str(plugin / ".claude-plugin" / "plugin.json"),
        str(plugin / "hooks" / "hooks.json"),
        str(plugin / ".mcp.json"),
    ):
        assert expected in written
    assert result["validation"]["status"] in {"passed", "skipped"}


def test_install_codex_writes_config_and_hooks(tmp_path: Path) -> None:
    tomllib = pytest.importorskip("tomllib")

    result = cmd_install(harness="codex", target_dir=str(tmp_path), base_dir=".es", executable="evidencespine")
    assert result["status"] == "ok"
    config = (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")
    data = tomllib.loads(config)
    assert data["features"]["hooks"] is True
    assert data["mcp_servers"]["evidencespine"]["command"] == "evidencespine"
    hooks = json.loads((tmp_path / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    assert set(hooks["hooks"]) == {"SessionStart", "SessionEnd", "PreCompact"}
    assert hooks["hooks"]["SessionStart"][0]["hooks"][0]["timeout"] == 30


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


def test_debug_harness_reports_health_and_clients(tmp_path: Path) -> None:
    cmd_session_stop(thread_id="demo", summary="debug seed", base_dir=str(tmp_path / ".es"))
    payload = cmd_debug(base_dir=str(tmp_path / ".es"), thread_id="demo", target_dir=str(tmp_path))
    assert payload["status"] == "ok"
    assert payload["events_total"] >= 1
    assert payload["brief_ok"] is True
    assert payload["thread_id"] == "demo", "explicit debug thread is honored"
    assert "client_configs" in payload
    assert "windsurf" in payload["client_configs"]
    from evidencespine import __version__

    assert payload["evidencespine_version"] == __version__
