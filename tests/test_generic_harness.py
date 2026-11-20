"""Tests for the universal (generic) MCP installer and client renderer."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from evidencespine.harness import generic


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load(path: Path):
    return json.loads(_read(path))


def test_install_writes_canonical_and_projections(tmp_path: Path) -> None:
    result = generic.install_generic(target_dir=str(tmp_path))

    assert result["status"] == "ok"
    assert result["harness"] == "generic"
    assert result["action"] == "added"
    canonical = tmp_path / generic.CANONICAL_FILENAME
    assert canonical.exists()
    assert result["path"] == str(canonical)
    entry = _load(canonical)["mcpServers"]["evidencespine"]
    assert entry["command"] == "evidencespine"
    assert entry["args"] == ["mcp", "--base-dir", ".evidencespine"]
    assert entry["env"] == {}
    for rel in generic.DEFAULT_PROJECTIONS:
        path = tmp_path / rel
        assert path.exists(), f"missing projection {rel}"
        assert "evidencespine" in _load(path)["mcpServers"]
    notes = tmp_path / generic.NOTES_FILENAME
    assert notes.exists()
    assert len(_read(notes).splitlines()) <= generic.MAX_NOTES_LINES
    assert generic.CANONICAL_FILENAME in _read(notes)
    assert all(path in result["wrote"] for path in [str(canonical)])


def test_install_merges_existing_servers_and_keys(tmp_path: Path) -> None:
    canonical = tmp_path / generic.CANONICAL_FILENAME
    canonical.write_text(
        json.dumps({"mcpServers": {"other": {"command": "other-bin"}}, "customKey": True}),
        encoding="utf-8",
    )
    mcp_json = tmp_path / ".mcp.json"
    mcp_json.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}), encoding="utf-8")

    result = generic.install_generic(target_dir=str(tmp_path))

    payload = _load(canonical)
    assert result["action"] == "updated"
    assert payload["customKey"] is True
    assert payload["mcpServers"]["other"] == {"command": "other-bin"}
    assert payload["mcpServers"]["evidencespine"]["command"] == "evidencespine"
    assert _load(mcp_json)["mcpServers"]["other"] == {"command": "x"}


def test_second_install_is_idempotent(tmp_path: Path) -> None:
    first = generic.install_generic(target_dir=str(tmp_path))
    before = _read(Path(first["path"]))
    second = generic.install_generic(target_dir=str(tmp_path))

    assert second["status"] == "ok"
    assert second["action"] == "unchanged"
    assert second["wrote"] == []
    assert _read(Path(second["path"])) == before
    assert all(status == "unchanged" for status in second["projections"].values())


def test_unparseable_canonical_is_invalid_and_untouched(tmp_path: Path) -> None:
    canonical = tmp_path / generic.CANONICAL_FILENAME
    original = "{not valid json"
    canonical.write_text(original, encoding="utf-8")

    result = generic.install_generic(target_dir=str(tmp_path))

    assert result["status"] == "invalid"
    assert result["wrote"] == []
    assert _read(canonical) == original
    assert not (tmp_path / ".mcp.json").exists()


def test_unparseable_projection_is_skipped_with_warning(tmp_path: Path) -> None:
    cursor_dir = tmp_path / ".cursor"
    cursor_dir.mkdir()
    projection = cursor_dir / "mcp.json"
    original = "not json either"
    projection.write_text(original, encoding="utf-8")

    result = generic.install_generic(target_dir=str(tmp_path))

    assert result["status"] == "ok"
    assert (tmp_path / generic.CANONICAL_FILENAME).exists()
    assert _read(projection) == original
    assert result["projections"][os.path.join(".cursor", "mcp.json")] == "invalid"
    assert any("skipped" in warning for warning in result["warnings"])


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    result = generic.install_generic(target_dir=str(tmp_path), dry_run=True)

    assert result["status"] == "ok"
    assert result["dry_run"] is True
    assert result["wrote"] == []
    assert not (tmp_path / generic.CANONICAL_FILENAME).exists()
    assert not (tmp_path / ".mcp.json").exists()
    assert "evidencespine" in result["content"]
    planned = [Path(path).name for path in result["planned"]]
    assert ".mcp.json" in planned


def test_uninstall_removes_only_evidence_spine_entry(tmp_path: Path) -> None:
    canonical = tmp_path / generic.CANONICAL_FILENAME
    canonical.write_text(
        json.dumps({"mcpServers": {"other": {"command": "other-bin"}}, "customKey": 1}),
        encoding="utf-8",
    )
    cursor = tmp_path / ".cursor"
    cursor.mkdir()
    other_projection = cursor / "mcp.json"
    other_projection.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}), encoding="utf-8")
    generic.install_generic(target_dir=str(tmp_path))

    result = generic.uninstall_generic(target_dir=str(tmp_path))

    assert result["status"] == "ok"
    assert result["action"] == "removed"
    payload = _load(canonical)
    assert payload["customKey"] == 1
    assert "evidencespine" not in payload["mcpServers"]
    assert "other" in payload["mcpServers"]
    assert "evidencespine" not in _load(other_projection)["mcpServers"]
    assert not (tmp_path / generic.NOTES_FILENAME).exists()


def test_uninstall_deletes_files_it_created(tmp_path: Path) -> None:
    generic.install_generic(target_dir=str(tmp_path))
    result = generic.uninstall_generic(target_dir=str(tmp_path))

    assert result["status"] == "ok"
    assert result["action"] == "removed"
    for rel in generic.DEFAULT_PROJECTIONS:
        assert not (tmp_path / rel).exists(), rel
    assert not (tmp_path / generic.CANONICAL_FILENAME).exists()
    assert not (tmp_path / generic.NOTES_FILENAME).exists()


def test_uninstall_unparseable_canonical_is_invalid(tmp_path: Path) -> None:
    canonical = tmp_path / generic.CANONICAL_FILENAME
    canonical.write_text("broken", encoding="utf-8")

    result = generic.uninstall_generic(target_dir=str(tmp_path))

    assert result["status"] == "invalid"
    assert _read(canonical) == "broken"


def test_multitoken_executable_splits_command_and_args(tmp_path: Path) -> None:
    result = generic.install_generic(
        target_dir=str(tmp_path),
        base_dir=".es",
        executable=sys.executable + " -m evidencespine.cli",
    )
    entry = _load(Path(result["path"]))["mcpServers"]["evidencespine"]

    assert entry["command"] == sys.executable
    assert entry["args"] == ["-m", "evidencespine.cli", "mcp", "--base-dir", ".es"]


def test_default_install_is_path_portable(tmp_path: Path) -> None:
    result = generic.install_generic(target_dir=str(tmp_path))
    text = _read(Path(result["path"]))

    assert "evidencespine" in text
    assert str(tmp_path) not in text
    assert "env" in text


def test_global_scope_skips_missing_parent_dirs(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()

    result = generic.install_generic(target_dir=str(home), scope="global")

    assert result["status"] == "ok"
    assert (home / generic.CANONICAL_FILENAME).exists()
    assert (home / ".mcp.json").exists()
    assert not (home / ".cursor").exists()
    assert result["projections"][os.path.join(".cursor", "mcp.json")] == "skipped"
    assert any("parent directory does not exist" in warning for warning in result["warnings"])


def test_notes_can_be_disabled(tmp_path: Path) -> None:
    generic.install_generic(target_dir=str(tmp_path), notes=False)
    assert not (tmp_path / generic.NOTES_FILENAME).exists()


def test_render_all_clients_contract() -> None:
    for client in generic.SUPPORTED_CLIENTS:
        result = generic.render_client_config(
            client,
            executable="evidencespine",
            base_dir=".evidencespine",
            project_dir="/tmp/demo-project",
        )
        assert result["client"] == client
        assert isinstance(result["path"], str)
        assert isinstance(result["paths"], dict)
        assert result["snippet"].strip(), client
        assert result["verify"].strip(), client
        assert result["format"] in {"json", "toml", "yaml"}
        assert result["grade"] in {"C", "I", "U"}
        assert "note" in result, f"{client} should explain path/shape confidence"


def test_render_unknown_client_raises() -> None:
    with pytest.raises(ValueError):
        generic.render_client_config("nope")


def test_render_aliases_resolve() -> None:
    assert generic.render_client_config("devin")["client"] == "windsurf"
    assert generic.render_client_config("vscode-copilot")["client"] == "vscode-copilot"
    assert generic.render_client_config("copilot")["client"] == "vscode-copilot"
    assert generic.render_client_config("vscode-legacy")["client"] == "vscode"
    assert generic.render_client_config("gemini-cli")["client"] == "gemini"


def test_render_alias_records_requested_client() -> None:
    result = generic.render_client_config("devin")
    assert result["client"] == "windsurf"
    assert result["requested_client"] == "devin"


def test_render_http_transport_uses_remote_shape() -> None:
    result = generic.render_client_config("codex", transport="http")

    assert result["format"] == "toml"
    assert result["transport"] == "http"
    assert result["requested_transport"] == "http"
    assert "bearer_token_env_var" in result["note"] + result["snippet"]
    assert "[mcp_servers.evidencespine]" in result["snippet"]


def test_render_json_snippets_parse_and_have_expected_shape() -> None:
    base = {"executable": "evidencespine", "base_dir": ".es", "project_dir": "/tmp/p"}
    standard = {
        "claude-code",
        "claude-desktop",
        "cursor",
        "gemini",
        "antigravity",
        "junie",
        "windsurf",
        "generic",
    }
    for client in standard:
        payload = json.loads(generic.render_client_config(client, **base)["snippet"])
        assert "evidencespine" in payload["mcpServers"], client
        assert payload["mcpServers"]["evidencespine"]["command"] == "evidencespine"

    amp = json.loads(generic.render_client_config("amp", **base)["snippet"])
    assert "evidencespine" in amp["amp.mcpServers"]

    zed = json.loads(generic.render_client_config("zed", **base)["snippet"])
    assert zed["context_servers"]["evidencespine"]["args"][-2:] == ["--base-dir", ".es"]

    legacy = json.loads(generic.render_client_config("vscode", **base)["snippet"])
    assert "servers" in legacy and "inputs" in legacy
    assert legacy["servers"]["evidencespine"]["type"] == "stdio"
    assert json.loads(generic.render_client_config("vscode-legacy", **base)["snippet"]) == legacy

    portable = json.loads(generic.render_client_config("vscode-copilot", **base)["snippet"])
    assert portable["mcpServers"]["evidencespine"]["type"] == "stdio"

    cline = json.loads(generic.render_client_config("cline", **base)["snippet"])
    assert cline["mcpServers"]["evidencespine"]["disabled"] is False

    roo = json.loads(generic.render_client_config("roo", **base)["snippet"])
    assert roo["mcpServers"]["evidencespine"]["cwd"] == "/tmp/p"

    opencode = json.loads(generic.render_client_config("opencode", **base)["snippet"])
    server = opencode["mcp"]["servers"]["evidencespine"]
    assert server["type"] == "local"
    assert server["command"] == ["evidencespine", "mcp", "--base-dir", ".es"]


def test_render_codex_toml_parses() -> None:
    tomllib = pytest.importorskip("tomllib")
    result = generic.render_client_config("codex", executable="/usr/bin/evidencespine", base_dir=".es")

    assert result["format"] == "toml"
    assert result["toml"] == result["snippet"]
    data = tomllib.loads(result["snippet"])
    assert data["mcp_servers"]["evidencespine"]["command"] == "/usr/bin/evidencespine"
    assert data["mcp_servers"]["evidencespine"]["args"] == ["mcp", "--base-dir", ".es"]


def test_render_dsh_insert_row_yaml() -> None:
    result = generic.render_client_config(
        "dsh",
        executable="/opt/evidencespine",
        base_dir=".evidencespine",
        project_dir="/tmp/demo-project",
    )

    snippet = result["snippet"]
    assert result["format"] == "yaml"
    assert result["yaml"] == snippet
    assert snippet.startswith("# BEGIN EvidenceSpine (managed)")
    assert "    - id: evidencespine-mcp" in snippet
    assert "name: '@deepseek-ai/dsh-mcp-client'" in snippet
    assert "serverName: evidencespine" in snippet
    assert "transport: stdio" in snippet
    assert "cwd: '" + os.path.realpath("/tmp/demo-project") + "'" in snippet
    assert result["paths"]["project"].endswith(os.path.join(".dsh", "evidencespine.dsh.patch.yml"))
    assert result["paths"]["user"].endswith("cordis.patch.yml")
    assert "--dump-config --profile headless" in result["verify"]


def test_render_continue_yaml() -> None:
    result = generic.render_client_config("continue", executable="evidencespine", base_dir=".es")

    assert result["format"] == "yaml"
    assert "mcpServers:" in result["snippet"]
    assert "type: stdio" in result["snippet"]
    assert result["yaml"] == result["snippet"]


def test_render_jetbrains_is_ui_only() -> None:
    result = generic.render_client_config("jetbrains")

    assert result["path"] == ""
    assert result["grade"] == "U"
    assert "UI-managed" in result["note"]
    assert json.loads(result["snippet"])["mcpServers"]


def test_render_paths_respect_scope_and_placeholders() -> None:
    placeholder = generic.render_client_config("cursor", project_dir="")
    assert placeholder["paths"]["project"].startswith("<repo>/")

    global_render = generic.render_client_config("cursor", scope="global")
    assert global_render["path"] == global_render["paths"]["user"]

    project_render = generic.render_client_config("cursor", project_dir="/tmp/abc")
    assert project_render["path"] == os.path.join("/tmp/abc", ".cursor", "mcp.json")


def test_render_notes_are_short_and_actionable() -> None:
    notes = generic.render_setup_notes(base_dir=".evidencespine", executable="evidencespine")
    lines = notes.splitlines()

    assert len(lines) <= generic.MAX_NOTES_LINES
    assert generic.CANONICAL_FILENAME in notes
    assert "print-config" in notes
    assert ".mcp.json" in notes


def test_install_with_agents_md_writes_managed_block(tmp_path: Path) -> None:
    result = generic.install_generic(target_dir=str(tmp_path), with_agents_md=True)

    agents = tmp_path / "AGENTS.md"
    assert result["status"] == "ok"
    assert agents.exists()
    assert result["instructions"]["harness"] == "instructions"
    assert str(agents) in result["wrote"]
    assert "<!-- evidencespine:begin -->" in _read(agents)


def test_install_dry_run_with_agents_md_does_not_write(tmp_path: Path) -> None:
    result = generic.install_generic(target_dir=str(tmp_path), with_agents_md=True, dry_run=True)

    assert result["status"] == "ok"
    assert result["instructions"]["changed"] is True
    assert not (tmp_path / "AGENTS.md").exists()


# ---------------------------------------------------------------------------
# Streamable-HTTP variants (research section 4.2 + 6.4)
# ---------------------------------------------------------------------------

HTTP_URL = "http://127.0.0.1:8000/mcp"


def _http(client: str, **kw) -> dict:
    return generic.render_client_config(
        client,
        executable="evidencespine",
        base_dir=".evidencespine",
        project_dir="/tmp/http-demo",
        transport="http",
        **kw,
    )


def test_http_defaults_and_url_override() -> None:
    result = _http("cursor")

    assert result["requested_transport"] == "http"
    assert result["transport"] == "http"
    assert result["http_url"] == HTTP_URL
    assert HTTP_URL in result["snippet"]

    override = generic.render_client_config(
        "cursor", executable="evidencespine", transport="http", url="http://127.0.0.1:9000/mcp"
    )
    assert "http://127.0.0.1:9000/mcp" in override["snippet"]
    alias = generic.render_client_config(
        "cursor", executable="evidencespine", transport="http", mcp_url="http://127.0.0.1:9100/mcp"
    )
    assert "http://127.0.0.1:9100/mcp" in alias["snippet"]


def test_http_claude_code_requires_type_http() -> None:
    server = json.loads(_http("claude-code")["snippet"])["mcpServers"]["evidencespine"]

    assert server["type"] == "http"
    assert server["url"] == HTTP_URL
    authorization = server["headers"]["Authorization"]
    assert "EVIDENCESPINE_MCP_AUTH_TOKEN" in authorization
    assert "${" in authorization


def test_http_typed_discriminators_match_notes() -> None:
    cline = json.loads(_http("cline")["snippet"])["mcpServers"]["evidencespine"]
    assert cline["type"] == "streamableHttp"
    assert cline["url"] == HTTP_URL

    roo = json.loads(_http("roo")["snippet"])["mcpServers"]["evidencespine"]
    assert roo["type"] == "streamable-http"
    assert roo["url"] == HTTP_URL

    legacy = json.loads(_http("vscode")["snippet"])
    assert legacy["servers"]["evidencespine"]["type"] == "http"
    assert legacy["servers"]["evidencespine"]["url"] == HTTP_URL
    assert json.loads(_http("vscode-legacy")["snippet"]) == legacy

    portable = json.loads(_http("vscode-copilot")["snippet"])["mcpServers"]["evidencespine"]
    assert portable["type"] == "http"
    assert portable["url"] == HTTP_URL

    opencode = json.loads(_http("opencode")["snippet"])["mcp"]["servers"]["evidencespine"]
    assert opencode["type"] == "remote"
    assert opencode["url"] == HTTP_URL
    assert opencode["oauth"] is False


def test_http_gemini_uses_http_url_key() -> None:
    server = json.loads(_http("gemini")["snippet"])["mcpServers"]["evidencespine"]

    assert server["httpUrl"] == HTTP_URL
    assert "url" not in server
    assert server["headers"]["Authorization"].endswith("EVIDENCESPINE_MCP_AUTH_TOKEN")


def test_http_antigravity_uses_server_url() -> None:
    server = json.loads(_http("antigravity")["snippet"])["mcpServers"]["evidencespine"]

    assert server["serverUrl"] == HTTP_URL
    assert "url" not in server
    assert "SSE and httpUrl are unsupported" in _http("antigravity")["note"]


def test_http_windsurf_uses_server_url() -> None:
    server = json.loads(_http("windsurf")["snippet"])["mcpServers"]["evidencespine"]

    assert server["serverUrl"] == HTTP_URL
    assert "headers" in server
    assert "[U]" in _http("windsurf")["note"]


def test_http_dsh_uses_loader_js_header() -> None:
    result = _http("dsh")
    snippet = result["snippet"]

    assert result["format"] == "yaml"
    assert result["yaml"] == snippet
    assert "transport: streamable-http" in snippet
    assert HTTP_URL in snippet
    assert "failOnStartupError: true" in snippet
    assert "Authorization: !!js '\"Bearer \" + process.env.EVIDENCESPINE_MCP_AUTH_TOKEN'" in snippet
    assert "!!js" in snippet


def test_http_zed_uses_context_servers() -> None:
    payload = json.loads(_http("zed")["snippet"])

    assert "context_servers" in payload
    assert "mcpServers" not in payload
    server = payload["context_servers"]["evidencespine"]
    assert server["url"] == HTTP_URL
    assert "Authorization" in server["headers"]
    assert "(I)" in _http("zed")["note"] or "[I]" in _http("zed")["note"]


def test_http_continue_requires_streamable_type() -> None:
    result = _http("continue")

    assert result["format"] == "yaml"
    assert "type: streamable-http" in result["snippet"]
    assert HTTP_URL in result["snippet"]
    assert result["yaml"] == result["snippet"]


def test_http_codex_uses_url_and_bearer_env() -> None:
    tomllib = pytest.importorskip("tomllib")
    result = _http("codex")
    data = tomllib.loads(result["snippet"])
    server = data["mcp_servers"]["evidencespine"]

    assert server["url"] == HTTP_URL
    assert server["bearer_token_env_var"] == generic.AUTH_TOKEN_ENV
    assert "command" not in server
    assert result["toml"] == result["snippet"]


def test_http_claude_desktop_is_ui_only() -> None:
    result = _http("claude-desktop")

    assert result["grade"] == "U"
    assert "Connectors" in result["note"]
    assert HTTP_URL not in result["snippet"], "must not emit a misleading url entry"
    assert "mcpServers" in json.loads(result["snippet"])


def test_http_amp_uses_amp_key_and_env_dialect() -> None:
    payload = json.loads(_http("amp")["snippet"])
    server = payload["amp.mcpServers"]["evidencespine"]

    assert server["url"] == HTTP_URL
    assert "${EVIDENCESPINE_MCP_AUTH_TOKEN}" in server["headers"]["Authorization"]


def test_http_documented_clients_emit_url_and_token() -> None:
    for client in generic.HTTP_FORMAT_CLIENTS:
        result = _http(client)
        assert result["requested_transport"] == "http", client
        assert HTTP_URL in result["snippet"], client
        assert generic.AUTH_TOKEN_ENV in result["snippet"], client
        assert result["note"], client
        assert result["grade"] in {"C", "I", "U"}, client


def test_http_untyped_clients_use_url_plus_headers() -> None:
    for client in ("junie", "warp", "jetbrains", "generic"):
        server = json.loads(_http(client)["snippet"])["mcpServers"]["evidencespine"]
        assert server["url"] == HTTP_URL, client
        assert "Authorization" in server["headers"], client
        assert "type" not in server, client

    assert _http("warp")["grade"] == "U"
    assert "verify in the Warp" in _http("warp")["note"]
    assert _http("jetbrains")["grade"] == "U"


def test_http_does_not_change_stdio_snippets() -> None:
    stdio = generic.render_client_config("claude-code", executable="evidencespine", base_dir=".es")

    assert stdio["transport"] == "stdio"
    server = json.loads(stdio["snippet"])["mcpServers"]["evidencespine"]
    assert server["command"] == "evidencespine"
    assert "url" not in server
    assert "requested_transport" not in stdio
