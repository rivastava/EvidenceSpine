"""Tests for the DeepSeek Harness (DSH) patch-layer installer."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from evidencespine.harness import dsh

EXE = "/opt/evidencespine/bin/evidencespine"


@pytest.fixture()
def dsh_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "dsh-home"
    monkeypatch.setenv("DSH_HOME", str(home))
    monkeypatch.delenv("DSH_CLI", raising=False)
    return home


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    return project


def _install(tmp_path: Path, project: Path, **kw):
    kwargs = {
        "target_dir": str(project),
        "base_dir": ".evidencespine",
        "executable": EXE,
        "verify": False,
    }
    kwargs.update(kw)
    return dsh.install_dsh(**kwargs)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_install_creates_one_managed_block_with_absolute_paths(tmp_path: Path, dsh_home: Path) -> None:
    project = _project(tmp_path)
    result = _install(tmp_path, project)
    patch = dsh_home / "cordis.patch.yml"

    assert result["status"] == "ok"
    assert result["harness"] == "dsh"
    assert result["action"] == "added"
    assert result["path"] == str(patch)
    assert result["wrote"] == [str(patch)]
    text = _read(patch)
    assert text.count(dsh.MANAGED_BEGIN) == 1
    assert text.count(dsh.MANAGED_END) == 1
    assert "- insert:" in text
    assert "    - id: evidencespine-mcp" in text
    assert "name: '@deepseek-ai/dsh-mcp-client'" in text
    assert "serverName: evidencespine" in text
    assert "transport: stdio" in text
    assert "command: '" + EXE + "'" in text
    assert "          - 'mcp'" in text
    assert "          - '--base-dir'" in text
    abs_base = os.path.join(str(project), ".evidencespine")
    assert "          - '" + abs_base + "'" in text
    assert "cwd: '" + os.path.realpath(str(project)) + "'" in text
    assert "failOnStartupError: false" in text
    assert "startupTimeoutMs: 20000" in text
    if os.name != "nt":
        assert (patch.stat().st_mode & 0o777) == 0o600
    assert not [name for name in os.listdir(str(dsh_home)) if name.endswith(".tmp")]


def test_second_install_is_byte_idempotent(tmp_path: Path, dsh_home: Path) -> None:
    project = _project(tmp_path)
    first = _install(tmp_path, project)
    before = _read(Path(first["path"]))
    second = _install(tmp_path, project)

    assert second["status"] == "ok"
    assert second["action"] == "unchanged"
    assert second["wrote"] == []
    assert _read(Path(second["path"])) == before
    assert before.count(dsh.MANAGED_BEGIN) == 1


def test_reinstall_updates_block_and_preserves_user_yaml(tmp_path: Path, dsh_home: Path) -> None:
    project = _project(tmp_path)
    dsh_home.mkdir(parents=True)
    patch = dsh_home / "cordis.patch.yml"
    user_yaml = "# user comment\n- id: user-row\n  config: {}\n"
    patch.write_text(user_yaml, encoding="utf-8")

    first = _install(tmp_path, project)
    assert first["action"] == "added"
    assert "user-row" in _read(patch)
    backups = sorted(patch.parent.glob("cordis.patch.yml.bak-*"))
    assert backups, "a backup is expected before modifying an existing file"
    assert _read(backups[0]) == user_yaml

    second = _install(tmp_path, project, executable="/opt/other-evidencespine")
    assert second["action"] == "updated"
    text = _read(patch)
    assert text.count(dsh.MANAGED_BEGIN) == 1
    assert "user-row" in text
    assert "command: '/opt/other-evidencespine'" in text
    assert EXE not in text


def test_guard_rejects_non_sequence_yaml_without_corruption(tmp_path: Path, dsh_home: Path) -> None:
    project = _project(tmp_path)
    dsh_home.mkdir(parents=True)
    patch = dsh_home / "cordis.patch.yml"
    original = "mcp-manager:\n  servers: {}\n"
    patch.write_text(original, encoding="utf-8")

    result = _install(tmp_path, project)

    assert result["status"] == "invalid"
    assert "top-level YAML sequence" in result["reason"]
    assert result["wrote"] == []
    assert _read(patch) == original


def test_guard_allows_comments_only_file(tmp_path: Path, dsh_home: Path) -> None:
    project = _project(tmp_path)
    dsh_home.mkdir(parents=True)
    patch = dsh_home / "cordis.patch.yml"
    patch.write_text("# my own comment\n\n", encoding="utf-8")

    result = _install(tmp_path, project)

    assert result["status"] == "ok"
    assert result["action"] == "added"
    text = _read(patch)
    assert "# my own comment" in text
    assert "- insert:" in text


def test_unmanaged_row_is_a_conflict_not_a_duplicate(tmp_path: Path, dsh_home: Path) -> None:
    project = _project(tmp_path)
    dsh_home.mkdir(parents=True)
    patch = dsh_home / "cordis.patch.yml"
    original = "- id: evidencespine-mcp\n  name: '@deepseek-ai/dsh-mcp-client'\n"
    patch.write_text(original, encoding="utf-8")

    result = _install(tmp_path, project)

    assert result["status"] == "conflict"
    assert result["wrote"] == []
    assert _read(patch) == original


def test_unterminated_block_is_invalid(tmp_path: Path, dsh_home: Path) -> None:
    project = _project(tmp_path)
    dsh_home.mkdir(parents=True)
    patch = dsh_home / "cordis.patch.yml"
    original = dsh.MANAGED_BEGIN + "\n- id: x\n"
    patch.write_text(original, encoding="utf-8")

    result = _install(tmp_path, project)

    assert result["status"] == "invalid"
    assert "unterminated" in result["reason"]
    assert _read(patch) == original


def test_dry_run_returns_planned_content_without_writing(tmp_path: Path, dsh_home: Path) -> None:
    project = _project(tmp_path)
    result = _install(tmp_path, project, dry_run=True)

    assert result["status"] == "ok"
    assert result["dry_run"] is True
    assert result["wrote"] == []
    assert "id: evidencespine-mcp" in result["content"]
    assert not (dsh_home / "cordis.patch.yml").exists()


def test_uninstall_restores_preexisting_content(tmp_path: Path, dsh_home: Path) -> None:
    project = _project(tmp_path)
    dsh_home.mkdir(parents=True)
    patch = dsh_home / "cordis.patch.yml"
    original = "- id: user-row\n  config:\n    value: 1\n"
    patch.write_text(original, encoding="utf-8")
    _install(tmp_path, project)

    result = dsh.uninstall_dsh(target_dir=str(project), verify=False)

    assert result["status"] == "ok"
    assert result["action"] == "removed"
    assert result["wrote"] == [str(patch)]
    assert _read(patch) == original


def test_uninstall_deletes_file_it_created(tmp_path: Path, dsh_home: Path) -> None:
    project = _project(tmp_path)
    installed = _install(tmp_path, project)
    patch = Path(installed["path"])
    assert patch.exists()

    result = dsh.uninstall_dsh(target_dir=str(project), verify=False)

    assert result["status"] == "ok"
    assert result["action"] == "removed"
    assert result["removed"] == [str(patch)]
    assert not patch.exists()
    assert sorted(patch.parent.glob("cordis.patch.yml.bak-*"))


def test_uninstall_absent_is_ok(tmp_path: Path, dsh_home: Path) -> None:
    result = dsh.uninstall_dsh(target_dir=str(tmp_path / "project"), verify=False)

    assert result["status"] == "ok"
    assert result["action"] == "absent"
    assert result["wrote"] == []


def test_uninstall_dry_run_does_not_touch_file(tmp_path: Path, dsh_home: Path) -> None:
    project = _project(tmp_path)
    installed = _install(tmp_path, project)
    patch = Path(installed["path"])
    before = _read(patch)

    result = dsh.uninstall_dsh(target_dir=str(project), dry_run=True, verify=False)

    assert result["status"] == "ok"
    assert result["dry_run"] is True
    assert result["would_remove_file"] is True
    assert result["content"] == ""
    assert _read(patch) == before


def test_profile_scope_writes_manager_row(tmp_path: Path, dsh_home: Path) -> None:
    project = _project(tmp_path)
    profile_dir = dsh_home / "profiles" / "headless"
    profile_dir.mkdir(parents=True)

    result = dsh.install_dsh(
        target_dir=str(project),
        scope="profile:headless",
        executable=EXE,
        verify=False,
    )

    path = profile_dir / "cordis.patch.yml"
    assert result["status"] == "ok"
    assert result["action"] == "added"
    assert result["path"] == str(path)
    text = _read(path)
    assert "- id: mcp-manager" in text
    assert "    servers:" in text
    assert "      evidencespine:" in text
    assert "enabled: true" in text
    assert "id: evidencespine-mcp" not in text

    removed = dsh.uninstall_dsh(target_dir=str(project), scope="profile:headless", verify=False)
    assert removed["action"] == "removed"
    assert not path.exists()


def test_profile_scope_requires_provisioned_profile(tmp_path: Path, dsh_home: Path) -> None:
    project = _project(tmp_path)
    result = dsh.install_dsh(
        target_dir=str(project),
        scope="profile:web",
        executable=EXE,
        verify=False,
    )

    assert result["status"] == "invalid"
    assert "not provisioned" in result["reason"]
    assert result["wrote"] == []


def test_profile_scope_conflicts_with_user_manager_row(tmp_path: Path, dsh_home: Path) -> None:
    project = _project(tmp_path)
    profile_dir = dsh_home / "profiles" / "headless"
    profile_dir.mkdir(parents=True)
    patch = profile_dir / "cordis.patch.yml"
    original = "- id: mcp-manager\n  config:\n    servers:\n      other:\n        transport: stdio\n"
    patch.write_text(original, encoding="utf-8")

    result = dsh.install_dsh(
        target_dir=str(project),
        scope="profile:headless",
        executable=EXE,
        verify=False,
    )

    assert result["status"] == "conflict"
    assert result["snippet"]
    assert "evidencespine:" in result["snippet"]
    assert _read(patch) == original


def test_coexistence_with_profile_manager_entry_warns(tmp_path: Path, dsh_home: Path) -> None:
    project = _project(tmp_path)
    profile_dir = dsh_home / "profiles" / "web"
    profile_dir.mkdir(parents=True)
    (profile_dir / "cordis.patch.yml").write_text(
        "- id: mcp-manager\n  config:\n    servers:\n      evidencespine:\n        transport: stdio\n",
        encoding="utf-8",
    )

    result = _install(tmp_path, project)

    assert result["status"] == "ok"
    assert any("mcp-manager" in warning and "evidencespine" in warning for warning in result["warnings"])
    assert (dsh_home / "cordis.patch.yml").exists()


def test_invalid_scope_is_reported(tmp_path: Path, dsh_home: Path) -> None:
    result = _install(tmp_path, _project(tmp_path), scope="bogus")

    assert result["status"] == "invalid"
    assert "unsupported dsh scope" in result["reason"]


def test_verification_unavailable_never_fails_install(
    tmp_path: Path, dsh_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dsh, "find_dsh_cli", lambda: "")
    result = _install(tmp_path, _project(tmp_path), verify=True)

    assert result["status"] == "ok"
    verification = result["verification"]
    assert verification["composed_headless"] is None
    assert verification["composed_web"] is None
    assert verification["error"]


def test_real_dsh_cli_composes_and_drops_the_row(tmp_path: Path, dsh_home: Path) -> None:
    if not dsh.find_dsh_cli():
        pytest.skip("dsh CLI not available on this machine")
    project = _project(tmp_path)

    installed = dsh.install_dsh(target_dir=str(project), executable=EXE, verify=True)
    verification = installed["verification"]
    assert verification["composed_headless"] is True, verification
    assert verification["composed_web"] is True, verification
    assert verification["error"] == ""

    removed = dsh.uninstall_dsh(target_dir=str(project), verify=True)
    assert removed["action"] == "removed"
    assert removed["verification"]["composed_headless"] is False


def test_build_insert_row_uses_absolute_cwd_and_base(tmp_path: Path) -> None:
    project = _project(tmp_path)
    row = dsh.build_insert_row(executable=EXE, base_dir=".evidencespine", target_dir=str(project))

    assert row["id"] == "evidencespine-mcp"
    assert row["name"] == "@deepseek-ai/dsh-mcp-client"
    config = row["config"]
    assert config["command"] == EXE
    assert config["args"][:3] == ["mcp", "--base-dir", os.path.join(str(project), ".evidencespine")]
    assert config["cwd"] == os.path.realpath(str(project))
    assert config["failOnStartupError"] is False
    assert config["startupTimeoutMs"] == 20000


def test_multitoken_executable_is_split_into_command_and_args(tmp_path: Path) -> None:
    project = _project(tmp_path)
    row = dsh.build_insert_row(
        executable=sys.executable + " -m evidencespine.cli",
        base_dir=".es",
        target_dir=str(project),
    )

    config = row["config"]
    assert config["command"] == sys.executable
    assert config["args"][:2] == ["-m", "evidencespine.cli"]
    assert config["args"][-2:] == ["--base-dir", os.path.join(str(project), ".es")]


def test_bare_missing_executable_warns_but_writes(tmp_path: Path, dsh_home: Path) -> None:
    project = _project(tmp_path)
    result = _install(tmp_path, project, executable="definitely-not-on-path-xyz")

    assert result["status"] == "ok"
    assert any("not absolute" in warning for warning in result["warnings"])
    assert "command: 'definitely-not-on-path-xyz'" in _read(dsh_home / "cordis.patch.yml")


def test_windows_cmd_shim_warning() -> None:
    warnings = dsh._platform_warnings("evidencespine.cmd")
    assert any("Windows" in warning for warning in warnings)
    assert dsh._platform_warnings("/usr/local/bin/evidencespine") == []


def test_install_with_agents_md_writes_managed_block(tmp_path: Path, dsh_home: Path) -> None:
    project = _project(tmp_path)
    result = _install(tmp_path, project, with_agents_md=True)

    agents = project / "AGENTS.md"
    assert result["status"] == "ok"
    assert agents.exists()
    assert result["instructions"]["harness"] == "instructions"
    assert str(agents) in result["wrote"]
    assert "<!-- evidencespine:begin -->" in agents.read_text(encoding="utf-8")


def test_install_dry_run_with_agents_md_does_not_write(tmp_path: Path, dsh_home: Path) -> None:
    project = _project(tmp_path)
    result = _install(tmp_path, project, with_agents_md=True, dry_run=True)

    assert result["status"] == "ok"
    assert result["instructions"]["changed"] is True
    assert not (project / "AGENTS.md").exists()


def test_render_http_patch_block_uses_loader_js_header() -> None:
    text = dsh.render_http_patch_block("http://127.0.0.1:9000/mcp")

    assert text.count(dsh.MANAGED_BEGIN) == 1
    assert text.count(dsh.MANAGED_END) == 1
    assert "    - id: evidencespine-mcp" in text
    assert "        transport: streamable-http" in text
    assert "        url: 'http://127.0.0.1:9000/mcp'" in text
    assert "Authorization: !!js '\"Bearer \" + process.env.EVIDENCESPINE_MCP_AUTH_TOKEN'" in text
    assert "failOnStartupError: true" in text
    assert "startupTimeoutMs" not in text


def test_render_http_patch_block_supports_custom_auth_env() -> None:
    text = dsh.render_http_patch_block(auth_env="MY_SPINE_TOKEN")

    assert "process.env.MY_SPINE_TOKEN" in text
    assert "process.env.EVIDENCESPINE_MCP_AUTH_TOKEN" not in text
