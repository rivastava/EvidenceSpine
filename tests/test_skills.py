"""Tests for skill rendering and DSH/.agents skill materialization."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from evidencespine.harness import skills


def _frontmatter(text: str) -> dict:
    lines = text.splitlines()
    assert lines[0] == "---"
    end = lines.index("---", 1)
    meta = {}
    for line in lines[1:end]:
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


def _repo_root() -> Path:
    return Path(skills.__file__).resolve().parents[3]


def test_render_skill_markdown_frontmatter_and_body() -> None:
    text = skills.render_skill_markdown()
    meta = _frontmatter(text)

    assert meta["name"] == "evidencespine"
    assert meta["description"].endswith(".")
    assert "\n" not in meta["description"]
    assert meta["description"].count(".") == 1
    assert "before claiming" in meta["whenToUse"].lower()
    assert meta["user-invocable"] == "true"
    assert "## The evidence model" in text
    assert "## How to reach the spine" in text
    assert skills.GENERATED_MARKER not in text


def test_skill_rel_path_contract() -> None:
    assert skills.SKILL_REL_PATH == "skills/evidencespine/SKILL.md"


def test_canonical_skill_file_exists_and_matches_frontmatter() -> None:
    canonical = _repo_root() / skills.SKILL_REL_PATH
    assert canonical.is_file(), canonical
    text = canonical.read_text(encoding="utf-8")

    assert _frontmatter(text) == _frontmatter(skills.render_skill_markdown())
    assert text.count("## The evidence model") == 1


def test_canonical_skill_path_resolves_repo_file() -> None:
    resolved = skills.canonical_skill_path()

    assert resolved.endswith(os.path.join("skills", "evidencespine", "SKILL.md"))
    assert os.path.isfile(resolved)


def test_materialize_dsh_uses_relative_symlink(tmp_path: Path) -> None:
    result = skills.materialize(target_dir=str(tmp_path), harnesses=["dsh"])
    link = tmp_path / ".dsh" / "skills" / "evidencespine" / "SKILL.md"

    assert result["status"] == "ok"
    assert result["mode"] == "symlink"
    assert link.exists()
    if os.name != "nt":
        assert link.is_symlink()
        assert not os.path.isabs(os.readlink(link))
    assert os.path.realpath(str(link)) == skills.canonical_skill_path()
    assert any(Path(path).name == "SKILL.md" for path in result["wrote"])
    second = skills.materialize(target_dir=str(tmp_path), harnesses=["dsh"])
    assert second["wrote"] == []
    assert second["warnings"] == []


def test_materialize_generic_uses_agents_skills_root(tmp_path: Path) -> None:
    result = skills.materialize(target_dir=str(tmp_path), harnesses=["generic"])
    link = tmp_path / ".agents" / "skills" / "evidencespine" / "SKILL.md"

    assert result["status"] == "ok"
    assert link.exists()
    assert os.path.realpath(str(link)) == skills.canonical_skill_path()


def test_materialize_both_harnesses(tmp_path: Path) -> None:
    result = skills.materialize(target_dir=str(tmp_path), harnesses=["dsh", "generic"])

    assert (tmp_path / ".dsh" / "skills" / "evidencespine" / "SKILL.md").exists()
    assert (tmp_path / ".agents" / "skills" / "evidencespine" / "SKILL.md").exists()
    assert len(result["targets"]) == 2
    assert len(result["wrote"]) == 2


def test_copy_fallback_when_canonical_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skills, "canonical_skill_path", lambda target_dir="": "")
    result = skills.materialize(target_dir=str(tmp_path), harnesses=["dsh"])
    target = tmp_path / ".dsh" / "skills" / "evidencespine" / "SKILL.md"

    assert result["mode"] == "copy"
    assert target.exists()
    assert not target.is_symlink()
    text = target.read_text(encoding="utf-8")
    assert skills.GENERATED_MARKER in text
    assert text.startswith("---\nname: evidencespine")
    second = skills.materialize(target_dir=str(tmp_path), harnesses=["dsh"])
    assert second["wrote"] == []


def test_symlink_failure_falls_back_to_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(src: str, dst: str) -> None:
        raise OSError("symlink not permitted")

    monkeypatch.setattr(os, "symlink", _boom)
    result = skills.materialize(target_dir=str(tmp_path), harnesses=["dsh"])
    target = tmp_path / ".dsh" / "skills" / "evidencespine" / "SKILL.md"

    assert target.exists()
    assert not target.is_symlink()
    assert any("symlink unavailable" in warning for warning in result["warnings"])


def test_user_modified_copy_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skills, "canonical_skill_path", lambda target_dir="": "")
    skills.materialize(target_dir=str(tmp_path), harnesses=["dsh"])
    target = tmp_path / ".dsh" / "skills" / "evidencespine" / "SKILL.md"
    target.write_text("user edited skill\n", encoding="utf-8")

    result = skills.materialize(target_dir=str(tmp_path), harnesses=["dsh"])

    assert target.read_text(encoding="utf-8") == "user edited skill\n"
    assert result["wrote"] == []
    assert any("preserved user-modified" in warning for warning in result["warnings"])


def test_generated_copy_is_updated_in_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(skills, "canonical_skill_path", lambda target_dir="": "")
    skills.materialize(target_dir=str(tmp_path), harnesses=["dsh"])
    target = tmp_path / ".dsh" / "skills" / "evidencespine" / "SKILL.md"
    stale = "---\nname: evidencespine\n---\n\nstale body\n\n" + skills.GENERATED_MARKER + "\n"
    target.write_text(stale, encoding="utf-8")

    result = skills.materialize(target_dir=str(tmp_path), harnesses=["dsh"])

    assert result["wrote"] == [str(target)]
    refreshed = target.read_text(encoding="utf-8")
    assert "stale body" not in refreshed
    assert "## The evidence model" in refreshed


def test_unknown_harness_is_skipped_with_warning(tmp_path: Path) -> None:
    result = skills.materialize(target_dir=str(tmp_path), harnesses=["ada"])

    assert result["status"] == "ok"
    assert result["wrote"] == []
    assert result["targets"] == {}
    assert any("no skill root" in warning for warning in result["warnings"])


def test_global_scope_uses_home_skills_root(tmp_path: Path) -> None:
    result = skills.materialize(target_dir=str(tmp_path), harnesses=["dsh"], scope="global")
    target = tmp_path / "skills" / "evidencespine" / "SKILL.md"

    assert result["status"] == "ok"
    assert target.exists()
    assert os.path.realpath(str(target)) == skills.canonical_skill_path()


def test_target_dir_canonical_source_is_preferred(tmp_path: Path) -> None:
    local = tmp_path / skills.SKILL_REL_PATH
    local.parent.mkdir(parents=True)
    local.write_text("---\nname: evidencespine\ndescription: local copy.\n---\n\nlocal body\n", encoding="utf-8")

    assert skills.canonical_skill_path(str(tmp_path)) == os.path.realpath(str(local))
    result = skills.materialize(target_dir=str(tmp_path), harnesses=["dsh"])
    link = tmp_path / ".dsh" / "skills" / "evidencespine" / "SKILL.md"
    assert os.path.realpath(str(link)) == os.path.realpath(str(local))
    assert result["source"] == os.path.realpath(str(local))


def test_materialize_never_self_links_canonical_file() -> None:
    repo = _repo_root()
    result = skills.materialize(target_dir=str(repo), harnesses=["dsh"], scope="global")

    assert result["wrote"] == []
    assert (repo / skills.SKILL_REL_PATH).is_file()


def test_usage_import_is_lazy() -> None:
    source = Path(skills.__file__).read_text(encoding="utf-8")
    header = source.split("def ", 1)[0]
    assert "from evidencespine.usage" not in header
    assert "def render_skill_markdown" in source


def test_materialize_dry_run_plans_without_writing(tmp_path: Path) -> None:
    result = skills.materialize(target_dir=str(tmp_path), harnesses=["dsh", "generic"], dry_run=True)

    assert result["status"] == "ok"
    assert result["dry_run"] is True
    assert result["wrote"] == []
    assert len(result["planned"]) == 2
    assert not (tmp_path / ".dsh").exists()
    assert not (tmp_path / ".agents").exists()


def test_install_instructions_merges_and_preserves_user_text(tmp_path: Path) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# My project\n\nSome user notes.\n", encoding="utf-8")

    first = skills.install_instructions(target_dir=str(tmp_path))

    assert first["status"] == "ok"
    assert first["harness"] == "instructions"
    assert first["changed"] is True
    assert first["wrote"] == [str(agents)]
    text = agents.read_text(encoding="utf-8")
    assert text.startswith("# My project")
    assert "Some user notes." in text
    assert skills.AGENTS_MD_BEGIN in text and skills.AGENTS_MD_END in text
    assert "## The evidence model" in text

    second = skills.install_instructions(target_dir=str(tmp_path))
    assert second["changed"] is False
    assert second["wrote"] == []
    assert agents.read_text(encoding="utf-8") == text

    removed = skills.uninstall_instructions(target_dir=str(tmp_path))
    assert removed["action"] == "removed"
    assert agents.read_text(encoding="utf-8") == "# My project\n\nSome user notes.\n"


def test_install_instructions_creates_and_deletes_file(tmp_path: Path) -> None:
    first = skills.install_instructions(target_dir=str(tmp_path))
    agents = tmp_path / "AGENTS.md"

    assert agents.exists()
    assert first["changed"] is True
    removed = skills.uninstall_instructions(target_dir=str(tmp_path))
    assert removed["action"] == "removed"
    assert not agents.exists()

    absent = skills.uninstall_instructions(target_dir=str(tmp_path))
    assert absent["action"] == "absent"


def test_install_instructions_dry_run(tmp_path: Path) -> None:
    result = skills.install_instructions(target_dir=str(tmp_path), dry_run=True)

    assert result["changed"] is True
    assert result["wrote"] == []
    assert not (tmp_path / "AGENTS.md").exists()


def test_render_agents_block_matches_usage_guide() -> None:
    block = skills.render_agents_block()

    assert block.startswith(skills.AGENTS_MD_BEGIN)
    assert block.rstrip("\n").endswith(skills.AGENTS_MD_END)
    assert "## The evidence model" in block
    assert skills.merge_agents_text("") == block
