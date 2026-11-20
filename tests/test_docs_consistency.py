"""Docs consistency guards: README examples must be copy-paste correct.

These tests are intentionally active assertions (no xfail for the README
quickstart checksum): documentation that ships a checksum must ship the right
one, otherwise users following the quick start get an ungrounded claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
HARNESS_MATRIX = ROOT / "docs" / "HARNESS_MATRIX.md"

_EVIDENCE_ITEM_RE = re.compile(
    r"--evidence-item-json\s+(?P<quote>['\"])(?P<payload>.+?)(?P=quote)",
    re.DOTALL,
)
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _code_fences(text: str) -> list[str]:
    return re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)


def _evidence_item_examples(text: str) -> list[dict]:
    examples: list[dict] = []
    for fence in _code_fences(text):
        for match in _EVIDENCE_ITEM_RE.finditer(fence):
            payload = match.group("payload").strip()
            try:
                parsed = json.loads(payload)
            except ValueError as exc:  # pragma: no cover - defensive
                raise AssertionError(f"README --evidence-item-json is not valid JSON: {exc}") from exc
            if isinstance(parsed, dict):
                examples.append(parsed)
    return examples


def test_readme_quickstart_checksum_matches_excerpt() -> None:
    examples = _evidence_item_examples(README.read_text(encoding="utf-8"))
    assert examples, "README must document an --evidence-item-json quickstart example"
    checked = 0
    for item in examples:
        excerpt = item.get("excerpt")
        checksum = item.get("checksum")
        assert isinstance(excerpt, str) and excerpt, f"evidence item missing excerpt: {item}"
        assert isinstance(checksum, str) and checksum, f"evidence item missing checksum: {item}"
        digest = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
        accepted = {digest, f"sha256:{digest}"}
        assert checksum.lower() in accepted, (
            "README quickstart checksum does not match sha256(excerpt): "
            f"excerpt={excerpt!r} documented_checksum={checksum!r} expected=sha256:{digest}"
        )
        checked += 1
    assert checked >= 1


#: Generated benchmark artifacts are gitignored; documentation must link
#: committed files, not local run output.
_GENERATED_PREFIXES = ("benchmarks/results/",)


def test_readme_doc_links_exist() -> None:
    text = README.read_text(encoding="utf-8")
    missing: list[str] = []
    seen: set[str] = set()
    for target in _MARKDOWN_LINK_RE.findall(text):
        raw = target.strip()
        if raw.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path_part = raw.split("#", 1)[0]
        if not path_part.lower().endswith(".md"):
            continue
        if path_part in seen:
            continue
        seen.add(path_part)
        normalized = path_part.lstrip("./")
        if normalized.startswith(_GENERATED_PREFIXES):
            continue
        candidate = (ROOT / path_part.lstrip("./")).resolve()
        if not candidate.exists():
            missing.append(path_part)
    assert not missing, f"README references missing documentation files: {missing}"


HARNESS_ALIASES: dict[str, tuple[str, ...]] = {
    "claude-code": ("claude-code", "claude code"),
    "opencode": ("opencode",),
    "codex": ("codex",),
    "cursor": ("cursor",),
    "vscode": ("vscode", "vs code"),
    "vscode-copilot": ("vscode-copilot", "vscode copilot", "copilot"),
    "git": ("git",),
    "agents-md": ("agents-md", "agents.md", "agents md"),
    "dsh": ("dsh", "deepseek harness"),
    "generic": ("generic", "generic mcp"),
    "instructions": ("instructions", "instruction", "agents.md", "agents-md"),
}


def _scan_parser_choices() -> list[str]:
    from evidencespine.cli import build_parser

    def _walk(parser: argparse.ArgumentParser) -> list[str]:
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for choice in action.choices.values():
                    found = _walk(choice)
                    if found:
                        return found
            if getattr(action, "dest", "") in {"harness", "harness_target"} and getattr(action, "choices", None):
                return [str(x) for x in action.choices]
        return []

    return [name for name in _walk(build_parser()) if name != "all"]


def _resolve_harness_choices() -> list[str]:
    # Prefer the canonical registry kept in sync with the CLI.
    try:
        from evidencespine.harness.install import KNOWN_HARNESSES

        registered = [str(name) for name in KNOWN_HARNESSES if str(name) != "all"]
    except Exception:  # pragma: no cover - fall back during harness refactors
        registered = []
    if not registered:
        registered = _scan_parser_choices()
    return registered


def test_harness_matrix_covers_registered_harnesses() -> None:
    registered = _resolve_harness_choices()
    assert registered, "could not resolve registered harness names from the CLI parser"
    if not HARNESS_MATRIX.exists():
        pytest.xfail(
            "docs/HARNESS_MATRIX.md has not landed yet (docs task); it must cover: "
            + ", ".join(registered)
        )
    text = HARNESS_MATRIX.read_text(encoding="utf-8").lower()
    uncovered = []
    for name in registered:
        aliases = HARNESS_ALIASES.get(name, (name, name.replace("-", " "), name.replace("-", "_")))
        if not any(alias.lower() in text for alias in aliases):
            uncovered.append(name)
    assert not uncovered, f"docs/HARNESS_MATRIX.md does not cover registered harnesses: {uncovered}"
