from __future__ import annotations

from pathlib import Path

from evidencespine.usage import usage_guide_markdown


def test_committed_agents_md_matches_canonical_guide() -> None:
    """The committed AGENTS.md must never drift from the canonical guide."""
    repo_root = Path(__file__).resolve().parents[1]
    agents_md = (repo_root / "AGENTS.md").read_text(encoding="utf-8")
    assert agents_md == usage_guide_markdown(), (
        "AGENTS.md drifted from usage_guide_markdown(); regenerate with: "
        "python -c \"from evidencespine.usage import usage_guide_markdown; "
        "open('AGENTS.md','w').write(usage_guide_markdown())\""
    )


def test_usage_guide_covers_decision_rules() -> None:
    guide = usage_guide_markdown()
    for rule in (
        "Before asserting project state",
        "When claiming something is fixed/done/verified",
        "After editing files a claim cites",
        "When a test/gate verifies a claim",
        "On receiving a handoff packet",
        "On role change or session end",
        "When claims conflict",
    ):
        assert rule in guide, f"missing decision rule: {rule}"
    assert "`verified` REQUIRES grounding" in guide
    assert "//guide" in guide
