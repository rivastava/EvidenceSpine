from __future__ import annotations

from pathlib import Path

from evidencespine.grounding import ground_file


def grounded_item(tmp_path: Path, name: str = "src/evidence.py", text: str = "def run():\n    return True\n", start: int = 1, end: int = 2) -> dict:
    """Create a temp file and return a grounded evidence item for its lines."""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    item = ground_file(name, start, end, source_root=str(tmp_path))
    assert item is not None, f"grounding failed for {name}#L{start}-{end}"
    return item
