"""Enable `python -m evidencespine` as a CLI entry point."""

from __future__ import annotations

from evidencespine.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
