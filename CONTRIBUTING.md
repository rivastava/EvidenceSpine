# Contributing to EvidenceSpine

Thanks for contributing.

## Scope
EvidenceSpine is a memory and handoff quality layer. Keep changes bounded:
- prioritize correctness and backward compatibility
- avoid unnecessary runtime dependencies
- avoid framework-specific lock-in

## Development
```bash
cd oss/evidencespine
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
PYTHONPATH=src pytest -q
```

## Pull Requests
- Keep patches narrow and test-backed.
- Update docs when public behavior changes.
- Preserve existing runtime contracts unless a breaking change is explicitly approved.

## License and DCO
By contributing, you agree your contribution is licensed under Apache-2.0.

This repository uses the Developer Certificate of Origin (DCO). Sign off each commit:

```bash
git commit -s -m "feat: your change"
```

The `-s` flag adds a `Signed-off-by` line required for contribution provenance.
