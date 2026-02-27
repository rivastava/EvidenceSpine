# Publish Guide

## 1) Validate locally

```bash
cd oss/evidencespine
python -m venv .venv
source .venv/bin/activate
pip install -U pip build twine pytest
pip install -e .
pytest
python -m build
python -m twine check dist/*
```

Optional benchmark artifact:

```bash
PYTHONPATH=src python benchmarks/bench_evidencespine.py --out benchmarks/results/latest.json
```

## 2) Create GitHub repo
- Create a new public repo, e.g. `evidencespine`.
- Push this folder as repo root.

```bash
cd oss/evidencespine
git init
git add .
git commit -m "Initial OSS release: EvidenceSpine v0.1.0"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## 3) Optional PyPI publish

```bash
python -m build
python -m twine upload dist/*
```

## License note
- This project is source-available under **PolyForm Noncommercial 1.0.0**.
- Commercial usage requires separate permission.
- Licensing contact: `rivastava0@gmail.com`

## 4) Versioning policy
- Bump patch for bug fixes.
- Bump minor for backward-compatible features.
- Bump major for protocol-breaking changes.
