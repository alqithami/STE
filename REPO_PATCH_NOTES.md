# STE repository cleanup patch

This patch contains replacement repository-management files for the current STE GitHub repository.

## Replace directly

- `requirements.txt`
- `Makefile`
- `README.md`
- `README_NEURIPS_EXPERIMENTS.md`
- `.gitignore`
- `LICENSE`
- `CITATION.cff`
- `.github/workflows/ci.yml`

## Add

- `outputs/.gitkeep`
- `plots/.gitkeep`
- `data/README.md`
- `paper/README.md`
- `scripts/clean_generated.sh`
- `docs/repo_release_checklist.md`

## Remove from git tracking

Run from the repository root:

```bash
git rm -r --cached __pycache__ outputs plots || true
find . -type d -name __pycache__ -prune -exec rm -rf {} +
mkdir -p outputs plots
touch outputs/.gitkeep plots/.gitkeep
git add .gitignore outputs/.gitkeep plots/.gitkeep
```

Then add the replacement files and run:

```bash
python -m pip install -r requirements.txt
make test
make smoke
git status
```
