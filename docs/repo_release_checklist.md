# STE repository release checklist

Before sharing the repository with reviewers or linking it from the manuscript:

1. `python -m pip install -r requirements.txt` succeeds in a fresh environment.
2. `make test` passes.
3. `make smoke` completes and writes `outputs/neurips_smoke/summary_report.md`.
4. The repository contains no `__pycache__`, `.DS_Store`, `.venv`, or large generated outputs.
5. `outputs/` and `plots/` are ignored except for `.gitkeep`.
6. The README commands match the actual script names.
7. The manuscript does not claim calibrated probabilities unless reliability diagnostics support that claim.
8. Real-data outputs are described as diagnostics unless ground-truth core labels are available.
9. The reviewer artifact includes exact configs, seeds, raw outputs, and commit hash.
10. If the paper source is included, the conference version is anonymized.
