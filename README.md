# Soft Tournament Equilibrium (STE)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2604.04328-b31b1b.svg)](https://arxiv.org/abs/2604.04328)

This repository contains the runnable code and experiment pipeline for **Soft Tournament Equilibrium (STE)**: set-valued evaluation for non-transitive pairwise comparisons. The main empirical pipeline is designed to reproduce the synthetic planted-core benchmark, ablations, bootstrap diagnostics, runtime scaling, and optional real-data diagnostics used in the current STE manuscript.

## Repository status

This repository is intended to be the **code and experiment artifact** for the paper. Generated outputs are intentionally excluded from version control; publishable outputs should be regenerated from the committed configs and archived separately as a reviewer artifact or release.

## Main components

```text
ste_neurips/        NeurIPS-style synthetic and real-data experiment runner
ste_ops/            Core STE operators
baselines/          Ranking/rating baselines
configs/            Smoke, laptop, server, and final experiment configs
data/               Dataset loaders and templates; do not commit private/raw dumps
scripts/            Convenience run scripts
tests/              Sanity tests
tools/              Auditing and placeholder-detection utilities
outputs/            Generated locally; ignored by git except .gitkeep
plots/              Generated locally; ignored by git except .gitkeep
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Apple Silicon, the synthetic pipeline is CPU/NumPy-based and does not require GPU setup. Server-scale experiments can be run on CPU nodes unless you add neural contextual models.

## First checks

Run the unit/sanity tests and a small smoke experiment before launching larger jobs:

```bash
make test
make smoke
```

Expected smoke outputs are written under:

```text
outputs/neurips_smoke/
```

## Synthetic experiments

Laptop/Mac run:

```bash
make mac
```

Recommended final synthetic run:

```bash
make final
```

Server-scale run:

```bash
make server
```

The synthetic suite writes:

```text
oracle_sanity.csv
synthetic_recovery.csv
synthetic_ablation.csv
bootstrap_stability.csv
runtime_scaling.csv
negative_controls.csv
synthetic_threshold_sensitivity.csv
synthetic_pairwise_reliability.csv
synthetic_membership_reliability.csv
synthetic_edge_margins.csv
summary_report.md
paper_tables.tex
figures/*.png
```

## Real-data diagnostics

Arena-style human preference CSV schema:

```text
model_a, model_b, winner, category
```

Run:

```bash
bash scripts/run_arena_human_preferences.sh /path/to/arena.csv outputs/arena_full
```

AgentBench-style score log schema:

```text
environment, agent, task_id, score
```

or use `success` / `status` instead of `score`.

Run:

```bash
bash scripts/run_agentbench_logs.sh /path/to/agentbench_scores.csv outputs/agentbench_full
```

Real-data outputs are diagnostics. They should be reported as evidence about cyclic structure and stability, not as ground-truth core accuracy.

## Reviewer-safety rules

Do not claim that STE scores are calibrated probabilities unless reliability diagnostics support that claim. For real data, report STE outputs as diagnostic membership scores. In synthetic experiments, ranking baselines can be converted to top-|C| sets using the true core size; this is favorable to the baselines and should be stated explicitly.

## Cleaning generated files

```bash
make clean-generated
```

To remove generated files from git if they were accidentally committed:

```bash
git rm -r --cached __pycache__ outputs plots || true
find . -type d -name __pycache__ -prune -exec rm -rf {} +
mkdir -p outputs plots
touch outputs/.gitkeep plots/.gitkeep
git add .gitignore outputs/.gitkeep plots/.gitkeep
```

## Citation

Use `CITATION.cff` once the public paper/preprint identifier is finalized.
