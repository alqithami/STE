# Soft Tournament Equilibrium (STE)

Research software for **Soft Tournament Equilibrium: Differentiable Set-Valued
Inference for Non-Transitive Pairwise Comparisons**, by Saad Alqithami.
[Public preprint: arXiv:2604.04328](https://arxiv.org/abs/2604.04328).

The current research artifact preserves separate synthetic and human-profile
studies, their original decision rules, and negative findings. Differentiating
through tournament structure can help some synthetic comparisons; it has not
established a general advantage over count/posterior controls on the examined
human profiles. See [the revision record](docs/ARXIV_REVISION.md) for the scope
of the added evidence and supporting proofs.

## Completed evidence and corrected software

The additional posterior/certificate study completed using **frozen v1** code.
At 10% of recorded ballots, the six-source macro summaries were:

| Native system | F1 | Exact set recovery |
| --- | ---: | ---: |
| Jeffreys posterior + GFM | **0.8742** | 0.7656 |
| Learned independent edges + GFM | 0.8704 | 0.7614 |
| Learned two-component mixture + GFM | 0.8366 | 0.7301 |
| Raw-count UC | 0.8481 | **0.7950** |

GFM is the established expected-F1 decision rule, not a new tournament solution.
The cohort contains 36 human ordinal profiles from six sources, with 33
singleton and only three selective UC references. These are exploratory finite
record reconstruction results. Repeated subsamples and initializations are not
additional independent profiles. No p-values, population superiority or
posterior-calibration claims follow from them. All native baselines and
cardinality limitations remain part of the study.

An independent review verified the native outputs and exposed a defect in the
**secondary v1 mixture hard-q decoder**: rounding could create self-loops and
empty sets. Its entire aggregate is excluded. The native GFM decisions sampled
strict irreflexive tournaments and remain verified. Original outputs and source
are retained; corrected calculations are never substituted silently.

[Corrected forward software v2](experiments/posterior_certificates/README.md)
repairs the marginal/graph contracts and strengthens the audit. **V2 has software
QA only; no v2 production training or results are claimed.** Its changed fitting
numerics require a fresh run before reporting v2 results. Read the
[correction notice](experiments/posterior_certificates/CORRECTION_NOTICE.md).

## Reproduction map

| Directory | Role |
| --- | --- |
| `experiments/posterior_certificates/` | Corrected v2 forward pipeline; future runs use a new result root. |
| `experiments/frozen/posterior_certificates_v1/` | Original software for the completed posterior/certificate study. |
| `experiments/frozen/hard_uc_confirm_v1/` | Original separate hard-UC confirmation pipeline. |
| `experiments/frozen/real_profiles_learn_v1/` | Original exploratory human-profile learning pipeline. |
| `experiments/frozen/uc_equal_compute_v1/` | Original equal-allocation soft-UC comparison pipeline. |
| `paper/` | Public manuscript material and paper-specific documentation. |
| `ste_ops/`, `ste_neurips/`, `configs/` | Legacy operators and original CPU/NumPy suite, preserved below. |

Use each frozen study's own instructions, configuration, and original
checkpoints for replay. Their data targets, readouts, budget designs and
inferential scopes differ; results must not be pooled into a new endpoint.
No completed study is recreated by running the legacy root `make` targets.

## Current software checks

From the repository root, with Python 3.11 or 3.12 and a compatible PyTorch
installation:

```bash
cd experiments/posterior_certificates
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

The directory's [README](experiments/posterior_certificates/README.md) supplies
software smoke, source checksums, fresh CUDA launch, completion, archive, and
download commands. A smoke pass or idle GPU is not production completion.
Training budgets, source/data/checkpoint hashes, selected-state replay, saved
joint predictions and independently reconstructed summaries must be inspected.
Generated experiment outputs remain separate from version-controlled code.

## Citation

Use [CITATION.cff](CITATION.cff) and cite the
[preprint](https://arxiv.org/abs/2604.04328). The repository revision and an arXiv
replacement are separate actions; repository contents do not assert that a new
arXiv version has been posted.

## Legacy pipeline documentation (preserved)

The historical instructions below describe the original CPU/NumPy suite.
Their old descriptions of a “current manuscript” or “final” run refer to that
legacy suite. They do not replace the study-specific protocols above, and
oracle-size diagnostics do not represent deployed selection accuracy.

<details>
<summary>Original repository README</summary>

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

</details>
