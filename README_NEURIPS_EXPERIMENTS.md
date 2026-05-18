# STE experiment pipeline

This document describes the reviewer-shareable experiment pipeline for Soft Tournament Equilibrium (STE). It is aligned with the STE and is designed to support the central empirical claim:

> In cyclic pairwise-evaluation settings, the useful object is often a set-valued tournament core rather than a forced scalar ranking.

## Implemented STE modes

The code distinguishes the following modes.

1. **Hard-threshold TC/UC**: threshold empirical pairwise probabilities at 1/2 and run classical Top Cycle / Uncovered Set.
2. **Plug-in soft STE**: compute differentiable Top-Cycle and Uncovered-Set scores from a soft edge matrix.
3. **Posterior-edge STE**: use edge-direction posterior evidence to avoid treating missing or ambiguous comparisons as confident bidirectional reachability. This is the main finite-sample reporting estimator.

## Synthetic outputs

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

## Real-data outputs

Arena/human-preference runs write:

```text
real_arena_scores.csv
real_arena_bootstrap.csv
real_arena_high_confidence_cycles.csv
real_arena_report.md
```

AgentBench-style score-log runs write:

```text
agentbench_pairwise.csv
agentbench_scores.csv
agentbench_bootstrap.csv
agentbench_high_confidence_cycles.csv
agentbench_status_counts.csv
agentbench_report.md
```

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Smoke test

```bash
make smoke
```

Expected output directory:

```text
outputs/neurips_smoke/
```

## Laptop-scale synthetic run

```bash
make mac
```

This uses `configs/synthetic_mac.yaml`.

## Recommended final synthetic run

```bash
make final
```

This uses `configs/synthetic_final_neurips.yaml` and should be the preferred source for final synthetic tables when compute is available.

## Server-scale run

```bash
make server
```

Run this only after `make smoke` and `make final` complete successfully.

## Real human-preference / Arena-style diagnostics

CSV schema:

```text
model_a, model_b, winner, category(optional)
```

Run:

```bash
bash scripts/run_arena_human_preferences.sh /path/to/arena.csv outputs/arena_full
```

Use `real_arena_high_confidence_cycles.csv` to report concrete non-transitivity examples if any are found under the configured count/confidence thresholds.

## AgentBench-style execution-log diagnostics

CSV schema:

```text
environment, agent, task_id, score
```

or use `success` / `status` in place of `score`.

Run:

```bash
bash scripts/run_agentbench_logs.sh /path/to/agentbench_scores.csv outputs/agentbench_full
```

## One-command complete run

After installing requirements, run:

```bash
bash scripts/run_complete_neurips.sh
```

This should run tests, the smoke suite, the final synthetic suite, any configured real-data downloads, the real-data suite, and reviewer-artifact packaging. Dataset download details belong in `DATA_DOWNLOADS.md`. Execution-agent benchmarks require per-agent per-task score logs; templates should be replaced by real run logs before enabling those datasets in the manifest.
