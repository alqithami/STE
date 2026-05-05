# STE NeurIPS-grade experiment pipeline

This package contains the reviewer-shareable experiment pipeline for **Soft Tournament Equilibrium (STE)**. It is aligned with the current NeurIPS draft and implements the empirical package needed to defend the paper's central claim: pairwise agent evaluation can be non-transitive, so the useful object is often a set-valued tournament core rather than a forced scalar ranking.

## Implemented STE modes

The code distinguishes three modes used in the paper.

1. **Hard-threshold TC/UC**: threshold empirical pairwise probabilities at 1/2 and run classical Top Cycle / Uncovered Set.
2. **Plug-in soft STE**: compute differentiable Top-Cycle and Uncovered-Set scores from a soft edge matrix using soft reachability and soft covering.
3. **Posterior-edge STE**: sample hard tournaments from Beta edge-direction posteriors and average hard TC/UC membership indicators. This is the main finite-sample reporting estimator.

The default main method is `ste_posterior_edge_uc`, which corresponds to the paper's posterior-edge Uncovered-Set membership estimator.

## What the pipeline produces

Synthetic suite outputs:

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

Real-data commands produce:

```text
real_arena_scores.csv
real_arena_bootstrap.csv
real_arena_high_confidence_cycles.csv
real_arena_report.md
agentbench_pairwise.csv
agentbench_scores.csv
agentbench_bootstrap.csv
agentbench_high_confidence_cycles.csv
agentbench_status_counts.csv
agentbench_report.md
```

## Installation

```bash
cd ste_neurips_experiment_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke test

Run this first. It checks all major code paths on a tiny grid.

```bash
make smoke
```

Expected outputs are written to `outputs/neurips_smoke/`.

## Laptop-scale run

```bash
make mac
```

This uses `configs/synthetic_mac.yaml`: 40 seeds, multiple sample sizes, missingness/noise/margin sweeps, ablations, bootstrap stability, extra diagnostics, controls, and runtime scaling.

## Recommended final synthetic run

```bash
make final
```

This uses `configs/synthetic_final_neurips.yaml`. It is the recommended setting for final NeurIPS tables if compute is available.

## Server-scale run

```bash
make server
```

This is larger than needed for most drafts. Use it only after the smoke and final configs succeed.

## Real human-preference / Arena-style diagnostics

CSV schema:

```text
model_a, model_b, winner, category(optional)
```

Run:

```bash
bash scripts/run_arena_human_preferences.sh /path/to/arena.csv outputs/arena_full
```

Use `real_arena_high_confidence_cycles.csv` to provide concrete non-transitivity examples if any are found under the configured count/confidence thresholds.

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

The command converts overlapping task scores into pairwise comparisons within each environment.

## Paper integration

- `paper/ste_neurips2026_v7/` contains the current NeurIPS draft source from the STE build.
- `paper/experiment_to_paper_map.md` maps outputs to main-paper/appendix claims.
- `paper/paper_tables.tex` inside each output directory contains ready-to-paste LaTeX tables.

## Reviewer-safety notes

- Do not claim membership scores are calibrated probabilities unless reliability diagnostics support that claim.
- For real data, report STE outputs as diagnostics rather than ground truth.
- Ranking baselines receive the true core size in synthetic top-|C| conversion, which is favorable to them.
- The random-label negative control should not show strong planted-core recovery. If it does, treat the main result as suspect until debugged.


## Real-dataset suite support

The pipeline can now run multiple real datasets from a single manifest:

```bash
python -m ste_neurips.neurips_suite real-suite \
  --manifest configs/real_datasets_manifest_template.yaml \
  --out outputs/real_suite_final
```

Supported real-data input types:

- `pairwise`: human or judge pairwise preferences with columns for two agents and a winner.
- `scorelog`: task-level scores with columns for environment, agent, task_id, and score/success/status. The runner converts same-task scores into pairwise comparisons.

The standard method list includes STE variants plus BTL, Elo, TrueSkill, Rank Centrality, HodgeRank, PageRank/MC-style, Copeland, Schulze, Minimax/Simpson, Ranked Pairs, and approximate Kemeny local search.

Real-data outputs include high-confidence 3-cycle audits, bootstrap top-set stability, selected-set dominance/error diagnostics, and per-environment/category summaries. These diagnostics should be reported as real-data evidence of cyclic structure and stability, not as ground-truth core accuracy.

## One-command complete run with dataset download

After installing requirements, run:

```bash
export HF_TOKEN=hf_your_token_if_needed
bash scripts/run_complete_neurips.sh
```

This runs tests, smoke synthetic checks, the final synthetic suite, downloads/standardizes available public pairwise preference datasets, runs the real-data suite, and builds a reviewer artifact.

Dataset download details are in `DATA_DOWNLOADS.md`. Human preference datasets can be downloaded automatically when Hugging Face access is available. Execution-agent benchmarks require per-agent per-task score logs; templates are written under `data/` and should be replaced by real run logs before enabling those datasets in the manifest.
