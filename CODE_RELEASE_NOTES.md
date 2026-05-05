# STE reviewer-code release notes

This repository is the experiment and paper-support package for **From Rankings to Cores: Soft Tournament Equilibrium for Non-Transitive Agent Evaluation**.

## What is implemented

The code implements the experiment layer corresponding to the NeurIPS draft:

- hard Top Cycle and Uncovered Set algorithms;
- plug-in soft STE using soft reachability and soft covering;
- posterior-edge STE membership: sample hard tournaments from Beta edge-direction posteriors and average TC/UC membership indicators;
- ranking/rating baselines converted to top-|C| sets for controlled synthetic comparisons;
- synthetic planted cyclic-core generation;
- missing-pair, label-noise, margin, and sample-size sweeps;
- ablations over edge estimator, reachability operator, path length, and temperature;
- bootstrap stability diagnostics;
- pairwise and membership reliability diagnostics;
- posterior threshold-sensitivity diagnostics;
- sanity and negative controls;
- Chatbot-Arena-style human-preference diagnostics;
- AgentBench-style execution-log diagnostics;
- runtime scaling.

## Claims that the code can support

The code is designed to support the paper's narrow empirical claim:

> In controlled cyclic-core regimes, posterior-edge STE recovers the hard tournament-theoretic core more reliably than scalar ranking/rating baselines once moderate pairwise evidence is available; real preference/execution logs can then be summarized diagnostically as tournament cores rather than forced rankings.

The code does **not** certify absolute model quality, factuality, safety, fairness, or universal agent superiority.

## Reviewer execution path

Run a quick check:

```bash
make install
make smoke
```

Run the recommended final synthetic suite:

```bash
make final
```

Regenerate paper tables and figures from existing outputs:

```bash
python -m ste_neurips.neurips_suite summarize --out outputs/neurips_final
```

## Real data inputs

Arena/human-preference CSVs should have:

```text
model_a, model_b, winner, category(optional)
```

AgentBench-style CSVs should have:

```text
environment, agent, task_id, score
```

or replace `score` with `success` or `status`.

## Anonymization

For double-blind submission, do not include private raw logs or non-public benchmark dumps in this repository. The example CSVs are tiny schema examples only. Use external paths for real data and report provenance in the paper.

## v3 real-dataset upgrade

This version adds a real-dataset manifest runner and stronger reviewer-facing comparison support.

New commands:

```bash
python -m ste_neurips.neurips_suite scorelog --input data/osworld_scores.csv --out outputs/osworld
python -m ste_neurips.neurips_suite real-suite --manifest configs/real_datasets_manifest_template.yaml --out outputs/real_suite_final
```

New baselines:

- Schulze strongest-path Condorcet method
- Minimax / Simpson Condorcet method
- Ranked Pairs / Tideman-style locking
- Approximate Kemeny-Young local search

New real-data outputs:

- `real_selection_diagnostics.csv`: selected-set dominance/error diagnostics.
- `all_real_scores.csv`: merged per-agent method scores across real datasets.
- `all_real_selection_diagnostics.csv`: merged external-attack/stability diagnostics.
- `all_real_high_confidence_cycles.csv`: merged high-confidence cycle witnesses.

The intended final evidence package combines direct human pairwise preference data, LLM-as-judge pairwise data, and execution-based agent benchmarks. Real-data results must be reported as diagnostics unless ground-truth core labels are available.
