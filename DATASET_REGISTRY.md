# STE real-dataset registry and reviewer-facing experiment plan

This registry defines the real-data evidence package for the STE NeurIPS paper. The central reviewer-facing goal is not only to show synthetic planted-core recovery, but also to demonstrate that the method behaves sensibly on multiple real evaluation regimes where rankings are commonly reported but cyclic evidence may exist.

## Priority tiers

### Tier 1: direct human pairwise preference data

1. **Chatbot Arena Conversations 33K**. Direct human pairwise preferences between model responses. Use this as the primary human-comparison dataset.
2. **Arena Human Preference 55K**. Larger human pairwise preference set from the LMSYS/Chatbot Arena preference-prediction setting. Use as a robustness replication of the human-preference analysis.

### Tier 2: automatic pairwise judging data

3. **Arena-Hard-Auto / Arena-Hard-v2 pairwise outputs**. Treat these as judge-generated pairwise labels, not human ground truth. Report separately from human comparisons.
4. **MT-Bench pairwise mode / FastChat pairwise judgments**, where available. Treat as LLM-judge evidence.

### Tier 3: execution-based agent benchmarks converted to pairwise tournaments

5. **AgentBench**. Convert same-environment, same-task score differences into pairwise wins.
6. **WebArena**. Convert task success / score logs into pairwise wins by task and website category.
7. **OSWorld**. Convert real desktop/web task outcomes into pairwise wins by domain.
8. **SWE-bench Verified / Lite**. Convert pass/fail issue-resolution outcomes into pairwise wins by repository or issue family.

## Required paper tables

**Table R1: Real dataset summary.** Dataset, source, label type, human vs judge vs execution, number of agents, number of tasks/prompts, decisive pairwise comparisons, ties, categories/environments, and license/access notes.

**Table R2: Cyclicity and core structure.** For each dataset/category: high-confidence 3-cycle count, cycle rate per observed triple, hard TC size, hard UC size, posterior-edge TC/UC expected core size, and pair coverage.

**Table R3: Algorithm comparison on real data.** For each dataset: selected-set stability under bootstrap, top-1 entropy, mean pairwise selected-set Jaccard, selected-vs-outside external attack rate, selected dominance rate, and dominance gap.

**Table R4: Synthetic + real bridge.** Synthetic recovery/F1 when ground truth is known, plus real stability/error diagnostics when ground truth is unknown.

## Required baselines

The default real suite includes:

- Win rate / empirical mean
- BTL / Bradley-Terry maximum likelihood
- Elo
- TrueSkill, if the optional package is installed; otherwise it falls back to Elo
- Copeland
- Rank Centrality
- PageRank / Markov centrality
- HodgeRank
- Schulze strongest-path Condorcet method
- Minimax / Simpson Condorcet method
- Ranked Pairs / Tideman-style locking
- Approximate Kemeny-Young local search
- Hard Top Cycle and hard Uncovered Set
- STE plug-in Top Cycle / Uncovered Set
- STE posterior-edge Top Cycle / Uncovered Set

## Required statistical controls

For every real dataset report:

- At least 1,000 bootstrap resamples for final results.
- Category/environment stratification when categories are available.
- Pair-coverage statistics so sparse comparisons are not confused with true indifference.
- Ties and undecided comparisons separately from decisive wins.
- High-confidence cycle audit with count and posterior-confidence thresholds.
- Selected-set stability across bootstrap resamples.
- External attack/error rate: how often agents outside a selected top set beat selected agents in observed comparisons.

## Claims allowed from real data

Allowed:

- The dataset contains cyclic preference/execution structure under stated thresholds.
- STE produces stable set-valued diagnostics on this dataset.
- Scalar ranking methods compress away structure that is visible in tournament-core diagnostics.
- On real datasets without ground-truth cores, results are diagnostic rather than oracle accuracy claims.

Not allowed:

- Claiming that STE recovers a true real-world core unless the dataset has ground-truth core labels.
- Claiming membership scores are calibrated probabilities without reliability evidence.
- Claiming human preference results from LLM-judge datasets.
