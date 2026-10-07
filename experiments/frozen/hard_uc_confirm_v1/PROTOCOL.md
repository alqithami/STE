# STE Hard UC prospective confirmation v1

This is a new synthetic experiment, not a reanalysis, an extension of the previous inferential family, or evidence already obtained. Its fresh root seed is fixed at 9872513. CPU smoke checks and CUDA software checks are engineering QA only and never scientific outcomes. Completion never implies superiority.

## Question and six complete systems

Does soft-UC supervision learn a relation whose prospectively selected hard-UC decoder improves selective set prediction over native direct heads under equal measured training/development budgets?

The original audited generator, PairModel architecture, pair likelihood, soft UC operator and all training-loss formulas are retained unchanged in frozen/ste. No differentiable hard operator is introduced. STE continues to optimize the original soft-UC binary cross-entropy plus pair likelihood; only checkpoint/lambda selection and final deployment use hard UC. The auxiliary models retain their original head-supervision objectives.

| System | Training objective | Development selection and deployed readout |
| --- | --- | --- |
| pair-hard | Pair likelihood only | Hard UC of the predicted majority relation |
| aux-direct | Pair likelihood + ordinary head BCE | Native direct head with development-only absolute threshold |
| relational-direct | Pair likelihood + relational head BCE | Native relational head with development-only absolute threshold |
| ste-hard | Pair likelihood + original soft-UC BCE | Hard UC of the predicted majority relation |
| aux-hard | Pair likelihood + ordinary head BCE | Hard UC of the predicted majority relation |
| relational-hard | Pair likelihood + relational head BCE | Hard UC of the predicted majority relation |

All six are separately timed and separately trained; the two auxiliary hard systems are not free extra selections from native-head fits. Their decoder-specific development selection avoids penalizing a hard readout through native-head checkpoint selection. Structural-loss alignment remains part of the treatment and is not a proof of universal superiority.

The majority relation has edge a→b exactly when predicted P_ab > 0.5. Each stored direction is thresholded independently. Float32 reciprocity is approximate: one stored direction can round to exactly 0.5 while its reverse remains slightly above 0.5. Pairs with neither direction above 0.5 are unoriented. Hard UC uses the unchanged graph oracle, without a membership score cutoff, oracle cardinality, or per-size calibration. The metadata threshold 0.5 denotes only the majority edge rule. Upper-triangle exact-tie counts, actually unoriented pair counts and one-sided boundary-rounding counts are reported. Tied predictions therefore form an incomplete directed graph, not a strict tournament.

## Independent units, pairing and splits

Twelve independent collections, each with three paired initialization seeds. Every collection contains 512 training and 128 development graphs at n=12 and 256 held-out test graphs at each of n=24 and n=48. Families are ordered, planted, separated, rank-mixture and random; count intensity 10, missing probability 0.25 and tie probability 0.2. No graph or initialization is treated as an independent inferential unit. Three initialization-specific graph means are averaged equally within collection.

All models and lambda candidates share identical encoder/pair initial states within a collection/initialization. Ordinary/relational head states follow the unchanged architectures. Identical semantic seeds give paired minibatch permutations for each epoch; method-specific update counts may differ under equal time. Candidate order rotates over all six systems and reverses every six paired units; lambda order alternates. Entire collections, including all systems and initializations, are the only allowed shards.

Every system's choices for all three initializations in a collection are persisted and hashed before any test split in that collection is generated or evaluated. Hard systems maximize development mean per-case F1 on selective non-singleton UC graphs (1 < core size < n). Direct heads use the same development stratum and the unchanged threshold grid. Ties select earlier checkpoint, then smaller lambda; direct thresholds break remaining ties by proximity to 0.5 then smaller threshold.

## Equal measured budgets

Exactly 220 charged wall-clock seconds per complete system and paired initialization. Pair-hard fits its one zero-core-weight candidate for 220 seconds; each supervised system fits two candidates, lambda 0.1 and 1.0, for 110 seconds each. Five synchronized checkpoint fractions are 0.025, 0.1, 0.25, 0.5 and 1.0. Learning rate 0.001, hidden dimension 32, training batch 16, evaluation batch 8, gradient clipping 5 and canonical soft temperature 0.035 remain unchanged.

Charged cost includes training, development prediction, hard decoding or direct threshold selection, checkpoint/development serialization, in-loop bookkeeping, and final cross-candidate winner selection/freezing. Final winner scans are measured per system; shared freeze and candidate-universe serialization cost is divided equally among all six systems. These costs enter complete-system charged totals without changing any candidate allocation. The cost-audit record itself is post-budget bookkeeping. Dataset generation, common warmup, model initialization/data transfer, initial-state hashing, post-budget integrity bookkeeping and test evaluation are recorded separately. GPU synchronization makes asynchronous work part of elapsed cost; CUDA event times are additional diagnostics. The scheduler reserves the largest prior checkpoint cost before each target and stops training at a synchronized minibatch boundary. Both undershoot and overshoot are reported.

The fixed validity tolerance is ±2% for every candidate and every complete-system total, including final selection/freezing cost. A missed training interval between checkpoints also invalidates confirmatory inference. Measured work quanta explain budget error but never relax tolerance. Invalid-budget runs are retained with descriptive summaries and formal p-values/intervals suppressed. Aborted attempts are archived; resumed candidates restart from their fixed initial state, and previous expenditure remains visible rather than silently reused.

Total allocated candidate time is 12×3×6×220 = 47,520 seconds (13 h 12 min). Two disjoint six-collection shards allocate 6 h 36 min per server. Setup, QA, dataset generation, test evaluation, independent replay, analysis and packaging add time; these are allocations, not guaranteed completion times.

## Primary inference, fixed before launch

Exactly two two-sided primary comparisons, selective non-singleton UC mean F1 at n=24:

1. ste-hard minus aux-direct.
2. ste-hard minus relational-direct.

Use twelve paired collection differences, exact 2^12 sign-flip p-values and Holm correction across this separate two-test family. Report 95% t intervals and all twelve differences, including unfavorable values. Sign-flip validity assumes independent collection differences exchangeable under independent sign reversals under the null; t intervals assume independent differences and approximate normality. No missing selective collection is dropped. A full 12-collection analysis is mandatory; partial shards never generate primary inference.

## Secondary descriptive diagnostics

Common-hard contrasts against pair-hard, aux-hard and relational-hard; all-class overlap; n=48; exact recovery; precision/recall; selected and target cardinalities; latent pair-probability MSE, expected Brier and expected Bernoulli log loss; family/stratum summaries; tie incidence; soft structural scores archived at test. These receive no new confirmatory significance claims. The primary family is not changed after results are seen. This experiment does not establish human-preference transfer, faithful scalable TC training, global learnability or general theoretical novelty.

## Runtime, sharding and reproducibility

Production requires CUDA, torch 2.6.0+cu124, CUDA runtime 12.4, NumPy 1.26.4, SciPy 1.13.1 and pandas 2.2.3. Float32, deterministic algorithms, TF32 disabled, no mixed precision. GPU placement, optimizer state, update histories, initial states, every development prediction and all selected test predictions are audited. Launch refuses other compute jobs on the selected GPU. Distinct servers may have different hostnames/UUIDs but all shards must use the same GPU model/capability and numerical runtime. Every shard carries the same immutable global protocol/configuration/source fingerprint and its own immutable collection-ID list. Merge requires disjoint coverage of all twelve IDs and recomputes collection-level inference from raw rows, never pools shard tests.

Full results retain datasets, all candidate checkpoints/development predictions/histories and selected states/predictions. The compact review archive retains raw test datasets, development predictions and choices, initial states/device proofs, selected checkpoints/predictions, all timing records and derived tables for independent replay. Omitted unselected files remain individually hashed in the full archive. Large review archives split into ≤190 MiB share parts with hashes and an explicit reassembly manifest. No production completion marker is written by software QA. Global delivery is declared complete only after analysis, selected-checkpoint replay, archive payload checks and checksum verification pass.
