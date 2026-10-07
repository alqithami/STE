# STE Posterior Certificates v2: corrected exploratory human-profile protocol

**Software status:** v2 is a corrected future-run implementation. The completed
study was generated with immutable v1 code. No v2 production or retraining is
claimed. Read `CORRECTION_NOTICE.md`; do not replace v1 metrics or replay its
checkpoints under v2. Exact ties remain unresolved in hard relations.

The v2 marginal orientation matrix is the float64 marginal of the sampler's
normalized component weights and upper-edge probabilities. Its reverse entries
are exact complements and its diagonal is exactly .5. The hard-q diagnostic
clears its diagonal explicitly and passes strict partial-relation validation.
Native tournament samples remain irreflexive and complete. The normalization
changes future fitting numerics: v2 must be retrained at a fresh output root.

This package is a separate research extension. It does not replace the frozen
submitted STE code, results, or claims. No remote experiment is launched by
preparing it. The seed `20261007` was fixed for this extension, but the cohort and
its earlier outcomes were already examined. A new random seed does not make this
study confirmatory. All comparisons are descriptive; this protocol authorizes no
p-values, superiority tests, or claim of independent replication.

## Cohort, target and observed inputs

Use exactly the 36 genuine PrefLib profiles listed in
`reference/data/ELIGIBLE_PROFILES.csv`, retaining its original source and profile
order. Verify each bundled raw-file SHA-256 before parsing. There are 33 singleton
UC targets and only three selective non-singleton targets:

| Profile | Source | Alternatives | Full-profile UC size |
| --- | --- | ---: | ---: |
| `00007-00000068.soi` | Electoral Reform Society | 11 | 3 |
| `00007-00000086.soi` | Electoral Reform Society | 4 | 3 |
| `00019-00000004.toi` | Oakland elections | 8 | 3 |

The target is reconstruction of the finite full-profile empirical UC. It is not
the UC of an identified voter population or a latent utility model. Parse actual
rank groups and their positive integer multiplicities. An explicit tied rank
contributes a symmetric tie count `T`; it contributes no decisive wins. If either
alternative is omitted from a ballot, their comparison remains unobserved. Never
impute an omitted alternative beneath the ranked alternatives. No synthetic
ballots, latent graphs, or generated preferences enter this experiment.

Form full decisive counts `fullW`, full tie counts `fullT`, and the adjacency
`fullA = fullW > fullW.T`. Every cohort profile must have a full decisive
comparison for each pair and no exact empirical majority tie. Validate UC truth
with the cover-relation oracle, an independent set-inclusion oracle, and strict
tournament two-step reachability. Record raw hashes, alternative IDs, ranks,
original multiplicities, full counts and truth.

For each profile and each of eight resamples, draw 5%, 10% and 25% of its actual
voters without replacement. The sample size is `max(1, floor(fraction*voters))`.
Draw increments by multivariate hypergeometric sampling from the remaining
ballot-type multiplicities. The three fractions are nested within a resample;
the eight resamples are new draws from the same finite profile. Record the actual
sample multiplicities, `W`, `T`, profile/resample/fraction IDs and NPZ checksums.
Neither fractions nor resamples are independent profiles. Their induced pair
counts share ballots; this dependence is retained in the inputs even when a
baseline uses a factorized pairwise model.

## Source separation

Use six folds in this order, with no method-dependent changes:

| Fold | Held-out test source | Development source | Training sources |
| ---: | --- | --- | --- |
| 0 | `00007` | `00019` | `00021`, `00022`, `00067`, `00073` |
| 1 | `00019` | `00021` | `00007`, `00022`, `00067`, `00073` |
| 2 | `00021` | `00022` | `00007`, `00019`, `00067`, `00073` |
| 3 | `00022` | `00067` | `00007`, `00019`, `00021`, `00073` |
| 4 | `00067` | `00073` | `00007`, `00019`, `00021`, `00022` |
| 5 | `00073` | `00007` | `00019`, `00021`, `00022`, `00067` |

Training may use full-profile majority orientations and UC membership as labels
for the four training sources. Full training counts are labels, prior-fitting
information or provenance, not inference features. Fit the scalar empirical-Bayes
prior once from the full decisive counts of unique training profiles, averaging
pairs within profile, profiles within source and sources equally. Its objective
is a composite Beta-binomial working likelihood because pair observations share
ballots. Repeated resampled cases are not additional prior-fit evidence. Development references are used only for checkpoint,
threshold and fixed candidate selection. Forward and held-out inference receive
observed `W` and `T` only; they receive no full counts, target membership, target
cardinality, or core-class flag. Freeze and serialize choices before comparing
test predictions with test reference labels. `observed_view` and `fold_split`
provide this separation explicitly. Different folds reuse source series in
different roles; treat the final source summaries descriptively.

## Fixed baselines and five fitted systems

Include raw-count UC reconstruction and a non-fitted Jeffreys baseline. For the
latter, model each decisive pair probability with independent
`Beta(W_ab + 1/2, W_ba + 1/2)`, obtain orientation probability
`q_ab = Pr(p_ab > 1/2 | observed decisive counts)`, sample strict tournaments and
decode their UC sets by expected F1. Ties are recorded but are not divided into
fractional wins. This conditional decisive-pair model does not model the full
distribution of ranked ballots.

The five fitted systems are:

1. **Empirical Bayes:** fit one symmetric `Beta(alpha, alpha)` prior scalar from
   full decisive counts of unique training profiles through a source-balanced,
   profile-balanced, pair-mean composite Beta-binomial working likelihood, then
   use the same UC sampling and expected-F1 decoder as Jeffreys. Repeated cases do
   not increase prior-fit evidence or the number of independent profiles. This
   baseline receives richer training-reference information than Jeffreys and
   majority/membership-only neural supervision. Do not claim an exact match in
   supervision; all methods receive only observed counts at held-out inference.
2. **Ordinary native head:** a standard membership head trained on full-profile
UC membership, with its native development-selected set decision.
3. **Relational native head:** a relational membership head trained on the same
   observed cases and full-profile UC labels, with its native decision.
4. **Learned independent edges:** train reciprocal edge-orientation probabilities
   against full-profile majority labels with BCE. Independently sample edges
   under those probabilities and apply UC and expected-F1 decoding.
5. **Learned two-component distribution:** train a conditional mixture using
   edge BCE plus sampled UC expected-F1 utility. Draw a single global component
   for each sampled tournament, then draw all its edges conditionally independently
   within that component. The mixture therefore represents dependent edge
   orientations. Compare the fixed utility weights `lambda in {0.1, 1.0}` within
   one system's total allocation.

The learned distributions are predictive models. Their names do not establish
that they are exact Bayesian posteriors or calibrated probabilities. Mixture
dependence is different from an arbitrary joint posterior; neither the mixture
nor the independent learned-edge system receives an automatic calibration claim.
Report marginal Brier scores when appropriate, while keeping membership
calibration, set F1 and exact recovery as separate endpoints.

Expected-F1 set decoding is established General F-measure Maximizer prior art,
not a new theorem. For each proposed cardinality `k = 1,...,n`, estimate
`d_i(k) = mean_s[2*y_si/(k + |Y_s|)]`, take its top `k` alternatives and choose the
`k` with the largest summed score. This optimizes the empirical utility of the
joint sampled UC sets; finite Monte Carlo error remains. Membership threshold
`0.5` is not generally equivalent. See Waegeman et al.,
[JMLR 2014](https://jmlr.org/papers/volume15/waegeman14a/waegeman14a.pdf).

## Budgets, selection and Monte Carlo work

Production allocates 120 wall-clock seconds per fitted system, fold and
initialization, with three paired initializations. Five systems across six folds
therefore have a total maximum allocation of 10,800 seconds (three hours), plus
data preparation, fixed baselines, held-out inference and analysis. The separate
timing pilot allocates 30 seconds per system/fold and one initialization: 900
seconds (15 minutes) of allocated fitting work. A short smoke run is software QA
only. Do not substitute pilot or smoke output for production results.

Charge initialization, training, in-budget development inference and Monte Carlo work, threshold
and candidate selection, checkpoint serialization and bookkeeping to that same
system allocation. Provide three checkpoint opportunities at 20%, 45% and 70%
of each candidate allocation. Stop training at 85% and reserve the remaining 15%
for final development inference and serialization. Divide the mixture system
allocation equally between its two utility weights. Record requested time, measured fitting
and development time, checkpoint costs, candidate allocations, actual stopping
time and overshoot. An empirical-Bayes scalar fit can converge early and leave
its allocation unused; report the cost and do not pad it. The allocation is a
common upper bound, not proof of equal actual compute or equal statistical
complexity. A fixed 2% upper-allocation tolerance flags `EQUAL_ALLOCATION_VALID`
false for any overrun. Still complete and archive the descriptive outputs with
status `exploratory_budget_deviation`. Measured overshoot does not authorize
confirmatory inference or p-values.

Use 64 UC draws for the sampled training utility, 512 joint UC draws for each
development evaluation and 4,096 for each held-out posterior set decision. Smoke
uses 8 training, 32 development and 64 held-out draws, one resample and two folds.
Record actual draw counts and held-out inference wall time for
every method. Test inference is outside fitting allocations and is measured
separately; large method-dependent inference costs must appear alongside quality
metrics. Use fixed case/method/initialization seeds, and preserve seeds and
serialized predictions for audit.

Select checkpoints, utility weight and native-head thresholds using only the
development source's 10% cases. Average resamples within profile, then profiles
equally. Native and probability plug-in thresholds use the same fixed grid
0.00, 0.05, ..., 0.95, 1.00, including explicit all-set and empty-set endpoints.
The GFM decoder keeps its own native expected-F1 decision. Break exact
selection ties by earlier checkpoint, smaller utility weight, threshold nearest
0.5 and then smaller threshold. No selective-core-only development objective is
allowed: several development sources contain no selective non-singleton target.

## Certificates and reporting

Logical necessary/possible membership statements must identify the completion
family defined by their allowed edge orientations. Independent Beta distributions
with positive parameters support every strict tournament, so unrestricted
posterior support alone yields vacuous individual membership bounds for `n >= 2`.
Compute completion bounds only conditional on explicit edge restrictions. The
implemented count intervals use simultaneous Hoeffding edge restrictions and
orientation-probability tree bounds use the supplied edge probabilities.

All such quantities on human profiles are finite-cohort descriptive diagnostics.
The formal simultaneous Hoeffding coverage theorem assumes independent Bernoulli
comparisons for each pair with fixed comparison counts. It applies to the stated
IID model and its controlled synthetic validation, not these correlated human
ballot subsamples. No synthetic ballots are added to the human-profile cohort.
Probability tree bounds are mathematically conditional on the supplied joint
model (with union bounds available without independence); learned probabilities
do not acquire external calibration by satisfying the calculation. Keep logical
validity within the completion family, an IID theorem's sampling assumptions,
and empirical calibration as distinct claims.

The main descriptive summary uses all 36 profiles at 10% sampling. Average
resamples and paired initializations within profile, profiles within source, then
the six sources equally. Report the six source rows beside the source-macro
mean. The 5% and 25% summaries are secondary nested-fraction diagnostics. Include
F1, exact-set recovery, selected size, target size and cardinality error. Report
Brier score, certificate width and inference cost where they are meaningful;
never substitute a surrogate score for exact UC truth.

For each of the three selective profiles, show its own exact-set recovery, F1
and predicted-cardinality results, including the frequency of selecting the
correct cardinality. Do not treat their 24 nested/resampled inputs as 24
independent selective profiles or average their two sources into a confirmatory
superiority claim. The large singleton fraction can obscure inflation errors;
keep singleton and selective summaries visible. Save per-case predictions and
metrics before aggregation, including fold, source, profile, fraction, resample,
method, initialization and selected configuration.

The deliverable is an auditable exploratory runner and evidence. It is not a
validated dependent posterior, a new GFM decoder, a demonstration of novelty,
or support for revising the frozen manuscript's empirical claims without a
separate substantiated analysis.

## Separate fixed IID certificate diagnostic

`run_diagnostics.py` evaluates the supporting certificate bounds under their
stated independent Bernoulli count model. This is separate from human-profile
learning and does not generate ballots or add profiles to the human cohort. Fix
seed `20261007`, `n in {12, 24}`, decisive comparisons per unordered pair
`m in {16, 64, 256, 1024}`, 256 repetitions per setting and simultaneous failure
level `delta = 0.05` per simulated case. Every pair has exactly `m` independent
Bernoulli comparison outcomes with a fixed probability. Different pairs and
repetitions use independent draws under the generator.

Use these three fixed generator families:

| Family | Probability structure | Diagnostic purpose |
| --- | --- | --- |
| `condorcet_weak_rest` | One alternative beats all others with probability 0.85; remaining pair orientations have winning probability 0.51. | Can a singleton UC be certified while irrelevant edges remain unresolved? |
| `cycle_core_weak_rest` | Three alternatives form a probability-0.85 directed cycle and beat all outsiders with probability 0.85; outsider pair orientations have winning probability 0.51. | Can a selective size-three UC be certified with unresolved outsider edges? |
| `uniform_margin` | Random fixed orientations with winning probability 0.65 for every pair. | Compare membership certification with whole-graph orientation confidence when every edge has the same margin. |

The generator's probability matrix `P` and majority UC are simulation truth only.
The interval and certificate procedures receive observed decisive counts `W`;
they receive neither `P` nor truth UC. After predictions are fixed, use truth to
check simultaneous interval coverage and the UC membership envelope. Save `P`,
actual pair wins and per-repetition results. Report envelope width, exact
membership certification frequency, all-edge orientation frequency and empirical
coverage. The `delta` guarantee is per case, not a simultaneous promise across
all simulated repetitions and settings.

These chosen families illustrate the difference between certifying an output
and estimating an entire graph. They support a conditional mathematical bound;
they establish neither a new method's general advantage nor novelty, and they
provide no coverage guarantee for the human ballot experiment.
