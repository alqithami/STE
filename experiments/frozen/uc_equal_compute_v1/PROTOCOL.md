# Prospective protocol: STE UC equal-compute v1

This document is frozen in the delivered package before any production outcomes.
The earlier completed study remains distinct. This follow-up tests complete UC
systems under an equal measured training/development budget; it does not alter
the canonical STE operators or reinterpret the earlier primary endpoint family.

## Question and systems

Can structural UC supervision improve held-out selective-core F1 relative to
ordinary and relational auxiliary membership heads at the same measured selection
budget? Compare four systems with the original pairwise model architecture:

| System | Training objective | Primary readout |
|---|---|---|
| Pairwise-only | Pairwise likelihood | Structural UC (secondary control) |
| Auxiliary | Pairwise likelihood + direct-core binary cross-entropy | Direct membership head |
| Relational auxiliary | Pairwise likelihood + relational direct-core binary cross-entropy | Relational membership head |
| STE | Pairwise likelihood + structural-core binary cross-entropy | Structural UC |

Loss weights are `0.1` and `1.0` for supervised systems. The original generators,
observation model, feature construction, optimizer settings, UC formulas, truth
oracles, and temperature `0.035` remain unchanged. The `frozen/` files and the
original configuration are bundled and hashed. This experiment studies finite
synthetic tournaments; no real-data neural-transfer claim follows from it.

## Data and independence

Seed `2026100407` defines twelve fresh, independent collections. Each has 512
training graphs and 128 development graphs at 12 alternatives, with 256 test
graphs at each of 24 and 48 alternatives. Use the original five generator families
in their original deterministic allocation: ordered, planted, separated,
rank-mixture, and random. Missingness is 0.25, tie probability 0.20, and count
intensity 10. Test distributions retain all natural core classes; no cases are
resampled to favor STE.

Each collection uses three paired initializations and the original paired
training-order construction. A collection, not a graph or initialization, is the
independent unit for inference. Initializations are averaged within collections.
Data hashes, model initialization seeds, and CUDA placement evidence are retained.

## Measured budget and model selection

Each supervised system gets two candidate budgets of 110 seconds: one for each
fixed loss weight. The pairwise-only system gets a single 220-second candidate.
Therefore each system has a nominal 220-second training/development budget per
collection and initialization. The total default allocation is
`12 × 3 × 4 × 220 = 31,680 seconds` (8 hours 48 minutes).

This includes training and development evaluation/checkpoint selection. CUDA
operations are synchronized for timing. Checkpoint fractions are fixed in advance
at `0.025, 0.1, 0.25, 0.5, 1.0`. Their measured timing, actual updates/epochs,
training and development costs, final selection, and total end-to-end costs are
logged. Small timing overshoots are audited against the prospectively fixed
candidate and system tolerance of ±2%; a violation prevents an unqualified
equal-compute claim. The full actual audit is included even if it fails.

The model/checkpoint, loss weight, and decision threshold are selected from
development data only by mean per-case F1 on selective non-singleton UC at
12 alternatives, using the prespecified absolute decision rule. Threshold
grid spacing is 0.025. Test labels, test cardinalities, and primary-test outcomes
are never used to choose a model, threshold, or checkpoint. Ties favor the earlier
checkpoint, then the smaller loss weight, then a threshold nearest 0.5, then the
smaller threshold. Save all choices for all three initializations in a collection
before generating its test data. No oracle cardinality is
used to construct predicted sets.

Setup, isolated software QA/smoke, dataset generation, model initialization/data
transfer, common warmup, post-budget integrity bookkeeping, and final held-out
testing are outside the selection budget and are reported separately. They explain why
the allocation is not a completion-time guarantee. The idle-device preflight
refuses another visible compute process; it never kills or modifies another job.

## Endpoints and analysis

The primary evaluation subset contains test cases whose true UC has cardinality
`2 <= |UC| < n`: selective non-singleton cores. This is a prespecified evaluation
stratum, not a data-generation or decision rule.

Two primary contrasts, both at 24 alternatives:

1. STE structural UC F1 minus ordinary auxiliary own-head F1.
2. STE structural UC F1 minus relational auxiliary own-head F1.

For each contrast, average case F1 within each initialization/collection on this
subset, then average the three initializations. Apply the paired procedure across
the twelve collection-level differences. Report mean differences, 95% paired
Student-t confidence intervals, exact two-sided sign-flip p-values, and Holm-adjusted p-values
across exactly these two contrasts. No per-case pseudo-replication is permitted.

Only each system's native readout is evaluated and calibrated during its timed
development selection: direct for auxiliary systems and structural for pairwise-only
and STE. Auxiliary development does not run the UC operator. Held-out prediction
stores pair probabilities and both available score arrays outside that budget.

Descriptive secondary outputs include full mixed-class results, selective results
at 48 alternatives, singleton and full-core strata, exact set recovery,
predicted/true cardinality, average precision, AUROC, saved structural-score diagnostics,
and pairwise-only controls. Save predicted pair matrices and report held-out
pairwise Brier score/likelihood and hard UC decoding. These outputs help interpret
whether a change reflects set selection or relation quality; they are not new
confirmatory endpoints.

The analysis retains positive, negative, and inconclusive outcomes. Missing units,
nonfinite data, budget violations, or mismatched provenance are visible and cannot
be bypassed with a completion marker. Training completion, complete packaging,
and validity of the equal-compute comparison are separate records.

## What this run can and cannot establish

A positive primary result would strengthen complete-system UC utility under this
fixed synthetic distribution and measured budget. It would not establish
universal dominance, an application to human preferences, TC scalability,
calibration across new domains, or priority over all differentiable tournament
formulations. A null/negative result constrains the practical claim and must be
reported. The existing matched-epoch study remains evidence for its original
limited claim and is not silently replaced by this follow-up.

The short smoke protocol has fewer cases, one initialization, and subsecond
candidate budgets. It is labeled `smoke=true`, is stored outside production
results, and is never eligible for manuscript inference.
