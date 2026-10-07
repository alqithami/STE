# Public STE revision: evidence and software boundaries

This record describes material prepared for the public STE manuscript and its
research repository. The preprint identifier is
[arXiv:2604.04328](https://arxiv.org/abs/2604.04328); the public author is Saad
Alqithami. Preparing a replacement and updating code do not themselves publish
a new arXiv version.

## What the added study establishes

The completed posterior/certificate benchmark used the byte-preserved v1
pipeline on 36 actual human ordinal profiles from six source series. It held
out whole sources and reconstructed each profile's full-record empirical UC
from subsampled records. Its main descriptive endpoint is source-macro F1 at
10% of records; repeated subsamples and initializations are averaged within
profiles before profiles and sources are aggregated. Only three references are
selective; the other 33 are singletons. The cohort had already been examined,
so these findings remain exploratory, without p-values or general/population
superiority claims.

The native main results are shown in the root README. Jeffreys-GFM has the
highest recorded F1, while raw-count UC has the highest exact recovery. Learned
independent edges nearly match Jeffreys F1, but this does not establish a
learning advantage. The learned mixture is weaker on this endpoint. Selective
references expose underselection and limited exact recovery; those limitations
are preserved rather than hidden by the all-class macro. Native ordinary and
relational heads, empirical-Bayes and other count controls remain in the full
report. No alternative readout replaces a native endpoint after results are
seen.

The prior soft-set null and separate hard-UC synthetic confirmation remain
separate experiments. The new exploratory comparison neither replaces nor
pools them. Earlier results, their data targets, and inferential qualifications
remain visible in the manuscript.

## Correction policy

Independent inspection verified raw counts, selected-state replays, native
membership/cardinality samples, decisions and summaries. It also found a
secondary mixture hard-q defect: float32 weight-sum rounding could raise the
nominal diagonal above one half, produce self-loops, and allow self-covering in
the unguarded oracle. Exclude the entire v1 mixture hard-q aggregate. The native
GFM sampler used normalized component weights and strict irreflexive graphs;
its verified endpoint remains reportable.

[The correction notice](../experiments/posterior_certificates/CORRECTION_NOTICE.md)
records the defect and v2 repair. Frozen v1 outputs and source are unchanged;
replay them with [the original pipeline](../experiments/frozen/posterior_certificates_v1/README.md).
Any corrected calculation on those predictions must be labeled post hoc and
cannot overwrite the historical endpoint.

The [active v2 implementation](../experiments/posterior_certificates/README.md)
uses the normalized native sampling law for its mixture marginal, exact
reciprocal/diagonal conventions, graph validation, and stronger audit checks.
It has software QA only. Changed floating-point fitting numerics and validation
cost can alter future trajectories, so **no v2 training findings exist until a
fresh independently inspected production run is completed**. Never resume into
an old v1 root or combine unfinished work with a new run.

## Supporting mathematics

The additional proofs distinguish three objects: a smooth STE score, a
specified joint tournament law, and the decision made under that law. Positive
independent Beta posteriors support every strict tournament, while confidence
intervals can restrict a separate completion set. Mixture marginal edges alone
do not encode their joint dependence or UC-size dependence.

Logical inner/outer UC bounds use two-step reachability and covering witnesses.
Their conditional structural soundness can hold without recovering all edges.
The stated frequentist coverage requires a fixed comparison design and i.i.d.
Bernoulli observations within each pair. It does not apply to the finite human
ballot subsamples and establishes no anytime, adaptive-query, population, or
allocated-comparison-savings guarantee. Fixed-matrix diagnostics do not supply
such an extension.

The GFM size-aware decoder, depth-two tournament support structure, and
likelihood-ratio gradient method are established components, with primary
sources cited in the manuscript. Complete supporting proofs specify their use
here. No new decision-rule novelty, posterior calibration, optimizer
convergence, or learning superiority follows from those proofs. The previous
TC-bound vacuity at the executed smoothing settings remains explicit.

## Finding the implementations

- `experiments/posterior_certificates/`: corrected forward v2 software and its
  complete launch/audit/archive instructions.
- `experiments/frozen/posterior_certificates_v1/`: original completed-study code.
- `experiments/frozen/hard_uc_confirm_v1/`: separate synthetic hard-UC pipeline.
- `experiments/frozen/real_profiles_learn_v1/`: earlier exploratory human-profile
  training pipeline.
- `experiments/frozen/uc_equal_compute_v1/`: original soft-UC allocation comparison.

Study-specific protocols take precedence over the historical root README and
its legacy `make` targets. Original evidence, native negative results, selection
records, budget qualifications and exact-recovery limits are retained.
