# Corrected software v2; completed evidence remains v1

The completed posterior/certificate study used `STE-Posterior-Certificates-v1`.
Its independent inspection verified the saved native GFM outputs, model replays,
raw ballot counts, metric reconstruction, selection records, budgets and archive
hashes. It also exposed a defect in a **secondary** mixture hard-q diagnostic.
The internal v1 audit replayed that defect, so its original COMPLETE marker alone
did not establish scientific validity of this diagnostic.

## Defect and evidence policy

V1 computed the mixture marginal as a float32 weighted sum over all matrix
entries. The exponentiated log-softmax weights could sum slightly above one.
Its nominal .5 diagonal could consequently exceed .5; hard thresholding created
self-loops, and the unguarded partial-cover oracle then permitted self-covering
and returned an empty set. The independent review found invalid hard relations
in 320 of 2,592 mixture predictions across all fractions, including 101 of 864
at the main 10% fraction. Seven saved mixture q matrices had a probability just
outside [0,1] from roundoff.

Exclude the **entire v1 mixture hard-q aggregate**, rather than retaining only
apparently unaffected cases. Preserve all original files, checkpoints, manifests
and archives. Any corrected calculation on them is a separately labeled post
hoc diagnostic; it is not a replacement for the original native endpoint.

V1 native GFM/half decoders sampled an irreflexive strict tournament from upper
component probabilities and normalized component weights. They did not
threshold the faulty marginal diagonal. Independent reconstruction verified
their membership/cardinality draws and decisions. This correction does not turn
the native results into a learning win: at 10% of records the six-source macro
F1 was 0.874194 for Jeffreys-GFM, 0.870415 for learned-edge GFM and 0.836585 for
mixture GFM. The known cohort has 36 profiles, 33 singleton references and only
three selective references; comparisons remain exploratory and descriptive.

## V2 implementation

1. The differentiable q is the float64 marginal of **the same normalized native
   component/upper-edge sampling law**. Reverse entries are complements and the
   diagonal is exactly .5. Components, their sampled upper-edge probabilities
   and the normalized component draw rule are retained.
2. The hard decoder explicitly clears its diagonal. Graph boundaries reject
   self-loops and bidirectional pairs; strict tournament oracles also reject
   missing edges. Exact .5 pairs stay absent in partial hard relations.
3. The audit checks the corrected marginal, diagonal, graph contracts and
   version, and regenerates selected-model outputs and native draws. Code/config
   hashes, selected checkpoints and archive payloads remain part of completion.
4. The corrected checkpoint feature version and software release differ from
   v1. Default outputs and archive prefixes use v2. Historical v1 directories
   and preexisting run/archive destinations are refused.

This changes future fitting numerics and validation cost. **V2 must be retrained
at a fresh root before reporting v2 experiment results.** V2 has software QA
only; no new GPU training or v2 production findings are asserted. The original
v1 source is identified in `FROZEN_V1_PARENT_MANIFEST.json`; replay v1 only with
the byte-preserved original source. The regression's native-sampler parity
establishes parity for fixed components, weights, seeds and draws, not equal
future training trajectories or outcomes.

The logical certificate proofs in `THEORY.md` are supporting mathematics.
Their novelty is unestablished. Bernoulli fixed-count coverage does not extend
to the dependent finite human-cohort subsamples in this study.
