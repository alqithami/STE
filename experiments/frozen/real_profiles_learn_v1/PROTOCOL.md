# Exploratory real-profile learning protocol

This study trains the existing permutation-equivariant pair model on genuine
archived human preference records. It reconstructs the UC of each full finite
record from actual ballot subsamples. No latent population-core labels are
available. It is exploratory: the cohort and previous estimator outcomes have
already been inspected. It does not replace the paper's completed primary test.

## Frozen cohort and scope

The earlier archive contains 462 profiles from 11 source series. Of these, the
36 profiles listed in `data/ELIGIBLE_PROFILES.csv` meet the following criteria:
they originate from ERS, Oakland, San Francisco, San Leandro, CSES or Voter
Autrement; have 4–48 alternatives; have every decisive pair observed; and have
no exact full-record majority ties. All listed raw bytes must match the existing
archive's SHA-256 manifest. The two independent UC oracles must agree. Exact
full-count duplicates are forbidden.

These profiles belong to six source series: ERS 9, Oakland 7, San Francisco 3,
San Leandro 2, CSES 10 and Voter Autrement 5. There are only THREE selective
non-singleton UC references: two ERS profiles and one Oakland profile. The other
33 references are singletons. Strictness is an eligibility condition on the
full reference; subsampled input counts may contain missing comparisons or
majority ties. This benchmark therefore gives limited evidence about learning
non-singleton sets. Report all three selective profiles separately.

The cohort was selected by an earlier metadata/core-class feasibility screen;
it is not representative of all elections or surveys. Its previous test
outcomes have been seen. Source exclusion reduces leakage and does not turn
this study into an unseen confirmatory benchmark. CSES parties/leaders from the
same country-year and Voter Autrement variants may share respondents; the entire
source is assigned together. Cross-source population independence is not
asserted. Sports rankings are excluded. UK election files are also excluded:
their nominal voters are constituencies ranked by electoral results, rather
than individual preference ballots.

## Recorded input construction and supervised target

For each profile, sample eight nested ordinal-record subsets without replacement at
5%, 10% and 25% of its recorded multiplicity. The seed depends on the profile
and replicate; all systems share these exact inputs. Weighted ordinal records
are sampled by multivariate hypergeometric draws. These are election ballots,
survey-derived rankings, or online voting records according to the source;
not all 36 profiles are actual election ballots. Ranked ties yield tie counts;
unranked alternatives remain unobserved. The training sampler draws a source
uniformly, then a training profile, fraction and replicate uniformly. Duplicate
case draws during training reuse real recorded subsets; they do not create
synthetic ballots or independent experimental units.

The label is hard UC of the full profile's strict empirical majority graph.
Full counts and labels may supervise training-source profiles. Held-out source
counts/labels never enter model training or model/cutoff selection. The
prediction inputs contain only subsample decisive and tied counts. No fitted
probability model generates the learning observations. Random posterior
tournament draws are used only to compute the explicit posterior baseline.

## Source-separated training, development and testing

There are six folds in the frozen source order shown in `config.json`. Each
source is the test once. The cyclic next source is development only; the
remaining four are training only. Thus train/development/test have disjoint
source series and whole profiles. Each source is evaluated only in its held-out
fold. Holdout assignments are written before learning.

Three initializations are paired across the four existing systems: pair-loss
only, ordinary auxiliary membership head, relational auxiliary membership head,
and STE with canonical soft UC membership supervision. Frozen model/operator
files are copied byte-for-byte. All use hidden width 32, Adam learning rate
0.003, gradient clipping 5, float32, no mixed precision/TF32, and canonical
temperature 0.035. Count likelihood always uses observed subsample counts;
auxiliary/STE membership loss uses the full-reference training label. Core loss
weights are 0.1 and 1; pair-only uses weight zero. Auxiliary and STE systems
receive full-reference membership supervision from the four training sources;
the count-only baselines receive no cross-profile core training labels. Thus
their comparison asks whether the learned supervision adds reconstruction
utility, rather than comparing identical supervision.

Each system receives 120 measured seconds per fold and initialization, including
training, development prediction and checkpoint serialization. Pair-only has
one candidate; each supervised system splits its allocation into two 60-second
weight candidates. Four checkpoint opportunities occur at equally spaced
candidate wall times. A final batch/checkpoint can exceed its boundary; record
the actual duration and optimizer steps. Report measured costs rather than
claiming exact identical execution times. The configured allocation is 8,640
seconds (2 hours 24 minutes); count baselines, test evaluation and packaging are
additional. Incomplete candidates restart with the same seed and full candidate
budget; completed candidates are resumed only after all saved hashes pass.

For each system, select checkpoints separately for threshold-free hard UC and
soft structural decoding, and for native membership heads where present. Each
choice maximizes development macro-profile F1 at 10% input. Soft/head absolute
cutoffs use the frozen 0.02 grid, including all/none endpoints. Break exact ties
by earlier checkpoint, smaller core-loss weight, cutoff closer to 0.5, then
smaller cutoff. Hard decoding applies the existing exact UC oracle to P > 0.5
and has no score cutoff. Exact predicted probability ties omit both arcs. Store
all development candidates and final choices before test model prediction.

## Baselines and estimands

Evaluate Jeffreys independent-edge posterior inclusion, posterior half-cutoff,
posterior expected-F1 decoding (GFM), raw hard majority UC, count plug-in soft
UC, Copeland, smooth Copeland, win rate, BTL, Hodge and Rank Centrality, plus
all/none controls. All consume exactly the same subsample counts. Posterior
membership uses 4,096 sampled strict tournaments per case with frozen seeds;
report its Monte Carlo draw count. Calibrate score-based baselines on the
held-out development source at 10%, using the same cutoff grid. Native count
hard UC, half-cutoff and GFM have no development cutoff. Including posterior
GFM is essential: posterior inclusion plus one scalar threshold need not
maximize F1.

Neural features include tied-count fractions. The independent-edge posterior
and count ranking baselines model decisive directions, conditioning on decisive
counts; their predictions do not use tied counts to infer a direction. All have
the same available observations, but they use different statistical models of
those observations. Keep this distinction explicit when interpreting a gain.

The descriptive primary estimand is all-class F1 at 10%: average resamples and
initializations within each profile, then profiles within source, then all six
source series equally. Report exact recovery, precision/recall, cardinalities,
and per-source F1 differences. Report 5%/25% and the THREE selective profiles
separately. Do not treat resamples, fractions, shared-source profiles, or model
initializations as independent population replicates. Generate no p-values or
automatic real-data superiority statement. A useful observation does not erase
the previous equal-budget soft-set null finding or the previous posterior
advantage. Any completed run must be described as an additional exploratory
experiment; negative outcomes remain in its package.

## Software verification and completion

CPU and GPU smoke output is clearly classified as software QA, with reduced
budgets/draws/initializations. Production refuses CPU fallback. Raw/profile
eligibility, oracle agreement, subsample multiplicity conservation, CUDA
placement, finite loss/gradients, encoder weight changes, per-candidate elapsed
budgets, train/dev/test source exclusion and hashes are checked. Retain raw
ballot multiplicities, full counts, sampled counts, selected weight files,
optimizer steps, development decisions and test predictions. `COMPLETE.json`
appears only after every fold, summary and output hash is verified. Packaging
must verify this marker and archive bytes; launch status is not completion.

## Data attribution

PrefLib: https://www.preflib.org and https://github.com/PrefLib/PrefLib-Data.
Source cards: https://preflib.github.io/PrefLib-Jekyll/dataset/00007,
https://preflib.github.io/PrefLib-Jekyll/dataset/00019,
https://preflib.github.io/PrefLib-Jekyll/dataset/00021,
https://preflib.github.io/PrefLib-Jekyll/dataset/00022,
https://preflib.github.io/PrefLib-Jekyll/dataset/00067,
https://preflib.github.io/PrefLib-Jekyll/dataset/00073.

CSES source requests citations to its Modules 1–3 full releases (2015), Module
4 (2018), Module 5 (2022), and Niclas Boehmer's 2023 dissertation,
"Application-oriented Collective Decision Making: Experimental Toolbox and
Dynamic Environments." ERS source describes the original tabulation by Nicolaus
Tideman and studies by Tideman and Plassmann. The raw files retain their original
source metadata; no private participant text is added. Verify source conditions
before redistributing beyond the existing research archive.
