# Fixed CPU certificate diagnostic — audited results

The full fixed run completed on CPU in **7.16 seconds**: three synthetic
families, `n∈{12,24}`, four fixed counts per pair (`16,64,256,1024`), and
256 replicates per cell, totaling **6,144 cases**. Each family/size uses one
seeded probability matrix; the four budgets use separate, nonnested batches.
The interval error allocation is `δ=0.05` for one whole graph in one case.

The independent audit imports no `stepc` implementation. It regenerated all
six probability matrices and **1,050,624 binomial count entries** from the
declared seeds, reconstructed directed counts and scalar Hoeffding radii,
checked true UC using both set-cover and adjacency-list traversal oracles,
reconstructed outer membership with a possible-completion construction, and
matched every raw row and summary value. **All 26 original payload SHA256
hashes passed.** The audit preserves its runtime and source hashes.

Across the saved cases, the probability intervals contained every true pair
probability in **6,132/6,144 cases (99.805%)**. The UC inner/outer envelope
contained the true UC in **6,144/6,144 cases**. Two confidently wrong edges
occurred among the 12 interval failures. The inner and outer sets coincided
and equaled truth in 3,177 cases. These are observed counts, not a claim of
zero failure probability.

Representative outcomes below report **exact and correct UC recovery**, which
checks both equality of the bounds and agreement with truth.

| Fixed family | n | Counts per pair | Exact and correct UC / 256 | All pairs oriented / 256 | Mean unresolved memberships |
|---|---:|---:|---:|---:|---:|
| Condorcet, weak remainder | 12 | 64 | 237 | 0 | 0.891 |
| Condorcet, weak remainder | 24 | 64 | 80 | 0 | 16.500 |
| Three-cycle core, weak remainder | 12 | 64 | 240 | 0 | 0.109 |
| Three-cycle core, weak remainder | 24 | 64 | 59 | 0 | 7.234 |
| Condorcet or cycle core, each size | 12 or 24 | 256 | 256 in each of four cells | 0 in each cell | 0 |
| Uniform margin | 12 | 256 | 1 | 0 | 5.633 |
| Uniform margin | 24 | 256 | 0 | 0 | 10.277 |
| Uniform margin, each size | 12 or 24 | 1024 | 256 in each of two cells | 256 in each cell | 0 |

The structured families have strong `0.85/0.15` edges and weak `0.51/0.49`
remainder edges. At 256 observations per pair, certificates recover the
singleton Condorcet UC or the three-vertex cycle UC in every saved replicate
while many remainder edges stay unresolved. This demonstrates outcome
certification before full edge orientation in these fixed examples. Every
pair still receives the same budget, so it establishes no allocation savings.
The uniform-margin matrices use `0.65/0.35` and have true UC sizes 11 and 24;
their observed recovery pattern differs.

The rates condition on six fixed matrices and have finite Monte Carlo
variation with only 256 replicates per cell. They do not average over tournament
instances. The `0.05` guarantee is per case, not simultaneous over all 6,144
cases, budgets, or checkpoints. These runs provide no human-cohort population
coverage, learned-system comparison, or anytime/adaptive-stopping guarantee.
No large-graph completion enumeration was attempted; the audit independently
checks traversal, covering, and constructive membership on every saved case.

Replay with `python audit_diagnostics.py`. The preserved outputs are
`validation/certificate_diagnostics_full/summary.csv`, `raw_metrics.csv`, all
24 source `.npz` files, `DIAGNOSTICS_COMPLETE.json`, `RUN_STDOUT.log`, and the
replicate-level `INDEPENDENT_RECONSTRUCTION.csv`. The machine-readable audit is
`validation/CERTIFICATE_DIAGNOSTIC_AUDIT.json`.
