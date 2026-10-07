# Posterior uncovered-set decisions and honest certificates

This note supplies proofs for the new `stepc` implementation. It is a supporting
theory note, not a claim that the results below establish a novel contribution.
The existing paper, code, and experiment archives are unchanged. The formal
frequentist certificate is restricted to the synthetic fixed-design Bernoulli
experiment; real finite-cohort count envelopes are descriptive diagnostics.

## 1. Objects, assumptions, and the two uncertainties

Let `V={0,...,n-1}`, with `n>=1`. A **strict tournament** has exactly one of
`i→j` and `j→i` for each distinct pair. Vertex `u` covers `v` if `u→v` and
every out-neighbor of `v` is an out-neighbor of `u`. The uncovered set `UC(T)`
consists of vertices with no coverer. It is equivalently the set of vertices
that reach every other vertex in at most two directed edges. Indeed, if `u`
covers `v`, neither `v→u` nor `v→w→u` is possible. Conversely, if `v` cannot
reach `u` in two edges, the tournament forces `u→v`, and forces `u→w` whenever
`v→w`; thus `u` covers `v`.

The stochastic frequentist target is a fixed matrix `p` satisfying
`p_ij+p_ji=1` and `p_ij != 1/2` off diagonal, with

\[
T(p)_{ij}=\mathbf 1\{p_{ij}>1/2\}.
\]

For each observed unordered pair, the number `m_ij` of outcomes is fixed before
seeing outcomes, and its observations are i.i.d. Bernoulli(`p_ij`). Independence
between *different* pairs is not needed for the simultaneous confidence proof.
A pair omitted by this fixed design has `m_ij=0`. The code records directed
counts `W_ij`, so `m_ij=W_ij+W_ji`. These assumptions exclude outcome-dependent
stopping, adaptive allocation, and without-replacement sampling from a fixed
empirical cohort. No anytime or finite-population guarantee is claimed.

The separate Bayesian object is a posterior on tournament orientations. Write

\[
q_{ij}=\Pr(P_{ij}>1/2\mid D),\qquad q_{ji}=1-q_{ij}.
\]

This is an **orientation probability**, not the posterior mean
`E[P_ij|D]`. Under independent pairwise Beta priors with positive parameters,
factorized pairwise Bernoulli likelihoods give independent Beta posteriors and

\[
q_{ij}=1-I_{1/2}(a_{ij}+W_{ij},b_{ij}+W_{ji}).
\]

The posterior puts no mass at `P_ij=1/2`. More generally, independent edge
orientations must be assumed explicitly when using a product formula. Shared
latent parameters or a learned neural model can create dependent orientations;
their marginal `q` matrix alone does not identify the joint tournament law.
The union-bound certificate below survives such dependence.

An uncertainty claim conditional on an assumed posterior is not a frequentist
coverage claim about `p`. A test on human cohort records also does not establish
population coverage. Repeating a count envelope on a finite set of ranked
records without replacement does not turn these records into i.i.d. Bernoulli
draws conditional on the cohort. A separate finite-population analysis would
have to identify its estimand, design, and a proved bound before being used.

## 2. Simultaneous fixed-design intervals

**Proposition 1 (standard Hoeffding union bound).** Let
`M=n(n-1)/2` and `0<δ<1`. For each observed `i<j`, define

\[
r_{ij}=\sqrt{\frac{\log(2M/\delta)}{2m_{ij}}},\qquad
L_{ij}=\max\{0,\widehat p_{ij}-r_{ij}\},\qquad
U_{ij}=\min\{1,\widehat p_{ij}+r_{ij}\}.
\]

Set `L_ji=1-U_ij`, `U_ji=1-L_ij`; give omitted pairs `[0,1]`. Then

\[
\Pr_p\!\left(\bigcap_{i<j}\{p_{ij}\in[L_{ij},U_{ij}]\}\right)
\ge 1-\delta.
\]

For `n=1` there are no pair events and coverage is trivial; no logarithm with
`M=0` is evaluated. Diagonal entries are conventions `[1/2,1/2]` and do not
participate in the union bound.

**Proof.** For an observed pair, Hoeffding's inequality gives
`Pr(|p_hat-p|>r) <= 2 exp(-2m r²)=δ/M`. Clipping to `[0,1]` cannot exclude
an admissible `p` previously inside the interval. Unobserved intervals always
contain `p`. A union bound over at most `M` observed unordered pairs bounds the
failure probability by `δ`. Reverse-direction intervals are equivalent events,
so they do not consume a second allocation. No cross-pair independence is used.
For the singleton there is only the empty intersection. \(\square\)

The implementation spends `δ/M` per pair from the predefined `M`, even if only
some pairs are observed. It does not replace `M` by a number selected after
observing outcomes. It uses strict orientations: `i→j` is **known** if
`L_ij>1/2` (equivalently `U_ji<1/2`); every other pair remains unknown. On the
simultaneous event, all known edges are edges of the strict true tournament.
Equality at `1/2` never licenses an edge. Latent ties are excluded from the
target assumption; this note does not silently orient them.

This is one fixed comparison design and one fixed `δ`. Running the rule at
several checkpoints does not produce an anytime guarantee. A finite union over
predeclared checkpoints requires an additional error allocation; adaptive
sampling or stopping requires a proved sequential confidence construction.
The code's `coverage_guarantee` flag is an assumption declaration, not an
automatic audit of how counts were generated.

## 3. Logical completion bounds and witnesses

Let `G` be any partial tournament, and `C(G)` its unrestricted strict tournament
completions. Define exact individual membership sets

\[
N(G)=\bigcap_{T\in C(G)} UC(T),\qquad
P(G)=\bigcup_{T\in C(G)} UC(T).
\]

These are coordinatewise necessary and possible winners. A subset of `P(G)`
need not itself be the UC of any completion. The code enumerates all `2^r`
completions only when the number `r` of unknown edges is at most a supplied cap
(20 by default); this exact small-graph oracle is separate from the certificate
proof below.

Define the constructive inner set `A(G)` by the following test: for every
`u!=v`, either `v→u` is known, or there is `w` with both `v→w` and `w→u`
known. Define the outer set `B(G)` by removing `v` when there is a known
`u→v` such that, for every `w` distinct from `u,v`, either `w→v` is known
or `u→w` is known.

**Proposition 2 (sound UC certificate bounds).** For every partial tournament,

\[
A(G)\subseteq N(G)\subseteq UC(T)\subseteq P(G)\subseteq B(G)
\quad\text{for every }T\in C(G).
\]

The implementation only needs these inclusions; it does not use an unproved
claim of completeness of the constructive tests. If `A(G)=B(G)`, this common
set equals `UC(T)` for every completion.

**Proof.** Every known path is preserved in every completion. Thus a vertex
passing the inner test reaches every opponent in at most two edges in every
completion, and is a UC member by the two-step characterization. For a removed
vertex `v`, retain its witness `u`. Any completion has `u→v`. If `v→w`
in this completion, the alternative known edge `w→v` cannot hold; hence the
test supplies known `u→w`. Therefore every out-neighbor of `v` is an
out-neighbor of `u`, so `u` covers `v` in every completion. This proves the
outer exclusion. The middle inclusions follow from intersection and union.
If the two endpoints coincide, all intermediate sets coincide. \(\square\)

For an inner member, choose one parent for each vertex at distance two and
retain the direct edges from the root to its known out-neighbors. Every
nonroot vertex has exactly one incoming retained edge; all such edges point
from the root or one of its direct children. This gives a spanning out-tree of
depth at most two, with exactly `n-1` edges. Its preservation alone proves the
root's UC membership. For an excluded vertex, the code records its fixed
covering witness. The scalable implementation takes `O(n³)` time; it is not an
adaptive edge-query algorithm.

**Corollary 2.1 (frequentist UC bounds).** Under Proposition 1's assumptions
and the strict-`p` assumption, applying Proposition 2 to the known-edge graph
gives

\[
\Pr_p\{A(G(D))\subseteq UC(T(p))\subseteq B(G(D))\}\ge1-\delta.
\]

**Proof.** On the *single whole-graph simultaneous event*, `T(p)` is a
completion of `G(D)`. Proposition 2 applies to that completion. The event has
probability at least `1-δ`. Any certificate selected after viewing the data is
covered by the same event, so no uncorrected selection of an individual
pairwise interval is used. \(\square\)

If the reported inner and outer bounds coincide, the returned UC set is
correct on this event. This is simultaneous set recovery, not a statement that
each reported member individually has posterior probability `1-δ`. For a
finite-cohort or adaptively stopped diagnostic, only Proposition 2's conditional
structural statement remains, with no asserted event probability.

## 4. Posterior certificate probabilities

**Proposition 3 (conditional posterior lower bound).** Fix data `D` and a
depth-two spanning out-tree `C_v` rooted at `v`. Its directed edges specify
the required orientations. For any joint posterior law of orientations with
marginals `q_e`,

\[
\pi_v:=\Pr(v\in UC(T)\mid D)
\ge \max\!\left\{0,1-\sum_{e\in C_v}(1-q_e)\right\}.
\]

If the required orientations are jointly independent under the posterior,

\[
\pi_v\ge\prod_{e\in C_v}q_e
\ge\max\!\left\{0,1-\sum_{e\in C_v}(1-q_e)\right\}.
\]

**Proof.** Whenever every required orientation holds, the root has a path of
length at most two to every vertex, so it belongs to the UC. The probability
that at least one required edge fails is at most the sum of its marginal
failure probabilities. Complementing and clipping at zero proves the first
inequality. Under joint independence the probability that all required edges
hold is exactly their product. The same union bound applied to independent
edge events gives the final inequality. \(\square\)

The tree may be selected using `D` or the fixed posterior `q`; this is a
conditional posterior assertion about the selected deterministic tree. It does
not establish repeated-data coverage. Frequentist selection must instead be
protected by simultaneous confidence information. `posterior_certificate_bound`
returns the product only when `independent_edges=True`; otherwise it returns
the dependence-robust union bound. Its automatic tree choice uses `q>1/2` and
does not claim to find the most probable tree. The supplied tree is validated
as a spanning depth-two out-tree before computing either bound.

## 5. An exact posterior UC marginal and Rao–Blackwellization

The following identity is derived here to specify and test the estimator.
Its novelty is unverified; the act of conditioning and the variance argument
are standard and are not presented as new statistical theory.

**Proposition 4 (independent-orientation identity).** Under a fixed posterior
with mutually independent unordered edge orientations, let
`W=N_T^+(v)` be the random set of out-neighbors of `v`. Each `w!=v` belongs
to `W` independently with probability `q_vw`. Define

\[
H_v(W)=\prod_{u\in V\setminus(W\cup\{v\})}
\left(1-\prod_{w\in W}q_{uw}\right).
\]

Then

\[
\pi_v=E[H_v(W)\mid D]
=\sum_{W\subseteq V\setminus\{v\}}
\left(\prod_{w\in W}q_{vw}\prod_{u\notin W\cup\{v\}}q_{uv}\right)H_v(W).
\]

**Proof.** Conditional on `W`, exactly the vertices outside `W∪{v}` beat
`v`. Such a vertex `u` covers `v` precisely when `u→w` for every `w∈W`.
Those edge orientations remain independent with probabilities `q_uw`, because
conditioning on `W` fixes only edges incident to `v`. Thus its conditional
covering probability is `∏_(w∈W) q_uw`. Covering events for distinct outside
vertices involve disjoint unordered edges, one endpoint in `W` and one
outside; they are conditionally independent. Multiplying the probabilities
of not covering gives `H_v(W)=Pr(v∈UC(T)|W,D)`. Total expectation gives the
first equality. Expanding the independent distribution of `W` gives the
finite sum. \(\square\)

Empty products matter: if `W` is empty and `n>1`, every outside vertex covers
`v`, so `H=0`. If `W` contains all other vertices, the outside product is empty
and `H=1`. The singleton also gives `H=1`. The finite sum is exponential in
`n-1`; it is an exact small-`n` oracle, not a polynomial-time algorithm.

For three vertices `a,b,c`, direct enumeration gives

\[
\pi_a=q_{ab}q_{ac}+q_{ab}(1-q_{ac})q_{bc}
+(1-q_{ab})q_{ac}(1-q_{bc}).
\]

At all `q=1/2`, this is `1/2`. A differentiable STE score with a different
value, such as `5/8` in the existing surrogate, therefore cannot be identified
with this posterior membership probability without a separate calibration
argument. Smoothness alone gives no such argument.

**Proposition 5 (scalar Rao–Blackwell variance reduction).** Draw independent
copies of `W` under the posterior. The sample mean of `H_v(W)` is unbiased for
`π_v`. For a single copy,

\[
\operatorname{Var}(H_v(W)\mid D)
=\pi_v(1-\pi_v)-E[H_v(W)(1-H_v(W))\mid D]
\le\pi_v(1-\pi_v).
\]

Both sample-mean variances divide by the number of independent draws.

**Proof.** Let `Y=1{v∈UC(T)}`. Proposition 4 states `H_v(W)=E[Y|W,D]`.
Unbiasedness follows by total expectation. Total variance gives

\[
\operatorname{Var}(Y\mid D)=\operatorname{Var}(E[Y\mid W,D]\mid D)
+E[\operatorname{Var}(Y\mid W,D)\mid D].
\]

Since `Y` is Bernoulli, `Var(Y|D)=π_v(1-π_v)`, while
`Var(Y|W,D)=H_v(W)(1-H_v(W))`. Rearranging
proves the formula and the nonnegative difference. Independent averages divide
variances by the draw count. \(\square\)

This is a marginal variance statement. Conditioning separately for different
vertices does not automatically reduce cross-vertex covariance or the variance
of a set-level score. Marginal membership probabilities alone do not supply
the cardinality-dependent moments needed for expected F1.

## 6. Bayes expected-F1 decoding and finite Monte Carlo regret

Let `S=UC(T)` under any fixed posterior joint tournament law, possibly dependent.
`S` is nonempty: a maximum-outdegree vertex cannot be covered, since a coverer
would beat it and every one of its out-neighbors, and have strictly larger
outdegree. For an action `A⊆V`, define

\[
F(A,S)=\frac{2|A\cap S|}{|A|+|S|},\qquad
U(A)=E[F(A,S)\mid D].
\]

The empty action has utility zero. The unrestricted Bayes action maximizes
`U(A)` over all subsets of `V`; it need not be a UC set realizable by a
posterior tournament. Requiring that feasibility is a different optimization
problem, not solved by unrestricted GFM.

**Proposition 6 (established General F-measure Maximizer).** Define

\[
\Delta_{i,k}=E\!\left[\frac{2\mathbf1\{i\in S\}}{k+|S|}\ \middle|\ D\right]
\quad (i\in V,\;k=1,\ldots,n).
\]

For each `k`, take the `k` largest `Δ_i,k`, then select the cardinality whose
sum is largest. This gives an unrestricted Bayes expected-F1 action.

**Proof.** For `|A|=k`, linearity of expectation gives
`U(A)=Σ_(i∈A) Δ_i,k`. Among sets of size `k`, this sum is maximized by taking
the largest `k` weights, with arbitrary deterministic tie breaking. Searching
all cardinalities optimizes over all nonempty actions. At least one such
action has positive utility because `S` is nonempty, so the empty action
cannot improve the maximum. This proof uses no independence of labels or
edges. \(\square\)

This is the established GFM rule of Waegeman et al. (2014), applied to UC
indicator vectors; it is not a new decision theorem. Finite posterior draws
maximize the empirical posterior objective rather than integrating it exactly.

**Proposition 7 (standard uniform Monte Carlo regret).** Fix `ε>0` and a
positive integer `B`. Conditional on fixed `D` and a fixed posterior, let
`S_1,...,S_B` be i.i.d. posterior UC sets and
`U_hat(A)=B^{-1}Σ_b F(A,S_b)`. Let `A_hat` maximize `U_hat` and `A*` maximize
`U`. For `0<δ_MC<1`, if

\[
B\ge \frac{\log(2\cdot2^n/\delta_{MC})}{2\epsilon^2},
\]

then with probability at least `1-δ_MC` over the posterior simulation,

\[
\sup_A|U_{hat}(A)-U(A)|\le\epsilon,\qquad
U(A^*)-U(A_{hat})\le2\epsilon.
\]

**Proof.** Each F1 summand is in `[0,1]`. Hoeffding's inequality bounds the
two-sided error for a fixed action by `2 exp(-2Bε²)`. A union bound over at most
`2^n` actions proves uniform error. On that event,
`U(A*) <= U_hat(A*)+ε <= U_hat(A_hat)+ε <= U(A_hat)+2ε`.
The selected action may depend on the very same simulation because the bound
is simultaneous over actions. \(\square\)

For `η>0`, an alternative bounds the **derived weights** directly. Every summand defining
`Δ_hat_i,k` is in `[0,1]`, so

\[
B\ge\frac{\log(2n^2/\delta_{MC})}{2\eta^2}
\Longrightarrow\max_{i,k}|\Delta_{hat,i,k}-\Delta_{i,k}|\le\eta
\]

with probability at least `1-δ_MC`. For `|A|=k`, weight error causes utility
error at most `kη<=nη`, yielding regret at most `2nη` by the same argument.
This is a standard finite-class concentration consequence, not a novelty claim.
The weight bound is not a bound on raw cardinality probability-table entries.
The Monte Carlo failure probability concerns posterior integration, whereas
`δ` in Proposition 1 concerns comparison-data sampling. Neither alone proves
the posterior model is correct or a learned surrogate is calibrated.

## 7. A finite-mixture likelihood-ratio training identity

This section specifies a separate stochastic training objective. It is a
supporting finite-state likelihood-ratio identity with unverified novelty;
likelihood-ratio differentiation and independent baselines are standard.

Let `c∈{1,...,C}` index finitely many mixture components and let `e=(i,j)`
run over unordered pairs with `i<j`. Let `x_e(T)=1` indicate the orientation
`i→j`. With finite differentiable logits `α_c(θ)` and `z_c,e(θ)`, define

\[
\rho_c(\theta)=\frac{\exp(\alpha_c(\theta))}
{\sum_{d=1}^{C}\exp(\alpha_d(\theta))},\qquad
q_{c,e}(\theta)=\sigma(z_{c,e}(\theta)).
\]

Conditional on its sampled component, the edge orientations are independent:

\[
p_\theta(c,T)=\rho_c(\theta)\prod_e
q_{c,e}(\theta)^{x_e(T)}
(1-q_{c,e}(\theta))^{1-x_e(T)}.
\]

All joint state probabilities are positive. The marginal tournament law
`p_θ(T)=Σ_c p_θ(c,T)` usually has dependent edges. A learned law of this form
is only a posterior when a specified inference model justifies that
interpretation; the training identity does not supply that justification.

Fix a nonempty reference set `S*`, independent of `θ`, and define the reward
and training objective

\[
R(T)=F(UC(T),S^*),\qquad
J(\theta)=\sum_{c,T}p_\theta(c,T)R(T).
\]

**Proposition 8 (finite-state score identity and unbiased leave-one-out
baseline).** At any finite parameter value where the logits are
differentiable,

\[
\nabla_\theta J(\theta)
=E_\theta[R(T)g_\theta(c,T)],
\]

where the joint-state score is

\[
\begin{aligned}
g_\theta(c,T)
&=\nabla_\theta\log p_\theta(c,T)\\
&=\nabla_\theta\alpha_c-
\sum_d\rho_d\nabla_\theta\alpha_d+
\sum_e(x_e(T)-q_{c,e})\nabla_\theta z_{c,e}.
\end{aligned}
\]

For `s>=2` independent draws `(c_b,T_b)` from this same current law, put

\[
b_b=\frac{1}{s-1}\sum_{a\ne b}R(T_a),\qquad
\widehat g=\frac{1}{s}\sum_{b=1}^{s}(R(T_b)-b_b)g_\theta(c_b,T_b).
\]

Then `E_θ[g_hat]=∇_θ J(θ)`. Rewards and baseline values must be held constant
when differentiating the sampled log-probability surrogate.

**Proof.** The state space has `C·2^M` elements. Since the sum is finite and
`R(T)` does not depend on `θ`, differentiation commutes with summation:

\[
\nabla_\theta J=
\sum_{c,T}R(T)\nabla_\theta p_\theta(c,T)
=\sum_{c,T}p_\theta(c,T)R(T)\nabla_\theta\log p_\theta(c,T).
\]

The second equality uses positivity. Differentiating softmax gives
`∇ log ρ_c=∇α_c-Σ_d ρ_d∇α_d`. For each Bernoulli factor, differentiating
`x log σ(z)+(1-x)log(1-σ(z))` gives `(x-σ(z))∇z`. Adding these terms proves
the displayed score expression. Normalization also gives

\[
E_\theta[g_\theta(c,T)]
=\sum_{c,T}\nabla_\theta p_\theta(c,T)
=\nabla_\theta 1=0.
\]

For each `b`, `b_b` is a function only of the other independent draws, so it
is independent of `g_θ(c_b,T_b)`. Thus
`E[b_b g_b]=E[b_b]E[g_b]=0`. Each uncentered reward-score term has expectation
`∇J` by the first part; averaging the centered terms preserves that
expectation. Treating the baseline or reward as a differentiable part of the
surrogate would introduce additional terms not present in this argument.
\(\square\)

Sampling and the scored log probability must describe the same law. Clipping
`q` in one of them while retaining unclipped logits in the other would break
the stated identity. Stable log-sigmoid evaluation expresses the same finite
logit law without an additional mathematical clipping operation. The theorem
does not assert optimizer convergence or reduction of gradient variance from
every baseline choice. The leave-one-out baseline is undefined at `s=1`;
using a zero baseline at that draw count retains the uncentered identity.

**Corollary 8.1 (component-conditional UC integration).** For each component,
use its oriented `q_c` matrix in Proposition 4 and denote the result by
`H_v,c(W)`. Then under this mixture law,

\[
\Pr_\theta(v\in UC(T))=
\sum_c\rho_c E[H_{v,c}(W)\mid c]
=E[H_{v,c}(W)].
\]

**Proof.** Conditional on `c`, the independent-edge proof of Proposition 4
applies. Averaging its conditional probability by the law of total probability
proves the first equality and sampling `(c,W)` gives the second. In fact
`H_v,c(W)=E[1{v∈UC(T)}|c,W]`, so the total-variance proof of Proposition 5
applies to this scalar estimator as well. No marginal edge independence is
required after mixing. \(\square\)

Using mixture marginal edge probabilities
`q_bar_e=Σ_c ρ_c q_c,e` inside Proposition 4's product formula is generally
incorrect. Likewise, conditional component independence does not license a
certificate probability `∏_e q_bar_e`. A fixed certificate instead has event
probability `Σ_c ρ_c∏_e q_c,e`, or can use the dependence-robust union bound
from mixture marginals. Joint GFM weights require the dependence between
membership and UC size; averaging scalar component membership marginals does
not provide those weights.

The training objective `J` scores a randomly sampled **whole UC set** against
a fixed reference truth. The final GFM decoder optimizes the expected F1 of
an **action** against a random modeled UC set. Evaluation against held-out
reference truth is a third quantity. Optimizing `J` does not, by this theorem,
guarantee improvement of the final GFM action, held-out F1, calibration, or
learning convergence. These objectives must be named separately in results.

## 8. Prior art and research claims

The following primary sources delimit what may reasonably be claimed.

| Primary source | Established component used here | Scope distinction |
|---|---|---|
| [Waegeman et al., JMLR 2014, *On the Bayes-Optimality of F-Measure Maximizers*](https://jmlr.org/papers/volume15/waegeman14a/waegeman14a.pdf), Theorems 8–9 | GFM uses a quadratic matrix of derived weights under an arbitrary joint label law. | UC posterior labels are an application, not new expected-F1 decoding. |
| [Aziz et al., JAIR 2015, *Possible and Necessary Winners of Partial Tournaments*](https://webspace.maths.qmul.ac.uk/felix.fischer/publications/abfo_partial.pdf), Section 4.4, Theorems 5–7 | Individual possible and necessary UC winners are polynomial-time computable; the two-step characterization is used. | Whether a specified entire set is a possible UC is NP-complete. Individual bounds do not solve that feasibility problem. |
| [Contet, Grandi, and Mengin, AAAI 2026, *Explaining Tournament Solutions with Minimal Supports*](https://arxiv.org/pdf/2509.09312), Corollaries 4–6 | UC minimal supports are depth-two rooted out-trees with `n-1` edges; a smallest support can be computed in `O(n²)`. | Depth-two certificate structure is prior art. The linked primary manuscript identifies itself as the extended AAAI 2026 version. |
| [Ramamohan, Rajkumar, and Agarwal, NIPS 2016, *Dueling Bandits: Beyond Condorcet Winners to General Tournament Solutions*](https://papers.nips.cc/paper/2016/file/fccb3cdc9acc14a6e70a12f74560c026-Paper.pdf), Section 4.2 | UCB methods target UC/TC/Banks winners and give logarithmic cumulative-regret guarantees. | A fixed-confidence whole-set or expected-F1-loss stopping guarantee is a different objective, and would still require comparison with this work. |

**Support caveat.** With finite counts and positive independent Beta prior
parameters, each `q_ij` lies strictly between zero and one. Consequently every
strict tournament has positive posterior support. For `n>=2`, each vertex is
in some supported UC (make it a Condorcet winner) and absent from another
(make it a Condorcet loser covered by another vertex). Thus unrestricted
posterior-support possible winners are all vertices and necessary winners are
empty. Informative confidence-completion sets must be justified separately;
they are not the support of this Beta posterior.

**Research goal A — unproved, novelty unverified.** Develop an adaptive noisy
comparison algorithm with a fixed-confidence whole-UC or F1-regret stopping
rule whose instance-dependent cost accounts for discovering and verifying
outcome certificates. A displayed tree alone gives no bound on discovery cost.
The fixed-design intervals in this note do not establish such a theorem.

**Research goal B — unproved, novelty unverified.** Obtain a differentiable
approximation to posterior decision risk or the GFM cardinality-dependent
moments with quantitative approximation and gradient-error control, including
dependent learned edge models where relevant. The exact scalar identity and
Rao–Blackwell estimator here do not prove this research goal, nor a guarantee
for the existing STE training surrogate.

No claimed novelty, superiority, active-learning sample complexity, or
population-coverage result follows from the software tests. They check
implementation identities and every-completion logical soundness, including
all 729 partial tournaments at `n=4`, 100 generated partial tournaments at
`n=5` with every completion, and all 1024 complete tournaments at `n=5`.
