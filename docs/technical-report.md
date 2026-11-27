# Technical report

## Abstract

howhard selects a model and reasoning effort for an AI agent task with the aim
of reducing the cost of reaching an accepted result. It combines independent
LLM judgments with learning from actual project executions. A coarse reader
estimates which candidates have enough capability; a fine reader forecasts
outcomes and resource use within that band. Selection accounts for repeated
caught failures and defects that escape acceptance. When a later investigation
traces a defect to an earlier execution, the original outcome can be corrected
and its learning replayed.

The statistical design combines weighted Bayesian updates for a latent capability
threshold, approximate predictive-evidence weights for memories, log-normal beliefs for
active time and token use, and heuristic uncertainty estimates for selection. These
components support decisions under
explicit assumptions. Demonstrating net credit savings requires chronological
evaluation that includes routing, learning and worker costs.

## 1 Problem and objective

The decision is a pair: model and reasoning effort. A capable model at high effort
can spend more than a routine task warrants. A cheaper invocation can also become
expensive through retries, review and later repairs. howhard evaluates candidates
against the task and current project, rather than relying on a universal ranking.

Let {math}`a=(m,e)` be a supplied model/effort candidate and {math}`x` the original
task and captured project context. The outcome categories are:

| Outcome | Recorded meaning | Consequence for selection |
| --- | --- | --- |
| `clean` | No defect has been observed | A non-loud result is accepted; later evidence may correct the label |
| `loud` | A gate or review caught a model failure | The cost model charges another attempt on the same candidate |
| `silent` | A defect escaped acceptance and was discovered later | Selection includes an escaped-defect penalty and an accepted-result risk limit |
| `interrupted` | Execution stopped operationally (`usage_limit`, `provider_error`, `host_error`, `cancelled`, `unrecorded`) | Preserved for usage/cost accounting and audit; excluded from ternary forecasts, capability thresholds, active-duration calibration and belief scoring |

Acceptance is therefore observable workflow behavior, not proof of correctness.
An execution retains its actual model, effort, outcome and measured usage. An
alternative that never ran has forecasts but no observed outcome.

For each candidate, define {math}`C_{\mathrm{run}}` as the expected worker cost of one
run, {math}`T` as the expected active execution hours of one run, {math}`p_l` as
loud-failure probability and {math}`p_s` as silent-failure probability. The selection
objective is:

```{math}
J(a)=\frac{C_{\mathrm{run}}(a)+\lambda T(a)
             +C_l p_l(a)+C_s p_s(a)}
            {\max(\delta,1-p_l(a))}.
```

Here {math}`\lambda` prices elapsed time, {math}`C_l` prices extra overhead per
caught failure, {math}`C_s` prices an escaped defect, and {math}`\delta` is a
numerical probability floor. These penalties express policy assumptions; actual
charges and rework remain separately measured evidence.

### Beliefs for active time and tokens

{math}`T` and {math}`C_{\mathrm{run}}` are averages of beliefs, not forecasts taken
at face value. Active time and each token kind have one log-normal belief per
candidate. Let {math}`A` be such an amount, with

```{math}
\ln A\sim\mathcal{N}(m,s^2),\qquad
s=\max\!\left(s_g,\frac{\tau}{\sqrt{N_{\mathrm{start}}}}\right),\qquad
m=\ln(\mathrm{prediction})-\frac{s^2}{2}.
```

- {math}`\mathrm{prediction}` is where the belief starts: the experts' forecasts of
  the amount, averaged on the log scale with their support weights.
- {math}`s_g` is the stated spread: each expert's stated 95% range read as
  {math}`(\ln hi-\ln lo)/3.92`, combined with the disagreement between experts,
  {math}`s_g^2=\sum_k w_k\,(s_k^2+(m_k-m_g)^2)`, where {math}`w_k` is expert
  {math}`k`'s weight, {math}`m_k` the log of its forecast, {math}`s_k` its stated
  spread and {math}`m_g=\sum_k w_k m_k`.
- {math}`\tau=D\sqrt{\text{scatter share}}` is the run-to-run scatter of the amount
  on one task. {math}`D` is the typical remaining miss of a prediction, as a spread
  of {math}`\ln(\mathrm{actual}/\mathrm{prediction})` (1.3 by default), and the
  scatter share is the part of {math}`D^2` that is scatter (0.5 by default).
- {math}`N_{\mathrm{start}}=1/(1/\mathrm{worth}+1/N_{\mathrm{task}})` is how many
  runs the prediction is worth: {math}`\mathrm{worth}` (5) is what the record says
  about the model family's own level, and {math}`N_{\mathrm{task}}` (1) is the most
  a record on other tasks is worth for one task.
- Subtracting {math}`s^2/2` makes the belief's average equal the prediction.

A stated range can widen the belief and never narrow it below
{math}`\tau/\sqrt{N_{\mathrm{start}}}`, which is 1.007 with the defaults. For a
prediction of 3,600 seconds with no range, {math}`N_{\mathrm{start}}=0.833`,
{math}`\tau=0.919`, {math}`s=1.007` and {math}`m=7.682`: an average of 1 hour and a
median of 0.60 hours. A stated range of 600 to 60,000 seconds gives a spread of 1.175,
which is used.

Recorded runs of the same candidate then update the belief conjugately. The runs and
their weights are those of section 3's evidence weights: task similarity, a 14-day
half-life and at most one run's weight per ticket. With
{math}`w_i` the weight of run {math}`i`, {math}`W=\sum_i w_i` and {math}`y_i` the log
of the amount it recorded:

```{math}
\frac{1}{s'^2}=\frac{1}{s^2}+\frac{W}{\tau^2},\qquad
m'=m+s'^2\,\frac{\sum_i w_i\,(y_i-m)}{\tau^2},\qquad
\mathbb{E}[A]=\exp\!\left(m'+\frac{s'^2}{2}\right).
```

Each run pulls the centre {math}`m'` toward its own amount and narrows the spread
{math}`s'`, by its weight. Two runs with weights 0.5 and 0.3 that recorded 1,800 and
5,400 seconds give {math}`W=0.8`, {math}`s'=0.719`, {math}`m'=7.792` and an average of
3,137 seconds. {math}`T` is the average of the active-time belief, in hours.
{math}`C_{\mathrm{run}}` prices the averages of the token beliefs at the candidate's
rates; section 7 gives its spread. A token kind that every expert forecasts at zero,
with no non-zero upper end, is exactly zero.

### Why the retry denominator matters

Under the model's stationary same-candidate retry assumption, attempts continue
until a non-loud result is accepted. Away from the numerical floor:

```{math}
\mathbb{E}[N]=\frac{1}{1-p_l},
\qquad
P(\mathrm{silent}\mid\mathrm{accepted})=\frac{p_s}{1-p_l}.
```

The denominator charges repeated worker cost and time. The loud penalty adds
detection or rework overhead beyond those repeated attempts. The model does not
assume that a failed attempt switches to another candidate. Real retries can be
correlated or change the task, so the geometric calculation is an approximation
to assess against observed workflow behavior.

As an illustrative calculation, assume known comparable host charges and ignore
time and defect penalties:

| Candidate | Cost per attempt | Loud probability | Expected worker cost to acceptance |
| --- | ---: | ---: | ---: |
| A | $0.40 | 0.50 | $0.80 |
| B | $0.65 | 0.05 | About $0.684 |

Candidate B has a higher invocation cost and a lower expected worker cost to
acceptance. These hypothetical values illustrate the objective, not measured
howhard savings. The full decision also considers escaped-defect risk.

The {ref}`selection contract <accepted-result-selection>` specifies the objective,
policy fields and admission tiers.

## 2 Architecture and independent LLM judgments

The workflow separates capability filtering from detailed forecasting:

```text
Task + project snapshot + actual host inventory
                    |
Independent LLM capability prior + retrieved experience
                    |
Learn task-specific order and fit band
                    |
Independent LLM fine forecast + memory-conditioned forecasts
                    |
Compare cost, accepted-result risk and credit feasibility
                    |
Host executes selected model and effort
                    |
Record actual outcome and measured usage
                    |
Replay predictive scores and update written lessons
                    ^
                    |
Supported later defect attribution amends the original outcome
```

Separate stateless calls produce cold-start capability priors and fine
no-memory forecasts once per unseen `(prototype, model, effort)` pair against a
synthesized `(representative_task, representative_context)` cluster in
`payloads.sqlite3` (matched on canonicalized task-only text with positive cross-encoder
relevance). Neither receives memories, historical
executions, learned tables or prior conversation. Warm routes whose candidates are
already covered reuse the prototype's materialized priors and empirical sufficient
statistics directly, calling a model only when active semantic memories require
conditional overrides or new unseen candidates enter the inventory. A memory reader separately
supplies conditional adjustments grounded in retrieved evidence and cannot
rewrite those baseline judgments.

This separation preserves a reference forecast for evaluating whether experience
helped. It does not imply that the baseline LLM is unbiased or calibrated.
Unavailable or invalid independent forecasts stop routing explicitly.

The full supplied inventory is considered at the capability stage. Only the fit
band receives detailed outcome and usage forecasts. If the learned band exceeds
the forecast budget, an explicit re-filter is requested; a second oversized band
is refused. If every forecast candidate is excluded by the risk limit, bounded
widening can request new forecasts for stronger candidates before a cell is drawn.

The [capability contract](capability.md) defines these transitions, while the
[context contract](context.md) defines the frozen task and project inputs.

## 3 Bayesian capability learning

### Latent threshold and task-specific order

For an inventory of {math}`n` candidates, the LLM proposes an order from weakest
to strongest for this task, with lower and upper fit cut points. A discrete latent
threshold {math}`H` represents the capability required. Its support includes a
position beyond the strongest supplied candidate, allowing the inventory to be
insufficient.

Each expert's cut points define a smoothed prior cumulative distribution over
this threshold. Below the lower cut {math}`\ell`, the prior CDF at rank {math}`r`
is {math}`0.8\,p_{\min}\,2^{-(\ell-1-r)}`. Above the upper cut, the prior allocates
half of the remaining overpowered-tail mass per step, normalized to the finite tail.
Predictive weights mix the independent and memory-conditioned priors.
Consensus ordering preserves each model's effort chain. A monotone
projection makes the mixed cumulative distribution consistent with that order.
Call the resulting threshold prior {math}`\pi_0(h)`.

### Weighted likelihood update

The evidence for one decision is a single set of candidate runs: recorded runs with
outcome `clean`, `loud` or `silent` that belong to at least one prototype the new task
matches. Prototypes only choose the candidates; no prototype score and no membership
weight enters a weight. A task that matches no prototype has no candidate. At most 256
candidates, the most recent by observation time, are compared with the new task, once,
when the request is prepared.

Only candidates whose executed cell is present in the inventory contribute threshold
likelihoods. Let {math}`r_i` be an executed candidate's
position in the proposed order, and {math}`z_i=1` for a clean outcome and zero
for a loud or silent outcome. Given observation noise {math}`\eta`, define:

```{math}
q_i(h)=
\begin{cases}
1-\eta & h\leq r_i,\\
\eta & h>r_i.
\end{cases}
```

The update is:

```{math}
\pi(h\mid D,x)\propto\pi_0(h)
\prod_{i\in D}
\left[q_i(h)^{z_i}(1-q_i(h))^{1-z_i}\right]^{w_i}.
```

Each weight is a product of four factors, followed by a limit per ticket:

```{math}
\mathrm{sim}_i=\frac{1}{1+e^{-\rho_i}},\qquad
a_i=0.5^{\,d_i/h},\qquad
w_i=m_i\,\mathrm{sim}_i\,s_i\,a_i,\qquad
w_i\leftarrow\frac{w_i}{\max(1,W_t)},\quad W_t=\sum_{j\in t}w_j.
```

Here {math}`\rho_i` is the cross-encoder's score for the new task's text against run
{math}`i`'s own task text, and {math}`\mathrm{sim}_i` is that score on a scale from 0
to 1. {math}`d_i` is the number of days from the run's observation to the time the
request was prepared, never below zero, and {math}`h` is the half-life, 14 days, so the
age factor {math}`a_i` halves a run's weight for every 14 days of age. {math}`m_i` is
the stored evidence weight of the run's model, 1 unless configured. {math}`s_i` is the
severity/cause strength: 1 for a clean run, 0.75 for a loud and 1 for a silent failure
attributed to capability, and half of that when the cause is unknown. {math}`t` is the
run's ticket and {math}`W_t` the total weight of that ticket's counted runs; a run
with no ticket is not limited.

In plain words: a run counts more when its task reads like the new one and less as it
ages; there is no similarity cut-off, so a run with a negative score still counts at a
small weight; and all runs of one ticket together count for at most one run, because
retries follow failures. For example, a loud run with no recorded cause,
{math}`\mathrm{sim}=0.8`, recorded 7 days before the request, weighs
{math}`0.8\times0.75\times0.5\times0.7071=0.2121`. Failures
explicitly attributed to capability retain their severity strength; unknown
causes contribute less. Supported assertions that a failure was not capability
caused can remove its threshold contribution without removing the outcome.

This is a weighted-likelihood, or generalized Bayesian, update. The relevance
weights are heuristic transfer strengths, not calibrated probabilities that two
tasks are equivalent. Related historical tasks are treated as noisy evidence
about the current threshold under the proposed order.

### From posterior to fit band

The posterior cumulative probability {math}`P(H\leq r\mid D,x)` estimates how
likely a candidate is to clear the task's capability threshold; write
{math}`P_{\mathrm{clear}}(o)` for its value at cell {math}`o`. The clearing and
timely-clearing posteriors each define an interval. An interval starts at the first
position meeting its lower edge and retains a reliable candidate when one exists. It
ends at the first position at or above that anchor where the remaining probability
gain is below its margin. When reliability is unattainable, the lower boundary is the
gain rule's anchor. When the timely minimum is also unattainable, only the timely gain
cut is kept.

The two edges of the clearing interval follow the host, the agent that asks for the
route and accepts or rejects the result. A failure the host finds before acceptance is
loud and one found afterwards is silent, so the share of its failures a host catches
decides how much weakness the band can afford. The caught share is estimated from the
failed routed runs on record, shrunk towards all hosts and towards a reference share:

```{math}
d_0=\frac{1-2\varepsilon}{1-\varepsilon},\qquad
d_{\mathrm{all}}=\frac{C_{\mathrm{all}}+\kappa_d\,d_0}{F_{\mathrm{all}}+\kappa_d},\qquad
d=\frac{C_{\mathrm{host}}+\kappa_d\,d_{\mathrm{all}}}{F_{\mathrm{host}}+\kappa_d},
\qquad F=\sum_i a_i,\quad C=\sum_i a_i c_i.
```

Here {math}`\varepsilon` is the risk class's accepted-defect limit. The sums run over
the failed runs: finished loud or silent runs whose decision receipt holds a forecast
for the cell that ran, of all hosts for {math}`F_{\mathrm{all}}` and
{math}`C_{\mathrm{all}}` and of the asking host for {math}`F_{\mathrm{host}}` and
{math}`C_{\mathrm{host}}`. {math}`c_i` is 1 for a loud run and 0 for a silent one.
{math}`a_i` is the age factor of the weighted likelihood update above, half as much
for every 14 days between the run and the request, divided by {math}`\max(1,A_t)`
where {math}`A_t` is the total age factor of the failed runs of the same ticket, so
one ticket's failed runs count as one run at most.
A run's host is the name in its decision receipt; a receipt with no name, or older
than decision v11, belongs to one unnamed host. {math}`\kappa_d=12` is the number of
failed runs the prior is worth. {math}`d_0` is the caught share at which both edges
below equal the configured ones.

In plain words: {math}`d` is the share of its failures that the host caught. A host
with few failures of its own leans on all hosts, and an empty record gives
{math}`d_0`. For example, at {math}`\varepsilon=0.10`, {math}`d_0=0.889`. With
{math}`F_{\mathrm{all}}=60` and {math}`C_{\mathrm{all}}=42`,
{math}`d_{\mathrm{all}}=(42+12\times0.889)/72=0.731`. With
{math}`F_{\mathrm{host}}=20` and {math}`C_{\mathrm{host}}=8`,
{math}`d=(8+12\times0.731)/32=0.524`.

```{math}
e=\max\!\left(p_{\min},\;1-\frac{\varepsilon}{1-d+\varepsilon d}\right),\qquad
g=\min\!\left(g_{\max},\;\frac{g_{\max}\,\varepsilon}{(1-\varepsilon)(1-d)}\right).
```

The clearing interval starts at the first cell with
{math}`P_{\mathrm{clear}}(o)\geq e` and ends at the first reliable cell with
{math}`P_{\mathrm{clear}}(\text{strongest})-P_{\mathrm{clear}}(o)<g`. {math}`p_{\min}`
is `min_clear_probability` (0.5), the floor of {math}`e`, and {math}`g_{\max}` is
`overpowered_gain_margin` (0.05), the widest margin. A reliable cell has
{math}`P_{\mathrm{clear}}` at or above `reliability_probability`, and the strongest
cell is the last of the order. At {math}`d\geq d_0` both edges are the configured
values, and below it both tighten.

In plain words: a cell too weak for the task fails. With
{math}`q=1-P_{\mathrm{clear}}`, a share {math}`q(1-d)` of its results escapes the host
and a share {math}`1-qd` is accepted. Holding the escaped share of the accepted results
at {math}`\varepsilon` gives {math}`e`. At the top, 0.05 more
{math}`P_{\mathrm{clear}}` removes about {math}`(1-d)\times0.05` of the silent
results, so {math}`g` shrinks to keep {math}`(1-d)\,g` constant. At
{math}`\varepsilon=0.10`, a host that catches 90% or more keeps {math}`e=0.5` and
{math}`g=0.05`; {math}`d=0.8` gives 0.643 and 0.0278; {math}`d=0.5` gives 0.818 and
0.0111; {math}`d=0` gives 0.900 and 0.0056. For example, {math}`d=0.524` gives
{math}`e=0.811` and {math}`g=0.0117`: with {math}`P_{\mathrm{clear}}` of 0.30, 0.55,
0.72, 0.84, 0.91, 0.955, 0.985 and 0.99 along the order, the configured edges give
cells 2 to 6 and these edges give cells 4 to 7.

If no cell reaches {math}`e`, the band starts at the first cell with the highest
{math}`P_{\mathrm{clear}}` when that value reaches {math}`p_{\min}`, and the request
is refused otherwise. The timely interval and the prior CDF keep the configured values.

When the span of the two intervals exceeds the forecast budget, the fine reader
forecasts their union. The gap lies beyond the clearing gain cut and before the
timely interval. This retains both clearing and timely alternatives for {math}`J` to
compare, while the receipt keeps probabilities for every rank. If the union itself
exceeds the budget, request a sharper independent prior over the same inventory.

The band alone can leave out a family that has never run this kind of task, or every
cell of one agent. Of the {math}`B` cells the forecast budget allows, {math}`R=2` are
reserved:

```{math}
\mathrm{core}=\text{the first }B-R\text{ cells of the band},\qquad
\mathrm{forecast}=\mathrm{core}\;\cup\;\{\text{up to }R\text{ cells }o_f\text{ with }P_{\mathrm{clear}}(o_f)\geq b_f\}.
```

Here {math}`f` is a family with no core cell and {math}`o_f` its cell nearest to the
core in the order, the stronger one on a tie. The bar {math}`b_f` is 0.2 when no run
of {math}`f` is among the decision's candidate runs, and {math}`e` otherwise. Slots go
first to those unproven families, each on its own and nearest first. The remaining
slots go to agents with no core cell, at most one slot per agent: a proven family
whose agent is missing takes the slot on that agent's behalf, at the agent's
qualifying cell nearest to the core, and an agent whose unproven family already took a
slot counts as served. In plain words: a slot buys one forecast for a family or an agent
the band leaves out. For example, a band of 11 cells with {math}`B=12` has a core of
ten. A family with no candidate run whose nearest cell has
{math}`P_{\mathrm{clear}}=0.35` takes the first slot, and an agent with no core cell
whose nearest cell is the band's eleventh takes the second. A slot that no family
takes goes back to the band, whose next cell beyond the core is then forecast. A
reserved cell keeps its position's classification, and no decision forecasts more
than {math}`B` cells.

Cross-model placements can change through constrained local search using marginal
threshold evidence and an order-distance penalty. This is approximate posterior/MAP
search, rather than exact inference over all rankings. The scalar ordering and
monotone threshold are modeling assumptions: different tasks can induce different
orders, and higher effort does not guarantee success.

An unseen candidate's placement remains prior-based. Observations at other
positions can still change its threshold-clearing probability, conditional on
that placement. The receipt distinguishes this inference from an observed outcome.
See [capability filtering](capability.md) for the exact cuts and evidence rules.

## 4 Learning which memories improve predictions

The fine reader forecasts clean/loud/silent probabilities and resource use for
each retained candidate. It combines an independent baseline with conditional
forecasts informed by written lessons or retrieved executions. These forecast
sources act as predictive experts.

### Predictive scores

For expert {math}`k` and execution {math}`i`, let {math}`p_{k,i}(y_i)` be the
probability assigned to the actual outcome before execution. Let expert zero be
the independent fine baseline. The outcome score is a floored log probability
ratio:

```{math}
S_{k,i}=\log\frac{\max(\delta,p_{k,i}(y_i))}
                         {\max(\delta,p_{0,i}(y_i))}.
```

Measured tokens and effective active duration (`active_seconds` when recorded,
otherwise wall-clock `seconds`) add broad log-normal resource scores relative
to the baseline, excluding passive external queue wait when broken out. Unknown
measurements add no resource evidence. A compact
expression for the resulting fine-expert support is:

```{math}
\widetilde W_k=\pi_k\exp\!\left[
\tau\sum_{i\in E_k}\gamma^{a_{k,i}}v_i
\left(S_{k,i}+\rho R_{k,i}\right)\right],
\qquad
W_k=\frac{\widetilde W_k}{\sum_{j\in A}\widetilde W_j}.
```

Here {math}`\pi_k` is initial expert mass, {math}`\tau` tempers evidence,
{math}`\gamma` discounts older exposures, {math}`v_i` is the stored evidence
weight, and {math}`R_{k,i}` is the resource-score advantage with scale
{math}`\rho`. The set {math}`E_k` contains eligible earlier forecast exposures;
{math}`a_{k,i}` counts subsequent eligible exposures for that expert. The set
{math}`A` contains participating forecast sources for this decision. Probability
means are mixtures of their forecasts using normalized support {math}`W_k`.

Capability-prior experts use the same support machinery separately, scoring
clean versus non-clean predictions against the independent fine forecast
coarsened to those two outcomes. Their scores do not include resource terms.

These are approximate predictive-evidence weights. They do not estimate the
probability that a natural-language lesson is true, establish a causal explanation,
or form an exact joint posterior over memories, rankings and outcomes.

### Prospective evidence and writable lessons

A memory earns exposure only when it supplied an explicit changed forecast for
the candidate actually executed. Unretrieved memories and unchanged overrides
receive no predictive credit. A lesson created after a failure can explain that
failure and guide later work, but the triggering failure cannot validate its
predictive performance. Revised lessons start a new content version without
inheriting the previous version's score.

After feedback is durable, an LLM writer can add, edit, remove, restore, merge or
split written memories. Mutations preserve history and reject stale revisions.
A writer failure leaves the execution outcome intact for retry. Raw execution
experts remain read-only.

Learning is shared among candidate models and providers inside the project's
store. A lesson may inform a forecast for another model, while its source outcome
remains attributed to the invocation that actually ran. The
[semantic-memory contract](semantic-memory.md) specifies scoring and mutation;
the [retrieval guide](retrieval.md) specifies precedent selection.

## 5 Outcome posteriors and bounded exploration

Each expert's fine forecast is a prior, not a measurement. For one candidate cell,
howhard treats the floored forecast {math}`p_0=(p_c,p_l,p_s)` as the mean of a
Dirichlet prior with concentration {math}`\kappa_0\le\kappa_{\max}` (the configured
`prior_strength`, 8 by default). An optional 95% credible interval on {math}`p_l` or
{math}`p_s` lowers {math}`\kappa_0` by moment matching,
{math}`\kappa=p(1-p)/\sigma^2-1` with {math}`\sigma` equal to the interval width
divided by 3.92, but never raises it: a forecaster cannot claim the authority of many
real runs. The same candidate runs that feed the capability threshold update the prior
conjugately: for each forecasted cell, the candidates that ran on that cell count with
{math}`w_i=m_i\,\mathrm{sim}_i\,\delta(c_i,c)\,a_i` and the same limit per ticket,
where {math}`\delta(c_i,c)` is 1 when run {math}`i`'s cell {math}`c_i` is the cell
{math}`c` being judged and 0 otherwise, so frequently executed cells cannot crowd out
rarer ones. The counts {math}`n_c,n_l,n_s` are the sums of those weights over the clean,
loud and silent candidates, and they update the prior
while decaying cold-start prior concentration linearly over empirical
weight {math}`W_{\mathrm{emp}}=n_c+n_l+n_s` (`\tau_{\mathrm{emp}}=2.0`) toward a unit
empirical base prior {math}`b_0=(0.88,0.08,0.04)`,
{math}`\alpha=\kappa_0\max(0,1-W_{\mathrm{emp}}/2)p_0+\min(1,W_{\mathrm{emp}}/2)b_0+(n_c,n_l,n_s)`.
Once {math}`W_{\mathrm{emp}}\ge 2`, cold-start zero-shot guesses have exact zero weight so
pessimistic priors on unfamiliar model IDs cannot permanently suppress empirical winners.
Because each count halves for every 14 days of age, a regression or a
recovery moves the posterior as older runs fade. Active time is calibrated from the
same cell's candidates that record an active time, each at the weight the outcome
counts give that run. A cell with no
observations keeps its prior variance regardless of how much evidence other cells have;
global expert support weights experts but is not a per-cell sample size.

Reported probabilities are the support-weighted mixture of the experts' posterior
marginals {math}`\mathrm{Beta}(\alpha_i,\sum_j\alpha_j-\alpha_i)`, with equal-tailed
95% credible intervals. Under a Dirichlet, accepted-result defect risk
{math}`p_s/(p_c+p_s)` is exactly {math}`\mathrm{Beta}(\alpha_s,\alpha_c)`, which
preserves each expert's loud/silent relationship. The intervals are Bayesian under the
stated prior and exchangeability assumptions; they are not calibrated frequentist
guarantees, and epsilon is a selection rule, not an empirical guarantee about the
future defect rate.

### The chance of meeting the limit

Admission asks one question of each forecast cell: how likely is it that the cell keeps
silently wrong accepted results within the risk class's limit?

```{math}
\pi(c)=P(S\le\varepsilon)
      =\sum_k w_k\,I(\varepsilon;\,\alpha_{s,k},\,\alpha_{c,k}).
```

- {math}`c`: a forecast cell.
- {math}`S`: the accepted-defect share, silent results divided by accepted results,
  {math}`p_s/(p_c+p_s)`.
- {math}`\varepsilon`: the risk class's limit on {math}`S`.
- {math}`\pi(c)`: the share of the cell's posterior for {math}`S` at or below
  {math}`\varepsilon`.
- {math}`k`, {math}`w_k`: a participating expert and its support weight. The weights
  sum to 1.
- {math}`\alpha_{s,k}`, {math}`\alpha_{c,k}`: the silent and clean entries of expert
  {math}`k`'s posterior {math}`\alpha` for the cell.
- {math}`I(x;a,b)`: the regularized incomplete beta function, the share of a
  {math}`\mathrm{Beta}(a,b)` distribution at or below {math}`x`.

In plain words, {math}`\pi` is the chance, under the cell's current belief, that no
more than {math}`\varepsilon` of its accepted results are silently wrong.

Example: {math}`\varepsilon` is 0.10. Two experts weigh 0.6 and 0.4, and their
posteriors for {math}`S` are {math}`\mathrm{Beta}(1,30)` and
{math}`\mathrm{Beta}(1,9)`. Then {math}`I` is {math}`1-0.9^{30}=0.958` and
{math}`1-0.9^{9}=0.613`, so {math}`\pi=0.6\times0.958+0.4\times0.613=0.820`.

### Three tiers

```{math}
\begin{aligned}
\text{trusted}  &\iff \pi(c)\ge q,\\
\text{on trial} &\iff t\le\pi(c)<q,\\
\text{excluded} &\iff \pi(c)<t.
\end{aligned}
```

- {math}`q`: the trusted line, `admission_quantile`, 0.80.
- {math}`t`: the on-trial line, `trial_probability`, 0.20.

In plain words, a cell that very likely meets the limit is trusted. An uncertain cell
is on trial. Only a cell that very likely misses the limit is excluded. A wide
posterior, which is all an unproven cell has, puts the cell on trial instead of
shutting it out for width alone, while a forecast or history with substantial silent
risk is still excluded.

Example, with one expert and {math}`\varepsilon=0.10`:

| Posterior for {math}`S` | Average | {math}`\pi` | Tier | 80% point of {math}`S` |
| --- | --- | --- | --- | --- |
| {math}`\mathrm{Beta}(1,30)` | 3.2% | 0.958 | trusted | 0.052 |
| {math}`\mathrm{Beta}(1,9)` | 10.0% | 0.613 | on trial | 0.164 |
| {math}`\mathrm{Beta}(4,8)` | 33.3% | 0.019 | excluded | 0.445 |

The last column is the receipt's `admission_risk`, the {math}`q` point of the same
mixture. It is at most {math}`\varepsilon` exactly when {math}`\pi\ge q`, so the
trusted tier is the same test read from the other side.

### Who competes

```{math}
\begin{aligned}
\mathcal{P}&=\{c:\ \mathrm{credit\_ok}(c)\ \text{and}\ c\ \text{is trusted or on trial}\}
  &&\text{when not empty},\\
\mathcal{P}&=\{c:\ \mathrm{credit\_ok}(c)\ \text{and}\ \pi(c)\ge f\cdot\max_{c'}\pi(c')\}
  &&\text{otherwise},\\
c^{*}&=\arg\min_{c\in\mathcal{P}}J(c),\\
\mathrm{eligible}(c)&\iff c\in\mathcal{P}\ \text{and}\
  J(c)-J(c^{*})\le u\,\bigl(\mathrm{sd}(c^{*})+\mathrm{sd}(c)\bigr),\\
\text{chosen}&=\arg\min_{c\ \mathrm{eligible}}\tilde J(c).
\end{aligned}
```

- {math}`\mathcal{P}`: the pool of cells that compete.
- {math}`\mathrm{credit\_ok}(c)`: the cell has enough credit; the maximum in the
  second line runs over the cells {math}`c'` that have enough credit.
- {math}`f`: `all_excluded_factor`, 0.5.
- {math}`J(c)`, {math}`\mathrm{sd}(c)`: the cell's mean cost and the delta-method
  standard deviation of that cost. It combines the Dirichlet covariance of the cell's
  outcome probabilities with the variance of its active time,
  {math}`(\lambda/(1-p_l))^2\,\mathbb{E}[T]^2\,(e^{s'^2}-1)`; run cost is exact in it.
- {math}`c^{*}`: the pool cell with the lowest mean cost.
- {math}`u`: `uncertainty_multiplier`, 1. A risk class's `explore_within` additionally
  caps the gap at that fraction of {math}`|J(c^{*})|`.
- {math}`\tilde J(c)`: the cell's cost under the seed's draw. Selection samples a
  predictive expert in proportion to its support and draws outcome probabilities from
  that expert's per-cell Dirichlet posterior, and each cell's active time from its
  belief, {math}`\ln T\sim\mathcal{N}(m',s'^2)`; run cost stays at its average
  (Thompson sampling).

In plain words, trusted and on-trial cells compete as equals, and nothing limits how
often an on-trial cell is chosen. A cell must still be close enough to the cheapest
pool cell for their cost distributions to overlap; by default a small point-forecast
difference cannot lock in a favoured cell and starve an unseen cheaper one of
evidence. When no trusted or on-trial cell has enough credit, the cells nearest the
best chance compete instead, so a sample is always drawn. Selection is refused only
when no cell has enough credit.

Example of the second pool: three cells with enough credit have {math}`\pi` 0.12, 0.08
and 0.05. The bar is {math}`0.5\times0.12=0.06`, so the first two form the pool and the
third is left out.

### The flag

```{math}
\begin{aligned}
\texttt{cap\_met}&\iff \text{the chosen cell is trusted},\\
\texttt{needs\_review}&\iff \text{not}\ \texttt{cap\_met}.
\end{aligned}
```

In plain words, the flag says that the chosen cell is not trusted. It is advice to the
caller, and nothing is enforced or recorded for it.

Example: a chosen cell with {math}`\pi=0.613` is on trial, so `cap_met` is false and
`needs_review` is true.

When every forecast cell is excluded, the engine first forecasts the next stronger
cell with no forecast, one per round, up to `widening_rounds` and within the forecast
budget; one trusted or on-trial cell stops this widening. The
{ref}`selection reference <admission-tiers>` defines the receipt fields, and the
[capability filter](capability.md) defines widening.

## 6 Retrospective defect attribution

Delayed feedback matters because a task that passes today's checks can contain
a defect discovered later. During a bug fix or requested audit, the companion
`attribute-failure` skill can investigate that earlier acceptance:

1. Reproduce the defect or obtain other concrete evidence of the failure.
2. Trace the behavior through the original task, pre-fix history and execution
   provenance; use bisection when helpful. Last-touch blame alone is insufficient.
3. When attribution is supported, append an amendment to the original execution's
   outcome and replay its frozen forecasts against the effective corrected result.
4. Update written lessons through the existing learning protocol. Record the fixing
   execution's own outcome separately.

The amendment replaces the earlier predictive score; it does not create another
independent execution or a new forecast exposure. Original identity, context and
measurements remain attached. Uncertain attribution stays unscored, and finding
an escaped defect does not itself make the fixing worker fail.

This is an investigation workflow invoked during later work. Installing the skill
adds neither continuous bug monitoring nor automatic historical attribution.
See the {ref}`escaped-defect attribution contract <escaped-defect-attribution>`
for the evidence and provenance requirements.

## 7 Credit accounting and operational boundaries

Money comparisons require known host charges on a comparable basis. API list
prices cannot stand in for unknown subscription credits or native host charges.
If evaluated candidates are unpriced or money bases are incompatible, worker
money is excluded uniformly and the decision identifies the partial risk/time
comparison. Unknown usage and cost remain unknown.

The credit one run needs is uncertain because its token counts are. Let
{math}`a` and {math}`b` index token kinds, {math}`\mathrm{price}_a` be the candidate's
rate for kind {math}`a` per million tokens, and {math}`\mathbb{E}[\mathrm{tokens}_a]`
and {math}`s_a` the average and spread of that kind's belief after its runs:

```{math}
c_a=\frac{\mathrm{price}_a\,\mathbb{E}[\mathrm{tokens}_a]}{10^6},\qquad
\mathbb{E}[K]=\sum_a c_a,\qquad
\mathrm{Var}(K)=\sum_a\sum_b c_a c_b\left(e^{\rho_{ab}s_a s_b}-1\right).
```

{math}`K` is the credit of one run. {math}`\rho_{ab}` says how kinds {math}`a` and
{math}`b` move together; it is 1, because no run records how the kinds co-vary, so the
spread is not understated by assuming independence. A log-normal with the same
average and variance has spread {math}`\sigma_K` and 80% point {math}`K_{80}`:

```{math}
\sigma_K^2=\ln\!\left(1+\frac{\mathrm{Var}(K)}{\mathbb{E}[K]^2}\right),\qquad
K_{80}=\mathbb{E}[K]\,\exp\!\left(0.8416\,\sigma_K-\frac{\sigma_K^2}{2}\right),
```

where 0.8416 is the 80% point of the standard normal. With input at 2 and output at
10 dollars per million tokens, an input forecast of 1,000,000 tokens with no range
(spread 1.007) and an output forecast of 100,000 tokens with a range of 10,000 to
1,000,000 (spread 1.175): {math}`c=2.00` and {math}`1.00`, {math}`\mathbb{E}[K]=3.00`,
{math}`\mathrm{Var}(K)=19.06`, {math}`\sigma_K=1.066` and {math}`K_{80}=4.17`.

Optional candidate credit budgets add a scheduling constraint, separate from the
admission tiers. They require uniform pricing and can represent remaining session and
weekly windows. A candidate is schedulable when
{math}`\max(\mathbb{E}[K],K_{80})` is at most its available credit: a run must fit
the remaining credit four times out of five, not only on average. The larger of the
two is taken because a belief with {math}`\sigma_K` above 1.68 has its 80% point below
its average, and the check must not get easier as a belief gets wider. In the example,
4.00 of available credit fails although the average fits, and 5.00 passes. Selection
refuses routing when none of the evaluated candidates is schedulable. This check does
not reserve the full uncertain cost of future retries or prevent concurrent external
spending.

Receipts freeze the inputs, retrieved evidence, memory versions, forecasts,
capability state and selection details. A recorded seed reproduces selection
given those frozen forecasts; fresh LLM inference can vary. Git worktrees share
the project evidence store while capturing their own current checkout context.

The host executes the chosen model and effort. Plugin discovery alone does not
enforce routing at every worker launch. Execution feedback must identify the
invocation that actually ran, and usage collectors operate on explicitly owned
artifacts. These boundaries are defined in [model and storage contracts](reference.md),
[host skills](skills.md) and [concurrency](concurrency.md).

## 8 Evaluating savings and learning

Net savings require evidence from genuine project work that includes the cost
of choosing and learning. The evaluation protocol compares a stateless LLM,
plain experience retrieval, writable memory with fixed weights, and memory with
learned predictive weights. Freeze candidate inventories, prices, worker tools,
acceptance gates and review budgets before outcomes are available.

Assess three different questions:

| Question | Evidence to report |
| --- | --- |
| Do forecasts improve? | Log loss and Brier score on matched observed actions; reliability with counts and uncertainty |
| Does the routing policy save money or credits? | Comparable worker charges plus routing and amortized writer costs; retries, latency, review and rework |
| Does delayed learning help? | Prospective forecasts on later related tasks, amendments after supported attribution, and follow-up coverage |

Configured failure penalties and priced time belong to policy loss. Report
measured review/rework separately so they are not counted twice through those
penalties. A policy's chosen-action logs cannot reveal what an unrun alternative
would have done. Comparative policy claims need actual alternative executions
or a prospective randomized trial; a reproducible random seed is not an action
assignment probability.

Evaluate chronologically using only information available at decision time.
Release delayed feedback when it became known, group related tasks and retries,
and retain failed decisions and inconvenient outcomes. The protocol follows
accepted work after 7 and 30 days. Missing follow-up is not evidence that a result
was clean. The initial 12-task feasibility study checks measurement and replay;
it cannot establish general superiority or a small silent-failure rate.

Implementation tests verify behavior. Retrieval diagnostics measure a component.
Neither establishes net savings for deployed routing. Field evidence must justify
the additional routing and learning cost against simpler comparators. The
[real-project evaluation guide](evaluation.md) provides the study procedure.

## 9 Statistical interpretation

The method is best described as **LLM priors, Bayesian capability learning,
evidence-weighted semantic memory and retrospective defect attribution**.
The components have distinct statistical meanings:

| Component | Interpretation | Main assumption or limit |
| --- | --- | --- |
| Independent LLM judgments | Cold-start task-specific prototype orders and forecasts | LLM judgment needs empirical validation |
| Capability threshold | Generalized Bayesian update with weighted noisy likelihoods | Scalar order and heuristic transfer from related tasks |
| Evidence weights | Task similarity, a 14-day half-life and at most one run's weight per ticket, over the runs of matched prototypes | Similarity is a heuristic transfer strength; the half-life and the 256-run cap are defaults, not fitted values |
| Cross-model ordering | Constrained local posterior/MAP search | Approximate search over rankings |
| Memory support | Tempered, recency-weighted predictive evidence | Forecast support does not establish truth or causality |
| Outcome posterior | Per-cell Dirichlet–multinomial update with continuous prior decay (`tau_emp = 2`) | Prior decay scale, base floor and the 14-day evidence half-life are fixed modelling choices |
| Active time and token beliefs | Per-cell log-normal belief with a conjugate update from same-cell runs | The miss `D`, scatter share and worths are fixed modelling choices; token kinds are taken to move together |
| Credit check | 80% point of a log-normal matched to the average and variance of run credit | A moment match, not the exact distribution of a sum |
| Admission tiers | Share of the mixture accepted-defect posterior within epsilon, cut at 0.80 and 0.20 | No calibrated coverage guarantee; the two lines are policy choices |
| Selection sampling | Thompson sampling within posterior cost overlap, among trusted and on-trial cells | Delta-method overlap is approximate |
| Outcome amendments | Correction and replay of an earlier execution | Requires supported attribution and preserved provenance |

Together, these mechanisms give routing access to both immediate outcomes and
mistakes discovered later. The practical question is whether that information
reduces total cost on future work enough to pay for acquiring and using it.
