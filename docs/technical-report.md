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
threshold, approximate predictive-evidence weights for memories, and heuristic
uncertainty estimates for selection. These components support decisions under
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
| `silent` | A defect escaped acceptance and was discovered later | Selection includes an escaped-defect penalty and an accepted-result risk cap |
| `interrupted` | Execution stopped operationally (`usage_limit`, `provider_error`, `host_error`, `cancelled`, `unrecorded`) | Preserved for usage/cost accounting and audit; excluded from ternary forecasts, capability thresholds, active-duration calibration and belief scoring |

Acceptance is therefore observable workflow behavior, not proof of correctness.
An execution retains its actual model, effort, outcome and measured usage. An
alternative that never ran has forecasts but no observed outcome.

For each candidate, define {math}`C_{\mathrm{run}}` as forecast worker cost,
{math}`T` as calibrated expected active execution hours (combining expert active-duration
forecasts with shrunk log-residuals from similar executions and pricing log-normal
right-tail variance {math}`\mathbb{E}[T]=\exp(\mu+\sigma^2/2)`), {math}`p_l` as loud-failure
probability and {math}`p_s` as silent-failure probability. The selection objective is:

```{math}
J(a)=\frac{C_{\mathrm{run}}(a)+\lambda T(a)
             +C_l p_l(a)+C_s p_s(a)}
            {\max(\delta,1-p_l(a))}.
```

Here {math}`\lambda` prices elapsed time, {math}`C_l` prices extra overhead per
caught failure, {math}`C_s` prices an escaped defect, and {math}`\delta` is a
numerical probability floor. These penalties express policy assumptions; actual
charges and rework remain separately measured evidence.

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
policy fields and fallback behavior.

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
is refused. If no candidate satisfies the risk cap, bounded widening can request
new forecasts for stronger candidates before applying the review policy.

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
likely a candidate is to clear the task's capability threshold. The clearing and
timely-clearing posteriors each define an interval. An interval starts at the first
position meeting the minimum probability and retains a reliable candidate when one
exists. It ends at the first position at or above that anchor where the remaining
probability gain is below the margin. When reliability is unattainable, the lower
boundary is the gain rule's anchor. When the timely minimum is also unattainable,
only the timely gain cut is kept. The clearing minimum still applies to all
forecasted cells.

When the span of the two intervals exceeds the forecast budget, the fine reader
forecasts their union. The gap lies beyond the clearing gain cut and before the
timely interval. This retains both clearing and timely alternatives for {math}`J` to
compare, while the receipt keeps probabilities for every rank. If the union itself
exceeds the budget, request a sharper independent prior over the same inventory.

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
preserves each expert's loud/silent relationship. A candidate passes the risk policy
when the 80% posterior quantile of that mixture is at most the risk class's epsilon.
The 80% quantile, rather than the 97.5% bound, avoids rejecting every unseen cell for
prior width alone while still rejecting forecasts or histories with substantial silent
risk. The intervals are Bayesian under the stated prior and exchangeability
assumptions; they are not calibrated frequentist guarantees, and epsilon is a selection
rule, not an empirical guarantee about the future defect rate.

When the cap is met, selection samples a predictive expert in proportion to its
support and draws outcome probabilities from that expert's per-cell Dirichlet posterior
(Thompson sampling). Exploration is restricted to candidates that satisfy risk and
credit constraints and whose posterior cost distribution overlaps the mean-best
candidate's, measured by delta-method standard deviations. A fixed fraction of the best
mean cost caps the gap only when a risk class sets it explicitly; by default a small
point-forecast difference cannot lock in a favoured cell and starve an unseen cheaper
one of evidence. After widening is exhausted with no passing candidate, selection
disables sampling and applies a deterministic fallback that requires review. The
{ref}`selection reference <accepted-result-selection>` defines the posterior,
admission and fallback policies.

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

Optional candidate credit budgets add a scheduling constraint against forecast
invocation cost, separate from risk admissibility. They require uniform pricing
and can represent remaining session and weekly windows. Selection refuses a
candidate whose forecast exceeds its available credit, and refuses routing when
none of the evaluated candidates is schedulable. This forecast-based check does
not reserve the full uncertain cost of future retries or prevent concurrent
external spending.

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
| Risk admission | 80% quantile of the mixture accepted-defect posterior | No calibrated coverage guarantee |
| Selection sampling | Thompson sampling within posterior cost overlap | Delta-method overlap is approximate |
| Outcome amendments | Correction and replay of an earlier execution | Requires supported attribution and preserved provenance |

Together, these mechanisms give routing access to both immediate outcomes and
mistakes discovered later. The practical question is whether that information
reduces total cost on future work enough to pay for acquiring and using it.
