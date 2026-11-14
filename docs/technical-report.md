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
| `loud` | A gate or review caught a failure | The cost model charges another attempt on the same candidate |
| `silent` | A defect escaped acceptance and was discovered later | Selection includes an escaped-defect penalty and an accepted-result risk cap |

Acceptance is therefore observable workflow behavior, not proof of correctness.
An execution retains its actual model, effort, outcome and measured usage. An
alternative that never ran has forecasts but no observed outcome.

For each candidate, define {math}`C_{\mathrm{run}}` as forecast worker cost,
{math}`T` as forecast hours, {math}`p_l` as loud-failure probability and
{math}`p_s` as silent-failure probability. The selection objective is:

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

Separate stateless calls produce the initial capability prior and the fine
no-memory forecast. Each receives the current task/project and the relevant
candidate identities and constants. Neither receives memories, historical
executions, learned tables or prior conversation. A memory reader separately
supplies conditional adjustments grounded in retrieved evidence and cannot
rewrite those independent judgments.

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
this threshold. Predictive weights mix the independent and memory-conditioned
priors. Consensus ordering preserves each model's effort chain; a monotone
projection makes the mixed cumulative distribution consistent with that order.
Call the resulting threshold prior {math}`\pi_0(h)`.

### Weighted likelihood update

Only retrieved executions with an actual candidate present in the inventory
contribute threshold likelihoods. Let {math}`r_i` be an executed candidate's
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

Each {math}`w_i` multiplies the stored evidence weight, sigmoid-transformed
retrieval relevance, severity/cause strength and a recency discount. Failures
explicitly attributed to capability retain their severity strength; unknown
causes contribute less. Supported assertions that a failure was not capability
caused can remove its threshold contribution without removing the outcome.

This is a weighted-likelihood, or generalized Bayesian, update. The relevance
weights are heuristic transfer strengths, not calibrated probabilities that two
tasks are equivalent. Related historical tasks are treated as noisy evidence
about the current threshold under the proposed order.

### From posterior to fit band

The posterior cumulative probability {math}`P(H\leq r\mid D,x)` estimates how
likely a candidate is to clear the task's capability threshold. The lower fit
boundary reaches the minimum clearing probability. The upper boundary retains a
reliable candidate when one exists, then excludes stronger candidates whose
remaining clearing-probability gain is below the margin. If the reliability target
is unattainable, the gain rule is anchored at the lower boundary instead.

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

Measured tokens and elapsed time add broad log-normal resource scores relative
to the baseline. Unknown measurements add no resource evidence. A compact
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

## 5 Risk envelopes and bounded exploration

The fine forecast supplies both an expected cost and an accepted-result defect
envelope. For each expert, howhard first computes the joint ratio
{math}`p_s/\max(\delta,1-p_l)`. It then takes weighted 2.5% and 97.5% quantiles
of those ratios and expands the envelope to include the ratio of aggregate means.
This preserves each expert's loud/silent relationship. Dividing unrelated
marginal bounds would lose it.

A candidate passes the risk policy when the upper accepted-result defect envelope
is at most the risk class's epsilon. These bounds are heuristic forecast
envelopes, not Bayesian credible intervals or calibrated frequentist confidence
intervals. Epsilon is a selection rule, not an empirical guarantee about the
future defect rate.

When the cap is met, selection samples a predictive expert and Dirichlet outcome
probabilities. Concentration is moment-matched from expert disagreement, envelope
width and recorded evidence counts, within fixed bounds. This is an approximate
sampling model, rather than a conjugate posterior obtained by counting outcomes.

Exploration is restricted to candidates that satisfy risk and credit constraints
and whose mean objective is close to the best by both a configured cost fraction
and an uncertainty-based allowance. After widening is exhausted with no passing
candidate, selection disables sampling and applies a deterministic fallback that
requires review. The {ref}`selection reference <accepted-result-selection>` defines the
envelopes, concentration and fallback policies.

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
| Independent LLM judgments | Initial task-specific orders and forecasts | LLM judgment needs empirical validation |
| Capability threshold | Generalized Bayesian update with weighted noisy likelihoods | Scalar order and heuristic transfer from related tasks |
| Cross-model ordering | Constrained local posterior/MAP search | Approximate search over rankings |
| Memory support | Tempered, recency-weighted predictive evidence | Forecast support does not establish truth or causality |
| Risk envelope | Weighted expert quantiles including the aggregate estimate | No calibrated coverage guarantee |
| Selection sampling | Moment-matched Dirichlet with bounded exploration | Approximate uncertainty model |
| Outcome amendments | Correction and replay of an earlier execution | Requires supported attribution and preserved provenance |

Together, these mechanisms give routing access to both immediate outcomes and
mistakes discovered later. The practical question is whether that information
reduces total cost on future work enough to pay for acquiring and using it.
