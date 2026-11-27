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

The selection objective is the cost to an accepted result, averaged over what is
believed about the candidate:

```{math}
J(a)=\bigl(C_{\mathrm{run}}(a)+\lambda\,\mathbb{E}[T]\bigr)\,\mathbb{E}[R]
     +C_l\,\bigl(\mathbb{E}[R]-1\bigr)+C_s\,\mathbb{E}[S],
\qquad R=\frac{1}{1-\theta_l}=1+e^{u},
\qquad C_{\mathrm{run}}=q\,\mathbb{E}[K].
```

- {math}`\theta_l`, {math}`u`: the candidate's loud rate, the share of attempts that
  fail loudly, and its log-odds.
- {math}`R`: the attempts until a run is not loud, so {math}`R-1` is the loud failures
  per accepted result.
- {math}`S`: the accepted-defect share, silent results divided by accepted results.
- {math}`K`: the credit of one run, and {math}`q` the candidate's cost per credit, or
  0 when worker money is excluded from the comparison (section 7).
- {math}`T`: the active execution hours of one run.
- {math}`\lambda`, {math}`C_l`, {math}`C_s`: the risk class's price of an hour, of the
  extra overhead of a caught failure and of an escaped defect.

In plain words, every attempt costs its credit and its time, a loud run is retried,
and each loud failure and each accepted defect has a price. The averages
{math}`\mathbb{E}[R]` and {math}`\mathbb{E}[S]` are taken over the candidate's two
outcome beliefs (section 5), and {math}`\mathbb{E}[T]` and {math}`\mathbb{E}[K]` over
its beliefs for active time and tokens (below). The penalties express policy
assumptions; actual charges and rework remain separately measured evidence.

Example, with {math}`\lambda=25`, {math}`C_l=50` and {math}`C_s=1{,}000`:
{math}`C_{\mathrm{run}}=2`, {math}`\mathbb{E}[T]=0.5`, {math}`\mathbb{E}[R]=1.25` and
{math}`\mathbb{E}[S]=0.04` give
{math}`J=14.5\times1.25+50\times0.25+1{,}000\times0.04=70.625`.

### Beliefs for active time and tokens

{math}`\mathbb{E}[T]` and {math}`C_{\mathrm{run}}` are averages of beliefs, not
forecasts taken at face value. Active time and each token kind have one log-normal belief per
candidate. Let {math}`A` be such an amount, with

```{math}
\ln A\sim\mathcal{N}(m,s^2),\qquad
s=\max\!\left(s_g,\frac{\tau}{\sqrt{N_{\mathrm{start}}}}\right),\qquad
m=\ln(\mathrm{prediction})-\frac{s^2}{2}.
```

- {math}`\mathrm{prediction}` is where the belief starts: the corrected prediction
  of section 5. Its guess is the experts' forecasts of the amount, averaged on the log
  scale with their support weights; with an empty record the prediction is that guess.
- {math}`s_g` is the stated spread: each expert's stated 95% range read as
  {math}`(\ln hi-\ln lo)/3.92`, combined with the disagreement between experts,
  {math}`s_g^2=\sum_k w_k\,(s_k^2+(m_k-m_g)^2)`, where {math}`w_k` is expert
  {math}`k`'s weight, {math}`m_k` the log of its forecast, {math}`s_k` its stated
  spread and {math}`m_g=\sum_k w_k m_k`.
- {math}`\tau=D\sqrt{\text{scatter share}}` is the run-to-run scatter of the amount
  on one task. {math}`D` is the typical remaining miss of a prediction, as a spread
  of {math}`\ln(\mathrm{actual}/\mathrm{prediction})`: section 5 measures it from the
  record, and it is 1.3 with an empty record. The scatter share is the part of
  {math}`D^2` that is scatter (0.5 by default).
- {math}`N_{\mathrm{start}}=1/(1/\mathrm{worth}+1/N_{\mathrm{task}})` is how many
  runs the prediction is worth: {math}`\mathrm{worth}` is what the record says
  about the model family's own level (the family's own runs plus 5), and
  {math}`N_{\mathrm{task}}` (1) is the most a record on other tasks is worth for one
  task.
- Subtracting {math}`s^2/2` makes the belief's average equal the prediction.

A stated range can widen the belief and never narrow it below
{math}`\tau/\sqrt{N_{\mathrm{start}}}`, which is 1.007 with the defaults and an
empty record. For a
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
3,137 seconds. {math}`\mathbb{E}[T]` is the average of the active-time belief, in
hours. {math}`C_{\mathrm{run}}` prices the averages of the token beliefs at the
candidate's rates; section 7 gives its spread. A token kind that every expert forecasts
at zero, with no non-zero upper end, is exactly zero. A belief is updated by the runs
that record its value: a run with a complete usage record and no active time moves the
token beliefs and leaves the active-time belief unchanged.

### Why the retry factor matters

Under the model's stationary same-candidate retry assumption, attempts continue
until a non-loud result is accepted. At a known loud rate {math}`\theta_l`, with
{math}`p_s` the share of attempts that end silently wrong:

```{math}
\mathbb{E}[N\mid\theta_l]=R=\frac{1}{1-\theta_l},
\qquad
P(\mathrm{silent}\mid\mathrm{accepted})=S=\frac{p_s}{1-\theta_l}.
```

The retry factor {math}`R` charges repeated worker cost and time. The loud penalty adds
detection or rework overhead beyond those repeated attempts. The model does not
assume that a failed attempt switches to another candidate. Real retries can be
correlated or change the task, so the geometric calculation is an approximation
to assess against observed workflow behavior.

The loud rate is not known, and the retry factor is curved in it, so the cost is
averaged over the belief instead of being read at the belief's average. For example, a
belief split evenly between loud rates of 0.2 and 0.6 averages 0.4. The cost at the
average charges {math}`1/0.6=1.67` attempts, and the average over the belief charges
{math}`(1.25+2.5)/2=1.875`. At the same average loud rate, a wider belief therefore
charges more attempts and a higher cost.

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

Separate stateless calls produce cold-start capability priors once per unseen
`(prototype, model, effort)` cell and fine no-memory forecasts once per unseen
`(prototype, model, effort, host)`, against a
synthesized `(representative_task, representative_context)` cluster in
`payloads.sqlite3` (matched on canonicalized task-only text with positive cross-encoder
relevance). Neither receives memories, historical
executions, learned tables or prior conversation. Warm routes whose candidates are
already covered reuse the prototype's capability priors and stored forecasts
directly, calling a model only when active semantic memories require
conditional overrides or a cell has no usable stored forecast. A memory reader separately
supplies conditional adjustments grounded in retrieved evidence and cannot
rewrite those baseline judgments.

### Stored forecasts are reused per host and merged on each value's own scale

The host is the agent that asks for a route, makes the forecast and accepts or rejects
the result. The level of a forecast follows the host that made it, so a stored forecast
is reused only for requests of that host. A prototype keeps one stored forecast per cell
and host: the forecaster's memory-free row, made against the prototype's representative
task and context. A new forecast is stored on the request's primary prototype and
replaces the same host's forecast whole. Capability priors are shared by all hosts.

A stored forecast is usable for a cell of a new request when its prototype matches the
task, it was made under the request's host, and it states every value the cell needs,
including the six token kinds for a cell priced by kind. A request with no host name
belongs to the one unnamed host and reuses only forecasts stored with no host name,
which includes every forecast stored before hosts were named. Each value of the cell's
row is merged over the usable forecasts:

```text
P       = the matched prototypes with a usable stored forecast for the cell
omega_p = w_p / (sum of w_q over q in P)
m       = sum over p in P of omega_p * m_p
s^2     = sum over p in P of omega_p * (s_p^2 + (m_p - m)^2)
```

- `w_p`: prototype `p`'s match weight for the new task. The weights of all matched
  prototypes sum to 1.
- `omega_p`: `p`'s share within `P`. The shares sum to 1.
- `m_p`: the stored value's centre on its own scale: `logit(loud)` for the loud rate
  `theta_l`, `logit(silent / (clean + silent))` for the accepted-defect share `S`,
  `ln(seconds)` for active time and `ln(tokens_a)` for token kind `a`, where
  `logit(p) = ln(p / (1 - p))`.
- `s_p`: the stored value's stated spread, `(phi(hi) - phi(lo)) / 3.92` for a stored
  95% range `[lo, hi]` on the same scale `phi`, or 0 when no range is stored.
- `m`, `s`: the merged centre and spread.

In plain words: the merged guess is the share-weighted average on the value's scale.
The merged spread holds each forecast's own spread plus the distance between the
forecasts, so forecasts that disagree widen the range.

For example, two prototypes with shares 0.7 and 0.3 store loud guesses of 20% and 40%,
each with a spread of 0.5. The centres are `logit(0.2) = -1.386` and
`logit(0.4) = -0.405`, so `m = 0.7 * (-1.386) + 0.3 * (-0.405) = -1.092`, which is
25.1%, and `s^2 = 0.7 * (0.25 + 0.294^2) + 0.3 * (0.25 + 0.687^2) = 0.452`, so
`s = 0.672`. With both guesses at 20%, `s` stays 0.5.

For active time, for example, three prototypes match with weights 0.5, 0.3 and 0.2 and
the third stores nothing usable, so the shares are `0.5 / 0.8 = 0.625` and
`0.3 / 0.8 = 0.375`. The first stores 600 seconds with a range of 300 to 1,200, a
spread of `(ln 1200 - ln 300) / 3.92 = 0.354`; the second stores 2,400 seconds with no
range. The merged guess is `exp(0.625 * ln 600 + 0.375 * ln 2400) = 1,009` seconds with
spread 0.727.

One prototype in `P` has its forecast reused unchanged. A token kind that every
forecast in `P` states as exactly zero stays exactly zero; any other kind is merged over
the forecasts that state it above zero and multiplied by their total share. `tokens` is
the sum of the merged kinds for a cell priced by kind, and is merged itself otherwise.

The merged forecast is the cell's memory-free row, in place of a forecaster's answer:

```text
loud    = logistic(m_loud)            S        = logistic(m_S)
clean   = (1 - loud) * (1 - S)        silent   = (1 - loud) * S
seconds = exp(m_time)                 tokens_a = k_a * exp(m_a)
range   = [inverse_phi(m - 1.96 * s), inverse_phi(m + 1.96 * s)]    for each value with s > 0
```

- `m_loud`, `m_S`, `m_time`, `m_a`: the merged centres of the loud rate, `S`, active
  time and token kind `a`.
- `logistic(x) = 1 / (1 + exp(-x))`, the inverse of `logit`.
- `k_a`: the total share of the forecasts that state kind `a` above zero; 1 when all do.
- `inverse_phi`: `logistic` for the two rates and `exp` for active time and tokens.
- `range`: the value's stated 95% range. The silent range is the range of `S` times
  `1 - loud`, and a kind's range is multiplied by `k_a` like its value. A value with
  `s = 0` has no range.

In plain words: the row states the merged guesses, each with the range from which the
beliefs of sections 1 and 5 read the merged centre and spread back, exactly as from a
forecaster's row.

For example, the loud merge above gives `loud = 0.251` with the range
`[logistic(-1.092 - 1.96 * 0.672), logistic(-1.092 + 1.96 * 0.672)] = [0.082, 0.556]`.
With a merged `S` of 10% the row is `clean = 0.749 * 0.9 = 0.674`, `loud = 0.251` and
`silent = 0.749 * 0.1 = 0.075`.

A cell with no usable stored forecast goes to the forecaster. No forecast, and no
value of one, is built from a prototype's recorded runs: a record with runs and no
stored forecast asks the forecaster once per cell and host, and the runs then count as
evidence beside that forecast. The inference cache of an explicit backend is keyed by
the host's name as well, with the unnamed host as a key of its own, so the forecast
stored under a host is always an answer made for that host. The receipt lists each reused cell's sources (prototype,
match weight, share, host and stored row) and each value's merged centre and spread.
Loading the receipt recomputes the shares, the merge and the row from those sources and
refuses a source stored under another host. The [reference](reference.md) gives the
stored fields.

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
{math}`A` contains participating forecast sources for this decision. Their
forecasts of each rate are merged on the log-odds scale using normalized support
{math}`W_k` (section 5).

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

## 5 Outcome beliefs and bounded exploration

Each expert's fine forecast is a prior, not a measurement. A candidate cell holds two
independent beliefs, each on the log-odds scale
{math}`\operatorname{logit}(p)=\ln\bigl(p/(1-p)\bigr)`: one for its loud rate
{math}`\theta_l`, the share of attempts that fail loudly, and one for its
accepted-defect share {math}`S`, silent results divided by accepted results. A belief
starts from a prediction, the forecast corrected from the record of the same host on
other tasks, and is reshaped by the runs counted for the cell: its own runs in full,
and runs of related cells at a discount.

### One forecast per rate

```{math}
\begin{aligned}
g_l&=\text{loud}, & g_S&=\frac{\text{silent}}{\text{clean}+\text{silent}},\\
m_k&=\operatorname{logit}(g_k), &
s_k&=\frac{\operatorname{logit}(hi_k)-\operatorname{logit}(lo_k)}{3.92},\\
m&=\sum_k w_k m_k, & s_{\mathrm{stated}}^2&=\sum_k w_k\bigl(s_k^2+(m_k-m)^2\bigr),\\
\mathrm{guess}&=\operatorname{logistic}(m)\ \text{held in}\ [0.0001,\,0.9998].
\end{aligned}
```

- clean, loud, silent: one expert's forecast for the cell, floored at 0.0001 and
  normalized. {math}`g_l` and {math}`g_S` are its guesses for {math}`\theta_l` and
  {math}`S`.
- {math}`k`, {math}`w_k`: a participating expert and its support weight. The weights
  sum to 1.
- {math}`lo_k`, {math}`hi_k`: the ends of the expert's stated 95% range, held in
  {math}`[0.0001,0.9999]`: `loud_ci` for {math}`\theta_l`, and `silent_ci` divided by
  clean + silent for {math}`S`.
- {math}`s_k`: the stated spread on the log-odds scale, 0 when no range is stated. A
  95% range spans 3.92 spreads.
- {math}`m`, {math}`s_{\mathrm{stated}}`, guess: the merged centre, the merged
  spread and the merged guess of the rate for this cell on this task.
- {math}`\operatorname{logistic}(x)=1/(1+e^{-x})`, the inverse of the logit.

In plain words, each rate gets one forecast per cell, and experts that disagree make it
less certain.

Example: two experts with weights 0.75 and 0.25 guess {math}`S` at 10% and 25%, each
with spread 0.4. Then {math}`m=0.75\times(-2.197)+0.25\times(-1.099)=-1.92`,
a guess of 12.8% and
{math}`s_{\mathrm{stated}}=\sqrt{0.16+0.75\times0.275^2+0.25\times0.824^2}=0.62`.

### The corrected prediction

The level of a forecast follows the agent that makes it, and outcomes depend on the
agent that accepts the result. So the record, not the forecast, gives the level of
every value, and the forecast only says whether this task looks higher or lower than
that forecaster usually states:

```{math}
\varphi(\mathrm{prediction})=\varphi(\mathrm{level})
  +\gamma\,\bigl(\varphi(\mathrm{guess})-\varphi(\mathrm{usual})\bigr).
```

- {math}`\varphi`: the value's own scale. It is
  {math}`\operatorname{logit}` for {math}`\theta_l` and {math}`S`, each held in
  {math}`[0.0001,0.9999]`, and {math}`\ln` for active time and each token count.
- guess: the merged forecast of the value for this cell on this task.
- level: what happened in the recorded runs of this cell under this host.
- usual: what the forecaster had stated for those runs.
- {math}`\gamma`: the weight on the forecaster's judgment, from 0 to 1.
- prediction: where the belief starts, before the similar runs are counted.

In plain words, the record gives the level and the forecast only says whether this task
looks harder or easier than usual. An empty record gives the guess, {math}`\gamma=1`
the guess moved by level minus usual, and {math}`\gamma=0` the level alone.

Example: level 0.0566, usual 0.188, guess 0.10 and {math}`\gamma=0.74` give
{math}`-2.81+0.74\times(-2.20+1.46)=-3.36` in log-odds, a prediction of 0.034.

Level and usual come down a ladder of five rungs, each leaning on the rung above it:

```{math}
\mathrm{level}(r)=\frac{\sum_i a_i\,y_i+\kappa_r\,\mathrm{level}(r^{+})}{\sum_i a_i+\kappa_r},
\qquad
\mathrm{usual}(r)=\frac{\sum_i a_i\,g_i+\kappa_r\,\mathrm{usual}(r^{+})}{\sum_i a_i+\kappa_r}.
```

- {math}`r`, {math}`r^{+}`: a rung and the one above it. From the top: all hosts, this
  host, this agent under it, this family under it, this cell under it. Above all hosts
  both ladders equal the guess.
- {math}`i`: a finished routed run at rung {math}`r`: its outcome is clean, loud or
  silent and its decision stored a forecast for the cell that ran. Its host and agent
  are the names that decision stores. The evidence candidates of the new decision are
  left out, because they count through their similarity to the new task; a run with no
  decision enters no ladder.
- {math}`a_i`: the age factor of section 3's evidence weights, halving every 14 days,
  divided by {math}`\max(1,A_t)` with {math}`A_t` the summed age factors of the runs of
  ticket {math}`t` in the ladder. A run with no ticket is not limited.
- {math}`y_i`: what happened: 1 for a failure and 0 otherwise for a rate, the logarithm
  of the recorded amount for an amount.
- {math}`g_i`: the merged guess that run's own decision stored, on the same footing.
- {math}`\kappa_r`: the worth of the rung above: 5 at the family rung (`model_worth`)
  and 40 elsewhere (`rung_worth`).

{math}`S` counts accepted results, with silent as the failure; {math}`\theta_l` counts
every finished run, with loud as the failure; an amount counts the runs whose recorded
amount and guess are above zero. In plain words, a cell with few runs leans on its
family, then its agent, its host and all hosts.

Example, for {math}`S` with guess 0.10:

| Rung | {math}`\sum a` | {math}`\sum a\,y` | {math}`\sum a\,g` | level | usual |
| --- | --- | --- | --- | --- | --- |
| All hosts | 160 | 24 | 16 | 0.140 | 0.100 |
| This host | 60 | 6 | 9 | 0.116 | 0.130 |
| This agent | 40 | 2 | 6 | 0.083 | 0.140 |
| This family | 15 | 1 | 3 | 0.071 | 0.185 |
| This cell | 10 | 0 | 2 | 0.057 | 0.188 |

The family level is {math}`(1+5\times0.083)/(15+5)=0.071`.

The weight {math}`\gamma` is measured, one per kind of value, from all hosts' runs:

```{math}
\hat\gamma=\frac{O_H-O_L}{F_H-F_L}\ \text{held in}\ [0,1],\qquad
\gamma=\frac{n\,\hat\gamma+\kappa}{n+\kappa}.
```

- {math}`H`, {math}`L`: the ladder's runs with
  {math}`d_i=\varphi(g_i)-\varphi(\mathrm{usual}_i)` above zero, and the rest.
- {math}`\mathrm{usual}_i`, {math}`\mathrm{level}_i`: both ladders at run {math}`i`'s
  own cell over all the runs, with the record's weighted averages of {math}`g` and
  {math}`y` above all hosts.
- {math}`F_H`, {math}`F_L`: each group's weighted average of {math}`d_i`.
- {math}`O_H`, {math}`O_L`: how far each group's results sat from its levels. For a
  rate,
  {math}`\operatorname{logit}\frac{\sum a_i y_i+1/2}{\sum a_i+1}-\operatorname{logit}\frac{\sum a_i\,\mathrm{level}_i+1/2}{\sum a_i+1}`
  over the group; for an amount, its weighted average of
  {math}`y_i-\mathrm{level}_i`.
- {math}`n`, {math}`\kappa`: the runs' total weight, and 40 (`rung_worth`).
- {math}`\hat\gamma` is 1 when a group is empty or {math}`F_H-F_L` is under 0.05.

In plain words, {math}`\gamma` is the gap that happened divided by the gap that was
forecast, and it starts at 1.

Example: {math}`H` holds 60 runs' worth with 13 silent and levels summing to 9, and
{math}`L` holds 100 with 11 silent and levels summing to 15. Then
{math}`O_H=\operatorname{logit}(13.5/61)-\operatorname{logit}(9.5/61)=0.43` and
{math}`O_L=-0.34`. With {math}`F_H-F_L=1.15`, {math}`\hat\gamma=0.675` and
{math}`\gamma=(160\times0.675+40)/200=0.74`.

The family's own runs set how firmly the prediction is held:

```{math}
\mathrm{worth}=n_{\mathrm{family}}+\kappa_m,\qquad
N_{\mathrm{start}}=\frac{1}{1/\mathrm{worth}+1/N_{\mathrm{task}}},\qquad
s=\max\!\left(s_{\mathrm{stated}},\frac{\sigma_1}{\sqrt{N_{\mathrm{start}}}}\right).
```

- {math}`n_{\mathrm{family}}`, {math}`\kappa_m`: {math}`\sum a_i` at the family rung,
  and 5 (`model_worth`).
- {math}`N_{\mathrm{task}}`: the most a record on other tasks is worth for one task:
  32 for {math}`S`, 16 for {math}`\theta_l`, 1 for an amount.
- {math}`\sigma_1`: one run's spread: {math}`1/\sqrt{p(1-p)}` for a rate with
  {math}`p` the prediction, and {math}`\tau=D\sqrt{\text{scatter share}}` for an amount.
- {math}`s_{\mathrm{stated}}`, {math}`s`: the merged forecast's spread, and the
  belief's starting spread, at most the flat belief's 1.869 for a rate.

Example: {math}`\mathrm{worth}=15+5=20` gives
{math}`N_{\mathrm{start}}=1/(1/20+1/32)=12.3`. With {math}`p=0.034`,
{math}`\sigma_1=5.54` and {math}`\sigma_1/\sqrt{N_{\mathrm{start}}}=1.58`. A stated
range of 4% to 20% gives 0.46, so {math}`s=1.58`.

For an amount, the same ladder over squared misses gives {math}`D`, the typical miss
left after the correction:

```{math}
D^2(r)=\frac{\sum_i a_i\,e_i^2+\kappa_r\,D^2(r^{+})}{\sum_i a_i+\kappa_r},\qquad
e_i=y_i-\varphi(\mathrm{prediction}_i),\qquad D=\sqrt{D^2(\text{this cell})}.
```

- {math}`\mathrm{prediction}_i`: the rule above for past run {math}`i`, with
  {math}`\mathrm{level}_i`, {math}`\mathrm{usual}_i` and the {math}`\gamma` of its kind
  of value.
- {math}`D^2(r^{+})` above all hosts: `duration_miss` squared, {math}`1.3^2`.

Example: 100 runs' worth on one cell, each off by a factor of {math}`e^{0.5}`, give
{math}`D^2` = 0.661, 0.368, 0.284, 0.252 and 0.250 down the rungs, so {math}`D=0.50`.

Total tokens and each of the six token kinds have their own ladders, weight and
{math}`D`, empty until runs record them. A token kind forecast at zero stays zero. The
receipt stores every sum under `semantic.prediction`, and each derived number is
recomputed from those sums when the receipt is loaded.

### The start of each belief

```{math}
\begin{aligned}
N_{\mathrm{start}}&=\frac{1}{1/\mathrm{worth}+1/N_{\mathrm{task}}}, &
\sigma_1&=\frac{1}{\sqrt{p(1-p)}},\\
s&=\min\Bigl(1.869,\ \max\bigl(s_{\mathrm{stated}},\ \sigma_1/\sqrt{N_{\mathrm{start}}}\bigr)\Bigr),\\
x_j&=-16+0.01\,j,\quad j=0,\dots,2500,\\
q_j(m)&=\frac{\exp\bigl(-(x_j-m)^2/(2s^2)\bigr)}{\sum_i\exp\bigl(-(x_i-m)^2/(2s^2)\bigr)},\\
\text{centre}&:\ \text{the}\ m\ \text{with}\ \sum_j q_j(m)\operatorname{logistic}(x_j)=p.
\end{aligned}
```

- {math}`p`: the corrected prediction of the rate, held in {math}`[0.0001,0.9998]`.
- worth: how many runs the record says about this family: the family's own runs plus
  `model_worth`, 5.
- {math}`N_{\mathrm{task}}`: the most a record on other tasks is worth for one task,
  32 for {math}`S` (`task_worth_silent`) and 16 for {math}`\theta_l`
  (`task_worth_loud`).
- {math}`N_{\mathrm{start}}`: how many runs the prediction is worth. With an empty
  record it is 4.32 for {math}`S` and 3.81 for {math}`\theta_l`.
- {math}`\sigma_1`: the spread of one run's result on the log-odds scale.
- {math}`s`: the spread at the start. A stated range can widen it, never narrow it:
  a forecaster cannot claim the authority of many real runs. 1.869 is the flat belief,
  the spread of a 95% range of {math}`[0.025,0.975]`.
- {math}`x_j`, {math}`q_j(m)`: the grid of 2,501 log-odds points, the same for every
  belief, and a bell curve of centre {math}`m` and spread {math}`s` on it.
- centre: found by 60 halvings of {math}`[-25,25]`, keeping the upper half when the
  grid average at the midpoint is below {math}`p`. The grid's top value is
  {math}`\operatorname{logistic}(9)=0.99988`, so every prediction up to 0.9998 can be
  met.

In plain words, the start is a bell curve on the log-odds scale whose average, not its
median, equals the prediction, and it is as wide as a record worth
{math}`N_{\mathrm{start}}` runs allows. One fixed grid makes every belief a finite sum,
so a stored decision is recomputed exactly when it is loaded.

Example, with an empty record: clean 0.72, loud 0.20, silent 0.08 with
`silent_ci = [0.04, 0.16]` gives {math}`p=0.10` for {math}`S` and {math}`s_{\mathrm{stated}}=0.40` (a range of 5% to
20%). {math}`\sigma_1/\sqrt{N_{\mathrm{start}}}=3.33/2.08=1.60`, so {math}`s=1.60` and
the centre is {math}`-3.04`. For {math}`\theta_l`, {math}`p=0.20` and
`loud_ci = [0.10, 0.35]` give {math}`s=1.28` and centre {math}`-1.79`.

### Counted runs reshape the belief

The same candidate runs that feed the capability threshold update the belief. Every
candidate is kept, whatever cell ran it, and counts for each forecasted cell {math}`c`
at a discount set by how related the two cells are:

```{math}
\begin{aligned}
\delta(c_i,c)&=\begin{cases}
1&\text{same family, same effort (the same cell)}\\
\delta_{\mathrm{effort}}&\text{same family, another effort}\\
\delta_{\mathrm{model}}&\text{another family of the same agent}\\
\delta_{\mathrm{agent}}&\text{another agent}
\end{cases}\\
w_i&=m_i\,\mathrm{sim}_i\,a_i\,\delta(c_i,c),\qquad
W_t=\sum_{i\in t}w_i,\qquad
w_i\leftarrow\frac{w_i}{\max(1,W_t)},\qquad
n_k=\sum_{i:\,y_i=k}w_i.
\end{aligned}
```

- {math}`c`, {math}`c_i`: the cell being judged, and the cell that ran candidate run
  {math}`i`.
- {math}`\delta`: the relatedness discount. The first line that fits is used. The
  defaults are {math}`\delta_{\mathrm{effort}}=0.5`, {math}`\delta_{\mathrm{model}}=0.25`
  and {math}`\delta_{\mathrm{agent}}=0.25` (`sampling.effort_discount`,
  `sampling.model_discount`, `sampling.agent_discount`).
- {math}`m_i`, {math}`\mathrm{sim}_i`, {math}`a_i`: the run's recorded weight, its
  similarity to the new task and its age factor.
- {math}`t`, {math}`W_t`: a ticket, and the weight its candidate runs carry for cell
  {math}`c`. The limit follows the discount, so it is taken per judged cell.
- {math}`y_i`, {math}`n_k`: run {math}`i`'s outcome, and the counted amount of outcome
  {math}`k`. {math}`n_c`, {math}`n_l` and {math}`n_s` are the clean, loud and silent
  counts.

In plain words: a run by a related cell counts as a fraction of a run by the cell
itself, and one ticket still counts for at most one run.

A cell's family comes from the model registry and its agent from the label on its
candidate. A run's agent is the label stored for the chosen cell in the receipt that
the run's `decision_id` names. A run with no decision, a receipt older than v11 or an
unlabeled candidate belongs to the one unnamed agent, as does an unlabeled cell of the
new request.

Example: the judged cell is family `f1` at effort `high` under agent `a1`, and every
run has {math}`m_i=1`.

| Run | Ran on | Relation | {math}`\mathrm{sim}` | {math}`a` | {math}`\delta` | Weight | Ticket | After the limit | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `f1` `high`, `a1` | Same cell | 0.8 | 1 | 1 | 0.80 | `t1` | 0.64 | clean |
| 2 | `f1` `low`, `a1` | Same family | 0.9 | 1 | 0.5 | 0.45 | `t1` | 0.36 | loud |
| 3 | `f2` `high`, `a1` | Same agent | 0.8 | 0.5 | 0.25 | 0.10 | `t2` | 0.10 | clean |
| 4 | `f3` `high`, `a2` | Another agent | 0.4 | 1 | 0.25 | 0.10 | `t3` | 0.10 | silent |

Ticket `t1` carries {math}`0.80+0.45=1.25`, so both of its runs are divided by 1.25.
The counts are {math}`n_c=0.74`, {math}`n_l=0.36` and {math}`n_s=0.10`. With the three
discounts at 0 only run 1 counts, as 0.80 clean.

The counts then reshape each belief:

```{math}
\begin{aligned}
b_j&\propto q_j(m)\,\operatorname{logistic}(x_j)^{n_{\mathrm{yes}}}\,
     \bigl(1-\operatorname{logistic}(x_j)\bigr)^{n_{\mathrm{no}}},\qquad \sum_j b_j=1,\\
S&:\ n_{\mathrm{yes}}=n_s,\ n_{\mathrm{no}}=n_c;\qquad
\theta_l:\ n_{\mathrm{yes}}=n_l,\ n_{\mathrm{no}}=n_c+n_s,\\
E[S]&=\sum_j b_j\operatorname{logistic}(x_j),\\
E\!\left[\frac{1}{1-\theta_l}\right]&=\sum_j b_j\,(1+e^{x_j}),\qquad
E\!\left[\frac{\theta_l}{1-\theta_l}\right]=\sum_j b_j\,e^{x_j},\\
q\ \text{point}&=\operatorname{logistic}(x_j)\ \text{at the first}\ j\
  \text{whose running total of}\ b\ \text{reaches}\ q.
\end{aligned}
```

- {math}`b_j`: the belief after the counted runs, computed in logarithms.
- {math}`E[S]`: the average of the belief of {math}`S`; the average of the belief of
  {math}`\theta_l` is read the same way.
- {math}`E[1/(1-\theta_l)]`, {math}`E[\theta_l/(1-\theta_l)]`: expected attempts, and
  expected loud failures per accepted result.
- {math}`q` point: the 2.5% and 97.5% points of each rate, for reports.

In plain words, the start counts as {math}`N_{\mathrm{start}}` runs and each counted
run as its weight. The forecast never fades out: however many runs are counted, it
keeps the weight of {math}`N_{\mathrm{start}}` runs. Because each count halves for
every 14 days of age, a regression or a recovery moves the belief as older runs fade.
A cell with no counted run keeps its start; global expert support weights experts but
is not a per-cell sample size. The discount applies to these outcome counts only: active
time and tokens are calibrated from the same cell's candidates alone, each at
{math}`m_i\,\mathrm{sim}_i\,a_i` after the per-ticket limit over that cell's own runs,
and the capability threshold keeps its own strength factors.

Example: with the forecast above, at the start {math}`E[S]=0.100` and the 95% range is
0.2% to 52%. With {math}`n_c=4` and {math}`n_s=0`, {math}`E[S]=0.053`. For
{math}`\theta_l` at the start, expected attempts are 1.38, against 1.25 at the average
rate.

Selection reads the two averages:

```{math}
P_{\mathrm{loud}}=E[\theta_l],\qquad
P_{\mathrm{silent}}=(1-E[\theta_l])\,E[S],\qquad
P_{\mathrm{clean}}=(1-E[\theta_l])\,(1-E[S]).
```

These three means are reports. The cost {math}`J` of section 1 is averaged over the
two beliefs, through {math}`E[1/(1-\theta_l)]=\mathbb{E}[R]` and {math}`E[S]`, and is
not the cost at these means. For example, {math}`E[\theta_l]=0.20` and
{math}`E[S]=0.10` give {math}`P_{\mathrm{silent}}=0.08` and
{math}`P_{\mathrm{clean}}=0.72`. The reported ranges are those of the two beliefs.
They are Bayesian under the stated start and exchangeability assumptions; they are not
calibrated frequentist guarantees, and epsilon is a selection rule, not an empirical
guarantee about the future defect rate.

### The chance of meeting the limit

Admission asks one question of each forecast cell: how likely is it that the cell keeps
silently wrong accepted results within the risk class's limit?

```{math}
\pi(c)=P(S\le\varepsilon)=\sum_{j:\ \operatorname{logistic}(x_j)\le\varepsilon} b_j.
```

- {math}`c`: a forecast cell.
- {math}`S`: the accepted-defect share, silent results divided by accepted results.
- {math}`\varepsilon`: the risk class's limit on {math}`S`.
- {math}`b_j`, {math}`x_j`: the cell's belief for {math}`S` and its grid points.
- {math}`\pi(c)`: the share of that belief at or below {math}`\varepsilon`.

In plain words, {math}`\pi` is the chance, under the cell's current belief, that no
more than {math}`\varepsilon` of its accepted results are silently wrong.

Example: {math}`\varepsilon` is 0.10. The cell forecast at clean 0.72, loud 0.20,
silent 0.08 has {math}`E[S]=0.100`, and 70% of its belief lies at or below 0.10, so
{math}`\pi=0.70`. Four counted clean runs move {math}`\pi` to 0.85.

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
belief, which is all an unproven cell has, puts the cell on trial instead of
shutting it out for width alone, while a forecast or history with substantial silent
risk is still excluded.

Example, with one expert, no stated range, no counted run and
{math}`\varepsilon=0.10`:

| Forecast of {math}`S` | {math}`\pi` | Tier | 80% point of {math}`S` |
| --- | --- | --- | --- |
| 2% | 0.957 | trusted | 0.021 |
| 10% | 0.702 | on trial | 0.156 |
| 50% | 0.011 | excluded | 0.692 |

The last column is the receipt's `admission_risk`, the {math}`q` point of the same
belief. It is at most {math}`\varepsilon` exactly when {math}`\pi\ge q`, so the
trusted tier is the same test read from the other side.

### The spread of the cost

The cost of section 1 is uncertain because four things about a cell are: the credit
{math}`K` and active time {math}`T` of one run, its loud rate and its accepted-defect
share. Each has its own belief, and the four are independent, so the spread of the cost
follows from their means and variances with no approximation. Write the cost before
averaging as {math}`X R-C_l+C_s S`, with {math}`X=qK+\lambda T+C_l`:

```{math}
\begin{aligned}
\mathrm{Var}(J)&=\mathbb{E}[X^2]\,\mathbb{E}[R^2]-\bigl(\mathbb{E}[X]\,\mathbb{E}[R]\bigr)^2
                +C_s^2\,\mathrm{Var}(S),\\
\mathbb{E}[X]&=C_{\mathrm{run}}+\lambda\,\mathbb{E}[T]+C_l,\\
\mathbb{E}[X^2]&=q^2\,\mathrm{Var}(K)+\lambda^2\,\mathrm{Var}(T)+\mathbb{E}[X]^2,\\
\mathrm{Var}(T)&=\mathbb{E}[T]^2\,\bigl(e^{s'^2}-1\bigr),\\
\mathrm{sd}(c)&=\sqrt{\mathrm{Var}(J)}.
\end{aligned}
```

- {math}`X`: what one attempt costs, plus {math}`C_l`.
- {math}`R=1+e^{u}`: the attempts until a run is not loud. {math}`\mathbb{E}[R]` and
  {math}`\mathbb{E}[R^2]` are the averages of {math}`1+e^{x_j}` and of its square under
  the belief {math}`b_j` of the loud rate on its grid.
- {math}`\mathrm{Var}(S)`: the average of {math}`S^2` minus {math}`\mathbb{E}[S]^2`,
  summed on the grid of the belief of {math}`S`.
- {math}`\mathrm{Var}(K)`: the variance of the credit of one run (section 7), and
  {math}`q` the cost per credit, or 0 when worker money is excluded.
- {math}`s'`: the spread of the active-time belief after its counted runs (section 1).
- {math}`\mathrm{sd}(c)`: the spread of the cell's cost, the receipt's `cost_sd`.

In plain words, the cost is a product of what one attempt costs and how many attempts
are needed, plus the price of accepted defects, and the variance of a product of
independent quantities is exact in their first two moments.

Example, continuing the example of section 1 with {math}`q=1`,
{math}`\mathrm{sd}(K)=1`, {math}`s'=0.6`, {math}`\mathbb{E}[R^2]=1.60` and
{math}`\mathrm{sd}(S)=0.03`: {math}`\mathrm{Var}(T)=0.25\,(e^{0.36}-1)=0.1083`,
{math}`\mathbb{E}[X]=64.5` and
{math}`\mathbb{E}[X^2]=1+625\times0.1083+4{,}160.25=4{,}228.96`. Then
{math}`\mathrm{Var}(J)=4{,}228.96\times1.60-80.625^2+900=1{,}165.94` and
{math}`\mathrm{sd}(c)=34.15`.

A wider stated range on active time or on a token kind raises the spread of that cell's
cost. It does not move the cell's {math}`J`, because a belief averages to its prediction
however wide it is, and it changes no number of another cell.

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
- {math}`J(c)`, {math}`\mathrm{sd}(c)`: the cell's cost averaged over its beliefs
  (section 1) and the exact spread of that cost (above).
- {math}`c^{*}`: the pool cell with the lowest {math}`J`.
- {math}`u`: `uncertainty_multiplier`, 1. A risk class's `explore_within` additionally
  caps the gap at that fraction of {math}`|J(c^{*})|`.
- {math}`\tilde J(c)`: the cell's cost under the seed's draw (below).

Example of eligibility: {math}`c^{*}` has {math}`J` 70.6 and spread 34.2. A cell with
{math}`J` 110 and spread 20 is eligible, because the gap of 39.4 is at most 54.2. A cell
with {math}`J` 140 and spread 10 is not, because 69.4 exceeds 44.2.

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

### The draw

Each forecast cell draws one value of everything it is uncertain about, and the lowest
sampled cost among the eligible cells wins (Thompson sampling).

```{math}
\begin{aligned}
\tilde S&=\operatorname{logistic}(x_j),
  &&j=\text{the first grid point where the running total of the belief of }S\text{ reaches }U_S,\\
\tilde R&=1+e^{x_j},
  &&j=\text{the same on the belief of the loud rate, with }U_u,\\
\tilde T&=\exp\bigl(m'+s'\,z(U_T)\bigr)/3600,\\
\tilde n_a&=\exp\bigl(m_a+s_a\,z(U_K)\bigr)&&\text{for every token kind }a,\\
\tilde J&=\bigl(q\tilde K+\lambda\tilde T\bigr)\,\tilde R+C_l\,(\tilde R-1)+C_s\,\tilde S.
\end{aligned}
```

- {math}`U_S`, {math}`U_u`, {math}`U_T`, {math}`U_K`: four uniform numbers strictly
  between 0 and 1, drawn in this order for every forecast cell.
- {math}`x_j`: a point of the grid of log-odds. It is the last point when rounding
  keeps the running total below {math}`U`.
- {math}`z(U)`: the standard normal value with probability {math}`U` below it.
- {math}`m'`, {math}`s'`: the centre and spread of the active-time belief, in log
  seconds; {math}`\tilde T` is in hours.
- {math}`m_a`, {math}`s_a`: the centre and spread of the belief of token kind
  {math}`a`, and {math}`\tilde n_a` its sampled count. One {math}`U_K` serves every
  kind, because the kinds are taken to move together (section 7).
- {math}`\tilde K`: the credit of the counts {math}`\tilde n_a` at the candidate's
  rates.
- {math}`\tilde J`: the sampled cost.

In plain words, a wide belief sometimes draws a low cost, so its cell is tried, and the
trial's outcome then narrows the belief. No predictive expert is drawn: every value of
a cell is one belief merged over the experts, and experts that disagree have already
widened it.

Example: {math}`\tilde S=0.02`, {math}`\tilde R=1.25`, {math}`\tilde T=0.4` and
{math}`q\tilde K=1.50` give {math}`\tilde J=11.5\times1.25+12.5+20=46.875`.

The numbers come from the PCG64 generator seeded with the decision's stored seed. Each
is {math}`((r\gg12)+0.5)/2^{52}` from one raw 64-bit output {math}`r`, so replaying a
decision does not depend on a library's distribution methods. Every forecast cell
draws, in a stored order and whether or not it is eligible, so the stream of numbers
does not depend on the pool. A stored decision is checked by replaying its draw from
its seed.

### The flag

```{math}
\begin{aligned}
\texttt{cap\_met}&\iff \text{the chosen cell is trusted},\\
\texttt{needs\_review}&\iff \text{not}\ \texttt{cap\_met}.
\end{aligned}
```

In plain words, the flag says that the chosen cell is not trusted. It is advice to the
caller, and nothing is enforced or recorded for it.

Example: a chosen cell with {math}`\pi=0.702` is on trial, so `cap_met` is false and
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
| Outcome beliefs | Two independent per-cell beliefs on a fixed log-odds grid, for the loud rate and the accepted-defect share: a bell-curve start worth `N_start` runs, reshaped by the cell's weighted counts | The two rates are treated as independent; the worths (5, 32 and 16) and the 14-day evidence half-life are fixed modelling choices |
| Active time and token beliefs | Per-cell log-normal belief with a conjugate update from same-cell runs | The miss `D`, scatter share and worths are fixed modelling choices; token kinds are taken to move together |
| Credit check | 80% point of a log-normal matched to the average and variance of run credit | A moment match, not the exact distribution of a sum |
| Admission tiers | Share of the accepted-defect belief within epsilon, cut at 0.80 and 0.20 | No calibrated coverage guarantee; the two lines are policy choices |
| Cost and its spread | The cost to an accepted result averaged over four independent per-cell beliefs, with the exact variance of that cost | Credit, active time, the loud rate and the accepted-defect share are treated as independent; same-cell retries are taken to be stationary |
| Selection sampling | Thompson sampling within cost overlap, among trusted and on-trial cells: one draw of both rates, active time and token counts per cell | Overlap within one spread on each side is a rule, not a test; token kinds are drawn together |
| Outcome amendments | Correction and replay of an earlier execution | Requires supported attribution and preserved provenance |

Together, these mechanisms give routing access to both immediate outcomes and
mistakes discovered later. The practical question is whether that information
reduces total cost on future work enough to pay for acquiring and using it.
