# Technical report

## 1 Introduction

This section states the problem, follows one example through a decision and outlines
the method.

An AI agent working in a software project hands many of its tasks to worker models, and
for each task it must pick a model and a reasoning effort. A capable model at high
effort can spend more than a routine task needs. A cheaper choice can cost more in the
end: a review or a test gate may catch its failure and force another attempt, or the
gate may pass a defect that surfaces weeks later. howhard makes this choice for each
task. For every candidate it estimates the expected cost of reaching an accepted result,
including retries, active time and the price of an escaped defect. Among the
candidates whose costs are too close to separate, it chooses with a randomized rule
that keeps learning.

No fixed ranking of models settles the choice, because it depends on the task and on
the project. Evidence is scarce: a project records tens or hundreds of runs spread over
many models and efforts, and some outcomes are known only when a later bug fix traces a
defect back to an earlier accepted run. We therefore start from the judgment of a
language model that reads the task and the project, and we correct that judgment with
the project's own record. When an investigation attributes a defect to an earlier run,
that run's outcome is amended and the learning is replayed.

### A running example

We follow one decision through the report. A host agent asks for a route for the task
"Make transaction rollback safe". The task's risk class prices an hour of active work at
25 USD, a caught failure at 50 USD and an escaped defect at 1,000 USD, and it limits
silently wrong accepted results to 10%. We call one model family at one reasoning effort
a *cell*. The host offers three cells, each priced per token, and the project has no
recorded runs yet. A language model forecasts each cell's outcome probabilities, active
time and tokens:

| Cell | Model and effort | USD per million tokens | Clean | Loud | Silent | Active seconds | Tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | small model, high | 1 | 0.30 | 0.40 | 0.30 | 900 | 150,000 |
| B | mid-sized model, medium | 3 | 0.72 | 0.20 | 0.08 | 1,800 | 400,000 |
| C | large model, high | 15 | 0.931 | 0.05 | 0.019 | 3,600 | 600,000 |

A loud result is a failure that the host catches before acceptance, and a silent result
is one that it accepts with a defect. The forecaster also states that B's loud
probability lies between 0.10 and 0.35 and its silent probability between 0.04 and 0.16.

The language model first orders the three cells for this task, A weakest, and judges
B and C capable enough. No run of A's family has been recorded on this kind of task, so
A still receives a forecast through a slot reserved for unproven families. Selection
then compares the three cells. Half of A's accepted results are expected to be wrong,
so A is excluded. B meets the 10% limit with probability 0.70 and is admitted on trial,
and C meets it with probability 0.96 and is trusted. The expected cost to an accepted
result is 137.8 USD for B and 60.8 USD for C, although one run of C costs 9.00 USD and
one run of B 1.20 USD: B's expected escaped defects alone cost 100 USD. B's belief is
wide enough that its cost overlaps C's, so B stays eligible for a randomized draw. With
the decision's seed the draw picks C. C is trusted, so the decision carries no review
flag.

The host runs C and records the outcome and the measured usage. Suppose a bug fix later
the same day traces a defect to C's accepted run, and its outcome is amended to silent.
The next similar task counts that run as silent evidence about C, and at a discount as
evidence about B. C falls to on trial with an expected cost of 152.9 USD, B rises to
168.7 USD, and the draw for that task picks B, flagged for review.

### Overview

```{figure} assets/decision-flow.svg
:alt: The decision flow. A task and its candidate cells pass through a capability stage, a forecast stage and selection before the host runs the chosen cell. The project record feeds the first two stages and receives each outcome, and a later defect attribution amends the record.
:width: 100%

The decision flow. Each stage reads the project record, and every outcome, including a
later amendment, returns to it.
```

A decision has three stages. The capability stage orders every candidate cell for the
task and keeps a band of cells strong enough without being needlessly strong. The
forecast stage obtains outcome and usage forecasts for those cells, corrects them with
the record of earlier tasks and turns them into beliefs that recorded runs of similar
tasks reshape. Selection computes each cell's expected cost to an accepted result,
admits cells by their chance of meeting the defect limit and draws one cell among those
whose costs overlap.

Section 2 relates this design to earlier work and section 3 states the problem and the
notation. Sections 4 to 12 describe the method in the order in which its parts depend on
one another: the forecasts (4), how forecast sources are weighted (5), which recorded
runs count as evidence (6), the capability band (7), the corrected predictions (8), the
outcome beliefs (9), the beliefs for time, tokens and credit (10), selection (11) and
the re-estimation of constants (12). Section 13 covers amendments after a defect is
found, section 14 the operational boundaries, section 15 the evaluation protocol and
section 16 the statistical assumptions and limits.

## 2 Related work

This section places howhard among methods for routing requests to language models and
names the statistical results it builds on.

Several methods reduce the cost of language-model inference by sending each query to the
cheapest model likely to answer it well. RouteLLM (Ong et al., 2025) trains routers on
human preference data to choose between a stronger and a weaker model for each query.
Hybrid LLM (Ding et al., 2024) routes each query to a small or a large model from its
predicted difficulty and a quality level that can be tuned at test time. FrugalGPT
(Chen, Zaharia and Zou, 2024) learns a cascade that queries models in sequence until a
learned scorer accepts an answer, and AutoMix (Aggarwal et al., 2024) escalates to a
larger model when a smaller model's few-shot self-verification is not confident, using a
POMDP-based router. BEST-Route (Ding et al., 2025) chooses a model together with the
number of responses to sample from it. MixLLM (Wang et al., 2025) treats routing among
several models as a contextual bandit that trades quality against cost and latency and
keeps learning from feedback.

These methods route single queries among language models and are trained and evaluated
on benchmark datasets. howhard routes agent tasks, which run for minutes to hours with
tools in a changing repository, and it chooses a reasoning effort as well as a model. Its
evidence is the project's own record of runs. An outcome distinguishes failures caught
before acceptance from defects found later, and the record can change after the fact
when a defect is attributed to an earlier run.

The method combines standard statistical tools. Selection is Thompson sampling
(Thompson, 1933; Russo et al., 2018): each cell draws one value of everything it is
uncertain about, and the lowest sampled cost wins. Evidence from related tasks enters
the capability threshold through likelihoods raised to fractional weights, a form of
generalized Bayesian updating (Bissiri, Holmes and Walker, 2016). Forecast sources are
weighted by their tempered, recency-discounted log scores, as in exponentially weighted
expert aggregation (Cesa-Bianchi and Lugosi, 2006), and the log score is a strictly
proper scoring rule (Gneiting and Raftery, 2007). Outcome rates are held as normal
beliefs on the log-odds scale, a logistic-normal family (Aitchison and Shen, 1980).
Active time and token counts have log-normal beliefs with the conjugate normal update,
and the levels of a forecaster's record are estimated by partial pooling across nested
groups (Gelman et al., 2013). The credit of one run is matched to a log-normal by its
first two moments (Fenton, 1960), and the mixed capability prior is made monotone by the
pool-adjacent-violators algorithm (Ayer et al., 1955).

## 3 Problem setup

This section defines the candidates, the outcomes, the record and the objective, and it
fixes the notation for the rest of the report.

### Candidates, hosts and risk classes

A route is requested by a *host*: the agent that asks for the route, makes the forecasts
and accepts or rejects the result. The host supplies an inventory {math}`\mathcal{C}` of
candidates. Each candidate names a model and an effort, may carry an *agent* label for
the agent product that would run it, and may carry prices. A model belongs to a *family*
in the model registry, and a cell {math}`c` is one family at one effort. Each task has a
task type, which maps to a risk class. The risk class fixes the price of an hour of
active work {math}`\lambda`, the extra cost {math}`C_l` of a caught failure, the cost
{math}`C_s` of an escaped defect and the limit {math}`\varepsilon` on the share of
accepted results that are silently wrong. The running example uses the packaged
`standard` class: {math}`\lambda=25`, {math}`C_l=50`, {math}`C_s=1{,}000` and
{math}`\varepsilon=0.10`.

### Outcomes

Every run ends in one of four outcomes:

| Outcome | Meaning | Use in learning |
| --- | --- | --- |
| `clean` | Accepted, and no defect has been found | Counts as a success |
| `loud` | A gate or review caught a failure before acceptance | Counts as a caught failure |
| `silent` | Accepted, and a defect was found later | Counts as an escaped defect |
| `interrupted` | Stopped for an operational reason: `usage_limit`, `provider_error`, `host_error`, `cancelled` or `unrecorded` | Kept for usage and cost accounting and audit; excluded from every learned quantity |

Acceptance is the observable behaviour of the workflow. A clean label records that no
defect has been found so far, and later evidence can amend it.

### The record

The *record* is the project's store of recorded runs. Each run keeps its model, effort,
outcome, measured usage, the time it was observed and, when it has one, its *ticket*: the
unit of work it belongs to, shared by retries. A run made through a decision also keeps
the decision's *receipt*, which freezes the decision's inputs, forecasts and choice. A
*routed run* is a run whose receipt holds a forecast for the cell that ran. A
*prototype* is a cluster of earlier tasks of one task type, represented by a synthesized
task and project context. A new task matches some prototypes, each with a match weight,
and the weights of all matched prototypes sum to one. The
[reference](reference.md) describes the stored forms.

### The objective

Let {math}`\theta_l` be a cell's *loud rate*, the share of its attempts that fail
loudly, and {math}`\theta_s` the share that end silently wrong. The cost model retries a
loud attempt on the same cell until a non-loud result is accepted. The number of
attempts is then geometric, with mean {math}`R`, and the *accepted-defect share*
{math}`S`, the share of accepted results that are silently wrong, follows from the two
rates:

```{math}
:label: attempts

R=\frac{1}{1-\theta_l}=1+e^{u},\qquad
S=\frac{\theta_s}{1-\theta_l},
```

where {math}`u=\operatorname{logit}\theta_l` and
{math}`\operatorname{logit}p=\ln\bigl(p/(1-p)\bigr)`. Let {math}`K` be the credit of one
run, its token counts priced at the candidate's list prices, {math}`\psi` the
candidate's cost per credit and {math}`T` the active hours of one run. The cost of one
attempt is {math}`\psi K+\lambda T`, and with {math}`R` and {math}`S` from
{eq}`attempts` the expected cost of reaching an accepted result is

```{math}
:label: objective

J(c)=\bigl(C_{\mathrm{run}}+\lambda\,\mathbb{E}[T]\bigr)\,\mathbb{E}[R]
     +C_l\,\bigl(\mathbb{E}[R]-1\bigr)+C_s\,\mathbb{E}[S],
\qquad C_{\mathrm{run}}=\psi\,\mathbb{E}[K].
```

Every attempt costs its credit and its time, each loud failure adds {math}`C_l` for the
work of catching it, and each accepted defect costs {math}`C_s`. The expectations are
taken over the cell's beliefs about {math}`\theta_l`, {math}`S`, {math}`T` and
{math}`K`, which sections 9 and 10 construct. {math}`\psi` is 1 for pay-as-you-go
credit. It is 0 for every candidate when a candidate that receives a forecast is
unpriced or the candidates' prices have different bases; worker money then drops out of
the comparison uniformly, and the decision reports that it compared risk and time only.

The retry factor charges repeated worker cost and time. For example, a candidate that
costs 0.40 USD per attempt and fails loudly half the time costs 0.80 USD to acceptance,
and one that costs 0.65 USD and fails loudly 5% of the time costs about 0.684 USD. The
more expensive invocation is the cheaper route.

Because {math}`1/(1-\theta_l)` is convex, the expected number of attempts exceeds the
number at the expected loud rate. A belief split evenly between loud rates of 0.2 and
0.6 averages 0.4, which would charge {math}`1/0.6=1.67` attempts, while the average over
the belief charges {math}`(1.25+2.5)/2=1.875`. At the same average loud rate, a wider
belief therefore costs more.

### Notation

Every symbol below keeps one meaning throughout the report. Configuration keys, where a
constant has one, are given in code font; the [configuration guide](configuration.md)
lists them all.

| Symbol | Meaning |
| --- | --- |
| {math}`\mathcal{C}`, {math}`c`, {math}`c_i`, {math}`c^{*}` | The inventory; a cell; the cell that ran run {math}`i`; the pool cell with the lowest {math}`J` |
| {math}`f`, {math}`c_f` | A model family; its cell nearest to the core of the band |
| {math}`i`, {math}`t` | A recorded run; a ticket |
| {math}`j` | A point of the log-odds grid |
| {math}`k`, {math}`\mathcal{K}` | A forecast source (expert); the experts that take part in a decision |
| {math}`p`, {math}`\mu_p`, {math}`\mathcal{U}` | A prototype; its match weight; the matched prototypes with a usable stored forecast |
| {math}`\theta_l`, {math}`\theta_s`, {math}`u` | Loud rate; silent share of attempts; {math}`\operatorname{logit}\theta_l` |
| {math}`R`, {math}`S` | Expected attempts until a run is not loud, {math}`1/(1-\theta_l)`; accepted-defect share |
| {math}`T`, {math}`K`, {math}`A`, {math}`A_\kappa` | Active hours of one run; credit of one run; an amount (active seconds or a token count); token count of kind {math}`\kappa` |
| {math}`\kappa` | A token kind: `input`, `output`, `cache_read`, `cache_write`, `cache_write_5m` or `cache_write_1h`, or the single total |
| {math}`\lambda`, {math}`C_l`, {math}`C_s`, {math}`\varepsilon` | Price of an hour; cost of a caught failure; cost of an escaped defect; limit on {math}`S` |
| {math}`\psi`, {math}`C_{\mathrm{run}}` | Cost per credit (`cost_per_credit`); expected money cost of one run |
| {math}`J`, {math}`\tilde J`, {math}`\operatorname{sd}(J)` | Expected cost to an accepted result; its sampled value; its standard deviation |
| {math}`\operatorname{logit}`, {math}`\operatorname{logistic}` | {math}`\operatorname{logit}p=\ln\bigl(p/(1-p)\bigr)` and its inverse {math}`\operatorname{logistic}(x)=1/(1+e^{-x})` |
| {math}`\varphi` | A value's own scale: {math}`\operatorname{logit}` for a rate, {math}`\ln` for an amount |
| {math}`m`, {math}`s` | A centre and a spread on a value's own scale, with subscripts: {math}`m_k,s_k` of expert {math}`k`; {math}`m_p,s_p` of prototype {math}`p`; {math}`\bar m,s_{\mathrm{stated}}` merged; {math}`m_0,s_0` at a belief's start; {math}`m',s'` after the counted runs |
| {math}`g`, {math}`g_i`, {math}`\hat g` | Merged guess of a value; the guess stored by run {math}`i`'s decision; the corrected prediction |
| {math}`\nu_\bullet` | The weight of one forecast in a merge |
| {math}`\mu^{+}_\kappa` | Total share of the stored forecasts that state kind {math}`\kappa` above zero |
| {math}`\alpha_k`, {math}`\alpha^0_k` | Expert {math}`k`'s normalized support weight; its prior mass |
| {math}`P_k(\mathrm{out}_i)`, {math}`\hat A_{k,i}` | Probability expert {math}`k` gave run {math}`i`'s outcome; its forecast of an amount the run measured |
| {math}`\mathcal{E}_k`, {math}`\mathrm{later}_{k,i}` | The runs that scored expert {math}`k`; how many scored it after run {math}`i` |
| {math}`v_i`, {math}`\rho_i`, {math}`\mathrm{sim}_i` | Run {math}`i`'s weight on record; its reranker score against the new task; that score on a 0-to-1 scale |
| {math}`\mathrm{age}_i`, {math}`h`, {math}`a_i` | Days from the run to the request; half-life, 14 (`half_life_days`); age factor |
| {math}`w_i`, {math}`W`, {math}`W_t` | A run's evidence weight; a sum of evidence weights; the sum over ticket {math}`t` |
| {math}`\omega_i` | A run's age factor after the per-ticket limit within one statistic |
| {math}`\delta(c_i,c)` | Relatedness discount: 1, {math}`\delta_{\mathrm{effort}}=0.5`, {math}`\delta_{\mathrm{model}}=0.25` or {math}`\delta_{\mathrm{agent}}=0.25` (`effort_discount`, `model_discount`, `agent_discount`) |
| {math}`o(c)`, {math}`o_i`, {math}`o_{\mathrm{lo}}`, {math}`o_{\mathrm{hi}}` | Position of a cell in the task's order, 1 for the weakest; position of {math}`c_i`; positions of the lower and upper cut points |
| {math}`H` | Capability threshold: the lowest position strong enough for the task, from 1 to {math}`\lvert\mathcal{C}\rvert+1` |
| {math}`P_0` | The prior distribution of {math}`H` |
| {math}`P_{\mathrm{clear}}`, {math}`P_{\mathrm{timely}}` | Probability that a cell clears the threshold; that it clears it without overrunning its time |
| {math}`P_{\min}`, {math}`P_{\mathrm{rel}}`, {math}`G_{\max}` | 0.5, 0.9 and 0.05 (`min_clear_probability`, `reliability_probability`, `overpowered_gain_margin`) |
| {math}`P_{\mathrm{fit}}`, {math}`g_{\mathrm{tail}}` | Prior probability at the lower cut; prior gain above the upper cut |
| {math}`P_{\mathrm{lo}}`, {math}`G` | Lower edge and upper margin of the clearing interval |
| {math}`P_{\mathrm{bar}}(f)`, {math}`P_{\mathrm{unproven}}` | The bar a family's cell must reach for a reserved slot; that bar for an unproven family, 0.2 (`unproven_clear_probability`) |
| {math}`B`, {math}`B_{\mathrm{res}}` | Forecast budget, 12 cells (`max_forecast_cells`); reserved slots, 2 (`reserved_forecast_cells`) |
| {math}`\eta`, {math}`\xi_i`, {math}`z_i`, {math}`\ell_i` | Observation noise, 0.05 (`observation_noise`); run {math}`i`'s strength for the threshold; 1 for a clean run and 0 otherwise; its likelihood |
| {math}`d`, {math}`d_0`, {math}`d_{\mathrm{all}}` | The host's caught share; its reference value; the caught share of all hosts |
| {math}`F`, {math}`F^{\mathrm{loud}}` | Weight of failed routed runs; of the loud ones among them |
| {math}`N_{\mathrm{caught}}`, {math}`N_{\mathrm{rung}}`, {math}`N_{\mathrm{model}}`, {math}`N_{\mathrm{task}}` | Worths in runs: 12 (`caught_share_worth`), 40 (`rung_worth`), 5 (`model_worth`), and 32, 16 or 1 (`task_worth_silent`, `task_worth_loud`, `task_worth_amount`) |
| {math}`N_{\mathrm{family}}`, {math}`N_{\mathrm{worth}}`, {math}`N_{\mathrm{start}}` | Weight of the family's own runs; what the record is worth for the family; what a prediction is worth |
| {math}`r`, {math}`r^{+}`, {math}`N_r` | A rung of the ladder; the rung above; the worth of the rung above |
| {math}`\mathrm{level}`, {math}`\mathrm{usual}` | What happened in a cell's record; what its forecaster had guessed for those runs |
| {math}`y_i`, {math}`e_i` | Run {math}`i`'s recorded value on the value's scale; the miss of its prediction |
| {math}`\gamma`, {math}`\hat\gamma` | Weight on the forecaster's judgment; its raw estimate |
| {math}`\Gamma_i`, {math}`\mathcal{G}_\pm`, {math}`\Gamma_\pm`, {math}`O_\pm`, {math}`\Omega` | How far run {math}`i`'s guess sat from usual; the runs guessed above usual and the rest; each group's average {math}`\Gamma_i`; how far its results sat from their levels; the total ladder weight |
| {math}`D`, {math}`\tau`, {math}`\beta` | Typical miss of an amount's prediction, 1.3 with no record (`duration_miss`); run-to-run scatter; scatter share, 0.5 (`scatter_share`) |
| {math}`s_1`, {math}`s_{\mathrm{flat}}` | One run's spread; the flat belief's spread, 1.869 |
| {math}`x_j`, {math}`b^0_j`, {math}`b_j` | Grid point on the log-odds scale; a belief's start on the grid; the belief after the counted runs |
| {math}`n_{\mathrm{clean}}`, {math}`n_{\mathrm{loud}}`, {math}`n_{\mathrm{silent}}` | Counted weight of each outcome for a cell |
| {math}`q` | A probability level of a belief's point |
| {math}`\pi(c)`, {math}`\pi_{\mathrm{trust}}`, {math}`\pi_{\mathrm{trial}}` | Chance that {math}`S\le\varepsilon`; the trusted line, 0.80 (`admission_quantile`); the on-trial line, 0.20 (`trial_probability`) |
| {math}`\mathrm{credit\_ok}(c)` | The cell fits its available credit |
| {math}`\mathcal{P}`, {math}`\chi`, {math}`M` | The pool of competing cells; 0.5 (`all_excluded_factor`); 1 (`uncertainty_multiplier`) |
| {math}`X` | Cost of one attempt plus {math}`C_l` |
| {math}`\mathrm{price}_\kappa`, {math}`K_\kappa` | The candidate's price of kind {math}`\kappa` per million tokens; the average credit of that kind |
| {math}`s_K`, {math}`K_{80}` | Spread of the log-normal matched to {math}`K`; its 80% point |
| {math}`U_S`, {math}`U_u`, {math}`U_T`, {math}`U_K`, {math}`\Phi^{-1}` | The four uniform numbers of a cell's draw; the standard normal quantile |
| {math}`\tilde S`, {math}`\tilde R`, {math}`\tilde T`, {math}`\tilde A_\kappa`, {math}`\tilde K` | Values drawn from a cell's beliefs |
| {math}`L_{k,i}`, {math}`V_{k,i}` | Expert {math}`k`'s outcome and resource scores on run {math}`i` |
| {math}`\Delta_i`, {math}`n` | Score difference of run {math}`i` between two values of a constant; the number of scored runs |

## 4 Forecasts from the language model

This section describes the two judgments that a language model contributes to a decision
and how a host's forecasts are stored and reused.

The *capability judgment* covers the full inventory. A stateless call receives the task,
the project snapshot and the candidate identities, and it returns an order of every cell
from weakest to strongest for this task with an inclusive lower and upper cut point,
which delimit the cells it judges fit. Within one model, known efforts keep their
natural order. A memory reader can propose up to eight adjusted orders, each grounded in
a retrieved memory or execution, and each such proposal is a separate expert.

The *fine forecast* covers only the cells that the capability stage keeps, its forecast
cells. For each it states the probabilities of `clean`, `loud` and `silent`, the active
seconds and the total tokens, and for a cell priced by token kind the six token counts.
A forecast may add a central 95% range to its loud and silent probabilities, its active
time and its token kinds. An independent stateless call produces the baseline row, and the memory
reader may override rows for the cells where a retrieved memory or execution changes
its expectation. The independent calls receive no memories, recorded runs or learned
tables, so the baseline remains a reference against which experience can be scored. An
unavailable or invalid independent forecast stops routing with an error.

Each value has its own scale {math}`\varphi`. A rate is read on the log-odds scale,
held in {math}`[0.0001,0.9999]`. An amount is read on the log scale, held at 1 or more.
The loud rate of a row is its `loud` probability and its accepted-defect share is
{math}`\text{silent}/(\text{clean}+\text{silent})`, after the three probabilities are
floored at 0.0001 and normalized. A stated 95% range {math}`[\mathrm{lo},\mathrm{hi}]`
gives the spread

```{math}
:label: spread

s=\frac{\varphi(\mathrm{hi})-\varphi(\mathrm{lo})}{3.92},
```

because a central 95% range spans 3.92 standard deviations. The range on {math}`S` is
the `silent` range divided by {math}`\text{clean}+\text{silent}`. A value with no range
has spread 0.

Several forecasts of one value, the forecast {math}`\bullet` with centre
{math}`m_\bullet`, spread {math}`s_\bullet` and weight {math}`\nu_\bullet` where the
weights sum to one, are merged on the value's scale into one centre and one spread:

```{math}
:label: merge

\bar m=\sum_\bullet \nu_\bullet\,m_\bullet,\qquad
s_{\mathrm{stated}}^2=\sum_\bullet \nu_\bullet\bigl(s_\bullet^2+(m_\bullet-\bar m)^2\bigr).
```

These are the mean and variance of the mixture of the forecasts, so forecasts that
disagree widen the merged spread and equal forecasts keep it. Section 8 merges the
experts' forecasts with this rule, and the stored forecasts below are merged with it too.

### Stored forecasts

A new fine forecast is stored on the task's primary prototype, the matched prototype
with the largest match weight, under the name of the host that asked. A prototype keeps
one stored forecast per cell and host, and a new one replaces the same host's earlier
forecast whole. The level of a forecast follows the host that made it, so a stored
forecast is reused only for that host; a request without a host name belongs to one
unnamed host. Capability priors are shared by all hosts.

A stored forecast is usable for a cell of a new task when its prototype matches the
task, it was made under the task's host, and it states every value the cell needs. Let
{math}`\mathcal{U}` be the matched prototypes with a usable forecast for the cell. Each
value is merged by {eq}`merge` over {math}`\mathcal{U}` with the shares
{math}`\mu_p/\sum_{p'\in\mathcal{U}}\mu_{p'}`, from the stored centre {math}`m_p` and the
stored spread {math}`s_p` of {eq}`spread`. A token kind that every forecast in
{math}`\mathcal{U}` states as exactly zero stays zero; any other kind is merged over the
forecasts that state it above zero and multiplied by their total share
{math}`\mu^{+}_\kappa`. A cell priced by kind sums its kinds into `tokens`. A single
usable forecast is reused unchanged. The merged row is

```{math}
:label: merged-row

\begin{aligned}
\text{loud}&=\operatorname{logistic}(\bar m_{\mathrm{loud}}), &
S&=\operatorname{logistic}(\bar m_S),\\
\text{clean}&=(1-\text{loud})(1-S), &
\text{silent}&=(1-\text{loud})\,S,\\
\text{seconds}&=\exp(\bar m_{\mathrm{time}}), &
A_\kappa&=\mu^{+}_\kappa\exp(\bar m_\kappa),
\end{aligned}
```

with {math}`\operatorname{logistic}(x)=1/(1+e^{-x})`, and each value with a positive
spread states the range {math}`\varphi^{-1}(\bar m\pm1.96\,s_{\mathrm{stated}})`, the
silent range being the range of {math}`S` times {math}`1-\text{loud}` and a kind's range
being multiplied by {math}`\mu^{+}_\kappa`. The row stands in place of the forecaster's
answer, and the beliefs read its centres and spreads back exactly as from a forecaster's
row.

For example, two prototypes with shares 0.7 and 0.3 store loud rates of 20% and 40%,
each with spread 0.5. The centres are {math}`\operatorname{logit}0.2=-1.386` and
{math}`\operatorname{logit}0.4=-0.405`, so {math}`\bar m=-1.092`, a loud rate of 25.1%,
and {math}`s_{\mathrm{stated}}^2=0.7(0.25+0.294^2)+0.3(0.25+0.687^2)=0.452`, a spread
of 0.672 and a range of {math}`[0.082,0.556]`. With a merged {math}`S` of 10% the row is
clean 0.674, loud 0.251 and silent 0.075.

A cell with no usable stored forecast goes to the forecaster, and a task whose cells
are all covered, with no active memory to consult, is routed without any model call. In
the running example the project is new, so both stages call the forecaster, and the
three rows are stored for the host. The {ref}`stored-forecast reference
<stored-forecasts>` gives the stored fields and the receipt's record of each reused cell.

## 5 Expert support

This section weights the forecast sources by how well their earlier forecasts predicted
what then happened.

Every forecast source is an *expert*. The independent forecast is the expert `llm`, a
written lesson in memory is the expert `memory:ID:VERSION`, and a recorded execution
shown to the reader is the read-only expert `execution:RUN_ID`. The capability stage has
its own experts, `filter:llm` and `filter:memory:ID:VERSION`. An expert earns a score
only from a run whose receipt holds its forecast for the cell that actually ran, made
before the run. A memory earns it only when it supplied an explicit changed forecast for
that cell; a memory that was not retrieved, or that left the baseline unchanged, earns
nothing. A lesson written after a failure can guide later work, but the failure that
prompted it cannot score it, and a revised lesson starts a new version without its
predecessor's score.

On run {math}`i`, a fine expert {math}`k` scores the floored log ratio of the
probability it gave to the actual outcome against the baseline's,

```{math}
:label: outcome-score

L_{k,i}=\ln\max\bigl(0.0001,P_k(\mathrm{out}_i)\bigr)-\ln\max\bigl(0.0001,P_{\mathrm{llm}}(\mathrm{out}_i)\bigr),
```

and, for each of the run's measured tokens and active seconds {math}`A_i`, a resource
score {math}`V_{k,i}` that sums
{math}`\bigl[(\ln A_i-\ln\hat A_{\mathrm{llm},i})^2-(\ln A_i-\ln\hat A_{k,i})^2\bigr]/(2\cdot1.5^2)`,
where {math}`\hat A` is the forecast and every amount is held at 1 or more. Active time
is the run's active seconds when recorded and its wall-clock seconds otherwise; an
unmeasured amount adds nothing. The support of expert {math}`k` is

```{math}
:label: support

\ln\tilde\alpha_k=\ln\alpha^0_k+0.5\sum_{i\in\mathcal{E}_k}0.98^{\,\mathrm{later}_{k,i}}\;v_i\,\bigl(L_{k,i}+0.25\,V_{k,i}\bigr),
\qquad
\alpha_k=\frac{\tilde\alpha_k}{\sum_{k'\in\mathcal{K}}\tilde\alpha_{k'}}.
```

Here {math}`\mathcal{E}_k` is the set of runs that scored expert {math}`k`,
{math}`\mathrm{later}_{k,i}` counts the runs that scored {math}`k` after run
{math}`i`, {math}`v_i` is the run's weight on record (the weight configured for its
model, 1 unless set) and {math}`\alpha^0_k` is 1 for the independent forecast and 0.25
for a memory. The factor 0.5 tempers the evidence and 0.98 lets older exposures fade.
The weights are normalized over the experts {math}`\mathcal{K}` that take part in the
decision, so growth in unrelated memories does not dilute them. These constants ship in
the packaged `semantic.toml`.

A capability expert scores, on the cell that ran, the log ratio of its
{math}`P_{\mathrm{clear}}` against the independent fine forecast's `clean` probability
for a clean run, and of {math}`1-P_{\mathrm{clear}}` against {math}`1-\text{clean}`
otherwise, floored as in {eq}`outcome-score`, with no resource term. An amendment
replaces the run's outcome, and its forecasts are scored once against the effective
outcome.

After each observation, a language-model writer can add, edit, remove, restore, merge
or split written lessons. Every change keeps history and is refused against a stale
revision, and a failed writer leaves the observation intact for a retry. Learning is
shared across models inside the project's store: a lesson may inform a forecast for
another model, while its source outcome stays attributed to the run that produced it.
The [semantic-memory contract](semantic-memory.md) specifies the writer and its
operations, and the [retrieval guide](retrieval.md) specifies how lessons and executions
are retrieved.

## 6 Recorded runs as evidence

This section chooses the recorded runs that bear on a new task and gives each a weight,
which three later readers use.

A run is a *candidate* for a new task when its outcome is `clean`, `loud` or `silent` and
it belongs to at least one prototype that the task matches. The prototypes only choose
the candidates; their match weights enter no run weight. A task that matches no
prototype has no candidates. At most the 256 most recent candidates are scored
(`comparison_cap`), each compared with the new task once, when the request is prepared.
The cross-encoder scores the new task's text against run {math}`i`'s own task text,
giving {math}`\rho_i`, and run {math}`i` weighs

```{math}
:label: evidence-weight

\mathrm{sim}_i=\operatorname{logistic}(\rho_i),\qquad
a_i=0.5^{\,\mathrm{age}_i/h},\qquad
w_i=v_i\,\mathrm{sim}_i\,a_i\times(\text{the reader's factor}),
```

where {math}`\mathrm{age}_i`, never below zero, counts the days from the run to the
request. A run counts more when its task reads like the new one and half as much for
every {math}`h` days of age. There is no cut-off, so a run with a negative score still
counts at a small weight. Retries of one ticket follow from one another's failures, so
the runs of one ticket together count for at most one run:

```{math}
:label: ticket-limit

w_i\leftarrow\frac{w_i}{\max(1,W_{t})},\qquad W_t=\sum_{i'\in t}w_{i'},
```

where the sum runs over the reader's counted runs of ticket {math}`t`. A run with no
ticket is not limited. For example, a run of the judged cell with {math}`\rho=1.386`,
recorded 7 days before the request, has {math}`\mathrm{sim}=0.8`,
{math}`a=0.5^{7/14}=0.7071` and {math}`w=0.8\times0.7071=0.5657`.

The three readers differ in their factor. The capability threshold (section 7) uses the
run's strength {math}`\xi_i` and counts every candidate whose cell is in the inventory.
The outcome beliefs (section 9) count every candidate for every forecast cell
{math}`c`, whatever cell ran it, at the relatedness discount

```{math}
:label: discount

\delta(c_i,c)=
\begin{cases}
1&\text{the same family at the same effort (the same cell)},\\
\delta_{\mathrm{effort}}&\text{the same family at another effort},\\
\delta_{\mathrm{model}}&\text{another family of the same agent},\\
\delta_{\mathrm{agent}}&\text{another agent},
\end{cases}
```

where the first line that fits applies, and the limit {eq}`ticket-limit` is taken
separately for each judged cell. A cell's agent is the label on its candidate, and a
run's agent is the label its receipt stores for the cell that ran; a run without one
takes the agent an operator imported for its family, and otherwise belongs, like an
unlabeled candidate, to one unnamed agent. The beliefs for active time and tokens
(section 10) count only the candidates of the same cell, with factor 1.

For example, let the judged cell be family `f1` at effort `high` under agent `a1`, with
every {math}`v_i=1`:

| Run | Ran on | Relation | {math}`\mathrm{sim}` | {math}`a` | {math}`\delta` | Weight | Ticket | After the limit | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `f1` `high`, `a1` | Same cell | 0.8 | 1 | 1 | 0.80 | `t1` | 0.64 | clean |
| 2 | `f1` `low`, `a1` | Same family | 0.9 | 1 | 0.5 | 0.45 | `t1` | 0.36 | loud |
| 3 | `f2` `high`, `a1` | Same agent | 0.8 | 0.5 | 0.25 | 0.10 | `t2` | 0.10 | clean |
| 4 | `f3` `high`, `a2` | Another agent | 0.4 | 1 | 0.25 | 0.10 | `t3` | 0.10 | silent |

Ticket `t1` carries {math}`0.80+0.45=1.25`, so both of its runs are divided by 1.25, and
the cell counts {math}`n_{\mathrm{clean}}=0.74`, {math}`n_{\mathrm{loud}}=0.36` and
{math}`n_{\mathrm{silent}}=0.10`. With all three discounts at 0, only run 1 counts, as
0.80 clean.

Two statistics of later sections read finished routed runs on other tasks instead of
the candidates: the host's caught share (section 7) and the corrected prediction
(section 8). Each weighs its runs by their age factor alone, limited per ticket over the
runs of that statistic:

```{math}
:label: ladder-weight

\omega_i=\frac{a_i}{\max\bigl(1,\sum_{i'\in t}a_{i'}\bigr)}.
```

In the running example the record is empty, so no reader counts any run. The
[retrieval guide](retrieval.md) gives the candidate rules in full.

## 7 Capability threshold and the fit band

This section turns the capability judgments and the weighted candidates into the
probability that each cell is strong enough, and it chooses the cells that receive fine
forecasts.

### The threshold prior

We model the capability a task requires as a latent position {math}`H` in the task's
order: the cells at positions {math}`o\ge H` are strong enough and the weaker ones are
not. {math}`H` can take the value {math}`\lvert\mathcal{C}\rvert+1`, which says that no
supplied cell is strong enough. One expert's cut points define a smoothed prior
distribution function over {math}`H`:

```{math}
:label: prior-cdf

P_0(H\le o)=
\begin{cases}
0.8\,P_{\min}\,2^{-(o_{\mathrm{lo}}-1-o)} & o<o_{\mathrm{lo}},\\[2pt]
P_{\mathrm{fit}}+\bigl(P_{\mathrm{rel}}-P_{\mathrm{fit}}\bigr)\dfrac{o-o_{\mathrm{lo}}}{o_{\mathrm{hi}}-o_{\mathrm{lo}}} & o_{\mathrm{lo}}\le o\le o_{\mathrm{hi}},\\[6pt]
P_{\mathrm{rel}}+g_{\mathrm{tail}}\dfrac{1-2^{-(o-o_{\mathrm{hi}})}}{1-2^{-(\lvert\mathcal{C}\rvert-o_{\mathrm{hi}})}} & o>o_{\mathrm{hi}},
\end{cases}
```

with {math}`P_{\mathrm{fit}}=\max(P_{\min},P_{\mathrm{rel}}-0.2)`,
{math}`g_{\mathrm{tail}}=\min\bigl((1-P_{\mathrm{rel}})/2,\,G_{\max}/2\bigr)`, and the
fit cell at {math}`P_{\mathrm{rel}}` when the two cuts coincide. Below the lower cut each
step toward weaker cells halves the probability, so a long list of weak cells does not
spread mass across them. Above the upper cut the remaining gain stays below
{math}`G_{\max}` and is concentrated near the cut. In the running example, with A, B
and C at positions 1 to 3 and the cuts at B and C, the prior gives
{math}`P_0(H\le1)=0.4`, {math}`P_0(H\le2)=0.7` and {math}`P_0(H\le3)=0.9`.

The experts' orders are combined into one consensus order by their support-weighted
positions, keeping each model's effort chain in the order the independent judgment gave
it. Each cell's prior probability is the support-weighted average of the experts'
{eq}`prior-cdf` at that cell, and the pool-adjacent-violators algorithm projects these
averages onto a distribution function that is monotone along the consensus order; we
write {math}`P_0` for the result. When matched prototypes already store priors for some
cells, the stored probabilities replace the independent judgment's values for those
cells before the projection, and when they cover every cell, the stored priors stand in
for the independent judgment.

### The weighted likelihood update

A candidate run {math}`i` contributes to the threshold when its cell is in the inventory.
At position {math}`o_i`, a clean outcome is likely when the threshold lies at or below
the run's position and unlikely otherwise, up to the observation noise {math}`\eta`.
For a threshold at position {math}`o`, the probability of a clean outcome is

```{math}
:label: likelihood

\ell_i(o)=
\begin{cases}
1-\eta & o\le o_i,\\
\eta & o>o_i.
\end{cases}
```

With {math}`z_i=1` for a clean run and 0 for a loud or silent one, each likelihood
{eq}`likelihood` is raised to the run's weight:

```{math}
:label: posterior

P(H=o\mid\text{runs})\;\propto\;P_0(H=o)\prod_i
\Bigl[\ell_i(o)^{z_i}\bigl(1-\ell_i(o)\bigr)^{1-z_i}\Bigr]^{w_i},
\qquad
P_{\mathrm{clear}}(c)=P\bigl(H\le o(c)\mid\text{runs}\bigr).
```

The weight is {eq}`evidence-weight` with the strength {math}`\xi_i` as factor, limited by
{eq}`ticket-limit` over this reader's runs. A clean run has strength 1. A loud failure
has 0.75 and a silent one 1 when its recorded cause is `capability`, and half of that
when the cause is `depth` or `unrecorded`; an explicit `capability_caused=false` in the
note of such a failure sets its strength to 0. For example, a loud run with no recorded
cause, {math}`\mathrm{sim}=0.8`, recorded 7 days before the request, weighs
{math}`0.8\times0.75\times0.5\times0.7071=0.2121`.

The order itself can change. A local search swaps the positions of observed cells of
different models, keeping every effort chain, when the swap raises the marginal
likelihood of the runs net of a penalty of 0.15 times the normalized distance from the
initial order. The prior's probabilities stay with the positions, so a cell that moves
takes the prior of its new position. A cell that has never run keeps its prior
placement, and runs at other positions can still change its
{math}`P_{\mathrm{clear}}` through the shared threshold.

### The host's caught share

How much weakness the band can afford depends on the host. A failure the host finds
before acceptance is loud and one found afterwards is silent, so the share of its
failures that a host catches decides how many weak results escape. We estimate this
*caught share* from the failed routed runs on record, those that ended loud or silent,
on any task, with weights {eq}`ladder-weight`. Let {math}`F` be their total weight and
{math}`F^{\mathrm{loud}}` that of the loud ones, over all hosts or over the asking host.
A run's host is the name its receipt stores, or the name an operator imported for an
older receipt, and otherwise the unnamed host. The estimate is shrunk toward all hosts
and toward a reference share:

```{math}
:label: caught-share

d_0=\frac{1-2\varepsilon}{1-\varepsilon},\qquad
d_{\mathrm{all}}=\frac{F^{\mathrm{loud}}_{\mathrm{all}}+N_{\mathrm{caught}}\,d_0}{F_{\mathrm{all}}+N_{\mathrm{caught}}},\qquad
d=\frac{F^{\mathrm{loud}}_{\mathrm{host}}+N_{\mathrm{caught}}\,d_{\mathrm{all}}}{F_{\mathrm{host}}+N_{\mathrm{caught}}}.
```

A host with few failures of its own leans on all hosts, and an empty record gives
{math}`d_0`. At {math}`\varepsilon=0.10`, {math}`d_0=0.889`. With
{math}`F_{\mathrm{all}}=60` and {math}`F^{\mathrm{loud}}_{\mathrm{all}}=42`,
{math}`d_{\mathrm{all}}=(42+12\times0.889)/72=0.731`, and a host with
{math}`F_{\mathrm{host}}=20` and {math}`F^{\mathrm{loud}}_{\mathrm{host}}=8` has
{math}`d=(8+12\times0.731)/32=0.524`.

### The fit band

The *clearing interval* starts at the first cell whose {math}`P_{\mathrm{clear}}`
reaches the lower edge {math}`P_{\mathrm{lo}}`. Its anchor is the first cell from there
that is *reliable*, with {math}`P_{\mathrm{clear}}\ge P_{\mathrm{rel}}`, or the start
when no cell is. It ends at the first cell from the anchor whose remaining gain to the
strongest cell, {math}`P_{\mathrm{clear}}(\text{strongest})-P_{\mathrm{clear}}(c)`, is
below the upper margin {math}`G`. Both edges follow the caught share:

```{math}
:label: band-edges

\bigl(P_{\mathrm{lo}},\,G\bigr)=
\begin{cases}
\bigl(P_{\min},\,G_{\max}\bigr) & d\ge d_0,\\[4pt]
\Bigl(\max\Bigl(P_{\min},\,1-\dfrac{\varepsilon}{1-d+\varepsilon d}\Bigr),\;
\min\Bigl(G_{\max},\,\dfrac{G_{\max}\,\varepsilon}{(1-\varepsilon)(1-d)}\Bigr)\Bigr) & d<d_0.
\end{cases}
```

A cell that is too weak fails with probability {math}`1-P_{\mathrm{clear}}`. A share
{math}`(1-P_{\mathrm{clear}})(1-d)` of its results escapes the host and a share
{math}`1-(1-P_{\mathrm{clear}})d` is accepted, and holding the escaped share of the
accepted results at {math}`\varepsilon` gives the lower edge. At the top, 0.05 more
{math}`P_{\mathrm{clear}}` removes about {math}`(1-d)\times0.05` of the silent results,
so {math}`G` shrinks to keep {math}`(1-d)\,G` constant. At {math}`\varepsilon=0.10`:

| {math}`d` | {math}`P_{\mathrm{lo}}` | {math}`G` |
| --- | --- | --- |
| 0.889 or more | 0.500 | 0.0500 |
| 0.80 | 0.643 | 0.0278 |
| 0.524 | 0.811 | 0.0117 |
| 0.50 | 0.818 | 0.0111 |
| 0.00 | 0.900 | 0.0056 |

With {math}`P_{\mathrm{clear}}` of 0.30, 0.55, 0.72, 0.84, 0.91, 0.955, 0.985 and 0.99
along the order, the configured edges give cells 2 to 6 and the edges at
{math}`d=0.524` give cells 4 to 7. If no cell reaches {math}`P_{\mathrm{lo}}`, the band
starts at the first cell with the highest {math}`P_{\mathrm{clear}}` when that value
reaches {math}`P_{\min}`, and the request is refused otherwise. With no counted run,
the independent judgment as the only expert and {math}`d\ge d_0`, the band is exactly
that judgment's cut points.

A second interval keeps faster stronger cells within reach. The *timely* probability
{math}`P_{\mathrm{timely}}` comes from {eq}`posterior` with {math}`z_i` replaced by a
timely strength: 0 for a failure, {math}`(1.5/\text{overrun})^2` for a clean run
that took more than 1.5 times its reference time, and 1 for any other clean run. The
reference is the smaller of the run's own forecast and the weighted geometric mean time
of the clean runs of stronger cells, and the overrun is the run's time divided by it.
The timely interval uses the configured {math}`P_{\min}`, {math}`P_{\mathrm{rel}}` and
{math}`G_{\max}` from the clearing interval's start, and when no cell reaches
{math}`P_{\min}` it keeps only the first near-best timely cell. The band runs from the
clearing interval's start to the later of the two ends. When that span exceeds the
forecast budget and a gap separates the two intervals, the band is their union, and the
cells of the gap keep their probabilities as intermediate cells. A band that still
exceeds the budget prompts one request for a sharper judgment over the same inventory,
and a second oversized band is refused.

### Forecast cells and reserved slots

The band alone can leave out a family that has never run this kind of task, or every
cell of one agent. Of the {math}`B` forecast cells, {math}`B_{\mathrm{res}}` are
reserved:

```{math}
:label: forecast-cells

\text{core}=\text{the first }B-B_{\mathrm{res}}\text{ cells of the band},\qquad
\text{forecast}=\text{core}\;\cup\;\{\text{up to }B_{\mathrm{res}}\text{ cells }c_f\text{ with }P_{\mathrm{clear}}(c_f)\ge P_{\mathrm{bar}}(f)\}.
```

Here {math}`f` is a family with no core cell and {math}`c_f` its cell nearest to the
core in the order, the stronger one on a tie. A family is *unproven* when none of its
runs is among the decision's candidates. Unproven families take slots first, at the bar
{math}`P_{\mathrm{bar}}(f)=P_{\mathrm{unproven}}`, each on its own and nearest first.
The remaining slots go to agents with no core cell, at most one slot per agent: a
proven family takes a slot on its agent's behalf at the bar
{math}`P_{\mathrm{bar}}(f)=P_{\mathrm{lo}}`, and an agent whose unproven family already
took a slot counts as served. A slot that no family takes goes back to the band, whose
next cell beyond the core is forecast. A reserved cell keeps the classification of its
position, and no decision forecasts more than {math}`B` cells.

In the running example the record is empty, so {math}`P_{\mathrm{clear}}` equals the
prior, {math}`d=d_0` and the band is B and C, which also form the core. A's family has
no core cell and no candidate run, and its {math}`P_{\mathrm{clear}}` of 0.4 reaches the
bar of 0.2, so A takes a reserved slot and is forecast as an underpowered cell. The
[capability contract](capability.md) defines the band, the widening rounds and the
classifications in full.

## 8 Corrected predictions

This section merges the experts' fine forecasts of each value and corrects the merged
guess with the record of the same host on other tasks.

For each forecast cell of {eq}`forecast-cells` and each value (the loud rate, the
accepted-defect share, active time and each token count), the experts' forecasts are
merged by {eq}`merge` with the support weights {math}`\alpha_k` of {eq}`support`. A row
reused from stored forecasts by {eq}`merged-row` enters as the independent expert's row.
The merged *guess* is {math}`g=\varphi^{-1}(\bar m)`, held in {math}`[0.0001,0.9998]`
for a rate. For example, two experts with weights 0.75 and 0.25 guess {math}`S` at 10%
and 25%, each with spread 0.4. Then
{math}`\bar m=0.75\times(-2.197)+0.25\times(-1.099)=-1.92`, a guess of 12.8%, and
{math}`s_{\mathrm{stated}}=\sqrt{0.16+0.75\times0.275^2+0.25\times0.824^2}=0.62`.

The level of a forecast follows the agent that makes it, and outcomes depend on the
agent that accepts the result. We therefore let the record set the level of every value
and let the forecast say only whether this task looks higher or lower than its
forecaster usually states:

```{math}
:label: correction

\varphi(\hat g)=\varphi(\mathrm{level})+\gamma\,\bigl(\varphi(g)-\varphi(\mathrm{usual})\bigr),
```

where *level* is what happened in the recorded runs of this cell under this host,
*usual* is what the forecaster had guessed for those runs, {math}`\gamma` is the weight
on the forecaster's judgment, estimated by {eq}`forecaster-weight` below, and
{math}`\hat g` is the *corrected prediction* from which the cell's belief starts. A
rate's prediction is held in {math}`[0.0001,0.9998]`. An empty record gives the guess,
{math}`\gamma=0` gives the level alone and {math}`\gamma=1` moves the guess by the
difference between level and usual. For example, level 0.0566, usual 0.188, guess 0.10
and {math}`\gamma=0.74` give {math}`-2.81+0.74\times(-2.20+1.46)=-3.36` in log-odds, a
prediction of 0.034.

### The ladder

A cell has few runs of its own, so level and usual come down a *ladder* of five rungs,
from all hosts through this host, this agent under it and this family under it to this
cell, each rung leaning on the one above:

```{math}
:label: ladder

\mathrm{level}(r)=\frac{\sum_{i\in r}\omega_i\,y_i+N_r\,\mathrm{level}(r^{+})}{\sum_{i\in r}\omega_i+N_r},\qquad
\mathrm{usual}(r)=\frac{\sum_{i\in r}\omega_i\,g_i+N_r\,\mathrm{usual}(r^{+})}{\sum_{i\in r}\omega_i+N_r}.
```

The runs are the finished routed runs with outcome clean, loud or silent, weighted by
{eq}`ladder-weight` over the runs of the ladder. A run's host and agent are the names its
receipt stores, or the names imported for an older receipt. The runs of every prototype
the new task matches are left out, because they already count as candidates (section
6). {math}`y_i` is what happened: for a rate, 1 for a failure and 0 otherwise;
for an amount, the logarithm of the recorded amount. {math}`g_i` is the merged guess
that run's own decision stored, on the same footing. The accepted-defect share counts
accepted runs with silent as the failure, the loud rate counts every finished run with
loud as the failure, and an amount counts the runs whose recorded amount and guess are
both above zero. {math}`N_r` is the worth of the rung above: {math}`N_{\mathrm{model}}=5`
at the family rung and {math}`N_{\mathrm{rung}}=40` elsewhere. Above all hosts, both
ladders equal the guess, and a rung with no run passes the value above through.

Suppose B's cell had this record for {math}`S`, against its guess of 0.10:

| Rung | {math}`\sum\omega` | {math}`\sum\omega\,y` | {math}`\sum\omega\,g` | Level | Usual |
| --- | --- | --- | --- | --- | --- |
| All hosts | 160 | 24 | 16 | 0.140 | 0.100 |
| This host | 60 | 6 | 9 | 0.116 | 0.130 |
| This agent | 40 | 2 | 6 | 0.083 | 0.140 |
| This family | 15 | 1 | 3 | 0.071 | 0.185 |
| This cell | 10 | 0 | 2 | 0.057 | 0.188 |

The all-hosts level is {math}`(24+40\times0.10)/(160+40)=0.140` and the family level
{math}`(1+5\times0.083)/(15+5)=0.071`. With {math}`\gamma=0.74` these rungs give the
prediction 0.034 of the example above.

### The weight on the forecaster

The weight {math}`\gamma` measures how much of the forecaster's stated difference from
its usual level shows up in what happens. It is estimated once per kind of value, from
the runs of all hosts. Let {math}`\Gamma_i=\varphi(g_i)-\varphi(\mathrm{usual}_i)`,
where {math}`\mathrm{level}_i` and {math}`\mathrm{usual}_i` are both ladders at run
{math}`i`'s own cell over all the runs by {eq}`ladder`, with the record's weighted
averages above all hosts. The runs with {math}`\Gamma_i>0` form the group
{math}`\mathcal{G}_+` and the rest {math}`\mathcal{G}_-`. With {math}`\Gamma_\pm` each
group's weighted average of {math}`\Gamma_i` and {math}`O_\pm` how far its results sat
from their levels,

```{math}
:label: forecaster-weight

\hat\gamma=\frac{O_+-O_-}{\Gamma_+-\Gamma_-}\ \text{held in}\ [0,1],\qquad
\gamma=\frac{\Omega\,\hat\gamma+N_{\mathrm{rung}}}{\Omega+N_{\mathrm{rung}}},
```

where {math}`\Omega=\sum_i\omega_i`. For a rate,
{math}`O=\operatorname{logit}\frac{\sum\omega_i y_i+1/2}{\sum\omega_i+1}-\operatorname{logit}\frac{\sum\omega_i\,\mathrm{level}_i+1/2}{\sum\omega_i+1}`
over the group; for an amount, it is the group's weighted average of
{math}`y_i-\mathrm{level}_i`. {math}`\hat\gamma` is 1 when a group is empty or
{math}`\Gamma_+-\Gamma_-<0.05`, so {math}`\gamma` starts at 1. For example, if
{math}`\mathcal{G}_+` holds 60 runs' worth with 13 silent and levels summing to 9, and
{math}`\mathcal{G}_-` holds 100 with 11 silent and levels summing to 15, then
{math}`O_+=\operatorname{logit}(13.5/61)-\operatorname{logit}(9.5/61)=0.43` and
{math}`O_-=-0.34`. With {math}`\Gamma_+-\Gamma_-=1.15`, {math}`\hat\gamma=0.675` and
{math}`\gamma=(160\times0.675+40)/200=0.74`.

### The remaining miss of an amount

For an amount, the same ladder over squared misses gives {math}`D`, the typical miss
left after the correction:

```{math}
:label: miss

D^2(r)=\frac{\sum_{i\in r}\omega_i\,e_i^2+N_r\,D^2(r^{+})}{\sum_{i\in r}\omega_i+N_r},\qquad
e_i=y_i-\varphi(\hat g_i),\qquad D=\sqrt{D^2(\text{this cell})},
```

where {math}`\hat g_i` is the prediction {eq}`correction` for past run {math}`i`, with
its own level and usual and the {math}`\gamma` of its kind of value, and the default
{math}`D^2=1.3^2` above all hosts. For example, 100 runs' worth on one cell, each off by
a factor of {math}`e^{0.5}`, give {math}`D^2` of 0.661, 0.368, 0.284, 0.252 and 0.250
down the rungs, so {math}`D=0.50`. Part of the miss is the scatter between runs of one
task, which no prediction can remove: the *run-to-run scatter* of an amount is
{math}`\tau=D\sqrt{\beta}`, where {math}`\beta` is the share of {math}`D^2` that is
scatter. Total tokens and each token kind have their own ladders, {math}`\gamma` and
{math}`D`.

### How firmly the prediction is held

The family's own runs set how many runs the prediction is worth. With
{math}`N_{\mathrm{family}}` the total {math}`\omega` at the family rung,

```{math}
:label: worth

N_{\mathrm{worth}}=N_{\mathrm{family}}+N_{\mathrm{model}},\qquad
N_{\mathrm{start}}=\frac{1}{1/N_{\mathrm{worth}}+1/N_{\mathrm{task}}},\qquad
s_0=\max\Bigl(s_{\mathrm{stated}},\frac{s_1}{\sqrt{N_{\mathrm{start}}}}\Bigr),
```

where {math}`N_{\mathrm{task}}`, the most a record on other tasks is worth for one task,
is 32 for {math}`S`, 16 for {math}`\theta_l` and 1 for an amount. The spread of one
run's result is {math}`s_1=1/\sqrt{\hat g(1-\hat g)}` for a rate and the run-to-run
scatter {math}`\tau=D\sqrt{\beta}` of {eq}`miss` for an amount. For a rate, {math}`s_0`
is at most the flat belief's {math}`s_{\mathrm{flat}}=1.869`, the spread of a 95% range
of {math}`[0.025,0.975]`. A stated range can widen the start and never narrow it. With
an empty record {math}`N_{\mathrm{worth}}=5`, so {math}`N_{\mathrm{start}}` is 4.32 for
{math}`S`, 3.81 for {math}`\theta_l` and 0.833 for an amount. In the example above,
{math}`N_{\mathrm{worth}}=15+5=20` gives {math}`N_{\mathrm{start}}=1/(1/20+1/32)=12.3`;
with {math}`\hat g=0.034`, {math}`s_1=5.54` and
{math}`s_1/\sqrt{N_{\mathrm{start}}}=1.58`, which exceeds B's stated spread of 0.40, so
{math}`s_0=1.58`.

In the running example the record is empty, so every prediction is the merged guess of
the single expert: for B, {math}`\hat g=0.10` for {math}`S` and 0.20 for
{math}`\theta_l`. The {ref}`corrected-prediction reference <corrected-prediction>` lists
the sums a receipt stores and how loading a receipt recomputes them.

## 9 Outcome beliefs and admission

This section builds each cell's beliefs about its loud rate and its accepted-defect
share, and it sorts the cells by their chance of meeting the defect limit.

### Beliefs on a log-odds grid

A cell holds two independent beliefs, one for {math}`\theta_l` and one for {math}`S`.
Each is a normal distribution on the log-odds scale, held on one fixed grid:

```{math}
:label: grid

x_j=-16+0.01\,j,\quad j=0,\dots,2500,\qquad
b^0_j\propto\exp\Bigl(-\frac{(x_j-m_0)^2}{2s_0^2}\Bigr),\qquad
\sum_j b^0_j\operatorname{logistic}(x_j)=\hat g.
```

The start {math}`b^0` has the spread {math}`s_0` of {eq}`worth`, and its centre
{math}`m_0` is chosen so that its average, rather than its median, equals the
prediction; it is found by 60 halvings of {math}`[-25,25]`. The grid's top rate,
{math}`\operatorname{logistic}(9)=0.99988`, lets every prediction up to 0.9998 be met,
and one fixed grid makes every belief a finite sum that a stored decision recomputes
exactly. For B, {math}`\hat g=0.10` for {math}`S`. B's stated range of 5% to 20% on
{math}`S` gives {math}`s_{\mathrm{stated}}=0.40`, while
{math}`s_1/\sqrt{N_{\mathrm{start}}}=3.33/2.08=1.60`, so {math}`s_0=1.60` and
{math}`m_0=-3.04`. For {math}`\theta_l`, {math}`\hat g=0.20` and the range 0.10 to 0.35
give {math}`s_0=1.28` and {math}`m_0=-1.79`.

### Counted runs

Every candidate run counts for every forecast cell {math}`c` at the weight
{eq}`evidence-weight` with the discount {eq}`discount` as factor, limited per ticket for
that cell. The counted weights of the three outcomes, {math}`n_{\mathrm{clean}}`,
{math}`n_{\mathrm{loud}}` and {math}`n_{\mathrm{silent}}`, then reshape the start
{math}`b^0` of {eq}`grid`:

```{math}
:label: belief-update

b_j\propto b^0_j\,\operatorname{logistic}(x_j)^{n_{\mathrm{yes}}}\bigl(1-\operatorname{logistic}(x_j)\bigr)^{n_{\mathrm{no}}},\qquad
\begin{cases}
S: & n_{\mathrm{yes}}=n_{\mathrm{silent}},\ n_{\mathrm{no}}=n_{\mathrm{clean}},\\
\theta_l: & n_{\mathrm{yes}}=n_{\mathrm{loud}},\ n_{\mathrm{no}}=n_{\mathrm{clean}}+n_{\mathrm{silent}}.
\end{cases}
```

The start counts as {math}`N_{\mathrm{start}}` runs and each counted run as its weight,
so the forecast keeps its weight however many runs are counted. Because each run's
weight halves every {math}`h` days, a regression or a recovery moves the belief as older
runs fade. A cell with no counted run keeps its start, and an expert's global support
never stands in for a cell's counted runs.

Selection reads the belief's moments and points, each a sum over the grid:

```{math}
:label: belief-summaries

\begin{aligned}
\mathbb{E}[S]&=\sum_j b_j\operatorname{logistic}(x_j), &
\operatorname{Var}(S)&=\sum_j b_j\bigl(\operatorname{logistic}(x_j)-\mathbb{E}[S]\bigr)^2,\\
\mathbb{E}[R]&=\sum_j b_j\,(1+e^{x_j}), &
\mathbb{E}[R-1]&=\sum_j b_j\,e^{x_j},
\end{aligned}
```

the last two on the belief of {math}`\theta_l`, which gives {math}`\mathbb{E}[\theta_l]`
the same way as {math}`\mathbb{E}[S]`. The *{math}`q` point* of a belief is
{math}`\operatorname{logistic}(x_j)` at the first {math}`j` whose running total of
{math}`b` reaches {math}`q`, and the reported range of a rate is its 2.5% and 97.5%
points. B's belief of {math}`S` starts with {math}`\mathbb{E}[S]=0.100` and a range of
0.2% to 52.5%. Four clean runs counted at full weight would move it to
{math}`\mathbb{E}[S]=0.053`. Its
belief of {math}`\theta_l` charges 1.38 expected attempts at the start, against 1.25 at
the average rate.

The outcome probabilities follow from the two averages:

```{math}
:label: probabilities

P_{\mathrm{loud}}=\mathbb{E}[\theta_l],\qquad
P_{\mathrm{silent}}=(1-\mathbb{E}[\theta_l])\,\mathbb{E}[S],\qquad
P_{\mathrm{clean}}=(1-\mathbb{E}[\theta_l])\,(1-\mathbb{E}[S]).
```

For B, {math}`\mathbb{E}[\theta_l]=0.20` and {math}`\mathbb{E}[S]=0.10` give
{math}`P_{\mathrm{silent}}=0.08` and {math}`P_{\mathrm{clean}}=0.72`. These are reports;
the cost {eq}`objective` averages over the two beliefs instead of reading these means.

### Admission tiers

Admission asks of each forecast cell how likely it is to keep its silently wrong
accepted results within the limit:

```{math}
:label: admission-chance

\pi(c)=P(S\le\varepsilon)=\sum_{j:\ \operatorname{logistic}(x_j)\le\varepsilon}b_j,
```

the share of the belief {eq}`belief-update` of {math}`S` at or below
{math}`\varepsilon`. The chance {eq}`admission-chance` sorts the cell into one of three
tiers:

```{math}
:label: tiers

\text{trusted}\iff\pi(c)\ge\pi_{\mathrm{trust}},\qquad
\text{on trial}\iff\pi_{\mathrm{trial}}\le\pi(c)<\pi_{\mathrm{trust}},\qquad
\text{excluded}\iff\pi(c)<\pi_{\mathrm{trial}}.
```

A cell that very likely meets the limit is trusted, an uncertain cell is on trial, and
only a cell that very likely misses it is excluded. A wide belief, which is all an
unproven cell has, does not exclude a cell by itself; a forecast or a history with
substantial silent risk does. The {math}`\pi_{\mathrm{trust}}` point of the same belief
is at most {math}`\varepsilon` exactly when {math}`\pi(c)\ge\pi_{\mathrm{trust}}`, so
the trusted tier is the same test read from the other side. In the running example, with
{math}`\varepsilon=0.10` and no counted run:

| Cell | Prediction of {math}`S` | {math}`s_0` | {math}`\pi` | Tier | 80% point of {math}`S` |
| --- | --- | --- | --- | --- | --- |
| A | 0.50 | 0.96 | 0.011 | excluded | 0.692 |
| B | 0.10 | 1.60 | 0.702 | on trial | 0.156 |
| C | 0.02 | 1.869 | 0.957 | trusted | 0.021 |

The same four clean runs would move B's {math}`\pi` to 0.85, and B would be trusted. The
{ref}`outcome-belief reference <outcome-posterior>` and the
{ref}`admission reference <admission-tiers>` give the receipt fields.

## 10 Active time, tokens and credit

This section builds the beliefs for a cell's active time and token counts and derives
the credit of one run.

### Amount beliefs

Active time and each token kind have one log-normal belief per cell. For an amount
{math}`A` with prediction {math}`\hat g` and starting spread {math}`s_0` from
{eq}`worth`, where one run's spread is the scatter {math}`\tau`,

```{math}
:label: amount-start

\ln A\sim\mathcal{N}(m_0,s_0^2),\qquad
m_0=\ln\hat g-\frac{s_0^2}{2}.
```

Subtracting {math}`s_0^2/2` makes the belief's average equal the prediction. With the
defaults and an empty record, {math}`N_{\mathrm{start}}=0.833` and
{math}`\tau=1.3\sqrt{0.5}=0.919`, so the smallest starting spread is
{math}`0.919/\sqrt{0.833}=1.007`. C's forecast of 3,600 seconds with no range gives
{math}`s_0=1.007` and {math}`m_0=7.682`: an average of 1 hour and a median of 0.60
hours. A stated range of 600 to 60,000 seconds would give a spread of 1.175, which would
be used.

The cell's own candidate runs that record the amount then update the start
{eq}`amount-start` conjugately. With run {math}`i`'s weight {math}`w_i`
({eq}`evidence-weight` with factor 1, limited per ticket over the cell's own runs),
{math}`W=\sum_i w_i` and {math}`y_i` the logarithm of its recorded amount,

```{math}
:label: amount-update

\frac{1}{s'^2}=\frac{1}{s_0^2}+\frac{W}{\tau^2},\qquad
m'=m_0+s'^2\,\frac{\sum_i w_i\,(y_i-m_0)}{\tau^2},\qquad
\mathbb{E}[A]=\exp\Bigl(m'+\frac{s'^2}{2}\Bigr).
```

Each run pulls the centre toward its own amount and narrows the spread, by its weight.
Had C's record held two runs with weights 0.5 and 0.3 that took 1,800 and 5,400
seconds, {math}`W=0.8`, {math}`s'=0.719`, {math}`m'=7.792` and the average would be
3,137 seconds. {math}`\mathbb{E}[T]` is the average of the active-time belief in hours.
A run's active time is its recorded active seconds, or its total seconds when no active
time was recorded. Each belief reads only the runs that record its value: a run with a
complete usage record and no active time moves the token beliefs alone, and a run that
records only a total token count moves the total-token belief alone. A token kind that
every expert forecasts at zero, with no non-zero upper end, is exactly zero. A cell whose
forecasts do not all state usage by kind has one belief, for total tokens.

### The credit of one run

The credit of one run is uncertain because its token counts are. With
{math}`\mathrm{price}_\kappa` the candidate's rate for kind {math}`\kappa` per million
tokens and {math}`s'_\kappa` the spread of that kind's belief,

```{math}
:label: credit

K_\kappa=\frac{\mathrm{price}_\kappa\,\mathbb{E}[A_\kappa]}{10^6},\qquad
\mathbb{E}[K]=\sum_\kappa K_\kappa,\qquad
\operatorname{Var}(K)=\sum_\kappa\sum_{\kappa'}K_\kappa K_{\kappa'}\bigl(e^{s'_\kappa s'_{\kappa'}}-1\bigr).
```

The double sum treats the kinds as moving together, with correlation 1 on the log
scale. A log-normal with the same average and variance has spread {math}`s_K` and 80%
point {math}`K_{80}`:

```{math}
:label: credit-p80

s_K^2=\ln\Bigl(1+\frac{\operatorname{Var}(K)}{\mathbb{E}[K]^2}\Bigr),\qquad
K_{80}=\mathbb{E}[K]\exp\Bigl(0.8416\,s_K-\frac{s_K^2}{2}\Bigr),
```

where 0.8416 is the 80% point of the standard normal. For example, with input at 2 and
output at 10 USD per million tokens, an input forecast of 1,000,000 tokens with no range
(spread 1.007) and an output forecast of 100,000 tokens with a range of 10,000 to
1,000,000 (spread 1.175) give {math}`K=2.00` and 1.00 by kind,
{math}`\mathbb{E}[K]=3.00`, {math}`\operatorname{Var}(K)=19.06`, {math}`s_K=1.066` and
{math}`K_{80}=4.17`. In the running example, C's single total-token belief gives
{math}`\mathbb{E}[K]=9.00` and {math}`K_{80}=12.65`.

A candidate may carry a credit budget, the credit available to it in its current
session and weekly windows. With {math}`K_{80}` from {eq}`credit-p80`, a cell is
schedulable when

```{math}
:label: credit-check

\max\bigl(\mathbb{E}[K],K_{80}\bigr)\le\text{available credit},
```

so a run must fit the remaining credit four times out of five, as well as on average.
The larger of the two is taken because a belief with {math}`s_K` above 1.68 has its 80%
point below its average, and the check must not get easier as a belief widens. In the
example above, 4.00 of available credit fails although the average fits, and 5.00
passes. A cell with no budget passes. The {ref}`amount-belief reference <amount-beliefs>`
and the {ref}`credit-budget reference <credit-budget>` give the fields and the budget
rules.

## 11 Selection

This section computes each cell's expected cost and its spread, forms the pool of
competing cells and draws the chosen cell.

### Expected cost and its spread

The cost {eq}`objective` is averaged over the four beliefs of the cell:
{math}`\mathbb{E}[R]` and {math}`\mathbb{E}[S]` from {eq}`belief-summaries`,
{math}`\mathbb{E}[T]` from the active-time belief {eq}`amount-update` and
{math}`C_{\mathrm{run}}=\psi\,\mathbb{E}[K]` from {eq}`credit`. For the running
example, with the tiers of {eq}`tiers`:

| Cell | Tier | {math}`C_{\mathrm{run}}` | {math}`\mathbb{E}[T]` | {math}`\mathbb{E}[R]` | {math}`\mathbb{E}[S]` | {math}`J` | {math}`\operatorname{sd}(J)` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | excluded | 0.15 | 0.25 | 2.0515 | 0.500 | 565.70 | 220.04 |
| B | on trial | 1.20 | 0.50 | 1.3783 | 0.100 | 137.80 | 148.12 |
| C | trusted | 9.00 | 1.00 | 1.0807 | 0.020 | 60.78 | 76.18 |

For C, {math}`J=(9.00+25\times1.00)\times1.0807+50\times0.0807+1{,}000\times0.020=60.78`,
and for B, {math}`(1.20+25\times0.50)\times1.3783+50\times0.3783+1{,}000\times0.100=137.80`.
B's escaped defects cost 100 of its 137.80.

The four beliefs of a cell are built independently of one another, so the spread of the
cost follows exactly from their means and variances. Write the cost before averaging as
{math}`XR-C_l+C_sS` with {math}`X=\psi K+\lambda T+C_l`. Then

```{math}
:label: cost-spread

\begin{aligned}
\operatorname{Var}(J)&=\mathbb{E}[X^2]\,\mathbb{E}[R^2]-\bigl(\mathbb{E}[X]\,\mathbb{E}[R]\bigr)^2+C_s^2\operatorname{Var}(S),\\
\mathbb{E}[X]&=C_{\mathrm{run}}+\lambda\,\mathbb{E}[T]+C_l,\qquad
\mathbb{E}[X^2]=\psi^2\operatorname{Var}(K)+\lambda^2\operatorname{Var}(T)+\mathbb{E}[X]^2,\\
\operatorname{Var}(T)&=\mathbb{E}[T]^2\bigl(e^{s'^2}-1\bigr),
\end{aligned}
```

with {math}`\mathbb{E}[R^2]` summed on the grid like {math}`\mathbb{E}[R]` and
{math}`s'` the spread of the active-time belief {eq}`amount-update`. For C,
{math}`\operatorname{Var}(T)=1^2\times(e^{1.007^2}-1)=1.7566`,
{math}`\operatorname{Var}(K)=142.29`, {math}`\mathbb{E}[X]=84`,
{math}`\mathbb{E}[X^2]=142.29+625\times1.7566+84^2=8{,}296.2`,
{math}`\mathbb{E}[R^2]=1.3757` and {math}`\operatorname{Var}(S)=0.002632`, so
{math}`\operatorname{Var}(J)=8{,}296.2\times1.3757-(84\times1.0807)^2+1{,}000^2\times0.002632=5{,}804`
and {math}`\operatorname{sd}(J)=76.2`. A wider stated range on active time or on a token
kind raises the spread of that cell's cost without moving its {math}`J`, because a
belief averages to its prediction however wide it is.

### The pool and eligibility

Only schedulable cells compete. Trusted and on-trial cells form the pool when any
exists; otherwise the cells nearest the best chance compete:

```{math}
:label: pool

\mathcal{P}=
\begin{cases}
\{c:\ \mathrm{credit\_ok}(c),\ c\ \text{trusted or on trial}\} & \text{when not empty},\\
\{c:\ \mathrm{credit\_ok}(c),\ \pi(c)\ge\chi\max_{c'}\pi(c')\} & \text{otherwise},
\end{cases}
```

where {math}`\mathrm{credit\_ok}(c)` is the check {eq}`credit-check` and the maximum
runs over the schedulable cells. Selection is refused only when no cell is schedulable.
For example, three schedulable cells with {math}`\pi` of 0.12, 0.08
and 0.05 and none admitted give the bar {math}`0.5\times0.12=0.06`, so the first two
form the pool. Trusted and on-trial cells compete as equals, and nothing limits how
often an on-trial cell is chosen.

A cell of the pool {eq}`pool` competes when its cost overlaps the lowest cost within
their spreads {eq}`cost-spread`:

```{math}
:label: eligible

c^{*}=\arg\min_{c\in\mathcal{P}}J(c),\qquad
\text{eligible}(c)\iff c\in\mathcal{P}\ \text{and}\ J(c)-J(c^{*})\le M\bigl(\operatorname{sd}(J(c^{*}))+\operatorname{sd}(J(c))\bigr).
```

A risk class's `explore_within` additionally caps the gap at that fraction of
{math}`\lvert J(c^{*})\rvert`. A small difference in expected cost therefore cannot lock
in one cell and starve a cheaper, less known one of evidence. In the running example the
pool is B and C, {math}`c^{*}` is C, and B's gap of 77.02 is within
{math}`76.18+148.12=224.30`, so both are eligible.

### The draw

Each forecast cell draws one value of everything it is uncertain about, and the cell
with the lowest sampled cost among the eligible cells of {eq}`eligible` is chosen:

```{math}
:label: draw

\begin{aligned}
\tilde S&=\operatorname{logistic}(x_j)\ \text{at the first }j\text{ where the running total of the belief of }S\text{ reaches }U_S,\\
\tilde R&=1+e^{x_j}\ \text{at the first }j\text{ where the running total of the belief of }\theta_l\text{ reaches }U_u,\\
\tilde T&=\exp\bigl(m'_{\mathrm{time}}+s'_{\mathrm{time}}\,\Phi^{-1}(U_T)\bigr)/3600,\qquad
\tilde A_\kappa=\exp\bigl(m'_\kappa+s'_\kappa\,\Phi^{-1}(U_K)\bigr),\\
\tilde J&=\bigl(\psi\tilde K+\lambda\tilde T\bigr)\tilde R+C_l(\tilde R-1)+C_s\tilde S,
\end{aligned}
```

where {math}`m'` and {math}`s'` are the centres and spreads of {eq}`amount-update` and
{math}`\tilde K` prices the sampled counts {math}`\tilde A_\kappa` as {eq}`credit`
prices their averages. The four uniform numbers come in this order from a
generator seeded with the decision's stored seed, and every forecast cell draws them, in
a stored order and whether or not it is eligible, so a stored decision replays its draw
exactly. One {math}`U_K` serves every token kind, matching the joint movement assumed in
{eq}`credit`. Ties go to the lower {math}`J`. A wide belief sometimes draws a low cost,
so its cell is tried, and the trial's outcome then narrows the belief. No expert is
drawn: each value of a cell is one belief merged over the experts, and experts that
disagree have already widened it.

With seed 42 in the running example, C draws {math}`\tilde S=0.0180`,
{math}`\tilde R=1.0106`, {math}`\tilde T=1.7763` hours and {math}`\tilde K=9.122`, so
{math}`\tilde J=(9.122+25\times1.7763)\times1.0106+50\times0.0106+1{,}000\times0.0180=72.6`.
B draws a high loud rate, {math}`\tilde R=3.075`, with {math}`\tilde S=0.0058`,
{math}`\tilde T=0.6156` hours and {math}`\tilde K=1.606`, so
{math}`\tilde J=(1.606+25\times0.6156)\times3.075+50\times2.075+1{,}000\times0.0058=161.8`.
C is chosen.

### The review flag and widening

The decision carries a flag when the chosen cell is not trusted:

```{math}
:label: flag

\texttt{cap\_met}\iff\text{the chosen cell is trusted},\qquad
\texttt{needs\_review}\iff\text{not}\ \texttt{cap\_met}.
```

The flag on the cell chosen by {eq}`draw` advises the caller to have an on-trial or
excluded choice reviewed independently; nothing enforces it. C is trusted, so
`needs_review` is false. Had the draw chosen B, with {math}`\pi=0.702`, the decision
would carry `needs_review`.

When every forecast cell is excluded, the engine first asks for a forecast of the next
stronger cell without one, one per round: the band cells beyond the core, then the
cells above the band, in capability order, up to two rounds (`widening_rounds`) within
the forecast budget. One trusted or on-trial forecast cell stops the widening. The
{ref}`selection reference <cost-spread-and-draw>` gives the stored draw and the
explanation of each cheaper alternative.

## 12 Refined constants

This section re-estimates eleven of the constants above from the store's own record.

The half-life {math}`h`, the three discounts, {math}`N_{\mathrm{rung}}`, the three task
worths, the miss {math}`D` with no record, the scatter share {math}`\beta` and
{math}`N_{\mathrm{caught}}` are estimated constants. Their defaults work on an empty
record. Once the record holds 200 scored runs, each is re-estimated from it after an
observation or a flush, never during a decision, and with no tuning by the user. A
constant set in the configuration file keeps that value.

A *scored run* is a finished routed run whose receipt has the current format. Its
beliefs are rebuilt with a tried value from what its receipt stores and from the runs and
amendments observed before its decision, and it is scored by the log probability those
beliefs gave to what happened:

```{math}
:label: scores

\begin{aligned}
\text{outcome}_i&=
\begin{cases}
\ln\mathbb{E}[\theta_l] & \text{loud},\\
\ln(1-\mathbb{E}[\theta_l])+\ln\mathbb{E}[S] & \text{silent},\\
\ln(1-\mathbb{E}[\theta_l])+\ln(1-\mathbb{E}[S]) & \text{clean},
\end{cases}\\
\text{time}_i&=\ln\mathcal{N}\bigl(y_i;\,m',\,s'^2+\tau^2\bigr),\qquad
\text{caught}_i=
\begin{cases}
\ln d & \text{loud},\\
\ln(1-d) & \text{silent},
\end{cases}
\end{aligned}
```

where {math}`y_i` is the logarithm of the run's active seconds, {math}`m'` and
{math}`s'` belong to its cell's active-time belief, and {math}`d` is its host's caught
share before the decision. The half-life, the discounts, {math}`N_{\mathrm{rung}}` and
the two rate worths are judged by the outcome score, the worth for {math}`S` on accepted
runs only; {math}`D`, {math}`\beta` and the amount worth by the time score; and
{math}`N_{\mathrm{caught}}` by the caught score. At most the 2,000 most recent scored
runs are used. The constants are tried one at a time in a fixed order, each with the
earlier ones at their refined values, and a value replaces the current one only when
the scores {eq}`scores` of the runs satisfy

```{math}
:label: rule

\Delta_i=\text{score}_i(\text{tried})-\text{score}_i(\text{current}),\qquad
\text{gain}=\frac{1}{n}\sum_i\Delta_i,\qquad
\text{se}=\frac{\operatorname{sd}(\Delta)}{\sqrt n},\qquad
n\ge200\ \text{and}\ \text{gain}>2\,\text{se}.
```

The best passing value wins, and a tie keeps the one nearer the default. For example,
400 runs whose differences average 0.012 with a standard deviation of 0.10 give
{math}`\text{se}=0.005`, so the constant moves; an average of 0.008 does not. {math}`D`
and {math}`\beta` also try the values the record measures: {math}`D` is the root mean
square of the timed runs' misses {math}`e_i`, and the scatter
{math}`\tau=\sqrt{\operatorname{mean}\bigl((y_{i'}-y_i)^2\bigr)/2}` comes from the
differences between consecutive timed runs of one cell on one ticket, once 30 such pairs
exist, giving {math}`\beta=\min(1,\tau^2/D^2)`. The
{ref}`refined-constants reference <refined-constants>` and the
{ref}`configuration guide <refined-constants-configuration>` list the values tried and
the cached result.

## 13 Retrospective defect attribution

This section corrects the record when a defect is traced to an earlier accepted run.

A task that passes today's checks can contain a defect found later. During a bug fix or
a requested audit, the companion `attribute-failure` skill investigates whether the
defect escaped an earlier accepted task:

1. Reproduce the defect or obtain other concrete evidence of the failure.
2. Trace the behaviour through the original task, the history before the fix and the
   execution's provenance, using bisection when it helps. The last change to a line
   alone does not establish the cause.
3. When the attribution is supported, append an amendment to the original execution's
   outcome and replay its frozen forecasts against the corrected outcome.
4. Update written lessons through the ordinary learning step, and record the fixing
   execution's own outcome separately.

The amendment replaces the run's earlier score; it creates neither another execution
nor a new forecast exposure, and the run keeps its identity, context and measurements.
An uncertain attribution stays unscored, and finding an escaped defect does not by
itself make the fixing worker fail.

In the running example, suppose such an investigation amends C's run to silent on the
day of the run, and the same host then routes a similar task with the same three cells
and the same stored forecasts. The amended run changes the decision in three places. It
is now a failed routed run that the host did not catch, so the host's caught share
{eq}`caught-share` falls from 0.889 to 0.757, which tightens the band edges
{eq}`band-edges` to {math}`P_{\mathrm{lo}}=0.686` and {math}`G=0.0229`. It is a
candidate run with a similarity of 0.8, so it lowers every cell's
{math}`P_{\mathrm{clear}}` (A 0.33, B 0.57, C 0.73); the band holds C alone, and A and
B, whose families are still unproven, take the two reserved slots. Finally it counts in
the outcome beliefs: in full, at weight 0.8, for C, whose {math}`\pi` falls to 0.66 and
whose {math}`J` rises to 152.9, and at the discount {math}`\delta_{\mathrm{model}}=0.25`
for B, a cell of another family under the same unnamed agent, whose {math}`\pi` falls to
0.60 and whose {math}`J` rises to 168.7. B and C are now both on trial with overlapping
costs, and with seed 42 the draw picks B, so the decision carries `needs_review` by
{eq}`flag`. The {ref}`attribution contract <escaped-defect-attribution>` states the
evidence and provenance requirements.

## 14 Operational boundaries

This section states what the method relies on from the host and the store.

The host executes the chosen model and effort. An installed plugin does not by itself
enforce routing at every worker launch, and the feedback must identify the invocation
that actually ran. Money comparisons need known charges on a comparable basis: list
prices stand in for neither subscription credit nor native host charges, and unknown
usage and cost stay unknown. Usage collectors read only artifacts they are explicitly
given. Every decision stores a receipt that freezes its inputs, retrieved evidence,
memory versions, forecasts, capability state and selection, and loading a receipt
recomputes every derived number from what it stores. The stored seed reproduces a
selection given its frozen forecasts, while a fresh call to a language model can answer
differently. Git worktrees of one project share its store and capture their own
checkout. The [model and storage contracts](reference.md), the [host skills](skills.md)
and the [concurrency contract](concurrency.md) define these boundaries.

## 15 Evaluating savings and learning

This section sets out how to test whether the method saves money and learns on real
work.

Net savings require evidence from genuine project work that includes the cost of
choosing and of learning. The evaluation compares four arms: a stateless language
model, plain retrieval of earlier executions, writable memory with fixed weights, and
memory with learned predictive weights. Candidate inventories, prices, worker tools,
acceptance gates and review budgets are frozen before any outcome is known. Three
questions are assessed separately:

| Question | Evidence to report |
| --- | --- |
| Do forecasts improve? | Log loss and Brier score on matched observed actions; reliability with counts and uncertainty |
| Does the routing policy save money or credit? | Comparable worker charges plus routing and amortized writer costs; retries, latency, review and rework |
| Does delayed learning help? | Prospective forecasts on later related tasks, amendments after supported attribution, and follow-up coverage |

The configured failure penalties and the priced time belong to the policy's loss, so
measured review and rework are reported separately and not counted twice. The chosen
actions of a policy cannot reveal what an unrun alternative would have done, so
comparative claims need actual alternative executions or a prospective randomized
trial; a reproducible seed is not an assignment probability. The evaluation runs in
chronological order with only the information available at each decision. Delayed
feedback is released when it became known, related tasks and retries stay grouped, and
failed decisions and inconvenient outcomes stay in. Accepted work is followed up after
7 and 30 days, and missing follow-up is not evidence that a result was clean. The first
feasibility study of 12 tasks checks measurement and replay; it cannot establish general
superiority or a small silent-failure rate. Implementation tests verify behaviour and
retrieval diagnostics measure one component; neither establishes savings for deployed
routing. Field evidence must justify the additional cost of routing and learning against
simpler comparators. The [real-project evaluation guide](evaluation.md) gives the
procedure.

## 16 Statistical interpretation

This section states once what each component means statistically and what it assumes.

The method combines language-model priors, Bayesian capability learning,
evidence-weighted semantic memory and retrospective defect attribution.

The independent language-model judgments are cold-start priors for each task's order
and outcomes, and their quality needs empirical validation. The corrected prediction
removes a forecaster's typical level, measured on its record; the method makes no
assumption that the baseline is unbiased or calibrated.

The capability threshold is a generalized Bayesian update (Bissiri, Holmes and Walker,
2016): likelihoods from related tasks are raised to fractional weights. It assumes that
one scalar order describes the demands of the task, which different tasks can violate,
and that a stronger cell does at least as well, which a higher effort does not
guarantee. The similarity weights are heuristic transfer strengths without a calibrated
probabilistic meaning. The order search is a constrained local search that can miss the
highest-posterior order. A cell that has never run is placed by the prior, and the
inference about it is conditional on that placement.

The evidence weights combine similarity, a half-life of 14 days by default and a
per-ticket limit. The limit treats the retries of one ticket as one run's worth of
evidence. The cap of 256 scored candidates and the model worth {math}`N_{\mathrm{model}}`
are fixed. The half-life, the discounts, the rung worth, the task worths, the default
miss, the scatter share and the caught-share worth start at defaults and move only on a
clear gain over at least 200 scored runs. A run rescored for refinement is rebuilt with
the extra matched runs beyond the cap as history, which its decision did not have.

Expert support is tempered, recency-weighted predictive evidence, similar in form to
exponentially weighted forecasters (Cesa-Bianchi and Lugosi, 2006). It measures how well
an expert predicted. Whether a lesson is true, and why an outcome happened, lie outside
what it measures, and it is a heuristic weighting with no exact joint posterior over
memories, orders and outcomes behind it.

The corrected prediction is a partial-pooling estimate (Gelman et al., 2013) whose worth
at each rung is set rather than estimated within the decision. The outcome beliefs are
logistic-normal (Aitchison and Shen, 1980) on a fixed grid. They treat the loud rate and
the accepted-defect share as independent, treat the counted runs as exchangeable, and
keep the forecast at the worth of {math}`N_{\mathrm{start}}` runs however many runs are
counted. Their ranges and the admission tiers are Bayesian under these assumptions and
carry no calibrated coverage guarantee. The limit {math}`\varepsilon` is a selection
rule that makes no promise about the future defect rate, and the tier lines 0.80 and
0.20 are policy choices.

The amount beliefs are log-normal with a conjugate update from the cell's own runs. The
credit check matches a log-normal to the mean and variance of run credit (Fenton, 1960),
which approximates the distribution of a sum, and it assumes that the token kinds move
together because no run records how they co-vary. The check covers one run: it reserves
nothing for the uncertain cost of later retries and does not see spending by other
clients.

The cost averages over four per-cell beliefs that it treats as independent: credit,
active time, the loud rate and the accepted-defect share. Retries are assumed to stay on
the same cell with a stationary loud rate, so the number of attempts is geometric; real
retries can be correlated or change the task, and a failed attempt may in practice move
to another cell. The penalties {math}`C_l` and {math}`C_s` express policy, and actual
charges and rework remain separately measured evidence.

Selection is Thompson sampling (Thompson, 1933; Russo et al., 2018) restricted to cells
whose costs overlap within one spread on each side. The overlap rule is a heuristic
with no test behind it, and the draw uses one shared number for all token kinds.

An amendment corrects an earlier outcome and replays its forecasts. It requires a
supported attribution and preserved provenance, and an unattributed defect leaves the
record as it was.

Together these mechanisms give routing access to both immediate outcomes and mistakes
discovered later. Whether that information reduces total cost on future work enough to
pay for acquiring it is the question the evaluation in section 15 asks.

## References

- Aggarwal, P., Madaan, A., Anand, A., Potharaju, S. P., Mishra, S., Zhou, P., Gupta, A.,
  Rajagopal, D., Kappaganthu, K., Yang, Y., Upadhyay, S., Faruqui, M. and Mausam (2024).
  AutoMix: Automatically mixing language models. *Advances in Neural Information
  Processing Systems 37 (NeurIPS 2024)*. <https://arxiv.org/abs/2310.12963>
- Aitchison, J. and Shen, S. M. (1980). Logistic-normal distributions: Some properties
  and uses. *Biometrika* 67(2), 261–272. <https://doi.org/10.1093/biomet/67.2.261>
- Ayer, M., Brunk, H. D., Ewing, G. M., Reid, W. T. and Silverman, E. (1955). An
  empirical distribution function for sampling with incomplete information. *The Annals
  of Mathematical Statistics* 26(4), 641–647. <https://doi.org/10.1214/aoms/1177728423>
- Bissiri, P. G., Holmes, C. C. and Walker, S. G. (2016). A general framework for
  updating belief distributions. *Journal of the Royal Statistical Society, Series B*
  78(5), 1103–1130. <https://doi.org/10.1111/rssb.12158>
- Cesa-Bianchi, N. and Lugosi, G. (2006). *Prediction, Learning, and Games*. Cambridge
  University Press. <https://doi.org/10.1017/CBO9780511546921>
- Chen, L., Zaharia, M. and Zou, J. (2024). FrugalGPT: How to use large language models
  while reducing cost and improving performance. *Transactions on Machine Learning
  Research*. <https://arxiv.org/abs/2305.05176>
- Ding, D., Mallick, A., Wang, C., Sim, R., Mukherjee, S., Rühle, V., Lakshmanan, L. V. S.
  and Awadallah, A. H. (2024). Hybrid LLM: Cost-efficient and quality-aware query
  routing. *International Conference on Learning Representations (ICLR 2024)*.
  <https://arxiv.org/abs/2404.14618>
- Ding, D., Mallick, A., Zhang, S., Wang, C., Madrigal, D., Hipolito Garcia, M. D. C.,
  Xia, M., Lakshmanan, L. V. S., Wu, Q. and Rühle, V. (2025). BEST-Route: Adaptive LLM
  routing with test-time optimal compute. *International Conference on Machine Learning
  (ICML 2025)*. <https://arxiv.org/abs/2506.22716>
- Fenton, L. (1960). The sum of log-normal probability distributions in scatter
  transmission systems. *IRE Transactions on Communications Systems* 8(1), 57–67.
  <https://doi.org/10.1109/TCOM.1960.1097606>
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A. and Rubin, D. B.
  (2013). *Bayesian Data Analysis*, third edition. Chapman and Hall/CRC.
  <https://doi.org/10.1201/b16018>
- Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and
  estimation. *Journal of the American Statistical Association* 102(477), 359–378.
  <https://doi.org/10.1198/016214506000001437>
- Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J. E., Kadous, M. W.
  and Stoica, I. (2025). RouteLLM: Learning to route LLMs from preference data.
  *International Conference on Learning Representations (ICLR 2025)*.
  <https://arxiv.org/abs/2406.18665>
- Russo, D. J., Van Roy, B., Kazerouni, A., Osband, I. and Wen, Z. (2018). A tutorial on
  Thompson sampling. *Foundations and Trends in Machine Learning* 11(1), 1–96.
  <https://arxiv.org/abs/1707.02038>
- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another
  in view of the evidence of two samples. *Biometrika* 25(3–4), 285–294.
  <https://doi.org/10.2307/2332286>
- Wang, X., Liu, Y., Cheng, W., Zhao, X., Chen, Z., Yu, W., Fu, Y. and Chen, H. (2025).
  MixLLM: Dynamic routing in mixed large language models. *Proceedings of the 2025
  Conference of the Nations of the Americas Chapter of the Association for
  Computational Linguistics (NAACL 2025)*, 10912–10922.
  <https://doi.org/10.18653/v1/2025.naacl-long.545>
