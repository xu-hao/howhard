# Model and storage contracts

Routing uses LLM forecasts and writable semantic memories. See the [reader/writer
contract](semantic-memory.md) for retrieval, isolated forecasts and predictive weights.
The [retrieval stages](retrieval.md) select execution precedents and written lessons
before support is normalized over participating forecasts. Observed runs retain their
joint model/effort, outcome and measured usage; no unrun action is labeled retrospectively.
The [coarse reader](capability.md) learns a task-specific capability threshold from the
same observed runs. Fine forecasts retain their independent LLM and memory protocol;
there is no token/time regression, lexical outcome predictor or statistical risk floor.

(accepted-result-selection)=
## Selection

For each supplied native model and effort/control cell, minimize

```text
J = (C_run + lambda_per_hour * T + C_loud * P_loud + C_silent * P_silent)
    / max(probability_floor, 1 - P_loud)
```

`C_run` is the expected actual session money cost (`session_cost_usd`). The candidate's
`rate_card` or `usd_per_mtok` gives raw public API list-price quota credit, reported as
`needed_credit_usd`; `cost_per_credit` converts that credit amount to session cost and
defaults to `1.0` for pay-as-you-go. A free tier can set it to `0.0`. `T` is the expected
active hours of one run. `T` and `needed_credit_usd` are the averages of the cell's
{ref}`beliefs for active time and tokens <amount-beliefs>`.

The caller supplies any nonempty number of distinct actual cells through `--candidates FILE` or Python
`native_candidates`. External routers can explicitly select a saved list from
`howhard.toml` with `--inventory NAME` or Python `inventory="NAME"` instead. Neither
option is implicit, and the two sources cannot be combined. An empty list is refused. Each has `model`,
`effort` and optional `agent`, `usd_per_mtok` or `rate_card`, `cost_per_credit`, and `credit_budget`. External routers can use stable
provider-qualified model IDs; execution and provider access belong to the caller.
See {ref}`inventory configuration <external-router-inventories>`.

(host-and-agent-names)=
Two optional names say who is involved in a route. Both are short labels of letters,
digits, `_` and `-`, at most 64 characters; `howhard` stores them and never verifies them.

- **Host**: the agent that asks for the route, makes the forecast and accepts or rejects
  the result. `pick`, `summary` and `context` take `--host NAME`
  (`Engine.pick/summary/prepare(..., host="NAME")`). The name travels on the prepared
  request, is returned with it, and is stored as `host` in the decision receipt. A
  request with no name belongs to the one unnamed host and stores `"host": null`.
- **Agent**: the agent product that would run a cell, given as `agent` on a candidate
  or on an entry of a configured inventory. The label is stored with its cell in the
  receipt, in `native_candidates` and as `agent` on each `capability.cells` entry. A
  candidate with no label belongs to the one unnamed agent (`"agent": null` on its
  capability cell). The label is not part of a cell's identity: two candidates with the
  same `model` and `effort` are duplicates whatever their labels.

Selection among the forecast cells reads neither name. The capability filter reads
both: the host's name picks whose failed runs set the two edges of the fit band, and
the agent labels decide which agents the reserved forecast slots count as missing (see
{ref}`the host's caught share <host-aware-band>` and
{ref}`reserved forecast slots <reserved-slots>`). `report` lists both under `routing`.
No-effort, null and `model_default` controls are refused. Unknown identities receive distinct
opaque family labels for protocol bookkeeping, without model assumptions or effects.
Configured identities retain their declared labels and evidence weights. Historical
identity metadata does not activate candidates in a later request.

A [task-specific capability filter](capability.md) first reads the full inventory.
Cold-start stateless LLM calls evaluate capability and fine forecasts at most once per
unseen `(prototype, model, effort)` cell against the prototype's synthesized
`(representative_task, representative_context)`, materializing priors and empirical cell
statistics in `payloads.sqlite3`. When matched prototypes already cover all cells and no
active editable memories exist, `pick`, `summary`, and `context --auto-decide` complete
locally without invoking a provider; when new unseen candidates appear on a warm
prototype, Stage 2 requests forecasts only for the unseen cells.
The reader supplies conditional overrides for memories it has seen; omitted cells and
unretrieved memories use the baseline forecast. No learned probability table
or synthetic forecast is supplied to isolated baseline calls. Cold-start provider absence, invalid forecasts
and provider failures are explicit errors; none produces a routing recommendation.

(outcome-posterior)=
### Per-cell outcome posterior

Each expert forecast row is a bounded Dirichlet prior for that `(family, effort)` cell,
updated only by that cell's own observed outcomes (`W_emp = n_clean + n_loud + n_silent`),
with continuous cold-start prior decay over `tau_emp = 2.0`:

```text
p_0         = floored, normalized (clean, loud, silent) forecast
kappa_0     = prior_strength, lowered by optional credible intervals, within
              [min_concentration, prior_strength]
decay       = max(0, 1 - W_emp / 2.0)
kappa_prior = kappa_0 * decay
kappa_base  = 1.0 * (1 - decay)
b_0         = (0.88, 0.08, 0.04)
alpha       = kappa_prior * p_0 + kappa_base * b_0 + (n_clean, n_loud, n_silent)
```

At `W_emp = 0`, `prior_concentration = kappa_0` and `alpha = kappa_0 * p_0`. As real
executions accumulate on the cell, the cold-start LLM prior decays linearly to exact
zero weight at `W_emp >= 2.0`, leaving only the unit empirical base smoothing `b_0` plus
the cell's own weighted counts so pessimistic guesses on unfamiliar or post-cutoff model
identifiers never depress observed winners permanently.

`n_*` are the counted amounts of each outcome in the cell's `outcome_observations`.
The runs are the decision's {ref}`evidence candidates <evidence-candidates>`: recorded
`clean`, `loud` and `silent` runs that belong to a prototype the new task matches
(`interrupted` runs are excluded), at most `comparison_cap` of them, most recent first.
For each forecasted `(family, effort)` cell (and any widened cell):

```text
w_i = m_i * sim_i * delta(c_i, c) * a_i
W_t = sum of w_i over the candidate runs of ticket t
w_i <- w_i / max(1, W_t)
n_k = sum of w_i over the candidate runs with outcome k
```

- `i`: a candidate run; `c_i` is the cell that ran it and `c` the cell being judged.
- `m_i`: the run's weight on record, the `weight` configured for its model (1 unless set).
- `sim_i`: `1 / (1 + exp(-r_i))`, where `r_i` is the reranker's score for the new task's
  text against the run's own task text.
- `delta(c_i, c)`: 1 for the same cell and 0 for any other, so only a cell's own runs count.
- `a_i`: the age factor `0.5^(days_i / half_life_days)`, where `days_i` runs from the
  run's `observed_at` to the time the request was prepared and is never below 0.
- `t`, `W_t`: a ticket (the row's `ticket` label) and the total weight of its candidate
  runs for cell `c`. A row with no ticket is not limited.
- `n_k`: the counted amount of outcome `k`, for `clean`, `loud` and `silent`.

In plain words: a run counts more when its task reads like the new one, half as much
for every 14 days of age, and all runs of one ticket together count for at most one
run. For example, a run of the same cell with `r = 1.386` recorded 7 days before the
request weighs `0.8 * 0.7071 = 0.5657`; three runs of one ticket weighing 0.6, 0.5 and
0.4 become 0.4, 0.3333 and 0.2667. Heavy history on another cell cannot crowd out this
cell's evidence, a regression or recovery dominates as older runs age, and a task that
matches no prototype counts nothing. A never-observed cell keeps its prior variance no
matter how many runs other cells have; an expert's global `evidence_count` affects only
its predictive weight, never a cell's sample size.

A forecast row may add `silent_ci` and/or `loud_ci` as `[lo, hi]` with
`0 <= lo <= value <= hi <= 1`, a central 95% credible interval for that probability.
Moment matching gives `kappa = p*(1-p)/((hi-lo)/3.92)^2 - 1`; the smallest such value is
clamped to `[min_concentration, prior_strength]`. Intervals can therefore widen a prior
but an overconfident narrow interval can never make a forecast stronger than
`prior_strength` pseudo-observations. Bare `clean`/`loud`/`silent` rows remain valid and
use `kappa_0 = prior_strength`.

(forecast-contract)=
### Forecast rows and stated ranges

A forecast row names one cell and states a value for each quantity a decision needs.
Every range is optional and is a central 95% range `[lo, hi]`.

| Value | Required | Optional range |
| --- | --- | --- |
| `clean`, `loud`, `silent` (sum to 1) | yes | `loud_ci`, `silent_ci` |
| `seconds` (active execution time) | yes | `seconds_ci` |
| `tokens` (total) | yes | none |
| `token_usage` (`input`, `output`, `cache_read`, `cache_write`, `cache_write_5m`, `cache_write_1h`) | only for a cell with a category rate card | `token_usage_ci: {kind: [lo, hi]}` for any of the six kinds |

- A range must contain its value, with `0 <= lo <= value <= hi`. A rate range also needs
  `hi <= 1`. A range that misses its value, or is not two finite non-negative numbers,
  is refused with a message that names the field (for example
  `forecast.token_usage_ci.input`).
- A rate range of `[0, 1]`, or anything wider than `[0.025, 0.975]`, states a flat
  belief: no knowledge of that probability. There is no other ceiling on a stated width.
  Selection reads such a range through the moment matching above, like any other
  stated rate range.
- A token kind forecast at zero is exactly zero unless a non-zero upper end is stated.
- `token_usage_ci` needs `token_usage` on the same row and accepts only the six kinds.
  There is no range on total `tokens` or on credit: token counts do not depend on
  prices, and a stored forecast is reused after prices change.

Stated ranges are stored unchanged in the receipt's forecast panels
(`semantic.worlds`). `loud_ci` and `silent_ci` are read as described above.
`seconds_ci` and `token_usage_ci` are read into the beliefs below: a stated range can
widen a belief and never narrow it.

(amount-beliefs)=
### Beliefs for active time and tokens

Active time and each token kind have one belief per cell, held on the log scale. A
belief starts at the forecast, is as wide as the forecast is unsure, and is updated by
the cell's own recorded runs. Rates keep the {ref}`outcome posterior <outcome-posterior>`.

**One forecast per value.** Several memory experts may forecast one cell. Their forecasts
of a value are merged before its belief is built.

```text
m_g   = sum_k w_k * m_k
s_g^2 = sum_k w_k * (s_k^2 + (m_k - m_g)^2)
```

- `k`, `w_k`: an expert in `semantic.beliefs.experts` and its weight. Weights sum to 1.
- `m_k`: the log of expert k's forecast: `seconds`, or one kind of `token_usage`.
- `s_k`: `(ln hi - ln lo) / 3.92` from the expert's stated range (`seconds_ci`,
  `token_usage_ci`), or 0 with no range.
- `m_g`, `s_g`: the log of the merged forecast, and its stated spread.

Forecasts are averaged on the log scale, and disagreement counts as stated uncertainty.
Amounts and range ends are held at 1 or more before logs are taken. For example, experts
with weights 0.75 and 0.25 forecast 3,000 and 6,000 seconds with no ranges. The merged
forecast is 3,568 seconds with stated spread 0.30.

**The active-time belief.**

```text
ln T ~ Normal(m, s^2)
s = max(s_g, tau / sqrt(N_start))        tau = D * sqrt(scatter_share)
N_start = 1 / (1/worth + 1/N_task)
m = ln(prediction) - s^2 / 2
```

- `T`: the expected active seconds of one run of this cell on this task.
- `prediction`: where the belief starts: the merged forecast.
- `s_g`: the stated spread of the merge above.
- `D`: the typical remaining miss, as a spread of `ln(actual / prediction)`
  (`sampling.duration_miss`, 1.3).
- `tau`: the run-to-run scatter on one task. `scatter_share` is the share of `D^2` that
  is scatter; at its default of 0.5, `tau = D / sqrt(2)`.
- `worth`: how many runs the record says about this family's own level
  (`sampling.model_worth`, 5).
- `N_task`: the most a record on other tasks is worth for one task
  (`sampling.task_worth_amount`, 1).
- `N_start`: how many runs the prediction is worth.
- `m`, `s`: the centre and spread. Subtracting `s^2 / 2` makes the belief's average
  equal the prediction.

The belief averages to the prediction. A stated range can widen it and never narrow it.
For example, the prediction is 3,600 seconds with no stated range, worth is 5 and D is
1.3. Then `N_start = 0.833`, `tau = 0.919`, `s = 1.007` and `m = 7.682`: an average of
1 hour and a median of 0.60 hours. A stated range of 600 to 60,000 seconds gives 1.175,
which is used. With the defaults the minimum spread is 1.007 for active time and for
every token kind.

Runs of the same cell then update the belief.

```text
1/s'^2 = 1/s^2 + W / tau^2
m'     = m + s'^2 * sum_i w_i * (y_i - m) / tau^2
E[T]   = exp(m' + s'^2 / 2)
```

- `i`, `w_i`, `W`: a same-cell run in `semantic.duration_observations`, its weight, and
  the sum of the weights.
- `y_i`: the log of run i's active seconds.
- `m'`, `s'`: the centre and spread after the runs.
- `E[T]`: the belief's average. In hours it is `T` in the cost.

Each counted run pulls the centre toward its own time and narrows the belief, by its
weight. For example, two runs with weights 0.5 and 0.3 recorded 1,800 and 5,400 seconds.
Then `W = 0.8`, `s' = 0.719`, `m' = 7.792` and `E[T] = 3,137` seconds.

The runs are the cell's own {ref}`evidence candidates <evidence-candidates>` that record
an active time. Each counts at the weight the outcome counts give that run, `w_i` after
the per-ticket limit: similarity to the new task, half as much for every
`half_life_days` of age, and at most one run's weight per ticket. A run of another cell
never counts. A run's time is `effective_active_seconds`: its `active_seconds`, or its
total `seconds` when no active time was recorded, so a total that includes waiting counts
as active time. `interrupted` runs are excluded.

**Token beliefs.** Each of the six token kinds gets a belief of the same form, with the
same `tau` and `N_start`, updated the same way and at the same weights from those
same-cell runs whose `execution_usage` is `complete`. Each scored candidate, and each row
of `duration_observations` counted from it, carries what its run recorded: `tokens` (the
total, or null) and `token_usage` (the count of each kind over a complete usage envelope,
or null). A run without complete usage leaves the kinds unchanged.

- Zero: a kind that every expert forecasts at zero, with no non-zero upper end, is
  exactly zero and has no spread. No run moves it.
- Priced per total token: a cell whose forecasts do not all state `token_usage` has one
  belief, for total `tokens`, with no stated range, updated from each run's recorded
  `tokens`. When every expert states usage by kind, each kind has its belief; a cell
  with `usd_per_mtok` prices all of them at the one rate.

**Credit: average, spread and check.**

```text
c_a       = price_a * E[tokens_a] / 1e6
E[K]      = sum_a c_a
Var(K)    = sum_a sum_b c_a * c_b * (exp(rho_ab * s_a * s_b) - 1)
sigma_K^2 = ln(1 + Var(K) / E[K]^2)
K_80      = E[K] * exp(0.8416 * sigma_K - sigma_K^2 / 2)
credit_ok <=> max(E[K], K_80) <= available credit
```

- `a`, `b`: token kinds (or the one total).
- `price_a`: the cell's rate for kind a per million tokens, at decision time.
- `E[tokens_a]`, `s_a`: the average and spread of kind a's belief after its update.
- `rho_ab`: how kinds a and b move together. It is 1: the kinds are treated as moving
  together, because no run records how they co-vary.
- `K`: the credit one run needs, with average `E[K]` and variance `Var(K)`.
- `sigma_K`: the spread of a log-normal with that average and variance.
- `K_80`: its 80% point. 0.8416 is the 80% point of the standard normal.
- `available credit`: `credit_budget.available_credit_usd`. With no budget the check
  passes.

A run must fit the remaining credit four times out of five, not only on average. The
larger of the two is taken because a belief with `sigma_K` above 1.68 has its 80% point
below its average, and the check must not get easier as a belief gets wider.

For example, input costs 2 and output 10 dollars per million tokens. Input is forecast
at 1,000,000 tokens with no range (spread 1.007), and output at 100,000 with a range of
10,000 to 1,000,000 (spread 1.175). Then `c = 2.00` and `1.00`, `E[K] = 3.00`,
`Var(K) = 19.06`, `sigma_K = 1.066` and `K_80 = 4.17`. With 4.00 of credit available the
cell fails, although its average fits. With 5.00 it passes.

`needed_credit_usd` is `E[K]`, `credit_variance` is `Var(K)` and `credit_p80_usd` is
`K_80`. `session_cost_usd` is `E[K] * cost_per_credit`, and `C_run` equals it. An
unpriced cell keeps its token beliefs and null credit fields.

Each row of `selection.cells` stores its beliefs:

| Field | Content |
| --- | --- |
| `active_time` | `forecast_seconds` (merged forecast), `stated_spread` (`s_g`), `worth`, `N_start`, `D`, `prior_centre` and `prior_spread` (`m`, `s`), `centre` and `spread` (`m'`, `s'`), `evidence_weight` (`W`), `expected_seconds` (`E[T]`) |
| `token_beliefs` | The same record for each of the six kinds, or one record under `tokens`, with `forecast_tokens` and `expected_tokens`. A kind that is exactly zero has null centres and zero spreads. |
| `credit_variance`, `credit_p80_usd` | `Var(K)` and `K_80`, beside `needed_credit_usd` |

`P_clean`, `P_loud` and `P_silent` report `{mean, lower, upper}` from the
support-weighted mixture of the experts' Beta marginals `Beta(alpha_i, sum(alpha) - alpha_i)`:
the mean and the equal-tailed 2.5%/97.5% quantiles, expanded to contain the mean. This is
a real posterior interval even when only the `llm` expert participates. Accepted-result
risk `P_silent / (1 - P_loud)` is, under each Dirichlet, `Beta(alpha_silent, alpha_clean)`;
`P_silent_accepted` reports the same mixture mean and 95% interval. There is no
additional statistical risk floor, and the posterior is a model assumption, not a
calibrated coverage guarantee. `J` uses the posterior means.

(admission-tiers)=
### Admission tiers

Every forecast cell is sorted by its chance of meeting the risk class's limit on
silently wrong accepted results:

```text
pi(c) = P(S <= epsilon)
      = sum over k of  w_k * I(epsilon; alpha_silent_k, alpha_clean_k)
```

- `c`: a forecast cell.
- `S`: the accepted-defect share, silent results / accepted results
  (`P_silent / (1 - P_loud)`).
- `epsilon`: the risk class's limit on `S`.
- `pi(c)`: the share of the cell's posterior for `S` at or below `epsilon`.
- `k`, `w_k`: an expert in `semantic.beliefs.experts` and its weight. The weights sum
  to 1.
- `alpha_silent_k`, `alpha_clean_k`: the silent and clean entries of expert `k`'s
  posterior `alpha` for the cell.
- `I(x; a, b)`: the regularized incomplete beta function, the share of a `Beta(a, b)`
  distribution at or below `x`.

In plain words, `pi` is the chance, under the cell's current belief, that no more than
`epsilon` of its accepted results are silently wrong.

Example: `epsilon` is 0.10. Two experts weigh 0.6 and 0.4, and their posteriors for `S`
are `Beta(1, 30)` and `Beta(1, 9)`. Then `I` is `1 - 0.9^30 = 0.958` and
`1 - 0.9^9 = 0.613`, so `pi = 0.6 * 0.958 + 0.4 * 0.613 = 0.820`.

```text
trusted   <=> pi(c) >= admission_quantile
on trial  <=> trial_probability <= pi(c) < admission_quantile
excluded  <=> pi(c) < trial_probability
```

- `admission_quantile`: the trusted line, 0.80 by default.
- `trial_probability`: the on-trial line, 0.20 by default.

In plain words, a cell that very likely meets the limit is trusted, an uncertain cell is
on trial, and only a cell that very likely misses the limit is excluded. A wide
posterior alone therefore puts an unproven cell on trial instead of shutting it out.

Example, with one expert and `epsilon` 0.10:

| Posterior for `S` | Average | `pi` | Tier | `admission_risk` |
| --- | --- | --- | --- | --- |
| `Beta(1, 30)` | 3.2% | 0.958 | trusted | 0.052 |
| `Beta(1, 9)` | 10.0% | 0.613 | on trial | 0.164 |
| `Beta(4, 8)` | 33.3% | 0.019 | excluded | 0.445 |

Each cell, in `pick` and in `summary`, reports `pi`, `tier` (`trusted`, `on-trial` or
`excluded`), `admission_risk` and `admissible`. `admission_risk` is the
`admission_quantile` point of the same mixture, so `admission_risk <= epsilon` holds
exactly when `pi >= admission_quantile`. `admissible` is true exactly when the tier is
trusted.

Trusted and on-trial cells compete together:

```text
pool = { c : credit_ok(c) and c is trusted or on trial }                                   when not empty
pool = { c : credit_ok(c) and pi(c) >= all_excluded_factor * max pi over credit_ok cells }  otherwise
best = the pool cell with the lowest J
eligible(c) <=> c in pool and J(c) - J(best) <= uncertainty_multiplier * (sd(best) + sd(c))
chosen = the eligible cell with the lowest sampled J
```

- `credit_ok(c)`: the cell has enough credit (see the
  {ref}`credit budget <credit-budget>`).
- `all_excluded_factor`: 0.5 by default.
- `J`, `sd`, sampled `J`: the cell's mean cost, its `cost_sd` and its cost under the
  seed's draw (`sampled_J`).
- `uncertainty_multiplier`: 1 by default. A risk class's `explore_within` additionally
  caps the gap at `explore_within * |J(best)|`.

In plain words, trusted and on-trial cells compete as equals, and nothing limits how
often an on-trial cell is chosen. When no such cell has enough credit, the cells nearest
the best chance compete instead, so a cell is always drawn. Selection is refused only
when no cell has enough credit.

Example of the second pool: three cells with enough credit have `pi` 0.12, 0.08 and
0.05. The bar is `0.5 * 0.12 = 0.06`, so the first two form the pool and the third is
left out.

```text
cap_met      <=> the chosen cell is trusted
needs_review <=> not cap_met
```

In plain words, the flag says that the chosen cell is not trusted. It is advice to the
caller to have the result reviewed independently; nothing is enforced or recorded for
it.

Example: a chosen cell with `pi` 0.613 is on trial, so `cap_met` is false and
`needs_review` is true.

`exploration.pool` records which pool competed: `trusted-and-on-trial` or
`all-excluded`. `exploration.enabled` is true when more than one cell is eligible, and
`exploration.reason` is `sampled-within-posterior-overlap` (the draw chose a cell other
than the mean-best), `mean-best` or `disabled-no-supported-alternative` (one eligible
cell). The three constants are frozen with the rest of the policy in
`selection.sampling_policy`.

### The draw

A sample is drawn for every decision, so `semantic.sampling.enabled` is always true.
`pick` samples one expert by its weight and draws outcome probabilities from that
expert's per-cell Dirichlet posterior `alpha`. Forecasted `seconds` represents expected
**active** execution duration excluding external queue wait. `T` and `C_run` in `J` are
the averages of the cell's {ref}`beliefs <amount-beliefs>`, which are merged over the
experts and so do not depend on the sampled expert. `cost_sd` is the delta-method
retry-cost standard deviation: the posterior Dirichlet covariance
`(diag(m) - m m^T) / (sum(alpha) + 1)` plus the active-time term
`(lambda_per_hour / (1 - P_loud))^2 * E[T]^2 * (exp(s'^2) - 1)`, raised to the
between-expert spread of per-expert posterior costs when larger. Run cost is exact in
`cost_sd` and in the draw: its spread enters the credit check only. Thompson sampling
considers every cell of the pool whose posterior cost overlaps the pool's mean-best
cell: `J - J_best <= uncertainty_multiplier * (sd_best + sd_cell)`. Only an explicit
risk-class `explore_within` additionally caps that gap at `explore_within * |J_best|`.
Thompson sampling draws each cell's active time from `ln T ~ Normal(m', s'^2)` in
`parts["time"]`. Receipts retain the scored `evidence_candidates`, the
`outcome_observations` and `duration_observations` counted from them, each sampled cell's
`outcome_posterior` (`prior`, `prior_concentration`, `counts`, `evidence_weight`,
`alpha`), concentrations, each cell's beliefs, `pi`, `tier` and
`admission_risk`, the seed and the reason for a sampled non-best choice. `summary`
returns the same posterior means, intervals, beliefs and tiers without sampling a selection. A
fixed seed reproduces selection given frozen forecasts and the request's scored evidence
candidates; it does not make inference deterministic. A receipt records the time its
request was prepared, because the age of every counted run is measured to that moment,
so two decisions prepared at different times are different receipts even when nothing
else differs. Loading a stored decision recomputes `pi` from its frozen forecasts,
expert weights and outcome observations and refuses a receipt whose `pi`, `tier`,
`admissible`, `exploration.pool`, chosen cell or flags disagree with them.

Loud attempts are retried on the same cell until a non-loud result is accepted.
`1/(1-P_loud)` charges the expected attempts; `C_loud` is extra detection/rework overhead
per loud failure, in addition to the repeated time/money. The probability floor keeps
a near-certain loud failure finite but very costly. Retry-on-another-cell is not assumed.

Risk-class `explore_within` can further tighten the posterior-overlap band; it cannot
permit exploration outside it. Each cheaper alternative is explained in `why`:
`insufficient-credit`, `silent-cap` for a cell outside the pool (the message quotes its
`pi`), `explore-within` or `posterior-overlap` for a pool cell outside the band, and
otherwise `sampled-cost` with signed cost components.

New receipts use `howhard-decision-v11` (`howhard-semantic-v8`, reader prompt
`semantic-reader-v8`, interval kind `dirichlet-posterior-mixture`). Loading a v11
receipt recomputes every belief from its stored forecasts, expert weights,
`duration_observations` and sampling policy, then `T`, the three credit numbers,
`session_cost_usd` and `credit_ok`; a receipt whose stored values differ is refused.
Frozen `howhard-decision-v10` receipts use the same outcome posterior without a host
name, agent labels, ranges on active time and token kinds, tiers or beliefs: they keep the
per-expert active-time calibration they were made with (each expert's forecast worth one
run against the same-cell runs, log spread capped at 2, averaged over the experts), price
each expert's forecast directly, and check `needed_credit_usd <= available_credit_usd`.
Frozen
`howhard-decision-v9` receipts keep undecayed `kappa_0 * p_0 + counts` posteriors, while
frozen `howhard-decision-v8` and `howhard-decision-v7` receipts keep their original weighted
forecast envelopes, upper-envelope admission and sampling policy shape; all load,
validate, accept outcomes and replay into predictive weights. Receipts older than v11
keep the cap as they recorded it:

| Item | Formats v7 to v10 | Format v11 |
| --- | --- | --- |
| `cap_met` | Some admissible cell has enough credit | The chosen cell is trusted |
| No cell passes | No draw; `selection.on_cap_unmet` records the policy and `exploration.reason` is `disabled-cap-unmet` | A draw over the second pool |
| `semantic.sampling.enabled` | Equals `cap_met` | Always true |
| `exploration.enabled` | `cap_met` and more than one eligible cell | More than one eligible cell |
| Cell fields | `admissible`, and `admission_risk` from v9 | Also `pi` and `tier` |
| `selection.sampling_policy` | Without the two tier constants | With `trial_probability` and `all_excluded_factor` |
| Widening labels | Trigger `silent-cap-unmet`; stop `cap-met` | Trigger `all-excluded`; stop `pickable` |

## Prices and measurements

Prices belong to the host inventory. If any forecasted band/widened candidate is unpriced, worker money is
omitted uniformly from every candidate's objective. Known `session_cost_usd` remains visible;
unknown values stay null, and `cost_policy` labels the partial risk/time comparison.
Fully priced inventories with a uniform rate basis use the full objective. Candidate
`usd_per_mtok` and `rate_card` rates measure list-price quota credit; they do not
establish actual host money cost when that differs. `cost_per_credit` supplies that
conversion. Tokens include cached/uncached input and output; T is expected active hours.
Actual `tokens`, `seconds`, optional `active_seconds` and `queue_wait_seconds`, and
`cost_usd` remain null unless measured. When both `active_seconds` and `queue_wait_seconds`
are supplied and `seconds` is omitted, `seconds` is derived as their sum; when all three
are supplied, `abs((active_seconds + queue_wait_seconds) - seconds) <= 1e-6` is enforced.
`effective_active_seconds(row)` uses `active_seconds` when non-null and falls back to
scalar `seconds` otherwise. Observations update both the threshold reader and shared
predictive evidence from one stored row.

(credit-budget)=
An optional candidate `credit_budget` object enforces hard scheduling feasibility against
forecasted `needed_credit_usd`, independently of the session money cost. It requires
nonnegative finite `available_credit_usd` and accepts optional
`session_5h_remaining_credit_usd` and `weekly_7d_remaining_credit_usd` windows; when window fields are
present, `available_credit_usd` must match their minimum within `1e-6`. Any candidate supplying
`credit_budget` must also supply `rate_card` or `usd_per_mtok`, and an inventory that uses
`credit_budget` requires uniform pricing across all candidates so `needed_credit_usd` is
known on a single credit basis. Each evaluated cell reports `session_cost_usd`,
`cost_per_credit`, `needed_credit_usd`, `credit_variance`, `credit_p80_usd`,
`credit_budget`, `available_credit_usd` and `credit_ok`. With a `credit_budget`,
`credit_ok` is `max(needed_credit_usd, credit_p80_usd) <= available_credit_usd`: the run
must fit the remaining credit on average and four times out of five (see
{ref}`credit: average, spread and check <amount-beliefs>`). With no budget it is `true`.
`tier` and `admissible` describe accepted-result silent risk only. Only schedulable
cells (`credit_ok == true`) enter either pool of the
{ref}`admission tiers <admission-tiers>`, so an excluded cell or a cell without enough
credit is never chosen while a trusted or on-trial cell has enough credit. Selection is
refused when no evaluated cell is schedulable, and cheaper unschedulable alternatives
are labeled `insufficient-credit` in `why`. Both messages name the amount that was
checked: `credit_p80_usd` when it exceeds `needed_credit_usd`, otherwise
`needed_credit_usd`.

Recorded usage sharpens these beliefs. A run that records active time apart from waiting
updates the active-time belief with the time the cell actually worked, and a run with a
complete execution usage envelope updates the belief of every token kind it records. A
run that records only a total time or a total token count updates what it can.

(portable-execution-accounting)=
## Portable execution accounting

Execution accounting is evidence about an actual invocation, independent of its outcome.
An optional `execution_usage` envelope (`howhard-execution-usage-v1`) belongs to the
existing observation store; no second ledger is created. `usage collect --manifest FILE`
reads only explicitly owned artifacts with an explicit adapter version and execution
span, and accepts optional finite nonnegative `seconds`, `active_seconds`, and
`queue_wait_seconds` fields (deriving `seconds` when both breakdown fields are given and
reconciling within `1e-6` when all three are present, independently of token telemetry
completeness). It never discovers transcripts, retains prompt/code text, or contacts providers.
`usage attach EXECUTION_ID --manifest FILE --id AMENDMENT_ID` appends a measurement
correction after the outcome is durable. Failed collection leaves that outcome intact;
retry the same execution once with the same amendment ID, then report the unavailable
measurement and park collection. Later evidence can amend the same execution.

Tokens are nonoverlapping `input`, `output`, `cache_read`, `cache_write`,
`cache_write_5m`, and `cache_write_1h` categories. Unknown is null. Codex input includes
cached input, which is subtracted once; reasoning output is already included in output.
Claude input excludes cache reads/writes; cache creation TTL buckets partition total
cache creation. A known aggregate never supplies a guessed partition. Optional segment `reported_tokens`
retains a measured inclusive aggregate when category coverage is partial; effective scalar
`tokens` uses it only when every segment aggregate and the completed boundary are known.
An interrupted boundary never becomes a complete scalar total. Collector sources
retain host/schema version, content hash, coordinate and span/source identity, provider,
model and tier. Supported schemas are Codex `0.166.0`, Claude sessions
`2.1.318`, and Claude SDK artifacts from the verified underlying Claude Code runtime
`2.1.318`; unknown runtime versions are refused. An SDK package version does not
establish the runtime's cumulative accounting semantics and is not a supported
`host_version` identity. A manifest's optional `coordinate_id` preserves append-only artifact
identity across moves; different coordinate offsets cannot establish disjoint spans
for one session. Explicit Claude `agent_id` must match the sidechain's `agentId` and
parent session identity. For child-inclusive SDK results, `included_sessions` contains
only independent host session IDs. Native sidechains sharing the parent's session instead
use `included_agents: ["EXACT_AGENT_ID"]` under the manifest's verified `session_id`;
the canonical `PARENT_SESSION:agent:AGENT_ID` claim matches the sidechain import and
prevents duplicate attachment in either order. Agent IDs are never guessed from flat
session aliases. Both arrays require distinct IDs and inclusive scope when nonempty.
Claude SDK cumulative results require a same-session pre-invocation `modelUsage` baseline
(an empty map explicitly means a fresh session), result UUID and declared child identities.
Missing SDK baseline is unavailable; counter resets are refused. A resumed Codex session
span requires raw measured starting cumulative usage in `baseline`; missing baseline
or unreconciled cumulative/last-frame deltas keeps it partial. Explicit event identities
and exclusive execution spans prevent inherited/forked usage
from becoming new evidence. Child-inclusive result scopes cannot also be counted as
separate executions. Duplicate source identities across effective original runs are
refused. Clearing or replacing a wrongly attributed effective claim permits attachment
to its truthful owner; raw history remains intact. Amendments replace usage, never add
another execution.

Accounting completeness is independent of task outcome. Claude SDK `success`,
`error_max_turns`, and `error_max_budget_usd` results have a bounded terminal usage
boundary when `is_error` agrees with the subtype. A bounded failed task can therefore
have complete measured usage. Contradictory or untyped error flags are refused;
uncertain execution/crash results remain partial. Source provenance and its primary
ownership claim must agree, and a changed source during collection is refused.

`estimated_cost` records a client estimate or rate-card calculation with its source.
A client estimate may be known at a completed source boundary while token categories
remain partial. Rate-card calculation still requires the category measurements it
prices; an incomplete boundary cannot claim a complete client estimate.
`billed_cost` requires an explicitly authoritative billing record tied to the execution,
with `authoritative: true`, `execution_id`, `receipt_id`, source, version and USD amount;
client `total_cost_usd` and telemetry cost metrics are estimates. When an SDK aggregate
and all per-model costs are known, they must reconcile; a mismatch is an unsupported
cost scope and is refused. No unexplained fee is silently omitted or invented. Authoritative billing
is independent of token coverage and source boundary completeness: a bound unique
receipt remains valid with partial or unknown tokens. Billed report totals are known
when every segment has its authoritative receipt; partial tokens alone do not make a
known bill unknown. Subscription execution
may have no per-run billed charge. Measured tokens do not require prices. Historical
`cost_usd` retains an unspecified basis and is never relabeled billed. Reports show these
bases separately, summarize `seconds`, `active_seconds`, `queue_wait_seconds`, and
`effective_active_seconds` coverage in `worker_time` and `execution_accounting.duration`,
and preserve unknown coverage instead of treating missing values as zero.

Native candidates may provide an explicit versioned `rate_card` with source, effective
date, provider/model/tier, credit-rate basis and category rates in USD per million tokens.
These rates represent raw public API list-price quota credit. They must come from a known
applicable source; there are no built-in price assumptions. `cost_per_credit` converts the
resulting credit burn to actual session cost.
Category forecasts use `token_usage` alongside their consistent scalar `tokens` total;
a rate-card candidate requires the category forecast. Legacy blended `usd_per_mtok`
remains a total-token forecast with unspecified credit basis. Comparisons omit worker
money uniformly when rates are unknown or their bases differ. Current request, receipt
and semantic format versions exclude older predictors from replay while retaining raw
historical evidence.

## Records and amendments

Every JSONL row requires these base keys:

| Key | Value |
| --- | --- |
| `id` | Nonempty string, unique across logs and journal |
| `date` | `YYYY-MM-DD` |
| `ticket` | String or null |
| `task_type` | Known task name; zero-config also accepts new names as standard |
| `risk_class` | Recorded risk class |
| `model` | Concrete id from the registry |
| `effort` | Exact registered effort/control identity |
| `observed_at` | ISO timestamp with timezone |
| `outcome` | `clean`, `loud`, `silent` or `interrupted` |
| `cause` | Null for `clean`; `depth`, `capability` or `unrecorded` for `loud`/`silent`; `usage_limit`, `provider_error`, `host_error`, `cancelled` or `unrecorded` for `interrupted` |
| `tokens` | Nonnegative integer or null |
| `seconds`, `cost_usd` | Finite nonnegative number or null |
| `active_seconds`, `queue_wait_seconds` | Optional finite nonnegative number or null; reconciled with `seconds` within `1e-6` |
| `note` | String, possibly empty |
| `amends` | Earlier row id or null |

Types, enums, missing keys and extra keys are strict. `observe --id` matching retries
return the original row without appending. Automatically derived context, timestamp/date,
and risk class retain their first historical values; explicitly changed
inputs conflict. Equality never bypasses field/type validation. Raw `observe --native`
can admit a verified actual model/control without a receipt or retrospective forecast;
all feedback/ID validation precedes identity metadata persistence under the project lock.
For native worker feedback, an accepted explicit model/effort override followed by
completed execution is sufficient to match the requested cell when the host API applies
those controls and reports no launch failure, ignored controls, substitution or
contradictory execution metadata. Use the decision receipt so this feedback scores its
prospective forecast; record accepted override and completion in `note` if the host
does not return a separate resolved identity. This is host-API provenance, not backend
or provider attestation. Without accepted explicit controls, match only when reported
actual model/control exactly matches the receipt. A different known actual cell uses raw
native feedback without the unexecuted receipt. When neither accepted overrides nor an
exact reported actual cell establishes attribution, unknown actual identity/control
remains unscored. Recommendations and accepted but unexecuted requests are not evidence.
Disk duplicate rows remain errors. Errors name the file, line and key.
Original rows require `task_text`, `project_context` and `context_encoder` together,
with optional `decision_id`. Empty task text or null project context represents an
explicitly unknown input; missing fields are refused. Every row requires `observed_at`.
Amendments carry null measurements/cost and correct an outcome to `clean`, `loud`, `silent` or `interrupted`,
with the usual cause rule. Effective `interrupted` runs represent operational interruptions (such as usage limits,
provider/host errors, or cancellations) rather than model task failures: they retain their original ID, decision,
usage/cost/duration accounting, amendment chains and `report` counts (including `interrupted_causes`), while being
excluded from stage-1/stage-2 predictive belief replay, capability threshold likelihoods, active-duration
calibration, forecast error metrics, and `filter_quality` failure/disagreement resolution. Amending a misclassified
`loud` execution to `interrupted` removes its predictive and threshold penalty on replay. Superseded rows without timestamps/context and unresolved
`model_default` efforts are refused; no context or control is inferred retrospectively.
Chains replace the original run's effective outcome and cause, counting that run once
in original chronological journal order and retaining its original ID, decision, cell,
task text, project snapshot, project, weight and uncorrected measurements. Raw rows remain intact. Amendments cannot override
historical context through the observation API. Targets within a file must precede their amendments.
Across files, targets must exist, have dates no later than their amendments, and form an
acyclic dependency graph. Dependency ordering keeps amendments valid when flushing moves
them into the first log ahead of a target in an additional log.

An amendment may additionally carry `measurement_corrections`, a nonempty object with
only `tokens`, `seconds`, `active_seconds`, `queue_wait_seconds`, `cost_usd` and/or `execution_usage`. Values follow the corresponding base-field
types and ranges; null explicitly clears a value to unknown. Supplying both `active_seconds`
and `queue_wait_seconds` without `seconds` updates `seconds` to their sum, and effective
rows must still satisfy the `1e-6` duration breakdown constraint. Original rows refuse this
field, and ordinary amendment `tokens`, `seconds`, `active_seconds`, `queue_wait_seconds` and `cost_usd` must still be null.
For example, `Engine.observe(..., amends="run-id", measurement_corrections={"tokens": 100,
"cost_usd": None})` replaces only those effective values. Unlisted measurements retain
their current effective values across a chain. Supply the intended current outcome/cause
as usual, including when only usage changes. No forecast is created retrospectively.
Worker-cost totals and predictive memory evidence use the corrected original
once. Writer `observation` contains current effective usage; `outcome_history` preserves
the raw original and correction maps as an audit trail, never extra runs or additive usage.
Matching correction-ID retries remain idempotent and different maps conflict. The usage envelope is replaced as a whole; `{"execution_usage": null}` clears it.
Readers without this field's support refuse such rows.

`flush` validates all files before writing. Duplicate refusals list every duplicate and
leave log and journal unchanged. It appends, fsyncs the log, then truncates and fsyncs the
journal. Backend-owned [FIFO write admission](concurrency.md) serializes mutations per
store with bounded waiting, automatic release and observable queue status. Data locks
preserve consistent reads without making readers join the write queue. A crash
between append and truncate can leave duplicate rows, which are refused on the next read;
there is no claim of an atomic transaction across two files. Preserve a copy and remove
the already-flushed journal entries after comparing their ids with the log.

## Capture root and durable state

The current project capture root is independent of the durable state root. `paths --json`
reports both; `--state-dir PATH` and `Engine(state_directory=...)` support intentional
store selection. Fresh zero-config Git worktrees share `howhard` below the repository's
common Git directory, with current context from the current checkout. Checkout-local
`.howhard` directories do not override this shared default and are left untouched.
Explicit configuration retains its declared paths and policies; plain directories
keep local `.howhard`. A stable random project identity lives
in `howhard-project.json` in the Git common directory, or `.howhard/project.json` for a
plain folder, independently of a custom store. Moving the project with its metadata
preserves identity. A claimed store refuses another project's use before cache, registry
or memory writes. All receipts require that identity and their native inventory/cost
policy. The resolved state subtree is excluded from capture and
`project.read`, including symlink aliases; a state root equal to or containing the capture
root is refused. Prepared calling-agent requests bind the worktree and reject sibling
retrieve/decide. Requests use `howhard-agent-request-v7` and new receipts use
`howhard-decision-v11` (`howhard-semantic-v8` / `semantic-reader-v8`) with required
`host`, capability and accepted-result selection state. Newly aggregated capability state uses
`howhard-capability-v6`, whose cells each carry their `agent` label. Its `band` is an ordered list of fit cells that can omit
`intermediate` cells between its `lower` and `upper` endpoints. It also stores
`caught_share` (`host`, `F_all`, `C_all`, `F_host`, `C_host`, `d_0`, `d_all`, `d_host`
and the two band edges `e` and `g`), the `core` (the band's first
`max_forecast_cells - reserved_forecast_cells` cells), `proven_families` (the families
with a run among the evidence candidates) and `reserved` (each reserved cell's `key`,
its `reason`, `unproven-family` or `missing-agent`, and its `bar`). The cells that
carry forecasts are the core, the reserved cells, the band cells that took unfilled
slots and the widening cells. Loading recomputes the caught shares, the edges, the
band, the core, the reserved cells and the widening order from the stored numbers and
refuses a receipt that does not follow from them. See
[capability filtering](capability.md) for the band, reserved-slot and widening rules.
`howhard-decision-v11` is open: its shape can still change before the release that
freezes it, intermediate shapes are not kept readable, and requests prepared under
`howhard-agent-request-v6` are refused, so prepare a new request. Persisted
`howhard-decision-v10` (`howhard-semantic-v7` / `semantic-reader-v7`,
`howhard-capability-v5` or `howhard-capability-v4`),
`howhard-decision-v9` (`howhard-semantic-v6` / `semantic-reader-v7`,
`howhard-capability-v4` or `howhard-capability-v3`), `howhard-decision-v8`
(`howhard-semantic-v5` / `semantic-reader-v6`, `howhard-capability-v3`), and
`howhard-decision-v7` (`howhard-semantic-v4` / `semantic-reader-v5`,
`howhard-capability-v1` or `howhard-capability-v2`) continue to load, validate, accept
feedback, and participate in belief replay and reporting under their frozen version rules.
Receipts older than v11 carry no host name and no agent labels; they read as the one
unnamed host and the one unnamed agent, and they do not accept those fields or the
ranges on active time and token kinds.
Receipts v1–v6 remain recognizable as historical store entries but are not replayed as
new decisions; prepare a fresh request and decision to attach new feedback.
Frozen-receipt observations and learning share the store across that project's worktrees.

### Request, decision and prototype storage

`payloads.sqlite3` in the state root holds content-addressed `requests` and `decisions` rows
alongside dynamic synthesized `(task, context)` prototype tables (`prototypes`,
`prototype_members`, `prototype_cells`, `prototype_meta`, `prototype_row_state`).
Prototype tables maintain reusable representative `(task, context)` clusters, per-cell
cold-start priors (`prior_json`), and incrementally updated / amendment-replayable
empirical sufficient statistics (`stats_json`), splitting heterogeneous clusters once at
least 12 non-`interrupted` member runs diverge semantically or empirically.
A forecast stored on a prototype keeps its values, every stated range and, as `host`,
the name of the host whose request produced it (`null` for the unnamed host and for
forecasts stored before hosts were named).
`howhard.prototypes.stored_forecasts(config, task_type=...)` returns them. Reuse does
not depend on the host: when exactly one prototype matches, its stored ranges return
with the reused values, and a range is widened only as far as floating-point rounding
of the reused value requires; values averaged over several matched prototypes carry no
range.
Each request or decision row retains the complete original JSON byte sequence in a zstd-compressed `payload`
BLOB; the SHA-256 `request_id` or `decision_id` remains the digest of the parsed JSON
value. No semantic forecast, legacy receipt field, or reproducibility input is omitted.
If two bytewise JSON variants have the same content ID, the prior exact bytes remain
in `payload_variants` when the main row takes the migrated bytes.
The `decisions` table also stores uncompressed `created_at`, `chosen`,
`context_summary`, `selection`, `task_type`, `risk_class`, `cap_met`, and
`needs_review` columns. `chosen`, `context_summary`, and `selection` are small JSON
summaries; `context_summary` includes task text and project fingerprint, while
`selection` includes objective, retry policy and, for receipts older than v11, the
recorded cap-unmet policy. Indexes cover creation
time, task/risk, chosen cell, and review status. The helper
`howhard.decisions.summaries(config, task_type=...,
risk_class=..., cap_met=..., needs_review=..., limit=...)` reads them without decompressing receipts.
Load a full receipt by ID when the complete context or selection is needed.

Runtime commands and Python APIs read and write requests and decision receipts
exclusively in `payloads.sqlite3`; no automatic migration or loose-JSON fallback occurs
at runtime. To migrate pre-4.0 loose `requests/*.json` and `decisions/*.json` files in a
claimed store offline, run `python scripts/migrate-payloads.py` (with optional
`--config PATH`, `--directory PATH`, `--state-dir PATH`, `--write-timeout SEC`,
`--batch-size N`, and `--json`). The standalone script verifies store ownership, checks
each file's content-addressed SHA-256 ID, commits zstd-compressed exact bytes in bounded
batches under normal write admission, verifies pre- and post-commit round-trip
decompression before unlinking each loose file, leaves unverified files untouched on
error, and removes drained legacy directories. External receipt files and
`--save-decision PATH` continue to use ordinary JSON files.

## Inference cache and current interfaces

Validated LLM responses may be cached by complete request, adapter identity, settings,
evidence and memory revision. Corrupt caches are replaced. Prepared requests bind the
current worktree, native inventory and a consistent store snapshot. New requests validate
only surfaced memory/evidence dependencies and configuration, with belief and capability
weights frozen at preparation. Unrelated writes do not invalidate them. A real change
returns a typed conflict and refreshed request; see [concurrent clients](concurrency.md).
Calling-agent submissions are audited with unknown external
inference costs and latency rather than being counted as free provider calls.

Decision v1–v6 receipts and older requests are refused for new execution feedback
and excluded from replay and report scoring. Existing raw observations
and semantic memories remain stored. Old posterior/prior artifacts are unused and left
untouched. Removed interfaces include `--llm`, `HOWHARD_LLM`, `--cache`, `Engine(cache_path=...)`,
`fit`, posterior/prior attributes, `export-prior`, `import-prior`, and prior/cache/transfer
configuration. No migration of old posterior/prior artifacts or distribution upload is
performed automatically.

## Python API

```python
from howhard import Engine

engine = Engine()  # State, context, retrieval, warm prototypes and feedback need no provider.
request = engine.prepare("specified-implementation", "Make rollback safe",
                         native_candidates=[{"model": "actual-host-model", "effort": "high"}])
# When baseline_request is non-null, evaluate it in an independent model context;
# pass auto_decide=True to complete warm memory-free routes in a single call.
# choice = engine.decide(request["request_id"], response, seed=42)
# engine.observe_decision(choice["decision_id"], "clean", id="actual-execution")
```

`Engine(llm=adapter)` accepts an object with `complete(request) -> dict`. Standalone
`pick` and `summary` complete locally with zero LLM calls when matched prototypes cover
all cells and no active editable memories exist, and lazily resolve `HOWHARD_LLM_COMMAND`
or an authenticated Claude CLI only when cold-start baselines or active-memory reader
calls are required. Calling-agent `prepare(..., auto_decide=False|True)` / `retrieve` /
`decide` never invoke a provider. A fresh engine's
observations return pending learning; apply an external writer response or call
`memory_learn` explicitly. An engine already holding a provider adapter runs its writer
after durable observation. `refresh()` reloads external evidence; no fitting occurs.
`Engine(write_timeout=30, write_cancel=callback)` bounds admission and supports
cancellation. `write_status()` and CLI `writes --json` expose contention. See
[concurrent clients](concurrency.md) for boundaries, typed errors and stable-ID retries.
Python 3.11+ and POSIX are required; runtime dependencies include NumPy, FastEmbed,
USearch, and zstandard. SQLite is provided by Python's standard library.
