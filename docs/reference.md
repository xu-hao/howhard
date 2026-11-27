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
Cold-start stateless LLM calls evaluate capability at most once per unseen
`(prototype, model, effort)` cell, and fine forecasts at most once per unseen
`(prototype, model, effort, host)`, against the prototype's synthesized
`(representative_task, representative_context)`, materializing priors, stored forecasts
and empirical cell statistics in `payloads.sqlite3`. Stage 1 priors are shared by all
hosts. A Stage 2 forecast is {ref}`reused only under the host that made it
<stored-forecasts>`, and the forecasts of several matched prototypes are merged on each
value's own scale. When matched prototypes already cover all cells under the request's
host and no active editable memories exist, `pick`, `summary`, and
`context --auto-decide` complete locally without invoking a provider; when some cells
have no usable stored forecast, Stage 2 requests forecasts only for those cells. Stage 2
never builds a forecast from recorded runs.
The reader supplies conditional overrides for memories it has seen; omitted cells and
unretrieved memories use the baseline forecast. No learned probability table
or synthetic forecast is supplied to isolated baseline calls. Cold-start provider absence, invalid forecasts
and provider failures are explicit errors; none produces a routing recommendation.

(outcome-posterior)=
### Per-cell outcome beliefs

A `(family, effort)` cell holds two independent beliefs, each on the log-odds scale
`logit(p) = ln(p / (1 - p))`:

- one for its loud rate `theta_l`, the share of attempts that fail loudly;
- one for its accepted-defect share `S`, silent results / accepted results.

Each belief starts from the forecast and is reshaped by the cell's own counted runs.

**One forecast per rate.** The experts' forecasts for a cell are merged first:

```text
g_l = loud                     g_S = silent / (clean + silent)
m_k = logit(g_k)               s_k = (logit(hi_k) - logit(lo_k)) / 3.92
m   = sum_k w_k m_k            s_stated^2 = sum_k w_k (s_k^2 + (m_k - m)^2)
p   = logistic(m), held in [0.0001, 0.9998]
```

- `clean`, `loud`, `silent`: one expert's forecast for the cell, floored at the
  probability floor 0.0001 and normalized. `g_l` and `g_S` are its guesses for `theta_l`
  and `S`.
- `k`, `w_k`: an expert in `semantic.beliefs.experts` and its weight. The weights sum
  to 1.
- `lo_k`, `hi_k`: the ends of the expert's stated 95% range, held in
  `[0.0001, 0.9999]`: `loud_ci` for `theta_l`, and `silent_ci` divided by
  `clean + silent` for `S`.
- `s_k`: the stated spread on the log-odds scale; 0 when no range is stated.
- `m`, `s_stated`, `p`: the merged centre, the merged spread and the prediction.
- `logistic(x) = 1 / (1 + exp(-x))`, the inverse of `logit`.

In plain words, each rate gets one forecast per cell, and experts that disagree make it
less certain.

Example: two experts with weights 0.75 and 0.25 guess `S` at 10% and 25%, each with
spread 0.4. Then `m = 0.75 * (-2.197) + 0.25 * (-1.099) = -1.92`, `p = 12.8%` and
`s_stated = sqrt(0.16 + 0.75 * 0.275^2 + 0.25 * 0.824^2) = 0.62`.

**The start of each belief.**

```text
N_start = 1 / (1/worth + 1/N_task)       sigma_1 = 1 / sqrt(p (1 - p))
s = min(1.869, max(s_stated, sigma_1 / sqrt(N_start)))
x_j = -16 + 0.01 j,  j = 0 .. 2500
q_j(m) = exp(-(x_j - m)^2 / (2 s^2)) / sum_i exp(-(x_i - m)^2 / (2 s^2))
centre: the m with  sum_j q_j(m) logistic(x_j) = p
```

- `worth`: how many runs the record says about this family, `sampling.model_worth`
  (5).
- `N_task`: the most a record on other tasks is worth for one task,
  `sampling.task_worth_silent` (32) for `S` and `sampling.task_worth_loud` (16) for
  `theta_l`.
- `N_start`: how many runs the prediction is worth: 4.32 for `S` and 3.81 for
  `theta_l`.
- `sigma_1`: the spread of one run's result on the log-odds scale.
- `s`: the spread at the start. A stated range can widen it, never narrow it. 1.869 is
  the flat belief, the spread of a 95% range of `[0.025, 0.975]`.
- `x_j`, `q_j(m)`: the grid of 2,501 log-odds points, the same for every belief, and a
  bell curve of centre `m` and spread `s` on it.
- centre: found by 60 halvings of `[-25, 25]`, keeping the upper half when the grid
  average at the midpoint is below `p`. The grid's top value is `logistic(9) = 0.99988`,
  so every prediction up to 0.9998 can be met.

In plain words, the start is a bell curve on the log-odds scale whose average, not its
median, equals the prediction, and it is as wide as a record worth `N_start` runs
allows.

Example: clean 0.72, loud 0.20, silent 0.08 with `silent_ci = [0.04, 0.16]` gives
`p = 0.10` for `S` and `s_stated = 0.40` (a range of 5% to 20%).
`sigma_1 / sqrt(N_start) = 3.33 / 2.08 = 1.60`, so `s = 1.60` and the centre is -3.04.
For `theta_l`, `p = 0.20` and `loud_ci = [0.10, 0.35]` give `s = 1.28` and centre -1.79.

**Counted runs reshape the belief.**

```text
b_j = q_j(m) * logistic(x_j)^n_yes * (1 - logistic(x_j))^n_no,  rescaled so that sum_j b_j = 1
S:        n_yes = n_s,  n_no = n_c
theta_l:  n_yes = n_l,  n_no = n_c + n_s

E[S] = sum_j b_j logistic(x_j)
pi(c) = sum of b_j over the points with logistic(x_j) <= epsilon
E[1 / (1 - theta_l)] = sum_j b_j (1 + exp(x_j))
E[theta_l / (1 - theta_l)] = sum_j b_j exp(x_j)
q point = logistic(x_j) at the first j whose running total of b reaches q
```

- `n_c`, `n_l`, `n_s`: the counted clean, loud and silent weight of the cell (below).
- `b_j`: the belief after the counted runs, computed in logarithms.
- `E[S]`, `pi(c)`: the average of the belief of `S` and its share at or below
  `epsilon`, the risk class's silent limit ({ref}`admission tiers <admission-tiers>`).
- `E[1 / (1 - theta_l)]`, `E[theta_l / (1 - theta_l)]`: expected attempts, and expected
  loud failures per accepted result.
- q point: the 2.5% and 97.5% points of each rate, reported as `lower` and `upper`.

In plain words, the start counts as `N_start` runs and each counted run as its weight.
The forecast never fades out: however many runs are counted, it keeps the weight of
`N_start` runs. A never-observed cell keeps its start whatever other cells have
recorded.

Example, continuing with `epsilon = 0.10`: at the start `E[S] = 0.100`, `pi(c) = 0.70`
and the 95% range is 0.2% to 52%. With `n_c = 4` and `n_s = 0`, `E[S] = 0.053` and
`pi(c) = 0.85`. For `theta_l` at the start, expected attempts are 1.38, against 1.25 at
the average rate.

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
matches no prototype counts nothing. An expert's global `evidence_count` affects only
its predictive weight, never a cell's counted runs.

A forecast row may add `silent_ci` and/or `loud_ci` as `[lo, hi]` with
`0 <= lo <= value <= hi <= 1`, a central 95% range for that probability. It enters as
the stated spread `s_k` above: a wide range widens the start, up to the flat belief,
and a narrow one cannot make the start narrower than a record worth `N_start` runs
allows. Bare `clean`/`loud`/`silent` rows remain valid and state no spread.

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
  Selection reads such a range as the stated spread above, which stops at the flat
  belief's 1.869.
- A token kind forecast at zero is exactly zero unless a non-zero upper end is stated.
- `token_usage_ci` needs `token_usage` on the same row and accepts only the six kinds.
  There is no range on total `tokens` or on credit: token counts do not depend on
  prices, and a stored forecast is reused after prices change.

Stated ranges are stored unchanged in the receipt's forecast panels
(`semantic.worlds`). `loud_ci` and `silent_ci` are read as described above.
`seconds_ci` and `token_usage_ci` are read into the beliefs below: a stated range can
widen a belief and never narrow it.

(stored-forecasts)=
### Stored forecasts: reuse and merge

A stored forecast is the forecaster's memory-free row for one cell, made against a
prototype's representative task and context. A prototype keeps one stored forecast per
cell and host. The host is the name the request carried; a request with no name belongs
to the one unnamed host.

- **Store.** When the forecaster answers for a cell, its row is stored on the request's
  primary prototype (the matched prototype with the largest match weight, or the
  prototype the request creates) under the request's host. It replaces that host's
  earlier forecast for the cell whole, ranges included, and leaves every other host's
  forecast as it is. A stored forecast names no decision, because it is stored when the
  forecaster answers, before a decision exists.
- **Usable.** A stored forecast is usable for a cell of a new request when its prototype
  matches the task, it was made under the request's host, and it states every value the
  cell needs: `clean`, `loud`, `silent`, `seconds` and `tokens`, and also the six token
  kinds when the cell is priced by kind. A request with no host name reuses only
  forecasts stored with no host name, which includes every forecast stored before hosts
  were named. A request with a host name never reuses those.
- **Reuse.** The usable forecasts of the matched prototypes are merged into the cell's
  memory-free row, which stands in place of a forecaster's answer. The cell is left out
  of `baseline_request`.
- **Nothing usable.** The cell goes to the forecaster. No forecast, and no single value
  of one, is built from a prototype's recorded runs, and no stand-in forecast exists. A
  record with runs and no stored forecast therefore asks the forecaster once per cell
  and host. The recorded runs still count as evidence beside the forecaster's row.

Each value of the row is merged on its own scale:

```text
P       = the matched prototypes with a usable stored forecast for the cell
omega_p = w_p / (sum of w_q over q in P)
m       = sum over p in P of omega_p * m_p
s^2     = sum over p in P of omega_p * (s_p^2 + (m_p - m)^2)
```

- `w_p`: prototype `p`'s match weight for the new task. The weights of all matched
  prototypes sum to 1.
- `omega_p`: `p`'s share within `P`. The shares sum to 1.
- `m_p`: the stored value's centre on its own scale, read as the
  {ref}`outcome beliefs <outcome-posterior>` and the
  {ref}`amount beliefs <amount-beliefs>` read a forecast row: `logit(loud)` for the loud
  rate, `logit(silent / (clean + silent))` for the accepted-defect share `S`,
  `ln(seconds)` for active time and `ln(tokens_a)` for token kind `a`, where
  `logit(p) = ln(p / (1 - p))`.
- `s_p`: the stored value's stated spread, `(phi(hi) - phi(lo)) / 3.92` for a stored
  range `[lo, hi]` on the same scale `phi`, or 0 when no range is stored.
- `m`, `s`: the merged centre and spread.

In plain words: the merged guess is the share-weighted average on the value's scale.
The merged spread holds each forecast's own spread plus the distance between the
forecasts, so forecasts that disagree widen the range and equal forecasts keep it.

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

Three rules complete the merge:

- One prototype in `P`: its stored forecast is reused unchanged.
- A token kind that every forecast in `P` states as exactly zero (zero, with no non-zero
  upper end) stays exactly zero. Any other kind is merged over the forecasts that state
  it above zero, and the result is multiplied by their total share. For example, with
  shares 0.7 and 0.3, a kind stored as 500 by the first prototype and as exactly zero by
  the second is `0.7 * 500 = 350`.
- `tokens` is the sum of the merged kinds for a cell priced by kind. For any other cell
  `tokens` is merged itself, by the same rule as one kind, and has no range.

The merged row states:

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
- `range`: the value's stated range (`loud_ci`, `silent_ci`, `seconds_ci` or the kind's
  entry in `token_usage_ci`). The silent range is the range of `S` times `1 - loud`,
  and a kind's range is multiplied by `k_a` like its value. A value with `s = 0` has no
  range; a spread of at most `1e-9`, which is rounding in the average, counts as 0.

In plain words: the row states the merged guesses, each with the range from which the
beliefs read the merged centre and spread back, exactly as from a forecaster's row.

For example, the loud merge above gives `loud = 0.251` with the range
`[logistic(-1.092 - 1.96 * 0.672), logistic(-1.092 + 1.96 * 0.672)] = [0.082, 0.556]`.
With a merged `S` of 10% the row is `clean = 0.749 * 0.9 = 0.674`, `loud = 0.251` and
`silent = 0.749 * 0.1 = 0.075`.

The receipt lists every reused cell in `semantic.stored_forecasts`: the cell's `family`
and `effort`; its `sources`, each with the `prototype_id`, the match `weight`, the
`share`, the `host` and the stored `forecast` row; and `merged`, the centre and spread
of `loud`, `silent` (the share `S`), `seconds`, and either `tokens` or each kind under
`token_usage`, where a token entry also carries its total `share` `k_a`. A cell the
forecaster answered has no entry. Loading a receipt recomputes the shares from the
match weights, and the merged values and the memory-free row from the source rows. It
refuses a receipt in which any of them differs, and one with a source stored under a
host other than the receipt's own. A calling agent that returns the full panel must
repeat each reused row unchanged.

(amount-beliefs)=
### Beliefs for active time and tokens

Active time and each token kind have one belief per cell, held on the log scale. A
belief starts at the forecast, is as wide as the forecast is unsure, and is updated by
the cell's own recorded runs. The two rates have the {ref}`outcome beliefs <outcome-posterior>`.

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

`P_loud` and `P_silent_accepted` report `{mean, lower, upper}` from the two beliefs:
the average and the 2.5% and 97.5% points of `theta_l` and of `S`. `P_clean` and
`P_silent` carry a `mean` only, because neither has a belief of its own:

```text
P_loud = E[theta_l]    P_silent = (1 - E[theta_l]) * E[S]    P_clean = (1 - E[theta_l]) * (1 - E[S])
```

For example, `E[theta_l] = 0.20` and `E[S] = 0.10` give `P_loud = 0.20`,
`P_silent = 0.08` and `P_clean = 0.72`. `J` is the cost formula above at `P_loud` and
`P_silent`. There is no additional statistical risk floor, and a belief is a model
assumption, not a calibrated coverage guarantee.

(admission-tiers)=
### Admission tiers

Every forecast cell is sorted by its chance of meeting the risk class's limit on
silently wrong accepted results:

```text
pi(c) = P(S <= epsilon) = sum of b_j over the points with logistic(x_j) <= epsilon
```

- `c`: a forecast cell.
- `S`: the accepted-defect share, silent results / accepted results.
- `epsilon`: the risk class's limit on `S`.
- `b_j`, `x_j`: the cell's {ref}`belief <outcome-posterior>` for `S` and the grid of
  log-odds points it lives on.
- `pi(c)`: the share of that belief at or below `epsilon`.

In plain words, `pi` is the chance, under the cell's current belief, that no more than
`epsilon` of its accepted results are silently wrong.

Example: `epsilon` is 0.10. A cell forecast at clean 0.72, loud 0.20, silent 0.08 has a
belief for `S` with average 0.100, and 70% of it lies at or below 0.10, so `pi = 0.70`.
Four counted clean runs move the average to 0.053 and `pi` to 0.85.

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

Example, with one expert, no stated range, no counted run and `epsilon` 0.10:

| Forecast of `S` | `pi` | Tier | `admission_risk` |
| --- | --- | --- | --- |
| 2% | 0.957 | trusted | 0.021 |
| 10% | 0.702 | on trial | 0.156 |
| 50% | 0.011 | excluded | 0.692 |

Each cell, in `pick` and in `summary`, reports `pi`, `tier` (`trusted`, `on-trial` or
`excluded`), `admission_risk` and `admissible`. `admission_risk` is the
`admission_quantile` point of the same belief, so `admission_risk <= epsilon` holds
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
`pick` samples one expert by its weight. Then each cell, in order, takes one uniform
number for `theta_l` and one for `S`, each read at the first grid point whose running
total of the belief reaches it, and then its active-time draw:

```text
sampled J = run / (1 - theta_l) + C_loud * theta_l / (1 - theta_l) + C_silent * S
            + lambda_per_hour * hours / (1 - theta_l)
```

- `theta_l`, `S`: the sampled loud rate and accepted-defect share of the cell.
- `run`: the cell's run cost, the average of its {ref}`belief <amount-beliefs>`.
- `hours`: the sampled active time of the cell.
- `C_loud`, `C_silent`, `lambda_per_hour`: the risk class's costs.

For example, `run = 2`, one hour at 25 dollars, `C_loud = 50`, `C_silent = 1000`,
`theta_l = 0.20` and `S = 0.10` give `2.5 + 12.5 + 100 + 31.25 = 146.25`.

Forecasted `seconds` represents expected **active** execution duration excluding
external queue wait. `T` and `C_run` in `J` are the averages of the cell's
{ref}`beliefs <amount-beliefs>`. Every value of a cell, its two rates, its active time
and its run cost, is one belief merged over the experts, so none depends on the sampled
expert, and experts that disagree widen the beliefs themselves. `cost_sd` is the
delta-method spread of the retry cost. Its part from the two rates is

```text
Var_rates(J) = ((C_loud + J - C_silent * E[S]) / (1 - P_loud))^2 * Var(theta_l) + C_silent^2 * Var(S)
```

- `J`: the cell's mean cost.
- `Var(theta_l)`, `Var(S)`: the variances of the two beliefs on the grid. The beliefs
  are independent, so their parts add.

For example, `J = 146.25`, `C_loud = 50`, `C_silent = 1000`, `P_loud = 0.20`,
`E[S] = 0.10`, `Var(theta_l) = 0.0314` and `Var(S) = 0.0188` give
`120.3^2 * 0.0314 + 1000^2 * 0.0188`, about 19,300: a spread of 139. The active-time
term `(lambda_per_hour / (1 - P_loud))^2 * E[T]^2 * (exp(s'^2) - 1)` is added. Run cost
is exact in `cost_sd` and in the draw: its spread enters the credit check only. Thompson sampling
considers every cell of the pool whose posterior cost overlaps the pool's mean-best
cell: `J - J_best <= uncertainty_multiplier * (sd_best + sd_cell)`. Only an explicit
risk-class `explore_within` additionally caps that gap at `explore_within * |J_best|`.
Thompson sampling draws each cell's active time from `ln T ~ Normal(m', s'^2)` in
`parts["time"]`. Receipts retain the scored `evidence_candidates`, the
`outcome_observations` and `duration_observations` counted from them, each sampled cell's
`outcome_beliefs`, each cell's beliefs for active time and tokens, `pi`, `tier` and
`admission_risk`, the seed and the reason for a sampled non-best choice.
`outcome_beliefs.loud` and `outcome_beliefs.silent` each hold `prediction`,
`stated_spread`, `n_start`, `spread`, `centre`, `counts` (`[n_yes, n_no]`), `mean`,
`lower` and `upper`; `silent` adds `pi`, and `loud` adds `attempts` and `odds`, the two
expectations above. `summary`
returns the same means, ranges, beliefs and tiers without sampling a selection. A
fixed seed reproduces selection given frozen forecasts and the request's scored evidence
candidates; it does not make inference deterministic. A receipt records the time its
request was prepared, because the age of every counted run is measured to that moment,
so two decisions prepared at different times are different receipts even when nothing
else differs. Loading a stored decision recomputes both beliefs of every forecast cell
from its frozen forecasts, expert weights, outcome observations and policy, and refuses
a receipt in which a belief, a `P_*` field, `pi`, `tier`, `admissible`, `J`,
`exploration.pool`, the chosen cell or a flag disagrees with them.

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
`semantic-reader-v8`, interval kind `log-odds-grid`). Loading a v11
receipt recomputes every belief from its frozen forecast panels, expert weights,
`outcome_observations`, `duration_observations` and sampling policy, then the `P_*`
fields, `pi`, `T`, the three credit numbers, `session_cost_usd`, `credit_ok` and `J`,
and every {ref}`reused memory-free row <stored-forecasts>` from its sources; a
receipt whose stored values differ is refused.
Frozen `howhard-decision-v10` and `howhard-decision-v9` receipts keep the outcome
posterior they recorded (interval kind `dirichlet-posterior-mixture`): each expert's
forecast as a Dirichlet prior of at most `prior_strength` pseudo-observations plus the
cell's counts, mixed over the experts, with v10's decay of that prior over the first two
counted runs and without it in v9. v10 also lacks a host name, agent labels, ranges on
active time and token kinds, tiers and beliefs: these receipts keep the
per-expert active-time calibration they were made with (each expert's forecast worth one
run against the same-cell runs, log spread capped at 2, averaged over the experts), price
each expert's forecast directly, and check `needed_credit_usd <= available_credit_usd`.
Frozen `howhard-decision-v8` and `howhard-decision-v7` receipts keep their original weighted
forecast envelopes, upper-envelope admission and sampling policy shape; all load,
validate, accept outcomes and replay into predictive weights. Receipts older than v11
keep the cap as they recorded it:

| Item | Formats v7 to v10 | Format v11 |
| --- | --- | --- |
| `cap_met` | Some admissible cell has enough credit | The chosen cell is trusted |
| No cell passes | No draw; `selection.on_cap_unmet` records the policy and `exploration.reason` is `disabled-cap-unmet` | A draw over the second pool |
| `semantic.sampling.enabled` | Equals `cap_met` | Always true |
| `exploration.enabled` | `cap_met` and more than one eligible cell | More than one eligible cell |
| Cell fields | `admissible`, and `admission_risk` from v9; four `{mean, lower, upper}` probabilities | Also `pi` and `tier`; `P_clean` and `P_silent` carry `mean` only |
| `selection.sampling_policy` | Without the two tier constants; with `prior_strength`, `min_concentration` and `max_concentration` | With `trial_probability`, `all_excluded_factor`, `model_worth`, `task_worth_silent` and `task_worth_loud` |
| `semantic.sampling.cells[]` | `outcome_posterior` and `concentration` from v9 | `outcome_beliefs.loud` and `outcome_beliefs.silent` |
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
A cell's `prior_json` holds the Stage 1 prior, which all hosts share, and under
`forecasts` one stored Stage 2 forecast per host: its values, every stated range and,
as `host`, the name of the host whose request produced it (`null` for the unnamed
host). A forecast stored before forecasts were kept per host sits directly in
`prior_json`. It reads as the forecast of the host named there, or of the unnamed host
when none is named, and moves under `forecasts` the next time a forecast is stored on
its cell; an existing store needs no migration.
`howhard.prototypes.stored_forecasts(config, task_type=...)` returns one entry per
prototype, cell and host. {ref}`Reuse <stored-forecasts>` reads only the forecasts of
the request's host. A split copies the Stage 1 prior and every host's stored forecast
to each child. The timely statistic in `stats_json` compares a clean run's active time
with the forecast stored by the run's own host, which is the host named in the run's
decision receipt, or the unnamed host for a run with no receipt or an unnamed one; with
no such forecast the run has no overrun discount.
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
evidence and memory revision. Corrupt caches are replaced. The cache does not read the
host's name: an explicit backend adapter asked the same question for a second host
returns its cached answer, which is stored under that second host like any new forecast.
A cached forecast round is replayed only when it states the same memory-free rows that
the request {ref}`reuses from store <stored-forecasts>`. Prepared requests bind the
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
all cells under the request's host and no active editable memories exist, and lazily resolve `HOWHARD_LLM_COMMAND`
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
