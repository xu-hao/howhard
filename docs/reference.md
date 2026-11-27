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

For each supplied native model and effort/control cell, minimize the cost to an accepted
result, averaged over the cell's beliefs:

```text
J(c)  = (C_run + lambda_per_hour * E[T]) * E[R] + C_loud * (E[R] - 1) + C_silent * E[S]
R     = 1 / (1 - theta_l) = 1 + exp(u)
C_run = q * E[K]
```

- `c`: a cell, one model family at one effort.
- `theta_l`, `u`: the cell's loud rate and its log-odds.
- `R`: the attempts until a run is not loud, so `R - 1` is the loud failures per accepted
  result. `E[R]` is summed over the grid of the loud rate's
  {ref}`belief <outcome-posterior>` and stored as `expected_attempts`.
- `S`: the accepted-defect share, silent results divided by accepted results. `E[S]` is
  summed over the grid of its belief.
- `K`: the credit of one run, from the cell's
  {ref}`token beliefs <amount-beliefs>`. `E[K]` is `needed_credit_usd`.
- `q`: the cell's `cost_per_credit`, or 0 when the cost policy excludes worker money.
  `C_run` is the expected session money cost of one run, `session_cost_usd`.
- `T`: the active time of one run in hours. `E[T]` is the average of the cell's
  {ref}`active-time belief <amount-beliefs>` after its counted runs, stored as `T`.
- `lambda_per_hour`, `C_loud`, `C_silent`: the risk class's price of an hour, of a loud
  failure and of an accepted defect.

In plain words: every attempt costs its credit and its time, a loud run is retried, and
each loud failure and each accepted defect has a price.

The average is taken over the belief because the retry factor is curved. For example, a
belief split evenly between loud rates of 0.2 and 0.6 averages 0.4. The cost at the
average charges `1 / 0.6 = 1.67` attempts, and the average over the belief charges
`(1.25 + 2.5) / 2 = 1.875`.

Example, with the packaged `standard` class (`lambda_per_hour` 25, `C_loud` 50,
`C_silent` 1,000): `C_run = 2`, `E[T] = 0.5`, `E[R] = 1.25` and `E[S] = 0.04` give
`J = 14.5 * 1.25 + 50 * 0.25 + 1,000 * 0.04 = 70.625`.

The candidate's `rate_card` or `usd_per_mtok` gives raw public API list-price quota
credit, reported as `needed_credit_usd`; `cost_per_credit` converts that credit amount to
session cost and defaults to `1.0` for pay-as-you-go. A free tier can set it to `0.0`.
The {ref}`spread of the cost and the draw <cost-spread-and-draw>` follow from the same
beliefs. Every belief starts from a {ref}`corrected prediction <corrected-prediction>`:
the forecast, corrected from the record of the same host on other tasks.

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
Runs whose receipt stores no name (receipts older than v11 and runs with no decision)
can be named once with an {ref}`import <imported-names>`.
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

Each belief starts from the {ref}`corrected prediction <corrected-prediction>` of its
rate and is reshaped by the runs counted for the cell: its own runs in full, and runs
of related cells at a discount.

**One forecast per rate.** The experts' forecasts for a cell are merged first:

```text
g_l = loud                     g_S = silent / (clean + silent)
m_k = logit(g_k)               s_k = (logit(hi_k) - logit(lo_k)) / 3.92
m   = sum_k w_k m_k            s_stated^2 = sum_k w_k (s_k^2 + (m_k - m)^2)
guess = logistic(m), held in [0.0001, 0.9998]
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
- `m`, `s_stated`, `guess`: the merged centre, the merged spread and the merged guess
  of the rate for this cell on this task.
- `logistic(x) = 1 / (1 + exp(-x))`, the inverse of `logit`.

In plain words, each rate gets one forecast per cell, and experts that disagree make it
less certain.

Example: two experts with weights 0.75 and 0.25 guess `S` at 10% and 25%, each with
spread 0.4. Then `m = 0.75 * (-2.197) + 0.25 * (-1.099) = -1.92`, `guess = 12.8%` and
`s_stated = sqrt(0.16 + 0.75 * 0.275^2 + 0.25 * 0.824^2) = 0.62`.

**The start of each belief.**

```text
N_start = 1 / (1/worth + 1/N_task)       sigma_1 = 1 / sqrt(p (1 - p))
s = min(1.869, max(s_stated, sigma_1 / sqrt(N_start)))
x_j = -16 + 0.01 j,  j = 0 .. 2500
q_j(m) = exp(-(x_j - m)^2 / (2 s^2)) / sum_i exp(-(x_i - m)^2 / (2 s^2))
centre: the m with  sum_j q_j(m) logistic(x_j) = p
```

- `p`: the {ref}`corrected prediction <corrected-prediction>` of the rate, held in
  `[0.0001, 0.9998]`. With an empty record it is the merged guess.
- `worth`: how many runs the record says about this family: the family's own runs on
  record under this host and agent plus `sampling.model_worth` (5). With an empty
  record it is 5.
- `N_task`: the most a record on other tasks is worth for one task,
  `sampling.task_worth_silent` (32) for `S` and `sampling.task_worth_loud` (16) for
  `theta_l`.
- `N_start`: how many runs the prediction is worth. With an empty record it is 4.32
  for `S` and 3.81 for `theta_l`.
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

Example, with an empty record: clean 0.72, loud 0.20, silent 0.08 with
`silent_ci = [0.04, 0.16]` gives `p = 0.10` for `S` and `s_stated = 0.40` (a range of
5% to 20%).
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
The prediction never fades out: however many runs are counted, it keeps the weight of
`N_start` runs. A cell with no counted run keeps its start.

Example, continuing with `epsilon = 0.10`: at the start `E[S] = 0.100`, `pi(c) = 0.70`
and the 95% range is 0.2% to 52%. With `n_c = 4` and `n_s = 0`, `E[S] = 0.053` and
`pi(c) = 0.85`. For `theta_l` at the start, expected attempts are 1.38, against 1.25 at
the average rate.

**Which runs are counted.** `n_c`, `n_l` and `n_s` are the counted amounts of each
outcome for the cell. The runs are the decision's
{ref}`evidence candidates <evidence-candidates>`: recorded `clean`, `loud` and `silent`
runs that belong to a prototype the new task matches (`interrupted` runs are excluded),
at most `comparison_cap` of them, most recent first. Every candidate is kept, whatever
cell ran it, including cells with no forecast in this decision. For each forecasted
`(family, effort)` cell (and any widened cell):

```text
delta(c_i, c) = 1                 same family, same effort (the same cell)
              = effort_discount   same family, another effort
              = model_discount    another family of the same agent
              = agent_discount    another agent

w_i  = m_i * sim_i * a_i * delta(c_i, c)
W_t  = sum of w_i over the candidate runs of ticket t
w_i <- w_i / max(1, W_t)
n_k  = sum of w_i over the candidate runs with outcome k
```

- `i`: a candidate run; `c_i` is the cell that ran it and `c` the cell being judged.
- `delta(c_i, c)`: the relatedness discount. The first line that fits is used. The
  defaults are `sampling.effort_discount` = 0.5, `sampling.model_discount` = 0.25 and
  `sampling.agent_discount` = 0.25. A run whose discount is 0 is not counted.
- `m_i`: the run's weight on record, the `weight` configured for its model (1 unless set).
- `sim_i`: `1 / (1 + exp(-r_i))`, where `r_i` is the reranker's score for the new task's
  text against the run's own task text.
- `a_i`: the age factor `0.5^(days_i / half_life_days)`, where `days_i` runs from the
  run's `observed_at` to the time the request was prepared and is never below 0.
- `t`, `W_t`: a ticket (the row's `ticket` label) and the weight its candidate runs carry
  for cell `c`. The limit follows the discount, so it is taken per judged cell. A row
  with no ticket is not limited.
- `n_k`: the counted amount of outcome `k`, for `clean`, `loud` and `silent`.

In plain words: a run counts more when its task reads like the new one and half as much
for every 14 days of age; a run by a related cell counts as a fraction of a run by the
cell itself; and all runs of one ticket together count for at most one run.

**Each run's family and agent.** The judged cell takes its `family` from the model
registry and its `agent` label from its candidate in the request. A run's family is the
registry family of the model that ran it. A run's agent is the label stored for the
chosen cell in the receipt that the row's `decision_id` names. A run with no decision or
a receipt older than `howhard-decision-v11` stores no label: it takes the agent an
operator {ref}`imported <imported-names>` for its family, if any. Otherwise it belongs,
like an unlabeled candidate and an unlabeled cell of the new request, to the one unnamed
agent. Two unnamed cells are the
same agent; a named and an unnamed one are different agents. At the same family the
agent is not read: such a run gets 1 or `effort_discount`.

Example: the judged cell is family `f1` at effort `high` under agent `a1`, and every
run has `m_i = 1`.

| Run | Ran on | Relation | `sim` | `a` | `delta` | Weight | Ticket | After the limit | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `f1` `high`, `a1` | Same cell | 0.8 | 1 | 1 | 0.80 | `t1` | 0.64 | clean |
| 2 | `f1` `low`, `a1` | Same family | 0.9 | 1 | 0.5 | 0.45 | `t1` | 0.36 | loud |
| 3 | `f2` `high`, `a1` | Same agent | 0.8 | 0.5 | 0.25 | 0.10 | `t2` | 0.10 | clean |
| 4 | `f3` `high`, `a2` | Another agent | 0.4 | 1 | 0.25 | 0.10 | `t3` | 0.10 | silent |

Ticket `t1` carries `0.80 + 0.45 = 1.25`, so both of its runs are divided by 1.25. The
counts are `n_c = 0.64 + 0.10 = 0.74`, `n_l = 0.36` and `n_s = 0.10`. With the three
discounts at 0 only run 1 counts, as 0.80 clean.

Only the counts that update these two beliefs read the discount. The
{ref}`beliefs for active time and tokens <amount-beliefs>` count runs of the same cell
only, and the capability threshold keeps its own strength factors. An expert's global
`evidence_count` affects only its predictive weight, never a cell's counted runs.

**What the receipt stores.** Each row of `semantic.outcome_observations` is one
candidate run, stored once: `id`, the run's own `family`, `effort` and `agent` (null
when unnamed), `outcome`, `ticket`, and `weight`, which is `m_i * sim_i * a_i` before
the discount. Each forecast cell in `selection.cells` stores `outcome_counts`: the
sums `n_clean`, `n_loud` and `n_silent` and `runs`, the number of runs with a weight
above 0 behind them. The three discounts are frozen in the receipt's sampling policy.

The storage rule: a receipt never stores one row per run and cell. A run's `relation`
(`cell`, `family`, `agent` for another family of the same agent, or `other` for another
agent), `discount`, ticket share and `weight` for a cell are a function of what the
receipt already holds: the run's stored labels, the cell's label in
`native_candidates` and the frozen discounts. With the discounts above 0 nearly every
candidate counts in every cell, so such rows would number cells times
`comparison_cap`, about 3,000 at the defaults. Loading a `howhard-decision-v11`
receipt recomputes each run's relation, discount, ticket share and weight from the
stored labels and policy and refuses a receipt whose sums or run count differ; an
observation's own cell need not be a forecast cell. A reader that needs the rows
computes them on demand with `howhard.decisions.counted_observations(receipt, cell)`,
which returns each counted run's `id`, `relation`, `discount` and `weight` after the
limit for one selection cell (the chosen cell by default).

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
- **Cached answers.** The {ref}`inference cache <inference-cache>` of an explicit
  backend follows the same rule: its key includes the host's name, so the forecast
  stored under a host is always an answer made for that host.

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
belief starts at the {ref}`corrected prediction <corrected-prediction>` of its amount,
is as wide as that prediction is unsure, and is updated by the cell's own recorded
runs. The two rates have the {ref}`outcome beliefs <outcome-posterior>`.

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
- `prediction`: where the belief starts: the
  {ref}`corrected prediction <corrected-prediction>` of the amount. With an empty
  record it is the merged forecast.
- `s_g`: the stated spread of the merge above.
- `D`: the typical remaining miss, as a spread of `ln(actual / prediction)`. It is
  measured from the record, and is `sampling.duration_miss` (1.3) with an empty record.
- `tau`: the run-to-run scatter on one task. `scatter_share` is the share of `D^2` that
  is scatter; at its default of 0.5, `tau = D / sqrt(2)`.
- `worth`: how many runs the record says about this family's own level: the family's
  own runs on record under this host and agent plus `sampling.model_worth` (5).
- `N_task`: the most a record on other tasks is worth for one task
  (`sampling.task_worth_amount`, 1).
- `N_start`: how many runs the prediction is worth.
- `m`, `s`: the centre and spread. Subtracting `s^2 / 2` makes the belief's average
  equal the prediction.

The belief averages to the prediction. A stated range can widen it and never narrow it.
For example, the prediction is 3,600 seconds with no stated range, worth is 5 and D is
1.3. Then `N_start = 0.833`, `tau = 0.919`, `s = 1.007` and `m = 7.682`: an average of
1 hour and a median of 0.60 hours. A stated range of 600 to 60,000 seconds gives 1.175,
which is used. With the defaults and an empty record the minimum spread is 1.007 for
active time and for every token kind.

Runs of the same cell then update the belief.

```text
1/s'^2 = 1/s^2 + W / tau^2
m'     = m + s'^2 * sum_i w_i * (y_i - m) / tau^2
E[T]   = exp(m' + s'^2 / 2)
```

- `i`, `w_i`, `W`: a same-cell run in `semantic.duration_observations` that records an
  active time, its weight, and the sum of the weights.
- `y_i`: the log of run i's active seconds.
- `m'`, `s'`: the centre and spread after the runs.
- `E[T]`: the belief's average. In hours it is `T` in the cost.

Each counted run pulls the centre toward its own time and narrows the belief, by its
weight. For example, two runs with weights 0.5 and 0.3 recorded 1,800 and 5,400 seconds.
Then `W = 0.8`, `s' = 0.719`, `m' = 7.792` and `E[T] = 3,137` seconds.

The runs are the cell's own {ref}`evidence candidates <evidence-candidates>` that record
an active time. Each counts at `m_i * sim_i * a_i` after the per-ticket limit over the
cell's own runs: similarity to the new task, half as much for every `half_life_days` of
age, and at most one run's weight per ticket. A run of another cell never counts here,
whatever {ref}`relatedness discount <outcome-posterior>` the outcome counts give it,
and it does not share this limit. A run's time is `effective_active_seconds`: its `active_seconds`, or its
total `seconds` when no active time was recorded, so a total that includes waiting counts
as active time. `interrupted` runs are excluded.

**Token beliefs.** Each of the six token kinds gets a belief of the same form, with its
own prediction, `D` and `N_start`, updated the same way and at the same weight for each
run, `m_i * sim_i * a_i` after the limit over the cell's own runs, from the same-cell runs
whose `execution_usage` is `complete`, whether or not the run records an active time. Each scored candidate, and each row of
`duration_observations` counted from it, carries what its run recorded:
`active_seconds` (or null), `tokens` (the total, or null) and `token_usage` (the count of
each kind over a complete usage envelope, or null). A same-cell candidate has a row when
it records an active time, complete usage or a total `tokens` count, and each belief
reads only the rows that record its value. A run without complete usage leaves the
kinds unchanged, and a run with complete usage and no active time moves the token
beliefs and leaves the active-time belief unchanged. A run that records only a total
`tokens` count, with neither an active time nor complete usage, moves the total-`tokens`
belief alone: not the active-time belief, and no token-kind belief.

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
| `active_time` | `forecast_seconds` (merged forecast), `prediction_seconds` (corrected prediction), `stated_spread` (`s_g`), `worth`, `N_start`, `D`, `prior_centre` and `prior_spread` (`m`, `s`), `centre` and `spread` (`m'`, `s'`), `evidence_weight` (`W`), `expected_seconds` (`E[T]`) |
| `token_beliefs` | The same record for each of the six kinds, or one record under `tokens`, with `forecast_tokens`, `prediction_tokens` and `expected_tokens`. A kind that is exactly zero has null centres and zero spreads. |
| `credit_variance`, `credit_p80_usd` | `Var(K)` and `K_80`, beside `needed_credit_usd` |

`P_loud` and `P_silent_accepted` report `{mean, lower, upper}` from the two beliefs:
the average and the 2.5% and 97.5% points of `theta_l` and of `S`. `P_clean` and
`P_silent` carry a `mean` only, because neither has a belief of its own:

```text
P_loud = E[theta_l]    P_silent = (1 - E[theta_l]) * E[S]    P_clean = (1 - E[theta_l]) * (1 - E[S])
```

For example, `E[theta_l] = 0.20` and `E[S] = 0.10` give `P_loud = 0.20`,
`P_silent = 0.08` and `P_clean = 0.72`. The three means are reports: `J` is averaged
over the two beliefs and is not the cost at these means. There is no additional
statistical risk floor, and a belief is a model assumption, not a calibrated coverage
guarantee.

(corrected-prediction)=
### The corrected prediction

Every belief starts from a prediction, not from the forecast itself. The record gives
the level of a value, and the forecast only says whether this task looks higher or
lower than that forecaster usually states.

**One rule for every value.**

```text
phi(prediction) = phi(level) + gamma * (phi(guess) - phi(usual))
phi(p) = logit(p) = ln(p / (1 - p))   for theta_l and S
phi(x) = ln(x)                        for active time and each token count
```

- `guess`: the merged forecast of the value for this cell on this task (the merges
  above).
- `level`: what happened in the recorded runs of this cell under this host.
- `usual`: what the forecaster had stated for those runs.
- `gamma`: the weight on the forecaster's judgment, from 0 to 1.
- `phi`: the value's own scale. A rate is held in `[0.0001, 0.9999]` before its
  log-odds are taken; an amount is held at 1 or more before its logarithm is taken.
- `prediction`: where the belief starts, before the similar runs of the
  {ref}`evidence candidates <evidence-candidates>` are counted. A rate's prediction is
  held in `[0.0001, 0.9998]`.

In plain words: the record gives the level, and the forecast only says whether this
task looks harder or easier than usual. An empty record gives the guess, `gamma = 1`
the guess moved by level minus usual, and `gamma = 0` the level alone.

Example: level 0.0566, usual 0.188, guess 0.10 and gamma 0.74 give
`-2.81 + 0.74 * (-2.20 + 1.46) = -3.36` in log-odds, a prediction of 0.034.

**Level and usual.** Both come down a ladder of five rungs, each leaning on the rung
above it:

```text
level(r) = (sum_i a_i * y_i + kappa_r * level(r+)) / (sum_i a_i + kappa_r)
usual(r) = (sum_i a_i * g_i + kappa_r * usual(r+)) / (sum_i a_i + kappa_r)
```

- `r`, `r+`: a rung and the one above it. From the top: all hosts, this host, this
  agent under it, this family under it, this cell under it. Above all hosts both
  ladders equal the guess. A rung with no run passes the value above it through.
- `i`: a finished routed run at rung `r`: a recorded run with outcome `clean`, `loud`
  or `silent` whose decision stored a forecast for the cell that ran. Its host and
  agent are the {ref}`names <host-and-agent-names>` that decision stores; for a
  decision older than v11, the names {ref}`imported <imported-names>` for it and for
  the run's family, if any. The
  {ref}`evidence candidates <evidence-candidates>` of the new decision (every recorded
  run of a prototype the new task matches) are left out: they count through their
  similarity instead. A run with no decision, a run whose decision cannot be read, and
  an `interrupted` run enter no ladder.
- `a_i`: the run's age factor `0.5^(days_i / half_life_days)`, divided by
  `max(1, A_t)`, where `A_t` is the summed age factors of the runs of ticket `t` in the
  same ladder. A row with no ticket is not limited.
- `y_i`: what happened. For a rate it is 1 for a failure and 0 otherwise; for an amount
  it is the logarithm of the recorded amount.
- `g_i`: the merged guess that run's own decision stored for its cell, on the same
  footing: a probability for a rate, a logarithm for an amount. A receipt of any
  readable format gives it by the merge rules above.
- `kappa_r`: the worth of the rung above: `sampling.model_worth` (5) at the family
  rung and `sampling.rung_worth` (40) at every other rung.

Each kind of value has its own ladder. `S` counts accepted results (`clean` and
`silent`; the failure is `silent`), `theta_l` counts every finished run (the failure is
`loud`), and an amount counts the runs whose recorded amount and guess are both above
zero. A rate's level and usual are averages of probabilities; an amount's are averages
of logarithms, reported as amounts.

In plain words: a cell with few runs leans on its family, then its agent, its host and
all hosts.

Example, for `S` with guess 0.10:

| Rung | sum a | sum a·y | sum a·g | level | usual |
| --- | --- | --- | --- | --- | --- |
| All hosts | 160 | 24 | 16 | 0.140 | 0.100 |
| This host | 60 | 6 | 9 | 0.116 | 0.130 |
| This agent | 40 | 2 | 6 | 0.083 | 0.140 |
| This family | 15 | 1 | 3 | 0.071 | 0.185 |
| This cell | 10 | 0 | 2 | 0.057 | 0.188 |

The all-hosts level is `(24 + 40 * 0.10) / (160 + 40) = 0.140`, and the family level is
`(1 + 5 * 0.083) / (15 + 5) = 0.071`. Ten finished routed runs of one ticket, recorded
on the day of the decision, add 1 to `sum a`.

**The weight on the forecaster.**

```text
gamma_hat = (O_H - O_L) / (F_H - F_L), held in [0, 1]
gamma     = (n * gamma_hat + kappa) / (n + kappa)
```

- `H`, `L`: two groups of the ladder's runs. `H` holds the runs with
  `d_i = phi(g_i) - phi(usual_i)` above zero, the runs their forecaster rated above
  usual; `L` holds the rest.
- `usual_i`, `level_i`: both ladders at run `i`'s own cell over all the runs, with the
  record's weighted averages of `g` and `y` above all hosts.
- `F_H`, `F_L`: each group's weighted average of `d_i`, with weights `a_i`.
- `O_H`, `O_L`: how far each group's results sat from its levels. For a rate it is
  `logit((sum a_i y_i + 1/2) / (sum a_i + 1)) - logit((sum a_i level_i + 1/2) / (sum a_i + 1))`
  over the group; for an amount it is the group's weighted average of `y_i - level_i`.
- `n`: the total weight `sum a_i` of the runs. `kappa`: `sampling.rung_worth` (40).
- `gamma_hat` is 1 when a group is empty or `F_H - F_L` is under 0.05.

In plain words: the gap that happened divided by the gap that was forecast, starting at
1. One weight per kind of value (`theta_l`, `S`, active time, total tokens and each
token kind) is measured at every decision from all hosts' runs and shared by all hosts.

Example: `H` holds 60 runs' worth with 13 silent and levels summing to 9, and `L` holds
100 with 11 silent and levels summing to 15. Then
`O_H = logit(13.5 / 61) - logit(9.5 / 61) = 0.43` and `O_L = -0.34`. With
`F_H - F_L = 1.15`, `gamma_hat = 0.675` and `gamma = (160 * 0.675 + 40) / 200 = 0.74`.

**How firmly the prediction is held.**

```text
worth   = n_family + kappa_m
N_start = 1 / (1 / worth + 1 / N_task)
s_min   = sigma_1 / sqrt(N_start)
s       = max(s_stated, s_min), at most 1.869 for a rate
```

- `n_family`: `sum a_i` at the family rung of the value's ladder. `kappa_m`:
  `sampling.model_worth` (5).
- `N_task`: the most a record on other tasks is worth for one task:
  `sampling.task_worth_silent` (32) for `S`, `sampling.task_worth_loud` (16) for
  `theta_l` and `sampling.task_worth_amount` (1) for an amount.
- `N_start`: how many runs the prediction is worth.
- `sigma_1`: one run's spread: `1 / sqrt(p * (1 - p))` for a rate with `p` the
  prediction, and `tau = D * sqrt(scatter_share)` for an amount, which is `D / sqrt(2)`
  at the default.
- `s_stated`, `s`: the merged forecast's stated spread, and the belief's starting
  spread. 1.869 is the flat belief.

In plain words: the family's own runs set how firmly the prediction is held, and a
stated range can only widen it.

Example: `worth = 15 + 5 = 20` gives `N_start = 1 / (1/20 + 1/32) = 12.3`. With
`p = 0.034`, `sigma_1 = 5.54` and `s_min = 1.58`. A stated range of 4% to 20% gives
0.46, so `s = 1.58`.

**The remaining miss of an amount.**

```text
D2(r) = (sum_i a_i * e_i^2 + kappa_r * D2(r+)) / (sum_i a_i + kappa_r)
e_i   = y_i - phi(prediction_i)
D     = sqrt(D2(this cell))
```

- `prediction_i`: the rule above for past run `i`, with `level_i`, `usual_i` and the
  `gamma` of its kind of value.
- `D2(r+)` above all hosts: `sampling.duration_miss` squared (1.69). `r`, `a_i` and
  `kappa_r` are those of the level and usual ladders.

In plain words: `D` is the typical miss left after the correction.

Example: 100 runs' worth on one cell, each off by a factor of `e^0.5`, give `D2` =
0.661, 0.368, 0.284, 0.252 and 0.250 down the rungs, so `D = 0.50`.

**Token counts.** Each of the six token kinds has its own ladders, weight and `D`. They
are empty until runs record usage by kind (a complete `execution_usage`) for decisions
that forecast usage by kind. Total `tokens` has its own as well, from each run's
recorded `tokens`; it starts the one token belief of a cell whose forecasts do not all
state `token_usage`. A kind forecast at exactly zero stays zero.

**Receipt.** `semantic.prediction` stores the sums and every number derived from them.
The ladder sums are read once, when the request is prepared and outside the storage
lock, with the same prepare time as the evidence candidates' age factors.

| Field | Content |
| --- | --- |
| `kinds.<kind>` | For `loud`, `silent`, `seconds`, `tokens` and each token kind: `n`, the two groups `high` and `low`, `gamma_hat` and `gamma`. A group holds `weight` (`sum a_i`), `gap` (`sum a_i d_i`), `outcome` (`sum a_i y_i`) and `level` (`sum a_i level_i`). |
| `cells[].values.<kind>` | For each forecast cell and each of its values (`loud`, `silent`, `seconds`, and its six token kinds or `tokens`): `guess`, `rungs`, `level`, `usual`, `prediction`, `worth`, `N_start`, `s_stated`, `s` and, for an amount, `D`. |
| `rungs` | Five entries from `all-hosts` through `host`, `agent` and `family` to `cell`. Each holds `weight` (`sum a_i`), `outcome` (`sum a_i y_i`), `guess` (`sum a_i g_i`) and, for an amount, `miss` (`sum a_i e_i^2`). |

A rate's `guess`, `level`, `usual` and `prediction` are probabilities. An amount's are
amounts (seconds or tokens), while its rung sums are sums of logarithms. A kind
forecast at exactly zero stores a zero `guess`, `prediction` and `s`, and null `level`
and `usual`.

When a receipt is loaded, every guess is merged again from the frozen forecast panels
and every derived number (`n`, `gamma_hat`, `gamma`, `level`, `usual`, `prediction`,
`worth`, `N_start`, `D` and `s`) is recomputed from the stored sums. A receipt whose
stored numbers do not follow is refused, and the beliefs of each cell must start from
the stored prediction.

Receipts older than v11 name neither a host nor an agent: their runs sit under the
names an operator {ref}`imported <imported-names>` for their decision and family, and
otherwise under the unnamed host and the unnamed agent.

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
- `J`, `sd`, sampled `J`: the cell's cost averaged over its beliefs, the exact spread of
  that cost (`cost_sd`) and its cost under the seed's draw (`sampled_J`); see
  {ref}`the spread of the cost and the draw <cost-spread-and-draw>`.
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

(cost-spread-and-draw)=
### The spread of the cost and the draw

**The exact spread.** Credit, active time, the loud rate and the accepted-defect share
are four independent beliefs of a cell, so the spread of its cost follows from their
means and variances with no approximation.

```text
X      = q * K + lambda_per_hour * T + C_loud        so that  J = X * R - C_loud + C_silent * S
Var(J) = E[X^2] * E[R^2] - (E[X] * E[R])^2 + C_silent^2 * Var(S)
E[X]   = C_run + lambda_per_hour * E[T] + C_loud
E[X^2] = q^2 * Var(K) + lambda_per_hour^2 * Var(T) + E[X]^2
Var(T) = E[T]^2 * (exp(s'^2) - 1)
sd(c)  = sqrt(Var(J))
```

- `X`: what one attempt costs, plus `C_loud`.
- `E[R]`, `E[R^2]`: the averages of `1 + exp(u)` and of `(1 + exp(u))^2`, summed on the
  grid of the loud rate's belief.
- `Var(S)`: the average of `S^2` minus `E[S]^2`, summed on the grid of its belief.
- `Var(K)`: the variance of the credit of one run, `credit_variance`.
- `s'`: the spread of the active-time belief after its counted runs, in log seconds.
  `Var(T)` is in hours squared.
- `sd(c)`: the spread of the cell's cost, stored as `cost_sd` on the cell's entry of
  `semantic.sampling.cells`.

Example, continuing the cost example above with `q = 1`, `sd(K) = 1`, `s' = 0.6`,
`E[R^2] = 1.60` and `sd(S) = 0.03`: `Var(T) = 0.25 * (exp(0.36) - 1) = 0.1083`,
`E[X] = 64.5` and `E[X^2] = 1 + 625 * 0.1083 + 4,160.25 = 4,228.96`.
`Var(J) = 4,228.96 * 1.60 - 80.625^2 + 900 = 1,165.94`, so `sd(c) = 34.15`.

A wider stated range on active time or on a token kind raises that cell's `cost_sd`.
It changes neither the cell's `J`, because a belief averages to its prediction however
wide it is, nor any number of another cell.

**Eligibility.** A cell of the pool competes when its cost and the best cost overlap
within their spreads:

```text
eligible(c) <=> J(c) - J(c*) <= uncertainty_multiplier * (sd(c*) + sd(c))
```

- `c*`: the pool's cell with the lowest `J`.
- `uncertainty_multiplier`: the `[sampling]` key, 1 by default. When the risk class sets
  `explore_within`, the gap is also capped at `explore_within * |J(c*)|`.

Example: `c*` has `J` 70.6 and spread 34.2. A cell with `J` 110 and spread 20 is
eligible, because the gap of 39.4 is at most 54.2. A cell with `J` 140 and spread 10 is
not, because 69.4 exceeds 44.2.

**The draw.** A sample is drawn for every decision, so `semantic.sampling.enabled` is
always true. Each forecast cell draws one value of everything it is uncertain about,
and the lowest sampled cost among the eligible cells wins. No expert is drawn: every
value of a cell is one belief merged over the experts, and experts that disagree widen
the beliefs themselves.

```text
S_s = logistic(v_j)      j = the first point where the running total of the accepted-defect belief reaches U_S
R_s = 1 + exp(u_j)       j = the same on the loud belief, with U_u
T_s = exp(m' + s' * z(U_T)) / 3600
n_a = exp(m_a + s_a * z(U_K))        for every token kind a
J_s = (q * K_s + lambda_per_hour * T_s) * R_s + C_loud * (R_s - 1) + C_silent * S_s
chosen = the eligible cell with the lowest J_s
```

- `U_S`, `U_u`, `U_T`, `U_K`: four uniform numbers strictly between 0 and 1, drawn in
  this order for every forecast cell.
- `j`, `v_j`, `u_j`: a grid point and its log-odds on the two beliefs. It is the last
  point when rounding keeps the running total below `U`.
- `z(U)`: the standard normal value with probability `U` below it.
- `m'`, `s'`: the centre and spread of the active-time belief, in log seconds. `T_s` is
  in hours.
- `m_a`, `s_a`: the centre and spread of the belief of token kind `a`, and `n_a` its
  sampled count. One `U_K` serves every kind, because the kinds move together. A kind
  that is exactly zero stays zero.
- `K_s`: the credit of the counts `n_a`, priced at the cell's rates as their averages
  are.
- `J_s`: the sampled cost, stored as `sampled_J`.

In plain words: a wide belief sometimes draws a low cost, so its cell is tried.

Example: `S_s = 0.02`, `R_s = 1.25`, `T_s = 0.4` and `q * K_s = 1.50` give
`J_s = 11.5 * 1.25 + 12.5 + 20 = 46.875`.

The numbers come from `numpy.random.PCG64` seeded with the stored `seed`. Each `U` is
`((r >> 12) + 0.5) / 2^52` from one raw 64-bit output `r`, so a replay does not depend
on a library's distribution methods. Cells draw in the order of
`semantic.sampling.cells`, eligible or not, so the stream does not depend on the pool.
Ties go to the lower `J`, then family and effort. The seed is `--seed` or 64 random
bits. `summary` returns the same costs, ranges, beliefs and tiers and makes no draw.

Each entry of `semantic.sampling.cells` stores `family`, `effort`, the cell's
`outcome_beliefs`, `probabilities`, `cost_sd` and `sampled`:

| Field of `sampled` | Content |
| --- | --- |
| `S` | `S_s`, the sampled accepted-defect share |
| `u` | `u_j`, the sampled log-odds of the loud rate, so `R_s = 1 + exp(u)` |
| `seconds` | `3600 * T_s`, the sampled active seconds |
| `credit` | `K_s`, the sampled credit of one run; null for an unpriced cell |

`outcome_beliefs.loud` and `outcome_beliefs.silent` each hold `prediction`,
`stated_spread`, `n_start`, `spread`, `centre`, `counts` (`[n_yes, n_no]`), `mean`,
`lower` and `upper`; `silent` adds `pi`, and `loud` adds `attempts` and `odds`, which
are `E[R]` and `E[R] - 1`. A v11 receipt has no `semantic.sampled_expert`.

Forecasted `seconds` represents expected **active** execution duration excluding
external queue wait. Receipts retain the scored `evidence_candidates`, the
`outcome_observations` and `duration_observations` read from them, each cell's
`outcome_counts` and beliefs, `pi`, `tier` and `admission_risk`, the seed, the sampled values and the reason
for a sampled non-best choice. A fixed seed reproduces selection given frozen forecasts
and the request's scored evidence candidates; it does not make inference deterministic.
A receipt records the time its request was prepared, because the age of every counted
run is measured to that moment, so two decisions prepared at different times are
different receipts even when nothing else differs.

Loading a stored decision recomputes every belief of every forecast cell from its
frozen forecasts, expert weights, counted runs, agent labels and policy, then `J`,
`expected_attempts` and `cost_sd` from those beliefs on the same grid, and replays the
draw from `seed`. A receipt in which a counted sum, a belief, a `P_*` field, `pi`, `tier`,
`admissible`, `J`, `expected_attempts`, `cost_sd`, a sampled value, a `sampled_J`,
`exploration.pool`, `exploration.eligible_cells`, the chosen cell or a flag disagrees
is refused with a message that names the field.

Loud attempts are retried on the same cell until a non-loud result is accepted.
`E[R]` charges the expected attempts; `C_loud` is extra detection/rework overhead
per loud failure, in addition to the repeated time/money. The grid's highest loud rate,
0.99988, keeps a near-certain loud failure finite but very costly.
Retry-on-another-cell is not assumed.

Risk-class `explore_within` can further tighten the posterior-overlap band; it cannot
permit exploration outside it. Each cheaper alternative is explained in `why`:
`insufficient-credit`, `silent-cap` for a cell outside the pool (the message quotes its
`pi`), `explore-within` or `posterior-overlap` for a pool cell outside the band, and
otherwise `sampled-cost` with signed `components`: for each of the four parts of the
sampled cost, that cell's part minus the chosen cell's.

```text
run = q * K_s * R_s    time = lambda_per_hour * T_s * R_s    loud = C_loud * (R_s - 1)    silent = C_silent * S_s
```

Loading a receipt recomputes every belief from its frozen forecast panels, expert weights,
`outcome_observations`, `duration_observations` and sampling policy, then each cell's
`outcome_counts`, the `P_*`
fields, `pi`, `T`, the three credit numbers, `session_cost_usd`, `credit_ok`, `J`,
`expected_attempts` and `cost_sd`, and every
{ref}`reused memory-free row <stored-forecasts>` from its sources, and replays the draw
from the seed; a receipt whose stored values differ is refused. Receipts older than v11
keep the rules they were made with ({ref}`older receipts <frozen-formats>`).

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
retrieve/decide. Requests and receipts use the frozen formats below, and a receipt
requires its `host`, capability and accepted-result selection state.

(frozen-formats)=
### Frozen formats

Each stored shape carries a format label. The labels below are frozen: any later change
to one of these shapes takes a new label. The constant that holds each label is named
relative to the `howhard` package.

| Stored shape | Constant | Label |
| --- | --- | --- |
| Decision receipt | `context.DECISION_FORMAT` | `howhard-decision-v11` |
| Semantic state of a receipt | `semantic.FORMAT` | `howhard-semantic-v8` |
| Reader prompt of a receipt | `semantic.PROMPT_VERSION` | `semantic-reader-v8` |
| Interval kind of a receipt | `semantic.INTERVAL_KIND` | `log-odds-grid` |
| Capability state of a receipt | `capability.FORMAT` | `howhard-capability-v6` |
| Prepared request | `engine.REQUEST_FORMAT` | `howhard-agent-request-v7` |
| Project snapshot | `context.PROJECT_FORMAT` | `howhard-project-v1` |
| Context encoder | `context.FORMAT` | `howhard-context-v1` |
| Native registry | `native.FORMAT` | `howhard-native-v1` |
| Execution usage envelope | `accounting.USAGE_FORMAT` | `howhard-execution-usage-v1` |
| Rate card | `accounting.RATE_FORMAT` | `howhard-rate-card-v1` |

**A prepared request or a v11 receipt is readable only in its frozen shape.** A request
that lacks a part is refused with a message that names the part and says to run
`howhard context` again: the agent label on each scored candidate run, the failed runs
behind the host's caught share (`learned_state.caught_record`), the record on other
tasks behind the corrected prediction (`learned_state.prediction`), or any other field
of the frozen request. Nothing completes or migrates such a request; prepared requests
are short-lived, so prepare a new one. A request of another format is refused the same
way. A receipt labelled `howhard-decision-v11` whose shape differs from the frozen one,
including one without its `request_snapshot`, is refused by the checks of its fields.

The semantic format and reader prompt live in the packaged `data/semantic.toml`, and
`semantic.RECEIPT_SEMANTICS` pairs them with each decision format.

The capability state stores the cells, each with its `agent` label. Its `band` is an
ordered list of fit cells that can omit `intermediate` cells between its `lower` and
`upper` endpoints. It also stores `caught_share` (`host`, `F_all`, `C_all`, `F_host`,
`C_host`, `d_0`, `d_all`, `d_host` and the two band edges `e` and `g`), the `core` (the
band's first `max_forecast_cells - reserved_forecast_cells` cells), `proven_families`
(the families with a run among the evidence candidates) and `reserved` (each reserved
cell's `key`, its `reason`, `unproven-family` or `missing-agent`, and its `bar`). The
cells that carry forecasts are the core, the reserved cells, the band cells that took
unfilled slots and the widening cells. Loading recomputes the caught shares, the edges,
the band, the core, the reserved cells and the widening order from the stored numbers
and refuses a receipt that does not follow from them. See
[capability filtering](capability.md) for the band, reserved-slot and widening rules.
A stored `P_clear` or `P_timely` may exceed 1 by at most `1e-10`
(`capability.PROBABILITY_TOLERANCE`), the rounding that a running sum of posterior
masses can leave; a larger value is refused. The tolerance holds for every receipt
format.

**Older receipts.** These formats keep loading, validating, accepting outcomes, replaying
their stored forecasts for scoring and appearing in `report`, each under its own rules:

| Decision | Semantic state and reader prompt | Capability state |
| --- | --- | --- |
| `howhard-decision-v10` | `howhard-semantic-v7`, `semantic-reader-v7` | `howhard-capability-v5` or `howhard-capability-v4` |
| `howhard-decision-v9` | `howhard-semantic-v6`, `semantic-reader-v7` | `howhard-capability-v4` or `howhard-capability-v3` |
| `howhard-decision-v8` | `howhard-semantic-v5`, `semantic-reader-v6` | `howhard-capability-v3` |
| `howhard-decision-v7` | `howhard-semantic-v4`, `semantic-reader-v5` | `howhard-capability-v1` or `howhard-capability-v2` |

`howhard-decision-v10` and `howhard-decision-v9` receipts keep the outcome
posterior they recorded (interval kind `dirichlet-posterior-mixture`): each expert's
forecast as a Dirichlet prior of at most `prior_strength` pseudo-observations plus the
cell's counts, mixed over the experts, with v10's decay of that prior over the first two
counted runs and without it in v9. v10 also lacks a host name, agent labels, ranges on
active time and token kinds, tiers and beliefs: these receipts keep the
per-expert active-time calibration they were made with (each expert's forecast worth one
run against the same-cell runs, log spread capped at 2, averaged over the experts), price
each expert's forecast directly, and check `needed_credit_usd <= available_credit_usd`.
`howhard-decision-v8` and `howhard-decision-v7` receipts keep their original weighted
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
| `semantic.sampling.cells[]` | `outcome_posterior` and `concentration` from v9; the drawn `expert` | `outcome_beliefs.loud` and `outcome_beliefs.silent`; `sampled`; no `expert` |
| `J`, `expected_attempts` | The cost at the posterior means: `(C_run + lambda_per_hour * T + C_loud * P_loud + C_silent * P_silent) / max(0.0001, 1 - P_loud)`, with `1 / max(0.0001, 1 - P_loud)` attempts | The cost averaged over the beliefs, with `E[R]` attempts |
| `cost_sd` | A first-order spread that treats run cost as exact; not recomputed on load | The exact spread, recomputed on load |
| The draw | One expert drawn by weight (`semantic.sampled_expert`), then each cell's rates and active time; not replayed on load | Four uniform numbers per forecast cell from `PCG64`; replayed on load |
| Widening labels | Trigger `silent-cap-unmet`; stop `cap-met` | Trigger `all-excluded`; stop `pickable` |

Receipts older than v11 carry no host name and no agent labels, and they do not accept
those fields or the ranges on active time and token kinds. Their runs belong to the one
unnamed host and the one unnamed agent unless an operator
{ref}`imports names <imported-names>` for them. Receipts v1–v6 remain recognizable as
historical store entries but are refused on load and skipped when scoring; prepare a
fresh request and decision to attach new feedback. Frozen-receipt observations and
learning share the store across that project's worktrees.

`report --json` is the sweep of a store's receipts: it loads every stored receipt, and
its `receipts` block counts them by format (`formats`, including v1–v6 receipts, which
are not scored) and counts in `rounded_probabilities` the receipts whose `P_clear` or
`P_timely` lies above 1 within the tolerance. A receipt that does not load stops the
report with its error. To sweep a real store, run `howhard report --json` in the
project, with the same `--config` or `--state-dir` as its other commands; it changes
no receipt and no recorded run.

(imported-names)=
### Names for older decisions

A run whose receipt stores no name belongs to the unnamed host and the unnamed agent. An
operator can name such runs once, with one optional JSON file:

```sh
howhard names import --file NAMES.json --json
```

```json
{"hosts":  {"<decision id>": "<host name>"},
 "agents": {"<family>": "<agent label>"}}
```

- `hosts`: names the host of a stored decision whose receipt is older than v11. Its runs
  then belong to that host: in the failed runs behind the
  {ref}`host's caught share <host-aware-band>` and on the host rungs of the
  {ref}`corrected prediction <corrected-prediction>`.
- `agents`: names the agent of a model family for every run whose receipt stores no
  agent label, which is a run of a receipt older than v11 or a run recorded with no
  decision. A registered model id is accepted in place of its family. The name is that
  run's agent in the {ref}`relatedness discounts <outcome-posterior>` and on the agent
  rungs of the corrected prediction.
- Both tables are optional, and any other key refuses the whole file, as does a name
  that is not 1 to 64 letters, digits, `_` or `-`, or a model id and its family named
  differently in one file.
- An entry for a `howhard-decision-v11` receipt, for an identifier that names no stored
  decision, or for a family or model that is not registered is skipped and reported; the
  rest of the file is stored. A v11 receipt is never renamed: it stores its own host and
  agent labels, including the unnamed ones.

The result counts, per table, the names `stored` (new), `unchanged`, `replaced` and
`skipped`, and lists each skipped entry with its `table`, `key` and `reason`. The same
file twice changes nothing and writes nothing; a later file replaces the names an earlier
file gave. Python: `Engine.import_names(value)` with the parsed file.

The names live in two tables of `payloads.sqlite3`, `decision_hosts` and
`family_agents`, written under the storage lock. No receipt is rewritten, because its
identifier is the digest of its content. Readers consult the names only for a run whose
receipt stores none, so every stored v11 receipt still re-validates from its own
numbers, and a new decision freezes the names it read into its receipt as it does the
rest of its record. `report` shows the imported names under `routing`. A forecast stored
on a prototype before v11 names no decision and no host: it stays with the unnamed
host, and the import does not reach it ({ref}`stored forecasts <stored-forecasts>`);
neither does the timely statistic of a prototype cell, which reads only the host a
receipt stores. With no import, older runs belong to the unnamed host and the unnamed
agent. The ordinary workflow needs no import.

To name the runs of a real store, list its older decisions with
`howhard.decisions.summaries(config)` or `report --json` (`routing` and `receipts`),
write the file, run `howhard names import --file NAMES.json --json` in the project (with
the same `--config` or `--state-dir` as its other commands), and read the counts and the
skipped entries. Running it again with the same file is safe.

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

(inference-cache)=
## Inference cache and current interfaces

Validated LLM responses may be cached by complete request, adapter identity, settings,
evidence, memory revision and the host's name. Corrupt caches are replaced. The unnamed
host is a key of its own, so a cached answer is returned only to the host it was made
for and is never returned, or stored as a forecast, for another: an explicit backend
adapter is called again when a second host sends the same request. A cached forecast
round is replayed only when it states the same memory-free rows that the request
{ref}`reuses from store <stored-forecasts>`. Prepared requests bind the
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
