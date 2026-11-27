# Optional advanced configuration

The default workflow requires no file. `howhard.toml` in the current directory is loaded
when present; `--config PATH` chooses another file. Paths are relative to that file.
Its settings overlay packaged defaults, so a file states only what it changes.
Unknown keys are refused. This reference is for integrators, not a user setup checklist.

(external-router-inventories)=
## Inventories for external routers

Native agent use remains the ordinary workflow: the installed skill inspects the host's
live delegation tools and exposed settings and supplies its available model/control
list through `--candidates FILE`. The backend does not detect the native agent or
enumerate its inventory. Missing model/control capabilities are reported as unavailable.
Saved configuration never replaces the host-supplied list automatically.

A separate routing application can use howhard for selection and learning while owning
provider access and execution. Save explicitly named inventories in `howhard.toml`:

```toml
[inventories]
router = [
  { model = "provider-a/model-x", effort = "high" },
  { model = "provider-b/model-y", effort = "low" },
]
```

These are illustrative identities, not claims about any provider's model catalog or
controls. Use stable IDs that distinguish the routes your application executes. A
provider-qualified `model` ID keeps the same model name on different provider routes
separate; howhard treats it as an opaque identity and returns it unchanged. The router
owns its mapping to provider requests. Every candidate needs an exact, enforceable
effort/control identity. No access or control values are inferred from a name.

Select a saved list explicitly with Python `inventory="router"` or CLI
`--inventory router`. Python `pick`, `summary` and `prepare`, and CLI `pick`, `summary`
and `context` support this selector. For example:

```sh
howhard context specified-implementation --inventory router --text "Repair transaction ordering" --json
```

When availability changes, pass Python `native_candidates=[...]` or CLI
`--candidates FILE` instead. Supplied lists and named inventories are mutually exclusive;
there is no union or fallback between them. An empty list or unknown inventory name is
an error. Without an explicit list or inventory selector, the backend refuses selection
even when saved lists exist. The same nonempty, unbounded inventory validation and optional pricing,
`cost_per_credit`, `credit_budget` and `agent` fields apply to both sources; invalid configured lists
fail before creating state. `agent = "NAME"` labels the agent product that would run an
entry (see {ref}`host and agent names <host-and-agent-names>`). Candidate `rate_card` and `usd_per_mtok` values are raw
list-price quota credit rates. `cost_per_credit` converts forecast credit use to actual
session money cost and defaults to `1.0`; set it to `0.0` for a free tier. The
`[models]` registry describes identities and evidence weights, not availability.
Previously observed models never become candidates merely because they are in history.

An application with a stateless forecast adapter can use the Python API directly:

```python
from howhard import Engine

# forecast_adapter and router are supplied by your application.
engine = Engine("howhard.toml", llm=forecast_adapter)
decision = engine.pick("specified-implementation", task_text, inventory="router")
selected = decision["chosen"]
result = router.execute(model=selected["model"], effort=selected["effort"], task=task_text)
engine.observe_decision(decision["decision_id"], result.outcome,
                        id=result.execution_id, tokens=result.tokens,
                        seconds=result.seconds, cost_usd=result.cost_usd)
```

The adapter implements `identity` and `complete(request) -> dict` for independent
baseline forecasts, memory-conditioned forecasts and memory updates. Its inference
provider is separate from the execution providers in the candidate list. The router
must honor review requirements and verify that the selected identity/control actually
ran before attaching receipt feedback. Outcomes use `clean`, `loud` or `silent`; usage
fields stay `None` unless measured. With the adapter supplied, observation also invokes
the memory writer. Alternatively, use the
{ref}`calling-agent protocol <calling-agent-provider-interfaces>` to supply forecasts and
writer responses explicitly without a native plugin.

All providers' runs and lessons in the same project store can inform later selections,
including selections where the source model is unavailable. Each actual result stays
bound to its executed model/control; predictions for unrun options are not observations.
Use consistent configuration/state paths across callers and inspect `howhard paths --json`
to confirm the store. Separate project stores are not automatically synchronized.
Configured-list changes require a new engine/config load; prepared decisions retain
their frozen inventory, and pending requests reject changed configuration.

(fixed-retrieval-configuration)=
## Fixed retrieval configuration

Select the reranker in user/application-owned `howhard.toml`:

```toml
[retrieval]
reranker = "fast"
```

`fast` is the default MiniLM-L4 INT8 ONNX cross-encoder, using two CPU threads and
batches of 16 with a 512-token combined window. Set `reranker = "qwen"` to choose
the slower instruction-aware Qwen3-Reranker-0.6B instead; install `howhard[qwen]`
for that backend. Configured Qwen defaults to two CPU threads, batches of four and
an 8,192-token window. These are explicit fixed choices: there is no automatic
upgrade or fallback, and agent retrieval messages cannot supply a tier, model or
configuration override. The packaged skill instructs agents to keep this policy fixed.

Both profiles accept `threads` (1–256) and `batch_size` (1–1,024). To replace the
pretrained model, supply **both** `model` and its full 40-character immutable `revision`.
For example, a user can select another compatible ONNX artifact:

```toml
[retrieval]
reranker = "fast"
model = "Xenova/ms-marco-MiniLM-L-6-v2"
revision = "a09144355adeed5f58c8ed011d209bf8ee5a1fec"
model_file = "onnx/model_quantized.onnx"
threads = 2
batch_size = 16
```

`model_file` is fast-backend-only and must be a relative `.onnx` artifact path inside
the model snapshot. Qwen instead accepts `instruction`, `device` and `max_length`
(1–32,768). Unsupported backend/option combinations, unknown keys, invalid numbers and
unpinned model overrides are rejected before state is created. Selecting Qwen does not
load weights or invoke a provider while parsing configuration; if its optional runtime
is missing when inference is needed, the error is explicit.

The same configuration applies to initial context, per-cell outcome-observation
retrieval, `memory search`, and evidence search. Prepared requests are bound to
configuration; changing policy requires a new
request. An application-supplied `Engine(retrieval=pipeline)` still takes precedence
over file-based construction. See [retrieval stages](retrieval.md) for replacing
components through Python.

## Routing policy and storage

```toml
project = "local"
journal = ".howhard/pending.jsonl"

[[logs]]
path = ".howhard/outcomes.jsonl"
project = "local"

[task_types]
my-task = "standard"

[classes.standard]
C_loud = 50.0
C_silent = 1000.0
lambda_per_hour = 25.0
epsilon = 0.10
# explore_within = 0.20
```

An explicit config requires `--risk` when picking an unknown task. Known taxonomy types
can be reassigned with `task_types`. Packaged classes are critical, contract, standard, docs
and routine; explicit projects may define additional classes. The first log belongs to the current project and is the flush destination;
additional logs can name other projects.

Known model mappings can be overlaid by `[models."concrete-id"]`:

```toml
[models.claude-opus-5-5]
family = "opus"
efforts = ["low", "medium", "high", "xhigh", "max"]
weight = 1.0
```

An explicit mapping labels the model's protocol cells and supplies its evidence weight;
it does not inherit learned family effects. Omitted evidence weight means 1.0. Unknown
native identities need no configuration and receive distinct opaque identity labels.
Every `pick`, `summary` and `context` request uses an explicit inventory through
`--candidates FILE` (Python: `native_candidates`) or an explicitly selected saved list
with `--inventory NAME` (Python: `inventory`). Registry entries do not activate candidates. Supply raw list-price quota credit rates as
optional `usd_per_mtok` or `rate_card`, an optional `cost_per_credit`, and optional remaining quota windows as
`credit_budget = { available_credit_usd = 1.25, session_5h_remaining_credit_usd = 1.25, weekly_7d_remaining_credit_usd = 4.50 }`, in that inventory;
unknown prices remain omitted/null. Family selection flags, model default efforts,
component API prices and token-mix configuration are not accepted.

Statistical priors, imported priors, cache paths and transfer tuning are no longer
accepted. `--journal` remains available for an integration path override. Provider
selection has no mode switch: Python accepts an adapter, and standalone inference uses
`HOWHARD_LLM_COMMAND` or an authenticated Claude CLI. Calling-agent commands do not
resolve a provider.

Native host use does not require this configuration. The skill supplies discovered
model/control cells through `--candidates`, and the backend owns their minimal historical
metadata. Native price omission means unknown. `--state-dir` redirects standard `.howhard/` data paths while honoring custom
configured paths, and keeps current project capture separate. Explicit configuration
without that override retains its existing data locations and policies.


## Capability and sampling policy

Defaults work without configuration. These optional application-owned settings control
forecast work and approximate inference; reader/retrieval messages cannot change them.

```toml
[capability]
max_forecast_cells = 12
widening_rounds = 2
min_clear_probability = 0.5
reliability_probability = 0.9
overpowered_gain_margin = 0.05
consistency_margin = 0.2
observation_noise = 0.05
order_penalty = 0.15
loud_weight = 0.75
silent_weight = 1.0
unknown_cause_weight = 0.5
caught_share_worth = 12.0
reserved_forecast_cells = 2
unproven_clear_probability = 0.2

[sampling]
uncertainty_multiplier = 1.0
admission_quantile = 0.80
trial_probability = 0.20
all_excluded_factor = 0.50
half_life_days = 14.0
comparison_cap = 256
model_worth = 5.0
rung_worth = 40.0
task_worth_amount = 1.0
duration_miss = 1.3
scatter_share = 0.5
task_worth_silent = 32.0
task_worth_loud = 16.0
effort_discount = 0.5
model_discount = 0.25
agent_discount = 0.25
```

The full inventory has no size bound. If the span of the clearing and timely intervals
exceeds the forecast budget, the filter uses their union. If the forecast set still
exceeds the budget, it gets one re-filter round and then a clear error. Widening
forecasts the next stronger cell with no forecast per round, within the same total
budget.

`min_clear_probability` is the floor of the band's lower edge and
`overpowered_gain_margin` the widest upper margin. Both edges tighten when the asking
host catches a smaller share of its failures than `d_0`
({ref}`the host's caught share <host-aware-band>`); with no failed run on record they
are exactly these two values. `caught_share_worth` is the number of failed runs at
which a record weighs as much as what it leans on: a host with 12 failed runs of its
own gets a caught share halfway between its own and that of all hosts, and all hosts
with 12 failed runs get one halfway between theirs and `d_0`.
`reserved_forecast_cells` of the `max_forecast_cells` slots
are reserved for families and agents the band leaves out
({ref}`reserved forecast slots <reserved-slots>`); 0 turns the rule off, and the
forecast cells are then the band. `unproven_clear_probability` is the `P_clear` a
family with no run on this kind of task must reach to take a reserved slot. All three
are optional.

`max_forecast_cells` is a positive integer; widening rounds are nonnegative integers.
`reserved_forecast_cells` is a nonnegative integer below `max_forecast_cells`, and
`caught_share_worth` is positive and may exceed 1.
Require `0 < min_clear_probability < reliability_probability < 1` and observation noise
in `(0,0.5)`. Probability/strength margins are in `[0,1]`; order penalty is nonnegative.
The uncertainty multiplier is positive,
`admission_quantile` lies in `[0.5, 1)`, `trial_probability` in
`[0, admission_quantile]`, `all_excluded_factor` in `[0, 1]`, `half_life_days` is positive,
`comparison_cap` is a positive integer, `model_worth`, `rung_worth`,
`task_worth_silent` and `task_worth_loud` are positive, and `effort_discount`,
`model_discount` and `agent_discount` lie in `[0, 1]`. A value outside its range is
refused with a message that names the key.

The worths set how many runs a prediction counts for at the start of a cell's
{ref}`outcome beliefs <outcome-posterior>`:

```text
worth   = n_family + model_worth
N_start = 1 / (1/worth + 1/N_task)
```

- `n_family`: the weight of the family's own finished routed runs on other tasks,
  under the request's host and the cell's agent
  ({ref}`corrected prediction <corrected-prediction>`). It is 0 with an empty record.
- `model_worth`: how many runs the rest of the record says about a model family, 5.
- `N_task`: the most a record on other tasks is worth for one task:
  `task_worth_silent` (32) for the accepted-defect share and `task_worth_loud` (16) for
  the loud rate.
- `N_start`: how many runs the prediction is worth. Each counted run then adds its own
  weight, and the prediction keeps the weight of `N_start` runs.

In plain words, a larger worth makes the prediction harder to move and its start
narrower; a smaller one lets counted runs take over sooner. Example: with the defaults
and an empty record the prediction of the accepted-defect share is worth
`1 / (1/5 + 1/32) = 4.32` runs and that of the loud rate `1 / (1/5 + 1/16) = 3.81`
runs. A family with 15 runs' worth on record has `worth = 20`, and its prediction of
the accepted-defect share is worth `1 / (1/20 + 1/32) = 12.3` runs.

`rung_worth` and `model_worth` also set how much each rung of the
{ref}`corrected prediction <corrected-prediction>` leans on the rung above it:

```text
level(r) = (sum_i a_i * y_i + kappa_r * level(r+)) / (sum_i a_i + kappa_r)
```

- `r`, `r+`: a rung and the one above it: all hosts, this host, this agent, this
  family, this cell.
- `sum_i a_i`, `sum_i a_i * y_i`: the weight of the rung's runs, and their weighted
  results.
- `kappa_r`: how many runs the rung above is worth: `model_worth` (5) at the family
  rung and `rung_worth` (40) at every other rung.

`rung_worth` is also the worth at which the measured weight on the forecaster leans on
1: `gamma = (n * gamma_hat + rung_worth) / (n + rung_worth)`. In plain words, a larger
`rung_worth` makes a host, an agent or a cell need more runs of its own before its
level leaves the level above it. Example: an agent with 100 runs' worth, 5 of them
silent, under a host whose level is 0.0622 has the level
`(5 + 40 * 0.0622) / (100 + 40) = 0.0535`; with `rung_worth = 10` it is
`(5 + 10 * 0.0622) / (100 + 10) = 0.0511`.

`prior_strength`, `min_concentration` and `max_concentration` are retired. A file that
still sets one loads: the key is accepted, its value is ignored, and one notice per key
is printed on standard error. Standard output is unchanged, and decisions are the same
as without the key.

`half_life_days` and `comparison_cap` govern which recorded runs count as evidence and
how much ({ref}`evidence candidates <evidence-candidates>`). A run's weight halves for
every `half_life_days` between its `observed_at` and the time the request was prepared,
in the outcome counts, the active-time and token beliefs and the capability threshold alike.
At most `comparison_cap` candidate runs are compared with the new task per decision,
most recent first; the rest count nowhere. Both are optional. The former
`sampling.outcome_recency` is accepted and ignored: a file that sets it loads with one
notice on standard error, and standard output is unchanged.

The three discounts set how much a recorded run counts as outcome evidence for a cell
other than the one that ran it ({ref}`outcome beliefs <outcome-posterior>`):

```text
delta(c_i, c) = 1                 same family, same effort (the same cell)
              = effort_discount   same family, another effort
              = model_discount    another family of the same agent
              = agent_discount    another agent
w_i = m_i * sim_i * a_i * delta(c_i, c)
```

- `c`, `c_i`: the cell being judged, and the cell that ran candidate run `i`. The first
  line that fits is used.
- `m_i`, `sim_i`, `a_i`: the run's weight on record, its similarity to the new task and
  its age factor.
- `w_i`: what the run counts for cell `c`, before the per-ticket limit of that cell.

In plain words, a run by a related cell counts as a fraction of a run by the cell
itself. Example: a run that weighs 0.8 for its own cell counts as `0.8 * 0.5 = 0.4` for
the same family at another effort and as `0.8 * 0.25 = 0.2` for a cell of another
agent. A cell's agent is the optional `agent` label of its candidate; with no labels
every cell belongs to the one unnamed agent, so `agent_discount` is never used. The
discounts apply to the outcome counts only, not to active time, tokens or the capability
threshold. Each is optional and lies in `[0, 1]`; setting all three to 0 counts only a
cell's own runs. They are frozen into each receipt's sampling policy.

Three constants sort cells into the {ref}`admission tiers <admission-tiers>` by `pi`,
a cell's chance that its accepted-defect share is within the risk class's epsilon:

```text
trusted   <=> pi >= admission_quantile
on trial  <=> trial_probability <= pi < admission_quantile
excluded  <=> pi < trial_probability
second pool, when no trusted or on-trial cell has enough credit:
              pi >= all_excluded_factor * (the highest pi among cells with enough credit)
```

- `admission_quantile`: the trusted line, 0.80. It is also the posterior quantile that
  `admission_risk` reports.
- `trial_probability`: the on-trial line, 0.20. With `trial_probability = 0` no cell is
  excluded; with `trial_probability = admission_quantile` no cell is on trial.
- `all_excluded_factor`: 0.50. It sets how near the best chance a cell must be to
  compete when every cell with enough credit is excluded: 1 keeps only the best chance
  and 0 keeps every cell with enough credit.

Example: with the defaults, cells with `pi` 0.958, 0.613 and 0.019 are trusted, on trial
and excluded. If the only cells with enough credit have `pi` 0.12, 0.08 and 0.05, the
bar is `0.50 * 0.12 = 0.06` and the first two compete.

Four keys shape the {ref}`beliefs for active time and tokens <amount-beliefs>`.
All four are optional and are stored in each receipt's sampling policy.

| Key | Symbol | Meaning | Default |
| --- | --- | --- | --- |
| `model_worth` | `kappa_m` | How many runs the record says about a model family's own level, beside the family's own runs: `worth = n_family + model_worth` | 5 |
| `task_worth_amount` | `N_task` | The most a record on other tasks is worth for one task, for an amount | 1 |
| `duration_miss` | `D` | The typical remaining miss of a prediction, as a spread of `ln(actual / prediction)`, before any run is on record; the record then measures `D` for each cell | 1.3 |
| `scatter_share` | | The share of `D^2` that is run-to-run scatter on one task | 0.5 |

`model_worth`, `task_worth_amount` and `duration_miss` are positive, and `scatter_share`
lies in `(0, 1]`. They give the run-to-run scatter `tau = D * sqrt(scatter_share)`, the
number of runs a prediction is worth `N_start = 1 / (1/worth + 1/N_task)`, and the
minimum spread of a belief `tau / sqrt(N_start)`. With the defaults and an empty record
`tau = 0.919`, `N_start = 0.833` and the minimum spread is 1.007 for active time and
every token kind.
A larger `duration_miss` widens every belief, which raises `credit_p80_usd` and the
spread of the cost; a larger `scatter_share` makes each recorded run count for less.

Thompson sampling considers every cell of the pool whose posterior cost overlaps the
pool's mean-best cell: `J - J_best <= uncertainty_multiplier * (sd_best + sd_candidate)`
using delta-method retry-cost standard deviations. A risk class's optional `explore_within`
additionally caps the gap at that fraction of `|J_best|`; it applies only when set on
the class. The former global `sampling.explore_within` is rejected with an error that
points to the risk-class setting.

`on_cap_unmet` is retired. A file that still sets it loads: the key is accepted, its
value is ignored, and one notice is printed on standard error. Standard output,
including `--json` output, is unchanged, and decisions are the same as without the key.

`C_loud` now prices extra overhead for each loud failure on top of its redo. Epsilon
caps defects among accepted non-loud results, not the raw per-attempt silent rate.
Review these meanings when upgrading a custom policy. See [estimator assumptions](capability.md)
and the {ref}`selection objective <accepted-result-selection>`.
