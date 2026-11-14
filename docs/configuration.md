# Optional advanced configuration

The default workflow requires no file. `howhard.toml` in the current directory is loaded
when present; `--config PATH` chooses another file. Paths are relative to that file.
An explicit config must state `on_cap_unmet`; other settings overlay packaged defaults.
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
on_cap_unmet = "flag-review"

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
`cost_per_credit` and `credit_budget` fields apply to both sources; invalid configured lists
fail before creating state. Candidate `rate_card` and `usd_per_mtok` values are raw
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
on_cap_unmet = "flag-review"

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

The same configuration applies to initial context, `memory search`, and evidence
search. Prepared requests are bound to configuration; changing policy requires a new
request. An application-supplied `Engine(retrieval=pipeline)` still takes precedence
over file-based construction. See [retrieval stages](retrieval.md) for replacing
components through Python.

## Routing policy and storage

```toml
on_cap_unmet = "flag-review" # or lowest-bound
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

[sampling]
min_concentration = 8.0
max_concentration = 1000000.0
prior_strength = 8.0
explore_within = 0.10
uncertainty_multiplier = 1.0
```

The full inventory has no size bound. An oversized fit band gets one re-filter round
and then a clear error; it is never truncated. Widening forecasts the next stronger
overpowered cell per round, within the same total forecast budget.

`max_forecast_cells` is a positive integer; widening rounds are nonnegative integers.
Require `0 < min_clear_probability < reliability_probability < 1` and observation noise
in `(0,0.5)`. Probability/strength margins are in `[0,1]`; order penalty is nonnegative.
Concentration bounds satisfy `1 <= min <= max`, prior strength and uncertainty multiplier
are positive, and sampling `explore_within` lies in `[0,1]`. A risk class's optional
`explore_within` further narrows that fraction. The supported uncertainty allowance is
`uncertainty_multiplier * (sd_best + sd_candidate)` using delta-method retry-cost
standard deviations; a candidate must satisfy both that allowance and the mean fraction.
No exploration occurs when no cell passes the accepted-result silent cap.

`C_loud` now prices extra overhead for each loud failure on top of its redo. Epsilon
caps defects among accepted non-loud results, not the raw per-attempt silent rate.
Review these meanings when upgrading a custom policy. See [estimator assumptions](capability.md)
and the {ref}`selection objective <accepted-result-selection>`.
