# Model and storage contracts

Routing uses LLM forecasts and writable semantic memories. See the [reader/writer
contract](semantic-memory.md) for retrieval, isolated forecasts and predictive weights.
There is no fitted outcome regression, token/time regression, lexical predictor,
statistical safety floor or generic prior transfer.

## Selection

For each supplied native model and effort/control cell, minimize

```text
J = C_run + P_loud * C_loud + P_silent * C_silent + lambda_per_hour * T
```

The host supplies 1–32 distinct actual cells through `--candidates FILE` or Python
`native_candidates`. Each has `model`, `effort` and optional `usd_per_mtok`. No-effort,
null and `model_default` controls are refused. Unknown identities receive distinct
opaque family labels for protocol bookkeeping, without model assumptions or effects.
Configured identities retain their declared labels and evidence weights. Historical
identity metadata does not activate candidates in a later request.

An independent stateless LLM forecasts every cell using the task and project alone.
The reader supplies conditional overrides for memories it has seen; omitted cells and
unretrieved memories use the independent LLM forecast. No learned probability table
or synthetic forecast is supplied to either call. Provider absence, invalid forecasts
and provider failures are explicit errors; none produces a routing recommendation.

Probability means are weighted by predictive expert support. Lower/upper bounds use
the weighted 2.5%/97.5% forecast quantiles expanded to include the mean. This is a
heuristic forecast envelope, not a Bayesian credible interval or calibrated guarantee.
Admissibility compares the silent-failure upper envelope to the risk class epsilon.
There is no additional statistical risk floor.

`pick` samples one expert by its weight, samples outcome probabilities around that
expert's forecast using the packaged Dirichlet concentration, and chooses the admissible
cell with the smallest sampled cost. Tokens and time come directly from the forecast.
`summary` returns the same means and envelope without sampling a selection. A fixed
seed reproduces selection given frozen forecasts; it does not make inference deterministic.

When no cell meets the cap, `flag-review` selects by sampled cost and requires review;
`lowest-bound` selects the lowest silent upper envelope. `explore_within`, if supplied,
limits eligible cells by expected total cost. Cheaper alternatives include signed cost
component explanations or the policy that excluded them.

## Prices and measurements

Prices belong to the host inventory. If any candidate is unpriced, worker money is
omitted uniformly from every candidate's objective. Known `C_run` remains visible;
unknown values stay null, and `cost_policy` labels the partial risk/time comparison.
Fully priced inventories use the full objective. Never substitute provider API prices
for unknown host charges. Tokens include cached/uncached input and output; T is hours.
Actual `tokens`, `seconds` and `cost_usd` remain null unless measured. Observations
update memory predictive evidence, not a separate fitted statistical model.

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
| `outcome` | clean, loud or silent |
| `cause` | Null for clean; depth, capability or unrecorded otherwise |
| `tokens` | Nonnegative integer or null |
| `seconds`, `cost_usd` | Finite nonnegative number or null |
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
Amendments carry null measurements/cost and correct an outcome to clean, loud or silent,
with the usual cause rule. Superseded rows without timestamps/context and unresolved
`model_default` efforts are refused; no context or control is inferred retrospectively.
Chains replace the original run's effective outcome and cause, counting that run once
and retaining its original ID, decision, cell, task text, project snapshot, project,
weight and uncorrected measurements. Raw rows remain intact. Amendments cannot override
historical context through the observation API. Targets within a file must precede their amendments.
Across files, targets must exist, have dates no later than their amendments, and form an
acyclic dependency graph. Dependency ordering keeps amendments valid when flushing moves
them into the first log ahead of a target in an additional log.

An amendment may additionally carry `measurement_corrections`, a nonempty object with
only `tokens`, `seconds` and/or `cost_usd`. Values follow the corresponding base-field
types and ranges; null explicitly clears a value to unknown. Original rows refuse this
field, and ordinary amendment `tokens`, `seconds` and `cost_usd` must still be null.
For example, `Engine.observe(..., amends="run-id", measurement_corrections={"tokens": 100,
"cost_usd": None})` replaces only those effective values. Unlisted measurements retain
their current effective values across a chain. Supply the intended current outcome/cause
as usual, including when only usage changes. No forecast is created retrospectively.
Worker-cost totals and predictive memory evidence use the corrected original
once. Writer `observation` contains current effective usage; `outcome_history` preserves
the raw original and correction maps as an audit trail, never extra runs or additive usage.
Matching correction-ID retries remain idempotent and different maps conflict. This is
an additive row-semantics expansion; readers without this field's support refuse such rows.

`flush` validates all files before writing. Duplicate refusals list every duplicate and
leave log and journal unchanged. It appends, fsyncs the log, then truncates and fsyncs the
journal. A shared advisory lock serializes storage commands within a project. A crash
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
retrieve/decide. Requests use `howhard-agent-request-v3` and receipts use
`howhard-decision-v4`, with required LLM forecasts.
Earlier formats are refused; prepare a fresh request and decision.
Frozen-receipt observations and learning share the store across that project's worktrees.

## Inference cache and current interfaces

Validated LLM responses may be cached by complete request, adapter identity, settings,
evidence and memory revision. Corrupt caches are replaced. Prepared requests bind the
current worktree, native inventory, effective evidence and memory revision; stale requests
must be prepared again. Calling-agent submissions are audited with unknown external
inference costs and latency rather than being counted as free provider calls.

Receipts use `howhard-decision-v4`, requests `howhard-agent-request-v3`, semantic state
`howhard-semantic-v2`. Older receipts and requests are refused for execution/feedback;
old receipt forecasts are excluded from replay/report scoring. Existing raw observations
and semantic memories remain stored. Old posterior/prior artifacts are unused and left
untouched. Removed interfaces include `--llm`, `HOWHARD_LLM`, `--cache`, `Engine(cache_path=...)`,
`fit`, posterior/prior attributes, `export-prior`, `import-prior`, and prior/cache/transfer
configuration. No migration or distribution upload is performed automatically.

## Python API

```python
from howhard import Engine

engine = Engine()  # State, context, retrieval and feedback need no provider.
request = engine.prepare("specified-implementation", "Make rollback safe",
                         native_candidates=[{"model": "actual-host-model", "effort": "high"}])
# Evaluate baseline_request in an independent model context, then build the response.
# choice = engine.decide(request["request_id"], response, seed=42)
# engine.observe_decision(choice["decision_id"], "clean", id="actual-execution")
```

`Engine(llm=adapter)` accepts an object with `complete(request) -> dict`. Standalone
`pick` and `summary` lazily resolve `HOWHARD_LLM_COMMAND` or an authenticated Claude CLI.
Calling-agent `prepare`/`retrieve`/`decide` never invoke a provider. A fresh engine's
observations return pending learning; apply an external writer response or call
`memory_learn` explicitly. An engine already holding a provider adapter runs its writer
after durable observation. `refresh()` reloads external evidence; no fitting occurs.
Python 3.11+ and POSIX are required; runtime dependencies are limited to NumPy.
