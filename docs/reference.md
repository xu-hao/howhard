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
defaults to `1.0` for pay-as-you-go. A free tier can set it to `0.0`.

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

Neither name changes which cell is chosen. `report` lists both under `routing`.
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

`n_*` are recency-weighted counts from per-cell `outcome_observations`: for each
forecasted `(family, effort)` cell (and any widened cell), up to
`outcome_evidence_limit` (packaged default `24`) task-similar (`score > 0`) `clean`/`loud`/`silent`
executions (`interrupted` runs are excluded) are retrieved
independently from that cell's own historical runs and scored with the configured
reranker (`similarity = sigmoid(score)`), each weighted by
`evidence_weight * similarity * outcome_recency ** age`. `age` counts newer recorded
non-`interrupted` observations of that same cell in chronological journal order, so heavy history on
another cell cannot crowd out this cell's evidence, a regression or recovery dominates
promptly, and unrelated workload does not erase a cell's history. A never-observed
cell keeps its prior variance no matter how many runs other cells have; an expert's
global `evidence_count` affects only its predictive weight, never a cell's sample size.

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
- A token kind forecast at zero is exactly zero unless a non-zero upper end is stated.
- `token_usage_ci` needs `token_usage` on the same row and accepts only the six kinds.
  There is no range on total `tokens` or on credit: token counts do not depend on
  prices, and a stored forecast is reused after prices change.

Stated ranges are stored unchanged in the receipt's forecast panels
(`semantic.worlds`). `seconds_ci` and `token_usage_ci` are validated and stored only:
the duration calibration, cost and sampling described here do not read them, so a row
with these ranges selects exactly as the same row without them. `loud_ci` and
`silent_ci` are read as described above.

`P_clean`, `P_loud` and `P_silent` report `{mean, lower, upper}` from the
support-weighted mixture of the experts' Beta marginals `Beta(alpha_i, sum(alpha) - alpha_i)`:
the mean and the equal-tailed 2.5%/97.5% quantiles, expanded to contain the mean. This is
a real posterior interval even when only the `llm` expert participates. Accepted-result
risk `P_silent / (1 - P_loud)` is, under each Dirichlet, `Beta(alpha_silent, alpha_clean)`;
`P_silent_accepted` reports the same mixture mean and 95% interval. A cell is
`admissible` when its `admission_risk`, the `admission_quantile` (default 0.80) posterior
quantile of accepted risk, is at most the risk class epsilon. Using q80 rather than the
97.5% bound keeps unseen cells with an ordinary forecast admissible while still rejecting
high-silent cells. There is no additional statistical risk floor, and the posterior is a
model assumption, not a calibrated coverage guarantee. `J` uses the posterior means.

`pick` samples one expert by its weight and draws outcome probabilities from that
expert's per-cell Dirichlet posterior `alpha`, and calibrates active duration `T`
alongside outcomes. Forecasted `seconds` represents expected **active** execution
duration excluding external queue wait. For each cell, empirical log-duration residuals
from retrieved similar executions (`effective_active_seconds`, shrunk by
similarity/recency evidence weight so zero-evidence cells preserve the expert forecast)
calibrate log-normal parameters `(mu, sigma)` and price the right-tail expectation
`E[T] = exp(mu + sigma^2 / 2)` in `T` and `J`. `cost_sd` is the delta-method retry-cost
standard deviation from the posterior Dirichlet covariance
`(diag(m) - m m^T) / (sum(alpha) + 1)` and log-normal active-duration variance, raised to
the between-expert spread of per-expert posterior costs when larger. Thompson sampling
considers every admissible, schedulable cell whose posterior cost overlaps the mean-best
cell: `J - J_best <= uncertainty_multiplier * (sd_best + sd_cell)`. Only an explicit
risk-class `explore_within` additionally caps that gap at `explore_within * |J_best|`.
Thompson sampling draws log-normal active duration in `parts["time"]` when duration
variance is present. Receipts retain `outcome_observations`, each sampled cell's
`outcome_posterior` (`prior`, `prior_concentration`, `counts`, `evidence_weight`,
`alpha`), concentrations, duration calibration parameters, each cell's `admission_risk`,
the seed and the reason for a sampled non-best choice. `summary` returns the same
posterior means and intervals without sampling a selection. A fixed seed reproduces
selection given frozen forecasts; it does not make inference deterministic.

Loud attempts are retried on the same cell until a non-loud result is accepted.
`1/(1-P_loud)` charges the expected attempts; `C_loud` is extra detection/rework overhead
per loud failure, in addition to the repeated time/money. The probability floor keeps
a near-certain loud failure finite but very costly. Retry-on-another-cell is not assumed.

If the cap is unmet after stronger-only capability widening, no expert or probability
sample is drawn. `flag-review` chooses the minimum mean J; `lowest-bound` chooses the
lowest accepted-result `admission_risk`, with mean J breaking ties. Both require
review. `exploration.reason` records `disabled-cap-unmet`. Risk-class `explore_within`
can further tighten the posterior-overlap band; it cannot permit exploration outside
it. Cheaper alternatives include signed cost component explanations or the policy that
excluded them.

New receipts use `howhard-decision-v11` (`howhard-semantic-v8`, reader prompt
`semantic-reader-v8`, interval kind `dirichlet-posterior-mixture`). Frozen
`howhard-decision-v10` receipts use the same posterior and selection rules without a
host name, agent labels or ranges on active time and token kinds. Frozen
`howhard-decision-v9` receipts keep undecayed `kappa_0 * p_0 + counts` posteriors, while
frozen `howhard-decision-v8` and `howhard-decision-v7` receipts keep their original weighted
forecast envelopes, upper-envelope admission and sampling policy shape; all load,
validate, accept outcomes and replay into predictive weights.

## Prices and measurements

Prices belong to the host inventory. If any forecasted band/widened candidate is unpriced, worker money is
omitted uniformly from every candidate's objective. Known `session_cost_usd` remains visible;
unknown values stay null, and `cost_policy` labels the partial risk/time comparison.
Fully priced inventories with a uniform rate basis use the full objective. Candidate
`usd_per_mtok` and `rate_card` rates measure list-price quota credit; they do not
establish actual host money cost when that differs. `cost_per_credit` supplies that
conversion. Tokens include cached/uncached input and output; T is calibrated active hours.
Actual `tokens`, `seconds`, optional `active_seconds` and `queue_wait_seconds`, and
`cost_usd` remain null unless measured. When both `active_seconds` and `queue_wait_seconds`
are supplied and `seconds` is omitted, `seconds` is derived as their sum; when all three
are supplied, `abs((active_seconds + queue_wait_seconds) - seconds) <= 1e-6` is enforced.
`effective_active_seconds(row)` uses `active_seconds` when non-null and falls back to
scalar `seconds` otherwise. Observations update both the threshold reader and shared
predictive evidence from one stored row.

An optional candidate `credit_budget` object enforces hard scheduling feasibility against
forecasted `needed_credit_usd`, independently of the session money cost. It requires
nonnegative finite `available_credit_usd` and accepts optional
`session_5h_remaining_credit_usd` and `weekly_7d_remaining_credit_usd` windows; when window fields are
present, `available_credit_usd` must match their minimum within `1e-6`. Any candidate supplying
`credit_budget` must also supply `rate_card` or `usd_per_mtok`, and an inventory that uses
`credit_budget` requires uniform pricing across all candidates so `needed_credit_usd` is
known on a single credit basis. Each evaluated cell reports `session_cost_usd`,
`cost_per_credit`, `needed_credit_usd`,
`credit_budget`, `available_credit_usd` and `credit_ok` (`needed_credit_usd <= available_credit_usd`
when `credit_budget` is present, or `true` when omitted), while
`admissible` remains accepted-result silent-risk admissibility. Selection filters to
schedulable cells (`credit_ok == true`) before sampling, exploration or `on_cap_unmet`
fallback (`lowest-bound` or `flag-review`), refuses selection when no evaluated cell is
schedulable, and labels cheaper unschedulable alternatives with `insufficient-credit` in
`why`.

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
`howhard-capability-v6`, whose cells each carry their `agent` label. Its `band` is an ordered list of forecast cells that can omit
`intermediate` cells between its `lower` and `upper` endpoints. See
[capability filtering](capability.md) for the band and widening rules.
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
`selection` includes objective, retry policy, and cap fallback. Indexes cover creation
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
