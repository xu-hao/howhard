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
`effort` and optional `usd_per_mtok` or `rate_card`, `cost_per_credit`, and `credit_budget`. External routers can use stable
provider-qualified model IDs; execution and provider access belong to the caller.
See {ref}`inventory configuration <external-router-inventories>`.
No-effort, null and `model_default` controls are refused. Unknown identities receive distinct
opaque family labels for protocol bookkeeping, without model assumptions or effects.
Configured identities retain their declared labels and evidence weights. Historical
identity metadata does not activate candidates in a later request.

A [task-specific capability filter](capability.md) first reads the full inventory.
An independent stateless LLM then forecasts only the fit band using the task and project alone.
The reader supplies conditional overrides for memories it has seen; omitted cells and
unretrieved memories use the independent LLM forecast. No learned probability table
or synthetic forecast is supplied to either call. Provider absence, invalid forecasts
and provider failures are explicit errors; none produces a routing recommendation.

Probability means are weighted by predictive expert support. Lower/upper bounds use
the weighted 2.5%/97.5% forecast quantiles expanded to include the mean. This is a
heuristic forecast envelope, not a Bayesian credible interval or calibrated guarantee.
Admissibility compares the accepted-result defect envelope to the risk class epsilon.
For each expert, transform its joint row to `P_silent / max(probability_floor, 1-P_loud)`
before taking envelope quantiles, then include the ratio of weighted means. This retains
the expert's loud/silent relationship instead of dividing unrelated marginal bounds.
There is no additional statistical risk floor.

`pick` samples one expert by its weight and draws outcome probabilities using an
evidence-calibrated Dirichlet. The concentration uses the maximum of between-expert
variance, squared envelope width divided by `3.92²`, and
`p*(1-p)/(prior_strength + expert_evidence_count + 1)`. Moment-matching gives
`p*(1-p)/variance - 1`; take the most conservative outcome and apply configured bounds.
Agreeing, well-observed experts tighten; weak priors and disagreement stay wider.
Selection admits exploration only inside both the mean-J fraction and a delta-method
cost-uncertainty band. Receipts retain concentrations, evidence counts, variance sources,
the seed and the reason for a sampled non-best choice. Tokens and time come directly from the forecast.
`summary` returns the same means and envelope without sampling a selection. A fixed
seed reproduces selection given frozen forecasts; it does not make inference deterministic.

Loud attempts are retried on the same cell until a non-loud result is accepted.
`1/(1-P_loud)` charges the expected attempts; `C_loud` is extra detection/rework overhead
per loud failure, in addition to the repeated time/money. The probability floor keeps
a near-certain loud failure finite but very costly. Retry-on-another-cell is not assumed.

If the cap is unmet after stronger-only capability widening, no expert or probability
sample is drawn. `flag-review` chooses the minimum mean J; `lowest-bound` chooses the
lowest accepted-result silent upper envelope, with mean J breaking ties. Both require
review. `exploration.reason` records `disabled-cap-unmet`. Risk-class `explore_within`
can further tighten the default sampling band; it cannot permit exploration outside
the uncertainty-supported band. Cheaper alternatives include signed cost
component explanations or the policy that excluded them.

## Prices and measurements

Prices belong to the host inventory. If any forecasted band/widened candidate is unpriced, worker money is
omitted uniformly from every candidate's objective. Known `session_cost_usd` remains visible;
unknown values stay null, and `cost_policy` labels the partial risk/time comparison.
Fully priced inventories with a uniform rate basis use the full objective. Candidate
`usd_per_mtok` and `rate_card` rates measure list-price quota credit; they do not
establish actual host money cost when that differs. `cost_per_credit` supplies that
conversion. Tokens include cached/uncached input and output; T is hours.
Actual `tokens`, `seconds` and `cost_usd` remain null unless measured. Observations
update both the threshold reader and shared predictive evidence from one stored row.

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
span. It never discovers transcripts, retains prompt/code text, or contacts providers.
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
bases separately and preserve unknown coverage instead of treating missing values as zero.

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
only `tokens`, `seconds`, `cost_usd` and/or `execution_usage`. Values follow the corresponding base-field
types and ranges; null explicitly clears a value to unknown. Original rows refuse this
field, and ordinary amendment `tokens`, `seconds` and `cost_usd` must still be null.
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
retrieve/decide. Requests use `howhard-agent-request-v5` and receipts use
`howhard-decision-v7`, with required capability and accepted-result selection state.
Receipts v1–v6 remain recognizable as historical store entries but are not replayed as
new decisions; prepare a fresh request and decision to attach new feedback.
Frozen-receipt observations and learning share the store across that project's worktrees.

## Inference cache and current interfaces

Validated LLM responses may be cached by complete request, adapter identity, settings,
evidence and memory revision. Corrupt caches are replaced. Prepared requests bind the
current worktree, native inventory and a consistent store snapshot. New requests validate
only surfaced memory/evidence dependencies and configuration, with belief and capability
weights frozen at preparation. Unrelated writes do not invalidate them. A real change
returns a typed conflict and refreshed request; see [concurrent clients](concurrency.md).
Calling-agent submissions are audited with unknown external
inference costs and latency rather than being counted as free provider calls.

Receipts use `howhard-decision-v7`, requests `howhard-agent-request-v5`, semantic state
`howhard-semantic-v4`. Older receipts and requests are refused for new execution feedback;
v1–v6 receipt forecasts are excluded from replay/report scoring. Existing raw observations
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
`Engine(write_timeout=30, write_cancel=callback)` bounds admission and supports
cancellation. `write_status()` and CLI `writes --json` expose contention. See
[concurrent clients](concurrency.md) for boundaries, typed errors and stable-ID retries.
Python 3.11+ and POSIX are required; runtime dependencies are limited to NumPy.
