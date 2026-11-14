# LLM reasoning with writable semantic memory

howhard can use an LLM to read the original task and project context, retrieve earlier
experience, and reason with persistent semantic memories. Feedback updates predictive
weights and gives an LLM writer an opportunity to add, revise, remove, merge, split or
restore memories. There is no fixed list of semantic concepts or fixed-size memory bank.

Fine forecasts come from the independent no-memory LLM and memory-conditioned overrides.
A [coarse threshold reader](capability.md) first learns a task-specific fit band from
the same execution store and predictive weights. It adds no floor to fine-forecast risk. Task and project inputs
remain their original text.

## Provider and calling-agent use

Standalone `pick`/`summary` lazily use an authenticated Claude CLI or the command in
`HOWHARD_LLM_COMMAND`. Python can supply `Engine(llm=adapter)`. Provider failure or
absence stops routing explicitly; there is no fallback recommendation and no `--llm`
mode switch. Commands for state, memory, evidence and the calling-agent protocol need
no configured provider and do not start nested inference.

```sh
howhard context specified-implementation --candidates /tmp/host-candidates.json --text "Make transaction rollback safe" --json
# Evaluate the filter baseline independently, then add memory_filters:
howhard decide FILTER_REQUEST_ID --file /tmp/filter.json --json
# Evaluate the returned forecast baseline independently, then add memory_forecasts:
howhard decide FORECAST_REQUEST_ID --file /tmp/forecasts.json --json
howhard observe DECISION_ID clean --id EXECUTION_ID --json
howhard memory learn-context EXECUTION_ID --json
```

An observation made through a fresh engine returns pending learning. Apply the calling
agent's writer response, or explicitly invoke `memory learn EXECUTION_ID` for provider
writing. An engine already using a provider adapter also writes automatically after
observation; failures leave feedback durable and can be retried.

## Persistent state and real memory operations

Memories live in `memory.sqlite3` within the backend state root reported by `howhard paths`. Each has an ID, content version, title, free-form
body, project/shared scope, evidence references and parent links. Scope describes intended
applicability inside this project's store; `shared` does not automatically upload or copy
memory to other directories. Memories remain in this project; there is no generic prior transfer.

Within that store, learning is shared across candidate models and providers. Retrieval
uses the task and project, without filtering history to today's available models. An
experience from one provider can inform a prospective forecast for another through the
reader and written lessons. Its actual outcome remains attributed only to the executed
model/control, and predictive support requires a later observed forecast exposure.
An external router can supply a list or explicitly select a
{ref}`configured inventory <external-router-inventories>`
and use this learning without a native agent plugin.

```sh
howhard memory add --title "Migration boundaries" \
  --body "Cross-service migrations without integration coverage need boundary reasoning."
howhard memory edit MEMORY_ID --body-file revised-lesson.txt
howhard memory remove MEMORY_ID
howhard memory restore MEMORY_ID
howhard memory merge ID_A ID_B --title "Combined lesson" --body-file combined.txt
howhard memory split MEMORY_ID --file parts.json
howhard memory read MEMORY_ID --version 1
howhard memory history MEMORY_ID
howhard memory search "rollback"
howhard memory list --all
```

`parts.json` is a list of objects with `title` and `body`. A batch file accepted by
`memory apply --file PATH --expected-revision N` is a list of operations with an `op`
and the corresponding fields. For example:

```json
[
  {"op": "add", "title": "Rollback", "body": "A provisional hypothesis", "evidence_ids": []},
  {"op": "remove", "id": "existing-id"}
]
```

Batches are transactional: one invalid operation changes nothing. Writer edits retain
their expected revision through the backend's [write queue](concurrency.md). Memory
mutations share admission with observations, amendments, receipts and cache publication;
revision and evidence-event checks run after waiting. Clients do not coordinate writer
windows through conversation channels. The queue never silently rebases a stale patch.
Evidence IDs must refer to
recorded runs. Edits create new content versions and receive no inherited predictive
credit. Merge and split create new IDs and preserve parent/evidence links without
combining old scores. Remove stops active use; archived versions, evidence and audit
history remain readable. Restore reactivates the current content version with its
own earlier evaluation history. No operation deletes original outcome evidence.

## Reader and retrieval

The reader receives the original task, a freshly captured project snapshot, candidate
model/effort cells, current belief summaries, semantic memories and relevant recorded
experience. [Execution experience retrieval](retrieval.md) uses BM25 plus HNSW, fuses
up to 200 candidates, and applies a compact pretrained MiniLM-L4 INT8 cross-encoder to select up to eight
observed executions/written lessons. Relevance includes broader/narrower task demands
and project constraints. The calling agent remains the final reasoning stage.
It can request bounded additional `memory.search`, `memory.read`, `evidence.search`,
`evidence.read` and `project.read` operations. Retrieval/reranking stages are replaceable.
No retrieval recall or routing-quality guarantee is implied.

`project.read` is limited to regular project source/documentation/manifest files inside
the project and the bounded snapshot's file set. Ignored/excluded files cannot be read.
Source reads must match the prepared snapshot; concurrent project changes
require a fresh decision. Retrieved text is saved in the receipt so later feedback does
not read today's replacement file. Project text and notes are treated as data, not
instructions to execute commands.

Separate stateless calls produce the task-specific capability prior and the fine
no-memory forecast for the retained band. Its request contains only
the current task/project, candidate identities and policy constants: no memories,
historical evidence, learned probability table or prior conversation. A memory reader
cannot rewrite that baseline. Category rate-card cells carry `forecast_format: "category-v1"`
and require all six nonoverlapping `token_usage` values plus their consistent scalar
`tokens` sum. Scalar cells use the legacy total forecast; no category split is guessed.
See the {ref}`pricing and measurement contract <portable-execution-accounting>`. It supplies explicit conditional forecast overrides for
memories it has actually seen. This separation is necessary for meaningful comparisons.

The original semantic text, retrieved evidence, memory versions and forecasts are
retained. The final numerical forecasts are decision outputs, not the sole stored
representation of the task. Repeated identical requests can reuse a validated inference
cache keyed by inputs, model/adapter identity, prompt/settings, evidence and memory state.

## Feedback and approximate belief weights

Actual executions also supply read-only `execution:RUN_ID` experts; original model,
effort, outcome and measured usage remain attached. These experts gain predictive
credit only from later observed actions, never from their own source execution.
Amendments replace their effective content version. For a decision, support is normalized
over the independent forecast and memories that supply overrides; unrelated memories
do not dilute the selected evidence. See the [retrieval contract](retrieval.md).

For the cell actually executed, the system compares each pre-recorded expert forecast
with the independent no-memory LLM forecast. Log probability ratios for clean/loud/silent outcomes,
plus broad log-normal resource scores when measurements are present, accumulate into
tempered predictive weights. Older exposures are mildly discounted so weights can adapt.
All scales and priors ship in `data/semantic.toml`; users do not tune them.

Only explicit, changed memory overrides earn exposure. Unspecified/unretrieved cells
use the independent no-memory LLM forecast and provide no credit to that memory. A new or edited
memory cannot gain support from a run that preceded its forecast. Feedback on an
unrelated cell cannot validate a memory-specific claim. Model registry evidence weights
also scale these updates.

These are approximate predictive-evidence weights, not probabilities that statements
are true and not an exact posterior over all possible natural-language memories.
They measure predictive support and do not establish causal explanations. The LLM's
own confidence is not evidence. Outcome amendments to clean, loud or silent replay the
original run's forecasts against its effective result once, replacing its earlier score
rather than adding another independent run. Original measurements and registry evidence
weight still apply; an amendment does not create a new forecast exposure.

Selection samples the LLM and memory experts and retains the cost objective. Intervals
use the weighted forecast envelope, expanded to contain its weighted mean. The silent
cap checks this envelope directly, with no fitted statistical floor. Tiny-weight
pessimistic experts do not automatically veto every cell. This is a heuristic envelope,
not a calibrated risk guarantee. Results identify the interval method and preserve the
review-required policy when the cap is unmet.

## Writer, failures and replay

After `observe`, an LLM writer sees the effective result, the full `outcome_history` of
original/amendment rows and notes, original decision, supporting
experience, beliefs and current memory. It can emit any supported memory operation.
The backend validates and commits the operations atomically. It rejects stale memory
or an outcome that was amended while the writer ran. The triggering experience may
motivate a new memory, but does not become retrospective predictive validation.

Observation storage completes before inference. A failed writer does not lose the
outcome; the CLI returns `memory_update.status = failed`. Retry with
`howhard memory learn RUN_ID`. Repeated learning of an unchanged effective run is
idempotent. Each appended amendment makes a new effective learning event, even when a
chain returns to an earlier classification. Learning by the original ID or any amendment
ID resolves to the same current event. A failed correction writer can be retried without
recording another outcome. Pending writer responses are rejected after any correction,
including a chain that returns to the result the writer originally saw. Usage attachment
uses the same amendment and event workflow: the effective `execution_usage` replaces the
prior envelope, while `outcome_history` preserves raw claims and corrections. Collection
failure leaves the outcome durable and does not create a second execution.

Routing seeds reproduce selection given recorded forecasts. They do not make fresh
LLM calls deterministic. Cached forecasts and content-addressed receipts support replay;
operational timing/cache metadata may differ between invocations.
Receipts use `howhard-decision-v7` with capability and accepted-result selection state.
Requests use `howhard-agent-request-v5` and semantic state uses `howhard-semantic-v4`. Older
decision receipts v1–v6 remain in the store but are excluded from predictive scoring; raw
evidence already linked to them remains intact.

Native host selection supplies `--candidates FILE` to `context` or `pick` before any
baseline/reader forecasts. The inventory is generated from actual host controls, not a
project adapter. It is bound to the saved request; retrieve/decide cannot replace it.
Unpriced native inventories uniformly exclude worker money and label that limitation.
Direct `evidence RUN_ID [--raw]` reads feedback without preparing another decision.
Matching stable-ID retries preserve the first observation; conflicts remain explicit.
Raw `observe --native` records a different verified execution without attaching an
unexecuted forecast. State is shared automatically across fresh Git worktrees, with
current project capture kept separate; inspect `paths` for the resolved roots.

(calling-agent-provider-interfaces)=
## Calling-agent and provider interfaces

An agent that already has an LLM can use howhard as a stateful tool backend:

```sh
howhard context specified-implementation --text-file task.txt --candidates /tmp/host-candidates.json --json
howhard retrieve REQUEST_ID --file retrieval-queries.json --json
howhard decide FILTER_REQUEST_ID --file filter.json --json
howhard decide FORECAST_REQUEST_ID --file forecasts.json --seed 42 --json
howhard memory learn-context RUN_ID --json
howhard memory learn RUN_ID --file writer-response.json \
  --expected-revision N --event-key EVENT_KEY --json
```

`context` returns `stage=filter` with a clean `baseline_request` and a separate memory
request. Evaluate the baseline in a separate model context, then submit
`{filter, memory_filters, reason}`. `decide` returns a new `stage=forecast` request ID
for calibrated rows over the fit band. Evaluate that clean baseline independently and
submit `{forecasts, memory_forecasts, reason}`. Handle explicit re-filter or stronger-only
widening requests until a decision ID is returned. The full inventory remains bound
throughout; follow-up commands never replace it. Retrieval returns a new request ID including frozen results.
Unrelated writes do not invalidate an in-flight request: belief/expert weights and
capability inputs use the preparation snapshot. Changed surfaced dependencies return
`StaleRequest` with the changed IDs/versions and a refreshed request, as documented in
[concurrent clients](concurrency.md). This protocol never claims the calling agent's model
calls are free: externally supplied forecasts and updates are audited separately, with
unknown model cost and latency. The host can retain its own measured usage alongside
these submissions; they are not included in the package's measured call count.

Python integrations pass an object with `identity` and `complete(request) -> dict` to
`Engine(llm=adapter)`. Its calls must be stateless for the isolated baseline. Optional
`last_usage` reports `seconds`, `cost_usd`, raw `tokens` and resolved `models`.
`HOWHARD_LLM_COMMAND` may be a JSON argument array; the adapter sends
`{"system": "...", "request": {...}}` to stdin and accepts a JSON object on stdout.
It can instead return `{"response": {...}, "usage": {...}}`. No shell interpolation is
used. Timeouts stop the process group. Command identities are hashed so arguments do
not appear in receipts. Nested Claude inference uses the calling-agent protocol.

## Measurement and evaluation

Actual baseline/reader/writer calls, including errors, are recorded in
`inference.jsonl` in the backend state root with provider usage and resolved models. Decision attempts
record wall-clock duration, failures and an explicitly unknown action propensity in
`decision-events.jsonl` in that same store. Outcome rows record when feedback became available.
Inference cost stays unknown when the adapter cannot supply it; it is never silently
reported as zero. Writer cost is included in inference reports.
Claude CLI dollar figures are CLI-reported accounting estimates, not a statement of
the user's actual subscription invoice. Preserve and compare the same cost basis across
evaluation arms; custom adapters should document the meaning of their reported costs.

`howhard report --json` scores forecasts only for actions with observed outcomes,
reports log loss/Brier scores and measured cost/latency, and accounts for amendments.
These descriptive scores cannot recover unrun alternatives' outcomes or establish
routing superiority. Follow the [real-project evaluation protocol](evaluation.md).


## Two readers and accepted-result selection

Stage-1 prior/memory experts join the existing belief replay under `filter:` keys.
Their frozen binary clean/non-clean predictions are log-scored against the fine
independent forecast coarsened to the same binary outcome. Stage 2 retains ternary
outcome and measured resource scores. The same temperature, recency, prior weights
and evidence-count machinery serves both, normalized over each reader's participants.
The threshold posterior applies the retrieved censored likelihood to the weighted
stage-1 prior; it has no independent observation store or feedback path.

Fine selection charges expected same-cell retries to an accepted result. Its silent
cap uses joint-expert accepted-result defect ratios. Sampling uses envelope/spread and
recorded expert exposures to calibrate bounded Dirichlet concentration. Mean-cost and
cost-uncertainty bands restrict exploration; an unmet cap disables all draws and
requires review. Receipts preserve the formula, constants, calibration and explanation.
A consistency disagreement compares raw stage-2 `P_clean` with stage-1 `P_clear`; it
does not substitute the accepted-result risk ratio or silently alter the chosen cell.
See [capability filtering](capability.md) and {ref}`selection <accepted-result-selection>`.
