# LLM reasoning with writable semantic memory

howhard can use an LLM to read the original task and project context, retrieve earlier
experience, and reason with persistent semantic memories. Feedback updates predictive
weights and gives an LLM writer an opportunity to add, revise, remove, merge, split or
restore memories. There is no fixed list of semantic concepts or fixed-size memory bank.

The existing statistical model is retained as a competing predictor and a conservative
risk guard. The LLM path uses raw task text and available project text, not the lexical
hash vector as its semantic representation.

## Start without statistical configuration

```sh
howhard pick specified-implementation --text "Make transaction rollback safe"
# Execute using the selected model and effort, then record the printed receipt ID:
howhard observe DECISION_ID clean --tokens 12000 --seconds 45
howhard memory list
howhard memory beliefs
howhard report
```

The default `auto` backend reuses an installed Claude CLI and its existing authentication.
The adapter runs in a temporary directory with tools, MCP servers, skills, customizations
and session persistence disabled. It uses a bounded inference budget. An explicit
`--llm claude` request fails clearly if unavailable or invalid. `auto` reports a visible
statistical fallback on provider failure; it never labels that result as LLM reasoning.
`--llm off` uses the existing statistical workflow without inference calls.

An LLM service still requires access to a model. No statistical tuning or additional
API key is needed when an authenticated CLI is already present. Standalone use without
such access can use a command adapter, Python adapter, or the calling-agent protocol
below. This package does not manufacture credentials or silently invoke a different
provider.

## Persistent state and real memory operations

Memories live in `.howhard/memory.sqlite3`. Each has an ID, content version, title, free-form
body, project/shared scope, evidence references and parent links. Scope describes intended
applicability inside this project's store; `shared` does not automatically upload or copy
memory to other directories. Semantic memory is not included in generic prior exports.

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

Batches are transactional: one invalid operation changes nothing. Writer edits carry
the revision they read and reject concurrent stale writes. Evidence IDs must refer to
recorded runs. Edits create new content versions and receive no inherited predictive
credit. Merge and split create new IDs and preserve parent/evidence links without
combining old scores. Remove stops active use; archived versions, evidence and audit
history remain readable. Restore reactivates the current content version with its
own earlier evaluation history. No operation deletes original outcome evidence.

## Reader and retrieval

The reader receives the original task, a freshly captured project snapshot, candidate
model/effort cells, current belief summaries, semantic memories and relevant recorded
experience. It can request bounded additional `memory.search`, `memory.read`,
`evidence.search`, `evidence.read` and `project.read` operations. Search is a lightweight
text index/shortlist; the LLM judges relevance and can reformulate queries. This first
version does not claim a learned semantic retriever or guaranteed retrieval recall.

`project.read` is limited to regular project source/documentation/manifest files inside
the project and the bounded snapshot's file set. Ignored/excluded files cannot be read.
Source reads must match the prepared snapshot; concurrent project changes
require a fresh decision. Retrieved text is saved in the receipt so later feedback does
not read today's replacement file. Project text and notes are treated as data, not
instructions to execute commands.

A separate stateless call produces the no-memory forecast. Its request contains only
the current task/project, candidate identities and policy constants: no memories,
historical evidence, learned probability table or prior conversation. A memory reader
cannot rewrite that baseline. It supplies explicit conditional forecast overrides for
memories it has actually seen. This separation is necessary for meaningful comparisons.

The original semantic text, retrieved evidence, memory versions and forecasts are
retained. The final numerical forecasts are decision outputs, not the sole stored
representation of the task. Repeated identical requests can reuse a validated inference
cache keyed by inputs, model/adapter identity, prompt/settings, evidence and memory state.

## Feedback and approximate belief weights

For the cell actually executed, the system compares each pre-recorded expert forecast
with the statistical baseline. Log probability ratios for clean/loud/silent outcomes,
plus broad log-normal resource scores when measurements are present, accumulate into
tempered predictive weights. Older exposures are mildly discounted so weights can adapt.
All scales and priors ship in `data/semantic.toml`; users do not tune them.

Only explicit, changed memory overrides earn exposure. Unspecified/unretrieved cells
use the statistical baseline and provide no credit to that memory. A new or edited
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

Selection samples competing predictors and retains the cost objective. Semantic
intervals use a weighted forecast envelope together with the statistical model's
uncertainty. The statistical silent-failure upper bound remains a lower floor on the
cap check: semantic memory can widen it but cannot by itself clear a statistically
rejected cell. Tiny-weight pessimistic experts do not automatically veto every cell.
This is an explicit conservative policy heuristic, not a calibrated semantic 97.5%
posterior guarantee. Results identify the interval method and preserve review fallback.

## Writer, failures and replay

After `observe`, an LLM writer sees the effective result, original decision, supporting
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
including a chain that returns to the result the writer originally saw.

Routing seeds reproduce selection given recorded forecasts. They do not make fresh
LLM calls deterministic. Cached forecasts and content-addressed receipts support replay;
operational timing/cache metadata may differ between invocations. Existing logs,
v1 receipts and generic prior formats remain readable. Semantic receipts use v2.

## Calling-agent and provider interfaces

An agent that already has an LLM can use howhard as a stateful tool backend:

```sh
howhard context specified-implementation --text-file task.txt --json
howhard retrieve REQUEST_ID --file retrieval-queries.json --json
howhard decide REQUEST_ID --file forecasts.json --seed 42 --json
howhard memory learn-context RUN_ID --json
howhard memory learn RUN_ID --file writer-response.json \
  --expected-revision N --event-key EVENT_KEY --json
```

`context` returns a saved request and a separate clean `baseline_request`. For a genuine
no-memory ablation, evaluate that baseline in a separate model context. The response for
`decide` has `forecasts`, `memory_forecasts` and `reason`, as described in the request's
contract. Retrieval returns a new request ID including frozen results. Stale evidence or
memory requires preparing again. This protocol never claims the calling agent's model
calls are free: externally supplied forecasts and updates are audited separately, with
unknown model cost and latency. The host can retain its own measured usage alongside
these submissions; they are not included in the package's measured call count.

Python integrations pass an object with `identity` and `complete(request) -> dict` to
`Engine(llm=adapter)`. Its calls must be stateless for the isolated baseline. Optional
`last_usage` reports `seconds`, `cost_usd`, raw `tokens` and resolved `models`.
`HOWHARD_LLM_COMMAND` may be a JSON argument array; `--llm command` sends
`{"system": "...", "request": {...}}` to stdin and accepts a JSON object on stdout.
It can instead return `{"response": {...}, "usage": {...}}`. No shell interpolation is
used. Timeouts stop the process group. Command identities are hashed so arguments do
not appear in receipts. Nested Claude inference uses the calling-agent protocol.

## Measurement and evaluation

Actual baseline/reader/writer calls, including errors, are recorded in
`.howhard/inference.jsonl` with provider usage and resolved models. Decision attempts
record wall-clock duration, failures and an explicitly unknown action propensity in
`.howhard/decision-events.jsonl`. Outcome rows record when feedback became available.
Inference cost stays unknown when the adapter cannot supply it; it is never silently
reported as zero. Writer cost is included in inference reports.
Claude CLI dollar figures are CLI-reported accounting estimates, not a statement of
the user's actual subscription invoice. Preserve and compare the same cost basis across
evaluation arms; custom adapters should document the meaning of their reported costs.

`howhard report --json` scores forecasts only for actions with observed outcomes,
reports log loss/Brier scores and measured cost/latency, and accounts for amendments.
These descriptive scores cannot recover unrun alternatives' outcomes or establish
routing superiority. Follow the [real-project evaluation protocol](evaluation.md).
