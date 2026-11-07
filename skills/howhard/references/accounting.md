# Portable execution accounting

Use this reference when recording actual usage or supplying known category prices.
The backend accepts evidence; it does not obtain credentials, install capture hooks,
discover transcripts or infer subscription charges. Outcome assessment and accepted
model/effort controls do not establish token usage.

## Capture one explicitly owned invocation

For an authorized new CLI invocation, capture its structured output directly in a
task-owned file. Choose the host's actual supported controls before launching. These
examples illustrate capture, not permission to run another worker or repeat completed work:

```sh
codex exec --json --model ACTUAL_MODEL -c model_reasoning_effort=ACTUAL_EFFORT 'AUTHORIZED_TASK' > /tmp/EXECUTION_ID-codex.jsonl
claude -p --verbose --output-format stream-json --model ACTUAL_MODEL 'AUTHORIZED_TASK' > /tmp/EXECUTION_ID-claude.jsonl
```

Use `codex-exec-v1` for the first artifact and `claude-sdk-v1` for the second. Record
the concrete host version from `codex --version` or `claude --version`, the exact session
identity emitted by the invocation, and the completed source byte range. A failed launch
is not a completed invocation. Preserve the original artifact on failure and never run a
second task merely to manufacture usage for the first.

Supported source schemas are `0.166.0` for both Codex adapters,
`2.1.318` for Claude sessions, and the verified underlying Claude Code runtime
`2.1.318` for the SDK adapter. An SDK package version does not prove the runtime's
cumulative accounting behavior; do not supply an SDK package version as `host_version`.
Verify the actual underlying Claude Code runtime even when the SDK package version
differs. Set `host_version` to the exact supported runtime identity; another version is refused,
not silently parsed. A successful command alone does not prove supported telemetry.

For `claude-sdk-v1`, also capture the session's cumulative `modelUsage` immediately
before the invocation. Supply `baseline: {"session_id": "EXACT_EMITTED_SESSION_ID",
"modelUsage": {}}` only for an explicitly fresh session with no inherited usage; a
resumed session instead needs its actual prior per-model counters and cost. Missing
baseline makes usage unavailable; decreasing counters or mismatched identity are refused.
The result UUID identifies the invocation. Always supply `included_sessions`, even an
empty list when no independent host sessions were included. Native Claude children
sharing the parent session use the separate exact `included_agents` mapping below.

For an already running session, use `codex-session-v1` or `claude-session-v1` only when
the host exposes the exact owned session artifact and an exclusive byte span. Save its
starting byte position before the invocation and ending byte position after it; capture
session identity from that same artifact. A resumed session needs the exclusive new span,
not the whole cumulative history. A resumed Codex session additionally needs a
`baseline` object containing its measured starting cumulative `input_tokens`,
`cached_input_tokens`, `cache_write_input_tokens`, `output_tokens` and any other
reported fields. Missing baseline on a nonzero starting span, or gaps between cumulative
and last-frame usage, keeps the collection partial. If these boundaries were not captured, or native
collaboration exposes no exact artifact-to-child identity binding, leave child usage
unknown. A shared `CODEX_THREAD_ID`, matching model or nearby timestamp is insufficient.
Do not search session directories or select a transcript by recency.

The manifest is an explicit ownership claim. Write JSON outside project source,
replacing every example value with verified invocation evidence:

```json
{
  "format": "howhard-usage-manifest-v1",
  "execution_id": "EXECUTION_ID",
  "host_version": "0.166.0",
  "adapter": "codex-exec-v1",
  "source": "/tmp/EXECUTION_ID-codex.jsonl",
  "session_id": "EXACT_EMITTED_SESSION_ID",
  "scope": "self",
  "start": 0,
  "end": 1234,
  "completed": true,
  "provider": "openai",
  "model": "ACTUAL_MODEL",
  "service_tier": null
}
```

`start` and `end` are byte offsets, not lines or character counts; use the exact captured
range, with `end` exclusive. Optional `coordinate_id` names the same append-only
artifact across relocations; its default is the absolute source path plus adapter.
Offsets on different artifact coordinates cannot establish disjoint ownership for the
same session. A newly created output file starts at zero; obtain its final
length with `wc -c < /tmp/EXECUTION_ID-codex.jsonl`. The collector permits at most a 16 MiB
span in a source no larger than 32 MiB. Keep host/schema identity explicit. The SDK's
child-inclusive result needs `scope: "inclusive"` and explicit ownership identities.
Use `included_sessions` only for actual independent host session IDs. Native Claude
sidechains retain the parent's `sessionId` and have their own `agentId`; record those
exact child IDs in `included_agents`, never as session aliases. For example:

```json
{
  "adapter": "claude-sdk-v1",
  "host_version": "2.1.318",
  "session_id": "EXACT_PARENT_SESSION_ID",
  "scope": "inclusive",
  "included_sessions": [],
  "included_agents": ["EXACT_CHILD_AGENT_ID"],
  "baseline": {"session_id": "EXACT_PARENT_SESSION_ID", "modelUsage": {}}
}
```

These fields extend the full manifest with SDK-specific ownership; the empty baseline is still valid only
for an explicitly fresh session. An `included_agents` child claim uses the canonical
identity `EXACT_PARENT_SESSION_ID:agent:EXACT_CHILD_AGENT_ID`, matching a separately
bound Claude-session artifact with `session_id: "EXACT_PARENT_SESSION_ID"`,
`agent_id: "EXACT_CHILD_AGENT_ID"`, `isSidechain: true` and that exact `agentId`.
This prevents attachment of the same child under another execution in either order.
A flat `included_sessions: ["EXACT_CHILD_AGENT_ID"]` claims a different host session;
it never implicitly maps a native agent. Both child arrays require distinct actual IDs;
nonempty arrays require inclusive scope. Without exact parent/agent evidence, leave
child ownership and usage unknown. Do not also record included children as separate
executions. For an exact owned Claude sidechain, `agent_id` must match the artifact's `agentId`
and its parent `sessionId`; never infer either identity. Do not claim inclusion when
the host cannot expose child identities. The store refuses overlapping source/session
claims across effective original runs, including separate command invocations. Clear or
replace a wrongly attributed claim before attaching that source to its truthful owner;
raw history remains intact.

A final supported usage event and a completed boundary are both necessary for complete
usage. Task success is separate: Claude SDK `error_max_turns` and
`error_max_budget_usd` with `is_error: true` are bounded terminal results and may have
complete usage. `success` requires `is_error: false`; contradictory or untyped flags
are refused. An uncertain crash subtype such as `error_during_execution` remains
partial, even when counters are present. Interrupted, malformed, reset or unsupported records must never become guessed
complete totals. Unknown categories remain null. Codex input includes cached input:
subtract cache reads once; reasoning is already part of output. Claude input excludes
cache reads and writes. Cache-write TTL buckets partition creation; generic `cache_write`
is only the residual whose TTL is unknown, not another copy of the known TTL buckets.

## Persist outcome, collect, then attach

Record the actual outcome using the matched receipt or verified raw actual cell from
[the decision workflow](workflow.md). Collect and attach with task-owned temporary paths:

```sh
howhard usage collect --manifest /tmp/EXECUTION_ID-usage-manifest.json > /tmp/EXECUTION_ID-usage.json
howhard usage attach EXECUTION_ID --manifest /tmp/EXECUTION_ID-usage-manifest.json --id EXECUTION_ID-usage-1 --json
howhard evidence EXECUTION_ID --json
```

`collect` emits a `howhard-execution-usage-v1` envelope without changing outcomes.
`attach` appends a measurement correction to the existing execution. An outcome remains
durable if collection or attachment fails. Reconcile ambiguous writes with
`evidence EXECUTION_ID-usage-1 --raw --json` and retry unchanged with the same amendment
ID once. After a repeated failure, report an infrastructure incident, leave the outcome
intact and park collection. Later evidence can use a new correction ID; never create a
second original execution to retry. Do not retry an invocation to repair accounting.

If a validated envelope already exists before the original outcome is written, pass
`observe ... --execution-usage /tmp/EXECUTION_ID-usage.json`. Use
`--measurement-corrections FILE` with `{"execution_usage": null}` to explicitly clear
an incorrect envelope, or with a complete replacement envelope to correct it. Amendments
replace usage rather than add a run. Do not overwrite the original row or add a ledger.
Run memory learning for the current event after usage corrections as described in the
[decision workflow](workflow.md#corrections-and-learning).

## Supply explicit prices and category forecasts

Measured tokens need no prices. Client/SDK `total_cost_usd` is an estimate. If an SDK
aggregate and all per-model costs are known, their totals must reconcile; a mismatch
is refused as an unsupported cost scope. Preserve the outcome, report the discrepancy
and park collection after its bounded retry; do not silently omit or invent fees.
An absent aggregate still permits supported per-model client estimates.
`billed_cost` needs an authoritative billing record tied to this execution, with
`authoritative: true`, `execution_id`, `receipt_id`, `source`, `version`, and `usd`.
The source/receipt ID identifies a unique billing line within the envelope and across
effective executions; reusing that line is refused. A verified bill is independent of
token-category coverage or an incomplete source boundary. Preserve the authoritative
receipt with partial/unknown tokens; billed totals stay known when every segment has a
receipt. A client estimate requires a completed usage boundary but may remain known
when token categories are partial. A category rate calculation still needs its category
measurements; do not infer missing categories from a known estimate or bill. Never label
an API list-price calculation or subscription run estimate as a billed charge. Legacy
`cost_usd` has an unspecified basis. Reports keep estimated, billed and legacy totals
separate and retain unknown coverage.

Obtain an applicable rate card from the user's contract, host billing documentation, or
provider pricing source only when that source applies to the actual host product, model,
service tier and effective date. For example, a user's approved local JSON rate card can
be copied into the task's candidate file after checking those identities and all six
category units against the source. Retain its exact source reference and version. An API
price page alone does not establish a subscription host's charge. If applicability or
any required rate is unknown, leave the price unknown. There is no built-in price lookup.

An illustrative synthetic candidate (the numbers are fixture rates, not provider prices):

```json
{
  "model": "fixture-model",
  "effort": "high",
  "rate_card": {
    "format": "howhard-rate-card-v1",
    "source": "fixture://approved-contract",
    "version": "fixture-v1",
    "effective_at": "2026-11-07T00:00:00+00:00",
    "provider": "fixture-provider",
    "model": "fixture-model",
    "service_tier": null,
    "basis": "estimate",
    "rates": {
      "input": 2,
      "output": 8,
      "cache_read": 0.2,
      "cache_write": null,
      "cache_write_5m": 2.5,
      "cache_write_1h": 4
    }
  }
}
```

To validate an approved card as part of the real candidate inventory, copy it from its
known source into a task-owned file and construct the candidate input explicitly:

```python
import json
from pathlib import Path

card = json.loads(Path("/tmp/EXECUTION_ID-approved-rate-card.json").read_text())
candidates = [{"model": card["model"], "effort": "ACTUAL_SUPPORTED_EFFORT", "rate_card": card}]
Path("/tmp/EXECUTION_ID-candidates.json").write_text(json.dumps(candidates))
```

```sh
howhard context TASK --text-file /tmp/EXECUTION_ID-task.txt --candidates /tmp/EXECUTION_ID-candidates.json --json
```

`context` validates the versioned card before producing forecast panels. Validation
checks structure and identity; the agent still verifies that the source applies to the
host charge. Missing or failed source access leaves the price unknown; retry source
access once, then report the unavailable dependency and park price collection.

To price already captured measured usage, put the same approved card in the optional
manifest `rate_card` field. First validate its provider/model/tier against the actual
source identity. Preserve the existing execution, source, session, baseline and byte span:

```python
import json
from pathlib import Path

manifest_path = Path("/tmp/EXECUTION_ID-usage-manifest.json")
manifest = json.loads(manifest_path.read_text())
card = json.loads(Path("/tmp/EXECUTION_ID-approved-rate-card.json").read_text())
identity = ("provider", "model", "service_tier")
if any(card[key] != manifest[key] for key in identity):
    raise ValueError("Approved rate card does not match this source's provider/model/tier")
manifest["rate_card"] = card
manifest_path.write_text(json.dumps(manifest))
```

```sh
howhard usage collect --manifest /tmp/EXECUTION_ID-usage-manifest.json > /tmp/EXECUTION_ID-priced-usage.json
howhard usage attach EXECUTION_ID --manifest /tmp/EXECUTION_ID-usage-manifest.json --id EXECUTION_ID-usage-2 --json
howhard evidence EXECUTION_ID --json
```

When prior no-card/client usage was already attached as `EXECUTION_ID-usage-1`, this
changed envelope needs a **new** correction ID (`EXECUTION_ID-usage-2` above). Retry
that new correction unchanged under its own stable ID. Re-collect the saved artifact;
never replay the execution. Then refresh the current learning event. The collector
validates the card and computes `estimated_cost` from the known category measurements;
unknown categories cannot become a calculated complete price.

One collection manifest card applies to every segment. For an SDK multi-model result,
omit the card unless every segment has the same applicable identity; a mismatch is
refused. Retry collection once without the card to retain measured tokens and available
client estimates, preserving the durable outcome; never guess a matching price.

A candidate accepts either `rate_card` or legacy blended `usd_per_mtok`, never both.
`basis: "charge"` describes the card's source basis; multiplying forecast or measured
tokens by it is still an estimate, not a billing record. A required missing category rate
makes the corresponding total unknown. Money is omitted uniformly when candidates have
unknown prices or mixed bases; do not fill missing prices with zero.

For rate-card candidates, every independent and memory forecast must additionally supply
all six nonoverlapping finite nonnegative `token_usage` categories. For example,
`{"input": 100, "output": 20, "cache_read": 50, "cache_write": 0,
"cache_write_5m": 0, "cache_write_1h": 0}` has scalar `tokens: 170`. Forecast categories
are prospective estimates; unlike measured unknown coverage, forecast values are concrete
numbers. Preserve the returned opaque cell keys and response schema. Do not copy measured
usage into a past forecast. The backend validates candidate cards and forecast consistency
before selection.
