---
name: howhard
description: Use howhard to recommend a native model and reasoning effort for an actual task, delegate within host capabilities, and learn from observed feedback through writable semantic memory. Apply when routing work or recording outcomes; mechanical checks do not establish routing quality.
---

# howhard

Use the installed howhard CLI or Python API directly. This skill is the consumer
integration: no project bridge, model mapping, dependency lock or custom ledger is
required. The backend owns validated decisions, evidence, persistent state and memory;
the host's actual tools execute the task.

1. Read the actual task and current project instructions/context. Discover the host's
   executable model IDs and exact supported effort/control IDs from its live tools or
   settings. Read [host capabilities](references/hosts.md) when needed. Never invent
   controls, map an unknown effort to max/default, or infer provider equivalence. Omit
   cells whose identity/control cannot be established and report a capability gap if
   none remain. Respect the user's chosen model or other execution constraints.
2. Follow [the native decision workflow](references/workflow.md). Write the observed
   candidate list to a temporary JSON input and pass `--candidates` to `context` or
   `pick`. Pass `--host NAME` with the name of the agent you are (the host that asks
   for the route, makes the forecast and accepts the result), and put an `agent` label
   on each candidate naming the agent product that would run it; use the names your
   routing layer already uses. Both are short labels stored unverified, and both are
   optional: omit one you cannot establish instead of guessing. Use the same host name
   every time: the backend tightens the fit band for a host that catches a smaller
   share of its failures before acceptance, and it reads that share from the failed
   runs recorded under the name. Unknown identities receive separate identity labels automatically;
   no model assumptions or statistical settings are required. Read [portable accounting](references/accounting.md) for explicit rate cards and
   category forecasts when rates are known. Candidate `usd_per_mtok` and `rate_card`
   values represent raw list-price quota credit; optional `cost_per_credit` converts that
   amount to actual session cost and defaults to `1.0` for pay-as-you-go. When actual
   pricing is unknown, omit rates so the backend compares all cells without worker money
   and labels that limit. When a priced candidate also has a known remaining credit
   envelope, attach optional `credit_budget` (`available_credit_usd` and optional
   `session_5h_remaining_credit_usd`, `weekly_7d_remaining_credit_usd`) so cells whose
   needed credit exceeds available credit, on average (`needed_credit_usd`) or at its
   80% point (`credit_p80_usd`), are excluded before selection.
3. Run commands from the current project. `howhard paths --json` exposes the
   capture/state roots. Fresh Git worktrees share repository state automatically while
   each task captures its current checkout. Explicit store selection remains available.
   Keep protocol inputs/responses in temporary artifacts outside project source.
   Do not create project wrappers, state contracts or a second memory store.
   Submit concurrent operations normally: the backend owns FIFO write admission,
   bounded waiting and crash release. Do not exchange writer-window/acquire/release
   messages or maintain an agent-channel mutex. Inspect `howhard writes --json` for
   contention; use `--write-timeout` when needed. After a timeout resubmit normally
   with the same stable IDs. Routing uses snapshot weights; unrelated writes do not
   invalidate requests. For `stale_request`, inspect changed dependencies and use its
   `refreshed_request`; reuse an independent answer only if the baseline is identical.
   Regenerate affected reasoning. Writer patches still require current revisions.
   Honor the user/application's fixed retrieval configuration. Do not edit it or try
   to upgrade its reranker when results are weak; retrieval tools accept no tier,
   model or configuration override.
4. For calling-agent reasoning, pass `--auto-decide --seed SEED` to `context` when you
   want warm memory-free routes to finish in one step; if `context` returns a
   `decision_id`, selection is already complete. Otherwise evaluate each non-null
   `baseline_request` in a separate fresh model context without memory/history or learned
   tables. When `baseline_request` is `null` (matched prototypes already cover every
   cell), skip the isolated baseline call, omit `filter` or `forecasts`, and reason only
   over active memories (`{memory_filters, reason}` or `{memory_forecasts, reason}`).
   When `baseline_request` is non-null at `stage=filter`, supply a complete task-specific
   capability order over emitted opaque cell keys and nonempty lower/upper fit cuts plus
   up to eight cited `memory_filters`. Submit to `decide`, then use its NEW
   `stage=forecast` request ID: if Stage 2 `baseline_request` is non-null, forecast
   every cell in `baseline_request.cells` (only the forecast cells with no usable stored
   forecast made under this request's `--host`; the other cells reuse their stored
   forecasts automatically) and up to eight `memory_forecasts`. Pass the same `--host`
   on every request you make: forecasts are stored and reused per host, and recorded
   runs alone never stand in for a forecast; forecasted `seconds` is expected **active**
   execution duration excluding external scheduler/queue wait. When genuinely unsure of a
   cell's loud or silent probability, add `loud_ci`/`silent_ci` as a `[lo, hi]` 95%
   credible interval containing the value; wide intervals let observed outcomes dominate
   sooner, narrow ones cannot narrow the forecast's start, and rows without intervals
   remain valid. State the same kind of 95% range for active time as `seconds_ci`
   `[lo, hi]` and, on a row with `token_usage`, for any token kind as
   `token_usage_ci` `{kind: [lo, hi]}`; each range must contain its value, a kind
   forecast at zero is exactly zero unless you state a non-zero upper end, and there is
   no range on total tokens or cost. Give the range you actually believe rather than
   omitting it or narrowing it to look confident. A forecast never fades out: each of a cell's two outcome
   beliefs keeps it at the weight of about four runs, beside the runs counted for the cell (its own in full, a related cell's at a discount). Handle one explicit re-filter and stronger-only
   widening requests; never truncate an inventory/band or forecast excluded cells.
   The forecast request can hold up to two cells outside the fit band, reserved for a
   model family with no run on this kind of task or for an agent the band leaves out:
   forecast every requested cell honestly, including one classified `underpowered`.
   The backend learns the order and threshold from actual similar executions, with
   prior-only placement labels for unseen cells. Placement is task-specific; there is
   no universal model ranking. If genuine independence is unavailable when a non-null
   `baseline_request` is returned, disclose it and report routing unavailable until the
   independent LLM calls are possible.
   Prepared requests bind candidates, prices and worktree; follow-up retrieve/decide
   commands do not repeat or replace that inventory. Never invent unrun outcomes or
   edit read-only execution experts.
5. Inspect the chosen native ID/control, cost policy, tier/review status and
   exploration explanation. `J` includes same-cell retries after loud failures;
   epsilon limits `P_silent/(1-P_loud)` among accepted results. Each cell's `pi` is its
   chance of meeting that limit: `trusted` at 0.80 or more, `on-trial` from 0.20, else
   `excluded`. Trusted and on-trial cells compete together; if every cell is excluded,
   one of those nearest the best chance is still drawn. `needs_review` is true when the
   chosen cell is not trusted (`cap_met` is false): it advises independent review of
   the result, and nothing enforces or records it. Delegate
   through the actual host tool within the authorized scope, using only supported
   controls. On a host whose native spawn API applies explicit model and effort
   overrides, accepted controls plus completed task execution attribute feedback to
   the requested cell unless the host reports launch failure, ignored controls,
   substitution or contradictory metadata. If no resolved identity is returned, record
   acceptance and completion in the observation note as host-API provenance, not backend
   or provider attestation. A recommendation or unexecuted request is not execution
   evidence. Honor review requirements and report unsupported controls or mismatches.
6. Record real feedback with a stable per-execution `--id`; never observe an unrun
   action. Use the matched receipt for the accepted-override case above, or when
   reported actual model and control exactly match the chosen cell, so feedback scores
   its prospective forecast.
   For code tasks, retain the base/result commits, PR when applicable and acceptance
   evidence alongside that execution ID in existing task artifacts and observation
   notes. Use [attribute-failure](../attribute-failure/SKILL.md) during later bug fixing
   to investigate escaped defects; uncertain attribution must not change evidence.
   Matching retries return the original row; explicit conflicts are refused. Use
   `evidence ID --raw` to reconcile ambiguous writes. If a different known cell ran,
   record raw actual feedback with `--native`, the original frozen task/context and a
   provenance note, without attaching the unexecuted decision. If neither accepted
   controls nor exact reported identity establishes a match and the actual cell is
   unknown, leave it unscored and report that; do not fabricate a registration.
   Record the outcome before collecting usage, and **always** record measured wall-clock
   `--seconds` (plus `--active-seconds` and `--queue-wait-seconds` when external queue
   wait is known; if both breakdown fields are passed, `--seconds` may be omitted and is
   derived as their sum) even when token telemetry is unavailable. Read [portable accounting](references/accounting.md)
   to collect an explicitly owned artifact and attach it with a stable amendment ID.
   Missing token binding or usage remains unknown; host-control acceptance is not token evidence.
7. Record operational interruptions such as usage/rate limits, provider/host errors or
   cancellations as `interrupted` with `--cause` (`usage_limit`, `provider_error`,
   `host_error`, `cancelled` or `unrecorded`), never as `loud` model failures. Append
   changed feedback to `clean`/`loud`/`silent`/`interrupted` with `amends`, preserving original
   context and measurements. Use `memory learn-context` and its full `outcome_history`
   to consider original and correction notes. Apply the writer response with the exact
   revision/event key. Retry an unchanged failed writer once; if stale, refresh and
   regenerate once, then report repeated failure. Do not duplicate a run to retry learning.
   Memories can be read/searched, added, edited, removed/restored, merged or split.
   New lessons earn no retrospective predictive credit.

Keep feedback local to the backend's project store. Installing this skill does not
expand permissions, create provider access, enforce model controls or authorize publication.
