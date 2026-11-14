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
   `pick`. Unknown identities receive separate identity labels automatically;
   no model assumptions or statistical settings are required. Read [portable accounting](references/accounting.md) for explicit rate cards and
   category forecasts when rates are known. Candidate `usd_per_mtok` and `rate_card`
   values represent raw list-price quota credit; optional `cost_per_credit` converts that
   amount to actual session cost and defaults to `1.0` for pay-as-you-go. When actual
   pricing is unknown, omit rates so the backend compares all cells without worker money
   and labels that limit. When a priced candidate also has a known remaining credit
   envelope, attach optional `credit_budget` (`available_credit_usd` and optional
   `session_5h_remaining_credit_usd`, `weekly_7d_remaining_credit_usd`) so cells whose
   `needed_credit_usd` exceeds available credit are excluded before selection.
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
4. For calling-agent reasoning, evaluate each returned `baseline_request` in a
   separate fresh model context without memory/history or learned tables. `context`
   starts at `stage=filter`: supply a complete task-specific capability order over
   emitted opaque cell keys and nonempty lower/upper fit cuts. Reason separately over
   retrieved experience for up to eight cited `memory_filters`. Submit the combined
   filter to `decide`, then use its NEW `stage=forecast` request ID to obtain calibrated
   rows only for the band. Supply `forecasts` and up to eight `memory_forecasts` under
   the same independent-baseline rule. Handle one explicit re-filter and stronger-only
   widening requests; never truncate an inventory/band or forecast excluded cells.
   The backend learns the order and threshold from actual similar executions, with
   prior-only placement labels for unseen cells. Placement is task-specific; there is
   no universal model ranking. If genuine independence is unavailable, disclose it
   and report routing unavailable until the independent LLM calls are possible.
   Prepared requests bind candidates, prices and worktree; follow-up retrieve/decide
   commands do not repeat or replace that inventory. Never invent unrun outcomes or
   edit read-only execution experts.
5. Inspect the chosen native ID/control, cost policy, accepted-result cap/review
   status and exploration explanation. `J` includes same-cell retries after loud
   failures; epsilon caps `P_silent/(1-P_loud)` among accepted results. An unmet cap
   disables exploration, selects by mean cost (or configured lowest-bound) and requires
   review. Delegate
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
   Record the outcome before collecting usage. Read [portable accounting](references/accounting.md)
   to collect an explicitly owned artifact and attach it with a stable amendment ID.
   Missing binding or usage remains unknown; host-control acceptance is not token evidence.
7. Append changed feedback to clean/loud/silent with `amends`, preserving original
   context and measurements. Use `memory learn-context` and its full `outcome_history`
   to consider original and correction notes. Apply the writer response with the exact
   revision/event key. Retry an unchanged failed writer once; if stale, refresh and
   regenerate once, then report repeated failure. Do not duplicate a run to retry learning.
   Memories can be read/searched, added, edited, removed/restored, merged or split.
   New lessons earn no retrospective predictive credit.

Keep feedback local to the backend's project store. Installing this skill does not
expand permissions, create provider access, enforce model controls or authorize publication.
