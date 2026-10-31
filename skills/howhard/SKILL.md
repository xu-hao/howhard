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
   no model assumptions or statistical settings are required. Omit worker prices unless actually known:
   if any are unknown, the backend compares all cells without worker money and labels
   that limit. Do not supply zero or an API list price for unknown host charges.
3. Run commands from the current project. `howhard paths --json` exposes the
   capture/state roots. Fresh Git worktrees share repository state automatically while
   each task captures its current checkout. Explicit store selection remains available.
   Keep protocol inputs/responses in temporary artifacts outside project source.
   Do not create project wrappers, state contracts or a second memory store.
   Honor the user/application's fixed retrieval configuration. Do not edit it or try
   to upgrade its reranker when results are weak; retrieval tools accept no tier,
   model or configuration override.
4. For calling-agent reasoning, evaluate the returned `baseline_request` in a
   separate fresh model context without memory/history or learned tables. Then reason
   over the original task, project and retrieved experience for memory overrides.
   The backend retrieves/reranks up to eight useful precedents, including read-only
   `execution:RUN_ID` experts with actual model, effort, outcomes and measured usage.
   Broader/narrower tasks and relevant failures can inform estimates; do not infer unrun
   outcomes or edit execution experts. Copy
   emitted opaque cell keys into the forecast schema; generate prospective estimates
   honestly. If genuine independence is unavailable, disclose it and report routing unavailable until an independent LLM forecast is available. Prepared requests already bind candidates, prices and worktree;
   follow-up retrieve/decide commands do not repeat or replace that inventory.
5. Inspect the chosen native ID/control, cost policy and cap/review status. Delegate
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
   Matching retries return the original row; explicit conflicts are refused. Use
   `evidence ID --raw` to reconcile ambiguous writes. If a different known cell ran,
   record raw actual feedback with `--native`, the original frozen task/context and a
   provenance note, without attaching the unexecuted decision. If neither accepted
   controls nor exact reported identity establishes a match and the actual cell is
   unknown, leave it unscored and report that; do not fabricate a registration.
7. Append changed feedback to clean/loud/silent with `amends`, preserving original
   context and measurements. Use `memory learn-context` and its full `outcome_history`
   to consider original and correction notes. Apply the writer response with the exact
   revision/event key. Retry an unchanged failed writer once; if stale, refresh and
   regenerate once, then report repeated failure. Do not duplicate a run to retry learning.
   Memories can be read/searched, added, edited, removed/restored, merged or split.
   New lessons earn no retrospective predictive credit.

Keep feedback local to the backend's project store. Installing this skill does not
expand permissions, create provider access, enforce model controls or authorize publication.
