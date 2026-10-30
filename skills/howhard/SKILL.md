---
name: howhard
description: Use howhard to recommend a model and reasoning effort for an actual task, delegate within host capabilities, and learn from observed outcomes through writable semantic memory. Apply when routing work or recording feedback, not as a claim of routing quality.
---

# howhard

Use the installed `howhard` CLI or Python `Engine` as the stateful recommendation and
feedback backend. This skill provides the workflow; the host's tools execute the work.
Installing or loading it does not add a delegation tool or enforce a model selection.

1. Read the task and current project context. Consult any project-declared adapter in
   its `AGENTS.md` or runbook for supported host models/efforts, commands and state
   locations. Follow the project's taxonomy if it has one; otherwise use a meaningful
   task name and description. Do not replace actual task text with a category label.
2. Establish the host's actual executable inventory and controls before recommending
   an action. Read [host capabilities](references/hosts.md) for Codex, Claude Code or
   Antigravity. Intersect the inventory with validated howhard registry cells. Never
   pass a Claude ID to another provider's control, infer cross-provider equivalence,
   or invent a tool parameter. If no supported intersection or enforceable effort
   exists, report that capability gap and the actual available controls. A planning
   recommendation is not evidence that the selected action ran.
3. Use [the decision workflow](references/workflow.md) for standalone commands or the
   calling-agent protocol. Freeze a fresh task/project context. Evaluate the returned
   `baseline_request` in a separate, fresh model context without memories, history or
   the learned table; then reason over the original task and retrieved evidence for
   memory-conditioned overrides. If an independent context cannot be obtained, report
   that limit and use the explicitly labeled statistical path. Generate prospective
   estimates honestly; do not fabricate past forecasts, observations or baseline independence.
4. Validate the result against the same host inventory and preserve its decision ID,
   chosen concrete model, effort, context, cap/review status and reasons. Honor review
   requirements. Delegate through the host's actual tool using supported controls and
   the user's authorized scope. Retain the execution ID, requested controls and host
   reported actual model/effort separately. Report any mismatch; never relabel a run
   as if the requested action executed.
5. Record actual feedback after checks/review, with a stable execution-to-run ID
   mapping and measured usage when known. Reconcile an ambiguous write by reading the
   recorded ID before retrying; never create a second original row for the same run.
   Append corrections to clean/loud/silent with `amends`; keep original context and
   measurements. Unknown usage stays null. An unrun action has no outcome. Missing
   execution identity or unsupported controls stay in an honest host audit until a
   valid native observation can be attributed.
6. Use the mutable memory commands and writer protocol in the workflow reference.
   Inspect/read/search, add, revise, remove, restore, merge or split memories as
   evidence warrants. A new lesson earns no retrospective predictive credit. Retry a
   failed writer for the same effective event once; if stale, prepare a fresh context
   and retry once, then report the unresolved error. Retain outcomes even when learning
   fails. Do not turn mechanical fixtures or descriptive reports into routing-quality
   claims.

Keep project evidence and memories in the project's declared store. Installation does
not authorize copying evidence across projects, making provider calls or publishing work.
