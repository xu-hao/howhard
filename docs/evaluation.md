# Evaluation on real projects

Evaluate routing and learning with genuine work from maintained projects. No generated
tasks, artificial paraphrases or invented outcomes count as quality evidence. Fixtures
in the unit suite verify software mechanics only.

## First field study

Start with the next 12 consecutive eligible tasks in howhard and another real maintained
repository when available. Use the first four as warm-up and the next eight as a
chronological feasibility evaluation. Keep failures, fallbacks and inconvenient tasks
in the record. Twelve tasks test whether measurement and learning work; they cannot
establish general superiority or a small silent-failure rate.

Before any worker outcome, record forecasts from these arms:

| Arm | Information and state |
| --- | --- |
| Existing statistical router | Task/project features and outcome posterior |
| Stateless LLM | Current task and project only |
| Plain retrieval | Immutable earlier experience, without mutable semantic memory |
| Writable memory | Reader/writer with fixed weights |
| Weighted memory | Reader/writer with predictive evidence weighting |

The stateless LLM must run in a fresh context without memories, earlier outcomes or
learned probability tables. Freeze worker tools, candidate registry, prices, risk
constants, acceptance gate and review budget before the task.

Initially supply each arm the same observed incumbent history. This isolates forecast
learning under common experience; it does not establish the value of independent
deployed learning policies. Run the incumbent choice and one genuinely different
challenger choice in isolated issue worktrees when the task is safe and budget allows.
If all arms agree, one worker run is enough. Retain artifacts without publishing
experimental changes. Duplicate experimental worker cost belongs to the study budget;
also estimate the cost each deployed policy would require.

## Ground truth and time

Freeze the project's ordinary gate before seeing artifacts, then use a reviewer blinded
to model/router identity for an independent audit. Record separately:

- A defect caught by the ordinary gate: loud failure.
- A gate-passing defect found by the independent audit: an escape candidate, an offline proxy.
- A defect discovered after actual acceptance/release: confirmed silent failure.
- Work passing available checks: provisionally clean, with a stated follow-up horizon.

Follow accepted work after 7 and 30 days, recording rework and outcome amendments.
Missing follow-up is not evidence of cleanliness. Never relabel an offline audit finding
as a confirmed live escape. Do not use unsafe live operations as experimental tasks.

Archive the real task, exact decision-time project context/source, memory versions,
retrieval results, forecasts and toolchain identity. Historical replay must not inspect
later source, fixes, discussions, tests or Git history. Release feedback when it became
available, including delayed amendments. Related tasks/retries/follow-ups stay grouped
when creating development and holdout sets. Reset memory and inference caches when
starting an independent replay or evaluation arm.

## What to measure

Primary policy loss includes worker USD, routing USD, configured loud/silent failure
penalties, and priced elapsed time. Include amortized writer cost. Report measured
review/rework separately so it is not counted twice through configured penalties.
Unknown cost remains unknown. Report latency distributions and review/fallback rates.

Score pre-outcome forecasts using multiclass log loss and Brier score. Compare all arms
on matched, observed actions and display silent-probability reliability with counts and
uncertainty. A seed is not an action probability. Ordinary selected-action logs do not
identify counterfactual outcomes; use actual parallel worker artifacts or a prospective
randomized trial before making policy-value claims.

Look for learning on naturally recurring work: useful memory creation, benefit on later
related tasks with different vocabulary, non-application to unrelated tasks, correction
after contradictory evidence, and adaptation after real project changes. If corrections
or drift do not occur during the study, label those capabilities unmeasured rather than
manufacturing them. Plausible explanations and successful recall are diagnostic signals,
not evidence of better routing.

Use `howhard report --json` as an observed-action summary and retain the underlying
audits. Record additional field-study metadata outside the basic outcome labels:
experiment/arm/task-family IDs, archived artifacts, frozen gate results, reviewer,
feedback timestamps, follow-up horizons, and assignment probabilities when known.
Separate cold and warm caches, reader/writer calls, errors, token billing categories,
resolved model IDs, and end-to-end latency.

## Advancing beyond feasibility

The first study succeeds when records are complete, replay is possible, no future
information leaked, and overhead is measured. It may reveal useful examples without
showing an overall improvement.

For a later confirmation study, choose a stopping plan and meaningful improvement
target before collecting the holdout results. Include several projects and enough real
tasks to estimate uncertainty; the number needed depends on observed variability and
rare-failure frequency. Weighted memory must justify its additional complexity and
cost against the strongest simpler comparator. Report regressions by project and risk
class. Continue human acceptance and ordinary gates while the policy is being evaluated.
