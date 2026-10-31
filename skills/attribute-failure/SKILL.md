---
name: attribute-failure
description: Investigate a bug discovered after an agent task was accepted, trace it to the responsible execution using reproductions and Git history, and correct supported howhard outcomes. Use during bug fixing or a requested audit of completed work.
---

# Attribute an escaped defect

Connect a demonstrated defect to the execution that introduced it or accepted
responsibility for implementing the missing behavior. Use this skill alongside the
installed `howhard` skill for outcome corrections and learning. Blame produces
candidates; a last-touch commit alone does not justify changing learning evidence.

## Establish the failure and its origin

Read the bug report, original task and acceptance evidence. Record the expected and
actual behavior, the revision/environment affected, and when the defect was discovered.
Use the bug-fix task's reproducing test or other concrete evidence. A defect caught
before the original task's acceptance is `loud`; an escaped defect found later is
`silent`. A later edit, refactor, new requirement or unsupported environment does not
by itself establish a failure of the earlier task. An LLM suspicion needs verification.

Inspect the fix and the pre-fix history. `git blame -w -M -C` on the affected lines
at the pre-fix revision can suggest introducing commits; inspect their diffs and
requirements rather than attributing the defect to the latest editor. Follow renames,
deleted code and integration changes when relevant. Missing code has no line to blame:
look for the task that claimed to deliver the omitted requirement.

When useful, run the same reproducer before and after a candidate change, or use
`git bisect run` between verified good and bad revisions. Use isolated task-owned
worktrees and processes, preserving the active checkout and unrelated agent state.
Control environmental differences; unbuildable revisions are untestable, not evidence
of this defect. Bisection identifies a behavioral transition, which still needs to be
checked against the original task's scope. Do not force bisection when there is no
valid good revision or when direct evidence already establishes the omission.

## Resolve the actual execution

Trace the introducing commit or responsible task through its PR/ticket and retained
execution provenance. Account for squash, rebases and merge-conflict edits: the merge
author or nearest commit is not necessarily the worker that produced the defect.
Use `howhard paths --json` and `howhard evidence EXECUTION_ID --json` in the owning
project to inspect the existing effective outcome. Inspect raw evidence when needed.

Confirm both that the defect belongs to the original task's accepted scope and that
the code/task maps to an existing recorded execution. Retain the reproduction or other
causal evidence, relevant commits, original PR/task, acceptance/discovery evidence and
execution ID in the existing bug issue or task artifacts. If the cause, responsibility
or execution mapping remains ambiguous, report it as unresolved and leave learning
evidence unchanged. Do not invent a historical model, effort, execution or forecast,
or penalize every task that touched the file. Record the fixing execution separately.

## Correct supported evidence and learn

For a confirmed escaped defect with a verified original execution, follow the installed
`howhard` skill's [correction and learning workflow](../howhard/references/workflow.md#corrections-and-learning).
Use the original task type, model and effort from its evidence, a stable correction ID
for this incident, and an evidence-bearing note:

```sh
howhard observe TASK MODEL EFFORT silent --amends EXECUTION_ID --id CORRECTION_ID --note 'Verified escaped defect; original task/PR, introducing change, reproducer and discovery evidence' --json
howhard memory learn-context CORRECTION_ID --json
```

Replace placeholders with verified values. Do not attach a decision receipt to the
amendment or pass today's task/context or the fixing worker's model. Preserve original
usage unless separately correcting a measured error. Leave the failure cause
`unrecorded` unless there is evidence for capability or depth; a bug does not prove
that a stronger model or more reasoning would have prevented it.

Complete the explicit writer-response learning step using the returned revision/event
key, even when no memory edits are warranted. Preserve the original record: the
amendment replaces its effective outcome and scores that original execution once.
Reconcile retries with the existing evidence and stable correction ID; do not append
another original run or repeat an already recorded incident under a new ID. A writer
failure does not undo the durable observation; use howhard's bounded learning recovery.

## Make future attribution possible

For completed code tasks, retain the stable execution ID, base commit, resulting
commit(s), PR when applicable, and the task's acceptance evidence in existing task
artifacts and howhard observation notes. Capture actual available values, including
the published merge/squash mapping when known. Unknown provenance stays explicit;
do not add a second ledger or fabricate historical links. `clean` means no observed
defect yet, not a guarantee that no escaped defect exists.

Report confirmed, rejected or unresolved attribution with the supporting evidence,
the affected execution if known, and whether correction and learning completed.
