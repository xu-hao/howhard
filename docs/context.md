# Task text and changing project context

The LLM receives original task text and bounded project text, together with observable
project attributes. The independent LLM uses no lexical feature hashing or contextual regression.
Retrieval represents earlier task/project text, and the coarse threshold reader uses
its relevance scores; it does not replace the original task text. Historical observations preserve
the task and snapshot captured for the actual execution.

## Capturing project state

The current working directory (or the explicit config's directory) is inspected for
each `pick` and `summary`, including repeated calls on the same `Engine`. Git-backed
scans respect tracked files and untracked files allowed by Git's ignore rules. Other
directories use a bounded filesystem scan. Generated state such as `.howhard`, virtual
environments, dependency trees and build output is excluded.

The snapshot records parameters, bounded project text, the current Git revision when
available, a fingerprint and whether the file limit was reached. The fingerprint also
tracks file sizes and modification times; revision alone would miss working-tree edits.
Parameters are observable proxies, such as source suffix diversity for language count
and file naming for test presence. A bounded working-tree scan is not a semantic model
of the entire repository or an atomic filesystem transaction.

Callers may supply parameter overrides or a complete saved snapshot through the API's
`project_context` argument, or a JSON file via `--project-context PATH`. Parameter keys
are `files`, `source_bytes`, `languages`, `test_files`, `manifests`, `changed_files`,
`has_ci`, `has_lockfile`, `has_typing`, and `coverage`. Counts are nonnegative integers,
flags are booleans, and coverage is null or a fraction in `[0, 1]`. Missing overrides
are gathered automatically. No coverage value is guessed from the existence of tests.

## Record the decision-time context

```sh
howhard pick specified-implementation --candidates /tmp/host-candidates.json --text "Make transaction rollback safe" --json
# Copy decision_id from the result, then execute the task using its chosen cell.
howhard observe DECISION_ID clean --tokens 12000 --seconds 45
```

Every pick saves a content-addressed receipt under `decisions/` in the backend state
root reported by `howhard paths` (local `.howhard` for plain folders). Its ID is
printed in text and JSON output. The receipt includes the chosen cell, task description,
project snapshot, seed and risk class. A later observation resolves the receipt and
copies that original context into the append-only row. It does not rescan the newer
project. `--save-decision PATH` copies the receipt for another process; the short observe
form also accepts that file path. Explicit four-argument observe supports `--decision ID`
or `--decision PATH` and refuses conflicting task/cell/context arguments.

```python
import json
from pathlib import Path
from howhard import Engine

inventory = json.loads(Path("/tmp/host-candidates.json").read_text())
engine = Engine()
choice = engine.pick("specified-implementation", "Make transaction rollback safe", native_candidates=inventory, seed=42)
# The repository can change while the delegated task runs.
engine.observe_decision(choice["decision_id"], "clean", tokens=12000, seconds=45)
```

`--text-file PATH` reads a task description as UTF-8. The longer observe form remains
available and accepts `--text` or `--text-file`; without a receipt it captures context
at observation time. Use a receipt when the task changed project state. A receipt can
also be passed to `Engine.observe(..., decision=...)`. Amendments always retain the
original run's text, project snapshot and usage measurements.


## Two reader stages in a frozen request

`context` first returns a capability-filter baseline and memory request over the full
inventory. Its task-specific order is recomputed for each decision. Submitting the
filter to `decide` returns a new request ID for calibrated forecasts over the fit band.
Both stages keep the original task and project snapshot and independently reject stale
evidence or memory. Retrieval, one explicit re-filter, and each stronger-only widening
round also return bound request IDs. Follow the returned `stage` until a `decision_id`
is produced. Neither clean baseline includes earlier execution evidence or memories.

The receipt freezes full inventory coverage, capability positions/classes, prior-only
placement labels, learned threshold probabilities, widening, cross-reader disagreements,
and stage-2 forecasts/selection. Stage 2 records same-cell retry cost, accepted-result
silent-risk bounds, calibrated uncertainty and exploration status. One later observation
uses this original receipt to update both readers. See [capability filtering](capability.md)
and [upgrading to 3.0](upgrading-3.0.md).
