# Task text and changing project context

This page describes the statistical baseline. The [LLM semantic-memory path](semantic-memory.md)
reads original semantic text and mutable memories in addition to this model.

Each decision conditions on the task type, task description, and a fresh snapshot of
the current project. Project context belongs to a run: it can change between two tasks
in the same repository. Historical rows keep the snapshot captured for that run.

## Model architecture

The existing hierarchical categorical effects remain. For each failure class, the
outcome predictor gains three sets of slopes on the contextual feature vector `x`:

```text
eta_k = existing categorical predictor
      + x * shared_context_slopes[k]
      + x * family_context_slopes[k, family]
      + x * effort_context_slopes[k, effort]
```

Thus text and project attributes can change relative family and effort preferences.
The log-token and log-time regressions each gain their own shared contextual slopes.
Outcome fitting remains a joint Newton/Laplace fit; cost and time updates remain exact.
Inactive coefficients are marginalized under their normal prior and reconstructed with
their full conditional covariance, including imported correlations. This reduces work
on small logs without changing the statistical model or using sequential approximations.

The versioned feature vector contains:

| Input | Representation |
| --- | --- |
| Task text | 32 signed hash buckets for word unigrams and adjacent bigrams; normalized log counts |
| Task size | Log-scaled character and line counts |
| Project text | 16 signed hash buckets from a bounded summary of current README, manifests and source snippets |
| Project size | Log-scaled file count, source bytes, source suffix diversity and manifest count |
| Validation setup | Test-file count, CI, lockfile and typing indicators |
| Working-tree changes | Current staged, unstaged and untracked changed-file count |
| Measured coverage | Optional fraction and a separate indicator for whether coverage is known |
| Context availability | Indicator distinguishing an observed project snapshot from missing historical context |

Hashing uses stable BLAKE2 digests, Unicode normalization and case folding, rather than
Python's process-randomized hash. The encoder is local and needs no model download,
network service or credentials. The approach follows
[feature hashing](https://arxiv.org/abs/0902.2206), with a deliberately small fixed vector
to keep joint Bayesian fitting practical.

This is a lexical encoder. It learns associations from recorded outcomes; it has no
pretrained semantic understanding of tasks or code. Synonyms need not have similar
features, and unrelated terms can collide. Context slopes start at zero with normal
uncertainty, so initial expectations rely mainly on the categorical prior. Recorded or
transferred contextual evidence is needed to learn which words and project states
favor a particular model or effort. Prior values live in packaged `data/context.toml`.

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
howhard pick specified-implementation --text "Make transaction rollback safe" --json
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
from howhard import Engine

engine = Engine()
choice = engine.pick("specified-implementation", "Make transaction rollback safe", seed=42)
# The repository can change while the delegated task runs.
engine.observe_decision(choice["decision_id"], "clean", tokens=12000, seconds=45)
```

`--text-file PATH` reads a task description as UTF-8. The longer observe form remains
available and accepts `--text` or `--text-file`; without a receipt it captures context
at observation time. Use a receipt when the task changed project state. A receipt can
also be passed to `Engine.observe(..., decision=...)`. Amendments always retain the
original run's text, project snapshot and usage measurements.

## Compatibility, cache and transfer

Old rows remain readable. Their absent context contributes zero feature values and is
never filled from today's project. New rows store `task_text`, `project_context` and
`context_encoder` together, plus a `decision_id` when a receipt was used. Validation
refuses unknown fields, malformed snapshots and unsupported encoder versions.

Full-log cache hashes include the recorded context and encoder assumptions. Query
memoization keys also include the actual task text and current snapshot, so two
requests with the same type cannot reuse different contextual predictions. Cache format
v2 safely invalidates v1 files. Identical posterior, text, snapshot and seed reproduce
the same result and receipt ID across processes.

Contextual prior exports use `howhard-prior-v2`, with an exact feature vocabulary and
encoder ID. They export shared context slopes and covariance, including family and
effort slopes, and omit raw text, snapshots, revisions, fingerprints, paths, filenames,
receipts and project-specific terms. The coefficients still represent learned evidence;
this is not a formal differential-privacy guarantee. Legacy v1 imports remain supported;
their missing context slopes use the packaged defaults. Imports reject incompatible
encoder IDs or feature vocabularies. Configurations with only legacy evidence can still
export the legacy format until contextual terms are learned or imported.

## Worktree context and shared state

Project capture uses the current checkout even when a durable store is shared. Fresh
Git worktrees automatically use the repository common directory's howhard store; valid
legacy and explicit state remain in place. `paths --json` exposes capture/state roots
and `--state-dir` selects an intentional store without changing the source capture root.
Prepared requests are worktree-bound, preventing sibling checkouts from satisfying
retrieval against another checkout's saved source. Frozen receipts remain valid for
feedback from another worktree in the same project. Stable backend-owned identity markers
move with the project; new receipts reject other repositories even with the same project
label. The resolved store subtree, including aliases, is excluded from source capture and
retrieval. State cannot equal or contain the capture root. See the [state contract](reference.md).
