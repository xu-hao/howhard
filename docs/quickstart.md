# Start without configuration

Install from a checkout with Python 3.11 or later:

```sh
python -m pip install .
howhard pick specified-implementation --text "Make transaction rollback safe"
```

The result names a concrete model and effort, prints its seed, and explains the expected
cost and uncertainty of the alternatives. Run the delegated task with that choice, then
record what happened:

```sh
howhard observe DECISION_ID clean
howhard pick specified-implementation --text "Repair a failing fixture"
howhard summary specified-implementation
howhard flush
```

Replace `DECISION_ID` with the receipt ID printed by `pick`. It preserves the task text,
chosen model/effort and project snapshot while the task changes the repository. `clean` means no
defect was found. `loud` means a gate or review caught a defect before release or live use.
`silent` means a defect escaped. Optional `--tokens` and `--seconds` measurements help it
learn running cost and waiting time too. Without measurements it still learns outcomes.

Each directory starts from the packaged defaults and learns locally. Unknown task names
start with the standard risk policy and learn their own effects as outcomes arrive.
The shipped taxonomy includes `contract-change`, `sensitive-change`,
`specified-implementation`, `integration-recovery`, `guarded-landing`,
`docs-recording-rulings`, `docs-settled-rules`, `probe-or-measurement`, `read-only-survey`,
`live-operation-prep`, `live-checklist-run`, `release-qualification`, `preserving-rollout`,
`environment-setup`, `bookkeeping` and `worktree-cleanup`. Their definitions and default
classes are in the packaged `data/default_prior.toml`.

No initialization, credentials, network access or model training is needed. howhard
recommends a choice; the caller executes the task. It stores the pending journal, outcome
log, posterior cache, optional imported prior and a file lock in `.howhard/` in the working
directory. `observe` appends only to the journal. `flush` appends pending rows to the log
and empties the journal. Both refit using all evidence.

When no candidate meets the silent-failure cap, the default policy marks the recommendation
as needing independent review before release. JSON consumers must check `cap_met` and
`needs_review`; a successful command can still return a recommendation requiring review.

Reproduce a choice with `--seed N`. For automation use `--json`:

```sh
howhard pick bookkeeping --seed 42 --json
howhard export-prior --out shared-prior.json
# From another project directory:
howhard import-prior /path/to/shared-prior.json
```

Transfer contains shared statistical terms and generic names, with uncertainty inflated
for a new project. It excludes run records, notes, tickets, concrete model ids and project
effects. Review names of custom task types before sharing: they are part of the vocabulary.
Without configuration, imported custom types use the standard class for local policy.

See [task text and project context](context.md) for `--text-file`, receipt files,
optional project parameters and the encoder's lexical limits. The project is re-inspected
for every decision, and historical training rows retain their captured snapshots.

The built-in registry is a dated planning snapshot, not a live quote. All defaults are
maintained by the package; users do not need to tune statistical parameters.
Advanced integrations can use the [optional configuration](configuration.md).
