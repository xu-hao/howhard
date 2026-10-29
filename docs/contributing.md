# Contributor workflow

Track every repository change with a descriptive issue, and report its number with a
sentence describing the work. Develop and validate in an issue-specific worktree from
`main`, preserving unrelated changes. The initial implementation is tracked by issue #1,
which specifies the model and effort selector; issues #2–#9 specify its packaging,
models, storage, selection, cache, transfer and behavioral acceptance tests.

```sh
git worktree add -b issue-N-description ../howhard-issue-N main
cd ../howhard-issue-N
python -m venv .venv
.venv/bin/python -m pip install -e '.[test,docs]'
agc run --label "howhard tests" --resource cpu=1 -- .venv/bin/python -m pytest
```

Use the local AGCoord installation for checks and declare resources. Do not submit nested
coordinator commands from admitted jobs. Tests use synthetic configs and log files in
their own temporary directories, exercise public behavior, and own all state they create.
Statistical tests use fixed seeds. The 1,000-row refit has a two-second performance budget.
Limit BLAS to the granted CPU count when running numerical checks.

Update the affected canonical documents, list new documents in the index, and record
user-facing changes in `CHANGELOG.md`. `AGENTS.md` is the canonical instruction source;
`CLAUDE.md` is its relative symlink. Other agents read the top-level instructions too.

The complete validation command is `scripts/check`. It runs tests, builds both distributions,
verifies wheel contents/version and builds Sphinx with warnings treated as errors. CI runs
it on Python 3.11 and 3.12. Gate and publish through a single `agc land` request using that
command when a change is ready to land. Never push directly to the target branch or bypass
the coordinator verdict. Documentation-only changes follow the narrower exemption in
the root guardrails. End changed work with the [session handoff](session_handoff_format.md).
