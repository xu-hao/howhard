# howhard

*How hard should the agent think?*

`howhard` chooses the model and the reasoning effort for a task delegated to an AI
agent. It uses task descriptions and snapshots of the changing project alongside task
types. From recorded runs it learns how often each (model, effort) cell
finishes cleanly, fails loudly (the failure is caught before release by a test gate, a
refusal or a review) or fails silently (the failure escapes and is found later, or never),
and what a run costs in tokens and wall time. It then picks the cell with the lowest expected
total cost: the run's own cost, plus the expected cost of a loud and of a silent failure, plus
a price on waiting, while keeping the chance of a silent failure under a cap set for the
task's risk class.

The model is Bayesian and starts from an explicit, documented prior, so it gives useful
answers before any data exist and sharpens as outcomes are recorded. Model and effort are
two separate categorical axes with an interaction, not one ordered ladder. Choices are made
by Thompson sampling from the posterior, so cheaper cells keep being tried. A project can
export what it has learned about generic task types as a prior for another project, without
exporting any of its records.

The package is plain Python with NumPy and ships a command-line tool, `howhard`.
It works without configuration or statistical parameter tuning:

```sh
python -m pip install .
howhard pick specified-implementation --text "Make transaction rollback safe"
# Execute the task with the chosen cell, then use the printed decision ID:
howhard observe DECISION_ID clean
howhard summary specified-implementation
howhard flush
```

Decision receipts preserve task text and project context from before the task ran.
The project is inspected again for each new decision. Defaults include a model registry,
prices, task taxonomy, risk policies and priors. Learning is local to `.howhard/` in the
working directory. New task names start with the standard policy. If the silent-failure
cap cannot be met, the default recommendation requires independent review before release.

Use `--json` for automation and `--seed N` to reproduce a pick. There are six commands:
`pick`, `summary`, `observe`, `flush`, `export-prior` and `import-prior`.

```python
from howhard import Engine

engine = Engine()
choice = engine.pick("specified-implementation", "Make transaction rollback safe", seed=42)
```

See the [quickstart](docs/quickstart.md), [documentation index](docs/index.md),
[model and storage contracts](docs/reference.md), and [changelog](CHANGELOG.md).
The [context architecture](docs/context.md) explains text features, changing project
snapshots, receipts and the local lexical encoder's limits.
Python 3.11+ and a POSIX platform are required. Run tests with
`python -m pip install -e '.[test]'` followed by `python -m pytest`;
contributors use [AGCoord and isolated worktrees](docs/contributing.md).
