# howhard

*How hard should the agent think?*

`howhard` chooses the model and the reasoning effort for a task delegated to an AI
agent. For each task type it learns, from recorded runs, how often each (model, effort) cell
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

The package is plain Python with NumPy and ships a command-line tool, `howhard`; for example, `howhard pick <task-type>`.

## Status

Under construction. The build plan and acceptance criteria are tracked in this repository's
issues.
