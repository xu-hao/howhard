# howhard

*How hard should the agent think?*

[Read the documentation](https://howhard.readthedocs.io/en/latest/).

`howhard` chooses the model and reasoning effort for a task delegated to an AI
agent. An LLM reads the actual task, current project and writable semantic memory,
forecasting clean completion, caught failures, escaped failures, tokens and time for
each candidate. Recorded outcomes update the predictive support for memories.

Selection balances worker cost, expected failure cost and waiting time. Its risk cap
uses a weighted envelope of LLM forecasts; these estimates are not a calibrated
failure guarantee. Without a usable LLM forecast, routing stops explicitly.

## Quickstart

You need Python 3.11+, a POSIX shell, Git and a current Codex, Claude Code or
Antigravity client with native plugin support.

Install the Python backend from [PyPI](https://pypi.org/project/howhard/2.0.0/) in your
project directory:

```sh
python3 -m venv .venv-howhard
. .venv-howhard/bin/activate
python -m pip install "howhard==2.0.0"
howhard --version
```

Expect `howhard 2.0.0`. Keep the environment active and start your agent from the same
shell so it can find `howhard`. An existing GUI session may need the absolute path to
`.venv-howhard/bin/howhard` in your project.

Install the native plugin for your host:

```sh
# Codex
codex plugin marketplace add xu-hao/howhard
codex plugin add howhard@howhard
```

```sh
# Claude Code
claude plugin marketplace add xu-hao/howhard
claude plugin install howhard@howhard
```

```sh
# Antigravity
agy plugin install https://github.com/xu-hao/howhard
agy plugin list
```

The plugin supplies the `howhard` routing and learning skill and the `attribute-failure`
investigation skill. Backend and plugin updates are separate; the
[native installation guide](docs/skills.md) covers scopes and lifecycle commands.

Ask your coordinating agent to use howhard on a real, authorized task before starting
a worker:

> Use howhard to choose the model and reasoning effort for this task. Obtain an
> independent baseline, consult memory, execute with supported host controls, then
> record the actual outcome and measured usage and update memory.

Invoke the skill explicitly with `$howhard` or `/skills` in Codex,
`/howhard:howhard` in Claude Code, or `/howhard` in Antigravity. To require it before
delegation, add the [coordinating-agent instruction](docs/skills.md#require-howhard-before-delegation)
to your project instructions. Plugin installation alone does not enforce worker launches.

The [complete quickstart](docs/quickstart.md) walks through decision fields, feedback,
usage and memory learning. The host must support the chosen controls and a fresh context
for the independent baseline; otherwise routing is unavailable. Unknown usage and prices
stay unknown. Record actual executions rather than invented outcomes.

Inspect local state and observed-action summaries with:

```sh
howhard paths --json
howhard memory list --json
howhard report --json
```

The [model and storage contracts](docs/reference.md), [semantic-memory guide](docs/semantic-memory.md)
and [context contract](docs/context.md) describe the CLI/API and evidence boundaries.
Evaluate routing quality with [genuine project work](docs/evaluation.md).
External routing applications can [configure an available model list](docs/configuration.md#candidate-lists-for-external-routers)
and use the Python or calling-agent interfaces to select models and share learning
across providers in the same project store.
Contributors use [isolated worktrees and the documentation-only exemption](docs/contributing.md).
