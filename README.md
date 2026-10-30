# howhard

*How hard should the agent think?*

[Read the documentation](https://howhard.readthedocs.io/en/latest/).

`howhard` chooses the model and the reasoning effort for a task delegated to an AI
agent. An LLM can read task descriptions, current project context and writable semantic
memory. Recorded outcomes update predictive weights, while a statistical model learns
how often each (model, effort) cell
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

## Quickstart

You need Python 3.11+, Git and a current Codex, Claude Code or Antigravity client
with native plugin support. No statistical configuration, tuning or Node/npm setup is
required by howhard. From your project directory, install the Python backend from source:

```sh
python3 -m venv .venv-howhard
. .venv-howhard/bin/activate
python -m pip install "git+https://github.com/xu-hao/howhard.git"
howhard --version
```

Expect `howhard 0.6.0` (or later). This is a source installation, not a PyPI release
claim. Keep this environment active and start your coding agent from the same shell so
it can find `howhard`. Install the skill through **your agent's native plugin manager**;
choose the matching block below.
An existing GUI session may need the absolute path to `.venv-howhard/bin/howhard`
in your project; reopening a GUI alone does not activate this terminal's environment.

Codex:

```sh
codex plugin marketplace add xu-hao/howhard
codex plugin add howhard@howhard
```

Claude Code:

```sh
claude plugin marketplace add xu-hao/howhard
claude plugin install howhard@howhard
```

Antigravity (`agy`):

```sh
agy plugin install https://github.com/xu-hao/howhard
agy plugin list
```

These native flows use the user's plugin environment. Claude Code also supports explicit
`--scope project` or `--scope local` on both marketplace addition and plugin installation.
Choose the intended scope before running those commands. Reopen the agent if necessary.
All three load the same `skills/howhard/SKILL.md`: common Agent Skills content with thin,
host-specific plugin metadata, not three copies or a howhard installer.

In Codex, the installed skill can activate automatically when a task matches its
model-selection or feedback-learning description. Explicit invocation is optional:
use `$howhard` or `/skills` in Codex, `/howhard:howhard` in Claude Code, or `/howhard`
in Antigravity to request it directly. [Codex skill activation](https://learn.chatgpt.com/docs/build-skills)
is based on the task; starting a subagent alone does not guarantee activation.
The coordinating agent uses howhard **before** spawning a worker so it can choose that
worker's model and effort. The plugin supplies instructions, with no subagent-start hook.
To request the complete workflow explicitly, give it your real task:

> Use howhard for this task: repair transaction rollback and verify its failure
> boundary. Read the current project context, obtain an independent baseline, consult
> memory, and recommend a model and effort supported by this host. Execute the authorized
> work with that cell, then record the actual result and measured usage under a stable
> execution ID and update memory. Report any unsupported controls or identity mismatch.

The skill discovers the host's executable model/control inventory, creates temporary
protocol inputs, obtains an independent baseline and uses memory-conditioned forecasts.
It calls the native howhard CLI/API directly. You do not write a project bridge, add a
dependency-lock feature, maintain a separate ledger, map models to statistical families
or tune parameters. Unknown native identities get distinct neutral prior families;
unsupported or unknown execution controls are reported honestly.

Your host's existing LLM supplies the calling-agent reasoning without another API key.
An independent baseline needs a fresh context without memory/history; if unavailable,
the agent reports that limit and uses the labeled statistical path. Unknown worker
prices stay unknown. If any candidate is unpriced, every candidate is compared without
worker money; the partial objective is labeled rather than treating unknown charges as
free or assuming API list prices apply to the host.

The JSON decision includes `chosen.model`, `chosen.effort`, `decision_id`, `cap_met`,
`needs_review` and the native `cost_policy`. A recommendation does not execute the task.
Honor required review and check the actual host execution against the chosen cell.
Fresh Git worktrees automatically share repository state while capturing the current
checkout; plain folders use local `.howhard`, and explicit store selection is available.
Inspect the resolved roots with:

```sh
howhard paths --llm off --json
```

Only after execution, replace `DECISION_ID` with the returned receipt ID, `EXECUTION_ID`
with a stable ID for that actual invocation, and `OUTCOME` with `clean`, `loud` or `silent`:

```sh
howhard observe DECISION_ID OUTCOME --id EXECUTION_ID --llm off --json
howhard evidence EXECUTION_ID --raw --llm off --json
howhard memory learn-context EXECUTION_ID --llm off --json
howhard memory list --llm off --json
howhard report --llm off --json
```

`clean` means no observed defect; `loud` means a gate/review caught a failure; `silent`
means a failure escaped and was discovered later. Supply `--tokens`, `--seconds` and
`--cost-usd` only when measured; unknown usage stays null. Never record an unrun action
or attach a different executed cell to the selected receipt. A verified different cell
is recorded as raw native feedback with the original frozen context and provenance; an
unknown actual identity/control stays unscored. Matching stable-ID retries return the
first row, while conflicting feedback is refused. The skill uses the returned
learning context, including the original and correction notes, to revise mutable memory. After the host prepares a temporary `writer-response.json`
with `operations` and `reason` grounded in that result, apply it using `N` and `EVENT_KEY`
from `memory learn-context`:

```sh
howhard memory learn EXECUTION_ID --file writer-response.json --expected-revision N --event-key EVENT_KEY --llm off --json
```

An empty operations list is valid. New lessons receive no retrospective predictive
credit. Retry learning for the same effective event, and append changed feedback with
`--amends` instead of creating another original run.

The default provider adapter can also reuse an authenticated Claude CLI; explicit
`--llm off` keeps commands statistical and disables provider-backed writing. The
[skill guide](docs/skills.md) covers native lifecycle commands, native feedback/mismatch handling and local/pinned sources.
A third-party cross-agent `npx skills` route is optional, not a prerequisite.
Python backend upgrades and skill updates are separate.

See the [quickstart](docs/quickstart.md), [documentation index](docs/index.md),
[model and storage contracts](docs/reference.md), and [changelog](CHANGELOG.md).
The [context architecture](docs/context.md) explains text features, changing project
snapshots and receipts in the statistical baseline. Evaluate routing quality with the
[real-project protocol](docs/evaluation.md); unit fixtures establish implementation
behavior rather than product-quality evidence.
Python 3.11+ and a POSIX platform are required. Run tests with
`python -m pip install -e '.[test]'` followed by `python -m pytest`;
contributors use [AGCoord and isolated worktrees](docs/contributing.md).
