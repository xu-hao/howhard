# howhard

*How hard should the agent think?*

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

You need Python 3.11+, Git, Node.js 22.20.0+ with npm/npx, and a compatible coding
agent (Codex, Claude Code or Antigravity). No statistical configuration or tuning is
required. From the project where you want to use howhard, install the Python backend
from source and the skill with the standard [skills CLI](https://github.com/vercel-labs/skills):

```sh
python3 -m venv .venv-howhard
. .venv-howhard/bin/activate
python -m pip install "git+https://github.com/xu-hao/howhard.git"
npx --yes skills@1.7.4 add xu-hao/howhard --skill howhard --agent codex claude-code antigravity antigravity-cli --yes
howhard --version
npx --yes skills@1.7.4 list --agent codex claude-code antigravity antigravity-cli
```

Expect `howhard 0.5.0` (or later) and an installed `howhard` skill. This is a source
installation, not a PyPI release claim. The command installs into the current project;
keep only the agent IDs you use, or add `--global` to the `skills add` command for
user-wide installation. Antigravity App/IDE and CLI have separate agent IDs. The standard
installer manages its central skill, agent links and lock metadata; do not copy files
by hand. Reopen your coding agent if the new skill is not visible.

In Codex invoke `$howhard` (or select it in `/skills`); in Claude Code use `/howhard`;
in Antigravity use `/howhard` or ask for the skill by name. Give it your real task:

> Use howhard for this task: repair transaction rollback and verify its failure
> boundary. Read the current project context, obtain an independent baseline, consult
> memory, and recommend a model and effort supported by this host. Execute the authorized
> work with that cell, then record the actual result and measured usage under a stable
> execution ID and update memory. Report any unsupported controls or identity mismatch.

The calling-agent workflow uses your host's existing LLM; it needs no extra API key.
The skill explains the `context` / `retrieve` / `decide` and memory-writer protocol.
A genuinely independent baseline requires a fresh model context without memory/history;
if unavailable, the agent reports that limit and uses the labeled statistical path.
Installing a skill does not add model access or enforce execution controls. The shipped
registry contains Anthropic IDs: if those do not match actual host-supported controls,
report the capability gap instead of silently relabeling a different provider's model.
A project can declare an optional host adapter in its own instructions.

To inspect a first statistical planning recommendation without any model call:

```sh
howhard pick task --text "Repair transaction rollback and verify its failure boundary" --llm off --seed 42 --json
```

The JSON includes `chosen.model`, `chosen.effort`, `decision_id`, `cap_met` and
`needs_review`. It preserves the task and current project snapshot in a receipt under
`.howhard/`. This command recommends; it does not execute the task. Honor any required
review, verify the host can run the chosen cell, then execute and assess the actual work.

Only after execution, replace `DECISION_ID` with the returned receipt ID, `EXECUTION_ID`
with a stable ID for that actual run, and `OUTCOME` with `clean`, `loud` or `silent`:

```sh
howhard observe DECISION_ID OUTCOME --id EXECUTION_ID --llm off --json
howhard memory learn-context EXECUTION_ID --llm off --json
howhard memory list --llm off --json
howhard report --llm off --json
```

`clean` means no observed defect; `loud` means a gate/review caught a failure; `silent`
means a failure escaped and was discovered later. Supply `--tokens`, `--seconds` and
`--cost-usd` only when measured; unknown usage stays null. Never record an unrun action
or attach a different executed cell to the selected receipt. The skill uses the returned
learning context to revise mutable memory. After the host prepares `writer-response.json`
with `operations` and `reason` grounded in that result, apply it using `N` and `EVENT_KEY`
from `memory learn-context`:

```sh
howhard memory learn EXECUTION_ID --file writer-response.json --expected-revision N --event-key EVENT_KEY --llm off --json
```

An empty operations list is valid. New lessons receive no retrospective predictive
credit. Retry learning for the same effective event, and append changed feedback with
`--amends` instead of creating another original run.

The default provider adapter can also reuse an authenticated Claude CLI; explicit
`--llm off` keeps commands statistical and disables provider-backed writing. For standard
skill maintenance in this project:

```sh
npx --yes skills@1.7.4 check
npx --yes skills@1.7.4 update howhard --project --yes
npx --yes skills@1.7.4 remove howhard --yes
```

The [skill guide](docs/skills.md) covers pinned/local sources, global scope and migration
from the retired 0.4 copier. Python backend upgrades and skill updates are separate.

See the [quickstart](docs/quickstart.md), [documentation index](docs/index.md),
[model and storage contracts](docs/reference.md), and [changelog](CHANGELOG.md).
The [context architecture](docs/context.md) explains text features, changing project
snapshots and receipts in the statistical baseline. Evaluate routing quality with the
[real-project protocol](docs/evaluation.md); unit fixtures establish implementation
behavior rather than product-quality evidence.
Python 3.11+ and a POSIX platform are required. Run tests with
`python -m pip install -e '.[test]'` followed by `python -m pytest`;
contributors use [AGCoord and isolated worktrees](docs/contributing.md).
