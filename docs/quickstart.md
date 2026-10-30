# Quickstart

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

The [skill guide](skills.md) covers pinned/local sources, global scope and migration
from the retired 0.4 copier. Python backend upgrades and skill updates are separate.

The [semantic-memory guide](semantic-memory.md) describes complete reader/writer schemas,
mutable memory operations and provider options. The [context contract](context.md)
explains frozen task/project snapshots. Learning stays local to `.howhard/` by default;
new task names use the standard risk policy, and no parameter tuning is required.
Use `howhard flush --llm off` to move pending journal rows into the outcome log.
See [real-project evaluation](evaluation.md) for evaluating useful learning; mechanical
fixtures and observed-action reports cannot establish routing superiority.
