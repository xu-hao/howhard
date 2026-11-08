# Quickstart

You need Python 3.11+, a POSIX shell, Git and a current Codex, Claude Code or
Antigravity client with native plugin support.

## Install the backend

Run these commands from the project where you want to use howhard:

```sh
python3 -m venv .venv-howhard
. .venv-howhard/bin/activate
python -m pip install "howhard==2.0.0"
howhard --version
```

Expect `howhard 2.0.0`. The Python package is available on
[PyPI](https://pypi.org/project/howhard/2.0.0/).

Keep this environment active and start your coding agent from the same shell so it
can find `howhard`. If an existing GUI session cannot find it, configure the absolute
path to your project's `.venv-howhard/bin/howhard`; reopening a GUI does not activate
the shell's environment.

## Install the native plugin

The Python package supplies the backend. The native plugin supplies the `howhard`
routing and learning skill and the `attribute-failure` investigation skill. Install
the plugin through your agent's own plugin manager.

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

Antigravity:

```sh
agy plugin install https://github.com/xu-hao/howhard
agy plugin list
```

These commands use the user's plugin environment. Claude Code also supports explicit
`--scope project` or `--scope local` for both marketplace addition and installation.
Reopen the agent if the skill is not visible. Backend and plugin updates are separate;
see [native installation and lifecycle](skills.md) for update and scope commands.

## Use howhard on a real task

Give the coordinating agent an authorized task and ask it to use howhard before
starting a worker. For example:

> Use howhard for this task: repair transaction rollback and verify its failure
> boundary. Read the project context, obtain an independent baseline, consult memory,
> and choose a model and reasoning effort supported by this host. Execute the work,
> record the actual outcome and measured usage under a stable execution ID, then
> update memory. Report unavailable routing or unsupported controls.

Explicit skill invocation is `$howhard` or `/skills` in Codex, `/howhard:howhard` in
Claude Code, and `/howhard` in Antigravity. Codex can also activate the skill when the
task matches its description. Installing the plugin does not make every worker start
run it automatically. For persistent use, copy the
{ref}`coordinating-agent instruction <require-howhard-before-delegation>` into
your host's project instructions.

The agent discovers the host's supported model and effort controls, obtains a baseline
forecast in a fresh context without memory or history, and considers retrieved
experience before selecting a worker. It creates temporary protocol inputs and calls
the backend directly. You do not need to configure model-family mappings or tune
parameters. If an independent forecast or required execution control is unavailable,
the agent reports the limitation instead of inventing a selection.

The decision includes `chosen.model`, `chosen.effort`, `decision_id`, `cap_met`,
`needs_review` and `cost_policy`. Honor required review. The host executes the task;
a recommendation alone is not execution. Unknown worker prices stay unknown. If any
candidate is unpriced, the comparison excludes worker money for every candidate and
labels that limit.

## Record the outcome and learn

The skill records feedback only after actual execution. When inspecting the workflow,
replace `DECISION_ID` with the returned receipt, `EXECUTION_ID` with a stable ID for
that invocation, and `OUTCOME` with `clean`, `loud` or `silent`:

```sh
howhard observe DECISION_ID OUTCOME --id EXECUTION_ID --json
howhard evidence EXECUTION_ID --raw --json
howhard memory learn-context EXECUTION_ID --json
```

`clean` means no observed defect, `loud` means a gate or review caught a failure, and
`silent` means a failure escaped and was discovered later. Record real work and actual
outcomes; setup examples and mechanical checks do not establish routing quality.

Accepted explicit native model/effort overrides plus completed execution establish a
match to the selected receipt unless the host reports a launch failure, ignored control,
substitution or contradictory metadata. Record that as host-API provenance when no
separate resolved identity is returned. A known different executed cell uses raw native
feedback; an unknown identity stays unscored. See {ref}`feedback attribution <skill-capability-boundaries>`
for the full protocol.

The agent uses the learning context to prepare a writer response containing `operations`
and `reason`. Apply it with the exact revision and event key returned by
`memory learn-context`:

```sh
howhard memory learn EXECUTION_ID --file WRITER_RESPONSE.json --expected-revision N --event-key EVENT_KEY --json
```

An empty operations list is valid. Retry learning for the same event; amend changed
feedback with `--amends` instead of recording the execution twice. See the
[semantic-memory guide](semantic-memory.md) for writer schemas and memory operations.

Supply usage only when measured. `--tokens`, `--seconds` and `--cost-usd` accept known
measurements; unsupported or missing measurements remain unknown. For supported host
artifacts, `usage collect` and `usage attach` add bounded execution accounting to an
existing outcome. Exact invocation ownership is required, including child-session
binding when applicable. Follow the {ref}`portable accounting contract <portable-execution-accounting>`
for capture, token categories, rate cards and estimated versus billed cost.

## Inspect project state

```sh
howhard paths --json
howhard memory list --json
howhard report --json
```

Git worktrees share repository state while each task captures its current checkout.
Plain folders use local `.howhard`. `paths` shows the resolved locations; the
[context contract](context.md) explains task snapshots and explicit store selection.
`report` summarizes observed actions, not unrun alternatives or proven routing
superiority. Use the [real-project evaluation guide](evaluation.md) to assess learning
from genuine work.

Calling-agent commands use your host's LLM reasoning and need no provider configuration.
Standalone `pick` can use an authenticated Claude CLI or `HOWHARD_LLM_COMMAND`; see
[provider options](semantic-memory.md).
