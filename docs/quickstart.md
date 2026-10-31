# Quickstart

You need Python 3.11+, Git and a current Codex, Claude Code or Antigravity client
with native plugin support. No parameter tuning or Node/npm setup is
required by howhard. From your project directory, install the Python backend from source:

```sh
python3 -m venv .venv-howhard
. .venv-howhard/bin/activate
python -m pip install "git+https://github.com/xu-hao/howhard.git@v1.0.0"
howhard --version
```

Expect `howhard 1.0.0`. Read the [upgrade guide](upgrading-1.0.md) when moving
from 0.6.x. This is a source installation, not a PyPI release
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
All three load the same skills: `howhard` for routing and learning, and
`attribute-failure` for investigating escaped defects. They share common Agent Skills
content with thin host-specific plugin metadata; see [skill installation](skills.md).

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

For persistent use before worker delegation, copy the
[coordinating-agent instruction](skills.md)
into your agent's project instructions (`AGENTS.md` in Codex). The skill guide explains
the instruction's scope and the host enforcement needed for a hard guarantee.

The skill discovers the host's executable model/control inventory, creates temporary
protocol inputs, obtains an independent baseline and uses memory-conditioned forecasts.
It calls the native howhard CLI/API directly. You do not write a project bridge, add a
dependency-lock feature, maintain a separate ledger, map models to families
or tune parameters. Unknown native identities get distinct identity labels;
unsupported or unknown execution controls are reported honestly.

Your host's existing LLM supplies the calling-agent reasoning without another API key.
An independent baseline needs a fresh context without memory/history; if unavailable,
the agent reports routing unavailable until an independent LLM forecast is available. Unknown worker
prices stay unknown. If any candidate is unpriced, every candidate is compared without
worker money; the partial objective is labeled rather than treating unknown charges as
free or assuming API list prices apply to the host.

The JSON decision includes `chosen.model`, `chosen.effort`, `decision_id`, `cap_met`,
`needs_review` and the native `cost_policy`. A recommendation does not execute the task.
Honor required review. When the native host accepts explicit model and effort overrides
and the task completes, use the matched decision receipt for the requested cell unless
the host reports a launch failure, ignored control, substitution or contradictory
metadata; note that host-API provenance when no resolved identity is returned. Otherwise,
match only on exact reported actual identity/control, use raw feedback for a known
mismatch, and leave feedback unscored only when that identity/control is unknown.
Fresh Git worktrees automatically share repository state while capturing the current
checkout; plain folders use local `.howhard`, and explicit store selection is available.
Inspect the resolved roots with:

```sh
howhard paths --json
```

Only after execution, replace `DECISION_ID` with the returned receipt ID, `EXECUTION_ID`
with a stable ID for that actual invocation, and `OUTCOME` with `clean`, `loud` or `silent`:

```sh
howhard observe DECISION_ID OUTCOME --id EXECUTION_ID --json
howhard evidence EXECUTION_ID --raw --json
howhard memory learn-context EXECUTION_ID --json
howhard memory list --json
howhard report --json
```

`clean` means no observed defect; `loud` means a gate/review caught a failure; `silent`
means a failure escaped and was discovered later. Supply `--tokens`, `--seconds` and
`--cost-usd` only when measured; unknown usage stays null. Never record an unrun action
or attach a different executed cell to the selected receipt. A verified different cell
is recorded as raw native feedback with the original frozen context and provenance; an
unknown actual identity/control stays unscored when neither accepted overrides nor exact
reported identity establishes a match. Matching stable-ID retries return the first row,
while conflicting feedback is refused. The skill uses the returned
learning context, including the original and correction notes, to revise mutable memory. After the host prepares a temporary `writer-response.json`
with `operations` and `reason` grounded in that result, apply it using `N` and `EVENT_KEY`
from `memory learn-context`:

```sh
howhard memory learn EXECUTION_ID --file writer-response.json --expected-revision N --event-key EVENT_KEY --json
```

An empty operations list is valid. New lessons receive no retrospective predictive
credit. Retry learning for the same effective event, and append changed feedback with
`--amends` instead of creating another original run.

Standalone `pick` can reuse an authenticated Claude CLI or `HOWHARD_LLM_COMMAND`.
Calling-agent commands need no provider configuration and never start nested inference.
Observations made without a provider adapter return a pending memory update; use the
explicit writer workflow above or `memory learn RUN_ID` to invoke a provider. The
[skill guide](skills.md) covers native lifecycle commands, native feedback/mismatch handling and local/pinned sources.
A third-party cross-agent `npx skills` route is optional, not a prerequisite.
Python backend upgrades and skill updates are separate.

The [semantic-memory guide](semantic-memory.md) describes complete reader/writer schemas,
mutable memory operations and provider options. The [context contract](context.md)
explains frozen task/project snapshots. New task names use the standard risk policy;
no parameter tuning is required. `howhard flush` moves pending journal rows
into the outcome log. See [real-project evaluation](evaluation.md) for evaluating useful
learning; mechanical fixtures do not establish routing superiority.
