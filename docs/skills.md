# Native installation of the howhard skill

All three hosts support the common Agent Skills format: a `SKILL.md` with `name` and
`description` frontmatter and a Markdown workflow. The canonical content is
`skills/howhard/SKILL.md`, with progressive `references/hosts.md` and
`references/workflow.md`. Thin native plugin manifests point at this one root skill
folder. howhard implements no skill copier or plugin installer.

The complete [Quickstart](quickstart.md) covers backend installation, native skill
installation, invocation and real feedback. Keep the Python environment active when
starting the agent so it can find the `howhard` executable. No Node/npm installation
is required for the native path.

## First-party plugin installation

Use a current native client and choose the commands for your host:

| Host | Register/install | Explicit invocation |
| --- | --- | --- |
| Codex | `codex plugin marketplace add xu-hao/howhard`, then `codex plugin add howhard@howhard` | `$howhard` or `/skills` |
| Claude Code | `claude plugin marketplace add xu-hao/howhard`, then `claude plugin install howhard@howhard` | `/howhard:howhard` |
| Antigravity (`agy`) | `agy plugin install https://github.com/xu-hao/howhard` | `/howhard` |

These CLI flows install/register in the user's plugin environment. Claude Code also
supports `--scope project` and `--scope local` for marketplace addition and plugin
installation; select the desired scope explicitly for both steps. Native clients own
installation, caching, enable/disable and removal. Reopen the agent if a newly installed
skill is not visible.

For a source checkout, native marketplace commands can use the checkout directory in
place of `xu-hao/howhard`; Antigravity can install that directory directly. The portable
root `plugin.json` and `.agents/plugins/marketplace.json` support Codex. Claude Code uses
`.claude-plugin/plugin.json` plus its marketplace, while Antigravity selects
`.antigravity-plugin/plugin.json`. Default `skills/` discovery lets each wrapper load
identical content. The Codex marketplace's `source.path` is relative to the repository
root, not to `.agents/plugins/`.

The wrappers contain no hooks, authentication setup, MCP server, provider calls or model
controls. Plugin installation is separate from Python backend installation and does not
publish the plugin into any official public directory. The skill remains in the source
distribution; the Python wheel supplies the backend without another skill copy.

Official format/packaging references, checked 2026-10-30:

- [Codex portable plugins and marketplaces](https://developers.openai.com/plugins/build/plugins)
- [Claude Code plugin manifests](https://code.claude.com/docs/en/plugins-reference) and
  [marketplaces](https://code.claude.com/docs/en/plugin-marketplaces)
- [Antigravity plugin authoring](https://antigravity.google/docs/build-plugins) and
  [native plugin installation](https://antigravity.google/docs/plugins)

## Native lifecycle

Inspect installed plugins with `codex plugin list`, `claude plugin list`, or
`agy plugin list`. Codex refreshes marketplace snapshots through `codex plugin marketplace
upgrade`; use its `--help` for available selectors. Claude Code provides `claude plugin
update howhard@howhard`. Antigravity's plugin UI/manager owns installed updates; consult
`agy plugin help` for the current CLI operations instead of assuming a generic update
flag. Native uninstall commands are `codex plugin remove howhard@howhard`,
`claude plugin uninstall howhard@howhard` and `agy plugin uninstall howhard`.
Python package upgrades do not update the installed skill, and native plugin updates do
not upgrade the Python package.

## Automatic activation and delegation

[Codex can activate skills implicitly](https://learn.chatgpt.com/docs/build-skills)
when a task matches their descriptions; `$howhard` and `/skills` request the workflow
explicitly. Installation makes the skill discoverable. It does not run every skill
on each subagent start, and this plugin contains no startup hook.

Use howhard in the coordinating agent before spawning a worker: selecting the model and
effort must happen before that worker starts. Giving a worker a task does not prove it
loaded howhard, and the worker need not rerun routing just to execute its assigned task.
For a required workflow, include an explicit howhard instruction in the coordinating
agent's task or project instructions. Skill discovery and activation remain host-owned.

(require-howhard-before-delegation)=
### Require howhard before delegation

Add the following to the coordinating agent's persistent project instructions after
installing the skill and backend. For Codex, use the project's `AGENTS.md`;
[Codex loads project instructions before working](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
For another host, use its native project-instruction mechanism.

```text
Before delegating an authorized task to a worker, use the installed howhard skill to
select the worker's native model and reasoning effort from the host's supported
controls. Respect explicit user model choices and required review. Start the worker
with the selected controls; if routing is unavailable or the controls cannot be
enforced, report the limitation instead of silently bypassing routing.
After execution, record the outcome and measured usage under a stable execution ID, then
update howhard memory. On a host whose native spawn API applies explicit model and effort
overrides, accepted controls plus completed execution match the requested cell unless
the host reports launch failure, ignored controls, substitution or contradictory
metadata. If no resolved identity is returned, note acceptance and completion as host-API
provenance, not backend/provider attestation, and use the matched receipt. Otherwise,
match only on exact reported actual model/effort; use raw feedback for a known different
cell and leave feedback unscored only when neither accepted overrides nor an exact
reported identity/control establishes a match. Usage stays measured-only.
The independent baseline forecast inside howhard is part of routing; do not
recursively route that forecast through howhard.
```

This makes howhard use an instruction-level requirement for the coordinating agent.
The plugin does not enforce it at the delegation tool boundary. A host-enforced
guarantee would require the host to reject worker launches without a valid howhard
decision matching the requested model and effort. Installing the plugin or adding
this instruction does not implement that check or authorize additional delegation.

## Capability boundaries

The skill uses original task/current project context, an independent baseline, retrieved
memory and validated prospective forecasts. The calling-agent workflow uses the host's
existing LLM without an additional API key; an independent baseline still needs a fresh
context without memories/history. If that is unavailable, disclose it and use the labeled
statistical path.

The skill is the only consumer integration. It discovers actual host controls and
calls `context --candidates FILE`, `retrieve`, `decide`, `observe`, `evidence` and the
memory commands directly. Temporary JSON inputs/responses are generated by the host
agent, not user configuration or tracked project tooling. Unknown native model IDs get
separate neutral prior families; generic categorical controls such as ultra remain exact
rather than mapping to max/default. Missing candidate model/control values remain a
capability gap. A separate resolved worker identity need not be returned after execution
when accepted explicit overrides and task completion establish attribution as above.
No project bridge, model mapping, statistical settings, lock feature or extra audit
ledger is needed. Native manifests and the stateful Python backend remain separate.

The backend validates and binds the active native inventory before producing forecast
panels. Historical registry metadata does not reactivate old candidates. Omitted native
worker price means unknown even for a shipped model ID; any unpriced cell causes worker
money to be omitted uniformly from comparison and labeled in `cost_policy`. Known costs
and actual measurements remain visible, while unknown cost is null, not zero.

`howhard paths --llm off --json` reports current capture and durable state roots. Fresh
Git worktrees share the repository common directory's howhard store, while current task
context is captured from the current checkout. Explicit configuration owns its declared
paths; plain folders use local `.howhard`. A prepared request is
bound to its worktree, but frozen-receipt feedback can be recorded in a sibling worktree.
`--state-dir` supports intentional store selection and refuses another project's claimed
store. No symlink/config wrapper is required to preserve worktree memory.

Use stable execution IDs for feedback. `evidence ID --raw` reconciles an ambiguous write;
matching retries return the first observation and explicit conflicts are refused. If a
different verified cell ran, raw `observe --native` admits its identity without creating
a retrospective forecast, retaining the original task/snapshot and a provenance note.
It must not attach the unexecuted receipt. Corrections replace effective evidence once;
`memory learn-context` includes all original/amendment notes in `outcome_history` along
with the current event key. The installed skill's workflow reference has complete native
CLI/API examples. When neither accepted overrides nor exact reported identity establishes
a match, unknown actual identities remain unscored and reported, not invented.

## Optional third-party cross-agent installation

The [skills CLI](https://github.com/vercel-labs/skills) is an optional third-party tool,
not any host's first-party installer. Version 1.7.4 requires Node.js 22.20.0+ and npm/npx.
From the target project it can discover and install the same root skill:

```sh
npx --yes skills@1.7.4 add xu-hao/howhard --list
npx --yes skills@1.7.4 add xu-hao/howhard --skill howhard --agent codex claude-code antigravity antigravity-cli --yes
npx --yes skills@1.7.4 list --agent codex claude-code antigravity antigravity-cli
```

Project scope is the default. `--global` selects user-wide installation. Antigravity
App/IDE and CLI have separate agent IDs. This tool manages its central content, agent
links and lock metadata; do not recreate its destination logic or force independent
copies. Use its own lifecycle operations:

```sh
npx --yes skills@1.7.4 check
npx --yes skills@1.7.4 update howhard --project --yes
npx --yes skills@1.7.4 remove howhard --yes
```

The named removal without an agent filter clears that skill across agent references.
Local checkout sources such as `/path/to/howhard` are supported; local installs do not
have the same remote update tracking as Git sources. For immutable sources, replace
`FULL_COMMIT` with a full 40-character commit containing the current release:

```sh
python -m pip install "git+https://github.com/xu-hao/howhard.git@FULL_COMMIT"
npx --yes skills@1.7.4 add https://github.com/xu-hao/howhard/tree/FULL_COMMIT/skills/howhard --skill howhard --agent codex claude-code antigravity antigravity-cli --yes
```

These are source installs, not PyPI release claims. Choose one installation route per
host to avoid duplicate skill names and conflicting ownership.

## Contributor validation

Native validation is an explicit read-only contributor check:

```sh
agc run --label "native plugin validation" --resource cpu=1 -- .venv/bin/python scripts/check-plugins.py
```

It runs Claude's strict marketplace check and Antigravity's checkout validation, then
strictly validates an exact disposable Antigravity payload with its manifest/shared skill.
The multiprovider checkout intentionally has contributor `AGENTS.md` outside plugin
rules and a portable root manifest superseded by Antigravity's own manifest; that client
reports these as warnings. Claude's validator also prints separate public-directory
submission advisories for missing license/art; those are not its manifest-validation
verdict, and this project does not claim public-directory submission. Codex has no
standalone plugin validator in the checked CLI. Its command shapes/documented portable
schema were verified; unregistered read-only local listing returned no entries, so it is
not claimed as a successful installation. No global native install or live agent call
is exercised by these checks.

The optional third-party integration smoke is separate from the ordinary Python gate:

```sh
agc run --label "optional skills integration" --resource cpu=1 -- env npm_config_cache=/tmp/howhard-issue-N-npm-cache DO_NOT_TRACK=1 .venv/bin/python scripts/check-skills.py
```

Replace `N` with the tracking issue. It uses its own disposable project and task-owned
npm cache to discover/add all four agent IDs, read references, list/repeat/remove through
the real CLI, and stop/clean its processes and temporary project. No npm dependency or
network-enabled agent setup is added to ordinary tests. Retain coordinator logs as
validation artifacts.
