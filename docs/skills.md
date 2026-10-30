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

| Host | Register/install | Invoke |
| --- | --- | --- |
| Codex | `codex plugin marketplace add xu-hao/howhard`, then `codex plugin add howhard@howhard` | `$howhard` or `/skills` |
| Claude Code | `claude plugin marketplace add xu-hao/howhard`, then `claude plugin install howhard@howhard` | `/howhard:howhard` |
| Antigravity (`agy`) | `agy plugin install https://github.com/xu-hao/howhard` | `/howhard` |

These CLI flows install/register in the user's plugin environment. Claude Code also
supports `--scope project` and `--scope local` for marketplace addition and plugin
installation; select the desired scope explicitly for both steps. Native clients own
installation, caching, enable/disable and removal. Reopen the agent if a newly installed
skill is not visible. Older Codex clients with the built-in `$skill-installer` can instead
be asked to install the howhard skill from `xu-hao/howhard`, path `skills/howhard`; follow
that first-party skill's instructions rather than copying files manually.

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

## Capability boundaries

The skill uses original task/current project context, an independent baseline, retrieved
memory and validated prospective forecasts. The calling-agent workflow uses the host's
existing LLM without an additional API key; an independent baseline still needs a fresh
context without memories/history. If that is unavailable, disclose it and use the labeled
statistical path.

A skill provides instructions, the howhard CLI/API stores validated evidence, and the
host's tools enforce execution controls. Installation does not add model access or a
delegation tool. The shipped registry contains Anthropic IDs: never silently relabel
another provider's model. Restrict recommendations to actual supported cells and disclose
unsupported effort/model controls or unknown execution identity. Keep such runs honestly
unscored instead of attaching them to a different decision receipt. An optional project
adapter can be declared by that project's own instructions; standalone planning needs
no adapter or statistical tuning.

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
`FULL_COMMIT` with a full 40-character commit containing 0.5.0 or later:

```sh
python -m pip install "git+https://github.com/xu-hao/howhard.git@FULL_COMMIT"
npx --yes skills@1.7.4 add https://github.com/xu-hao/howhard/tree/FULL_COMMIT/skills/howhard --skill howhard --agent codex claude-code antigravity antigravity-cli --yes
```

These are source installs, not PyPI release claims. Choose one installation route per
host to avoid duplicate skill names and conflicting ownership.

## Migration and contributor validation

Version 0.5.0 retires the bespoke 0.4 `howhard skills install` command and
`howhard.skill_install` API before 1.0. Preserve local edits and archive those particular
old howhard copies outside skill discovery before native or optional third-party
installation; `howhard-install.json` identifies a former copy. Leave unrelated skills,
instruction files and settings alone. No compatibility copier remains.

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
