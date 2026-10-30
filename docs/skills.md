# Install the howhard skill with the standard CLI

The canonical reusable skill lives at `skills/howhard/SKILL.md` with progressive
`references/hosts.md` and `references/workflow.md`. Install it with the established
[skills CLI](https://github.com/vercel-labs/skills), which discovers standard root skill
folders and manages installation, central content, agent links, lock metadata and updates.
howhard supplies the skill and Python backend; it does not implement a skill installer.

## Install and choose scope

The complete [Quickstart](quickstart.md) covers Python backend installation, first use
and real feedback. Skill installation needs Git and Node.js 22.20.0+ with npm/npx.
The examples pin the tested `skills` CLI version 1.7.4. From the intended project:

```sh
npx --yes skills@1.7.4 add xu-hao/howhard --list
npx --yes skills@1.7.4 add xu-hao/howhard --skill howhard --agent codex claude-code antigravity antigravity-cli --yes
npx --yes skills@1.7.4 list --agent codex claude-code antigravity antigravity-cli
```

Project scope is the default; the current working directory is the target. Remove agent
IDs you do not use. `antigravity` and `antigravity-cli` are separate targets. For user-wide
installation, explicitly append `--global` to the same `add` command. The installer owns
agent destinations; do not recreate its path selection or force independent copies.
Default installation uses a managed central skill with agent links where appropriate.
Inspect the standard command's summary and listing for the installed scope/locations.
See its [agent IDs](https://github.com/vercel-labs/skills/blob/main/docs/agents.md) and
[commands](https://github.com/vercel-labs/skills/blob/main/docs/commands.md).

## Local and immutable sources

A source checkout is directly discoverable; run this from the target project:

```sh
npx --yes skills@1.7.4 add /path/to/howhard --skill howhard --agent codex claude-code antigravity antigravity-cli --yes
```

For reproducible backend and skill installation from the same public Git revision,
capture a full commit hash containing 0.5.0 or later and use it for both sources:

```sh
howhard_revision="$(git ls-remote https://github.com/xu-hao/howhard.git refs/heads/main | cut -f1)"
test "${#howhard_revision}" -eq 40
python -m pip install "git+https://github.com/xu-hao/howhard.git@${howhard_revision}"
npx --yes skills@1.7.4 add "https://github.com/xu-hao/howhard/tree/${howhard_revision}/skills/howhard" --skill howhard --agent codex claude-code antigravity antigravity-cli --yes
```

Record and reuse the same full hash instead of resolving `main` again when reproducing
an installation. These are source installs, not PyPI release claims. The skill is included
in the source distribution, while the Python wheel supplies the backend rather than a
second skill copy. Installing/upgrading the Python package does not install/update a skill;
standard skill commands do not upgrade the Python backend.

## Invoke and enforce capabilities

In Codex mention `$howhard` or select it with `/skills`. In Claude Code invoke `/howhard`.
In Antigravity invoke `/howhard` or request the skill by name; IDE Customizations lists
active skills. Reopen the relevant agent if a newly installed skill is not visible.
Standard frontmatter permits automatic discovery; supporting references load when needed.

The skill teaches task/context reasoning, independent baseline and memory-conditioned
forecasts, actual host delegation, stable execution feedback, corrections and mutable
memory. The calling-agent path uses the host's existing model access without an extra
API key. A genuinely independent baseline needs a separate fresh context; when that is
unavailable, disclose it and use the explicitly statistical path.

A skill provides instructions, the howhard CLI/API stores validated evidence, and the
host's actual tools enforce execution controls. Installation does not add model access
or a delegation tool. The shipped planning registry contains Anthropic IDs; do not pass
those IDs to another provider or silently equate models. Restrict recommendations to
actual supported model/effort cells. If the host cannot enforce or report them, disclose
the capability gap and retain honest unscored metadata rather than false matched outcomes.
A project may declare an optional adapter in its own `AGENTS.md` or runbook for host
inventory, candidate restrictions, state conventions and execution receipts. No project
adapter or statistical tuning is required for standalone planning.

## Check, update and remove

Use the standard tool for lifecycle operations:

```sh
npx --yes skills@1.7.4 check
npx --yes skills@1.7.4 update howhard --project --yes
npx --yes skills@1.7.4 remove howhard --yes
```

`check` reports available updates for tracked sources and can return nonzero when updates
exist. Local source installations do not provide the same remote update tracking as Git
sources. Review changes before updating; an update intentionally changes the installed
skill. Use `--global` for a global update/removal and `list --global` to inspect that scope.
Omitting an agent filter on the skill-specific removal clears howhard across agent
references, including shared universal locations, without removing other named skills.

Version 0.5.0 retires the bespoke 0.4 `howhard skills install` command and
`howhard.skill_install` API before 1.0. Existing 0.4 skill copies are not automatically
managed by the standard tool. Preserve local edits and archive those specific howhard
installations outside skill discovery before standard installation; the former
`howhard-install.json` marker identifies such a copy. Do not remove unrelated skills,
instruction files or settings. No compatibility copier remains.

## Contributor integration smoke

The ordinary Python suite and wheel/docs gate do not download npm packages. Run the real
standard-CLI integration explicitly through AGCoord, using a task-owned npm cache:

```sh
agc run --label "standard skills integration" --resource cpu=1 -- env npm_config_cache=/tmp/howhard-issue-N-npm-cache DO_NOT_TRACK=1 .venv/bin/python scripts/check-skills.py
```

Replace `N` with the issue tracking the work. The smoke uses the current canonical root
layout in its own disposable project, discovers howhard, adds it for all four agent IDs,
reads installed references, lists it, repeats installation and removes it through the
standard CLI. It stops its subprocesses and removes its temporary project. No global
installation, real agent call or npm dependency is added to the ordinary test suite.
Retain the coordinator log with the change's validation evidence.
