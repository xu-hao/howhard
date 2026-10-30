# Reusable howhard skill

howhard 0.4.0 packages one canonical skill for Codex, Claude Code and Antigravity (`agy`).
It guides task/context reasoning, model and effort recommendations, actual host execution,
outcome corrections and writable semantic memory. Its source is
`src/howhard/skills/howhard/SKILL.md`, with focused `references/hosts.md` and
`references/workflow.md` in the same bundle. The installer copies these packaged resources;
it works from an installed wheel without a source checkout or installation-time network.

A skill teaches a workflow. The howhard CLI/API validates decisions and stores evidence.
The host's delegation tool controls execution. Installing the skill does not add host
model access, a new delegation tool, or enforcement of a recommended effort.

## Install the package

From a current source checkout, use a Python 3.11+ environment:

```sh
python -m pip install .
howhard --version
```

Or capture the public `main` revision and install that exact immutable Git commit:

```sh
howhard_revision="$(git ls-remote https://github.com/xu-hao/howhard.git refs/heads/main | cut -f1)"
test "${#howhard_revision}" -eq 40
python -m pip install "git+https://github.com/xu-hao/howhard.git@${howhard_revision}"
howhard --version
```

Record the printed revision for reproducible later installs; replace the variable with
that same full commit hash when reusing it. Choose a revision containing 0.4.0 or later
for the skill installer. These are source installation commands, not a claim of a PyPI
release. Initial package/dependency installation can require network; `skills install`
uses only local packaged resources afterward.

## Install into a host

Run from the intended workspace, or supply its existing directory. Git is not required
for the target directory. Project scope is the default:

```sh
howhard skills install --host codex
howhard skills install --host claude-code
howhard skills install --host antigravity
howhard skills install --host all --directory /path/to/project --json
```

For all projects belonging to the current user, choose user scope explicitly:

```sh
howhard skills install --host codex --scope user
howhard skills install --host claude-code --scope user
howhard skills install --host antigravity --scope user
howhard skills install --host all --scope user --json
```

`--directory` applies only to project scope; combining it with user scope is an error.
User destinations use the current user's home directory. Installer output names the
package version, host, scope, exact absolute locations, associated hosts and a per-location
status of `installed` or `unchanged`.

| Host | Project destination | User destination |
| --- | --- | --- |
| Codex | `.agents/skills/howhard` | `~/.agents/skills/howhard` |
| Claude Code | `.claude/skills/howhard` | `~/.claude/skills/howhard` |
| Antigravity App/IDE | `.agents/skills/howhard` | `~/.gemini/config/skills/howhard` |
| Antigravity CLI | `.agents/skills/howhard` | `~/.gemini/antigravity-cli/skills/howhard` |

Codex and Antigravity share one project destination. `--host all` writes two project
locations or four user locations; Antigravity user installation covers both surfaces.
These paths follow the official [Codex](https://learn.chatgpt.com/docs/build-skills),
[Claude Code](https://code.claude.com/docs/en/skills) and
[Antigravity](https://antigravity.google/docs/skills) discovery documentation, checked
2026-10-30. Legacy Antigravity discovery paths are not installation defaults.

The installed bundle is a copy, with `howhard-install.json` identifying its package
version. The installer leaves project instruction files, authentication, host settings,
other skills and unrelated siblings alone. It does not construct an Engine, inspect
project context, create `.howhard` state or resolve an LLM provider.

## Repeat, upgrade and remove

An identical complete bundle is accepted without rewriting files. Different content,
extra files/directories, symlinked skill destinations and symlinked discovery ancestors
are conflicts. All destinations are checked before installation begins, so an existing
conflict in one host prevents writes to the other hosts. Filesystem/I/O failures during
copying can leave a partial new directory; inspect it before retrying. The installer
does not claim a transaction across destinations or automatic recovery after a crash.

There is no implicit replacement or upgrade. To upgrade or replace an edited install,
inspect and archive that particular `howhard` directory outside host skill discovery,
then rerun installation from the desired package version. Keep unrelated siblings.
To remove the skill, remove only the installed `howhard` directories listed by the
installer after preserving any local edits. A package upgrade alone does not alter
installed skill copies.

## Invoke and use

Automatic discovery uses the skill's name and description; its body and references load
when relevant. In Codex mention `$howhard` or select it with `/skills`; restart if a
change is not visible. In Claude Code invoke `/howhard`; use `/reload-skills` or restart
if a newly created skills directory is not visible. In Antigravity ask for howhard by
name; App and CLI also expose skills as slash commands, and IDE Customizations lists
active skills. Restart the relevant surface if discovery has not refreshed.

After loading, ask it to recommend a model/effort for the actual task and current project,
then record real feedback. The skill routes to standalone CLI/API examples and the
calling-agent protocol in its installed `references/workflow.md`. For a standalone
statistical planning check:

```sh
howhard pick task --text 'Repair transaction rollback at its public boundary.' --risk standard --llm off --json
```

This command recommends from the packaged registry; it does not execute the work.
The shipped registry contains Anthropic IDs. Installing into Codex or Antigravity does
not make those IDs valid host controls. Match actual host-supported model/effort cells
through a validated registry and restrict candidates with Python `Engine.pick(...,
candidates=...)` or an explicitly restricted project configuration. Report a capability
gap when no truthful mapping or enforceable control exists. Do not relabel another
provider's model or claim that a recommendation changed a running agent.

For independent calling-agent reasoning, give the returned `baseline_request` a fresh
stateless model context, then use the separate memory reader and validated forecasts.
If an independent call is unavailable, disclose that limitation and use the explicitly
statistical path. Follow the [semantic-memory contract](semantic-memory.md) for the
request/response protocol and [configuration](configuration.md) for optional registries.

Projects may declare an optional adapter in their own `AGENTS.md` or runbook to supply
host inventory, supported controls, candidate restrictions, state paths and stable
execution receipts. The reusable skill consults that declared contract; it does not
hardcode a project, taxonomy or provider-equivalence assumption. No adapter is required
when the existing registry already matches executable host cells.

Retain stable execution/run IDs and actual model/effort separately from requested
controls. Record measured outcomes and usage only for executed actions, append corrections
instead of duplicate original runs, and retry memory learning for the same effective
event. Memory remains mutable through add/edit/remove/restore/merge/split operations;
new content receives no retrospective predictive credit. Missing identity or unsupported
controls remain explicitly unscored rather than becoming false matched evidence.
Mechanical validation of this bundle does not establish routing quality; use the
[real-project evaluation protocol](evaluation.md) for that question.
