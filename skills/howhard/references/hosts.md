# Host inventory and execution controls

The same skill files work in all three hosts. Discovery, tool availability, account
access and execution controls are separate capabilities. Check the live tool schema,
current host settings and any project-declared adapter; documentation examples do not
prove a model is available to this account.

| Host | Discover the installed skill | Execution boundary |
| --- | --- | --- |
| Codex | Mention `$howhard` or select it with `/skills`; automatic discovery is enabled. Restart if a new skill is not visible. | Use the actual delegation tool schema. Supply model/effort only when those controls exist and the requested values are supported. A model available in an API or another host is not automatically available to this tool. |
| Claude Code | With the native howhard plugin invoke `/howhard:howhard`; a standalone skill uses `/howhard`. Relevance matching can load either. Restart if the installed skill is not visible. | Inspect the available agent/subagent tool and model settings. CLI defaults and provider names do not prove that a particular delegated task used the requested concrete model or effort. Preserve the resolved identity when available. |
| Antigravity (`agy`) | Ask the agent to use the howhard skill; check the host's skill listing/discovery UI and restart if it is not visible. | Use only the current surface's tools and controls. The documented custom-subagent model values are `inherit`, `flash`, and `pro`; that frontmatter table has no effort field. Do not assume an arbitrary concrete model ID or five-level effort can be passed to `invoke_subagent`. Check current model settings and report unsupported controls. |

The shipped howhard registry contains Anthropic model IDs. It is a planning snapshot,
not a universal inventory or a guarantee those IDs exist in the host. Installation
does not add native Codex or Antigravity registry entries. Without a validated compatible
registry and enforceable host controls, report the gap instead of routing under a false
identity. Existing family labels are a closed statistical vocabulary; do not relabel
another provider's model as an Anthropic model to make selection succeed.

A project may explicitly declare an optional adapter that owns a real inventory,
registry mapping, candidate restriction, state directory and execution receipts. Read
its contract before running it. A defensible approximate family mapping must be explicit,
documented and weighted as such; do not invent equivalence or prices. Keep requested
and actual host identities in that adapter's audit. If the mapping cannot represent the
actual executed cell, do not attach its outcome to a different cell's decision receipt.
Standalone use needs no adapter when registry cells already match the host controls.

If a host supports only an inherited model or a fixed effort, restrict recommendations
to the accurately known executable cell. If the inherited identity is unknown, disclose
it and keep the run unscored until it can be attributed. A host may require the user to
change a setting before a new execution; a recommendation alone cannot change a running
agent. Do not quietly substitute a model or effort after selecting a receipt.

Official discovery/control references, checked 2026-10-30:

- [Codex skills](https://learn.chatgpt.com/docs/build-skills)
- [Claude Code skills](https://code.claude.com/docs/en/skills)
- [Antigravity skills](https://antigravity.google/docs/skills)
- [Antigravity subagents](https://antigravity.google/docs/subagents/)
- [Antigravity model selection](https://antigravity.google/docs/models/)
