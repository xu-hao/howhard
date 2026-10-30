# Host capabilities and native identity

Installation/discovery and execution controls are separate. Inspect the actual tool
schema and current account/settings; a model listed by a provider is not automatically
available to this host. No project adapter or statistical configuration is required.

| Host | Invoke the installed skill | Execution boundary |
| --- | --- | --- |
| Codex | `$howhard` or `/skills` | Use the live delegation tool's supported model and effort fields; other Codex surfaces may expose different controls. |
| Claude Code | Native plugin: `/howhard:howhard`; standalone skill: `/howhard` | Inspect the actual agent/subagent controls and resolved identity when reported. Provider names or CLI defaults do not prove which cell ran. |
| Antigravity (`agy`) | `/howhard` or request it by name | Its documented custom-subagent model values include inherit/flash/pro, with no effort field in that frontmatter table. Inspect the current surface; do not invent a concrete model ID or effort parameter for it. |

Supply only cells with known native model and explicit effort/control identities to
howhard's `--candidates` input. The backend preserves exact values; it does not translate
provider IDs or map ultra to max. Bounded effort identifiers containing letters, digits,
underscores and hyphens are categorical controls, with no assumed ordering for newly
introduced values. Null/no-effort and `model_default` are not native inventory controls.
If a host cannot expose the model and effort/control values needed for a concrete
inventory, report the limitation instead of selecting a falsely labeled cell.

On any host whose native spawn API applies explicit model and effort overrides, an
accepted override followed by completed task execution is sufficient attribution to the
requested cell unless the host reports launch failure, ignored controls, substitution or
contradictory execution metadata. If the host does not return a separate resolved
identity, attach feedback to the matched decision receipt and record the accepted
explicit override and completed execution in the existing observation note. This is
provenance based on the host API contract, not backend or provider attestation. A
recommendation, planned call or accepted but unexecuted launch is not execution evidence.
Without accepted explicit controls, match only when the reported actual model and
control exactly match the chosen cell. Inherited or ambiguous settings alone do not
establish a match; use raw feedback for a known different actual cell and leave feedback
unscored only when the actual cell is unknown. Add usage only when measured.

Unregistered native model IDs automatically receive distinct opaque identity labels
and ordinary evidence weight, without fitted family effects. Backend-emitted family
keys are internal forecast keys, not a request to classify the model. Their identity/control metadata persists for later
feedback but does not make old candidates active in future decisions. Existing configured
model identities retain their historical mapping; an ambiguous configured family/effort
collision is refused rather than silently merging distinct forecasts.

Native worker pricing is unknown unless intentionally supplied for that execution
surface. A subscription or bundled allowance is not an observed zero price or an API
list-price quote. Omit it when unknown. The backend then excludes worker money uniformly
from the entire candidate comparison, keeps unknown `C_run` null and labels the partial
objective. It still records actual measured costs/resources when supplied.

Delegate using the chosen exact native controls. If another known cell actually ran, raw
`observe --native` records it with frozen task/project context and a provenance note,
without crediting the original forecast. When neither accepted explicit controls nor an
exact reported identity establishes a match, leave feedback unscored if actual
identity/control is unknown. Preserve the known facts in the host's existing task
response; create no project audit wrapper.

Official capability/discovery references, checked 2026-10-30:

- [Codex skills](https://learn.chatgpt.com/docs/build-skills)
- [Claude Code skills](https://code.claude.com/docs/en/skills)
- [Antigravity skills](https://antigravity.google/docs/skills)
- [Antigravity subagents](https://antigravity.google/docs/subagents/)
- [Antigravity model selection](https://antigravity.google/docs/models/)
