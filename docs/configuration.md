# Optional advanced configuration

The default workflow requires no file. `howhard.toml` in the current directory is loaded
when present; `--config PATH` chooses another file. Paths are relative to that file.
An explicit config must state `on_cap_unmet`; other settings overlay packaged defaults.
Unknown keys are refused. This reference is for integrators, not a user setup checklist.

```toml
on_cap_unmet = "flag-review" # or lowest-bound
project = "local"
journal = ".howhard/pending.jsonl"

[[logs]]
path = ".howhard/outcomes.jsonl"
project = "local"

[task_types]
my-task = "standard"

[classes.standard]
C_loud = 50.0
C_silent = 1000.0
lambda_per_hour = 25.0
epsilon = 0.10
# explore_within = 0.20
```

An explicit config requires `--risk` when picking an unknown task. Known taxonomy types
can be reassigned with `task_types`. Packaged classes are critical, contract, standard, docs
and routine; explicit projects may define additional classes. The first log belongs to the current project and is the flush destination;
additional logs can name other projects.

Known model mappings can be overlaid by `[models."concrete-id"]`:

```toml
[models.claude-opus-5-5]
family = "opus"
efforts = ["low", "medium", "high", "xhigh", "max"]
weight = 1.0
```

An explicit mapping labels the model's protocol cells and supplies its evidence weight;
it does not inherit learned family effects. Omitted evidence weight means 1.0. Unknown
native identities need no configuration and receive distinct opaque identity labels.
Every `pick`, `summary` and `context` request supplies its actual inventory through
`--candidates FILE` (Python: `native_candidates`). Registry entries do not activate
candidates. Supply known host worker prices as optional `usd_per_mtok` in that inventory;
unknown prices remain omitted/null. Family selection flags, model default efforts,
component API prices and token-mix configuration are not accepted.

Statistical priors, imported priors, cache paths and transfer tuning are no longer
accepted. `--journal` remains available for an integration path override. Provider
selection has no mode switch: Python accepts an adapter, and standalone inference uses
`HOWHARD_LLM_COMMAND` or an authenticated Claude CLI. Calling-agent commands do not
resolve a provider.

Native host use does not require this configuration. The skill supplies discovered
model/control cells through `--candidates`, and the backend owns their minimal historical
metadata. Native price omission means unknown. `--state-dir` redirects standard `.howhard/` data paths while honoring custom
configured paths, and keeps current project capture separate. Explicit configuration
without that override retains its existing data locations and policies.
