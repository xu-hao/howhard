# Optional advanced configuration

The default workflow requires no file. `howhard.toml` in the current directory is loaded
when present; `--config PATH` chooses another file. Paths are relative to that file.
An explicit config must state `on_cap_unmet`; other settings overlay packaged defaults.
Unknown keys are refused. This reference is for integrators, not a user setup checklist.

```toml
on_cap_unmet = "flag-review" # or lowest-bound
project = "local"
journal = ".howhard/pending.jsonl"
cache = ".howhard/posterior.npz"

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
can be reassigned with `task_types`. Class names are critical, contract, standard, docs
and routine. The first log belongs to the current project and is the flush destination;
additional logs can name other projects.

Known model mappings can be overlaid by `[models."concrete-id"]`:

```toml
[models.claude-opus-5-5]
family = "opus"
efforts = ["low", "medium", "high", "xhigh", "max"]
weight = 1.0
```

An explicitly mapped concrete ID inherits that family's learned effects. A model entry
requires a known family and supported efforts; omitted evidence weight means 1.0.
Unknown native IDs need no configuration: the backend assigns distinct neutral families.
Every `pick`, `summary` and `context` request supplies its actual inventory through
`--candidates FILE` (Python: `native_candidates`). Registry entries do not activate
candidates. Supply known host worker prices as optional `usd_per_mtok` in that inventory;
unknown prices remain omitted/null. Family selection flags, model default efforts,
component API prices and token-mix configuration are not accepted.

Maintainers can supply an entire strictly validated prior with `[prior] path = "prior.toml"`
or recursively replace individual assumptions under `[prior.overrides]`. For example,
`[prior.overrides.outcome.loud.a]` accepts `mean` and positive `sd`. All missing terms in
a supplied prior file, unknown terms, unknown vocabularies and invalid sds are refused.
This does not alter any provider model weights or train a provider model.

`[prior] import = ".howhard/imported-prior.json"` is the default transfer destination.
An absent file starts with the shipped prior. `import-prior FILE` validates before writing
there. `[transfer] variance_inflation = 2.0` overrides the exported covariance multiplier.
`--cache` and `--journal` are available for integration overrides.

Native host use does not require this configuration. The skill supplies discovered
model/control cells through `--candidates`, and the backend owns their minimal historical
metadata and neutral defaults. Native price omission means unknown. `--state-dir` redirects standard `.howhard/` data paths while honoring custom
configured paths, and keeps current project capture separate. Explicit configuration
without that override retains its existing data locations and prior settings.
