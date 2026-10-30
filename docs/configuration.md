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

The packaged registry is overlaid by `[families.NAME]` and `[models."concrete-id"]`:

```toml
[families.opus]
tier = 3
current = "claude-opus-5-5"
selectable = true

[models.claude-opus-5-5]
family = "opus"
default_effort = "medium"
efforts = ["low", "medium", "high", "xhigh", "max"]
weight = 1.0
usd_per_mtok = 0.638
```

A new concrete id inherits its family's learned effects. Evidence from another provider
can map to an equivalent family with a lower positive `weight`. A model entry requires
family, supported efforts and default effort. A missing weight means 1.0. Only current,
selectable family ids are candidates; older ids can remain registered for historical
rows. Set `selectable = false` to remove a family from recommendations.

Instead of `usd_per_mtok`, supply
`prices = {input=..., output=..., cache_write_5m=..., cache_write_1h=..., cache_read=...}`.
The project-level `[token_mix]` has exactly those five keys, nonnegative shares summing
to one within `1e-6`. A model can have only one price form. Specifying a blended price
for a shipped id replaces its component prices.

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
metadata and neutral defaults. Native price omission means unknown, not the packaged API
price. `--state-dir` redirects standard `.howhard/` data paths while honoring custom
configured paths, and keeps current project capture separate. Explicit configuration
without that override retains its existing data locations and prior settings.
