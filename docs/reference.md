# Model and storage contracts

These are the statistical baseline contracts. The optional [LLM semantic-memory layer](semantic-memory.md)
adds original-text reasoning, mutable memory and approximate predictive weights, with
its conservative interval policy reported separately.

## Selection

For task type `t`, family `m`, effort `e` and risk class `r`, minimize

```text
J = C_run + P_loud * C_loud + P_silent * C_silent + lambda_per_hour * T
```

Native callers supply a validated inventory of actual model/control cells before any
forecast. `pick`/`context` accept `--candidates FILE`; Python uses `native_candidates`.
Each cell has `model`, `effort` and optional `usd_per_mtok`. Native pricing is unknown
when omitted/null, even for a model with a packaged API price. If any candidate is
unpriced, the worker monetary term is omitted for **every** candidate. `cost_policy`
labels this partial comparison; unknown `C_run` is null and known `C_run` remains visible,
while `J` contains risk and time only. Fully priced inventories use the full objective.
The legacy registry-only path retains its original strict pricing behavior.

Costs are USD, and `T` is hours. Families and efforts are categorical axes. The selector
uses 4,000 joint posterior draws with a fixed internal seed for deterministic means and
95% intervals. It uses the 97.5th percentile of `P_silent` for admissibility. One fresh
joint posterior draw chooses the admissible cell with the smallest sampled `J`.
`explore_within`, when supplied, limits eligible cells by expected `J`.

The `flag-review` fallback samples among all cells and sets `needs_review` when no cell is
admissible. `lowest-bound` instead selects the lowest upper bound. The result includes the
seed, chosen concrete id, family, effort, class constants, cap status, every candidate's
probabilities, expected running cost, time, total cost, sampled cost, and explanations for
cheaper cells. Explanation components are signed differences and sum to the sampled cost
difference. Policy filters are explained separately. `summary` performs no Thompson draw.

## Outcome posterior

For each of `loud` and `silent`, the linear predictor is

```text
a + b[m] + c[e] + d[m,e] + u[t] + w[t,m] + v[t,e] + pi[p] + pi_task[p,t]
```

Softmax with the clean reference gives the three outcome probabilities. Normal priors
come entirely from the packaged TOML, optionally overridden by an advanced project.
At the prior mode, contract work at opus/high is 60% clean, 35% loud and 5% silent;
probability means differ because the prior is uncertain.

Contextual runs add shared, family-specific and effort-specific slopes on task-text
and project-snapshot features. The cost and time regressions add shared context slopes.
See the [context contract](context.md) for the complete encoding and historical semantics.

`fit(prior, rows, start=None)` uses weighted multinomial likelihood, exact gradients and
Hessians, Newton steps with step halving, and an absolute maximum Newton step below
`1e-9`. It refuses after 100 iterations. The joint covariance is the inverse negative
log-posterior Hessian at the mode. Full covariance imports are supported. Canonical cell
aggregation makes the result independent of row order. Every run's evidence weight comes
from its concrete model's registry entry. Project and project/task departures are fitted
but never transferred.

Unregistered native identities receive distinct backend-generated family labels with
zero-mean family effects, ordinary unit evidence weight and the existing packaged
uncertainties. New effort/control categories have zero-mean categorical effects; they
are not ordered or translated to known efforts. Identity/effort metadata persists in
`native-models.json` in the project store for later receipts/feedback, without making
historical candidates active. Known configured identities retain their original mapping;
ambiguous configured family/effort collisions are refused. The native input requires
1–32 distinct bounded cells; no-effort/null and `model_default` are unsupported native
controls. The skill discovers actual host capabilities; users supply no model priors,
family mappings or statistical tuning.

## Cost and time

Two independent exact conjugate normal regressions model log total tokens and log seconds:

```text
task[t] + family[m] + effort[e] + project[p] + normal noise
```

Only non-null measurements enter the corresponding regression. Zero measurements use
a one-token or one-second floor before taking a logarithm. The fixed noise sds are in
TOML. Conditional expected measurements are `exp(mu + noise_sd**2/2)`;
posterior draws also account for coefficient uncertainty. Tokens include all cached and
uncached input and output. `cost_usd` is retained for reporting and never fitted.

Model prices and the token mix are data. A blended USD-per-million price or five component
prices plus shares summing to one are accepted. Missing prices are refused on selection.
The built-in planning snapshot uses a cache-heavy mix and conservative long-prompt Haiku
pricing. Its model ids, effort support and prices were checked on 2026-10-29 against
[Anthropic's model registry](https://platform.claude.com/docs/en/models/overview),
[effort documentation](https://platform.claude.com/docs/en/build-with-claude/effort) and
[pricing](https://platform.claude.com/docs/en/about-claude/pricing).

## Records and amendments

Every JSONL row requires these base keys:

| Key | Value |
| --- | --- |
| `id` | Nonempty string, unique across logs and journal |
| `date` | `YYYY-MM-DD` |
| `ticket` | String or null |
| `task_type` | Known task name; zero-config also accepts new names as standard |
| `risk_class` | Recorded risk class; fitting uses the current mapping |
| `model` | Concrete id from the registry |
| `effort` | low, medium, high, xhigh, max, or model_default |
| `default_effort` | Historical resolved effort for model_default, otherwise null |
| `outcome` | clean, loud or silent |
| `cause` | Null for clean; depth, capability or unrecorded otherwise |
| `tokens` | Nonnegative integer or null |
| `seconds`, `cost_usd` | Finite nonnegative number or null |
| `note` | String, possibly empty |
| `amends` | Earlier row id or null |

Types, enums, missing keys and extra keys are strict. `observe --id` matching retries
return the original row without appending. Automatically derived context, timestamp/date,
risk and default resolution retain their first historical values; explicitly changed
inputs conflict. Equality never bypasses field/type validation. Raw `observe --native`
can admit a verified actual model/control without a receipt or retrospective forecast;
all feedback/ID validation precedes identity metadata persistence under the project lock.
Disk duplicate rows remain errors. Errors name the file, line and key.
Contextual rows additionally store `task_text`, `project_context` and `context_encoder`
together, with optional `decision_id`. New observations also record `observed_at`, an
ISO timestamp with timezone identifying when feedback became available. Legacy rows
keep their missing context and timestamp.
Amendments are new rows with `amends` set to an earlier row ID and null measurement
and cost fields. Since 0.3.1, they may correct an outcome to `clean`, `loud` or `silent`;
earlier versions accepted only `silent` amendments and cannot read the expanded cases.
The row keys and cache/prior formats are unchanged. The usual cause rule still applies:
null for clean, and depth, capability or unrecorded for loud/silent.
Chains replace the original run's effective outcome and cause, counting that run once
and retaining its original ID, decision, cell, task text, project snapshot, project,
weight and measurements. Raw rows remain intact. Amendments cannot override historical
context through the observation API. Targets within a file must precede their amendments.
Across files, targets must exist, have dates no later than their amendments, and form an
acyclic dependency graph. Dependency ordering keeps amendments valid when flushing moves
them into the first log ahead of a target in an additional log.

`flush` validates all files before writing. Duplicate refusals list every duplicate and
leave log and journal unchanged. It appends, fsyncs the log, then truncates and fsyncs the
journal. A shared advisory lock serializes storage commands within a project. A crash
between append and truncate can leave duplicate rows, which are refused on the next read;
there is no claim of an atomic transaction across two files. Preserve a copy and remove
the already-flushed journal entries after comparing their ids with the log.

## Capture root and durable state

The current project capture root is independent of the durable state root. `paths --json`
reports both; `--state-dir PATH` and `Engine(state_directory=...)` support intentional
store selection. Fresh zero-config Git worktrees share `howhard` below the repository's
common Git directory, with current context from the current checkout. Valid existing
checkout/main-worktree `.howhard` stores take precedence, with no automatic migration,
deletion or overwrite. Explicit configuration retains its declared paths and prior
settings; plain directories keep local `.howhard`. A stable random project identity lives
in `howhard-project.json` in the Git common directory, or `.howhard/project.json` for a
plain folder, independently of a custom store. Moving the project with its metadata
preserves identity. A claimed store refuses another project's use before cache, registry
or memory writes. New receipts carry that identity too; legacy receipts retain their
project-label validation. The resolved state subtree is excluded from capture and
`project.read`, including symlink aliases; a state root equal to or containing the capture
root is refused. Prepared calling-agent requests bind the worktree and reject sibling
retrieve/decide; old pending requests without that binding must be prepared again.
Frozen-receipt observations and learning share the store across that project's worktrees.

## Cache and transfer

The atomic NPZ cache stores labels, distributions, hashes, format and fit statistics.
SHA-256 covers the format, prior bytes, imported bytes, effective settings and registry,
and the bytes of every configured log plus the journal. Configuration is included because
weights, task mappings and project identities affect the posterior. Length prefixes keep
file boundaries distinct. Row reordering changes the data hash, but never the fit.

A matching cache performs zero fitting. Changed data with the same prior warm-starts the
same all-row Newton fit, matching labels and initializing new terms at their prior means.
A changed prior, invalid cache or missing cache starts a full fit. Cache deletion
reproduces the posterior. `fit.kind` and `fit.iterations` describe the current call.
An `Engine` is a snapshot: call `refresh()` after external writes.

Legacy exports use `howhard-prior-v1`; contextual exports use `howhard-prior-v2` with
encoder and feature vocabulary validation. Both contain labeled means, full shared covariances and
the fixed noise sds. Covariances are multiplied by `transfer.variance_inflation`, default
2.0. Project terms are excluded by label. Imports validate all keys, names, numerical
shapes, symmetry and positive definiteness before replacing the destination atomically.
Native identity-specific family/control terms are excluded from generic prior exports;
no concrete native identity mapping is transferred. Export vocabularies and existing
prior formats remain compatible. Shared terms absent from the file use hand-set priors. New-project terms always use
their independent hand-set priors. With explicit configuration, register custom task
types before importing them. Zero-config imports accept custom type names with standard
local policy, retaining their transferred distributions.

## Python API

```python
from howhard import Engine

engine = Engine()  # optional config path or directory=...
description = "Make transaction rollback safe"
choice = engine.pick("specified-implementation", description, seed=42)
table = engine.summary("specified-implementation", description)
engine.observe_decision(choice["decision_id"], "clean")
engine.flush()
engine.export_prior("shared-prior.json")
```

Reuse the engine for repeated picks: deterministic summaries are memoized by task text
and freshly captured project context until refresh. `engine.prior` and `engine.posterior` expose labeled means, covariances and fit
statistics. The package runs on Python 3.11+ on POSIX systems (the storage lock uses
`fcntl`). Runtime dependencies are limited to NumPy.
