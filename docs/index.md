# howhard

howhard helps you save AI credits by choosing the model and reasoning effort each
task needs. It starts with an LLM's judgment, learns from your project's real
results, and corrects earlier success records when bugs are discovered later.
Its selection accounts for retries, elapsed time and escaped defects.

Start with the [quickstart](quickstart.md) to use it on a real task. Read the
[technical report](technical-report.md) for the design, learning equations,
statistical assumptions and evaluation method. Savings depend on actual host
costs and the overhead of routing and learning; the [evaluation guide](evaluation.md)
explains how to measure them.

## Getting started

- [Quickstart](quickstart.md) — install the backend and plugin, route a task,
  record its outcome and learn.
- [Agent skills and host installation](skills.md) — native setup, routing and
  escaped-defect attribution in Codex, Claude Code and Antigravity.

```{toctree}
:hidden:
:caption: Getting started

quickstart
skills
```

## Technical deep dive

The technical report connects the full workflow. The component guides describe
each method and its operational boundaries in detail.

- [Technical report](technical-report.md) — the credit-saving objective, LLM
  priors, Bayesian capability learning, evidence-weighted memory and retrospective
  defect attribution.
- [Capability filtering](capability.md) — task-specific fit bands, threshold
  evidence, forecast budgets and widening.
- [LLM reasoning and semantic memory](semantic-memory.md) — independent
  forecasts, memory operations, predictive weights and feedback replay.
- [Experience retrieval](retrieval.md) — hybrid search, reranking and bounded
  context from earlier executions and written lessons.
- [Real-project evaluation](evaluation.md) — measure net cost and forecast
  quality with chronological trials on genuine work.
- [Rule retrieval benchmark](rule-benchmark.md) — representation and reranking
  diagnostics, separate from field routing-quality evidence.

```{toctree}
:hidden:
:caption: Technical deep dive

technical-report
capability
semantic-memory
retrieval
evaluation
rule-benchmark
```

## Reference

- [Model and storage contracts](reference.md) — selection, pricing, execution
  accounting, CLI/API behavior and persisted evidence.
- [Configuration](configuration.md) — optional inventories, policies, retrieval
  profiles and data locations.
- [Task and project context](context.md) — original task text, immutable project
  snapshots and decision receipts.
- [Concurrent clients](concurrency.md) — FIFO write admission, snapshot validity,
  refresh conflicts, bounded waits and retries.

```{toctree}
:hidden:
:caption: Reference

reference
configuration
context
concurrency
```

## Project

- [Contributor workflow](contributing.md) — issues, isolated worktrees and checks.
- [Packaging and release](releasing.md) — versioning, distribution validation
  and publication.
- [Changelog](changelog.md) — release history and links to archived upgrade guides.
- [Session handoff](session_handoff_format.md) — reporting completed and open work.

```{toctree}
:hidden:
:caption: Project

contributing
releasing
changelog
session_handoff_format
```
