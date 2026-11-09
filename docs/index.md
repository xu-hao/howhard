# howhard documentation

- [Quickstart](quickstart.md) — install, choose, record an outcome and learn, with no setup.
- [Model and storage contracts](reference.md) — LLM forecasts, cost/risk selection, strict rows,
  reproducibility and storage.
- [Task-specific capability filtering](capability.md) — learned fit bands, threshold evidence,
  bounded fine forecasts, widening and filter-quality reports.
- [Upgrading to 3.0](upgrading-3.0.md) — two-stage requests and accepted-result selection.
- [Concurrent clients](concurrency.md) — backend-owned FIFO write admission, bounded waits,
  cancellation, queue status and safe retries across worktrees.
- [Task text and changing project context](context.md) — original task/project text, immutable
  snapshots and decision receipts.
- [LLM reasoning and writable semantic memory](semantic-memory.md) — reader/writer
  adapters, memory operations, feedback weights and agent tools.
- [Execution experience retrieval](retrieval.md) — replaceable BM25/HNSW retrieval,
  pretrained instruction-aware reranking and bounded precedent context.
- [Reusable skills and host installation](skills.md) — routing, escaped-defect attribution,
  Codex, Claude Code and Antigravity discovery, native installation and execution limits.
- [Rule retrieval benchmark](rule-benchmark.md) — opt-in representation/reranking diagnostics
  with pytest-probability, separate from field routing-quality evidence.
- [Real-project evaluation](evaluation.md) — chronological field trials and genuine
  observed outcomes, with implementation fixtures kept separate.
- [Advanced configuration](configuration.md) — external-router candidate lists, optional
  registry, policies and data locations.
- [Upgrading to 2.0](upgrading-2.0.md) — portable execution accounting and explicit money bases.
- [Upgrading to 1.0](upgrading-1.0.md) — breaking interfaces, backend/skill updates and
  preserved historical state.
- [Contributor workflow](contributing.md) — issue tracking, worktrees and coordinated checks.
- [Packaging and release](releasing.md) — versions, distribution validation and publication.
- [Changelog](changelog.md) — user-facing changes by version.
- [Session handoff](session_handoff_format.md) — reporting changed work.

```{toctree}
:hidden:

quickstart
reference
capability
upgrading-3.0
concurrency
context
semantic-memory
retrieval
skills
evaluation
rule-benchmark
configuration
upgrading-1.0
upgrading-2.0
contributing
releasing
changelog
session_handoff_format
```
