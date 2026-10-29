# Changelog

## 0.2.0 — 2026-10-29

- Condition predictions on task descriptions and project snapshots that are captured
  afresh for every decision, including working-tree changes and optional measured coverage.
- Learn shared, family-specific and effort-specific contextual outcome effects, plus
  contextual token and time effects, using a deterministic local lexical encoder.
- Save decision receipts so later observations keep the original task and project
  context. Add task-text/file inputs and `observe DECISION_ID OUTCOME` to the CLI.
- Preserve missing context in legacy rows, retain original context on amendments,
  and version caches and prior transfers without exporting raw text or snapshots.
- Marginalize inactive coefficients exactly to keep fitting practical on small logs.

## 0.1.0 — 2026-10-29

- Start without configuration: packaged registry, prices, risk policies and priors,
  automatic local learning, and a standard policy for new task names.
- Choose model and effort with Bayesian clean/loud/silent outcomes, exact log-normal
  cost/time regressions, Thompson sampling, a silent-failure cap and explanations.
- Record outcomes in a strict append-only journal, amend escaped failures and flush
  into the outcome log. Cache posteriors and refit from all evidence with warm starts.
- Transfer shared learning between projects without exporting run records or project effects.
- Add the Python API, six CLI commands, acceptance tests, wheel checks, canonical
  documentation, contributor guardrails and release versioning.
