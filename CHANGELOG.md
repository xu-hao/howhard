# Changelog

## Unreleased

- Reduce the spiral project mark's uniform line weight from 20 to 17 units while
  retaining its 4.5 turns and solid black-and-white palette (#93).

- Increase the abstract geometric spiral project mark from 2.5 to 4.5 turns and
  thicken its uniform stroke from 14 to 20 units, preserving its solid
  black-and-white palette (#91).

- Replace the raster project mark with a geometric SVG spiral using only solid
  black and white, with uniform lines and no shading or texture (#89).

- Publish the clean monochrome spiral project mark in the README and documentation
  homepage, with image provenance recorded under `docs/assets/` (#87).

- Reorganize Read the Docs into Getting started, Technical deep dive, Reference and
  Project. Add a technical report explaining the credit-saving objective, independent
  LLM priors, Bayesian capability updates, predictive memory weights, risk envelopes,
  retrospective defect attribution and field evaluation. Remove historical upgrade
  pages from the live docs while retaining links to their release-tag archives (#85).

## 3.1.0 — 2026-11-13

- Version newly aggregated capability filter blocks as `howhard-capability-v2` with
  strict geometric overpowered-tail validation while accepting stored
  `howhard-capability-v1` decision v6 receipts whose expert priors uniformly match
  either the pre-#77 linear overpowered tail or the geometric overpowered tail, keeping
  receipt loading, belief replay, `observe`, and `report` working across the #74/#77
  prior change (#78).
- Concentrate the prior's overpowered tail geometrically near the upper cut so
  conditioning on weak-cell failures does not stretch the learned band in proportion to
  inventory size (#74).
- Support optional candidate `credit_budget` constraints (`available_usd` and optional
  `session_5h_remaining_usd`, `weekly_7d_remaining_usd`) in candidate lists and named
  inventories. Require uniform pricing across the inventory whenever any candidate
  declares `credit_budget`, report `needed_credit_usd`, `credit_budget`,
  `available_credit_usd` and `credit_ok` on evaluated cells while keeping `admissible`
  for accepted-result silent-risk admissibility, filter selection strictly to schedulable
  cells (`credit_ok == true`) including under `on_cap_unmet`, refuse selection when no
  evaluated cell has sufficient credit, and label cheaper unschedulable alternatives with
  `insufficient-credit` (#75).

## 3.0.2 — 2026-11-11

- Bound the learned capability band when no inventory cell reaches
  `reliability_probability`. Treat that as an explicit case: anchor the gain-margin cut
  at the lower cut and measure it against the best attained clearing probability,
  instead of running the band to the strongest cell. Easy tasks with low lower bounds
  are no longer refused with `max_forecast_cells exceeded after re-filter` merely
  because the inventory is large. Stronger cells are classified overpowered and remain
  reachable through stronger-only widening. A band that is genuinely wider than the
  budget still gets one re-filter and then the refusal. Receipt formats are unchanged
  (#72).

## 3.0.1 — 2026-11-10

- Keep concurrent routing requests valid across unrelated observations and memory
  writes. Capture one consistent read snapshot, freeze belief/capability weights, and
  validate only surfaced memory/evidence dependencies plus bound configuration (#69).
- Return typed `StaleRequest` conflicts with changed IDs/versions or evidence digests
  and a refreshed request. Preserve dependency tracking through extra retrieval,
  filtering, forecasting and widening; retain FIFO admission and strict writer revisions
  (#69).
- Preserve the original dependency conflict if refresh or failure-audit admission times
  out; return bounded recovery diagnostics without masking the changed inputs (#69).
- Record the snapshot policy and dependency manifest in new immutable receipts while
  retaining existing v6 receipt and conservative legacy v5 request compatibility (#69).

## 3.0.0 — 2026-11-09

- Accept any nonempty distinct candidate inventory and learn a task-specific capability
  order and fit band from similarity-weighted actual execution outcomes. Share the
  observation store and tempered expert-score replay with calibrated forecasts; expose
  prior-only placements, threshold evidence and filter-quality diagnostics (#66).
- Add filter → forecast requests, a default 12-cell fine-forecast budget with one explicit
  re-filter, and bounded stronger-only widening before cap-unmet fallback. Never truncate
  inventories or fabricate forecasts/outcomes for excluded cells (#66).
- Minimize expected cost to an accepted result with same-cell retries after loud failure.
  Apply epsilon to accepted-result silent defect risk, calibrate bounded sampling from
  forecast spread and predictive evidence, and restrict exploration by mean cost and
  supported uncertainty. Cap-unmet selection is deterministic and requires review (#67).
- Break the calling-agent/provider contract: decision v6 receipts include both readers
  and frozen accepted-result selection, agent requests use v5, and semantic state uses v4.
  Old receipts cannot supply new receipt-bound feedback or predictive scoring; raw
  historical evidence remains intact. Update adapters and skills; reprepare pending
  requests. C_loud and epsilon have new retry/accepted-result meanings. See the 3.0
  [archived upgrade guide](https://github.com/xu-hao/howhard/blob/v3.0.0/docs/upgrading-3.0.md)
  (#66, #67).

## 2.2.0 — 2026-11-08

- Coordinate concurrent writes inside howhard with per-store FIFO admission, finite
  waits, cancellation, automatic crash release and `writes --json` queue inspection.
  CLI `--write-timeout` and Python cancellation/error types expose contention without
  requiring agent-channel writer windows or a separate coordinator.
- Admit requests, receipts, evidence, usage amendments, memory and cache publication
  through the same store queue. Keep model calls and embedding outside write turns,
  let readers bypass waiting writers, and preserve independent store concurrency.
- Recheck evidence, memory revisions and learning events after admission. Preserve
  append-only evidence and matching stable-ID retries; refuse stale forecasts and
  conflicting patches. Unrelated historical model admission leaves disjoint inventories
  usable. Document concurrency boundaries and update native skill guidance.

## 2.1.0 — 2026-11-08

- Accept explicitly named `[inventories]` in `howhard.toml`, selected through Python
  `inventory="NAME"` or CLI `--inventory NAME`. Saved lists never override the native
  agent's supplied inventory automatically. Document external-router integration,
  distinct provider route identities and shared project learning across providers
  while retaining observed model/control attribution.

- Update the quickstart to install the published PyPI backend and explain the current
  plugin, real-task, feedback and memory workflow without release-history setup notes.

## 2.0.0 — 2026-11-07

This source version prepares the 2.0.0 release. Until the corresponding GitHub release
is published, it remains a release candidate. A GitHub tag, release and attached
distributions are separate from PyPI publication; this entry asserts neither.

- Add portable, bounded, offline usage collection for explicitly owned Codex exec/session
  and Claude SDK/session artifacts. Append usage to an existing execution through
  idempotent measurement amendments; preserve the original outcome on collection failure.
- Normalize nonoverlapping token categories, retain unknown coverage and provenance,
  refuse duplicate source claims including exact native parent-session/agent mappings, and keep client cost estimates separate from billed
  charges and historical costs with an unspecified basis. Keep task outcome, token
  coverage, bounded client estimates and authoritative billing independently recorded.
- Accept explicit category rate cards and prospective token-category forecasts. Omit
  worker money uniformly for unknown or mixed price bases; never infer host charges.
- Require decision v5, agent request v4 and semantic state v3. Older receipts cannot
  be replayed, while historical raw observations remain retained. Synchronize backend
  and native plugin versions at 2.0.0; see the [archived upgrade guide](https://github.com/xu-hao/howhard/blob/v2.0.0/docs/upgrading-2.0.md).
- Extend the host skill with owned artifact capture, rate-card provenance, honest
  partial usage, attachment retries and explicit child-session binding limits.

## 1.0.0 — 2026-10-31

### Compatibility

This is a breaking release of the LLM-only routing workflow, not a compatible patch
to the 0.6 maintenance series. See the
[upgrade guide](https://github.com/xu-hao/howhard/blob/v1.0.0/docs/upgrading-1.0.md).

- Remove statistical routing, its CLI/API/configuration surfaces and generic prior
  transfer. Native routing requires explicit host model/effort candidates and an
  independent LLM forecast; unavailable inference stops selection.
- Require `howhard-decision-v4`, `howhard-agent-request-v3` and
  `howhard-semantic-v2`. Older requests and receipts cannot be reused; no automatic
  state migration or deletion occurs. Preserve historical evidence and reprepare
  current requests; unsupported persisted inputs are refused explicitly.
- Local retrieval adds FastEmbed, ONNX Runtime and USearch dependencies and lazily
  downloads pinned models for larger searches. Qwen remains an optional extra.
- Synchronize Python and all native plugin manifests at 1.0.0. Upgrade the backend
  and installed skills separately; old GitHub releases remain available.

### Changes

- Add the portable `attribute-failure` skill for bug-fixing agents to investigate
  escaped defects through reproductions, Git history/blame and optional bisection,
  connect supported findings to original executions, and amend outcomes with explicit
  memory learning. Retain commit/PR provenance in existing artifacts and observation
  notes; uncertain attribution remains unscored. Native plugins discover both skills.

- Fix a native-runtime import-order crash in hybrid retrieval by initializing ONNX
  Runtime before USearch at the lazy dense-search boundary. Fresh and cached HNSW
  searches work without caller pre-imports; small corpora still load neither runtime.

- Add optional fixed `[retrieval]` settings for the fast ONNX or Qwen reranker,
  pinned model/artifact overrides and runtime controls. Apply the policy to routing,
  memory and evidence search; reject invalid combinations before creating state.
  Agent retrieval tools cannot change the tier, model or configuration, and no
  automatic upgrade/fallback is introduced. Policy changes invalidate pending requests.

- Use a pinned MiniLM-L4 INT8 ONNX cross-encoder as the default CPU reranker. A
  two-thread check of 200 pairs took 0.32–0.41 seconds for short inputs and 8.34–8.55
  seconds for full 512-token windows. Keep the 200-to-eight pipeline and calling-agent
  forecasts; retain instruction-aware Qwen through the optional `qwen` install extra.
  The compact default has a 512-token combined window and no instruction tuning.

- Retrieve transferable execution experience with local BM25 and HNSW, RRF fusion
  to 200 candidates and pretrained instruction-aware Qwen3-Reranker-0.6B selection
  of up to eight precedents. Keep observed model/effort/outcome/cost attached, include
  relevant failures and broader/narrower tasks, and retain the calling agent for final
  forecasts. Make every retrieval/reranking stage independently replaceable in Python.
- Expose observed runs as read-only memory experts with prospective predictive credit.
  Normalize decision support over participating experts so unrelated history cannot
  dilute retrieved experience. Rebuild derived indexes on edits/amendments without
  changing canonical evidence. Larger searches lazily download local pretrained models.

- Add an opt-in pytest-probability diagnostic benchmark comparing whole-rule versus
  applicability-condition embeddings, with and without cross-encoder reranking.
  Forty authored tasks cover paraphrases, misleading action words, exceptions,
  missing prerequisites and conjunctions. Retain labels separately, report candidate
  recall/ranking metrics and latency, and keep real-model runs outside the ordinary gate.

- Remove statistical-only routing, the internal statistical expert/risk floor, lexical
  features, outcome/resource fitting, posterior caches and generic prior transfer.
  Routing now requires independent LLM forecasts and semantic memory; missing or failed
  inference stops explicitly. Remove `--llm`, `HOWHARD_LLM`, `--cache`, prior commands,
  statistical Python APIs and configuration. State/protocol commands require no provider.
- Use LLM-reference predictive memory weights and a labeled weighted forecast envelope
  for risk checks. New receipts/requests use v4/v3 and semantic state v2; old artifacts
  remain untouched and old forecasts are unscored. Fresh-engine observations leave
  writer work pending for explicit provider/calling-agent learning.

- Clarify portable skill feedback attribution when native host APIs accept explicit
  model/effort overrides: completed execution can match the requested decision receipt
  without provider attestation; known mismatches use raw feedback and unknown cells
  remain unscored.
- Allow append-only measured-usage corrections with a validated amendment-only
  `measurement_corrections` map and CLI JSON input. Explicit replacements or null clears
  update effective regressions, worker costs and learning once while preserving raw history.
- Document a copyable coordinating-agent instruction requiring howhard before authorized
  worker delegation, with actual feedback, an independent-baseline recursion exception
  and an explicit distinction between agent instructions and host enforcement.
- Remove registry-only routing and the Python family/effort candidate filter. Every
  routing request now requires the host's native inventory; prices come only from that
  inventory. Remove packaged API prices, token mixes, family activation flags and
  unresolved default-effort handling.
- Remove checkout-local Git store discovery and old SQLite adoption inspection. Git
  worktrees use the common Git directory's store; explicit configuration/store selection
  and plain-folder state remain supported. Existing checkout data is left untouched.
- Require timestamped contextual original records, identity/inventory/cost-bound v3
  decision receipts and worktree-bound v2 agent requests. Only contextual v2 priors are
  imported/exported, including empty-store exports. Superseded records, receipts,
  requests and priors are refused rather than upgraded or read through compatibility paths.
- Explain automatic task-based Codex skill activation and optional explicit invocation.
  The coordinating agent selects a worker before spawning; subagent startup does not
  itself guarantee skill activation. Remove retired installation guidance.

## 0.6.1 — 2026-10-30

- Allow append-only measured-usage corrections with a validated amendment-only
  `measurement_corrections` map and `observe --measurement-corrections FILE`. Explicit
  replacements or null clears update effective regressions, worker costs and learning
  once while preserving the raw original, context and receipt identity.
- Preserve existing 0.6 persistence/API behavior. This additive row-semantics expansion
  leaves existing data readable; older readers refuse rows containing the new field.
- Link the live Read the Docs site from the README and release guide.

## 0.6.0 — 2026-10-30

- Accept actual native model/control inventories before forecasts, with distinct neutral
  families for new identities, persistent historical metadata and exact categorical efforts.
  The portable skill calls native CLI/API primitives without project bridges or tuning.
- Label unknown worker pricing explicitly and omit worker money uniformly when any native
  candidate is unpriced; preserve known costs and null unknowns without API-price assumptions.
- Separate current project capture from durable state, share fresh Git worktree state,
  retain valid legacy/explicit stores and refuse cross-project store reuse. Expose paths
  and intentional state selection; preserve identity on project moves, exclude resolved
  state from source capture/retrieval, and bind prepared requests to their originating worktree.
- Add direct evidence lookup, matching stable-ID feedback retries and validated raw native
  execution admission without retrospective forecasts. Include all correction notes in
  writer context while retaining amendment scoring and learning-event idempotence.
- Existing logs, receipts and prior formats remain readable. Native metadata/efforts and
  new receipts with project identity/native cost-policy fields require 0.6.0 or later;
  old unbound pending requests must be prepared again. No old state is migrated
  or deleted automatically, and generic prior exports exclude native identity-specific terms.

## 0.5.0 — 2026-10-30

- Move the canonical skill to standard `skills/howhard` discovery, with thin portable,
  Codex marketplace, Claude Code and Antigravity manifests for first-party installation.
- Retire the bespoke 0.4 `howhard skills install` command and `howhard.skill_install`
  API before 1.0. The source distribution retains the skill; the Python wheel supplies
  the backend without another skill copy. Preserve edited old copies before migration.
- Complete the existing Quickstart with source backend installation, native plugin
  setup/invocation, backend executable visibility, actual feedback and memory learning.
- Add explicit native validation and an optional third-party `skills` CLI smoke;
  keep native installations and npm downloads out of the ordinary Python gate.

## 0.4.0 — 2026-10-30

- Package one reusable howhard skill with progressive references for actual task/context
  reasoning, host-compatible delegation, outcome corrections and writable memory.
- Add offline `skills install` for Codex, Claude Code and Antigravity, with explicit
  project/user scopes, deduplicated discovery paths, identical-install reuse and
  preflight conflict refusal. Installation leaves model/provider settings unchanged.
- Document source and pinned Git installation, host capability gaps and optional
  project adapters; verify skill installation from a clean wheel.

## 0.3.1 — 2026-10-30

- Allow append-only outcome amendments to clean, loud or silent, retaining the original
  run's context and measurements and replacing Bayesian and semantic evidence once.
  Row keys and cache/prior formats are unchanged; earlier versions cannot read clean
  or loud amendments.
- Give every correction its own idempotent memory-learning event, including corrections
  back to an earlier result, and reject writer responses prepared before a correction.

## 0.3.0 — 2026-10-30

- Add LLM reasoning over original task/project text, persistent semantic memory and
  bounded evidence retrieval, with an isolated stateless forecast baseline.
- Support add, edit, remove, restore, merge and split memory operations through the
  CLI, Python API and LLM writer, with transactional edits and preserved history.
- Update approximate predictive weights from actual selected-action forecasts and
  effective outcomes, including resource evidence, amendments and fresh edit versions.
- Reuse an authenticated Claude CLI with tools disabled, accept custom adapters and
  expose a calling-agent protocol. Explicit LLM errors remain explicit; automatic
  statistical fallback is labeled.
- Preserve semantic forecasts and memory versions in receipts, audit inference and
  writer usage, and report observed-action forecast metrics and unknown costs honestly.
- Add a real-project evaluation protocol. Keep correctness fixtures separate from
  routing-quality evidence; defer Read the Docs publishing.

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
