# Task-specific capability filtering

howhard reads one execution evidence store in two ways. The coarse reader learns a
capability-fit band over the full inventory. The fine reader forecasts clean/loud/silent
outcomes, tokens and time only for that band. One `observe` row supplies both readers;
there is no separate fitness ledger or feedback command.

## An order for this task

Each decision starts with an independent LLM prior: a complete order of opaque cell
keys, weakest to strongest **for this task and project**, plus inclusive lower and
upper fit cut points. A model may rank differently for debugging and writing. No
universal model leaderboard is fitted or reused. Within a model, known efforts follow
`low < medium < high < xhigh < max < ultra`; custom controls retain the prior's
within-model order. Cross-model placements can change with related execution evidence.

Fitness is a band: underpowered, fit, then overpowered. Threshold-clearing probability
is monotone along this decision's order; fitness is not a score that improves forever
with capability. A stronger cell is overpowered when a weaker fit cell already gives
nearly all the clearing probability this inventory offers: the additional clearing
probability is below the gain margin. The lower cut is the first cell reaching
`min_clear_probability`. The upper cut is the first cell at or above its anchor whose
gain to the strongest cell is below the margin, so the entire excluded suffix
qualifies. The strongest cell attains the best clearing probability because the
probability is monotone along the order.

The anchor is the first cell reaching `reliability_probability`, so the band always
includes a reliable cell when one exists. If no cell reaches that target, the target
is unattainable within this inventory and cannot anchor the band. "No reliable cell" is
then an explicit case: the anchor is the lower cut, and the gain cut alone, measured
against the best attained probability, ends the band. The upper cut never falls back
to the strongest cell merely because the target is unmet. Stronger cells remain
reachable through stronger-only widening. Price, retry cost, and accepted-result silent
risk belong to stage 2.

The order assumes a useful scalar approximation to the demands of this task. Related
tasks and noisy outcomes do not establish a universal ranking, causal explanation,
or guarantee that a stronger cell succeeds. Receipts expose the prior, learned order,
observations, and assumptions so these approximations can be assessed.

## Similarity-weighted censored observations

The required capability is a discrete latent threshold with a possible value beyond
the strongest supplied cell. Each expert's cut points define a smoothed prior CDF:
underpowered positions lie below `min_clear_probability`, the upper fit position
reaches `reliability_probability`, and the overpowered tail's possible gain stays
below `overpowered_gain_margin`. With no execution evidence or memory adjustment,
the result retains the independent prior's order and cuts exactly.

The existing tempered log-score weights mix the independent and versioned-memory
prior CDFs. Consensus ranks preserve each model's effort chain. A pooled adjacent
violators projection makes the mixed CDF monotone. The threshold reader then applies
the execution likelihood once to this mixed prior; it does not compete as a second
independent learner.

For a clean run at position c, the noisy likelihood is `1-noise` for thresholds at or
below c and `noise` above it. Loud/silent outcomes reverse that likelihood. Each
likelihood is raised to a fractional evidence weight:

```text
row weight × sigmoid(relevance score) × severity/cause strength × recency discount
```

Use the existing task/project retrieval to select at most eight precedents. The
configured cross-encoder supplies the absolute relevance score. Small retrieval
corpora normally skip reranking for display; the threshold reader scores those
selected precedents too. Sigmoid scores are heuristic relevance strengths, not
calibrated probabilities of task equivalence. A lone weak match is not normalized to
unit weight. The receipt freezes the raw score, similarity and effective weight.

Clean outcomes have strength 1. Loud and silent strengths default to 0.75 and 1.
An explicit `cause=capability` retains that strength; depth/unrecorded failures get
half strength. For those otherwise unattributed failures, an explicit
`capability_caused=false` assertion in the note gives zero threshold evidence. Ordinary
prose is not interpreted as a causal label. Recorded cause takes precedence over a
contradictory note. Outcomes always remain clean/loud/silent in the single store.

Cross-model order updates use the same marginal threshold likelihood with an initial
normalized order-distance penalty. Deterministic improving cross-model swaps preserve effort
chains. This is approximate task-local posterior/MAP search, not exact inference over
all permutations. Observed placements can swap across unseen cells while preserving
each model's effort chain. Only actual executions contribute likelihoods. If a precedent's
executed cell is absent, the reader does not guess an equivalent identity/control.

New cells have `placement=prior-only`: their initial position comes from the task's
LLM prior. They can still receive `threshold=learned` because observations at other
positions inform the shared task threshold. A receipt distinguishes these sources.
This transfer is conditional on the proposed placement; it is not an observed outcome
for the new cell.

## Protocol and forecast budget

`context` returns `stage=filter`, the full validated inventory, a clean
`baseline_request`, and a separate memory reader request. The independent response is:

```json
{"filter": {"order": ["cell:key-a", "cell:key-b"], "lower": "cell:key-a", "upper": "cell:key-b"}, "reason": "Task-specific prior"}
```

The actual keys come from the request. Before submitting to `decide`, add
`memory_filters: []` or up to eight `{id, version, filter}` adjustments grounded in
retrieved active memories. Each adjustment supplies a complete order and cut points;
it can move a boundary or move a particular cell. Missing, duplicate or unknown keys,
nonordinal known efforts, reversed cuts and empty bands are refused.

`decide` applies the learned filter and returns a new `stage=forecast` request ID.
Evaluate its clean `baseline_request` in another independent context and supply the
normal `{forecasts, memory_forecasts, reason}` response. Only fit cells receive rows.
Neither independent call receives memory, execution evidence or learned tables.
`pick` and `summary` drive these same stages through a stateless provider adapter.

The inventory has no count limit. `max_forecast_cells` is a budget for fine forecasts,
not an inventory cap. If the learned band exceeds it, `decide` returns one re-filter
request with the full inventory. A second oversized band errors and asks for a tighter
filter. Because neither cut falls back to the strongest cell, an oversized band means
the posterior spreads material clearing-probability gains over more cells than the
budget; a sharper prior can narrow it. The backend never slices a band to fit the
budget, which would exclude cells that still add material clearing probability. If the
posterior puts every cell below the minimum clearing probability, it refuses routing
and asks for a more capable inventory; it does not relabel a known underpowered
strongest cell as fit.

If no forecast cell passes the accepted-result silent cap, request the next stronger
overpowered cell, one per widening round, in capability order. Each requires an actual
new prospective forecast before entering selection. Underpowered cells never enter
widening. The default is two rounds, within the same total forecast budget. Only after
widening ends does `on_cap_unmet` apply. No probability is invented for an unforecast
cell. The final {ref}`selection policy <accepted-result-selection>` includes retry cost and
never explores when the cap is unmet.

## Shared predictive scores and quality reports

Stage-1 experts use `filter:llm` and `filter:memory:ID:VERSION` in the same belief replay
as stage-2 experts. Both use the packaged probability floor, prior masses, temperature,
recency, evidence counts and support normalization. A stage-1 prior earns a binary
clean/non-clean log probability ratio against the independent fine forecast, coarsened
to the same binary outcome. Stage 2 keeps its ternary outcome and measured-usage log
scores. Normalize over participating experts separately for each reader. New or revised
memories have no historical predictive credit; unchanged overrides earn no exposure.
Amendments replace the effective outcome and score the original exposure once.

For every forecasted cell, compare the threshold posterior's `P_clear` with stage 2's
weighted `P_clean`. A difference above `consistency_margin` (default 0.2) is recorded,
without changing stage-2 selection. After the actual outcome, compare the frozen binary
log losses and report which reader was closer, including ties.

`howhard report --json` includes `filter_quality`: observed fit-cell loud/silent rates,
widening frequency, later clean executions of previously underpowered cells on retrieved
similar tasks, pending disagreements and resolved reader comparisons. Execution entries
retain the selected cell's original classification and position. Excluded cells that
never ran contribute no outcome. Reported rates are descriptive diagnostics.

Configuration defaults and validation are in [configuration](configuration.md).
Decision v6 receipts freeze the full filter, expert priors, task-specific order,
probabilities, band, widening, disagreement data, fine forecasts, and accepted-result
selection. Older receipts remain inspectable as raw artifacts but cannot supply new
receipt-bound feedback or predictive scores; old observation rows remain evidence.
