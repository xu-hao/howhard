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

Threshold-clearing probability `P_clear` is monotone along this decision's order.
The clearing interval starts at the first cell reaching `min_clear_probability`.
Its upper cut is the first cell at or above its anchor whose gain to the strongest
cell is below `overpowered_gain_margin`. The strongest cell attains the best clearing
probability because the probability is monotone along the order.

The anchor is the first cell reaching `reliability_probability`, so the band always
includes a reliable cell when one exists. If no cell reaches that target, the target
is unattainable within this inventory and cannot anchor the band. "No reliable cell" is
then an explicit case: the anchor is the lower cut, and the gain cut alone, measured
against the best attained probability, ends the band. The upper cut never falls back
to the strongest cell merely because the target is unmet.

The timely-clearing posterior `P_timely` provides a second interval. Clean precedents
whose effective active duration exceeds `1.5x` their decision forecast or faster
stronger clean precedents receive reduced timely-clearing strength
(`(1.5 / overrun_ratio)^2`). This keeps faster stronger options available for the fine
reader to compare time. The timely interval uses the same minimum, reliability, and
gain rules. When no candidate reaches the timely minimum, the interval contains only
the first near-best timely option among cells that meet the clearing minimum.

The forecast band covers the full span of the two intervals when it fits the budget.
If that span exceeds the budget, forecast the union of the intervals. A gap consists
of cells past the clearing gain cut and before the timely interval. These cells keep
their positions and probabilities and carry classification `intermediate`. The band
lists the `fit` cells, with `lower` and `upper` marking its first and last entries.
`underpowered` cells precede `lower`. `overpowered` cells follow `upper` and remain
reachable through stronger-only widening. Price, retry cost, and accepted-result
silent risk belong to stage 2.

The order assumes a useful scalar approximation to the demands of this task. Related
tasks and noisy outcomes do not establish a universal ranking, causal explanation,
or guarantee that a stronger cell succeeds. Receipts expose the prior, learned order,
observations, and assumptions so these approximations can be assessed.

## Similarity-weighted censored observations

The required capability is a discrete latent threshold with a possible value beyond
the strongest supplied cell. Each expert's cut points define a smoothed prior CDF.
At the position just before the lower cut the CDF is `0.8 * min_clear_probability`.
It halves with every further step toward weaker cells, so a long prefix does not
spread prior mass across distant weak cells. The upper fit position reaches
`reliability_probability`. The overpowered tail's gain stays below
`overpowered_gain_margin` and is concentrated geometrically near the upper cut.
With no execution evidence or memory adjustment, the result retains the independent
prior's order and cuts exactly.

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
contradictory note. Operational `interrupted` outcomes (`usage_limit`, `provider_error`,
`host_error`, `cancelled`, `unrecorded`) remain in the single observation store for
accounting and audit, but are excluded from threshold likelihoods and timely-clearing
updates rather than being recorded as `loud` model failures.

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
filter. Both threshold intervals remain intact when the gap is omitted. If the
posterior puts every cell below the minimum clearing probability, it refuses routing
and asks for a more capable inventory.

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
weighted prospective `P_clean` forecast (before per-cell outcome updates). A difference above `consistency_margin` (default 0.2) is recorded,
without changing stage-2 selection. After the actual outcome, compare the frozen binary
log losses and report which reader was closer, including ties.

`howhard report --json` includes `filter_quality`: observed fit-cell loud/silent rates,
widening frequency, later clean executions of previously underpowered cells on retrieved
similar tasks, pending disagreements and resolved reader comparisons. Execution entries
retain the selected cell's original classification and position. Excluded cells that
never ran contribute no outcome. Reported rates are descriptive diagnostics.

Configuration defaults and validation are in [configuration](configuration.md).
Decision v9 receipts freeze the full filter, expert priors, task-specific order,
clearing and timely-clearing probabilities, band, widening, disagreements, fine
forecasts, and accepted-result selection. Newly aggregated capability blocks use
`howhard-capability-v4`, with geometric prior mass outside both cuts and a band
checked against the frozen probabilities. Persisted decision v9 and v8 receipts
with capability v3 retain the linear prior prefix, geometric tail, and contiguous
band. Persisted decision v7 receipts support capability v2 and v1. Capability v1
priors must uniformly match the linear tail or the geometric tail across all experts.
These versions retain support for loading, feedback, belief replay, and reporting.
Older decision formats v1–v6 remain inspectable as raw
artifacts but cannot supply new receipt-bound feedback or predictive scores; old
observation rows remain evidence.
