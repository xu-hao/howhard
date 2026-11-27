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
It is the posterior probability that the task's threshold lies at or below the cell:
the running sum of the posterior masses up to that cell, held within [0, 1] so that
rounding cannot carry a saturated sum past 1. `P_timely`, below, is held the same way.
A receipt with a probability above 1 is refused.
The clearing interval starts at the first cell whose `P_clear` reaches the lower edge
`e`. Its upper cut is the first cell at or above its anchor whose gain to the strongest
cell is below the upper margin `g`. Both edges follow the share of its failures that
the asking host catches ({ref}`the host's caught share <host-aware-band>`); with no
failed run on record they are `min_clear_probability` and `overpowered_gain_margin`.
The strongest cell attains the best clearing probability because the probability is
monotone along the order.

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
reader to compare time. The timely interval uses the configured
`min_clear_probability`, `reliability_probability` and `overpowered_gain_margin`,
whatever the host's caught share. When no candidate reaches the timely minimum, the
interval contains only the first near-best timely option among the cells from the
clearing interval's lower cut onward.

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

(host-aware-band)=
### The host's caught share and the two edges

The host is the agent that asks for the route and accepts or rejects the result. A
failure it finds before acceptance is `loud`, and one found afterwards is `silent`.
The caught share is the share of its failures that a host found before acceptance:

```text
d_0     = (1 - 2 * epsilon) / (1 - epsilon)
d(all)  = (C_all  + kappa_d * d_0)    / (F_all  + kappa_d)
d(host) = (C_host + kappa_d * d(all)) / (F_host + kappa_d)
```

- `epsilon`: the accepted-defect limit of the task's risk class.
- failed run `i`: a finished `loud` or `silent` run whose decision receipt holds a
  forecast for the cell that ran. Every such run on record counts, whatever its task.
- `c_i`: 1 for a `loud` run (found before acceptance) and 0 for a `silent` one.
- `a_i`: the run's age factor `0.5^(days_i / half_life_days)`, with `days_i` counted
  from the run's `observed_at` to the time the request was prepared, divided by
  `max(1, A_t)`, where `A_t` is the total age factor of the failed runs of the same
  ticket. One ticket's failed runs therefore count as one run at most, and a run with
  no ticket is not limited.
- `F` and `C`: the sum of `a_i` and the sum of `a_i * c_i` over the failed runs of all
  hosts (`F_all`, `C_all`) or of the asking host (`F_host`, `C_host`).
- host of a run: the name in the run's decision receipt. A receipt with no name, or
  older than decision v11, belongs to the one unnamed host.
- `kappa_d`: `caught_share_worth`, 12 failed runs' worth.
- `d_0`: the caught share at which both edges below equal the configured ones.
- `d`: `d(host)` of the asking host.

In plain words: `d` is the share of its failures that the host caught. A host with few
failures of its own leans on the record of all hosts, and an empty record gives `d_0`.

For example, at `epsilon = 0.10`, `d_0 = 0.8 / 0.9 = 0.889`. With `F_all = 60` and
`C_all = 42`, `d(all) = (42 + 12 * 0.889) / 72 = 0.731`. With `F_host = 20` and
`C_host = 8`, `d = (8 + 12 * 0.731) / 32 = 0.524`.

```text
lower edge:  P_clear(o) >= e
             e = max(min_clear_probability, 1 - epsilon / (1 - d + epsilon * d))
upper edge:  the first reliable cell o with  P_clear(strongest) - P_clear(o) < g
             g = min(overpowered_gain_margin,
                     overpowered_gain_margin * epsilon / ((1 - epsilon) * (1 - d)))
```

- `o`: a cell in the task's order, weakest first, and `P_clear(o)` its chance of being
  strong enough for the task.
- `e`: the lowest `P_clear` of a band cell. Its floor is `min_clear_probability` (0.5).
- `g`: the upper margin. It is at most `overpowered_gain_margin` (0.05).
- reliable: `P_clear >= reliability_probability`. strongest: the last cell of the order.
- At `d >= d_0`, `e` is `min_clear_probability` and `g` is `overpowered_gain_margin`.
  Below `d_0` both tighten.

In plain words: a cell too weak for the task fails. With `q = 1 - P_clear`, a share
`q * (1 - d)` of its results escapes the host and a share `1 - q * d` is accepted;
holding the escaped share of the accepted results at `epsilon` gives `e`. At the top,
0.05 more `P_clear` removes about `(1 - d) * 0.05` of the silent results, so `g`
shrinks to keep `(1 - d) * g` constant.

| `d` | `e` | `g` |
| --- | --- | --- |
| 0.90 or more | 0.500 | 0.0500 |
| 0.80 | 0.643 | 0.0278 |
| 0.50 | 0.818 | 0.0111 |
| 0.00 | 0.900 | 0.0056 |

The table is for `epsilon = 0.10`. For example, `d = 0.524` gives `e = 0.811` and
`g = 0.0117`. With `P_clear` of 0.30, 0.55, 0.72, 0.84, 0.91, 0.955, 0.985 and 0.99
along the order, the configured edges give cells 2 to 6: 0.55 is the first value at
0.5 or more, 0.91 is the first reliable cell, and 0.955 is the first from there within
0.05 of 0.99. The edges at `d = 0.524` give cells 4 to 7: 0.84 is the first value at
0.811 or more, and 0.985 is the first reliable cell within 0.0117 of 0.99.

If no cell reaches `e`, the band starts at the first cell with the highest `P_clear`,
provided that value reaches `min_clear_probability`. Otherwise the request is refused.
`e` and `g` move the clearing interval only. The timely interval, the prior CDF below
and the filter synthesized from prototype priors keep the configured values.

## Similarity-weighted censored observations

The required capability is a discrete latent threshold with a possible value beyond
the strongest supplied cell. Each expert's cut points define a smoothed prior CDF.
At the position just before the lower cut the CDF is `0.8 * min_clear_probability`.
It halves with every further step toward weaker cells, so a long prefix does not
spread prior mass across distant weak cells. The upper fit position reaches
`reliability_probability`. The overpowered tail's gain stays below
`overpowered_gain_margin` and is concentrated geometrically near the upper cut.
With no execution evidence or memory adjustment, the result retains the independent
prior's order, and its cuts exactly when the host's caught share is at or above `d_0`.
Below `d_0` the band is computed from the prior's probabilities like any other band.

The existing tempered log-score weights mix the independent and versioned-memory
prior CDFs. Consensus ranks preserve each model's effort chain. A pooled adjacent
violators projection makes the mixed CDF monotone. The threshold reader then applies
the execution likelihood once to this mixed prior; it does not compete as a second
independent learner.

For a clean run at position c, the noisy likelihood is `1-noise` for thresholds at or
below c and `noise` above it. Loud/silent outcomes reverse that likelihood. Each
likelihood is raised to a fractional evidence weight:

```text
w_i = m_i * sim_i * strength_i * a_i
W_t = sum of w_i over this reader's runs of ticket t
w_i <- w_i / max(1, W_t)
```

- `i`: a run this reader counts.
- `m_i`: the run's weight on record, the `weight` configured for its model (1 unless set).
- `sim_i`: `1 / (1 + exp(-r_i))`, where `r_i` is the reranker's score for the new task's
  text against the run's own task text.
- `strength_i`: the factor for the run's outcome and its cause, defined below.
- `a_i`: the age factor `0.5^(days_i / half_life_days)`, half as much for every 14 days
  between the run and the request.
- `t`, `W_t`: a ticket and the total weight of its runs in this reader. A run with no
  ticket is not limited.

The runs are the decision's {ref}`evidence candidates <evidence-candidates>` whose
executed cell is in the inventory: recorded runs that belong to a prototype the new
task matches. A task that matches no prototype gives this reader no observation, so
unrelated tasks receive no weight. The reader has no relatedness discount: every
counted run informs the shared threshold from the position of the cell that ran it.
All runs of one ticket together count for at most one run, because retries follow
failures. Similarities are heuristic relevance strengths, not calibrated probabilities
of task equivalence. A lone weak match is not normalized to unit weight, and a run with
a negative score still counts, at a small weight.

For example, a loud run with no recorded cause, `sim = 0.8` and `a = 0.7071` weighs
`0.8 * 0.75 * 0.5 * 0.7071 = 0.2121`. The receipt stores each observation's `strength`,
`ticket_share` and `weight`; the run's score, similarity and age factor are stored once
with the scored candidate.

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

`context` matches the incoming task against active synthesized `(task, context)`
prototypes in `payloads.sqlite3` and returns `stage=filter` with the full validated
inventory, the memory reader `request`, and either a clean `baseline_request` (when at
least one inventory cell lacks prototype Stage-1 coverage, conditioned on the
prototype's synthesized representative task and context) or `baseline_request: null`
(when every inventory cell already has a materialized `P_clear` prior or
`capability_weight >= 0.5` on matched prototypes, pre-populating `no_memory_filter`).
When `baseline_request` is non-null, the independent response is:

```json
{"filter": {"order": ["cell:key-a", "cell:key-b"], "lower": "cell:key-a", "upper": "cell:key-b"}, "reason": "Task-specific prior"}
```

The actual keys come from the request. On a partial prototype hit where new unseen
models join already-covered models, previously covered cells preserve their materialized
prior `P_clear` while unseen cells take `prior_cdf(filter)`, projected monotone along
the merged order. When `baseline_request` is `null`, omit `filter` (or pass the unchanged
`no_memory_filter`) and submit `{memory_filters, reason}`; passing `--auto-decide` to
`context` advances warm stages automatically when no active editable semantic memories
exist. Before submitting to `decide`, add
`memory_filters: []` or up to eight `{id, version, filter}` adjustments grounded in
retrieved active memories. Each adjustment supplies a complete order and cut points;
it can move a boundary or move a particular cell. Missing, duplicate or unknown keys,
nonordinal known efforts, reversed cuts and empty bands are refused.

`decide` applies the learned filter and returns a new `stage=forecast` request ID.
A forecast cell is covered at Stage 2 when a matched prototype stores a usable forecast
for it made under the request's host; a forecast made under another host, and recorded
runs with no stored forecast, do not cover a cell. The Stage 1 priors above are shared
by all hosts. When every forecast cell is covered, Stage 2 sets
`baseline_request: null` and pre-populates `no_memory_forecasts` with the
{ref}`stored forecasts merged on each value's own scale <stored-forecasts>`; when only
some cells are covered, `baseline_request` contains only the other cells and their new
forecasts join the reused rows. A full panel returned in that case must repeat each
reused row unchanged. Otherwise evaluate its clean
`baseline_request` in another independent context and supply the normal
`{forecasts, memory_forecasts, reason}` response. Only the forecast cells receive
rows: the core of the band, the reserved cells and the band cells that took unfilled
slots ({ref}`reserved forecast slots <reserved-slots>`). Each new forecast is stored on
the primary prototype under the request's host, so one host asks once per cell.
Neither independent call receives memory, execution evidence or learned tables.
`pick` and `summary` drive these same stages, completing locally with zero LLM calls on
warm prototypes fully covered under the request's host when no active editable memories
exist.

The inventory has no count limit. `max_forecast_cells` is a budget for fine forecasts,
not an inventory cap. If the learned band exceeds it, `decide` returns one re-filter
request with the full inventory. A second oversized band errors and asks for a tighter
filter. Both threshold intervals remain intact when the gap is omitted. If the
posterior puts every cell below `min_clear_probability`, it refuses routing and asks
for a more capable inventory.

(reserved-slots)=
### Reserved forecast slots

The band alone can leave out a family nobody has tried on this kind of task, or every
cell of one agent. Two of the forecast budget's slots are reserved for them:

```text
core     = the first (max_forecast_cells - reserved_forecast_cells) cells of the band
forecast = core + up to reserved_forecast_cells cells o_f with P_clear(o_f) >= b_f
```

- band: the fit band, weakest first.
- `reserved_forecast_cells`: 2. Setting it to 0 turns the rule off.
- `f`: a family with no core cell. A band cell beyond the core counts as outside it.
- `o_f`: the cell of `f` nearest to the core in the order, the stronger one on a tie.
- `b_f`: the bar `o_f` must reach. It is `unproven_clear_probability` (0.2) when no
  run of `f` is among the decision's {ref}`evidence candidates <evidence-candidates>`,
  and the lower edge `e` otherwise.

In plain words: a slot buys one forecast for a family or an agent that the band leaves
out. Slots go first to unproven families, those with no candidate run. Each unproven
family competes on its own, the one nearest to the core first. The remaining slots go
to missing agents, those with no core cell. A proven family that qualifies only
because its agent is missing takes a slot on behalf of that agent, so a missing agent
gets at most one slot: it goes to the agent's qualifying cell nearest to the core, the
stronger one on a tie, and the next slot goes to the next missing agent or back to the
band. An agent whose unproven family already took a slot counts as served. A proven
family whose agent has a core cell takes no slot. An agent is the `agent` label of the
family's nearest cell, and cells with no label share the one unnamed agent.

For example, with the default budget of 12 a band of 11 cells has a core of ten.
Family X has no candidate run and its nearest cell has `P_clear = 0.35`, above 0.2: it
takes the first slot. Agent Q has no core cell and the band's eleventh cell is its
nearest: that cell takes the second slot. With neither X nor Q, the eleventh cell is
forecast as an ordinary band cell. With no unproven family, two proven families of
agent Q above the band and one proven family of another missing agent R beyond them,
Q's nearer cell takes the first slot and R's cell the second; Q's other family takes
none.

A slot that no family takes goes back to the band: the band's next cell beyond the
core is forecast. With no family to reserve for, the forecast cells are the band's
first `max_forecast_cells` cells. A reserved cell keeps the classification of its
position, so a reserved cell below the band is still `underpowered`. No decision
forecasts more than `max_forecast_cells` cells.

If every forecast cell is excluded by the {ref}`admission tiers <admission-tiers>`,
request the next stronger cell with no forecast, one per widening round: first the band
cells beyond the core, then the overpowered cells, each in capability order. A reserved
cell already has a forecast and is skipped. Each requires an actual new prospective
forecast before entering selection. Underpowered cells never enter widening. The
default is two rounds (`widening_rounds`), within the same total forecast budget. The
test reads tiers only; credit does not enter it, so one trusted or on-trial forecast
cell stops widening even when it lacks credit. Each round records the trigger
`all-excluded`. `widening_stop` records why widening ended: `pickable` when a forecast
cell is trusted or on trial, otherwise `forecast-budget`, `round-limit` or `exhausted`.
No probability is invented for an unforecast cell. The final
{ref}`selection policy <accepted-result-selection>` includes retry cost and still draws
a cell when every forecast cell stays excluded.

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
`filter_quality.bands` lists, for each decision whose receipt stores them, the
`caught_share` block (the host's name, `F` and `C` for the host and for all hosts,
`d_0`, `d_all`, `d_host`, `e` and `g`), the `core`, the `proven_families` and the
`reserved` cells with their reason and bar. Older receipts have no entry.

Configuration defaults and validation are in [configuration](configuration.md).
Decision v11 receipts freeze the full filter, expert priors, task-specific order,
clearing and timely-clearing probabilities, band, widening, disagreements, fine
forecasts, and accepted-result selection. Newly aggregated capability blocks use
`howhard-capability-v6`, supporting materialized prototype priors, mixed seen/unseen
`prior_sources` PAV projection, geometric prior mass outside both cuts, observations
drawn from the receipt's scored evidence candidates with weights that are recomputed on
load, and a band checked against the frozen probabilities. It also stores
`caught_share` (the asking host's name, `F_all`, `C_all`, `F_host`, `C_host`, `d_0`,
`d_all`, `d_host`, `e` and `g`), the `core`, the `proven_families` (the families with
a candidate run) and each `reserved` cell with its `reason` (`unproven-family` or
`missing-agent`) and `bar`. On load the caught shares and both edges are recomputed
from the recorded `F` and `C`, which are taken as recorded, and the band, the core,
the reserved cells and the widening order are recomputed from them; forecasts must
cover exactly the core, the reserved cells, the band cells that took unfilled slots
and the widening cells. A receipt whose band, core or reserved cells do not follow
from its stored numbers is refused. Each of its cells
also stores the candidate's {ref}`agent label <host-and-agent-names>` (`null` when
unnamed), which must match the receipt's inventory and does not affect the order, the
probabilities or the band; it decides only which agents the reserved slots count as
missing. Persisted decision v10 receipts use `howhard-capability-v5`, a block without
agent labels, caught share, core or reserved cells: its band uses the configured edges
and its forecasts cover exactly the band and the widening cells. Persisted decision
v10 and v9 receipts also accept `howhard-capability-v4`, while persisted decision v9 and
v8 receipts with capability v3 retain the linear prior prefix, geometric tail, and
contiguous band. Persisted decision v7 receipts support capability v2 and v1. Capability v1
priors must uniformly match the linear tail or the geometric tail across all experts.
These versions retain support for loading, feedback, belief replay, and reporting.
Older decision formats v1–v6 remain inspectable as raw
artifacts but cannot supply new receipt-bound feedback or predictive scores; old
observation rows remain evidence.
