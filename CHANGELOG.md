# Changelog

## Unreleased

- Forecast contract and `howhard-decision-v11` (#128). `pick`, `summary` and `context`
  accept an optional `--host NAME` (Python `host=`) for the agent that asks for the
  route, and each candidate or configured inventory entry accepts an optional `agent`
  label for the agent product that would run the cell. Both are stored in the receipt
  unverified and listed by `report` under `routing`; no name means one unnamed host or
  agent. Forecast rows accept optional 95% ranges `seconds_ci: [lo, hi]` and
  `token_usage_ci: {kind: [lo, hi]}` beside `loud_ci` and `silent_ci`; a range that
  misses its value or is malformed is refused by field name. Stored prototype forecasts
  keep every range and the host that made them
  (`howhard.prototypes.stored_forecasts`). Selection is unchanged: the new inputs are
  validated, stored and reported, not read by the selection math. New receipts use
  `howhard-decision-v11` (`howhard-semantic-v8`, `semantic-reader-v8`,
  `howhard-capability-v6`, `howhard-agent-request-v7`); `v10`, `v9`, `v8` and `v7`
  receipts keep loading, observing, replaying and reporting, and read as unnamed.
  Requests prepared under `howhard-agent-request-v6` must be prepared again. The v11
  shape stays open until the release that freezes it.

- Evidence by direct similarity (#129). The outcome counts, the active-time calibration
  and the capability threshold now read one set of candidate runs per decision: the
  recorded `clean`, `loud` and `silent` runs that belong to a prototype the new task
  matches. Prototypes only choose the candidates; a task that matches no prototype
  counts nothing, and there is no fallback search. Each candidate is compared with the
  new task once, when the request is prepared and outside the storage lock, and weighs
  `weight * similarity * 0.5^(days / half_life_days)`, where `similarity` is
  `1 / (1 + exp(-score))` of the run's own task text (a negative score still counts, at
  a small weight) and `days` runs from the run's `observed_at` to the prepare time. All
  runs of one `ticket` together count for at most one run; a run with no ticket is not
  limited. A cell's outcome counts and active-time weights come from that cell's own
  runs; the capability threshold keeps its outcome and cause strengths. New optional
  `[sampling]` settings `half_life_days = 14` and `comparison_cap = 256` (the most
  recent candidates scored per decision) replace count-based ageing and the per-cell
  bounds of 8 and 24 counted runs. `sampling.outcome_recency` is accepted and ignored: a
  file that sets it loads with one notice on standard error. A v11 receipt stores
  `semantic.evidence_candidates` (prepare time, both constants, the number of
  candidates before the cap and each scored candidate once), each outcome observation's
  `discount`, `ticket_share` and `weight`, and each capability observation's `strength`,
  `ticket_share` and `weight`; loading it recomputes every similarity, age factor,
  ticket share, weight and count and refuses a mismatch or more than `comparison_cap`
  candidates. Every scored candidate is a dependency of the prepared request. `v10`,
  `v9`, `v8` and `v7` receipts keep loading, observing, replaying and reporting with
  their own evidence shapes and limits. An `Engine(retrieval=...)` object must provide
  `reranker.score(query, documents)`, as `RetrievalPipeline` does.

- Tiered admission (#130). Every forecast cell reports `pi`, the posterior chance that
  its accepted-defect share is within the risk class's epsilon, and a `tier`: `trusted`
  when `pi >= admission_quantile` (0.80), `on-trial` when
  `trial_probability <= pi < admission_quantile` (0.20), `excluded` below that.
  `admissible` is true exactly when the tier is trusted, which is the cell the earlier
  cap admitted. Trusted and on-trial cells with enough credit compete together on one
  drawn sample; when there is none, the cells with enough credit whose `pi` is at least
  `all_excluded_factor` (0.5) times the best `pi` compete instead, so a sample is drawn
  for every decision and `exploration.pool` records `trusted-and-on-trial` or
  `all-excluded`. **Changed meaning:** `cap_met` is now true when the chosen cell is
  trusted, and `needs_review` is its opposite; the flag is advice to the caller and
  nothing is enforced or recorded for it. Widening runs while every forecast cell is
  excluded (round trigger `all-excluded`) and stops at the first trusted or on-trial
  cell (`widening_stop` `pickable`). `[sampling]` gains `trial_probability`, from 0 to
  `admission_quantile`, and `all_excluded_factor`, from 0 to 1; a value outside its
  range is refused by key name. **Retired:** `on_cap_unmet`. An explicit configuration
  file no longer has to state it, the packaged default drops it, and a file that still
  sets it loads with the value ignored and one notice on standard error. Receipts no
  longer record it, `exploration.reason` no longer takes `disabled-cap-unmet`, and
  `why` no longer takes `lowest-bound`; `silent-cap` now marks a cell outside the pool
  and quotes its `pi`. Loading a v11 decision recomputes `pi` and refuses a changed
  `pi`, `tier`, `admissible`, pool, chosen cell or flag. `v10`, `v9`, `v8` and `v7`
  receipts keep their recorded cap, policy and widening labels, and keep loading,
  observing, replaying and reporting.

- Beliefs for active time and each token kind, and a credit check on the 80% point
  (#132). A `howhard-decision-v11` cell holds one log-normal belief for active time and
  one per token kind (or one for total `tokens` when its forecasts do not state usage by
  kind). The experts' forecasts of a value are merged on the log scale, and their
  disagreement and the stated ranges `seconds_ci` and `token_usage_ci` set the belief's
  spread, never below the minimum (1.007 by default). Runs of the same cell update the
  beliefs, each at the weight the outcome counts give it: active time from every scored
  evidence candidate of the cell that recorded a time, a token kind from those with a
  `complete` `execution_usage`, and the total from those that recorded `tokens`. Each
  scored candidate and each row of `semantic.duration_observations` carries the run's
  `tokens` and `token_usage`. `T`,
  `needed_credit_usd` and `session_cost_usd` are the beliefs' averages; this replaces the
  per-expert active-time calibration for v11 decisions. Each cell reports `active_time`,
  `token_beliefs`, `credit_variance` and `credit_p80_usd`, and `credit_ok` now requires
  `max(needed_credit_usd, credit_p80_usd) <= available_credit_usd`, so a run must fit the
  remaining credit four times out of five and not only on average. A cell that passed
  the credit check on its average alone can now be refused; the `insufficient-credit`
  explanation and the no-schedulable-candidate error name the amount that was checked.
  The plain output of `pick` and `summary` adds a `credit_p80_usd` column. Optional
  `[sampling]` keys `model_worth` (5), `task_worth_amount` (1), `duration_miss` (1.3) and
  `scatter_share` (0.5) are stored in the receipt's policy. Loading a v11 receipt
  recomputes its beliefs, `T`, the credit numbers and `credit_ok`; `v10`, `v9`, `v8` and
  `v7` receipts keep their own active-time calibration and credit check and load,
  observe, replay and report as before. Hosts are asked to record usage by token kind and
  active time apart from waiting.

- Outcome beliefs on the log-odds scale (#133). A cell now holds two independent
  beliefs in place of each expert's Dirichlet posterior: one for its loud rate and one
  for its accepted-defect share (silent results / accepted results). The experts'
  forecasts for a cell are first merged by their weights on the log-odds scale into one
  forecast per rate, so experts that disagree make it less certain. Each belief starts
  as a bell curve on one fixed grid of 2,501 log-odds points whose average equals that
  forecast and whose width is that of a record worth `N_start` runs (4.32 runs for the
  accepted-defect share, 3.81 for the loud rate); a stated `loud_ci` or `silent_ci` can
  widen the start, up to the flat belief, and never narrows it. The cell's counted runs
  then reshape it. **Changed behaviour:** the forecast no longer fades out after two
  counted runs and there is no unit base of `(0.88, 0.08, 0.04)`; the forecast keeps
  the weight of `N_start` runs however many are counted, so `pi`, tiers, `J`, `cost_sd`
  and sampled costs differ from earlier builds for the same inputs. `pi` and
  `admission_risk` are read from the accepted-defect belief. `P_loud` and
  `P_silent_accepted` keep `{mean, lower, upper}`; **`P_clean` and `P_silent` now carry
  `mean` only**, because neither has a belief of its own, and the rendered table prints
  no range for them. Until the cost is averaged over the beliefs, `J` is the same
  formula at `P_loud = E[theta_l]` and `P_silent = (1 - E[theta_l]) * E[S]`, and the
  draw takes one uniform number per rate and cell, read at the first grid point whose
  running total reaches it. Every value of a cell is now one belief merged over the
  experts, so `cost_sd` adds the variances of the two rates to the active-time term and
  has no separate between-expert term, and its sampling cell no longer depends on the
  sampled expert. New optional `[sampling]` settings, each positive:
  `task_worth_silent = 32` and `task_worth_loud = 16`, beside the shared
  `model_worth = 5`. **Retired:**
  `sampling.prior_strength`, `sampling.min_concentration` and
  `sampling.max_concentration`; a file that still sets one loads with the value ignored
  and one notice per key on standard error, and decisions are the same as without it.
  A v11 receipt has `semantic.interval_kind` `log-odds-grid`, stores
  `outcome_beliefs.loud` and `outcome_beliefs.silent` (`prediction`, `stated_spread`,
  `n_start`, `spread`, `centre`, `counts`, `mean`, `lower`, `upper`, plus `pi` on
  `silent` and `attempts` and `odds` on `loud`) in each `semantic.sampling.cells` entry
  in place of `outcome_posterior` and `concentration`, and stores the three worths, not
  the retired keys, in its sampling policy. Loading a v11 decision recomputes both
  beliefs, the `P_*` fields, `pi` and `J`, and refuses a receipt with a changed number.
  v11 decisions stored by earlier unreleased builds no longer load. `v10`, `v9`, `v8`
  and `v7` receipts keep their recorded posteriors and policy and keep loading,
  observing, replaying and reporting.

- Host-aware forecast band and reserved forecast slots (#131). Both edges of the fit
  band's clearing interval now follow the share of its failures that the asking host
  catches before acceptance. From the failed routed runs on record (finished `loud` or
  `silent` runs whose decision receipt holds a forecast for the cell that ran, each
  weighing `0.5^(days / half_life_days)`, with one ticket's failed runs counting as one
  run at most) the filter computes `d_0 = (1 - 2 * epsilon) / (1 - epsilon)`,
  `d(all) = (C_all + 12 * d_0) / (F_all + 12)` and
  `d(host) = (C_host + 12 * d(all)) / (F_host + 12)`, where `F` sums the failed runs
  and `C` the `loud` ones, for all hosts and for the host named with `--host`. A
  receipt with no host name, or older than v11, counts under the one unnamed host. The
  band starts at the first cell with
  `P_clear >= e = max(min_clear_probability, 1 - epsilon / (1 - d + epsilon * d))` and
  ends at the first reliable cell within
  `g = min(overpowered_gain_margin, overpowered_gain_margin * epsilon / ((1 - epsilon) * (1 - d)))`
  of the strongest cell. At `d >= d_0`, and so with no failed run on record, `e` and
  `g` are the configured 0.5 and 0.05 and the band is unchanged; at `epsilon = 0.10` a
  host that catches half of its failures gets 0.818 and 0.0111. When no cell reaches
  `e`, the band starts at the first cell with the highest `P_clear` if that reaches
  `min_clear_probability`. The timely interval keeps the configured values.
  Two of the `max_forecast_cells` slots are reserved: the core is the band's first
  `max_forecast_cells - 2` cells, and up to two further cells are forecast, first for
  model families with no run among the evidence candidates (at `P_clear >= 0.2`), then
  for agents with no core cell (at `P_clear >= e`): a proven family whose `agent` is
  missing takes at most one slot for that agent, and an agent whose unproven family
  already took a slot counts as served. Each slot goes to the nearest qualifying cell.
  A slot that no family takes goes back to the band, so with no family to reserve for
  the forecast cells are the band as before. A
  forecast request can therefore include a cell classified `underpowered` or
  `overpowered`. Widening takes the next stronger cell with no forecast: band cells
  beyond the core, then overpowered cells, skipping reserved cells. New optional
  `[capability]` settings: `caught_share_worth = 12`, `reserved_forecast_cells = 2`
  (an integer below `max_forecast_cells`; 0 turns the rule off) and
  `unproven_clear_probability = 0.2`. A v11 receipt's capability block stores
  `caught_share` (`host`, `F_all`, `C_all`, `F_host`, `C_host`, `d_0`, `d_all`,
  `d_host`, `e`, `g`), `core`, `proven_families` and `reserved` (each cell's `key`,
  `reason` and `bar`); loading recomputes them and refuses a receipt whose band, core,
  reserved cells or widening do not follow from its stored numbers. `report` lists
  them per decision under `filter_quality.bands`. `v7` to `v10` receipts load, observe,
  replay and report as before.

- Fix: a capability probability can no longer round above 1 (#146). When many clean
  runs at the strongest cell left almost no posterior mass above it, the running sum
  behind `P_clear` and `P_timely` could come out a few units in the last place above 1
  (for example `1.0000000000000004`). The receipt was saved, then refused on load with
  `probabilities must be monotone along this task's order`, which also stopped
  `observe --decision` and `report` on that store. Because run weights depend on each
  run's age in seconds, the same scenario failed only some of the time. Both
  probabilities are now held within [0, 1] when they are computed; the receipt check
  is unchanged, so a receipt with a probability above 1 is still refused, including one
  stored by an earlier build.

- Stored forecasts are reused per host and merged on each value's own scale (#136). A
  prototype now keeps one stored Stage 2 forecast per cell and host, and a request
  reuses only the forecasts made under its own `--host` name; a request with no name
  reuses only forecasts stored with no name, which includes every forecast stored
  before hosts were named. A new forecast replaces the same host's forecast whole and
  leaves the other hosts' forecasts stored. When several matched prototypes store a
  usable forecast, the loud rate and the accepted-defect share are merged on the
  log-odds scale and active time and each token kind on the log scale, at each
  prototype's share of the match weight; the merged row always states ranges, which
  hold each forecast's own spread plus the distance between the forecasts, so forecasts
  that disagree widen them. One usable forecast is reused unchanged. A token kind that
  every usable forecast states as exactly zero stays exactly zero, and a stored forecast
  without the six token kinds is not reused for a cell priced by kind. Stage 2 no longer
  builds a forecast from prototype statistics or from the `(0.88, 0.08, 0.04)` base: a
  cell with recorded runs and no stored forecast under the request's host goes to the
  forecaster, once per cell and host, so a warm store answers its first request of each
  host with a non-null Stage 2 `baseline_request`. Stage 1 and prototype matching are
  unchanged, and a split copies every host's stored forecast to each child. A calling
  agent that returns the full forecast panel must repeat each reused row unchanged. A
  v11 receipt lists every reused cell in `semantic.stored_forecasts` (each source's
  prototype, match weight, share, host and stored row, and each value's merged centre
  and spread); loading it recomputes the shares, the merge and the row, and refuses an
  altered one or a source stored under another host. `howhard.prototypes.stored_forecasts`
  returns one entry per prototype, cell and host. The inference cache of an explicit
  backend is keyed by the host's name too, the unnamed host being a key of its own: an
  answer cached for one host is never returned or stored for another, a second host's
  identical request calls the backend again, and answers cached by earlier builds are
  not read. No configuration key is added and an existing store loads unchanged. v11 decisions stored by earlier unreleased builds no
  longer load; `v7` to `v10` receipts keep loading, observing, replaying and reporting.

- Corrected prediction (#139). Every belief now starts from a prediction, not from the
  merged forecast itself:
  `phi(prediction) = phi(level) + gamma * (phi(guess) - phi(usual))`, on the log-odds
  scale for the loud rate and the accepted-defect share and on the log scale for active
  time, total tokens and each token kind. `level` (what happened) and `usual` (what had
  been forecast) come down a ladder of five rungs: all hosts, this host, this agent
  under it, this family under it and this cell under it. Each rung is the weighted
  average of its runs and the rung above, which is worth `model_worth = 5` runs at the
  family rung and the new optional `[sampling]` setting `rung_worth = 40` (positive)
  elsewhere; above all hosts both equal the guess. The runs are the finished routed
  runs on other tasks: outcome `clean`, `loud` or `silent`, with a decision that stored
  a forecast for the cell that ran, under the host and agent that decision stores. The
  decision's evidence candidates, runs with no decision and `interrupted` runs enter no
  ladder. A run weighs its age factor, and all runs of one ticket together count for at
  most one run. `gamma`, the weight on the forecaster's judgment, is measured once per
  kind of value from all hosts' runs: the gap that happened between the runs forecast
  above usual and the rest, divided by the gap that was forecast, held in `[0, 1]` and
  leaning on 1 at the worth of `rung_worth` runs. The family's own runs set how firmly
  a prediction is held (`worth = n_family + model_worth`), and for an amount the ladder
  of squared misses gives `D`, the typical miss left after the correction, in place of
  the fixed `duration_miss`. **Changed behaviour:** with finished routed runs on
  record, predictions, `pi`, tiers, `T`, credit, `J`, `cost_sd` and sampled costs differ
  from earlier builds for the same inputs; with an empty record every prediction is the
  merged forecast, `gamma` is 1, `worth` is 5 and `D` is 1.3, as before. A v11 receipt
  stores `semantic.prediction`: per kind of value `n`, both groups' sums, `gamma_hat`
  and `gamma`; per forecast cell and value the `guess`, each rung's sums, `level`,
  `usual`, `prediction`, `worth`, `N_start`, `s_stated`, `s` and, for an amount, `D`.
  Each amount belief also reports `prediction_seconds` or `prediction_tokens` beside
  its forecast. Loading a v11 decision merges every guess again from the frozen
  forecasts, recomputes every derived number from the stored sums, and refuses a
  receipt whose numbers do not follow. v11 decisions stored by earlier unreleased
  builds no longer load. Runs with receipts older than v11 sit under the unnamed host
  and the unnamed agent; `v10`, `v9`, `v8` and `v7` receipts keep loading, observing,
  replaying and reporting unchanged.

- Relatedness discounts (#134). A recorded run now counts as outcome evidence for cells
  other than the one that ran it, at a discount set by how related the two cells are:
  1 for the same cell, `effort_discount` (0.5) for the same family at another effort,
  `model_discount` (0.25) for another family of the same agent and `agent_discount`
  (0.25) for another agent; the first that fits is used. A run's weight for a judged
  cell is `weight * similarity * age factor * discount`, and the per-ticket limit is
  applied after the discount, per judged cell, so one ticket still counts for at most
  one run. **Changed behaviour:** a cell's two outcome beliefs, and so `pi`, tiers, `J`
  and sampled costs, now move with the runs of related cells; a cell with no run of its
  own no longer always keeps its start. Only the outcome counts change: active time and
  tokens still count runs of the same cell alone, now limited per ticket over that
  cell's own runs, and the capability threshold keeps its own strength factors. A
  cell's family comes from the model registry and its agent from the optional `agent`
  label of its candidate. A run's agent is the label stored for the chosen cell in the
  receipt its `decision_id` names; a run with no decision, a receipt older than v11 or
  an unlabeled candidate belongs to the one unnamed agent, as does an unlabeled cell of
  the request. New optional `[sampling]` settings `effort_discount = 0.5`,
  `model_discount = 0.25` and `agent_discount = 0.25`, each in `[0, 1]` and refused by
  key outside it; with all three at 0 only a cell's own runs count. A v11 receipt
  freezes the three discounts in its sampling policy, stores each scored candidate's
  `agent`, and stores every candidate run once in `semantic.outcome_observations` with
  the run's own `family`, `effort`, `agent` (null when unnamed), `outcome`, `ticket` and
  its `weight` before the discount, in place of one row per counted cell with
  `discount` and `ticket_share`. Each forecast cell in `selection.cells` gains
  `outcome_counts`: the sums `n_clean`, `n_loud` and `n_silent` and `runs`, the number
  of runs behind them. No row per run and cell is stored, because each run's relation,
  discount, ticket share and weight follow from the stored labels and the frozen
  discounts; `howhard.decisions.counted_observations(receipt, cell)` computes them on
  demand (`id`, `relation` of `cell`, `family`, `agent` or `other`, `discount` and
  `weight` after the limit). Loading a v11 decision recomputes them from the stored
  labels and policy and refuses a receipt whose sums or run count differ; an
  observation's own cell need not be a forecast cell.
  v11 decisions stored by earlier unreleased builds no longer load. `v10`, `v9`, `v8`
  and `v7` receipts keep their own evidence shapes and keep loading, observing,
  replaying and reporting.

- Cost averaged over the beliefs, its exact spread and a draw of every uncertain value
  (#135). **Changed behaviour:** `J` is now the cost to an accepted result averaged
  over the cell's beliefs, `(C_run + lambda_per_hour * E[T]) * E[R] + C_loud * (E[R] - 1)
  + C_silent * E[S]`, where `R = 1 / (1 - loud rate)` is the attempts until a run is not
  loud and `S` the accepted-defect share; it replaces the cost at the posterior means.
  `expected_attempts` is `E[R]`, which is above `1 / (1 - P_loud)` because the retry
  factor is curved: at the same average loud rate a wider loud belief now charges more
  attempts and a higher `J`. `cost_sd` is the exact spread of that cost from the means
  and variances of four independent beliefs (credit, active time, the loud rate and the
  accepted-defect share); it replaces the first-order estimate that treated run cost as
  exact, so a wider stated range on active time or on a token kind raises that cell's
  `cost_sd` and changes no other cell's numbers. Eligibility keeps its rule,
  `J(c) - J(c*) <= uncertainty_multiplier * (sd(c*) + sd(c))` with a class
  `explore_within` capping the gap, on the new `J` and `cost_sd`. The draw no longer
  samples one expert: every forecast cell draws one value of its accepted-defect share,
  its loud rate, its active time and its token counts, priced at the cell's rates, and
  the lowest sampled cost among the eligible cells wins. The numbers are
  `((r >> 12) + 0.5) / 2^52` from raw `numpy.random.PCG64` output seeded with the stored
  `seed`, four per forecast cell in the order of `semantic.sampling.cells`, so **a seed
  gives a different choice than in earlier builds** and a replay does not depend on a
  library's distribution methods. Each `semantic.sampling.cells` entry gains `sampled`
  (`S`, `u`, `seconds`, `credit`) and drops `expert`; `semantic.sampled_expert` is no
  longer written. The `sampled-cost` explanation in `why` compares the parts
  `run = q * K_s * R_s`, `time = lambda_per_hour * T_s * R_s`, `loud = C_loud * (R_s - 1)`
  and `silent = C_silent * S_s`. Loading a v11 receipt recomputes `J`,
  `expected_attempts` and `cost_sd` from its beliefs and replays the draw from its seed:
  a receipt whose `J`, `expected_attempts`, `cost_sd`, sampled value, `sampled_J`,
  `exploration.eligible_cells` or chosen cell differs is refused with a message that
  names the field. No configuration key is added. The token beliefs now also count a
  same-cell run that records a `complete` `execution_usage` and no active time, at the
  weight a run of its own cell carries: such a run has a row in
  `semantic.duration_observations` with `active_seconds` null, moves the token beliefs
  and leaves the active-time belief unchanged. v11 decisions stored by earlier
  unreleased builds no longer load. `v10`, `v9`, `v8` and `v7` receipts keep the cost at
  the posterior means and their recorded draw, are not replayed, and keep loading,
  observing, replaying and reporting.

## 5.0.0 — 2026-11-24

- Dynamic synthesized `(task, context)` prototypes with materialized per-cell priors and
  sufficient statistics in `payloads.sqlite3` (`prototypes`, `prototype_members`,
  `prototype_cells`, `prototype_meta`): strip one-off worker wrappers, issue IDs, commit
  hashes, and temporary scratch paths when synthesizing representative tasks, match
  incoming tasks on task-only text (`score > 0`) without shared project boilerplate
  inflating similarity, maintain cell statistics incrementally on `observe`, replay
  affected prototypes reversibly across `amends` and `measurement_corrections` while
  excluding `interrupted` runs, backfill pre-existing `outcomes.jsonl` histories online,
  and split heterogeneous prototypes (`member_count >= 12`) into sharper child prototypes
  (#123).

- Cold-start-only `baseline_request` and provider-free warm routing: evaluate stateless
  Stage 1 (`filter`) and Stage 2 (`forecast`) priors once per unseen `(prototype, model,
  effort)` cell on the prototype's representative `(task, context)`, emit
  `baseline_request = null` when matched prototypes already cover all cells (or request
  Stage 2 forecasts only for newly added unseen cells), complete warm `pick`/`summary`
  and `howhard context --auto-decide` locally without an LLM provider when no active
  editable semantic memories exist, and retain bounded multi-corpus `BM25Index` (16 FTS5
  snapshots) and `HNSWIndex` (64 graphs) caches with query-vector memoization (#123).

- Continuous cold-start prior decay and `howhard-decision-v10`: decay cold-start forecast
  prior concentration linearly over empirical cell weight (`tau_emp = 2.0`) into an
  empirical base floor (`kappa_floor = 1.0`, `b_0 = (0.88, 0.08, 0.04)`) so zero-shot
  guesses carry exact zero weight once `W_emp >= 2.0`, retrieve per-cell capability,
  duration, and outcome precedents (`score > 0`) without global top-8 truncation, and
  emit `howhard-decision-v10` (`howhard-semantic-v7`, `howhard-capability-v5`,
  `howhard-agent-request-v6`) while preserving full `v9`, `v8`, and `v7` frozen receipt
  loading, observation, belief replay, and reporting (#123).

## 4.0.0 — 2026-11-21

- Breaking storage change: `payloads.sqlite3` replaces `<state_root>/requests/*.json`
  and `<state_root>/decisions/*.json`, storing complete request and decision JSON
  losslessly as zstd-compressed SQLite payloads and exposing indexed decision summaries
  without decompressing semantic evidence. Runtime commands and APIs have no legacy
  loose-JSON fallbacks and never perform automatic migration; pre-4.0 stores migrate
  offline via standalone `scripts/migrate-payloads.py` (included in the source
  repository and `.tar.gz` source distribution), which migrates and removes verified
  loose `requests/*.json` and `decisions/*.json` under store write admission
  (#109, #116).

- Concentrate the capability prior near the lower cut and keep separate clearing and
  timely-clearing intervals when their combined span exceeds `max_forecast_cells`.
  The complete inventory and both posteriors stay in the receipt. New receipts emit
  `howhard-capability-v4` blocks recording the exact forecast set, while frozen
  decision v9, v8, and v7 receipts in `payloads.sqlite3` continue to load, accept
  observations, replay, and report (#110).

## 3.4.0 — 2026-11-20

- Replace point outcome forecasts with per-cell Dirichlet–multinomial posteriors: each
  expert forecast is a bounded prior (`kappa_0 <= prior_strength`) updated only by that
  `(family, effort)` cell's own recency-weighted `outcome_observations` (up to
  `outcome_evidence_limit = 24` task-similar non-`interrupted` executions retrieved
  independently per forecasted cell, aged in chronological journal order with
  `sampling.outcome_recency = 0.95`), so global expert evidence counts no longer shrink
  unseen cells' uncertainty and heavy history on one cell cannot crowd out another.
  Forecast rows accept optional `silent_ci`/`loud_ci` 95% credible intervals that can
  only widen the prior (down to `min_concentration`, now 2.0 by default). `P_clean`,
  `P_loud`, `P_silent` and `P_silent_accepted` report real Beta mixture credible
  intervals even for `llm`-only forecasts; admission uses the 80% posterior quantile of
  accepted silent risk (`sampling.admission_quantile`, reported as `admission_risk`);
  Thompson eligibility uses posterior cost overlap, with `explore_within` honored only
  when set on a risk class. The global `sampling.explore_within` setting is rejected
  with a pointer to the risk-class setting. New receipts use `howhard-decision-v9`
  (`howhard-semantic-v6`, `semantic-reader-v7`) while frozen v8 and v7 receipts still
  load, validate and replay (#103).

- Add `interrupted` as an operational observation `outcome` alongside `clean`, `loud`, and
  `silent`, with structured causes (`usage_limit`, `provider_error`, `host_error`,
  `cancelled`, and `unrecorded`) so usage-limit and operational aborts are not misclassified
  as `loud` model failures. Keep LLM forecasts and `J` ternary while excluding `interrupted`
  runs from predictive belief replay, capability threshold learning, active-duration
  calibration, forecast error metrics, and `filter_quality` failure/disagreement resolution,
  preserving durable usage/cost/time accounting, amendment reversibility, and separate
  `outcomes.interrupted` / `interrupted_causes` report counts (#104).

## 3.3.0 — 2026-11-16

- Separate active execution duration from passive queue wait with optional
  `active_seconds` and `queue_wait_seconds` across `observe`, `--measurement-corrections`,
  `howhard-usage-manifest-v1` / `howhard-execution-usage-v1`, `evidence`, and `report`,
  scoring forecasted active `seconds` against `effective_active_seconds`, calibrating
  empirical active duration and log-normal right-tail variance (`exp(mu + sigma^2 / 2)`)
  in `J`, `cost_sd`, and Thompson `sample()`, preventing slow `clean` runs with severe
  active-duration overruns from prematurely capping capability `upper` (`P_timely`), and
  versioning new receipts as `howhard-decision-v8` (`howhard-semantic-v5`,
  `howhard-capability-v3`) while preserving frozen `howhard-decision-v7` validation and
  replay (#99).

## 3.2.0 — 2026-11-14

- Separate actual session money cost from list-price quota credit with optional
  `cost_per_credit`, explicit `session_cost_usd` and `needed_credit_usd` cell fields,
  credit-named budget fields, cost-based selection, and `howhard-decision-v7` receipt
  normalization while retaining v1–v6 receipts as historical store evidence (#95).

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
