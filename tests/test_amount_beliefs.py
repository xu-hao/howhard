"""Beliefs for active time and each token kind, the credit spread and the credit check (#132).

Every decision here is routed through the public calling-agent interface, and the numbers
asserted are the worked examples of the selection contract. Fixtures are synthetic and own
their state: prices, forecasts and earlier runs are test inputs, not provider claims.

Earlier runs count through the request's scored evidence candidates (#129): a run of the
judged cell weighs its similarity to the new task times its age factor, which is 1 for a
run recorded moments before the request. A test first routes the task with its forecast,
which the matched prototype stores; after the runs are recorded, routing again reuses that
forecast and the runs update the beliefs.
"""

from copy import deepcopy
import json
import math
import statistics

import pytest

from howhard import Engine
from howhard.capability import EFFORT_ORDER
from howhard.cli import main
from howhard.context import canonical, digest
from howhard.decisions import load
from howhard.payload_store import put
from howhard.report import report
from howhard.storage import locked
from howhard.validation import HowHardError
from test_capability_filter import filtering
from test_evidence_similarity import Scripted
from test_execution_accounting import artifact, card, codex_events, counts
from test_outcome_posterior import reference_belief

TASK = "Repair rollback ordering"
MODEL = "fixture/worker"
# tau / sqrt(N_start) with worth 5, one run per task, D = 1.3 and half of D^2 as scatter.
TAU = 1.3 * math.sqrt(.5)
MIN_SPREAD = TAU / math.sqrt(1 / (1 / 5 + 1 / 1))
RATES = counts(input=2, output=10, cache_read=.2, cache_write=3, cache_write_5m=2.5, cache_write_1h=4)
# A range of this ratio around a value states a spread of exactly 2.0.
WIDE = math.exp(3.92)


def stated(weight):
    """Task text for a run that counts for ``weight``: its similarity, stated as a reranker score."""
    return f"{TASK} relevance {math.log(weight / (1 - weight)):.12f}"


def forecast(seconds=3600, tokens=1000, **extra):
    return {"clean": .98, "loud": .01, "silent": .01, "tokens": tokens, "seconds": seconds, **extra}


def by_kind(usage, **extra):
    return forecast(tokens=sum(usage.values()), token_usage=usage, **extra)


def decide(engine, candidates, rows=None, *, memory_rows=(), seed=7):
    """One routed decision; ``rows`` maps (model, effort) to that cell's forecast fields.

    A cell whose forecast a matched prototype already stores reuses it, so ``rows`` is
    needed only the first time a task is routed.
    """
    pending = engine.prepare("task", TASK, native_candidates=candidates)
    if pending["baseline_request"] is None:
        pending = engine.decide(pending["request_id"], {"memory_filters": [], "reason": "Stored fit band"})
    else:
        ordered = sorted(pending["baseline_request"]["cells"], key=lambda c: (c["model"], EFFORT_ORDER[c["effort"]]))
        pending = engine.decide(pending["request_id"], filtering(pending["baseline_request"], 0, len(candidates) - 1,
                                                                 [c["key"] for c in ordered]))
    if pending["baseline_request"] is None:
        assert rows is None and not memory_rows
        response = {"memory_forecasts": [], "reason": "Stored forecasts"}
    else:
        cells = pending["baseline_request"]["cells"]
        assert len(cells) == len(candidates)
        forecasts = [{"family": c["family"], "effort": c["effort"], **rows[c["model"], c["effort"]]} for c in cells]
        memory = [{"id": m["id"], "version": m["version"],
                   "forecasts": [{"family": c["family"], "effort": c["effort"], **changed[c["model"], c["effort"]]}
                                 for c in cells]} for m, changed in memory_rows]
        response = {"forecasts": forecasts, "memory_forecasts": memory, "reason": "Synthetic belief fixture"}
    return pending["request_id"], response, engine.decide(pending["request_id"], response, seed=seed)


def record(engine, run_id, seconds, weight, *, effort="high", **options):
    """Record one clean run of the cell that counts for ``weight`` in a request prepared right after."""
    engine.observe("task", MODEL, effort, "clean", native=True, task_text=stated(weight), id=run_id,
                   active_seconds=seconds, **options)


def one_cell(tmp_path, row, *, candidate=None, runs=(), seed=7):
    """Route a single cell; with ``runs`` (id, active seconds, weight, observe options), route it again after them."""
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    cells = [{"model": MODEL, "effort": "high", **(candidate or {})}]
    _, _, result = decide(engine, cells, {(MODEL, "high"): row}, seed=seed)
    if runs:
        for run_id, seconds, weight, options in runs:
            record(engine, run_id, seconds, weight, **options)
        _, _, result = decide(engine, cells, seed=seed)
    return engine, result, result["cells"][0]


def cell(result, model, effort="high"):
    return next(c for c in result["cells"] if (c["model"], c["effort"]) == (model, effort))


# --- 1. One forecast per value ------------------------------------------------------------

def test_expert_forecasts_merge_on_the_log_scale_and_disagreement_is_stated_spread(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    memories = [engine.memory.add(f"Lesson {i}", f"Rollback ordering lesson number {i}.") for i in range(4)]
    same, slower = forecast(3000), forecast(6000)
    # The reader and two memories forecast 3,000 seconds; two memories forecast 6,000.
    _, _, result = decide(engine, [{"model": MODEL, "effort": "high"}], {(MODEL, "high"): same},
                          memory_rows=[(m, {(MODEL, "high"): same if i < 2 else slower}) for i, m in enumerate(memories)])
    weights = {e["key"]: e["weight"] for e in result["semantic"]["beliefs"]["experts"]}
    assert weights["llm"] == pytest.approx(.5) and len(weights) == 5
    slow = sum(weights[f"memory:{m['id']}:{m['version']}"] for m in memories[2:])
    assert slow == pytest.approx(.25)                                  # weights 0.75 and 0.25

    belief = result["cells"][0]["active_time"]
    assert belief["forecast_seconds"] == pytest.approx(3567.6, abs=.05)   # 3,568 seconds
    assert belief["stated_spread"] == pytest.approx(.30, abs=.001)
    # Disagreement narrower than the minimum spread leaves the belief at the minimum.
    assert belief["prior_spread"] == pytest.approx(MIN_SPREAD)
    assert result["cells"][0]["T"] == pytest.approx(3567.6 / 3600, rel=1e-4)
    load(engine.config, result["decision_id"])


def test_expert_disagreement_wider_than_the_minimum_widens_the_belief(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    memory = engine.memory.add("Slow path", "Rollback ordering can take a full day.")
    _, _, result = decide(engine, [{"model": MODEL, "effort": "high"}], {(MODEL, "high"): forecast(600)},
                          memory_rows=[(memory, {(MODEL, "high"): forecast(86400)})])
    belief = result["cells"][0]["active_time"]
    # Weights 0.8 and 0.2: s_g^2 = 0.8 * 0.2 * ln(144)^2.
    assert belief["stated_spread"] == pytest.approx(.4 * math.log(144))
    assert belief["prior_spread"] == belief["stated_spread"] > MIN_SPREAD
    assert belief["forecast_seconds"] == pytest.approx(600 * 144 ** .2)
    assert result["cells"][0]["T"] * 3600 == pytest.approx(belief["forecast_seconds"])


# --- 2. The active-time belief -------------------------------------------------------------

@pytest.mark.parametrize("stated, spread", [
    ({}, 1.007),                                    # no range: the minimum spread
    ({"seconds_ci": [600, 60000]}, 1.175),          # a wider stated range is used
    ({"seconds_ci": [3000, 4000]}, 1.007),          # a narrower one never narrows the belief
    ({"seconds_ci": [3600, 3600]}, 1.007),
])
def test_active_time_belief_starts_at_the_forecast_and_a_range_only_widens_it(tmp_path, stated, spread):
    engine, result, row = one_cell(tmp_path, forecast(3600, **stated))
    belief = row["active_time"]
    assert belief["prior_spread"] == pytest.approx(spread, abs=5e-4)
    assert (belief["worth"], belief["D"]) == (5., 1.3)
    assert belief["N_start"] == pytest.approx(.833, abs=5e-4)
    # The centre makes the belief's average equal the forecast.
    assert belief["prior_centre"] == pytest.approx(math.log(3600) - belief["prior_spread"] ** 2 / 2)
    assert belief["expected_seconds"] == pytest.approx(3600) and row["T"] == pytest.approx(1.)
    # No run yet: the belief after the runs is the belief before them.
    assert (belief["centre"], belief["spread"], belief["evidence_weight"]) == (
        belief["prior_centre"], belief["prior_spread"], 0.)
    if not stated:
        assert belief["prior_centre"] == pytest.approx(7.682, abs=5e-4)
        assert math.exp(belief["centre"]) / 3600 == pytest.approx(.60, abs=.005)   # the median, in hours
    load(engine.config, result["decision_id"])


def test_same_cell_runs_update_the_active_time_belief_by_their_weights(tmp_path):
    engine, result, row = one_cell(tmp_path, forecast(3600),
                                   runs=[("earlier", 1800., .5, {}), ("later", 5400., .3, {})])
    observed = {o["id"]: o for o in result["semantic"]["duration_observations"]}
    # The active-time reader counts each run at the weight the outcome counts give it.
    counted = {o["id"]: o["weight"] for o in result["semantic"]["outcome_observations"]}
    assert {k: o["weight"] for k, o in observed.items()} == counted
    assert observed["earlier"]["weight"] == pytest.approx(.5, rel=1e-4)
    assert observed["later"]["weight"] == pytest.approx(.3, rel=1e-4)
    belief = row["active_time"]
    assert belief["evidence_weight"] == pytest.approx(.8, rel=1e-4)
    assert belief["prior_spread"] == pytest.approx(1.007, abs=5e-4)
    assert belief["spread"] == pytest.approx(.719, abs=5e-4)
    assert belief["centre"] == pytest.approx(7.792, abs=5e-4)
    assert belief["expected_seconds"] == pytest.approx(3137, abs=.5)
    assert row["T"] == pytest.approx(belief["expected_seconds"] / 3600)
    load(engine.config, result["decision_id"])


def test_runs_at_the_forecast_keep_the_average_and_other_cells_do_not_count(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    cells = [{"model": MODEL, "effort": effort} for effort in ("high", "low")]
    decide(engine, cells, {(MODEL, "high"): forecast(900), (MODEL, "low"): forecast(900)})
    for i in range(3):
        record(engine, f"run-{i}", 900., .9, queue_wait_seconds=9900.)
    record(engine, "run-3", 90000., .9, effort="low")
    _, _, result = decide(engine, cells)
    high, low = cell(result, MODEL, "high")["active_time"], cell(result, MODEL, "low")["active_time"]
    # Waiting is not active time, and three runs at the forecast narrow the belief around it.
    assert high["expected_seconds"] == pytest.approx(900) and high["spread"] < high["prior_spread"]
    assert high["evidence_weight"] == pytest.approx(2.7, rel=1e-4)
    # The slow run belongs to the other cell only: a run of another cell never counts.
    assert low["evidence_weight"] == pytest.approx(.9, rel=1e-4)
    assert low["expected_seconds"] > 5 * high["expected_seconds"]
    assert {o["id"] for o in result["semantic"]["duration_observations"] if o["effort"] == "high"} == {
        "run-0", "run-1", "run-2"}


# --- 3. Token beliefs, and 4. credit: average, spread and check ----------------------------

def priced_pair(*budgets, rates=RATES):
    """The same rate card at two efforts, each with its own remaining credit."""
    return [{"model": MODEL, "effort": effort, "rate_card": card(model=MODEL, rates=rates),
             "credit_budget": {"available_credit_usd": budget}} for effort, budget in zip(("low", "high"), budgets)]


WORKED = by_kind(counts(input=1_000_000, output=100_000), token_usage_ci={"output": [10_000, 1_000_000]})


def test_credit_check_uses_the_80_percent_point_when_it_exceeds_the_average(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    _, _, result = decide(engine, priced_pair(4.00, 5.00), {(MODEL, "low"): WORKED, (MODEL, "high"): WORKED})
    short, enough = cell(result, MODEL, "low"), cell(result, MODEL, "high")
    for row in (short, enough):
        beliefs = row["token_beliefs"]
        assert set(beliefs) == set(counts())
        assert beliefs["input"]["spread"] == pytest.approx(1.007, abs=5e-4)
        assert beliefs["output"]["spread"] == pytest.approx(1.175, abs=5e-4)
        assert beliefs["input"]["expected_tokens"] == pytest.approx(1_000_000)
        assert beliefs["output"]["expected_tokens"] == pytest.approx(100_000)
        assert row["needed_credit_usd"] == pytest.approx(3.00)             # 2.00 input + 1.00 output
        assert row["credit_variance"] == pytest.approx(19.06, abs=.005)
        assert row["credit_p80_usd"] == pytest.approx(4.17, abs=.005)
        sigma = math.sqrt(math.log1p(row["credit_variance"] / row["needed_credit_usd"] ** 2))
        assert sigma == pytest.approx(1.066, abs=5e-4)
        assert row["session_cost_usd"] == pytest.approx(row["needed_credit_usd"] * row["cost_per_credit"])
        assert row["C_run"] == row["session_cost_usd"]
    # The average fits 4.00 of credit, the 80% point does not.
    assert not short["credit_ok"] and enough["credit_ok"]
    assert result["chosen"]["effort"] == "high"
    load(engine.config, result["decision_id"])


def test_no_candidate_fits_names_the_checked_amount(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    with pytest.raises(HowHardError, match=r"minimum credit_p80_usd 4\.16829 USD, maximum available_credit_usd 4 USD"):
        decide(engine, priced_pair(4.00, 3.50), {(MODEL, "low"): WORKED, (MODEL, "high"): WORKED})


def test_a_very_wide_belief_is_checked_on_its_average_not_its_lower_80_percent_point(tmp_path):
    usage = counts(input=1_000_000, output=100_000, cache_read=500_000, cache_write=10_000,
                   cache_write_5m=20_000, cache_write_1h=5_000)
    row = by_kind(usage, token_usage_ci={kind: [value / WIDE, value * WIDE] for kind, value in usage.items()})
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    probe = decide(engine, priced_pair(1e6, 1e6), {(MODEL, "low"): row, (MODEL, "high"): row})[2]["cells"][0]
    assert all(b["spread"] == pytest.approx(2.0) for b in probe["token_beliefs"].values())
    needed, p80 = probe["needed_credit_usd"], probe["credit_p80_usd"]
    assert needed == pytest.approx(sum(usage[k] * RATES[k] for k in usage) / 1e6)
    # sigma_K = 2.0 > 1.68: the 80% point exp(0.8416 * 2 - 2) is 0.73 of the average.
    assert p80 == pytest.approx(needed * math.exp(.8416 * 2 - 2)) and p80 < needed

    between, above = (p80 + needed) / 2, needed * 1.01
    other = Engine(directory=tmp_path / "budgets", retrieval=Scripted())
    _, _, result = decide(other, priced_pair(between, above), {(MODEL, "low"): row, (MODEL, "high"): row})
    assert not cell(result, MODEL, "low")["credit_ok"] and cell(result, MODEL, "high")["credit_ok"]
    load(other.config, result["decision_id"])

    # When no budget covers the average, the refusal names the average as the checked amount.
    short = Engine(directory=tmp_path / "short", retrieval=Scripted())
    with pytest.raises(HowHardError, match=rf"minimum needed_credit_usd {needed:.6g} USD, maximum available_credit_usd"):
        decide(short, priced_pair(between, between), {(MODEL, "low"): row, (MODEL, "high"): row})


def test_cheaper_cell_without_credit_is_explained_by_the_checked_amount(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    cheap, dear = priced_pair(4.00, 50.00)
    dear["rate_card"] = card(model=MODEL, rates={k: 2 * v for k, v in RATES.items()})
    _, _, result = decide(engine, [cheap, dear], {(MODEL, "low"): WORKED, (MODEL, "high"): WORKED})
    assert result["chosen"]["effort"] == "high"
    assert [w["message"] for w in result["why"]] == ["excluded: credit_p80_usd 4.16829 > available_credit_usd 4"]
    assert result["why"][0]["reason"] == "insufficient-credit"


def test_kind_forecast_at_zero_with_no_upper_end_adds_nothing_to_credit(tmp_path):
    engine, result, row = one_cell(tmp_path, WORKED, candidate={"rate_card": card(model=MODEL, rates=RATES)})
    for kind in ("cache_read", "cache_write", "cache_write_5m", "cache_write_1h"):
        belief = row["token_beliefs"][kind]
        assert (belief["forecast_tokens"], belief["expected_tokens"], belief["spread"], belief["prior_spread"]) == (0., 0., 0., 0.)
        assert belief["centre"] is None and belief["prior_centre"] is None
    # Only input and output enter E[K] and Var(K), although every kind has a price.
    assert row["needed_credit_usd"] == pytest.approx(3.00)
    assert row["credit_variance"] == pytest.approx(19.06, abs=.005)

    # A stated range of [0, 0] is still exactly zero; a non-zero upper end is not.
    stated = deepcopy(WORKED)
    stated["token_usage_ci"].update(cache_write=[0, 0], cache_read=[0, 500_000])
    _, _, other = one_cell(tmp_path / "stated", stated, candidate={"rate_card": card(model=MODEL, rates=RATES)})
    assert other["token_beliefs"]["cache_write"]["expected_tokens"] == 0.
    possible = other["token_beliefs"]["cache_read"]
    assert possible["prior_spread"] == pytest.approx(math.log(500_000) / 3.92) and possible["expected_tokens"] > 0
    assert other["needed_credit_usd"] > row["needed_credit_usd"]
    assert other["credit_variance"] > row["credit_variance"]
    load(engine.config, result["decision_id"])


def usage_run(tmp_path, engine, *, completed=True, seconds=600.):
    """Record one run of the cell at weight 0.5 and attach its owned usage: 60 input, 20 output, 40 cached."""
    record(engine, "measured", seconds, .5)
    manifest = artifact(tmp_path, codex_events(), execution_id="measured", model=MODEL, completed=completed)
    engine.attach_usage("measured", manifest, id="measured-usage")


@pytest.mark.parametrize("usage", ["complete", "partial", "none"])
def test_run_with_complete_execution_usage_moves_the_kinds_it_records(tmp_path, usage):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    cells = [{"model": MODEL, "effort": "high", "rate_card": card(model=MODEL, rates=RATES)}]
    row_forecast = by_kind(counts(input=6000, output=2000, cache_read=4000))
    decide(engine, cells, {(MODEL, "high"): row_forecast})
    if usage == "none":
        record(engine, "measured", 600., .5)
    else:
        usage_run(tmp_path, engine, completed=usage == "complete")
    _, _, result = decide(engine, cells)
    row = result["cells"][0]
    observed = result["semantic"]["duration_observations"][0]
    candidate = result["semantic"]["evidence_candidates"]["candidates"][0]
    # The scored candidate carries what the run recorded, and the reader's row repeats it.
    assert (observed["tokens"], observed["token_usage"]) == (candidate["tokens"], candidate["token_usage"])
    weight = observed["weight"]
    assert weight == pytest.approx(.5, rel=1e-4)
    assert weight == result["semantic"]["outcome_observations"][0]["weight"]
    # Active time is recorded either way and its belief is updated.
    assert row["active_time"]["evidence_weight"] == weight
    recorded = {"input": 60, "output": 20, "cache_read": 40}
    for kind, count in recorded.items():
        belief = row["token_beliefs"][kind]
        if usage != "complete":
            assert observed["token_usage"] is None
            assert (belief["centre"], belief["spread"], belief["evidence_weight"]) == (
                belief["prior_centre"], belief["prior_spread"], 0.)
            assert belief["expected_tokens"] == pytest.approx(row_forecast["token_usage"][kind])
            continue
        assert observed["token_usage"] == counts(**recorded)
        # The token belief counts the run at the same weight as the active-time belief.
        assert belief["evidence_weight"] == weight
        variance = 1 / (1 / belief["prior_spread"] ** 2 + weight / TAU ** 2)
        assert belief["spread"] == pytest.approx(math.sqrt(variance))
        assert belief["centre"] == pytest.approx(
            belief["prior_centre"] + variance * weight * (math.log(count) - belief["prior_centre"]) / TAU ** 2)
        assert belief["expected_tokens"] == pytest.approx(math.exp(belief["centre"] + variance / 2))
        assert belief["expected_tokens"] < row_forecast["token_usage"][kind]
    # A kind forecast at zero stays exactly zero whatever a run records.
    assert row["token_beliefs"]["cache_write"]["expected_tokens"] == 0.
    if usage == "complete":
        assert row["needed_credit_usd"] < .03
    else:
        assert row["needed_credit_usd"] == pytest.approx((6000 * 2 + 2000 * 10 + 4000 * .2) / 1e6)
    load(engine.config, result["decision_id"])


def test_cell_priced_per_total_token_has_one_belief_updated_from_recorded_tokens(tmp_path):
    engine, result, row = one_cell(tmp_path, forecast(tokens=10_000), candidate={"usd_per_mtok": 10.})
    assert list(row["token_beliefs"]) == ["tokens"]
    belief = row["token_beliefs"]["tokens"]
    assert belief["prior_spread"] == pytest.approx(MIN_SPREAD) and belief["stated_spread"] == 0.
    assert row["needed_credit_usd"] == pytest.approx(.1)
    # One belief at the minimum spread: sigma_K is that spread.
    assert row["credit_p80_usd"] == pytest.approx(.1 * math.exp(.8416 * MIN_SPREAD - MIN_SPREAD ** 2 / 2))
    assert row["credit_variance"] == pytest.approx(.01 * math.expm1(MIN_SPREAD ** 2))
    load(engine.config, result["decision_id"])

    # A run that records total tokens moves the one belief; a run that records none does not.
    for name, options, moved in (("counted", {"tokens": 1000}, True), ("uncounted", {}, False)):
        _, later, updated = one_cell(tmp_path / name, forecast(tokens=10_000), candidate={"usd_per_mtok": 10.},
                                     runs=[("run", 600., .5, options)])
        observed = later["semantic"]["duration_observations"][0]
        assert (observed["tokens"], observed["token_usage"]) == (options.get("tokens"), None)
        total = updated["token_beliefs"]["tokens"]
        assert total["evidence_weight"] == (observed["weight"] if moved else 0.)
        assert (updated["needed_credit_usd"] < .09) == moved
        assert updated["active_time"]["evidence_weight"] == pytest.approx(.5, rel=1e-4)


def test_usage_by_kind_on_a_per_total_token_cell_gives_each_kind_a_belief_at_the_one_rate(tmp_path):
    usage = counts(input=6000, output=1000, cache_read=3000)
    engine, result, row = one_cell(tmp_path, by_kind(usage, token_usage_ci={"output": [100, 10_000]}),
                                   candidate={"usd_per_mtok": 10.})
    assert set(row["token_beliefs"]) == set(counts())
    assert row["token_beliefs"]["output"]["prior_spread"] == pytest.approx(math.log(100) / 3.92)
    assert row["needed_credit_usd"] == pytest.approx(10. * 10_000 / 1e6)
    # The wider output belief raises the variance above the single-belief value, and the
    # 80% point is that of the log-normal with this average and variance.
    assert row["credit_variance"] > .01 * math.expm1(MIN_SPREAD ** 2)
    sigma_sq = math.log1p(row["credit_variance"] / row["needed_credit_usd"] ** 2)
    assert row["credit_p80_usd"] == pytest.approx(.1 * math.exp(.8416 * math.sqrt(sigma_sq) - sigma_sq / 2))
    load(engine.config, result["decision_id"])


def test_session_cost_follows_credit_and_no_budget_always_fits(tmp_path):
    engine, result, row = one_cell(tmp_path, forecast(tokens=10_000), candidate={"usd_per_mtok": 20., "cost_per_credit": .1})
    assert row["needed_credit_usd"] == pytest.approx(.2)
    assert row["session_cost_usd"] == pytest.approx(row["needed_credit_usd"] * .1)
    assert row["C_run"] == row["session_cost_usd"]
    assert row["available_credit_usd"] is None and row["credit_ok"] is True
    assert row["credit_p80_usd"] > row["needed_credit_usd"]

    # An unpriced cell keeps null credit fields, still holds its beliefs, and always fits.
    _, _, unpriced = one_cell(tmp_path / "unpriced", forecast(tokens=10_000))
    for field in ("needed_credit_usd", "credit_variance", "credit_p80_usd", "session_cost_usd", "C_run"):
        assert unpriced[field] is None
    assert unpriced["credit_ok"] is True
    assert unpriced["token_beliefs"]["tokens"]["expected_tokens"] == pytest.approx(10_000)
    assert unpriced["active_time"]["expected_seconds"] == pytest.approx(3600)


def test_a_stated_range_of_any_width_keeps_every_number_finite(tmp_path):
    usage = counts(input=6000, output=2000)
    row = by_kind(usage, seconds_ci=[0, 1e300], token_usage_ci={"input": [0, 1e300], "output": [2000, 1e200]})
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    request_id, response, result = decide(engine, priced_pair(1., 2.), {(MODEL, "low"): row, (MODEL, "high"): row})
    receipt = load(engine.config, result["decision_id"])
    assert json.loads(canonical(receipt)) == receipt                    # no infinity or NaN anywhere
    for stored in receipt["selection"]["cells"]:
        assert stored["active_time"]["spread"] == pytest.approx(math.log(1e300) / 3.92)
        assert stored["T"] == pytest.approx(1.) and stored["needed_credit_usd"] == pytest.approx(.032)
        # So wide a belief has its 80% point far below its average; the check falls back to the average.
        assert stored["credit_p80_usd"] < stored["needed_credit_usd"] and stored["credit_ok"]
    assert all(math.isfinite(c["cost_sd"]) for c in receipt["semantic"]["sampling"]["cells"])
    assert all(math.isfinite(engine.decide(request_id, response, seed=s)["cells"][0]["sampled_J"]) for s in range(20))


# --- 5. The cost spread and the draw keep their shape ---------------------------------------

def timed_engine(directory):
    """A risk class with no failure costs: the cost to an accepted result is run cost plus priced time."""
    directory.mkdir()
    (directory / "howhard.toml").write_text(
        'on_cap_unmet = "flag-review"\n[classes.timed]\nC_loud = 0.0\nC_silent = 0.0\n'
        'lambda_per_hour = 3600.0\nepsilon = 0.5\n[task_types]\ntask = "timed"\n')
    return Engine(directory=directory, retrieval=Scripted())


@pytest.mark.parametrize("stated", [{}, {"seconds_ci": [60, 216000]}])
def test_cost_spread_time_term_is_the_belief_variance_and_run_cost_is_exact(tmp_path, stated):
    engine = timed_engine(tmp_path / "store")
    cells = [{"model": MODEL, "effort": "high", "usd_per_mtok": 10.}]
    _, _, result = decide(engine, cells, {(MODEL, "high"): forecast(3600, tokens=10_000, **stated)})
    row, sampled = result["cells"][0], result["semantic"]["sampling"]["cells"][0]
    assert "duration" not in sampled
    spread = row["active_time"]["spread"]
    assert spread == pytest.approx(math.log(3600) / 3.92 if stated else MIN_SPREAD)
    # T and C_run are the beliefs' averages whatever the stated width.
    assert row["T"] == pytest.approx(1.) and row["C_run"] == pytest.approx(.1)
    d = 1 / row["expected_attempts"]
    assert row["J"] == pytest.approx((.1 + 3600 * 1.) / d)
    # With no failure costs the rates enter through retries alone: (J / (1 - P_loud))^2 * Var(theta_l).
    loud = sampled["outcome_beliefs"]["loud"]
    outcome_variance = (row["J"] / d) ** 2 * reference_belief(
        loud["prediction"], loud["stated_spread"], *loud["counts"], 5., 16.)["variance"]
    time_variance = (3600 / d) ** 2 * row["T"] ** 2 * math.expm1(spread ** 2)
    assert sampled["cost_sd"] == pytest.approx(math.sqrt(outcome_variance + time_variance))


def test_draw_takes_active_time_from_the_belief_after_the_runs(tmp_path):
    engine = timed_engine(tmp_path / "store")
    # An unpriced cell: the sampled cost is priced time alone, one dollar per second.
    cells = [{"model": MODEL, "effort": "high"}]
    decide(engine, cells, {(MODEL, "high"): {**forecast(3600), "clean": 1., "loud": 0., "silent": 0.}})
    record(engine, "earlier", 1800., .5)
    record(engine, "later", 5400., .3)
    request_id, response, first = decide(engine, cells)
    belief = first["cells"][0]["active_time"]
    assert belief["centre"] == pytest.approx(7.792, abs=5e-4) and belief["spread"] == pytest.approx(.719, abs=5e-4)
    draws = [math.log(engine.decide(request_id, response, seed=seed)["cells"][0]["sampled_J"]) for seed in range(300)]
    # ln T ~ Normal(m', s'^2) with m' = 7.792 and s' = 0.719; loud retries add a negligible factor.
    assert statistics.fmean(draws) == pytest.approx(belief["centre"], abs=.15)
    assert statistics.stdev(draws) == pytest.approx(belief["spread"], abs=.12)
    assert len(set(draws)) == len(draws)


# --- 6. Configuration and the receipt -------------------------------------------------------

def test_sampling_keys_are_optional_configurable_and_stored_in_the_policy(tmp_path, monkeypatch, capsys):
    engine, result, row = one_cell(tmp_path / "defaults", forecast(3600))
    receipt = load(engine.config, result["decision_id"])
    defaults = {"model_worth": 5., "task_worth_amount": 1., "duration_miss": 1.3, "scatter_share": .5}
    for policy in (receipt["selection"]["sampling_policy"], receipt["semantic"]["sampling"]["policy"]):
        assert {k: policy[k] for k in defaults} == defaults

    configured = tmp_path / "configured"
    configured.mkdir()
    (configured / "howhard.toml").write_text(
        'on_cap_unmet = "flag-review"\n[sampling]\nmodel_worth = 3\ntask_worth_amount = 1.5\n'
        'duration_miss = 0.8\nscatter_share = 0.25\n[task_types]\ntask = "standard"\n')
    other = Engine(directory=configured, retrieval=Scripted())
    _, _, changed = decide(other, [{"model": MODEL, "effort": "high"}], {(MODEL, "high"): forecast(3600)})
    belief = changed["cells"][0]["active_time"]
    assert (belief["worth"], belief["N_start"], belief["D"]) == (3., pytest.approx(1.), .8)
    assert belief["prior_spread"] == pytest.approx(.8 * math.sqrt(.25) / math.sqrt(1.))
    stored = load(other.config, changed["decision_id"])["selection"]["sampling_policy"]
    assert (stored["model_worth"], stored["task_worth_amount"], stored["duration_miss"], stored["scatter_share"]) == (3, 1.5, .8, .25)

    monkeypatch.chdir(configured)
    for setting in ("model_worth = 0", "task_worth_amount = -1", "duration_miss = 0", "scatter_share = 0",
                    "scatter_share = 1.5", 'duration_miss = "wide"'):
        (configured / "howhard.toml").write_text(f'on_cap_unmet = "flag-review"\n[sampling]\n{setting}\n')
        assert main(["paths"]) == 2
        assert "sampling" in capsys.readouterr().err


def budget_decision(tmp_path):
    """Two priced cells with earlier runs, ranges and budgets: every belief and credit number is in play."""
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    row = by_kind(counts(input=6000, output=2000, cache_read=4000), seconds=900,
                  seconds_ci=[100, 20_000], token_usage_ci={"output": [200, 20_000]})
    decide(engine, priced_pair(1., 2.), {(MODEL, "low"): row, (MODEL, "high"): row})
    usage_run(tmp_path, engine)
    _, _, result = decide(engine, priced_pair(1., 2.))
    assert all(c["credit_ok"] for c in result["cells"])
    measured = cell(result, MODEL, "high")
    assert measured["active_time"]["evidence_weight"] > 0 and measured["token_beliefs"]["input"]["evidence_weight"] > 0
    assert measured["active_time"]["stated_spread"] > MIN_SPREAD
    return engine, result


def tamper_cases():
    def high(receipt):
        return next(c for c in receipt["selection"]["cells"] if c["effort"] == "high")
    return [
        (lambda r: high(r)["active_time"].update(spread=high(r)["active_time"]["spread"] * 1.01), "active_time"),
        (lambda r: high(r)["active_time"].update(expected_seconds=high(r)["active_time"]["expected_seconds"] + 1), "active_time"),
        (lambda r: high(r)["active_time"].update(evidence_weight=0.), "active_time"),
        (lambda r: high(r)["active_time"].update(worth=6.), "active_time"),
        (lambda r: high(r)["active_time"].pop("prior_spread"), "active_time"),
        (lambda r: high(r)["token_beliefs"]["input"].update(centre=high(r)["token_beliefs"]["input"]["centre"] + .01), r"token_beliefs\.input"),
        (lambda r: high(r)["token_beliefs"]["cache_write"].update(expected_tokens=5.), r"token_beliefs\.cache_write"),
        (lambda r: high(r)["token_beliefs"].pop("output"), "token_beliefs"),
        (lambda r: high(r).update(token_beliefs={"tokens": high(r)["token_beliefs"]["input"]}), "token_beliefs"),
        (lambda r: high(r).pop("active_time"), "active_time"),
        (lambda r: high(r).update(credit_p80_usd=high(r)["credit_p80_usd"] * .5), "credit_p80_usd"),
        (lambda r: high(r).update(credit_p80_usd=None), "credit_p80_usd"),
        (lambda r: high(r).update(credit_variance=high(r)["credit_variance"] * 2), "credit_variance"),
        (lambda r: high(r).update(credit_ok=False), "credit_ok"),
        (lambda r: high(r).update(T=high(r)["T"] * 1.5), "expected time"),
        (lambda r: [high(r).update({k: v * 1.1 for k, v in high(r).items() if k in ("needed_credit_usd", "session_cost_usd", "C_run")})],
         "needed_credit_usd"),
        # The reader's rows are recomputed from the scored candidates, and the beliefs from those rows.
        (lambda r: r["semantic"]["duration_observations"][0]["token_usage"].update(input=6000), "duration_observations"),
        (lambda r: r["semantic"]["duration_observations"][0].update(tokens=5), "duration_observations"),
        (lambda r: r["semantic"]["duration_observations"][0].update(active_seconds=9000.), "duration_observations"),
        (lambda r: r["semantic"]["duration_observations"][0].pop("token_usage"), "duration_observations"),
        (lambda r: r["semantic"]["duration_observations"][0].update(weight=.9), "duration_observations"),
        (lambda r: r["semantic"]["evidence_candidates"]["candidates"][0]["token_usage"].update(input=6000),
         "duration_observations"),
        (lambda r: [row["token_usage"].update(input=6000) for row in (
            r["semantic"]["evidence_candidates"]["candidates"][0], r["semantic"]["duration_observations"][0])],
         r"token_beliefs\.input"),
        (lambda r: [row.update(active_seconds=9000.) for row in (
            r["semantic"]["evidence_candidates"]["candidates"][0], r["semantic"]["duration_observations"][0])],
         "active_time|capability"),
        (lambda r: r["semantic"]["evidence_candidates"]["candidates"][0].pop("tokens"), "candidate"),
        (lambda r: r["semantic"]["evidence_candidates"]["candidates"][0].update(tokens=-1), r"candidate\.tokens"),
        (lambda r: r["semantic"]["evidence_candidates"]["candidates"][0]["token_usage"].update(output=None), "token_usage"),
        (lambda r: r["semantic"]["evidence_candidates"]["candidates"][0]["token_usage"].pop("output"), "token_usage"),
        (lambda r: r["semantic"]["sampling"]["policy"].update(duration_miss=2.), "active_time"),
        (lambda r: [p.pop("scatter_share") for p in (r["semantic"]["sampling"]["policy"], r["selection"]["sampling_policy"])], "sampling"),
        (lambda r: [c.update(duration={}) for c in r["semantic"]["sampling"]["cells"]], "duration"),
    ]


@pytest.fixture(scope="module")
def stored_receipt(tmp_path_factory):
    """One loaded receipt for every alteration below; each test alters its own copy."""
    engine, result = budget_decision(tmp_path_factory.mktemp("stored-receipt"))
    return engine, load(engine.config, result["decision_id"])


def test_unaltered_receipt_loads_from_its_content(stored_receipt):
    engine, receipt = stored_receipt
    assert load(engine.config, deepcopy(receipt)) == receipt


@pytest.mark.parametrize("index", range(len(tamper_cases())))
def test_receipt_with_an_altered_belief_or_credit_number_is_refused(stored_receipt, index):
    engine, receipt = stored_receipt
    receipt = deepcopy(receipt)
    change, match = tamper_cases()[index]
    change(receipt)
    with pytest.raises(HowHardError, match=match):
        load(engine.config, receipt)


def test_receipt_whose_credit_check_ignores_the_80_percent_point_is_refused(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    _, _, result = decide(engine, priced_pair(4.00, 5.00), {(MODEL, "low"): WORKED, (MODEL, "high"): WORKED})
    receipt = load(engine.config, result["decision_id"])
    short = next(c for c in receipt["selection"]["cells"] if c["effort"] == "low")
    assert short["needed_credit_usd"] < short["available_credit_usd"] < short["credit_p80_usd"]
    short["credit_ok"] = True
    with pytest.raises(HowHardError, match="credit_ok"):
        load(engine.config, receipt)


# --- 7. Older receipts and the command line -------------------------------------------------

def test_v7_to_v10_receipts_keep_their_own_time_and_credit_rules(tmp_path, monkeypatch, capsys):
    from legacy_receipts import to_v7, to_v8, to_v9, to_v10
    from howhard.success import calibrate_duration
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    row = forecast(900, tokens=10_000, seconds_ci=[100, 20_000])
    cells = [{"model": MODEL, "effort": effort, "usd_per_mtok": 10.} for effort in ("low", "high")]
    decide(engine, cells, {(MODEL, "low"): row, (MODEL, "high"): row})
    usage_run(tmp_path, engine, seconds=7200.)
    _, _, result = decide(engine, cells)
    current = load(engine.config, result["decision_id"])
    assert current["format"] == "howhard-decision-v11"
    measured = cell(result, MODEL, "high")
    assert measured["active_time"]["evidence_weight"] == pytest.approx(.5, rel=1e-4)

    ids = {}
    for name, build in (("v10", to_v10), ("v9", to_v9), ("v8", to_v8), ("v7", to_v7)):
        legacy = build(current)
        ids[name] = digest(legacy)
        with locked(engine.config):
            put(engine.config, "decisions", ids[name], canonical(legacy) + b"\n")
        loaded = load(engine.config, ids[name])
        assert loaded == legacy and loaded["format"] == f"howhard-decision-{name}"
        old = next(c for c in loaded["selection"]["cells"] if c["effort"] == "high")
        assert not {"active_time", "token_beliefs", "credit_variance", "credit_p80_usd"} & old.keys()
        assert not {"model_worth", "task_worth_amount", "duration_miss", "scatter_share"} & loaded["selection"]["sampling_policy"].keys()
        assert old["needed_credit_usd"] == pytest.approx(.1)
        if name != "v7":
            # The frozen calibration, not the belief, prices the older receipt's time.
            family = old["family"]
            expected = calibrate_duration(900, (family, "high"), loaded["semantic"]["duration_observations"])
            assert old["T"] == pytest.approx(expected["expected_seconds"] / 3600)
            assert old["T"] != pytest.approx(measured["T"], rel=1e-3)
            assert all("tokens" not in o and "token_usage" not in o for o in loaded["semantic"]["duration_observations"])
            assert all("duration" in c for c in loaded["semantic"]["sampling"]["cells"])

    # Observe through each receipt, replay every forecast and report.
    monkeypatch.chdir(tmp_path)
    for name in ("v10", "v9", "v8"):
        engine.observe_decision(ids[name], "clean", id=f"from-{name}", active_seconds=10.)
    assert main(["observe", ids["v7"], "clean", "--id", "from-v7", "--json"]) == 0
    capsys.readouterr()
    engine.observe_decision(result["decision_id"], "clean", id="from-v11", active_seconds=10.)
    engine.refresh()
    beliefs = engine.memory_beliefs()
    assert beliefs["unscored_runs"] == 0
    assert next(e for e in beliefs["experts"] if e["key"] == "llm")["evidence_count"] == 5
    stats = report(engine)
    assert stats["unsupported_receipts"] == [] and stats["missing_receipts"] == []
    assert stats["forecast_metrics"]["llm"]["runs"] == 5

    # Each version requires its own shape.
    v10 = load(engine.config, ids["v10"])
    high = lambda r: next(c for c in r["selection"]["cells"] if c["effort"] == "high")
    for change, match in (
            (lambda r: high(r).update(active_time=measured["active_time"]), "unknown keys .'active_time'"),
            (lambda r: high(r).update(credit_p80_usd=.14), "unknown keys .'credit_p80_usd'"),
            (lambda r: r["selection"]["sampling_policy"].update(duration_miss=1.3), "unknown keys .'duration_miss'"),
            (lambda r: r["semantic"]["duration_observations"][0].update(tokens=120, token_usage=None), "unknown keys"),
            (lambda r: [c.pop("duration") for c in r["semantic"]["sampling"]["cells"]], "duration"),
            (lambda r: high(r).update(T=measured["T"]), "expected time")):
        bad = deepcopy(v10)
        change(bad)
        with pytest.raises(HowHardError, match=match):
            load(engine.config, bad)
    for change, match in (
            (lambda r: [high(r).pop(k) for k in ("active_time", "token_beliefs")], "active_time"),
            (lambda r: high(r).update(T=high(v10)["T"]), "expected time")):
        bad = deepcopy(current)
        change(bad)
        with pytest.raises(HowHardError, match=match):
            load(engine.config, bad)


def test_command_line_reports_the_beliefs_and_the_80_percent_credit_point(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps([{"model": MODEL, "effort": "high", "usd_per_mtok": 10.,
                                 "credit_budget": {"available_credit_usd": 1.}}]))
    for command in ("summary", "pick"):
        assert main([command, "task", "--text", TASK, "--candidates", str(path), "--json"]) == 0
        row = json.loads(capsys.readouterr().out)["cells"][0]
        # The fixture reader forecasts 10,000 tokens and 30 active seconds.
        assert row["needed_credit_usd"] == pytest.approx(.1)
        assert row["credit_p80_usd"] == pytest.approx(.1 * math.exp(.8416 * MIN_SPREAD - MIN_SPREAD ** 2 / 2))
        assert row["credit_ok"] is True
        assert row["active_time"]["expected_seconds"] == pytest.approx(30)
        assert row["token_beliefs"]["tokens"]["expected_tokens"] == pytest.approx(10_000)
    assert main(["pick", "task", "--text", TASK, "--candidates", str(path), "--seed", "1"]) == 0
    shown = capsys.readouterr().out
    assert "needed_credit_usd (credit)  credit_p80_usd (80% credit)" in shown
    assert "0.1000  0.1000  0.1406" in shown
