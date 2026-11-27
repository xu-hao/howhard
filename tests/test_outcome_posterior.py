"""Per-cell outcome beliefs on the log-odds scale and overlap-gated Thompson sampling (#103, #133).

A cell holds one belief for its loud rate and one for its accepted-defect share. Every
expected number below comes from ``reference_belief``, the formulas written out in plain
Python, or from the worked examples of the contract. Every fixture is synthetic and owns
its state.
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json
import math
import statistics

import pytest

from conftest import averaged_cost, own_runs_only, row
from howhard import Engine
from howhard.context import canonical, digest
from howhard.decisions import load
from howhard.payload_store import put
from howhard.report import report
from howhard.storage import locked
from howhard.prediction import assemble, by_cell
from howhard.success import DEFAULTS, merged_forecast, outcome_beliefs, uniforms
from howhard.validation import HowHardError
from legacy_receipts import to_v7, to_v8, to_v9, to_v10
from test_capability_filter import SimilarityDouble, filtering

FIELDS = ("P_clean", "P_loud", "P_silent", "P_silent_accepted")
RANGED = ("P_loud", "P_silent_accepted")  # The two rates that have a belief of their own.
WORTHS = {"loud": (5., 16.), "silent": (5., 32.)}  # (model_worth, the rate's task worth)
EPSILON = .10  # The packaged silent limit of the standard risk class.


def logit(p):
    return math.log(p / (1 - p))


@lru_cache(maxsize=None)
def reference_belief(prediction, stated_spread=0., n_yes=0., n_no=0., worth=5., task_worth=32.):
    """One belief from the contract's formulas alone, independent of the package's grid code."""
    xs = [-16 + .01 * j for j in range(2501)]
    rates = [1 / (1 + math.exp(-x)) for x in xs]
    n_start = 1 / (1 / worth + 1 / task_worth)
    spread = min(1.869, max(stated_spread, 1 / math.sqrt(prediction * (1 - prediction)) / math.sqrt(n_start)))

    def bell(m):
        q = [math.exp(-(x - m) ** 2 / (2 * spread ** 2)) for x in xs]
        total = math.fsum(q)
        return [v / total for v in q]
    lo, hi = -25., 25.
    for _ in range(60):
        mid = (lo + hi) / 2
        if math.fsum(q * r for q, r in zip(bell(mid), rates)) < prediction:
            lo = mid
        else:
            hi = mid
    centre = (lo + hi) / 2
    logs = [math.log(q) + n_yes * math.log(r) + n_no * math.log(1 - r) if q > 0 else -math.inf
            for q, r in zip(bell(centre), rates)]
    peak = max(logs)
    b = [math.exp(v - peak) for v in logs]
    total = math.fsum(b)
    b = [v / total for v in b]

    def point(q):
        running = 0.
        for share, rate in zip(b, rates):
            running += share
            if running >= q:
                return rate
        return rates[-1]
    mean = math.fsum(share * rate for share, rate in zip(b, rates))
    return {"n_start": n_start, "spread": spread, "centre": centre, "mean": mean,
            "lower": point(.025), "upper": point(.975), "point": point,
            "variance": math.fsum(share * rate ** 2 for share, rate in zip(b, rates)) - mean ** 2,
            "pi": lambda epsilon: math.fsum(share for share, rate in zip(b, rates) if rate <= epsilon),
            "attempts": math.fsum(share * (1 + math.exp(x)) for share, x in zip(b, xs)),
            "attempts_sq": math.fsum(share * (1 + math.exp(x)) ** 2 for share, x in zip(b, xs)),
            "odds": math.fsum(share * math.exp(x) for share, x in zip(b, xs))}


class RecentDouble:
    """Absolute-relevance fixture that retrieves the most recent executions first."""
    @property
    def reranker(self):
        return self

    def score(self, query, documents):
        return [12] * len(documents)

    def search(self, records, query, limit=8, **kwargs):
        evidence = [r for r in records if r["kind"] == "evidence"]
        # Input order is journal chronology; string IDs would put run-10 before run-2.
        chosen = list(reversed(evidence))[:limit]
        return {"records": deepcopy(chosen), "trace": {"seconds": 0, "selected": [
            {"key": r["key"], "score": 12} for r in chosen]}}


def spec(clean, loud, silent, **extra):
    return {"clean": clean, "loud": loud, "silent": silent, "tokens": 1000, "seconds": 60, **extra}


def forecasting(engine, candidates, specs, *, memory_forecasts=(), task_text="Repair rollback"):
    """Prepare one request up to its forecast stage; return its ID and the forecast response."""
    pending = engine.prepare("task", task_text, native_candidates=candidates)
    if pending["baseline_request"] is None:
        band = engine.decide(pending["request_id"], {"memory_filters": [], "reason": "Materialized prior"})
    else:
        keys_by_pair = {(c["model"], c["effort"]): c["key"] for c in pending["baseline_request"]["cells"]}
        order = [keys_by_pair[c["model"], c["effort"]] for c in candidates]
        band = engine.decide(pending["request_id"], filtering(pending["baseline_request"], 0, len(candidates) - 1, order))
    all_by_cell = {(c["family"], c["effort"]): c["model"] for c in band["request"]["cells"]}
    memory = [{"id": m["id"], "version": m["version"],
               "forecasts": [{"family": f, "effort": e, **s} for (f, e), model in all_by_cell.items()
                             for name, s in m["specs"].items() if name == model]} for m in memory_forecasts]
    if band["baseline_request"] is None:
        return band["request_id"], {"memory_forecasts": memory, "reason": "Synthetic posterior fixture"}
    by_cell = {(c["family"], c["effort"]): c["model"] for c in band["baseline_request"]["cells"]}
    rows = [{"family": f, "effort": e, **specs[m]} for (f, e), m in by_cell.items()]
    return band["request_id"], {"forecasts": rows, "memory_forecasts": memory, "reason": "Synthetic posterior fixture"}


def decide(engine, candidates, specs, *, seed=1, **options):
    request_id, response = forecasting(engine, candidates, specs, **options)
    return engine.decide(request_id, response, seed=seed)


def by_model(result):
    return {c["model"]: c for c in result["cells"]}


def calibration(result, model):
    row = by_model(result)[model]
    return next(c for c in result["semantic"]["sampling"]["cells"] if (c["family"], c["effort"]) == (row["family"], row["effort"]))


def observe(engine, model, outcomes, prefix):
    for i, outcome in enumerate(outcomes):
        extra = {} if outcome == "clean" else {"cause": "capability"}
        engine.observe("task", model, "high", outcome, native=True, task_text="Repair rollback", id=f"{prefix}-{i}", **extra)


WORKER = [{"model": "fixture/worker", "effort": "high", "usd_per_mtok": 1.0}]


def beliefs(result, model):
    return calibration(result, model)["outcome_beliefs"]


def matches_reference(belief, rate, *, stated_spread=0., epsilon=EPSILON):
    """A stored belief equals the plain-Python reference at its own prediction and counts."""
    worth, task_worth = WORTHS[rate]
    expected = reference_belief(belief["prediction"], stated_spread, *belief["counts"], worth, task_worth)
    for field in ("n_start", "spread", "centre", "mean", "lower", "upper"):
        assert belief[field] == pytest.approx(expected[field], abs=1e-9), (rate, field)
    if rate == "silent":
        assert belief["pi"] == pytest.approx(expected["pi"](epsilon), abs=1e-9)
    else:
        assert belief["attempts"] == pytest.approx(expected["attempts"], rel=1e-9)
        assert belief["odds"] == pytest.approx(expected["odds"], rel=1e-9)
    return expected


# The contract's worked forecast: p = 0.10 for S with a stated range of 5% to 20%, and p = 0.20 for theta_l.
EXAMPLE = dict(clean=.72, loud=.20, silent=.08, silent_ci=[.04, .16], loud_ci=[.10, .35])


def test_the_worked_examples_of_the_start_reproduce_at_the_decision_boundary(tmp_path):
    engine = Engine(directory=tmp_path)
    # One hour of active time and a run cost of 2 dollars: 1000 tokens at 2000 dollars per million.
    worker = [{"model": "fixture/worker", "effort": "high", "usd_per_mtok": 2000.0}]
    result = decide(engine, worker, {"fixture/worker": {**EXAMPLE, "tokens": 1000, "seconds": 3600}}, seed=3)
    assert set(result["semantic"]["worlds"]) == {"llm"} and result["constants"]["epsilon"] == EPSILON
    assert result["semantic"]["interval_kind"] == "log-odds-grid"
    cell = calibration(result, "fixture/worker")
    assert set(cell) == {"family", "effort", "outcome_beliefs", "probabilities", "cost_sd", "sampled"}
    loud, silent = cell["outcome_beliefs"]["loud"], cell["outcome_beliefs"]["silent"]
    common = {"prediction", "stated_spread", "n_start", "spread", "centre", "counts", "mean", "lower", "upper"}
    assert set(silent) == common | {"pi"} and set(loud) == common | {"attempts", "odds"}

    # S: N_start 4.32, s_stated 0.40, sigma_1 / sqrt(N_start) = 1.60, centre -3.04.
    assert silent["prediction"] == pytest.approx(.10) and silent["counts"] == [0., 0.]
    assert silent["stated_spread"] == pytest.approx(.40, abs=5e-3)
    assert silent["n_start"] == pytest.approx(4.32, abs=5e-3)
    assert silent["spread"] == pytest.approx(1.60, abs=5e-3)
    assert silent["centre"] == pytest.approx(-3.04, abs=5e-3)
    # At the start E[S] = 0.100, pi = 0.70 and the 95% range is 0.2% to 52%.
    assert silent["mean"] == pytest.approx(.100, abs=5e-4)
    assert silent["pi"] == pytest.approx(.70, abs=5e-3)
    assert silent["lower"] == pytest.approx(.002, abs=5e-4) and round(100 * silent["upper"]) == 52
    # theta_l: N_start 3.81, s 1.28, centre -1.79; expected attempts 1.38 against 1.25 at the average rate.
    assert loud["prediction"] == pytest.approx(.20) and loud["n_start"] == pytest.approx(3.81, abs=5e-3)
    assert loud["spread"] == pytest.approx(1.28, abs=5e-3) and loud["centre"] == pytest.approx(-1.79, abs=5e-3)
    assert loud["attempts"] == pytest.approx(1.38, abs=5e-3) and loud["odds"] == pytest.approx(loud["attempts"] - 1)
    for rate, belief in (("loud", loud), ("silent", silent)):
        # The belief's average, not its median, equals the prediction.
        assert belief["mean"] == pytest.approx(belief["prediction"], abs=1e-9)
        assert belief["lower"] < belief["mean"] < belief["upper"]
        matches_reference(belief, rate, stated_spread=belief["stated_spread"])

    # The selection cell reads the two beliefs; P_clean and P_silent have no belief and no range.
    row = result["cells"][0]
    assert row["P_loud"] == {k: loud[k] for k in ("mean", "lower", "upper")}
    assert row["P_silent_accepted"] == {k: silent[k] for k in ("mean", "lower", "upper")}
    assert row["P_silent"] == {"mean": pytest.approx(.8 * .1)} and row["P_clean"] == {"mean": pytest.approx(.8 * .9)}
    assert cell["probabilities"] == [row["P_clean"]["mean"], row["P_loud"]["mean"], row["P_silent"]["mean"]]
    assert row["pi"] == silent["pi"] and row["tier"] == "on-trial"
    # run = 2, one hour at 25, C_loud = 50, C_silent = 1000. The cost is averaged over the beliefs:
    # 27 for each of E[R] attempts, 50 for each of E[R] - 1 loud failures and 1000 * E[S].
    constants = result["constants"]
    assert (constants["C_loud"], constants["C_silent"], constants["lambda_per_hour"]) == (50., 1000., 25.)
    assert row["session_cost_usd"] == pytest.approx(2.) and row["T"] == pytest.approx(1.)
    assert row["expected_attempts"] == loud["attempts"] > 1.25
    assert row["J"] == pytest.approx(27 * loud["attempts"] + 50 * (loud["attempts"] - 1) + 1000 * silent["mean"])
    assert row["J"] == pytest.approx(averaged_cost(result, row))
    # The cost at the two averages is 146.25; the curved retry factor makes the average over the belief dearer.
    assert row["J"] > 146.25 == pytest.approx((2 + 25 + 50 * .2 + 1000 * .08) / .8)
    # The exact spread: X = K + 25 T + 50 is what one attempt costs plus C_loud, and
    # Var(J) = E[X^2] E[R^2] - (E[X] E[R])^2 + C_silent^2 Var(S).
    reference = {rate: reference_belief(belief["prediction"], belief["stated_spread"], *belief["counts"], *WORTHS[rate])
                 for rate, belief in (("loud", loud), ("silent", silent))}
    mean_x = 2 + 25 + 50
    second_x = row["credit_variance"] + 25 ** 2 * 1. ** 2 * math.expm1(row["active_time"]["spread"] ** 2) + mean_x ** 2
    variance = (second_x * reference["loud"]["attempts_sq"] - (mean_x * reference["loud"]["attempts"]) ** 2
                + 1000 ** 2 * reference["silent"]["variance"])
    assert cell["cost_sd"] == pytest.approx(math.sqrt(variance)) and row["credit_variance"] > 0
    receipt = load(engine.config, result["decision_id"])
    assert receipt["format"] == "howhard-decision-v11" and receipt["semantic"]["outcome_observations"] == []
    assert receipt["semantic"]["sampling"]["cells"] == result["semantic"]["sampling"]["cells"]


@pytest.mark.parametrize("clean,loud,silent", [(.9, .07, .03), (.47, .40, .13), (.005, .99, .005), (1., 0., 0.),
                                               (0., 1., 0.), (0., 0., 1.), (.3, .1, .6), (.98, .019, .001)])
def test_with_no_counted_run_each_belief_averages_its_prediction(tmp_path, clean, loud, silent):
    result = decide(Engine(directory=tmp_path), WORKER, {"fixture/worker": spec(clean, loud, silent)})
    floored = [max(.0001, p) for p in (clean, loud, silent)]
    total = sum(floored)
    guesses = {"loud": floored[1] / total, "silent": floored[2] / (floored[0] + floored[2])}
    for rate, belief in beliefs(result, "fixture/worker").items():
        # One expert and no stated range: the prediction is the guess, held in [0.0001, 0.9998].
        assert belief["prediction"] == pytest.approx(min(.9998, max(.0001, guesses[rate])), abs=1e-12)
        assert belief["stated_spread"] == pytest.approx(0., abs=1e-7) and belief["counts"] == [0., 0.]
        assert belief["mean"] == pytest.approx(belief["prediction"], abs=1e-9)
        assert 0 < belief["lower"] <= belief["mean"] <= belief["upper"] < 1
    row = result["cells"][0]
    assert sum(row[field]["mean"] for field in ("P_clean", "P_loud", "P_silent")) == pytest.approx(1.)
    assert row["J"] == pytest.approx(averaged_cost(result, row))


def test_experts_merge_into_one_forecast_per_rate_and_disagreement_widens_it():
    # Two experts with weights 0.75 and 0.25 guess S at 10% and 25%, each with spread 0.4.
    def row(share, loud=.2):
        ends = [1 / (1 + math.exp(-(logit(share) + z * .4))) for z in (-1.96, 1.96)]
        return dict(clean=(1 - loud) * (1 - share), loud=loud, silent=(1 - loud) * share,
                    silent_ci=[(1 - loud) * end for end in ends])
    pair = ("worker", "high")
    panels = {"llm": {pair: row(.10)}, "memory": {pair: row(.25)}}
    experts = [{"key": "llm", "weight": .75}, {"key": "memory", "weight": .25}]
    merged = merged_forecast(panels, experts, pair)
    assert logit(merged["silent"]["prediction"]) == pytest.approx(-1.92, abs=5e-3)
    assert merged["silent"]["prediction"] == pytest.approx(.128, abs=5e-4)
    assert merged["silent"]["stated_spread"] == pytest.approx(.62, abs=5e-3)
    # The experts agree on the loud rate and state no range for it.
    assert merged["loud"]["prediction"] == pytest.approx(.2) and merged["loud"]["stated_spread"] == pytest.approx(0., abs=1e-7)
    for key in panels:
        alone = merged_forecast(panels, [{"key": key, "weight": 1.}], pair)["silent"]
        assert alone["stated_spread"] == pytest.approx(.4, abs=1e-9)
        assert alone["stated_spread"] < merged["silent"]["stated_spread"]

    # Counted runs: with n_c = 4 and n_s = 0, E[S] = 0.053 and pi = 0.85.
    example = {"llm": {pair: {**EXAMPLE, "tokens": 1000, "seconds": 3600}}}
    runs = [{"family": "worker", "effort": "high", "agent": None, "ticket": None, "outcome": "clean", "weight": 1.}] * 4
    forecaster = [{"key": "llm", "weight": 1.}]
    # With no run on record on other tasks, each belief starts from the merged forecast itself.
    values = by_cell(assemble({"kinds": {}, "cells": {}}, example, forecaster, [("cell", pair)], DEFAULTS))[pair]
    assert values["silent"]["prediction"] == merged_forecast(example, forecaster, pair)["silent"]["prediction"]
    start = outcome_beliefs(values, pair, [], DEFAULTS, epsilon=EPSILON)
    after = outcome_beliefs(values, pair, runs, DEFAULTS, epsilon=EPSILON)
    assert start["silent"]["mean"] == pytest.approx(.100, abs=5e-4) and start["silent"]["pi"] == pytest.approx(.70, abs=5e-3)
    assert after["silent"]["counts"] == [0., 4.] and after["loud"]["counts"] == [0., 4.]
    assert after["silent"]["mean"] == pytest.approx(.053, abs=5e-4) and after["silent"]["pi"] == pytest.approx(.85, abs=5e-3)
    # The start itself is untouched by the runs: the forecast keeps its weight of N_start runs,
    # for four counted runs and for four hundred.
    many = outcome_beliefs(values, pair, runs * 96 + [{**runs[0], "outcome": "silent"}] * 16, DEFAULTS,
                           epsilon=EPSILON)
    assert many["silent"]["counts"] == [16., 384.] and many["loud"]["counts"] == [0., 400.]
    for counted in (after, many):
        for rate in ("loud", "silent"):
            for field in ("prediction", "stated_spread", "n_start", "spread", "centre"):
                assert counted[rate][field] == start[rate][field]
            matches_reference(counted[rate], rate, stated_spread=counted[rate]["stated_spread"])
    # The counted share is 16 / 400 = 0.04; the forecast of 0.10 still pulls the average towards it.
    assert .04 < many["silent"]["mean"] < .10


def test_a_stated_range_can_widen_the_start_but_never_narrow_it(tmp_path):
    def decided(name, **ranges):
        result = decide(Engine(directory=tmp_path / name), WORKER, {"fixture/worker": spec(.72, .20, .08, **ranges)})
        return result, beliefs(result, "fixture/worker")
    bare_result, bare = decided("bare")
    # Ranges on S of 0.4% to 75% and on theta_l of 1% to 90%: wider than a record worth N_start runs.
    wide_result, wide = decided("wide", silent_ci=[.0032, .6], loud_ci=[.01, .9])
    narrow_result, narrow = decided("narrow", silent_ci=[.0799, .0801], loud_ci=[.1999, .2001])
    flat_result, flat = decided("flat", silent_ci=[0., 1.], loud_ci=[0., 1.])

    stated = {"silent": (logit(.75) - logit(.004)) / 3.92, "loud": (logit(.9) - logit(.01)) / 3.92}
    for rate in ("loud", "silent"):
        assert bare[rate]["stated_spread"] == pytest.approx(0., abs=1e-7)
        assert bare[rate]["spread"] == pytest.approx(reference_belief(bare[rate]["prediction"], 0., 0., 0., *WORTHS[rate])["spread"])
        assert wide[rate]["stated_spread"] == pytest.approx(stated[rate])
        assert wide[rate]["spread"] == pytest.approx(stated[rate])
        assert bare[rate]["spread"] < wide[rate]["spread"] < 1.869
        assert wide[rate]["upper"] - wide[rate]["lower"] > bare[rate]["upper"] - bare[rate]["lower"]
        assert wide[rate]["lower"] < bare[rate]["lower"]
        # The average stays the prediction however wide the start is.
        assert wide[rate]["mean"] == pytest.approx(bare[rate]["mean"], abs=1e-9)
        # A narrow range states a small spread, and the start is exactly the bare one.
        assert 0 < narrow[rate]["stated_spread"] < .01
        assert {k: v for k, v in narrow[rate].items() if k != "stated_spread"} == {
            k: v for k, v in bare[rate].items() if k != "stated_spread"}
        # [0, 1] is the flat belief.
        assert flat[rate]["spread"] == 1.869 and flat[rate]["stated_spread"] > 1.869
        assert flat[rate]["mean"] == pytest.approx(bare[rate]["mean"], abs=1e-9)
    for field in FIELDS:
        assert narrow_result["cells"][0][field] == bare_result["cells"][0][field]
    stored = load(Engine(directory=tmp_path / "wide").config, wide_result["decision_id"])
    assert stored["semantic"]["worlds"]["llm"][0]["silent_ci"] == [.0032, .6]

    for idx, bad in enumerate(([.05, .2], [0., 1.2], [.1], "0-0.2", [.2, .01], [0., None])):
        with pytest.raises(HowHardError, match="silent_ci"):
            decide(Engine(directory=tmp_path / f"bad-{idx}"), WORKER, {"fixture/worker": spec(.9, .07, .03, silent_ci=bad)})
    with pytest.raises(HowHardError, match="loud_ci"):
        decide(Engine(directory=tmp_path / "bad-loud"), WORKER, {"fixture/worker": spec(.9, .07, .03, loud_ci=[.08, .1])})


def test_two_experts_that_disagree_give_a_wider_start_than_either_alone(tmp_path):
    # Worths of 1000 runs make the start as narrow as the forecast allows, so what the experts state shows.
    worth = '[task_types]\ntask = "standard"\n[sampling]\nmodel_worth = 2000.0\ntask_worth_silent = 2000.0\ntask_worth_loud = 2000.0\n'
    low, high = spec(.72, .20, .08), spec(.40, .20, .40)

    def engine(name):
        (tmp_path / name).mkdir()
        (tmp_path / name / "howhard.toml").write_text(worth)
        return Engine(tmp_path / name / "howhard.toml")
    alone = [beliefs(decide(engine(f"alone-{i}"), WORKER, {"fixture/worker": forecast}), "fixture/worker")
             for i, forecast in enumerate((low, high))]
    mixed_engine = engine("mixed")
    memory = mixed_engine.memory.add("Risk", "Silent defects are more common than forecast on rollback repairs")
    mixed = decide(mixed_engine, WORKER, {"fixture/worker": low},
                   memory_forecasts=[{**memory, "specs": {"fixture/worker": high}}])
    key = f"memory:{memory['id']}:{memory['version']}"
    assert set(mixed["semantic"]["worlds"]) == {"llm", key}
    weights = {e["key"]: e["weight"] for e in mixed["semantic"]["beliefs"]["experts"]}
    assert all(0 < w < 1 for w in weights.values())
    merged = beliefs(mixed, "fixture/worker")
    # The cell holds one belief per rate: m = sum w_k m_k and s_stated^2 = sum w_k (m_k - m)^2.
    guesses = {"llm": logit(.10), key: logit(.50)}
    centre = sum(weights[k] * guesses[k] for k in weights)
    stated = math.sqrt(sum(weights[k] * (guesses[k] - centre) ** 2 for k in weights))
    assert logit(merged["silent"]["prediction"]) == pytest.approx(centre)
    assert merged["silent"]["stated_spread"] == pytest.approx(stated) and stated > .5
    assert merged["silent"]["n_start"] == pytest.approx(1000.)
    assert merged["silent"]["spread"] == pytest.approx(stated)
    assert merged["silent"]["spread"] > max(belief["silent"]["spread"] for belief in alone)
    assert merged["silent"]["upper"] - merged["silent"]["lower"] > max(
        belief["silent"]["upper"] - belief["silent"]["lower"] for belief in alone)
    assert all(belief["silent"]["stated_spread"] == pytest.approx(0., abs=1e-7) for belief in alone)
    # They agree on the loud rate, which stays as certain as either alone.
    assert merged["loud"]["spread"] == pytest.approx(alone[0]["loud"]["spread"])
    row = mixed["cells"][0]
    assert row["P_silent_accepted"]["lower"] < row["P_silent_accepted"]["mean"] < row["P_silent_accepted"]["upper"]
    assert row["P_silent_accepted"]["mean"] == pytest.approx(merged["silent"]["prediction"], abs=1e-9)
    load(mixed_engine.config, mixed["decision_id"])


def test_without_discounts_counted_runs_change_only_their_own_cell_and_the_forecast_keeps_its_weight(tmp_path):
    candidates = [{"model": "fixture/unseen", "effort": "high", "usd_per_mtok": 1.0},
                  {"model": "fixture/seen", "effort": "high", "usd_per_mtok": 1.0}]
    specs = {"fixture/seen": spec(.6, .2, .2), "fixture/unseen": spec(.6, .2, .2)}
    for name in ("fresh", "history"):
        (tmp_path / name).mkdir()
        own_runs_only(tmp_path / name)
    fresh = decide(Engine(directory=tmp_path / "fresh", retrieval=SimilarityDouble()), candidates, specs)

    engine = Engine(directory=tmp_path / "history", retrieval=SimilarityDouble())
    observe(engine, "fixture/seen", ["clean"] * 8 + ["loud"] * 2 + ["silent"], "seen")
    # Many scored runs of an unrelated cell raise the llm expert's global evidence count
    # without crowding out fixture/seen's per-cell outcome history.
    other = decide(engine, [{"model": "fixture/other", "effort": "high", "usd_per_mtok": 1.0}],
                   {"fixture/other": spec(.84, .12, .04)})
    for i in range(30):
        engine.observe_decision(other["decision_id"], "clean", id=f"other-{i}")
    assert next(e for e in engine.memory_beliefs()["experts"] if e["key"] == "llm")["evidence_count"] >= 30
    result = decide(engine, candidates, specs)
    observed = result["semantic"]["outcome_observations"]
    seen_family = by_model(result)["fixture/seen"]["family"]
    assert {o["id"] for o in observed if o["family"] == seen_family} == {f"seen-{i}" for i in range(11)}
    assert not any(o["family"] == by_model(result)["fixture/unseen"]["family"] for o in observed)

    # The unseen cell is exactly as it was with no record at all.
    unseen, unseen_fresh = calibration(result, "fixture/unseen"), calibration(fresh, "fixture/unseen")
    assert unseen["outcome_beliefs"] == unseen_fresh["outcome_beliefs"]
    assert unseen["cost_sd"] == pytest.approx(unseen_fresh["cost_sd"])
    for field in (*FIELDS, "pi", "admission_risk", "tier"):
        assert by_model(result)["fixture/unseen"][field] == by_model(fresh)["fixture/unseen"][field]

    seen, seen_fresh = calibration(result, "fixture/seen"), calibration(fresh, "fixture/seen")
    counted = {outcome: math.fsum(o["weight"] for o in observed if o["outcome"] == outcome and o["family"] == seen_family)
               for outcome in ("clean", "loud", "silent")}
    assert counted["clean"] > 7 and counted["loud"] > 1.5 and counted["silent"] > .9
    assert by_model(result)["fixture/seen"]["outcome_counts"] == {
        "n_clean": pytest.approx(counted["clean"]), "n_loud": pytest.approx(counted["loud"]),
        "n_silent": pytest.approx(counted["silent"]), "runs": 11}
    assert by_model(result)["fixture/unseen"]["outcome_counts"] == {"n_clean": 0., "n_loud": 0., "n_silent": 0., "runs": 0}
    # S counts silent against clean runs; theta_l counts loud against clean and silent ones.
    assert seen["outcome_beliefs"]["silent"]["counts"] == pytest.approx([counted["silent"], counted["clean"]])
    assert seen["outcome_beliefs"]["loud"]["counts"] == pytest.approx(
        [counted["loud"], counted["clean"] + counted["silent"]])
    for rate in ("loud", "silent"):
        belief, start = seen["outcome_beliefs"][rate], seen_fresh["outcome_beliefs"][rate]
        # However many runs are counted, the start is the same bell curve worth N_start runs.
        for field in ("prediction", "stated_spread", "n_start", "spread", "centre"):
            assert belief[field] == start[field], (rate, field)
        expected = matches_reference(belief, rate)
        # The counted share is below the prediction, and the runs move the average below it too.
        assert belief["counts"][0] / sum(belief["counts"]) < belief["prediction"]
        assert belief["mean"] < start["mean"] == pytest.approx(belief["prediction"], abs=1e-9)
        assert expected["upper"] - expected["lower"] < start["upper"] - start["lower"]
    seen_row, unseen_row = by_model(result)["fixture/seen"], by_model(result)["fixture/unseen"]
    assert seen_row["P_clean"]["mean"] > .65 > unseen_row["P_clean"]["mean"]
    assert seen_row["pi"] > unseen_row["pi"] and seen["cost_sd"] < unseen["cost_sd"]
    assert seen_row["J"] == pytest.approx(averaged_cost(result, seen_row))
    load(engine.config, result["decision_id"])


RETIRED = "[sampling]\nprior_strength = 40.0\nmin_concentration = 3.0\nmax_concentration = 50.0\n"


def test_a_retired_sampling_key_loads_with_a_notice_and_changes_no_decision(tmp_path, capsys):
    task = '[task_types]\ntask = "standard"\n'
    candidates = [{"model": "fixture/a", "effort": "high", "usd_per_mtok": 1.0},
                  {"model": "fixture/b", "effort": "high", "usd_per_mtok": 2.0}]
    specs = {"fixture/a": spec(.80, .12, .08, silent_ci=[.01, .3]), "fixture/b": spec(.93, .05, .02)}
    results = {}
    for name, settings in (("plain", ""), ("retired", RETIRED)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "howhard.toml").write_text(task + settings)
        capsys.readouterr()
        engine = Engine(tmp_path / name / "howhard.toml")
        notices = capsys.readouterr().err.splitlines()
        if settings:
            # One line per retired key, on standard error.
            assert len(notices) == 3 and all("retired and ignored" in line for line in notices)
            assert [key for line in notices for key in ("prior_strength", "min_concentration", "max_concentration")
                    if f"sampling.{key} " in line] == ["prior_strength", "min_concentration", "max_concentration"]
        else:
            assert notices == []
        policy = engine.config.data["sampling"]
        assert not {"prior_strength", "min_concentration", "max_concentration"} & policy.keys()
        assert (policy["model_worth"], policy["task_worth_silent"], policy["task_worth_loud"]) == (5., 32., 16.)
        results[name] = [decide(engine, candidates, specs, seed=seed) for seed in range(6)]
        stored = load(engine.config, results[name][0]["decision_id"])
        # A v11 policy does not store the retired keys, in the selection or in the frozen draw.
        for frozen in (stored["selection"]["sampling_policy"], stored["semantic"]["sampling"]["policy"]):
            assert frozen == policy
    for plain, retired in zip(results["plain"], results["retired"]):
        for field in ("chosen", "seed", "cells", "selection", "exploration", "cap_met", "needs_review"):
            assert plain[field] == retired[field], field
        assert plain["semantic"]["sampling"] == retired["semantic"]["sampling"]

    # The three worths are optional, positive and refused by name otherwise.
    for index, (setting, key) in enumerate((("model_worth = 0", "model_worth"),
                                           ("task_worth_silent = -1.0", "sampling.task_worth_silent"),
                                           ('task_worth_loud = "16"', "sampling.task_worth_loud"),
                                           ("task_worth_loud = 0.0", "sampling.task_worth_loud"))):
        (tmp_path / f"refused-{index}").mkdir()
        (tmp_path / f"refused-{index}" / "howhard.toml").write_text(f"[sampling]\n{setting}\n")
        with pytest.raises(HowHardError, match=key):
            Engine(tmp_path / f"refused-{index}" / "howhard.toml")
    # A larger worth makes the forecast harder to move: the same start is narrower.
    (tmp_path / "worth").mkdir()
    (tmp_path / "worth" / "howhard.toml").write_text(task + "[sampling]\nmodel_worth = 20.0\n")
    heavy = decide(Engine(tmp_path / "worth" / "howhard.toml"), candidates, specs, seed=0)
    for rate, task_worth in (("loud", 16.), ("silent", 32.)):
        belief = beliefs(heavy, "fixture/b")[rate]
        assert belief["n_start"] == pytest.approx(1 / (1 / 20 + 1 / task_worth))
        assert belief["n_start"] > beliefs(results["plain"][0], "fixture/b")[rate]["n_start"]
    assert beliefs(heavy, "fixture/a")["loud"]["spread"] < beliefs(results["plain"][0], "fixture/a")["loud"]["spread"]


def test_the_same_request_record_and_seed_give_the_same_decision_and_sampled_values(tmp_path):
    candidates = [{"model": "fixture/a", "effort": "high", "usd_per_mtok": 1.0},
                  {"model": "fixture/b", "effort": "high", "usd_per_mtok": 2.0}]
    specs = {"fixture/a": spec(.86, .08, .06), "fixture/b": spec(.90, .06, .04)}

    def recorded(name):
        engine = Engine(directory=tmp_path / name, retrieval=SimilarityDouble())
        observe(engine, "fixture/a", ["clean", "clean", "loud"], "run")
        return engine
    engine = recorded("first")
    request_id, response = forecasting(engine, candidates, specs)
    first = engine.decide(request_id, response, seed=7)
    again = engine.decide(request_id, response, seed=7)
    assert first["decision_id"] == again["decision_id"]
    for field in ("chosen", "seed", "cells", "selection", "exploration", "why"):
        assert first[field] == again[field]
    sampled = lambda result: {c["model"]: c["sampled_J"] for c in result["cells"]}
    assert sampled(first) == sampled(again)
    assert first["semantic"]["sampling"] == again["semantic"]["sampling"]
    assert "sampled_expert" not in first["semantic"]
    assert any(sum(c["outcome_beliefs"]["loud"]["counts"]) > 2 for c in first["semantic"]["sampling"]["cells"])
    # The same record in another store, prepared a moment later, chooses the same cell.
    other = decide(recorded("second"), candidates, specs, seed=7)
    assert other["chosen"] == first["chosen"] and other["exploration"]["reason"] == first["exploration"]["reason"]
    assert sampled(other) == pytest.approx(sampled(first), rel=1e-4)
    # Other seeds draw other values from the same beliefs.
    others = [engine.decide(request_id, response, seed=seed) for seed in range(8)]
    assert len({tuple(sorted(sampled(result).items())) for result in others}) == 8
    # The seed moves the sampled values alone: the beliefs, the cost and its spread stay.
    frozen = lambda result: [{k: v for k, v in c.items() if k != "sampled"} for c in result["semantic"]["sampling"]["cells"]]
    assert all(frozen(result) == frozen(first) for result in others)
    assert all(result["cells"][0]["J"] == first["cells"][0]["J"] for result in others)

    # One cell: four uniform numbers in the order S, theta_l, active time, credit. Each rate is read
    # at the first grid point whose running total reaches its number.
    single = Engine(directory=tmp_path / "single")
    z = statistics.NormalDist().inv_cdf
    for seed in (0, 1, 2, 11):
        result = decide(single, WORKER, {"fixture/worker": spec(.72, .20, .08)}, seed=seed)
        u_share, u_loud, u_time, u_credit = uniforms(seed, 4)
        share = reference_belief(.10, 0., 0., 0., *WORTHS["silent"])["point"](u_share)
        theta = reference_belief(.20, 0., 0., 0., *WORTHS["loud"])["point"](u_loud)
        row, constants = result["cells"][0], result["constants"]
        hours = math.exp(row["active_time"]["centre"] + row["active_time"]["spread"] * z(u_time)) / 3600
        tokens = row["token_beliefs"]["tokens"]
        run = math.exp(tokens["centre"] + tokens["spread"] * z(u_credit)) * 1. / 1e6
        expected = ((run + constants["lambda_per_hour"] * hours) / (1 - theta) + constants["C_loud"] * theta / (1 - theta)
                    + constants["C_silent"] * share)
        assert row["sampled_J"] == pytest.approx(expected)
        sampled = result["semantic"]["sampling"]["cells"][0]["sampled"]
        assert sampled == {"S": pytest.approx(share), "u": pytest.approx(logit(theta)),
                           "seconds": pytest.approx(3600 * hours), "credit": pytest.approx(run)}


def test_a_v11_receipt_stores_both_beliefs_and_refuses_one_altered_number(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    candidates = [{"model": "fixture/a", "effort": "high", "usd_per_mtok": 1.0},
                  {"model": "fixture/b", "effort": "high", "usd_per_mtok": 2.0}]
    observe(engine, "fixture/a", ["clean", "clean", "silent", "loud"], "run")
    memory = engine.memory.add("Risk", "Silent defects are more common than forecast on rollback repairs")
    result = decide(engine, candidates, {"fixture/a": spec(.80, .12, .08, silent_ci=[.02, .3]), "fixture/b": spec(.93, .05, .02)},
                    seed=5, memory_forecasts=[{**memory, "specs": {"fixture/a": spec(.7, .12, .18)}}])
    receipt = load(engine.config, result["decision_id"])
    assert receipt["format"] == "howhard-decision-v11" and digest(receipt) == result["decision_id"]
    drawn = receipt["semantic"]["sampling"]["cells"]
    assert {(c["family"], c["effort"]) for c in drawn} == {(c["family"], c["effort"]) for c in receipt["semantic"]["worlds"]["llm"]}
    assert len(drawn) == 2 and any(sum(c["outcome_beliefs"]["silent"]["counts"]) > 0 for c in drawn)
    for cell in drawn:
        assert set(cell["outcome_beliefs"]) == {"loud", "silent"}
        assert "outcome_posterior" not in cell and "concentration" not in cell

    def refused(tamper, label):
        bad = deepcopy(receipt)
        tamper(bad)
        with pytest.raises(HowHardError):
            load(engine.config, bad)
            pytest.fail(f"{label}: accepted")

    def nudge(container, key):
        container[key] = container[key] + 1e-6 * max(1., abs(container[key]))

    # Every stored number of both beliefs, for every forecast cell, is recomputed.
    for index, cell in enumerate(drawn):
        for rate, belief in cell["outcome_beliefs"].items():
            for field, value in belief.items():
                if field == "counts":
                    for position in range(2):
                        refused(lambda r: nudge(r["semantic"]["sampling"]["cells"][index]["outcome_beliefs"][rate]["counts"], position),
                                f"{rate}.counts[{position}]")
                else:
                    refused(lambda r: nudge(r["semantic"]["sampling"]["cells"][index]["outcome_beliefs"][rate], field),
                            f"{rate}.{field}")
        for position in range(3):
            refused(lambda r: nudge(r["semantic"]["sampling"]["cells"][index]["probabilities"], position), "probabilities")
    # The selection check recomputes the P_* fields, pi, the 80% point and J.
    for index, row in enumerate(receipt["selection"]["cells"]):
        for field in FIELDS:
            assert set(row[field]) == ({"mean", "lower", "upper"} if field in RANGED else {"mean"})
            for name in row[field]:
                refused(lambda r: nudge(r["selection"]["cells"][index][field], name), f"{field}.{name}")
        for field in ("pi", "admission_risk", "J", "expected_attempts", "T"):
            refused(lambda r: nudge(r["selection"]["cells"][index], field), field)
    tampers = {
        "a range without a belief": lambda r: r["selection"]["cells"][0]["P_clean"].update(lower=0., upper=1.),
        "a missing range": lambda r: r["selection"]["cells"][0]["P_loud"].pop("upper"),
        "selection worth": lambda r: r["selection"]["sampling_policy"].update(model_worth=6.),
        "frozen draw worth": lambda r: r["semantic"]["sampling"]["policy"].update(task_worth_silent=30.),
        "missing worth": lambda r: r["selection"]["sampling_policy"].pop("task_worth_loud"),
        "retired policy key": lambda r: r["selection"]["sampling_policy"].update(prior_strength=8.),
        "retired frozen key": lambda r: r["semantic"]["sampling"]["policy"].update(max_concentration=1000000.),
        "missing belief": lambda r: r["semantic"]["sampling"]["cells"][0]["outcome_beliefs"].pop("loud"),
        "missing field": lambda r: r["semantic"]["sampling"]["cells"][0]["outcome_beliefs"]["silent"].pop("pi"),
        "extra field": lambda r: r["semantic"]["sampling"]["cells"][0]["outcome_beliefs"]["silent"].update(odds=1.),
        "v10 posterior": lambda r: r["semantic"]["sampling"]["cells"][0].update(concentration=8.),
        "v10 interval kind": lambda r: r["semantic"].update(interval_kind="dirichlet-posterior-mixture"),
        "expert weight": lambda r: [e.update(weight=.5) for e in r["semantic"]["beliefs"]["experts"]],
        "forecast": lambda r: r["semantic"]["worlds"]["llm"][0].update(clean=.5, loud=.3, silent=.2),
        "stated range": lambda r: [row.pop("silent_ci", None) for row in r["semantic"]["worlds"]["llm"]],
    }
    for label, tamper in tampers.items():
        refused(tamper, label)


def test_q80_admission_and_posterior_overlap_break_lock_in_but_reject_high_silent_traps(tmp_path):
    candidates = [{"model": "fixture/favored", "effort": "high", "usd_per_mtok": 1.0},
                  {"model": "fixture/native", "effort": "high", "usd_per_mtok": 1.0},
                  {"model": "fixture/trap", "effort": "high", "usd_per_mtok": .01}]
    specs = {"fixture/favored": spec(.94, .05, .01), "fixture/native": spec(.93, .05, .02),
             "fixture/trap": spec(.6, .1, .3)}
    config = tmp_path / "howhard.toml"
    config.write_text('on_cap_unmet = "flag-review"\n[task_types]\ntask = "standard"\n')
    engine = Engine(config)
    first = decide(engine, candidates, specs, seed=0)
    native, trap, favored = (by_model(first)[m] for m in ("fixture/native", "fixture/trap", "fixture/favored"))
    epsilon = first["constants"]["epsilon"]
    # The 97.5% bound of an unseen cell exceeds epsilon, but the q80 admission quantile does not.
    assert native["P_silent_accepted"]["upper"] > epsilon >= native["admission_risk"] and native["admissible"]
    assert trap["admission_risk"] > epsilon and not trap["admissible"]
    assert first["selection"]["sampling_policy"]["admission_quantile"] == .8
    # The point-guess gap exceeds the old 10% mean band but posterior costs overlap.
    assert native["J"] - favored["J"] > .1 * favored["J"]
    assert first["exploration"]["relative_band"] is None
    assert {c["model"] for c in first["exploration"]["eligible_cells"]} == {"fixture/favored", "fixture/native"}

    chosen = [decide(engine, candidates, specs, seed=seed)["chosen"]["model"] for seed in range(150)]
    assert "fixture/trap" not in chosen
    assert 15 <= chosen.count("fixture/native") <= 135
    load(engine.config, first["decision_id"])

    # An explicit risk-class explore_within is still respected.
    capped = tmp_path / "capped"
    capped.mkdir()
    (capped / "howhard.toml").write_text('on_cap_unmet = "flag-review"\n[task_types]\ntask = "standard"\n'
                                         '[classes.standard]\nexplore_within = 0.1\n')
    capped_engine = Engine(capped / "howhard.toml")
    capped_first = decide(capped_engine, candidates, specs, seed=0)
    assert capped_first["exploration"]["relative_band"] == .1
    assert {decide(capped_engine, candidates, specs, seed=s)["chosen"]["model"] for s in range(40)} == {"fixture/favored"}

    # The removed global sampling.explore_within cannot silently change meaning.
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "howhard.toml").write_text('on_cap_unmet = "flag-review"\n[sampling]\nexplore_within = 0.1\n')
    with pytest.raises(HowHardError, match="risk class"):
        Engine(bad / "howhard.toml")


def test_time_decay_detects_regression_and_recovers_before_undecayed_counts(tmp_path):
    worker = [{"model": "claude-opus-5-5", "effort": "high", "usd_per_mtok": 1.0}]

    def admission(directory, history, settings=""):
        """``history`` lists (outcome, days before the request) for each recorded run."""
        (directory / ".howhard").mkdir(parents=True)
        config = directory / "howhard.toml"
        config.write_text('on_cap_unmet = "flag-review"\n[task_types]\ntask = "standard"\n' + settings)
        now = datetime.now(timezone.utc)
        (directory / ".howhard" / "outcomes.jsonl").write_text("".join(json.dumps(row(
            i, task="task", task_text="Repair rollback", outcome=outcome,
            cause=None if outcome == "clean" else "depth",
            observed_at=(now - timedelta(days=days)).isoformat())) + "\n" for i, (outcome, days) in enumerate(history)))
        result = decide(Engine(config, retrieval=RecentDouble()), worker, {"claude-opus-5-5": spec(.94, .05, .01)})
        return result["cells"][0], result

    healthy, _ = admission(tmp_path / "healthy", [("clean", 0)] * 5)
    assert healthy["admissible"]
    regressed, result = admission(tmp_path / "regressed", [("clean", 28)] * 5 + [("silent", 0)] * 3)
    assert not regressed["admissible"] and regressed["P_silent"]["mean"] > .15
    scored = {c["id"]: c for c in result["semantic"]["evidence_candidates"]["candidates"]}
    weights = {o["id"]: o["weight"] for o in result["semantic"]["outcome_observations"]}
    assert scored["run-7"]["days"] == pytest.approx(0, abs=1e-3) and scored["run-0"]["days"] == pytest.approx(28, abs=1e-3)
    # Two half-lives of 14 days: a quarter of the weight of a run recorded now.
    assert weights["run-0"] == pytest.approx(weights["run-7"] * .25, rel=1e-3)

    recovering = [("clean", 60)] * 3 + [("silent", 45)] * 3 + [("clean", 18 - i) for i in range(18)]
    recovered, recovered_result = admission(tmp_path / "recovered", recovering)
    undecayed, undecayed_result = admission(tmp_path / "undecayed", recovering, "[sampling]\nhalf_life_days = 1000000\n")
    assert {o["id"] for o in recovered_result["semantic"]["outcome_observations"]} == {f"run-{i}" for i in range(24)}
    silent = lambda r: sum(o["weight"] for o in r["semantic"]["outcome_observations"] if o["outcome"] == "silent")
    assert silent(recovered_result) == pytest.approx(3 * .5 ** (45 / 14), rel=1e-2)
    assert silent(undecayed_result) == pytest.approx(3, rel=1e-2)
    assert recovered["admissible"] and not undecayed["admissible"]
    assert recovered["admission_risk"] < undecayed["admission_risk"]


def test_frozen_v7_to_v10_receipts_load_observe_replay_and_report(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    candidates = [{"model": f"fixture/model-{i}", "effort": "high", "usd_per_mtok": 1.0} for i in range(2)]
    specs = {c["model"]: spec(.95, .04, .01) for c in candidates}
    current_id = decide(engine, candidates, specs, seed=11)["decision_id"]
    current = load(engine.config, current_id)
    assert current["format"] == "howhard-decision-v11"
    ids = {}
    for name, legacy in (("v10", to_v10(current)), ("v9", to_v9(current)), ("v8", to_v8(current)), ("v7", to_v7(current))):
        ids[name] = digest(legacy)
        with locked(engine.config):
            put(engine.config, "decisions", ids[name], canonical(legacy) + b"\n")
        assert load(engine.config, ids[name])["format"] == f"howhard-decision-{name}"
    engine.observe_decision(ids["v7"], "clean", id="from-v7", seconds=12.)
    engine.observe_decision(ids["v8"], "loud", id="from-v8", cause="capability", active_seconds=10.)
    engine.observe_decision(ids["v9"], "clean", id="from-v9", active_seconds=10.)
    engine.observe_decision(ids["v10"], "clean", id="from-v10", active_seconds=10.)
    engine.observe_decision(current_id, "clean", id="from-v11", active_seconds=10.)
    beliefs = engine.memory_beliefs()
    assert beliefs["unscored_runs"] == 0
    assert next(e for e in beliefs["experts"] if e["key"] == "llm")["evidence_count"] == 5
    stats = report(engine)
    assert stats["unsupported_receipts"] == [] and stats["forecast_metrics"]["llm"]["runs"] == 5
    # The new decision sees those outcomes as per-cell evidence.
    later = decide(engine, candidates, specs, seed=12)
    assert {o["outcome"] for o in later["semantic"]["outcome_observations"]} == {"clean", "loud"}

    # Cross-version shapes are rejected in both directions.
    v8 = load(engine.config, ids["v8"])
    for tamper in (lambda r: r["semantic"].update(interval_kind="dirichlet-posterior-mixture"),
                   lambda r: r["semantic"].update(interval_kind="log-odds-grid"),
                   lambda r: r["semantic"].update(outcome_observations=[]),
                   lambda r: r["semantic"]["worlds"]["llm"][0].update(silent_ci=[0., .5])):
        bad = deepcopy(v8)
        tamper(bad)
        with pytest.raises(HowHardError):
            load(engine.config, bad)
    for tamper in (lambda r: r["semantic"].update(interval_kind="weighted-forecast-envelope"),
                   lambda r: r["semantic"].pop("outcome_observations"),
                   lambda r: r["selection"]["cells"][0].update(admission_risk=r["selection"]["cells"][0]["admission_risk"] + .01),
                   lambda r: r["selection"]["cells"][0]["P_silent"].update(upper=r["selection"]["cells"][0]["P_silent"]["mean"]),
                   lambda r: r["selection"]["sampling_policy"].update(explore_within=.1),
                   lambda r: r["semantic"].update(interval_kind="dirichlet-posterior-mixture"),
                   lambda r: r["semantic"]["sampling"]["cells"][0]["outcome_beliefs"]["silent"].update(n_start=100.),
                   lambda r: r["semantic"]["sampling"]["cells"][0].update(evidence_count=500)):
        bad = deepcopy(current)
        tamper(bad)
        with pytest.raises(HowHardError):
            load(engine.config, bad)

    # The older formats keep the posterior they recorded: a v10 or v9 receipt cannot borrow the
    # beliefs, and its own Dirichlet numbers are still recomputed.
    for name in ("v10", "v9"):
        frozen = load(engine.config, ids[name])
        assert frozen["semantic"]["interval_kind"] == "dirichlet-posterior-mixture"
        cell = frozen["semantic"]["sampling"]["cells"][0]
        assert "outcome_beliefs" not in cell and cell["outcome_posterior"]["prior_concentration"] == 8.
        assert cell["concentration"] == 8. and cell["outcome_posterior"]["alpha"] == pytest.approx([7.6, .32, .08])
        assert {"prior_strength", "min_concentration", "max_concentration"} <= frozen["selection"]["sampling_policy"].keys()
        assert not {"model_worth", "task_worth_silent", "task_worth_loud"} & frozen["selection"]["sampling_policy"].keys()
        for field in FIELDS:
            assert frozen["selection"]["cells"][0][field].keys() == {"mean", "lower", "upper"}
        for tamper in (lambda r: r["semantic"].update(interval_kind="log-odds-grid"),
                       lambda r: r["semantic"]["sampling"]["cells"][0].update(
                           outcome_beliefs=current["semantic"]["sampling"]["cells"][0]["outcome_beliefs"]),
                       lambda r: r["semantic"]["sampling"]["cells"][0]["outcome_posterior"].update(prior_concentration=100.),
                       lambda r: r["selection"]["cells"][0]["P_clean"].pop("upper"),
                       lambda r: r["selection"]["sampling_policy"].pop("prior_strength"),
                       lambda r: r["selection"]["sampling_policy"].update(model_worth=5.)):
            bad = deepcopy(frozen)
            tamper(bad)
            with pytest.raises(HowHardError):
                load(engine.config, bad)
