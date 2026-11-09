"""Accepted-result selection at public decision boundaries (issue #67)."""

import math

import pytest

from howhard import Engine
from howhard.decisions import load
from conftest import forecast_request
from test_capability_filter import SimilarityDouble, filtering

# Forecast inputs retained from a reported decision; these are not observed outcomes.
TABLE = [
    ("claude-haiku-5-5", "low", .03, .82, .15, 2.5),
    ("claude-haiku-5-5", "high", .07, .75, .18, 3.6),
    ("claude-opus-5-5", "xhigh", .47, .40, .13, 5.4),
    ("claude-opus-5-5", "high", .43, .42, .15, 4.6),
]


def prepare(engine, table):
    candidates = [{"model": r[0], "effort": r[1]} for r in table]
    pending = forecast_request(engine, engine.prepare("task", "Repair rollback", native_candidates=candidates))
    indexed = {(r[0], r[1]): r for r in table}
    response = {"forecasts": [], "memory_forecasts": [], "reason": "Synthetic selection regression"}
    for c in pending["baseline_request"]["cells"]:
        r = indexed[c["model"], c["effort"]]
        response["forecasts"].append({"family": c["family"], "effort": c["effort"],
            "clean": r[2], "loud": r[3], "silent": r[4], "seconds": r[5]*3600, "tokens": 1000})
    return pending, response


def test_reported_high_failure_cell_loses_on_cost_to_success_and_cap(tmp_path):
    engine = Engine(directory=tmp_path)
    request, response = prepare(engine, TABLE)
    result = engine.decide(request["request_id"], response, seed=42)
    assert result["chosen"]["model"] == "claude-opus-5-5"
    assert result["chosen"]["effort"] == "xhigh"
    indexed = {(c["model"], c["effort"]): c for c in result["cells"]}
    assert indexed["claude-haiku-5-5", "low"]["J"] == pytest.approx(253.5/.18)
    assert indexed["claude-haiku-5-5", "high"]["P_silent_accepted"]["mean"] == pytest.approx(.72)
    assert not indexed["claude-haiku-5-5", "high"]["admissible"]
    assert result["exploration"]["reason"] == "disabled-cap-unmet"
    assert not result["exploration"]["enabled"] and result["needs_review"]
    assert "sampled_expert" not in result["semantic"]
    receipt = load(engine.config, result["decision_id"])
    assert receipt["selection"]["objective"] == "cost-to-accepted-result"
    assert receipt["selection"]["exploration"] == result["exploration"]


def test_one_thousand_seeds_never_choose_the_low_clean_probability_cells(tmp_path):
    engine = Engine(directory=tmp_path)
    request, response = prepare(engine, TABLE)
    first = engine.decide(request["request_id"], response, seed=42)
    assert not first["exploration"]["enabled"]
    selected = [engine.decide(request["request_id"], response, seed=seed)["chosen"] for seed in range(1000)]
    assert all(c["model"] == "claude-opus-5-5" and c["effort"] == "xhigh" for c in selected)


@pytest.mark.parametrize("loud,silent,clean", [(.4, .13, .47), (1., 0., 0.)])
def test_retry_formula_and_denominator_floor_are_finite(tmp_path, loud, silent, clean):
    engine = Engine(directory=tmp_path)
    request, response = prepare(engine, [("fixture/worker", "high", clean, loud, silent, 2.)])
    result = engine.decide(request["request_id"], response, seed=1)
    expected = (25*2+50*loud+1000*silent)/max(.0001, 1-loud)
    assert result["cells"][0]["J"] == pytest.approx(expected)
    assert math.isfinite(result["cells"][0]["J"])


def test_cap_targets_the_accepted_result_even_when_raw_silent_risk_is_small(tmp_path):
    engine = Engine(directory=tmp_path)
    request, response = prepare(engine, [("fixture/worker", "high", .01, .98, .01, 1.)])
    result = engine.decide(request["request_id"], response, seed=1)
    row = result["cells"][0]
    assert row["P_silent"]["upper"] < result["constants"]["epsilon"]
    assert row["P_silent_accepted"]["upper"] == pytest.approx(.5)
    assert not result["cap_met"] and result["needs_review"]
    assert result["exploration"]["reason"] == "disabled-cap-unmet"


def test_evidence_tightens_agreeing_forecasts_while_a_weak_prior_can_explore(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    # Actual synthetic execution rows calibrate the same prospective baseline expert.
    history, history_response = prepare(engine, [("fixture/history-worker", "high", .99, .01, 0., 4.)])
    historical = engine.decide(history["request_id"], history_response, seed=0)
    table = [("fixture/a", "high", .99, .01, 0., 4.), ("fixture/b", "high", .99, .01, 0., 4.2)]
    weak, weak_response = prepare(engine, table)
    initial = engine.decide(weak["request_id"], weak_response, seed=0)
    assert "sampling" in initial["semantic"]
    weak_choices = {engine.decide(weak["request_id"], weak_response, seed=seed)["chosen"]["model"] for seed in range(1000)}
    assert weak_choices == {"fixture/a", "fixture/b"}
    for i in range(100):
        engine.observe_decision(historical["decision_id"], "clean", id=f"historical-{i}")
    strong, strong_response = prepare(engine, table)
    first = engine.decide(strong["request_id"], strong_response, seed=0)
    assert first["semantic"]["sampling"]["cells"][0]["concentration"] > initial["semantic"]["sampling"]["cells"][0]["concentration"]
    strong_choices = [engine.decide(strong["request_id"], strong_response, seed=seed)["chosen"]["model"] for seed in range(1000)]
    assert strong_choices.count("fixture/a") >= 990


def test_unmet_cap_does_not_create_a_random_generator(tmp_path, monkeypatch):
    engine = Engine(directory=tmp_path)
    request, response = prepare(engine, TABLE)
    def forbidden(*args, **kwargs):
        raise AssertionError("No random sampling is allowed when the cap is unmet")
    monkeypatch.setattr("numpy.random.default_rng", forbidden)
    result = engine.decide(request["request_id"], response, seed=42)
    assert result["chosen"]["effort"] == "xhigh"


def test_accepted_risk_envelope_uses_each_experts_joint_probabilities(tmp_path):
    engine = Engine(directory=tmp_path)
    memory = engine.memory.add("Risk", "A different correlated loud/silent prediction")
    request, response = prepare(engine, [("fixture/worker", "high", .81, .1, .09, 1.)])
    override = {**response["forecasts"][0], "clean": .01, "loud": .98, "silent": .01}
    response["memory_forecasts"] = [{"id": memory["id"], "version": memory["version"], "forecasts": [override]}]
    result = engine.decide(request["request_id"], response, seed=42)
    # Per-expert ratios are .1 and .5; mixing unrelated marginal upper bounds would exceed 1.
    assert result["cells"][0]["P_silent_accepted"]["upper"] == pytest.approx(.5)


def test_forecasted_band_owns_worker_money_comparison(tmp_path):
    engine = Engine(directory=tmp_path)
    cells = [{"model": "fixture/a", "effort": "high", "usd_per_mtok": 2.},
             {"model": "fixture/b", "effort": "high"}]
    first = engine.prepare("task", native_candidates=cells)
    band = engine.decide(first["request_id"], filtering(first["baseline_request"], 0, 0))
    row = band["baseline_request"]["cells"][0]
    response = {"forecasts": [{"family": row["family"], "effort": row["effort"], "clean": .99,
                               "loud": .009, "silent": .001, "tokens": 1000, "seconds": 10}],
                "memory_forecasts": [], "reason": "Synthetic priced-band forecast"}
    result = engine.decide(band["request_id"], response, seed=42)
    assert result["cost_policy"]["worker_money_included"]
    assert len(load(engine.config, result["decision_id"])["native_candidates"]) == 2



def test_lowest_bound_is_deterministic_and_still_requires_review(tmp_path):
    config = tmp_path / "howhard.toml"
    config.write_text('on_cap_unmet="lowest-bound"\n[task_types]\ntask="standard"\n')
    engine = Engine(config)
    request, response = prepare(engine, TABLE)
    result = engine.decide(request["request_id"], response, seed=42)
    assert result["chosen"]["model"] == "claude-opus-5-5"
    assert result["needs_review"] and result["exploration"]["reason"] == "disabled-cap-unmet"
    load(engine.config, result["decision_id"])
