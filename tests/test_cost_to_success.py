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
    if pending["baseline_request"] is None:
        return pending, {"memory_forecasts": [], "reason": "Synthetic selection regression"}
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
    indexed = {(c["model"], c["effort"]): c for c in result["cells"]}
    assert indexed["claude-haiku-5-5", "low"]["J"] == pytest.approx(253.5/.18)
    assert indexed["claude-haiku-5-5", "high"]["P_silent_accepted"]["mean"] == pytest.approx(.72)
    assert not indexed["claude-haiku-5-5", "high"]["admissible"]
    # No cell is trusted. The two stronger cells are on trial and compete on one drawn sample;
    # the high-failure cells are excluded, and the decision is flagged.
    assert {pair: c["tier"] for pair, c in indexed.items()} == {
        ("claude-haiku-5-5", "low"): "excluded", ("claude-haiku-5-5", "high"): "excluded",
        ("claude-opus-5-5", "xhigh"): "on-trial", ("claude-opus-5-5", "high"): "on-trial"}
    assert result["exploration"]["pool"] == "trusted-and-on-trial"
    assert {c["model"] for c in result["exploration"]["eligible_cells"]} == {"claude-opus-5-5"}
    assert result["needs_review"] and not result["cap_met"]
    assert result["semantic"]["sampling"]["enabled"] and "sampled_expert" in result["semantic"]
    receipt = load(engine.config, result["decision_id"])
    assert receipt["selection"]["objective"] == "cost-to-accepted-result"
    assert receipt["selection"]["exploration"] == result["exploration"]


def test_one_thousand_seeds_never_choose_the_low_clean_probability_cells(tmp_path):
    engine = Engine(directory=tmp_path)
    request, response = prepare(engine, TABLE)
    first = engine.decide(request["request_id"], response, seed=42)
    assert first["exploration"]["pool"] == "trusted-and-on-trial"
    selected = [engine.decide(request["request_id"], response, seed=seed)["chosen"] for seed in range(1000)]
    assert all(c["model"] == "claude-opus-5-5" and c["effort"] in ("xhigh", "high") for c in selected)


@pytest.mark.parametrize("loud,silent,clean", [(.4, .13, .47), (1., 0., 0.)])
def test_retry_formula_and_denominator_floor_are_finite(tmp_path, loud, silent, clean):
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    request, response = prepare(engine, [("fixture/worker", "high", clean, loud, silent, 2.)])
    result = engine.decide(request["request_id"], response, seed=1)
    row = result["cells"][0]
    p_loud, p_silent = row["P_loud"]["mean"], row["P_silent"]["mean"]
    expected = (25*2+50*p_loud+1000*p_silent)/max(.0001, 1-p_loud)
    assert row["J"] == pytest.approx(expected)
    assert math.isfinite(row["J"]) and math.isfinite(row["sampled_J"])
    if loud == 1.:
        # Push posterior P_loud past 1 - 1e-4 to exercise the denominator floor directly.
        heavy_dir = tmp_path / "heavy"
        heavy_dir.mkdir()
        (heavy_dir / "howhard.toml").write_text(
            'on_cap_unmet = "flag-review"\n'
            '[task_types]\ntask = "standard"\n'
            '[models."fixture/worker"]\nfamily = "worker"\nefforts = ["high"]\nweight = 1500.0\n'
        )
        heavy_engine = Engine(heavy_dir / "howhard.toml", retrieval=SimilarityDouble())
        for i in range(12):
            heavy_engine.observe("task", "fixture/worker", "high", "loud",
                                 task_text="Repair rollback", note="capability_caused=false", id=f"loud-{i}")
        floored_req, floored_resp = prepare(heavy_engine, [("fixture/worker", "high", clean, loud, silent, 2.)])
        floored = heavy_engine.decide(floored_req["request_id"], floored_resp, seed=1)["cells"][0]
        assert 1 - floored["P_loud"]["mean"] < .0001
        assert floored["expected_attempts"] == pytest.approx(1/.0001)
        assert floored["J"] == pytest.approx(
            (25*2+50*floored["P_loud"]["mean"]+1000*floored["P_silent"]["mean"])/.0001)
        assert math.isfinite(floored["J"])


def test_cap_targets_the_accepted_result_even_when_raw_silent_risk_is_small(tmp_path):
    engine = Engine(directory=tmp_path)
    request, response = prepare(engine, [("fixture/worker", "high", .005, .99, .005, 1.)])
    result = engine.decide(request["request_id"], response, seed=1)
    row = result["cells"][0]
    assert row["P_silent"]["upper"] < result["constants"]["epsilon"]
    assert row["P_silent_accepted"]["mean"] == pytest.approx(.5)
    assert row["P_silent_accepted"]["upper"] > row["admission_risk"] > result["constants"]["epsilon"]
    assert not result["cap_met"] and result["needs_review"]
    assert row["tier"] == "on-trial" and result["exploration"]["reason"] == "disabled-no-supported-alternative"


def test_evidence_tightens_only_the_observed_cell_while_a_weak_prior_can_explore(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    table = [("fixture/a", "high", .92, .06, .02, 4.2), ("fixture/b", "high", .92, .06, .02, 4.)]
    weak, weak_response = prepare(engine, table)
    initial = engine.decide(weak["request_id"], weak_response, seed=0)
    assert "sampling" in initial["semantic"]
    weak_choices = [engine.decide(weak["request_id"], weak_response, seed=seed)["chosen"]["model"] for seed in range(1000)]
    assert set(weak_choices) == {"fixture/a", "fixture/b"}
    # Actual synthetic executions of fixture/b update only fixture/b's own posterior
    # while keeping weaker unseen fixture/a inside the stage-1 fit band.
    for i in range(8):
        engine.observe("task", "fixture/b", "high", "clean", native=True, task_text="Repair rollback", id=f"historical-{i}")
    strong, strong_response = prepare(engine, table)
    first = engine.decide(strong["request_id"], strong_response, seed=0)
    concentration = lambda result: {c["family"]: c["concentration"] for c in result["semantic"]["sampling"]["cells"]}
    families = {c["model"]: c["family"] for c in first["cells"]}
    added_weight = sum(o["weight"] for o in first["semantic"]["outcome_observations"])
    assert added_weight > 6.5
    assert concentration(first)[families["fixture/b"]] == pytest.approx(1.0 + added_weight)
    assert concentration(first)[families["fixture/a"]] == concentration(initial)[families["fixture/a"]]
    strong_choices = [engine.decide(strong["request_id"], strong_response, seed=seed)["chosen"]["model"] for seed in range(1000)]
    assert strong_choices.count("fixture/b") > weak_choices.count("fixture/b")


def test_no_trusted_cell_still_draws_one_seeded_sample(tmp_path, monkeypatch):
    import numpy
    engine = Engine(directory=tmp_path)
    request, response = prepare(engine, TABLE)
    seeds, original = [], numpy.random.default_rng
    def recording(seed=None):
        seeds.append(seed)
        return original(seed)
    monkeypatch.setattr("numpy.random.default_rng", recording)
    result = engine.decide(request["request_id"], response, seed=42)
    assert seeds == [42] and result["seed"] == 42
    assert not result["cap_met"] and result["semantic"]["sampling"]["enabled"]
    assert result["chosen"]["model"] == "claude-opus-5-5"


def test_accepted_risk_envelope_uses_each_experts_joint_probabilities(tmp_path):
    engine = Engine(directory=tmp_path)
    memory = engine.memory.add("Risk", "A different correlated loud/silent prediction")
    request, response = prepare(engine, [("fixture/worker", "high", .81, .1, .09, 1.)])
    override = {**response["forecasts"][0], "clean": .01, "loud": .98, "silent": .01}
    response["memory_forecasts"] = [{"id": memory["id"], "version": memory["version"], "forecasts": [override]}]
    result = engine.decide(request["request_id"], response, seed=42)
    # Per-expert joint ratios are .1 and .5; mixing unrelated marginal bounds could exceed 1.
    weights = {e["key"]: e["weight"] for e in result["semantic"]["beliefs"]["experts"]}
    expected = weights["llm"] * .1 + weights[f"memory:{memory['id']}:{memory['version']}"] * .5
    accepted = result["cells"][0]["P_silent_accepted"]
    assert accepted["mean"] == pytest.approx(expected, rel=1e-3)
    assert accepted["lower"] < accepted["mean"] < accepted["upper"] <= 1


def test_forecasted_band_owns_worker_money_comparison(tmp_path):
    engine = Engine(directory=tmp_path)
    # Two efforts of one model, so the unpriced cell outside the band takes no reserved forecast slot.
    cells = [{"model": "fixture/a", "effort": "high", "usd_per_mtok": 2.},
             {"model": "fixture/a", "effort": "low"}]
    first = engine.prepare("task", native_candidates=cells)
    keys = {c["effort"]: c["key"] for c in first["baseline_request"]["cells"]}
    band = engine.decide(first["request_id"], filtering(first["baseline_request"], 1, 1, [keys["low"], keys["high"]]))
    assert [c["effort"] for c in band["baseline_request"]["cells"]] == ["high"]
    row = band["baseline_request"]["cells"][0]
    response = {"forecasts": [{"family": row["family"], "effort": row["effort"], "clean": .99,
                               "loud": .009, "silent": .001, "tokens": 1000, "seconds": 10}],
                "memory_forecasts": [], "reason": "Synthetic priced-band forecast"}
    result = engine.decide(band["request_id"], response, seed=42)
    assert result["cost_policy"]["worker_money_included"]
    assert len(load(engine.config, result["decision_id"])["native_candidates"]) == 2



def test_retired_lowest_bound_policy_changes_nothing_and_the_decision_is_still_flagged(tmp_path):
    results = []
    for name, policy in (("old", 'on_cap_unmet="lowest-bound"\n'), ("new", "")):
        (tmp_path / name).mkdir()
        config = tmp_path / name / "howhard.toml"
        config.write_text(policy + '[task_types]\ntask="standard"\n')
        engine = Engine(config)
        request, response = prepare(engine, TABLE)
        result = engine.decide(request["request_id"], response, seed=42)
        assert result["chosen"]["model"] == "claude-opus-5-5" and result["needs_review"]
        assert "on_cap_unmet" not in load(engine.config, result["decision_id"])["selection"]
        results.append(result)
    assert results[0]["chosen"] == results[1]["chosen"] and results[0]["selection"] == results[1]["selection"]
