"""Task-conditioned capability filtering through public boundaries (issue #66)."""

from copy import deepcopy
import json
import math

import pytest

from howhard import Engine
from howhard.decisions import load
from howhard.report import report
from howhard.validation import HowHardError


def inventory(count):
    return [{"model": f"fixture/model-{i:03d}", "effort": "high"} for i in range(count)]


def filtering(request, lower=0, upper=None, order=None):
    order = order or [c["key"] for c in request["cells"]]
    upper = min(len(order) - 1, lower + 3) if upper is None else upper
    return {"filter": {"order": order, "lower": order[lower], "upper": order[upper]},
            "memory_filters": [], "reason": "Synthetic task-specific fit band"}


def forecasts(request, silent=.01, clean=None):
    clean = 1 - silent - .05 if clean is None else clean
    return {"forecasts": [{"family": c["family"], "effort": c["effort"],
            "clean": clean, "loud": 1-clean-silent, "silent": silent,
            "tokens": 1000, "seconds": 10} for c in request["cells"]],
            "memory_forecasts": [], "reason": "Synthetic calibrated forecasts"}


class ProtocolDouble:
    identity = "capability-protocol-double"
    def __init__(self, lower=0, upper=None):
        self.calls, self.lower, self.upper = [], lower, upper
        self.last_usage = {"cost_usd": None}
    def complete(self, request):
        self.calls.append(deepcopy(request))
        if request["kind"] == "learn":
            return {"operations": [], "reason": "Synthetic no-op writer"}
        if request["kind"] == "filter":
            if request["phase"] == "baseline":
                result = filtering(request, self.lower, self.upper)
                result.pop("memory_filters")
                return result
            return {"memory_filters": [], "reason": "No fixture memory override"}
        if request["phase"] == "baseline":
            result = forecasts(request)
            result.pop("memory_forecasts")
            return result
        return {"memory_forecasts": [], "reason": "No fixture forecast override"}


class SimilarityDouble:
    """Absolute relevance fixture; no provider or pretrained model calls."""
    def search(self, records, query, limit=8, **kwargs):
        chosen = records[:limit]
        return {"records": deepcopy(chosen), "trace": {"seconds": 0,
                "selected": [{"key": r["key"], "score": 12 if 'rollback' in r['text'] and 'rollback' in query else -18} for r in chosen]}}


def start(engine, cells, **options):
    return engine.prepare("task", "Repair rollback", native_candidates=cells, **options)


def finish(engine, pending, response=None):
    if pending["stage"] == "filter":
        pending = engine.decide(pending["request_id"], response or filtering(pending["baseline_request"]), seed=42)
    while "decision_id" not in pending:
        assert pending["stage"] == "forecast"
        pending = engine.decide(pending["request_id"], forecasts(pending["baseline_request"]), seed=42)
    return pending


@pytest.mark.parametrize("count", [33, 200])
def test_unbounded_inventory_runs_both_stages_and_validates_receipt(tmp_path, count):
    engine = Engine(directory=tmp_path)
    pending = start(engine, inventory(count))
    assert pending["stage"] == "filter" and len(pending["baseline_request"]["cells"]) == count
    assert pending["baseline_request"]["kind"] == "filter"
    assert not {"memories", "beliefs", "evidence", "retrieval"} & pending["baseline_request"].keys()
    next_request = engine.decide(pending["request_id"], filtering(pending["baseline_request"], 3, 6))
    assert next_request["request_id"] != pending["request_id"]
    assert len(next_request["baseline_request"]["cells"]) == 4
    result = finish(engine, next_request)
    receipt = load(engine.config, result["decision_id"])
    assert len(receipt["native_candidates"]) == count
    assert len(receipt["capability"]["cells"]) == count
    assert len(receipt["semantic"]["worlds"]["llm"]) == 4
    assert receipt["format"] == "howhard-decision-v6"


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unknown", "empty"])
def test_invalid_filter_refused_before_forecasting(tmp_path, defect):
    engine = Engine(directory=tmp_path)
    pending = start(engine, inventory(5))
    response = filtering(pending["baseline_request"])
    order = response["filter"]["order"]
    if defect == "missing":
        order.pop()
    elif defect == "duplicate":
        order[-1] = order[0]
    elif defect == "unknown":
        order[-1] = "unknown-cell"
    else:
        response["filter"].update(lower=order[-1], upper=order[0])
    with pytest.raises(HowHardError, match="filter"):
        engine.decide(pending["request_id"], response)
    assert not (engine.config.state_root / "decisions").exists()


def test_oversized_band_gets_one_refilter_then_fails_without_truncation(tmp_path):
    engine = Engine(directory=tmp_path)
    pending = start(engine, inventory(33))
    response = filtering(pending["baseline_request"], 0, 32)
    retry = engine.decide(pending["request_id"], response)
    assert retry["stage"] == "filter" and retry["refilter_round"] == 1
    assert len(retry["baseline_request"]["cells"]) == 33
    with pytest.raises(HowHardError, match="max_forecast_cells.*tighter"):
        engine.decide(retry["request_id"], response)
    # A valid re-filter keeps its entire requested band.
    result = finish(engine, engine.decide(retry["request_id"], filtering(retry["baseline_request"], 9, 11)))
    assert len(result["cells"]) == 3


def test_widening_requests_only_next_stronger_cells_and_selects_forecasted_one(tmp_path):
    engine = Engine(directory=tmp_path)
    pending = start(engine, inventory(6))
    pending = engine.decide(pending["request_id"], filtering(pending["baseline_request"], 1, 2))
    initial = {c["model"] for c in pending["baseline_request"]["cells"]}
    wider = engine.decide(pending["request_id"], forecasts(pending["baseline_request"], silent=.8))
    assert wider["stage"] == "forecast"
    assert [c["model"] for c in wider["baseline_request"]["cells"]] == ["fixture/model-003"]
    result = engine.decide(wider["request_id"], forecasts(wider["baseline_request"]), seed=42)
    assert result["chosen"]["model"] == "fixture/model-003" and result["cap_met"]
    assert len(result["capability"]["widening"]) == 1
    assert {c["model"] for c in result["cells"]} == initial | {"fixture/model-003"}
    assert "fixture/model-000" not in {c["model"] for c in result["cells"]}
    load(engine.config, result["decision_id"])


def test_widening_is_bounded_before_existing_cap_unmet_policy(tmp_path):
    engine = Engine(directory=tmp_path)
    pending = start(engine, inventory(8))
    pending = engine.decide(pending["request_id"], filtering(pending["baseline_request"], 1, 1))
    while "decision_id" not in pending:
        pending = engine.decide(pending["request_id"], forecasts(pending["baseline_request"], silent=.8), seed=1)
    assert len(pending["capability"]["widening"]) == 2
    assert not pending["cap_met"] and pending["needs_review"]
    assert len(pending["cells"]) == 3


def test_pick_drives_both_stages_and_only_forecasts_the_band(tmp_path):
    provider = ProtocolDouble(3, 5)
    engine = Engine(directory=tmp_path, llm=provider)
    result = engine.pick("task", "Repair rollback", native_candidates=inventory(200), seed=42)
    assert [r["kind"] for r in provider.calls] == ["filter", "predict"]
    assert len(provider.calls[0]["cells"]) == 200
    assert len(provider.calls[1]["cells"]) == 3
    assert len(result["cells"]) == 3


@pytest.mark.parametrize("outcome", ["loud", "silent"])
def test_related_failures_raise_the_task_specific_lower_cut(tmp_path, outcome):
    engine = Engine(directory=tmp_path, llm=ProtocolDouble(), retrieval=SimilarityDouble())
    for i in range(8):
        engine.observe("task", "fixture/model-001", "high", outcome, native=True,
                       task_text="Repair rollback", cause="capability", id=f"run-{i}")
    pending = start(engine, inventory(5))
    next_request = engine.decide(pending["request_id"], filtering(pending["baseline_request"], 0, 3))
    state = next_request["capability"]
    classified = {c["model"]: c for c in state["cells"]}
    assert classified["fixture/model-001"]["classification"] == "underpowered"
    assert classified["fixture/model-000"]["classification"] == "underpowered"
    assert classified["fixture/model-004"]["placement"] == "prior-only"
    assert classified["fixture/model-004"]["threshold"] == "learned"


def test_clean_related_runs_lower_the_upper_cut_but_dissimilar_runs_do_not(tmp_path):
    engine = Engine(directory=tmp_path, llm=ProtocolDouble(), retrieval=SimilarityDouble())
    cells = inventory(5)
    pending = start(engine, cells)
    prior = engine.decide(pending["request_id"], filtering(pending["baseline_request"], 0, 4))["capability"]
    for i in range(8):
        engine.observe("task", cells[0]["model"], "high", "clean", native=True,
                       task_text="Repair rollback", id=f"clean-{i}")
    pending = start(engine, cells)
    learned = engine.decide(pending["request_id"], filtering(pending["baseline_request"], 0, 4))["capability"]
    assert prior["band"] != learned["band"]
    assert learned["cells"][-1]["classification"] == "overpowered"
    other = engine.prepare("task", "Write a poem", native_candidates=cells)
    unrelated = engine.decide(other["request_id"], filtering(other["baseline_request"], 0, 4))["capability"]
    assert unrelated["band"] == prior["band"]


def test_cross_model_order_is_learned_locally_and_can_reverse_for_another_task(tmp_path):
    engine = Engine(directory=tmp_path, llm=ProtocolDouble(), retrieval=SimilarityDouble())
    for i in range(4):
        engine.observe("task", "fixture/model-000", "high", "clean", native=True,
                       task_text="Repair rollback", id=f"success-{i}")
        engine.observe("task", "fixture/model-001", "high", "silent", native=True,
                       task_text="Repair rollback", cause="capability", id=f"failure-{i}")
    pending = start(engine, inventory(3))
    response = filtering(pending["baseline_request"], 0, 2)
    state = engine.decide(pending["request_id"], response)["capability"]
    assert state["order"].index(response["filter"]["order"][0]) > state["order"].index(response["filter"]["order"][1])
    other = engine.prepare("task", "Write a poem", native_candidates=inventory(3))
    other_response = filtering(other["baseline_request"], 0, 2)
    assert engine.decide(other["request_id"], other_response)["capability"]["order"] == other_response["filter"]["order"]


def test_one_observation_updates_both_readers_and_reports_disagreement(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    memory = engine.memory.add("Rollback", "Related capability and forecast guidance")
    pending = start(engine, inventory(2))
    f = filtering(pending["baseline_request"], 0, 1)
    f["memory_filters"] = [{"id": memory["id"], "version": memory["version"], "filter": deepcopy(f["filter"])}]
    band = engine.decide(pending["request_id"], f)
    r = forecasts(band["baseline_request"], silent=.01, clean=.2)
    r["memory_forecasts"] = [{"id": memory["id"], "version": memory["version"],
                              "forecasts": [{**c, "clean": .3, "loud": .69} for c in r["forecasts"]]}]
    result = engine.decide(band["request_id"], r, seed=2)
    assert result["capability"]["disagreements"]
    before = {e["key"]: e for e in engine.memory_beliefs()["experts"]}
    engine.observe_decision(result["decision_id"], "silent", id="actual-run", cause="capability")
    assert len(engine.raw_rows) == 1
    after = {e["key"]: e for e in engine.memory_beliefs()["experts"]}
    assert after["filter:llm"]["evidence_count"] == 1
    assert after["filter:llm"]["log_score"] < before["filter:llm"]["log_score"]
    assert after[f"memory:{memory['id']}:{memory['version']}"]["evidence_count"] == 1
    stats = report(engine)["filter_quality"]
    assert stats["fit_failures"]["silent"] == 1
    assert stats["disagreements"]["count"] > 0 and stats["disagreements"]["resolved"] == 1
    assert stats["executions"][0]["position"] >= 0
    old = load(engine.config, result["decision_id"])
    old["format"] = "howhard-decision-v5"
    with pytest.raises(HowHardError, match="format.*unsupported"):
        engine.observe_decision(old, "clean", id="old")


def test_memory_cut_adjustments_participate_with_tempered_weights(tmp_path):
    engine = Engine(directory=tmp_path)
    memory = engine.memory.add("Rollback", "More capable cells may still fit this task")
    pending = start(engine, inventory(4))
    response = filtering(pending["baseline_request"], 0, 0)
    alternate = filtering(pending["baseline_request"], 0, 3)["filter"]
    response["memory_filters"] = [{"id": memory["id"], "version": memory["version"], "filter": alternate}]
    band = engine.decide(pending["request_id"], response)
    state = band["capability"]
    assert len(state["band"]) > 1
    assert {e["key"]: e["weight"] for e in state["beliefs"]["experts"]} == {
        "filter:llm": pytest.approx(.8), f"filter:memory:{memory['id']}:{memory['version']}": pytest.approx(.2)}
    assert state["baseline"]["upper"] != state["memory_filters"][0]["filter"]["upper"]


def test_known_effort_order_is_validated_while_custom_control_order_is_prior_only(tmp_path):
    engine = Engine(directory=tmp_path)
    cells = [{"model": "fixture/worker", "effort": e} for e in ("low", "high")]
    pending = start(engine, cells)
    supplied = {c["effort"]: c["key"] for c in pending["baseline_request"]["cells"]}
    wrong = [supplied["high"], supplied["low"]]
    with pytest.raises(HowHardError, match="ordinal"):
        engine.decide(pending["request_id"], filtering(pending["baseline_request"], 0, 1, wrong))
    other = start(engine, [{"model": "fixture/custom", "effort": e} for e in ("fast", "deep")])
    order = [c["key"] for c in reversed(other["baseline_request"]["cells"])]
    state = engine.decide(other["request_id"], filtering(other["baseline_request"], 0, 1, order))["capability"]
    assert state["order"] == order
    assert all(c["placement"] == "prior-only" for c in state["cells"])


@pytest.mark.parametrize("field,value", [("max_forecast_cells", 0), ("widening_rounds", -1),
                                         ("observation_noise", .5), ("min_clear_probability", True)])
def test_capability_policy_is_validated_before_state(tmp_path, field, value):
    path = tmp_path / "howhard.toml"
    path.write_text('on_cap_unmet="flag-review"\n[capability]\n'+field+'='+json.dumps(value)+'\n')
    with pytest.raises(HowHardError, match="capability"):
        Engine(path)
    assert not (tmp_path / ".howhard").exists()


def test_noncapability_failure_notes_do_not_shift_the_threshold(tmp_path):
    engine = Engine(directory=tmp_path, llm=ProtocolDouble(), retrieval=SimilarityDouble())
    cells = inventory(3)
    prior = start(engine, cells)
    prior_state = engine.decide(prior["request_id"], filtering(prior["baseline_request"], 0, 2))["capability"]
    for i in range(4):
        engine.observe("task", cells[0]["model"], "high", "silent", native=True, task_text="Repair rollback",
                       note="capability_caused=false; infrastructure outage", id=f"outage-{i}")
    pending = start(engine, cells)
    state = engine.decide(pending["request_id"], filtering(pending["baseline_request"], 0, 2))["capability"]
    assert [r["P_clear"] for r in state["cells"]] == pytest.approx([r["P_clear"] for r in prior_state["cells"]])
    assert all(r["weight"] == 0 for r in state["evidence"])


@pytest.mark.parametrize("corruption", ["classification", "position", "probability", "band", "forecast", "disagreement"])
def test_receipt_rejects_inconsistent_filter_or_forecast_coverage(tmp_path, corruption):
    engine = Engine(directory=tmp_path)
    result = finish(engine, start(engine, inventory(5)))
    receipt = load(engine.config, result["decision_id"])
    state = receipt["capability"]
    if corruption == "classification":
        state["cells"][0]["classification"] = "overpowered"
    elif corruption == "position":
        state["cells"][0]["position"] = 99
    elif corruption == "probability":
        state["cells"][0]["P_clear"] = 1.1
    elif corruption == "band":
        state["band"].pop()
    elif corruption == "forecast":
        receipt["semantic"]["worlds"]["llm"].pop()
    else:
        state["disagreements"].append({"key": "fake"})
    with pytest.raises(HowHardError):
        load(engine.config, receipt)


def test_related_evidence_can_reorder_observed_models_across_unseen_cells(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    for i in range(4):
        engine.observe("task", "fixture/model-000", "high", "clean", native=True,
                       task_text="Repair rollback", id=f"good-{i}")
        engine.observe("task", "fixture/model-003", "high", "silent", native=True,
                       task_text="Repair rollback", cause="capability", id=f"bad-{i}")
    pending = start(engine, inventory(5))
    response = filtering(pending["baseline_request"], 0, 4)
    state = engine.decide(pending["request_id"], response)["capability"]
    a, b = response["filter"]["order"][0], response["filter"]["order"][3]
    assert state["order"].index(a) > state["order"].index(b)
    assert next(c for c in state["cells"] if c["model"] == "fixture/model-001")["placement"] == "prior-only"


def test_posterior_does_not_label_a_known_underpowered_strongest_cell_as_fit(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    for i in range(8):
        engine.observe("task", "fixture/model-002", "high", "silent", native=True,
                       task_text="Repair rollback", cause="capability", id=f"too-hard-{i}")
    pending = start(engine, inventory(3))
    with pytest.raises(HowHardError, match="no cell.*capability"):
        engine.decide(pending["request_id"], filtering(pending["baseline_request"], 0, 2))


def failed_at(tmp_path, cell):
    """One related capability failure; the posterior keeps every P_clear below reliability."""
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    engine.observe("task", cell["model"], cell["effort"], "loud", native=True,
                   task_text="Repair rollback", cause="capability", id="related-failure")
    return engine


@pytest.mark.parametrize("lower", [0, 2])
def test_unreliable_inventory_cuts_band_at_best_attained_probability_not_inventory_top(tmp_path, lower):
    cells = inventory(20)
    engine = failed_at(tmp_path, cells[-1])
    pending = start(engine, cells)
    band = engine.decide(pending["request_id"], filtering(pending["baseline_request"], lower, lower + 5))
    assert band["stage"] == "forecast"
    state, policy = band["capability"], band["capability"]["policy"]
    p = [c["P_clear"] for c in state["cells"]]
    assert max(p) < policy["reliability_probability"]
    lo = next(i for i, v in enumerate(p) if v >= policy["min_clear_probability"])
    # Running the band from the lower cut to the strongest cell would exceed the budget.
    assert len(p) - lo > policy["max_forecast_cells"]
    hi = next(i for i in range(lo, len(p)) if max(p) - p[i] < policy["overpowered_gain_margin"])
    assert hi < len(p) - 1
    assert state["band"] == state["order"][lo:hi+1]
    assert len(band["baseline_request"]["cells"]) == len(state["band"]) <= policy["max_forecast_cells"]
    assert [c["classification"] for c in state["cells"][hi+1:]] == ["overpowered"] * (len(p) - hi - 1)
    # Stronger cells remain reachable, one at a time, through stronger-only widening.
    wider = engine.decide(band["request_id"], forecasts(band["baseline_request"], silent=.8))
    assert [c["model"] for c in wider["baseline_request"]["cells"]] == [state["cells"][hi+1]["model"]]
    result = engine.decide(wider["request_id"], forecasts(wider["baseline_request"]), seed=42)
    assert result["cap_met"] and len(result["capability"]["widening"]) == 1
    load(engine.config, result["decision_id"])


def test_learned_band_genuinely_wider_than_budget_still_refilters_then_refuses(tmp_path):
    cells = inventory(40)
    engine = failed_at(tmp_path, cells[0])
    pending = start(engine, cells)
    wide = filtering(pending["baseline_request"], 0, 39)
    retry = engine.decide(pending["request_id"], wide)
    assert retry["stage"] == "filter" and retry["refilter_round"] == 1
    with pytest.raises(HowHardError, match="max_forecast_cells.*tighter"):
        engine.decide(retry["request_id"], wide)
    # A tighter prior narrows this posterior's band within the budget.
    narrow = engine.decide(retry["request_id"], filtering(retry["baseline_request"], 0, 5))
    state = narrow["capability"]
    assert narrow["stage"] == "forecast"
    assert max(c["P_clear"] for c in state["cells"]) < state["policy"]["reliability_probability"]
    assert len(state["band"]) <= state["policy"]["max_forecast_cells"]


def test_overpowered_tail_band_stays_bounded_across_inventory_sizes(tmp_path):
    widths = {}
    for size in (20, 50, 100):
        cells = inventory(size)
        engine = failed_at(tmp_path / f"s{size}", cells[0])
        pending = start(engine, cells)
        band = engine.decide(pending["request_id"], filtering(pending["baseline_request"], 0, 5))
        assert band["stage"] == "forecast"
        widths[size] = len(band["capability"]["band"])
        assert widths[size] <= band["capability"]["policy"]["max_forecast_cells"]
    assert widths[50] <= widths[20]
    assert widths[100] <= widths[20]


def linear_v1_prior(value, policy):
    """The original v1 calculation, used to produce a frozen historical receipt."""
    import numpy as np

    order = value["order"]
    lo, hi = order.index(value["lower"]), order.index(value["upper"])
    minimum, reliable = policy["min_clear_probability"], policy["reliability_probability"]
    before = np.linspace(minimum * .2, minimum * .8, lo).tolist()
    fit = np.linspace(max(minimum, reliable - .2), reliable, hi-lo+1).tolist() if hi > lo else [reliable]
    tail = min((1-reliable)*.5, policy["overpowered_gain_margin"]*.5)
    after = np.linspace(reliable, reliable+tail, len(order)-hi).tolist()[1:]
    return dict(zip(order, before+fit+after))


@pytest.fixture
def frozen_v1_receipt(tmp_path, monkeypatch):
    from howhard import capability

    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    memory = engine.memory.add("Rollback", "Related capability guidance")
    # Generate and persist the whole decision with the historical prior, including
    # the resulting posterior and selection, rather than rewriting a v2 receipt.
    with monkeypatch.context() as patch:
        patch.setattr(capability, "FORMAT", "howhard-capability-v1")
        patch.setattr(capability, "prior_cdf", linear_v1_prior)
        pending = start(engine, inventory(6))
        response = filtering(pending["baseline_request"], 1, 2)
        response["memory_filters"] = [{"id": memory["id"], "version": memory["version"],
                                       "filter": deepcopy(response["filter"])}]
        result = finish(engine, pending, response)
        receipt = load(engine.config, result["decision_id"])
        engine.observe_decision(result["decision_id"], "silent", id="historical-run", cause="capability")
    assert receipt["format"] == "howhard-decision-v6"
    assert receipt["capability"]["format"] == "howhard-capability-v1"
    baseline = receipt["capability"]["baseline"]
    assert len(baseline["order"]) - 1 - baseline["order"].index(baseline["upper"]) >= 2
    return engine, result["decision_id"], receipt


@pytest.mark.parametrize("boundary", ["prepare", "observe", "beliefs", "learn", "report"])
def test_frozen_v1_receipt_loads_and_scores_at_public_boundaries(frozen_v1_receipt, boundary):
    from howhard.semantic import learning_request

    engine, decision_id, receipt = frozen_v1_receipt
    if boundary == "prepare":
        assert start(engine, inventory(6))["stage"] == "filter"
    elif boundary == "observe":
        assert engine.observe_decision(decision_id, "clean", id="new-run")["decision_id"] == decision_id
    elif boundary == "beliefs":
        expert = next(e for e in engine.memory_beliefs()["experts"] if e["key"] == "filter:llm")
        assert expert["evidence_count"] == 1
    elif boundary == "learn":
        request, _ = learning_request(engine, "historical-run")
        assert request["decision"] == receipt
        assert next(e for e in request["beliefs"]["experts"] if e["key"] == "filter:llm")["evidence_count"] == 1
    else:
        quality = report(engine)["filter_quality"]
        assert quality["decisions"] == 1 and len(quality["executions"]) == 1


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_versioned_receipt_rejects_tampered_prior(tmp_path, frozen_v1_receipt, version):
    engine, _, receipt = frozen_v1_receipt
    if version == "v2":
        engine = Engine(directory=tmp_path / "v2")
        pending = start(engine, inventory(6))
        result = finish(engine, pending, filtering(pending["baseline_request"], 1, 2))
        receipt = load(engine.config, result["decision_id"])
    assert receipt["capability"]["format"] == f"howhard-capability-{version}"
    world = receipt["capability"]["worlds"]["filter:llm"]
    world[next(iter(world))] += .001
    with pytest.raises(HowHardError, match="decision.capability: prior probabilities do not match the filter"):
        load(engine.config, receipt)


def test_new_capability_receipts_use_geometric_v2_prior(tmp_path):
    engine = Engine(directory=tmp_path)
    pending = start(engine, inventory(6))
    result = finish(engine, pending, filtering(pending["baseline_request"], 1, 2))
    state = load(engine.config, result["decision_id"])["capability"]
    assert state["format"] == "howhard-capability-v2"
    order = state["baseline"]["order"]
    reliable = state["policy"]["reliability_probability"]
    tail = min((1-reliable)*.5, state["policy"]["overpowered_gain_margin"]*.5)
    expected = [reliable + tail * (1 - .5**i) / (1 - .5**3) for i in range(1, 4)]
    assert [state["worlds"]["filter:llm"][key] for key in order[3:]] == pytest.approx(expected)
    assert state["worlds"]["filter:llm"] != linear_v1_prior(state["baseline"], state["policy"])


def test_unknown_capability_format_is_rejected(frozen_v1_receipt):
    engine, _, receipt = frozen_v1_receipt
    receipt["capability"]["format"] = "howhard-capability-v3"
    with pytest.raises(HowHardError, match="decision.capability.format: unsupported version"):
        load(engine.config, receipt)


def test_geometric_prior_labeled_v1_is_rejected(tmp_path):
    engine = Engine(directory=tmp_path)
    pending = start(engine, inventory(6))
    result = finish(engine, pending, filtering(pending["baseline_request"], 1, 2))
    receipt = load(engine.config, result["decision_id"])
    receipt["capability"]["format"] = "howhard-capability-v1"
    with pytest.raises(HowHardError, match="decision.capability: prior probabilities do not match the filter"):
        load(engine.config, receipt)
