"""The forecast band's host-aware edges and its reserved forecast slots (#131)."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from howhard import Engine, capability
from howhard.context import FORMAT as CONTEXT
from howhard.decisions import load
from howhard.report import report
from howhard.validation import HowHardError
from legacy_receipts import to_v10
from test_capability_filter import SimilarityDouble, _rewrite_stored_receipt, filtering, forecasts, inventory

POLICY = capability.DEFAULTS
# The worked example: 60 failed runs of all hosts, 42 caught; 20 of the asking host, 8 caught.
EXAMPLE = {"host": "alpha", "F_all": 60., "C_all": 42., "F_host": 20., "C_host": 8.}


def configured(tmp_path, settings, **options):
    """An engine whose own configuration file sets ``[capability]``; task type ``task`` is standard risk."""
    path = tmp_path / "howhard.toml"
    path.write_text('[capability]\n' + settings + '\n[task_types]\ntask = "standard"\n')
    return Engine(path, **options)


def labelled(count, agents):
    cells = inventory(count)
    for cell, agent in zip(cells, agents):
        cell["agent"] = agent
    return cells


def filtered(engine, cells, lower, upper, **options):
    """Prepare and answer the filter stage; the returned forecast request carries the capability state.

    The reader's order is by model, then by effort from low to max.
    """
    pending = engine.prepare("task", "Repair rollback", native_candidates=cells, **options)
    request = pending["baseline_request"]
    order = request and [c["key"] for c in sorted(request["cells"], key=lambda c: (
        c["model"], capability.EFFORT_ORDER[c["effort"]]))]
    pending = engine.decide(pending["request_id"], filtering(request, lower, upper, order))
    assert pending["stage"] == "forecast", pending
    return pending


def decided(engine, pending, silent=.01):
    while "decision_id" not in pending:
        pending = engine.decide(pending["request_id"], forecasts(pending["baseline_request"], silent=silent), seed=42)
    return pending


def models(state, keys):
    by_key = {c["key"]: c["model"] for c in state["cells"]}
    return [by_key[k] for k in keys]


def asked(pending):
    """The models whose cells the forecast request asks about."""
    return sorted(c["model"] for c in pending["request"]["cells"])


def fail(engine, decision_id, name, outcome, *, days=0., ticket=None):
    """Record a run of a stored decision's chosen cell, observed ``days`` ago, in the test's own log.

    The note keeps the run out of the capability threshold, so only the caught share reads it.
    """
    receipt = load(engine.config, decision_id)
    observed = datetime.now(timezone.utc) - timedelta(days=days)
    row = {"id": name, "date": observed.date().isoformat(), "ticket": ticket, "task_type": receipt["task_type"],
           "risk_class": receipt["risk_class"], "model": receipt["chosen"]["model"], "effort": receipt["chosen"]["effort"],
           "outcome": outcome, "observed_at": observed.isoformat(), "task_text": receipt["context"]["task_text"],
           "project_context": receipt["context"]["project_context"], "context_encoder": CONTEXT,
           "cause": None if outcome == "clean" else "unrecorded", "tokens": None, "seconds": None, "cost_usd": None,
           "note": "capability_caused=false; synthetic", "amends": None, "decision_id": decision_id}
    with open(engine.config.path(engine.config.data["logs"][0]["path"]), "a") as handle:
        handle.write(json.dumps(row) + "\n")
    engine.refresh()


def proven_inventory(engine):
    """Seven cells of three agents; the families of cells 0, 4, 5 and 6 have a run among the candidate runs.

    The runs do not move the capability threshold, so the reader's prior probabilities stand.
    """
    cells = labelled(7, ["z", "a", "a", "a", "a", "q", "a"])
    for index in (0, 4, 5, 6):
        engine.observe("task", cells[index]["model"], "high", "silent", native=True, task_text="Repair rollback",
                       note="capability_caused=false; outage", id=f"proven-{index}")
    return cells


def share_asked_by(engine, cells, host):
    return filtered(engine, cells, 0, len(cells) - 1, host=host)["capability"]["caught_share"]


def inject(monkeypatch, failed, caught):
    """Stand in for the caught-share reader: every host has ``failed`` failed runs, ``caught`` of them loud."""
    monkeypatch.setattr(capability, "caught_record", lambda engine, host, candidates: {
        "host": host, "F_all": float(failed), "C_all": float(caught), "F_host": float(failed), "C_host": float(caught)})


# The caught share.

def test_worked_example_gives_the_caught_share_of_all_hosts_and_of_the_asking_host():
    share = capability.caught_share(EXAMPLE, .10, POLICY)
    assert share["d_0"] == pytest.approx(.889, abs=5e-4)
    assert share["d_all"] == pytest.approx(.731, abs=5e-4)
    assert share["d_host"] == pytest.approx(.524, abs=5e-4)
    assert (share["e"], share["g"]) == pytest.approx((.811, .0117), abs=5e-4)
    assert {k: share[k] for k in EXAMPLE} == EXAMPLE


def test_empty_record_gives_the_share_at_which_both_edges_are_the_configured_ones():
    for epsilon in (.05, .10, .15, .20, .5, .7):
        share = capability.caught_share({"host": None, "F_all": 0., "C_all": 0., "F_host": 0., "C_host": 0.}, epsilon, POLICY)
        assert share["d_all"] == share["d_host"] == share["d_0"] == (1 - 2 * epsilon) / (1 - epsilon)
        assert (share["e"], share["g"]) == (.5, .05)


def test_a_host_with_few_failures_of_its_own_leans_on_all_hosts():
    few = capability.caught_share({"host": "alpha", "F_all": 60., "C_all": 42., "F_host": 1., "C_host": 0.}, .10, POLICY)
    many = capability.caught_share({"host": "alpha", "F_all": 60., "C_all": 42., "F_host": 40., "C_host": 0.}, .10, POLICY)
    assert few["d_all"] == many["d_all"] == pytest.approx(.731, abs=5e-4)
    assert few["d_host"] == pytest.approx((0 + 12 * few["d_all"]) / 13)
    assert many["d_host"] == pytest.approx((0 + 12 * many["d_all"]) / 52) and many["d_host"] < few["d_host"]


def test_failed_routed_runs_count_by_age_by_ticket_and_under_the_host_that_asked(tmp_path):
    engine = Engine(directory=tmp_path)
    cells = inventory(4)

    def decision(host):
        return decided(engine, filtered(engine, cells, 0, 3, host=host))["decision_id"]

    old, retried, other, unnamed = decision("alpha"), decision("alpha"), decision("beta"), decision(None)
    assert share_asked_by(engine, cells, "alpha") == {
        "host": "alpha", "F_all": 0., "C_all": 0., "F_host": 0., "C_host": 0.,
        "d_0": pytest.approx(8 / 9), "d_all": pytest.approx(8 / 9), "d_host": pytest.approx(8 / 9), "e": .5, "g": .05}
    fail(engine, old, "old-loud", "loud", days=14)                # one half-life old: counts half
    for name, outcome in (("retry-1", "loud"), ("retry-2", "silent"), ("retry-3", "silent")):
        fail(engine, retried, name, outcome, ticket="T-1")        # one ticket: a third each
    fail(engine, other, "beta-silent", "silent")
    fail(engine, unnamed, "unnamed-loud", "loud")
    # Neither a clean run nor a failure with no decision receipt is a failed routed run.
    fail(engine, other, "beta-clean", "clean")
    engine.observe("task", cells[0]["model"], "high", "loud", native=True, task_text="Repair rollback",
                   note="capability_caused=false; synthetic", id="raw-loud")

    alpha = share_asked_by(engine, cells, "alpha")
    assert alpha["host"] == "alpha"
    assert (alpha["F_all"], alpha["C_all"]) == pytest.approx((.5 + 1 + 1 + 1, .5 + 1 / 3 + 1), rel=1e-4)
    assert (alpha["F_host"], alpha["C_host"]) == pytest.approx((.5 + 1, .5 + 1 / 3), rel=1e-4)
    d_all = (alpha["C_all"] + 12 * alpha["d_0"]) / (alpha["F_all"] + 12)
    assert alpha["d_all"] == pytest.approx(d_all)
    assert alpha["d_host"] == pytest.approx((alpha["C_host"] + 12 * d_all) / (alpha["F_host"] + 12))
    assert alpha["d_host"] < alpha["d_0"] and alpha["e"] > .5 and alpha["g"] < .05
    beta = share_asked_by(engine, cells, "beta")
    assert (beta["F_host"], beta["C_host"]) == pytest.approx((1., 0.), rel=1e-4)
    unnamed = share_asked_by(engine, cells, None)
    assert (unnamed["F_host"], unnamed["C_host"]) == pytest.approx((1., 1.), rel=1e-4)
    newcomer = share_asked_by(engine, cells, "gamma")
    assert (newcomer["F_host"], newcomer["C_host"]) == (0., 0.) and newcomer["d_host"] == newcomer["d_all"]
    assert newcomer["F_all"] == pytest.approx(alpha["F_all"], rel=1e-4)


def test_a_receipt_older_than_v11_belongs_to_the_unnamed_host(tmp_path):
    engine = Engine(directory=tmp_path)
    cells = inventory(4)
    current = decided(engine, filtered(engine, cells, 0, 3, host="alpha"))["decision_id"]

    def older(receipt):
        legacy = to_v10(receipt)
        receipt.clear()
        receipt.update(legacy)

    legacy_id, legacy = _rewrite_stored_receipt(engine, current, older)
    assert legacy["format"] == "howhard-decision-v10" and "host" not in legacy
    fail(engine, legacy_id, "legacy-loud", "loud")
    assert share_asked_by(engine, cells, "alpha")["F_host"] == 0.
    unnamed = share_asked_by(engine, cells, None)
    assert (unnamed["F_all"], unnamed["F_host"], unnamed["C_host"]) == pytest.approx((1., 1., 1.), rel=1e-4)
    # The older receipt still loads and reports, with no host-aware band of its own.
    assert load(engine.config, legacy_id)["capability"]["format"] == "howhard-capability-v5"
    stats = report(engine)["filter_quality"]
    assert legacy_id not in {entry["decision_id"] for entry in stats["bands"]}
    assert [e["execution_id"] for e in stats["executions"]] == ["legacy-loud"]


# The two edges.

@pytest.mark.parametrize("d,e,g", [(.95, .5, .05), (.90, .5, .05), (.80, .643, .0278), (.50, .818, .0111), (0., .900, .0056)])
def test_edges_tighten_below_the_reference_share_and_are_the_configured_values_at_or_above_it(d, e, g):
    assert capability.band_edges(d, .10, POLICY) == pytest.approx((e, g), abs=5e-4)
    if d >= .9:
        assert capability.band_edges(d, .10, POLICY) == (.5, .05)


def test_edges_never_pass_the_configured_floor_and_widest_margin():
    policy = {**POLICY, "min_clear_probability": .85, "overpowered_gain_margin": .02}
    assert capability.band_edges(.80, .10, policy) == pytest.approx((.85, .02 * .1 / (.9 * .2)))
    assert capability.band_edges(.50, .10, {**policy, "min_clear_probability": .6}) == pytest.approx((.818, .02 * .1 / (.9 * .5)), abs=5e-4)
    assert capability.band_edges(.95, .10, policy) == (.85, .02)


def test_worked_example_moves_the_band_from_cells_two_to_six_to_cells_four_to_seven():
    order = [f"cell-{i}" for i in range(1, 9)]
    clear = dict(zip(order, [.30, .55, .72, .84, .91, .955, .985, .99]))
    share = capability.caught_share(EXAMPLE, .10, POLICY)
    assert capability._forecast_band(order, clear, clear, POLICY) == order[1:6]
    assert capability._forecast_band(order, clear, clear, POLICY, share=share) == order[3:7]
    generous = capability.caught_share({**EXAMPLE, "C_all": 60., "C_host": 20.}, .10, POLICY)
    assert capability._forecast_band(order, clear, clear, POLICY, share=generous) == order[1:6]


def test_timely_interval_does_not_depend_on_the_caught_share():
    order = [f"cell-{i}" for i in range(1, 13)]
    clear = dict(zip(order, [.30, .55, .72, .84, .91, .955, .985, .99, .99, .99, .99, .99]))
    timely = dict(zip(order, [0., 0., 0., 0., 0., .1, .2, .3, .4, .6, .8, .95]))
    policy = {**POLICY, "max_forecast_cells": 4}  # The span exceeds the budget, so the two intervals show apart.
    share = capability.caught_share(EXAMPLE, .10, policy)
    assert capability._forecast_band(order, clear, timely, policy) == order[1:6] + order[9:12]
    assert capability._forecast_band(order, clear, timely, policy, share=share) == order[3:7] + order[9:12]


def test_with_no_cell_at_the_lower_edge_the_band_is_the_first_best_cell_when_it_reaches_the_floor():
    order = ["a", "b", "c", "d"]
    share = capability.caught_share(EXAMPLE, .10, POLICY)
    clear = dict(zip(order, [.30, .60, .70, .70]))
    assert share["e"] > .70
    assert capability._forecast_band(order, clear, clear, POLICY, share=share) == ["c"]
    below = dict(zip(order, [.10, .20, .40, .45]))
    for edges in (None, share):
        with pytest.raises(HowHardError, match="no cell reaches the required capability probability"):
            capability._forecast_band(order, below, below, POLICY, share=edges)


@pytest.mark.parametrize("caught,e,g,band", [(.5, .818, .0111, (4, 7)), (.9, .5, .05, (2, 6))])
def test_a_host_that_catches_half_gets_a_tighter_band_and_one_that_catches_nine_in_ten_keeps_it(
        tmp_path, monkeypatch, caught, e, g, band):
    inject(monkeypatch, 1e6, caught * 1e6)
    engine = Engine(directory=tmp_path)
    pending = filtered(engine, inventory(8), 2, 5, host="alpha")
    state = pending["capability"]
    share = state["caught_share"]
    assert share["host"] == "alpha" and share["d_host"] == pytest.approx(caught, abs=1e-4)
    assert (share["e"], share["g"]) == pytest.approx((e, g), abs=5e-4)
    # Reader's prior: 0.2, 0.4, 0.7, 0.767, 0.833, 0.9, 0.917, 0.925.
    assert state["band"] == state["order"][band[0]:band[1]]
    assert all(c["P_clear"] >= share["e"] for c in state["cells"] if c["key"] in state["band"])
    receipt = load(engine.config, decided(engine, pending)["decision_id"])
    assert receipt["host"] == "alpha" and receipt["capability"]["caught_share"] == share


def test_a_host_that_catches_nothing_at_a_strict_limit_gets_the_first_best_cell(tmp_path, monkeypatch):
    inject(monkeypatch, 1e6, 0)
    engine = Engine(directory=tmp_path)
    # Critical risk: epsilon = 0.05, so e = 0.95 at d = 0, above every prior P_clear.
    state = filtered(engine, inventory(6), 1, 3, risk="critical")["capability"]
    assert state["caught_share"]["e"] == pytest.approx(.95, abs=1e-4)
    assert max(c["P_clear"] for c in state["cells"]) < state["caught_share"]["e"]
    assert state["band"] == state["order"][-1:]


def test_empty_record_keeps_the_readers_band_and_no_reserved_slots_keeps_the_forecast_cells(tmp_path):
    engine = configured(tmp_path, "reserved_forecast_cells = 0")
    pending = filtered(engine, inventory(8), 2, 5)
    state = pending["capability"]
    assert state["caught_share"] == {"host": None, "F_all": 0., "C_all": 0., "F_host": 0., "C_host": 0.,
                                     "d_0": pytest.approx(8 / 9), "d_all": pytest.approx(8 / 9),
                                     "d_host": pytest.approx(8 / 9), "e": .5, "g": .05}
    assert state["band"] == state["core"] == state["order"][2:6] and state["reserved"] == []
    assert asked(pending) == models(state, state["band"])
    receipt = load(engine.config, decided(engine, pending)["decision_id"])
    assert len(receipt["semantic"]["worlds"]["llm"]) == 4


# Reserved slots.

def test_unproven_families_nearest_to_the_core_take_the_reserved_slots(tmp_path):
    engine = Engine(directory=tmp_path)
    pending = filtered(engine, inventory(9), 3, 5)
    state = pending["capability"]
    order = state["order"]
    assert state["band"] == state["core"] == order[3:6] and state["proven_families"] == []
    # Both neighbours are one step away; the stronger one is first. Prior P_clear below the band: 0.1, 0.2, 0.4.
    assert state["reserved"] == [{"key": order[6], "reason": "unproven-family", "bar": .2},
                                 {"key": order[2], "reason": "unproven-family", "bar": .2}]
    assert asked(pending) == models(state, order[2:7])
    receipt = load(engine.config, decided(engine, pending)["decision_id"])
    classes = {c["key"]: c["classification"] for c in receipt["capability"]["cells"]}
    # A reserved cell keeps its position's classification.
    assert (classes[order[2]], classes[order[6]]) == ("underpowered", "overpowered")
    assert len(receipt["semantic"]["worlds"]["llm"]) == 5


def test_an_unproven_family_below_its_bar_takes_no_slot_and_the_slot_goes_back_unused(tmp_path):
    engine = configured(tmp_path, "unproven_clear_probability = 0.3")
    # Prior P_clear: 0.05, 0.1, 0.2, 0.4 below a band that ends at the strongest cell.
    pending = filtered(engine, inventory(6), 4, 5)
    state = pending["capability"]
    assert state["reserved"] == [{"key": state["order"][3], "reason": "unproven-family", "bar": .3}]
    assert asked(pending) == models(state, state["order"][3:])


def test_the_first_slot_goes_to_an_unproven_family_and_the_second_to_a_missing_agent(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    pending = filtered(engine, proven_inventory(engine), 2, 3)
    state = pending["capability"]
    order = state["order"]
    assert state["core"] == state["band"] and state["band"][0] == order[2] and order[5] not in state["band"]
    assert len(state["proven_families"]) == 4
    # Cell 0 is proven and its agent is missing, but its P_clear of 0.2 is below e. Cells 4 and 6
    # are proven and their agent has a core cell. Cell 1 is unproven; cell 5's agent has no core cell.
    assert state["caught_share"]["e"] == .5 and state["cells"][0]["P_clear"] == pytest.approx(.2)
    assert state["reserved"] == [{"key": order[1], "reason": "unproven-family", "bar": .2},
                                 {"key": order[5], "reason": "missing-agent", "bar": .5}]
    assert asked(pending) == sorted(models(state, state["band"] + [order[1], order[5]]))
    load(engine.config, decided(engine, pending)["decision_id"])


def test_with_one_reserved_slot_the_unproven_family_comes_before_the_missing_agent(tmp_path):
    engine = configured(tmp_path, "reserved_forecast_cells = 1", retrieval=SimilarityDouble())
    state = filtered(engine, proven_inventory(engine), 2, 3)["capability"]
    assert state["reserved"] == [{"key": state["order"][1], "reason": "unproven-family", "bar": .2}]


def test_with_no_family_to_reserve_for_the_forecast_cells_are_the_bands_first_cells(tmp_path):
    engine = configured(tmp_path, "max_forecast_cells = 4")
    # One model: every cell outside the core belongs to a family that has a core cell.
    cells = [{"model": "fixture/worker", "effort": effort} for effort in ("low", "medium", "high", "xhigh", "max")]
    pending = filtered(engine, cells, 1, 4)
    state = pending["capability"]
    assert len(state["band"]) == 4 and state["core"] == state["band"][:2] and state["reserved"] == []
    assert sorted(c["effort"] for c in pending["request"]["cells"]) == ["high", "max", "medium", "xhigh"]
    assert asked(pending) == ["fixture/worker"] * 4
    receipt = load(engine.config, decided(engine, pending)["decision_id"])
    assert len(receipt["semantic"]["worlds"]["llm"]) == 4


def test_no_decision_forecasts_more_than_the_budget_and_an_unfilled_slot_returns_to_the_band(tmp_path):
    engine = configured(tmp_path, "max_forecast_cells = 4\nunproven_clear_probability = 0.3")
    # Two weak models, then one model at four efforts, which is the band: a core of two.
    # The nearest weak cell has P_clear 0.4 and takes one slot; the other, at 0.2, is under
    # the bar, so its slot goes back and the band's third cell is forecast.
    cells = inventory(2) + [{"model": "fixture/worker", "effort": e} for e in ("low", "medium", "high", "xhigh")]
    pending = filtered(engine, cells, 2, 5)
    state = pending["capability"]
    order = state["order"]
    assert state["band"] == order[2:6] and state["core"] == order[2:4]
    assert [c["P_clear"] for c in state["cells"][:2]] == pytest.approx([.2, .4])
    assert state["reserved"] == [{"key": order[1], "reason": "unproven-family", "bar": .3}]
    assert {(c["model"], c["effort"]) for c in pending["request"]["cells"]} == {
        ("fixture/model-001", "high"), ("fixture/worker", "low"), ("fixture/worker", "medium"), ("fixture/worker", "high")}
    result = decided(engine, pending, silent=.8)
    receipt = load(engine.config, result["decision_id"])
    # The budget is spent, so the cap-unmet decision does not widen to the band's last cell.
    assert len(receipt["semantic"]["worlds"]["llm"]) == 4 and receipt["capability"]["widening"] == []
    assert receipt["capability"]["widening_stop"] == "forecast-budget"


def test_band_cells_beyond_the_core_compete_for_the_slots_as_families_with_no_core_cell(tmp_path):
    engine = configured(tmp_path, "max_forecast_cells = 4")
    pending = filtered(engine, inventory(6), 2, 5)
    state = pending["capability"]
    order = state["order"]
    assert state["band"] == order[2:6] and state["core"] == order[2:4]
    # The band's third cell and the cell below the band are both one step from the core; the stronger is first.
    assert [item["key"] for item in state["reserved"]] == [order[4], order[1]]
    assert asked(pending) == models(state, order[1:5])
    receipt = load(engine.config, decided(engine, pending)["decision_id"])
    classes = {c["key"]: c["classification"] for c in receipt["capability"]["cells"]}
    assert (classes[order[1]], classes[order[4]], classes[order[5]]) == ("underpowered", "fit", "fit")
    assert len(receipt["semantic"]["worlds"]["llm"]) == 4


def test_a_band_longer_than_the_budget_still_gets_one_refilter_and_then_fails(tmp_path):
    engine = configured(tmp_path, "max_forecast_cells = 4")
    pending = engine.prepare("task", "Repair rollback", native_candidates=inventory(8))
    wide = filtering(pending["baseline_request"], 1, 6)
    retry = engine.decide(pending["request_id"], wide)
    assert retry["stage"] == "filter" and retry["refilter_round"] == 1
    with pytest.raises(HowHardError, match="max_forecast_cells.*tighter"):
        engine.decide(retry["request_id"], wide)


# Widening.

def test_widening_skips_a_reserved_cell_and_takes_the_next_stronger_cell_with_no_forecast(tmp_path):
    engine = Engine(directory=tmp_path)
    pending = filtered(engine, inventory(8), 1, 2)
    state = pending["capability"]
    order = state["order"]
    assert [item["key"] for item in state["reserved"]] == [order[3], order[0]]
    wider = engine.decide(pending["request_id"], forecasts(pending["baseline_request"], silent=.8))
    assert wider["stage"] == "forecast" and asked(wider) == models(state, [order[4]])
    result = decided(engine, wider)
    assert result["cap_met"] and result["chosen"]["model"] == models(state, [order[4]])[0]
    receipt = load(engine.config, result["decision_id"])
    assert receipt["capability"]["widening"] == [{"round": 1, "cells": [order[4]], "trigger": "all-excluded"}]
    # A receipt that widened to the reserved cell instead does not follow from its stored numbers.
    invalid = deepcopy(receipt)
    invalid["capability"]["widening"][0]["cells"] = [order[3]]
    with pytest.raises(HowHardError, match="widening must forecast the next stronger cell with no forecast"):
        load(engine.config, invalid)


def test_widening_order_takes_band_cells_beyond_the_core_before_overpowered_cells():
    order = [f"cell-{i}" for i in range(8)]
    classes = ["underpowered"] + ["fit"] * 5 + ["overpowered"] * 2
    state = {"band": order[1:6], "core": order[1:4],
             "cells": [{"key": k, "classification": c} for k, c in zip(order, classes)]}
    forecasted = set(order[1:4]) | {order[6]}  # The core and a reserved overpowered cell.
    assert capability.widening_order(state, forecasted) == [order[4], order[5], order[7]]
    assert capability.widening_order(state, forecasted | {order[4]}) == [order[5], order[7]]


# The receipt and the report.

@pytest.mark.parametrize("corruption,message", [
    ("failed-runs", "do not follow from the recorded failed runs"),
    ("edge", "do not follow from the recorded failed runs"),
    ("record", "band does not match the frozen probabilities"),
    ("host", "does not belong to the host that asked"),
    ("caught-above-failed", "caught runs must be among the failed runs"),
    ("core", "core does not match the band"),
    ("reserved-dropped", "reserved cells do not follow"),
    ("reserved-reason", "reserved cells do not follow"),
    ("reserved-bar", "reserved cells do not follow"),
    ("reserved-cell", "reserved cells do not follow"),
    ("proven", "proven families do not match the scored candidates"),
    ("forecast", None),  # Refused by whichever reader first misses the forecast.
])
def test_receipt_whose_band_core_or_reserved_cells_do_not_follow_from_its_numbers_is_refused(
        tmp_path, corruption, message):
    engine = Engine(directory=tmp_path)
    receipt = load(engine.config, decided(engine, filtered(engine, inventory(9), 3, 5, host="alpha"))["decision_id"])
    state = receipt["capability"]
    share = state["caught_share"]
    if corruption == "failed-runs":
        share.update(F_all=40., C_all=10., F_host=30., C_host=5.)
    elif corruption == "edge":
        share["e"] = .6
    elif corruption == "record":
        # A consistent record under which the stored band is no longer the band.
        state["caught_share"] = capability.caught_share(
            {"host": "alpha", "F_all": 1e6, "C_all": 5e5, "F_host": 1e6, "C_host": 5e5}, .10, state["policy"])
    elif corruption == "host":
        share["host"] = "beta"
    elif corruption == "caught-above-failed":
        share.update(F_host=1., C_host=2.)
    elif corruption == "core":
        state["core"].pop()
    elif corruption == "reserved-dropped":
        state["reserved"].pop()
    elif corruption == "reserved-reason":
        state["reserved"][0]["reason"] = "missing-agent"
    elif corruption == "reserved-bar":
        state["reserved"][0]["bar"] = .1
    elif corruption == "reserved-cell":
        state["reserved"][1]["key"] = state["order"][0]
    elif corruption == "proven":
        state["proven_families"].append("unrecorded-family")
    else:
        dropped = next(c for c in state["cells"] if c["key"] == state["reserved"][1]["key"])
        for panel in receipt["semantic"]["worlds"].values():
            panel[:] = [row for row in panel if row["family"] != dropped["family"]]
    with pytest.raises(HowHardError, match=message):
        load(engine.config, receipt)


def test_report_shows_the_caught_share_edges_core_and_reserved_cells_of_each_decision(tmp_path, monkeypatch):
    inject(monkeypatch, 1e6, 5e5)
    engine = Engine(directory=tmp_path)
    result = decided(engine, filtered(engine, inventory(8), 2, 5, host="alpha"))
    state = load(engine.config, result["decision_id"])["capability"]
    assert report(engine)["filter_quality"]["bands"] == [{
        "decision_id": result["decision_id"], "caught_share": state["caught_share"], "core": state["core"],
        "proven_families": state["proven_families"], "reserved": state["reserved"]}]
    assert state["caught_share"]["host"] == "alpha" and state["caught_share"]["e"] == pytest.approx(.818, abs=5e-4)


# Configuration.

@pytest.mark.parametrize("settings", ["caught_share_worth = 0", "caught_share_worth = -1", "caught_share_worth = true",
                                      "reserved_forecast_cells = 12", "reserved_forecast_cells = -1",
                                      "reserved_forecast_cells = 1.5", "max_forecast_cells = 2",
                                      "unproven_clear_probability = 1.5", "unproven_clear_probability = -0.1"])
def test_host_aware_settings_are_validated_before_state(tmp_path, settings):
    with pytest.raises(HowHardError, match="capability"):
        configured(tmp_path, settings)
    assert not (tmp_path / ".howhard").exists()


def test_host_aware_settings_are_optional_and_the_worth_may_exceed_one(tmp_path):
    for name in ("defaults", "set"):
        (tmp_path / name).mkdir()
    assert {k: Engine(directory=tmp_path / "defaults").config.data["capability"][k] for k in capability.HOST_AWARE_SETTINGS} == {
        "caught_share_worth": 12., "reserved_forecast_cells": 2, "unproven_clear_probability": .2}
    engine = configured(tmp_path / "set", "caught_share_worth = 30\nreserved_forecast_cells = 0\nunproven_clear_probability = 0.0")
    assert engine.config.data["capability"]["caught_share_worth"] == 30
