"""The corrected prediction: per-host ladders, the weight on the forecaster and firmness (#139).

Every belief starts from a prediction: the level the same host's record shows on other
tasks, moved by how far this task's forecast sits from that forecaster's usual forecast.
Each decision here is routed through the public calling-agent interface. The record is
synthetic and owned by the test: a history decision is routed on a "billing" task, its
finished runs are appended to the project's outcome log on the day of the decision, each
under its own ticket, and the new "rollback" task matches none of their prototypes, so no
history run is an evidence candidate of the new decision.
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math

import pytest

from howhard import Engine, load_config
from howhard.capability import EFFORT_ORDER
from howhard.context import FORMAT, canonical, digest
from howhard.decisions import load
from howhard.payload_store import put
from howhard.prediction import forecaster_weight, value
from howhard.report import report
from howhard.storage import locked
from howhard.validation import HowHardError
from legacy_receipts import to_v7, to_v8, to_v9, to_v10
from test_capability_filter import filtering
from test_evidence_similarity import Scripted
from test_execution_accounting import artifact, card, codex_events, counts

TASK = "Repair rollback ordering"
OPUS, SONNET, HAIKU = "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"
FLAT = 1.869
POLICY = {"model_worth": 5., "rung_worth": 40., "task_worth_silent": 32., "task_worth_loud": 16.,
          "task_worth_amount": 1., "duration_miss": 1.3, "scatter_share": .5}
RUNGS = ("all-hosts", "host", "agent", "family", "cell")


def logit(p):
    return math.log(p / (1 - p))


def logistic(x):
    return 1 / (1 + math.exp(-x))


def cell(model, effort="high", **extra):
    return {"model": model, "effort": effort, "usd_per_mtok": 1.0, **extra}


def forecast(share=.10, *, loud=.20, seconds=600, tokens=1000, **extra):
    """A forecast row that guesses the accepted-defect share at ``share`` and the loud rate at ``loud``."""
    return {"clean": (1 - loud) * (1 - share), "loud": loud, "silent": (1 - loud) * share,
            "tokens": tokens, "seconds": seconds, **extra}


def route(engine, task_type, text, cells, rows=None, *, host=None, seed=7):
    """One routed decision; ``rows`` maps a model, or a (model, effort) pair, to that cell's forecast fields."""
    pending = engine.prepare(task_type, text, native_candidates=cells, host=host)
    if pending["baseline_request"] is None:
        pending = engine.decide(pending["request_id"], {"memory_filters": [], "reason": "Stored fit band"})
    else:
        ordered = sorted(pending["baseline_request"]["cells"], key=lambda c: (c["model"], EFFORT_ORDER[c["effort"]]))
        pending = engine.decide(pending["request_id"], filtering(pending["baseline_request"], 0, len(cells) - 1,
                                                                 [c["key"] for c in ordered]))
    if pending["baseline_request"] is None:
        assert rows is None, "a stored forecast would be reused in place of the fixture's"
        response = {"memory_forecasts": [], "reason": "Stored forecasts"}
    else:
        asked = pending["baseline_request"]["cells"]
        assert len(asked) == len(cells)
        response = {"forecasts": [{"family": c["family"], "effort": c["effort"],
                                   **(rows[c["model"], c["effort"]] if (c["model"], c["effort"]) in rows else rows[c["model"]])}
                                  for c in asked],
                    "memory_forecasts": [], "reason": "Synthetic prediction fixture"}
    result = engine.decide(pending["request_id"], response, seed=seed)
    assert "decision_id" in result, result
    return result


def record(engine, decision, outcomes, *, prefix, ticket=None, days=0., fields=None, decision_id=None):
    """Append finished runs of a routed decision to the outcome log, each under its own ticket unless one is given."""
    observed = datetime.now(timezone.utc) - timedelta(days=days)
    causes = {"clean": None, "interrupted": "cancelled"}
    rows = [{"id": f"{prefix}-{i}", "date": observed.date().isoformat(), "ticket": ticket or f"{prefix}-ticket-{i}",
             "task_type": decision["task_type"], "risk_class": decision["risk_class"],
             "model": decision["chosen"]["model"], "effort": decision["chosen"]["effort"], "outcome": outcome,
             "cause": causes.get(outcome, "unrecorded"), "tokens": None, "seconds": None, "cost_usd": None,
             "note": "", "amends": None, "observed_at": observed.isoformat(),
             "task_text": decision["context"]["task_text"], "project_context": None, "context_encoder": FORMAT,
             "decision_id": decision_id or decision["decision_id"], **(fields(i) if fields else {})}
            for i, outcome in enumerate(outcomes)]
    path = engine.config.path(engine.config.data["logs"][0]["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write("".join(json.dumps(row) + "\n" for row in rows))
    engine.refresh()


def history(engine, name, candidate, row, outcomes, *, host=None, task_type=None, **options):
    """Route one earlier decision on a billing task and record ``outcomes`` finished runs of it."""
    decision = route(engine, task_type or name, f"Tune billing {name}", [candidate], {candidate["model"]: row}, host=host)
    record(engine, decision, outcomes, prefix=name, **options)
    return decision


def silent(wrong, total):
    return ["silent"] * wrong + ["clean"] * (total - wrong)


def predicted(result, model, effort="high"):
    """The corrected prediction of one forecast cell: its entry for each value."""
    family = next(c["family"] for c in result["cells"] if (c["model"], c["effort"]) == (model, effort))
    block = result["semantic"]["prediction"]
    return next(c["values"] for c in block["cells"] if (c["family"], c["effort"]) == (family, effort))


def weights(result, kind="silent"):
    return result["semantic"]["prediction"]["kinds"][kind]


def rungs(entry, field="weight"):
    assert [r["rung"] for r in entry["rungs"]] == list(RUNGS)
    return [r[field] for r in entry["rungs"]]


def beliefs(result, model, effort="high"):
    family = next(c["family"] for c in result["cells"] if (c["model"], c["effort"]) == (model, effort))
    return next(c["outcome_beliefs"] for c in result["semantic"]["sampling"]["cells"]
                if (c["family"], c["effort"]) == (family, effort))


def selection_cell(result, model, effort="high"):
    return next(c for c in result["cells"] if (c["model"], c["effort"]) == (model, effort))


def minimum_spread(kind, entry, policy=POLICY):
    """``s_min = sigma_1 / sqrt(N_start)`` from the stored prediction, worth and miss."""
    if kind in ("loud", "silent"):
        task_worth = policy["task_worth_" + kind]
        one_run = 1 / math.sqrt(entry["prediction"] * (1 - entry["prediction"]))
    else:
        task_worth = policy["task_worth_amount"]
        one_run = entry["D"] * math.sqrt(policy["scatter_share"])
    n_start = 1 / (1 / entry["worth"] + 1 / task_worth)
    assert entry["N_start"] == pytest.approx(n_start)
    return one_run / math.sqrt(n_start)


# --- 1. One rule for every value -------------------------------------------------------------

def test_an_empty_record_leaves_every_forecast_as_it_was_merged(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    usage = counts(input=6000, output=2000, cache_read=4000)
    cells = [cell(OPUS), {"model": SONNET, "effort": "high", "rate_card": card(model=SONNET, rates=counts(
        input=2, output=10, cache_read=.2, cache_write=3, cache_write_5m=2.5, cache_write_1h=4))}]
    rows = {OPUS: forecast(.10, seconds=3600, silent_ci=[.04, .16]),
            SONNET: forecast(.25, loud=.05, tokens=12000, token_usage=usage, seconds_ci=[60, 6000])}
    result = route(engine, "task", TASK, cells, rows, host="host-x")
    assert result["sampling_policy"]["rung_worth"] == 40.

    block = result["semantic"]["prediction"]
    assert set(block) == {"kinds", "cells"}
    kinds = {"loud", "silent", "seconds", "tokens", "input", "output", "cache_read", "cache_write",
             "cache_write_5m", "cache_write_1h"}
    assert set(block["kinds"]) == kinds
    for measured in block["kinds"].values():
        # No run is on record: the forecaster's judgment keeps its full weight.
        assert set(measured) == {"n", "high", "low", "gamma_hat", "gamma"}
        assert (measured["n"], measured["gamma_hat"], measured["gamma"]) == (0., 1., 1.)
        assert measured["high"] == measured["low"] == {"weight": 0., "gap": 0., "outcome": 0., "level": 0.}

    plain, by_kind = predicted(result, OPUS), predicted(result, SONNET)
    assert set(plain) == {"loud", "silent", "seconds", "tokens"}
    assert set(by_kind) == {"loud", "silent", "seconds", "input", "output", "cache_read", "cache_write",
                            "cache_write_5m", "cache_write_1h"}
    for entries in (plain, by_kind):
        for kind, entry in entries.items():
            amount = kind not in ("loud", "silent")
            assert set(entry) == {"guess", "rungs", "level", "usual", "prediction", "worth", "N_start",
                                  "s_stated", "s", *(("D",) if amount else ())}
            assert all(total == 0. for field in ("weight", "outcome", "guess") for total in rungs(entry, field))
            assert entry["prediction"] == entry["guess"] and entry["worth"] == 5.
            if amount:
                assert entry["D"] == 1.3
            if entry["guess"] > 0:
                assert entry["level"] == entry["usual"] == entry["guess"]
    assert plain["silent"]["guess"] == pytest.approx(.10) and plain["loud"]["guess"] == pytest.approx(.20)
    assert plain["seconds"]["guess"] == pytest.approx(3600) and plain["tokens"]["guess"] == pytest.approx(1000)
    assert {kind: by_kind[kind]["guess"] for kind in usage} == pytest.approx(usage)
    # A kind forecast at zero stays exactly zero and has no spread.
    assert (by_kind["cache_write"]["prediction"], by_kind["cache_write"]["s"]) == (0., 0.)

    # Every belief starts from its prediction, at the firmness the prediction states.
    start = beliefs(result, OPUS)
    for rate in ("loud", "silent"):
        assert (start[rate]["prediction"], start[rate]["n_start"], start[rate]["spread"]) == (
            plain[rate]["prediction"], plain[rate]["N_start"], plain[rate]["s"])
    active = selection_cell(result, OPUS)["active_time"]
    assert (active["forecast_seconds"], active["prediction_seconds"]) == (plain["seconds"]["guess"],) * 2
    assert (active["worth"], active["N_start"], active["D"], active["prior_spread"]) == (
        5., plain["seconds"]["N_start"], 1.3, plain["seconds"]["s"])
    assert selection_cell(result, SONNET)["token_beliefs"]["cache_write"]["expected_tokens"] == 0.
    assert load(engine.config, result["decision_id"])["semantic"]["prediction"] == block


def test_the_worked_examples_of_the_rule_reproduce():
    # Level 0.0566, usual 0.188, guess 0.10 and gamma 0.74: -2.81 + 0.74 * (-2.20 + 1.46) = -3.36, or 0.034.
    table = [(160, 24, 16), (60, 6, 9), (40, 2, 6), (15, 1, 3), (10, 0, 2)]
    sums = [{"weight": float(a), "outcome": float(y), "guess": float(g)} for a, y, g in table]
    entry = value("silent", {"guess": .10, "s_stated": .46}, sums, .74, POLICY)
    assert entry["level"] == pytest.approx(.057, abs=5e-4) and entry["usual"] == pytest.approx(.188, abs=5e-4)
    assert logit(entry["prediction"]) == pytest.approx(-3.36, abs=5e-3)
    assert entry["prediction"] == pytest.approx(.034, abs=5e-4)
    # worth = 15 + 5 = 20 and N_start = 1 / (1/20 + 1/32) = 12.3; sigma_1 = 5.54 and s_min = 1.58.
    assert entry["worth"] == 20. and entry["N_start"] == pytest.approx(12.3, abs=.05)
    assert 1 / math.sqrt(entry["prediction"] * (1 - entry["prediction"])) == pytest.approx(5.54, abs=.02)
    # A stated range of 4% to 20% gives 0.46, which is narrower, so the belief starts at 1.58.
    assert (logit(.20) - logit(.04)) / 3.92 == pytest.approx(.46, abs=5e-3)
    assert entry["s"] == pytest.approx(1.58, abs=5e-3)
    # gamma = 1 moves the guess by level minus usual; gamma = 0 gives the level alone.
    moved = value("silent", {"guess": .10, "s_stated": 0.}, sums, 1., POLICY)["prediction"]
    assert logit(moved) == pytest.approx(logit(entry["level"]) + logit(.10) - logit(entry["usual"]))
    assert value("silent", {"guess": .10, "s_stated": 0.}, sums, 0., POLICY)["prediction"] == pytest.approx(entry["level"])

    # The weight: H holds 60 runs' worth with 13 silent and levels summing to 9, L 100 with 11 and 15.
    high = {"weight": 60., "gap": 60 * .75, "outcome": 13., "level": 9.}
    low = {"weight": 100., "gap": 100 * -.40, "outcome": 11., "level": 15.}
    assert logit(13.5 / 61) - logit(9.5 / 61) == pytest.approx(.43, abs=5e-3)
    assert logit(11.5 / 101) - logit(15.5 / 101) == pytest.approx(-.34, abs=5e-3)
    measured = forecaster_weight("silent", high, low, POLICY)
    assert measured["n"] == 160. and measured["gamma_hat"] == pytest.approx(.675, abs=1e-3)
    assert measured["gamma"] == pytest.approx(.74, abs=5e-3)
    # An empty group, or a forecast gap under 0.05, measures nothing: gamma_hat is 1.
    assert forecaster_weight("silent", high, {**low, "weight": 0.}, POLICY)["gamma_hat"] == 1.
    assert forecaster_weight("silent", {**high, "gap": 60 * -.36}, low, POLICY)["gamma_hat"] == 1.
    # An amount's observed gap is the average of y - level in each group.
    amount = forecaster_weight("seconds", {"weight": 10., "gap": 10 * .5, "outcome": 72., "level": 70.},
                               {"weight": 10., "gap": 10 * -.5, "outcome": 68., "level": 70.}, POLICY)
    assert amount["gamma_hat"] == pytest.approx(.4) and amount["gamma"] == pytest.approx((20 * .4 + 40) / 60)

    # The remaining miss: 100 runs' worth on one cell, each off by a factor of e^0.5.
    missed = [{"weight": 100., "outcome": 100 * math.log(1000), "guess": 100 * math.log(1000), "miss": 25.}] * 5
    assert value("seconds", {"guess": math.log(1000), "s_stated": 0., "exact_zero": False}, missed, 1., POLICY)["D"] == (
        pytest.approx(.50, abs=5e-3))
    squares, above = [], 1.3 ** 2
    for name in RUNGS:
        above = (25 + (5 if name == "family" else 40) * above) / (100 + (5 if name == "family" else 40))
        squares.append(above)
    assert squares == pytest.approx([.661, .368, .284, .252, .250], abs=1e-3)


# --- 2. Level and usual --------------------------------------------------------------------

def test_each_host_is_corrected_from_its_own_record(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    opus, sonnet = cell(OPUS), cell(SONNET)
    # Host X states S at 0.20 on one cell, four times the level of its results; host Y states
    # 0.05 on another. Each is silently wrong 10 times in 200 accepted results.
    history(engine, "invoices", opus, forecast(.20), silent(10, 200), host="host-x")
    history(engine, "refunds", sonnet, forecast(.05), silent(10, 200), host="host-y")
    under_x = route(engine, "task", TASK, [opus], {OPUS: forecast(.20)}, host="host-x")
    under_y = route(engine, "chore", "Repair rollback retries", [sonnet], {SONNET: forecast(.05)}, host="host-y")
    for result, model, guess in ((under_x, OPUS, .20), (under_y, SONNET, .05)):
        assert result["semantic"]["evidence_candidates"]["candidates"] == []
        entry = predicted(result, model)["silent"]
        assert entry["guess"] == pytest.approx(guess)
        # The prediction follows the host's results; Y's forecast is not moved by X's record.
        assert .045 <= entry["prediction"] <= .055
        assert rungs(entry) == pytest.approx([400, 200, 200, 200, 200], rel=1e-4)
        assert rungs(entry, "outcome") == pytest.approx([20, 10, 10, 10, 10], rel=1e-4)
        assert entry["level"] == pytest.approx(.05, abs=1e-3) and entry["usual"] == pytest.approx(guess, abs=1e-3)
        assert entry["worth"] == pytest.approx(205, rel=1e-4)
        assert beliefs(result, model)["silent"]["prediction"] == entry["prediction"]
        assert selection_cell(result, model)["P_silent_accepted"]["mean"] == pytest.approx(entry["prediction"], abs=1e-6)
        load(engine.config, result["decision_id"])
    assert rungs(predicted(under_x, OPUS)["silent"], "guess") == pytest.approx([50, 40, 40, 40, 40], rel=1e-4)
    assert rungs(predicted(under_y, SONNET)["silent"], "guess") == pytest.approx([50, 10, 10, 10, 10], rel=1e-4)
    # No run was loud: every finished run enters the loud rate's ladder with y = 0.
    assert rungs(predicted(under_x, OPUS)["loud"], "outcome") == [0.] * 5
    assert rungs(predicted(under_x, OPUS)["loud"]) == pytest.approx([400, 200, 200, 200, 200], rel=1e-4)


def test_a_new_family_takes_the_level_and_usual_of_its_agent(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    history(engine, "invoices", cell(OPUS, agent="alpha"), forecast(.20), silent(5, 100), host="host-x")
    result = route(engine, "task", TASK, [cell(SONNET, agent="alpha"), cell(HAIKU, agent="beta")],
                   {SONNET: forecast(.20), HAIKU: forecast(.20)}, host="host-x")
    entry = predicted(result, SONNET)["silent"]
    assert rungs(entry) == pytest.approx([100, 100, 100, 0, 0], rel=1e-4)
    # all hosts (5 + 40 * 0.2) / 140 = 0.0929, this host 0.0622, this agent 0.0535; the two empty
    # rungs below pass the agent rung's level and usual through.
    assert entry["level"] == pytest.approx((5 + 40 * (5 + 40 * (5 + 40 * .2) / 140) / 140) / 140, rel=1e-4)
    assert entry["usual"] == pytest.approx(.20, abs=1e-6)
    assert .05 <= entry["prediction"] <= .06 and entry["prediction"] == pytest.approx(entry["level"], rel=1e-6)
    assert entry["worth"] == 5.
    # A family of another agent under the same host leans on the host rung instead.
    other = predicted(result, HAIKU)["silent"]
    assert rungs(other) == pytest.approx([100, 100, 0, 0, 0], rel=1e-4)
    assert other["level"] == pytest.approx((5 + 40 * (5 + 40 * .2) / 140) / 140, rel=1e-4) and other["worth"] == 5.


def test_a_cell_leans_on_its_family_agent_host_and_all_hosts(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    # The worked table for S: the cell's own 10 runs, 5 more of its family, 25 more of its agent,
    # 20 more of its host and 100 of another host.
    history(engine, "invoices", cell(OPUS, agent="alpha"), forecast(.20), silent(0, 10), host="host-a")
    history(engine, "refunds", cell(OPUS, "low", agent="alpha"), forecast(.20), silent(1, 5), host="host-a")
    history(engine, "ledgers", cell(SONNET, agent="alpha"), forecast(.12), silent(1, 25), host="host-a")
    history(engine, "taxes", cell(HAIKU, agent="beta"), forecast(.15), silent(4, 20), host="host-a")
    history(engine, "payouts", cell(HAIKU, agent="beta"), forecast(.07), silent(18, 100), host="host-b")
    result = route(engine, "task", TASK, [cell(OPUS, agent="alpha")],
                   {OPUS: forecast(.10, silent_ci=[.8 * .04, .8 * .20])}, host="host-a")
    entry = predicted(result, OPUS)["silent"]
    assert rungs(entry) == pytest.approx([160, 60, 40, 15, 10], rel=1e-4)
    assert rungs(entry, "outcome") == pytest.approx([24, 6, 2, 1, 0], rel=1e-4)
    assert rungs(entry, "guess") == pytest.approx([16, 9, 6, 3, 2], rel=1e-4)
    assert entry["level"] == pytest.approx(.057, abs=5e-4) and entry["usual"] == pytest.approx(.188, abs=5e-4)
    # The family rung leans on the agent rung at the worth of 5 runs, every other rung at 40.
    levels, usuals, level, usual = [], [], .10, .10
    for name, a, y, g in zip(RUNGS, (160, 60, 40, 15, 10), (24, 6, 2, 1, 0), (16, 9, 6, 3, 2)):
        kappa = 5 if name == "family" else 40
        level, usual = (y + kappa * level) / (a + kappa), (g + kappa * usual) / (a + kappa)
        levels.append(level)
        usuals.append(usual)
    assert levels == pytest.approx([.140, .116, .083, .071, .057], abs=5e-4)
    assert usuals == pytest.approx([.100, .130, .140, .185, .188], abs=5e-4)
    gamma = weights(result)["gamma"]
    assert 0 < gamma <= 1
    assert logit(entry["prediction"]) == pytest.approx(logit(level) + gamma * (logit(.10) - logit(usual)), abs=1e-3)
    # The family's own 15 runs set the firmness: worth 20, N_start 12.3; the stated range is narrower.
    assert entry["worth"] == pytest.approx(20, rel=1e-4) and entry["N_start"] == pytest.approx(12.3, abs=.05)
    assert entry["s_stated"] == pytest.approx(.46, abs=5e-3)
    assert entry["s"] == pytest.approx(minimum_spread("silent", entry)) and entry["s"] > entry["s_stated"]
    load(engine.config, result["decision_id"])


def test_ten_runs_of_one_ticket_count_as_one_run(tmp_path):
    def all_hosts(directory, **options):
        directory.mkdir()
        engine = Engine(directory=directory, retrieval=Scripted())
        history(engine, "invoices", cell(OPUS), forecast(.20, seconds=600), ["clean"] * 10,
                fields=lambda i: {"active_seconds": 300.}, **options)
        values = predicted(route(engine, "task", TASK, [cell(OPUS)], {OPUS: forecast(.20)}), OPUS)
        return {kind: rungs(values[kind])[0] for kind in ("loud", "silent", "seconds")}
    assert all_hosts(tmp_path / "one", ticket="T-1") == pytest.approx({"loud": 1, "silent": 1, "seconds": 1}, rel=1e-4)
    assert all_hosts(tmp_path / "ten") == pytest.approx({"loud": 10, "silent": 10, "seconds": 10}, rel=1e-4)


def test_a_run_counts_half_as_much_for_every_half_life_of_age(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    history(engine, "invoices", cell(OPUS), forecast(.20), silent(2, 8), days=14)
    entry = predicted(route(engine, "task", TASK, [cell(OPUS)], {OPUS: forecast(.20)}), OPUS)["silent"]
    assert rungs(entry) == pytest.approx([4] * 5, rel=1e-3)
    assert rungs(entry, "outcome") == pytest.approx([1] * 5, rel=1e-3)


def test_candidates_runs_without_a_decision_and_interrupted_runs_enter_no_ladder(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    opus = cell(OPUS)
    earlier = history(engine, "invoices", opus, forecast(.20), silent(1, 5))
    first = route(engine, "task", TASK, [opus], {OPUS: forecast(.20)})
    before = predicted(first, OPUS)
    assert rungs(before["silent"]) == pytest.approx([5] * 5, rel=1e-4)

    # A run of the same task is an evidence candidate of the next decision on it.
    engine.observe_decision(first["decision_id"], "silent", id="same-task", ticket="same-task", active_seconds=50.)
    # A run with no decision, and an interrupted run of a routed decision.
    engine.observe("invoices", OPUS, "high", "silent", task_text="Tune billing invoices", id="unrouted",
                   ticket="unrouted", active_seconds=50.)
    record(engine, earlier, ["interrupted"], prefix="stopped")
    second = route(engine, "task", TASK, [opus])
    assert [c["id"] for c in second["semantic"]["evidence_candidates"]["candidates"]] == ["same-task"]
    after = predicted(second, OPUS)
    for kind in ("loud", "silent", "seconds", "tokens"):
        for field in ("weight", "outcome", "guess"):
            assert rungs(after[kind], field) == pytest.approx(rungs(before[kind], field), rel=1e-4)
    for kind, measured in second["semantic"]["prediction"]["kinds"].items():
        assert measured["n"] == pytest.approx(first["semantic"]["prediction"]["kinds"][kind]["n"], rel=1e-4)
    # The candidate counts once, through its similarity, in the belief the prediction starts.
    assert beliefs(second, OPUS)["silent"]["counts"][0] > 0
    assert beliefs(second, OPUS)["silent"]["prediction"] == after["silent"]["prediction"]


def test_runs_of_receipts_older_than_v11_sit_under_the_unnamed_host_and_agent(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    named = history(engine, "invoices", cell(OPUS, agent="alpha"), forecast(.20), [], host="host-x")
    older = to_v10(load(engine.config, named["decision_id"]))
    with locked(engine.config):
        put(engine.config, "decisions", digest(older), canonical(older) + b"\n")
    record(engine, named, silent(2, 20), prefix="older", decision_id=digest(older))

    under_named = route(engine, "task", TASK, [cell(OPUS, agent="alpha")], {OPUS: forecast(.20)}, host="host-x")
    entry = predicted(under_named, OPUS)["silent"]
    assert rungs(entry) == pytest.approx([20, 0, 0, 0, 0], rel=1e-4)
    # The older receipt gives its guess by the same merge rule.
    assert rungs(entry, "guess")[0] == pytest.approx(20 * .20, rel=1e-4)
    unnamed = route(engine, "chore", "Repair rollback retries", [cell(OPUS)], {OPUS: forecast(.20)})
    assert rungs(predicted(unnamed, OPUS)["silent"]) == pytest.approx([20] * 5, rel=1e-4)


# --- 3. The weight on the forecaster ---------------------------------------------------------

@pytest.mark.parametrize("wrong_high, wrong_low", [(8, 8), (16, 4)])
def test_the_weight_is_the_gap_that_happened_over_the_gap_that_was_forecast(tmp_path, wrong_high, wrong_low):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    opus = cell(OPUS)
    history(engine, "invoices", opus, forecast(.20), silent(wrong_high, 80))
    history(engine, "refunds", opus, forecast(.05), silent(wrong_low, 80))
    result = route(engine, "task", TASK, [opus], {OPUS: forecast(.10)})
    measured = weights(result)
    assert measured["n"] == pytest.approx(160, rel=1e-4)
    assert (measured["high"]["weight"], measured["low"]["weight"]) == pytest.approx((80, 80), rel=1e-4)
    # The two groups were forecast logit(0.20) - logit(0.05) = 1.56 apart.
    forecast_gap = measured["high"]["gap"] / measured["high"]["weight"] - measured["low"]["gap"] / measured["low"]["weight"]
    assert forecast_gap == pytest.approx(logit(.20) - logit(.05), abs=1e-3)
    assert (measured["high"]["outcome"], measured["low"]["outcome"]) == pytest.approx((wrong_high, wrong_low), rel=1e-4)
    if wrong_high == wrong_low:
        # Both groups were wrong equally often: the forecaster's "higher" meant nothing.
        assert measured["gamma_hat"] == pytest.approx(0., abs=1e-9)
        assert measured["gamma"] == pytest.approx(.2, abs=1e-4)
        assert predicted(result, OPUS)["silent"]["prediction"] < .10
    else:
        observed = logit(16.5 / 81) - logit(4.5 / 81)
        assert measured["gamma_hat"] == pytest.approx(observed / forecast_gap, abs=1e-3) and measured["gamma_hat"] > .9
        assert measured["gamma"] == pytest.approx((160 * measured["gamma_hat"] + 40) / 200, abs=1e-4)
    # One weight per kind of value: no run was loud, and both groups guessed the loud rate alike.
    assert weights(result, "loud")["gamma_hat"] == 1.
    entry = predicted(result, OPUS)["silent"]
    assert logit(entry["prediction"]) == pytest.approx(
        logit(entry["level"]) + measured["gamma"] * (logit(.10) - logit(entry["usual"])), abs=1e-9)
    load(engine.config, result["decision_id"])


# --- 4 and 5. Firmness and the remaining miss -------------------------------------------------

def test_the_remaining_miss_of_active_time_is_the_typical_miss_after_the_correction(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    opus = cell(OPUS)
    history(engine, "invoices", opus, forecast(.10, seconds=1000), ["clean"] * 100,
            fields=lambda i: {"active_seconds": 1000 * math.exp(.5 if i % 2 else -.5)})
    result = route(engine, "task", TASK, [opus], {OPUS: forecast(.10, seconds=1000)})
    entry = predicted(result, OPUS)["seconds"]
    assert weights(result, "seconds")["n"] == pytest.approx(100, rel=1e-4)
    assert weights(result, "seconds")["gamma"] == 1.
    assert rungs(entry, "miss") == pytest.approx([25] * 5, rel=1e-4)
    assert .49 <= entry["D"] <= .51
    # The runs average 1,000 seconds on the log scale, so the level, the usual forecast and the prediction agree.
    assert (entry["level"], entry["usual"], entry["prediction"]) == pytest.approx((1000,) * 3, rel=1e-4)
    # The belief is as firm as the family's 100 runs allow, with tau = D * sqrt(1/2).
    assert entry["worth"] == pytest.approx(105, rel=1e-4)
    assert entry["s"] == pytest.approx(entry["D"] * math.sqrt(.5) / math.sqrt(1 / (1 / entry["worth"] + 1)))
    active = selection_cell(result, OPUS)["active_time"]
    assert (active["D"], active["worth"], active["N_start"], active["prior_spread"]) == (
        entry["D"], entry["worth"], entry["N_start"], entry["s"])
    assert active["prior_centre"] == pytest.approx(math.log(entry["prediction"]) - entry["s"] ** 2 / 2)
    assert active["expected_seconds"] == pytest.approx(1000, rel=1e-4)
    # No run recorded tokens: the token ladder is empty and keeps the configured miss.
    assert predicted(result, OPUS)["tokens"]["D"] == 1.3 and weights(result, "tokens")["n"] == 0.
    load(engine.config, result["decision_id"])


def test_total_tokens_follow_the_record_and_a_stated_range_can_only_widen(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    opus = cell(OPUS)
    # Forecast at 1,000 tokens and 600 seconds; every run recorded four times as much.
    history(engine, "invoices", opus, forecast(.10, seconds=600, tokens=1000), ["clean"] * 60,
            fields=lambda i: {"tokens": 4000, "active_seconds": 2400.})
    narrow = route(engine, "task", TASK, [opus], {OPUS: forecast(.10, seconds=600, seconds_ci=[590, 610])})
    wide = route(engine, "chore", "Repair rollback retries", [opus],
                 {OPUS: forecast(.10, seconds=600, seconds_ci=[6, 60000], loud_ci=[.001, .999], silent_ci=[0, .8])})
    for result in (narrow, wide):
        values = predicted(result, OPUS)
        for kind in ("tokens", "seconds"):
            entry = values[kind]
            # Down the rungs the level closes in on what the runs recorded, four times the usual forecast.
            assert entry["level"] / entry["usual"] == pytest.approx(4, rel=5e-3)
            assert entry["prediction"] == pytest.approx(4 * entry["guess"], rel=5e-3)
            # Every run missed by the same factor, which the correction removes.
            assert entry["D"] < .2
        for kind, entry in values.items():
            floor = minimum_spread(kind, entry)
            if kind in ("loud", "silent"):
                assert entry["s"] == pytest.approx(min(FLAT, max(entry["s_stated"], floor))) and entry["s"] <= FLAT
            else:
                assert entry["s"] == pytest.approx(max(entry["s_stated"], floor)) and entry["s"] >= floor
        row = selection_cell(result, OPUS)
        assert row["token_beliefs"]["tokens"]["expected_tokens"] == pytest.approx(values["tokens"]["prediction"])
        assert row["T"] * 3600 == pytest.approx(values["seconds"]["prediction"])
    # A range narrower than the record allows leaves the minimum; a wider one is used, up to the flat belief.
    assert predicted(narrow, OPUS)["seconds"]["s"] == pytest.approx(minimum_spread("seconds", predicted(narrow, OPUS)["seconds"]))
    stated = predicted(wide, OPUS)
    assert stated["seconds"]["s"] == stated["seconds"]["s_stated"] == pytest.approx(math.log(10000) / 3.92)
    assert stated["loud"]["s"] == stated["silent"]["s"] == FLAT
    assert beliefs(wide, OPUS)["silent"]["spread"] == FLAT


def test_each_token_kind_has_its_own_ladder_once_runs_record_usage_by_kind(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    rates = counts(input=2, output=10, cache_read=.2, cache_write=3, cache_write_5m=2.5, cache_write_1h=4)
    worker = {"model": OPUS, "effort": "high", "rate_card": card(model=OPUS, rates=rates)}
    usage = counts(input=600, output=20, cache_read=40)
    earlier = route(engine, "invoices", "Tune billing invoices", [worker],
                    {OPUS: forecast(.10, tokens=660, token_usage=usage)})
    for i in range(6):
        # Each run's owned usage, from a session of its own, records 60 input, 20 output and 40 cached tokens.
        engine.observe_decision(earlier["decision_id"], "clean", id=f"measured-{i}", ticket=f"measured-{i}",
                                active_seconds=600.)
        owned = tmp_path / f"session-{i}"
        owned.mkdir()
        events = [{"type": "thread.started", "thread_id": f"session-{i}"}, codex_events()[1]]
        engine.attach_usage(f"measured-{i}", artifact(owned, events, execution_id=f"measured-{i}", model=OPUS,
                                                      session_id=f"session-{i}"), id=f"measured-usage-{i}")
    record(engine, earlier, ["clean"] * 4, prefix="unmeasured")
    result = route(engine, "task", TASK, [worker], {OPUS: forecast(.10, tokens=660, token_usage=usage)})
    values = predicted(result, OPUS)
    assert weights(result, "loud")["n"] == pytest.approx(10, rel=1e-4)
    for kind, recorded in (("input", 60), ("output", 20), ("cache_read", 40)):
        entry = values[kind]
        # Only the six runs with complete usage enter a kind's ladder.
        assert weights(result, kind)["n"] == pytest.approx(6, rel=1e-4)
        assert rungs(entry) == pytest.approx([6] * 5, rel=1e-4)
        assert rungs(entry, "outcome") == pytest.approx([6 * math.log(recorded)] * 5, rel=1e-4)
        assert rungs(entry, "guess") == pytest.approx([6 * math.log(usage[kind])] * 5, rel=1e-4)
    # Input was forecast at ten times what the runs used; the other two kinds as recorded.
    assert values["input"]["prediction"] < .5 * values["input"]["guess"]
    assert values["output"]["prediction"] == pytest.approx(20, rel=1e-4)
    for kind in ("cache_write", "cache_write_5m", "cache_write_1h"):
        # A kind forecast at zero enters no ladder and stays zero.
        assert weights(result, kind)["n"] == 0. and rungs(values[kind]) == [0.] * 5
        assert (values[kind]["guess"], values[kind]["prediction"], values[kind]["s"]) == (0., 0., 0.)
    assert "tokens" not in values and weights(result, "tokens")["n"] == pytest.approx(6, rel=1e-4)
    load(engine.config, result["decision_id"])


# --- Configuration and receipt ---------------------------------------------------------------

def test_rung_worth_is_an_optional_sampling_setting(tmp_path, project):
    config = project(settings="[sampling]\nrung_worth = 10.0\n")
    assert config.data["sampling"]["rung_worth"] == 10.
    engine = Engine(config, retrieval=Scripted())
    history(engine, "invoices", cell(OPUS), forecast(.20), silent(5, 100), host="host-x", task_type="X")
    result = route(engine, "Y", TASK, [cell(SONNET)], {SONNET: forecast(.20)}, host="host-x")
    assert result["sampling_policy"]["rung_worth"] == 10.
    # Every rung above the family now leans on the rung above it at the worth of 10 runs.
    entry = predicted(result, SONNET)["silent"]
    assert entry["level"] == pytest.approx((5 + 10 * (5 + 10 * (5 + 10 * .2) / 110) / 110) / 110, rel=1e-4)
    for bad in ("0", "-1", '"forty"'):
        path = tmp_path / f"bad-{len(bad)}.toml"
        path.write_text(f"[sampling]\nrung_worth = {bad}\n")
        with pytest.raises(HowHardError, match="rung_worth"):
            load_config(path)


@pytest.fixture(scope="module")
def corrected(tmp_path_factory):
    """One stored decision whose prediction a record has moved, with its engine."""
    engine = Engine(directory=tmp_path_factory.mktemp("corrected"), retrieval=Scripted())
    opus = cell(OPUS, agent="alpha")
    history(engine, "invoices", opus, forecast(.20, seconds=1000), silent(6, 40), host="host-x",
            fields=lambda i: {"active_seconds": 1000 * math.exp(.5 if i % 2 else -.5), "tokens": 3000})
    history(engine, "refunds", opus, forecast(.05, seconds=2000), silent(3, 40) + ["loud"] * 4, host="host-x",
            fields=lambda i: {"active_seconds": 1500., "tokens": 500})
    result = route(engine, "task", TASK, [opus, cell(SONNET, agent="alpha")],
                   {OPUS: forecast(.15, seconds=1200), SONNET: forecast(.15, seconds=900)}, host="host-x")
    return engine, load(engine.config, result["decision_id"])


def stored(receipt, kind="silent", family="opus"):
    """One value of the cell that has a record of its own."""
    return next(c["values"][kind] for c in receipt["semantic"]["prediction"]["cells"]
                if (c["family"], c["effort"]) == (family, "high"))


def tamper_cases():
    block = lambda r: r["semantic"]["prediction"]
    return [
        (lambda r: block(r)["kinds"]["silent"].update(gamma=block(r)["kinds"]["silent"]["gamma"] * .9), r"kinds\.silent\.gamma"),
        (lambda r: block(r)["kinds"]["silent"].update(gamma_hat=.123), r"kinds\.silent\.gamma_hat"),
        (lambda r: block(r)["kinds"]["seconds"].update(n=1.), r"kinds\.seconds\.n"),
        (lambda r: block(r)["kinds"]["silent"]["high"].update(outcome=block(r)["kinds"]["silent"]["high"]["outcome"] + 5), r"kinds\.silent"),
        (lambda r: block(r)["kinds"]["silent"]["low"].update(outcome=1e6), "cannot exceed its weight"),
        (lambda r: block(r)["kinds"].pop("tokens"), "missing keys"),
        (lambda r: stored(r)["rungs"][4].update(outcome=stored(r)["rungs"][4]["outcome"] + 3), r"values\.silent\.level"),
        (lambda r: stored(r)["rungs"][3].update(weight=stored(r)["rungs"][3]["weight"] - 10), r"values\.silent"),
        (lambda r: stored(r)["rungs"][4].update(weight=1e6), "cannot outweigh"),
        (lambda r: stored(r)["rungs"].reverse(), "from all hosts down to the cell"),
        (lambda r: stored(r)["rungs"].pop(), "five rungs"),
        (lambda r: stored(r).update(level=stored(r)["level"] * 1.01), r"values\.silent\.level"),
        (lambda r: stored(r).update(usual=stored(r)["usual"] * 1.01), r"values\.silent\.usual"),
        (lambda r: stored(r).update(prediction=stored(r)["guess"]), r"values\.silent\.prediction"),
        (lambda r: stored(r).update(guess=stored(r)["guess"] * 1.1), r"values\.silent\.guess"),
        (lambda r: stored(r).update(worth=5.), r"values\.silent\.worth"),
        (lambda r: stored(r).update(N_start=4.32), r"values\.silent\.N_start"),
        (lambda r: stored(r).update(s=stored(r)["s"] * .5), r"values\.silent\.s"),
        (lambda r: stored(r, "loud").update(s_stated=.3), r"values\.loud\.s_stated"),
        (lambda r: stored(r, "seconds").update(D=1.3), r"values\.seconds\.D"),
        (lambda r: stored(r, "seconds")["rungs"][0].update(miss=stored(r, "seconds")["rungs"][0]["miss"] * 2), r"values\.seconds\.D"),
        (lambda r: stored(r, "tokens").update(prediction=stored(r, "tokens")["guess"]), r"values\.tokens\.prediction"),
        (lambda r: stored(r).pop("usual"), "missing keys"),
        (lambda r: stored(r).update(D=1.3), "unknown keys"),
        (lambda r: block(r)["cells"].pop(), "cover the forecast cells|unknown or duplicate"),
        (lambda r: block(r)["cells"].append(deepcopy(block(r)["cells"][0])), "unknown or duplicate"),
        (lambda r: r["semantic"].pop("prediction"), "missing keys .'prediction'"),
        (lambda r: [p.pop("rung_worth") for p in (r["selection"]["sampling_policy"], r["semantic"]["sampling"]["policy"])], "rung_worth"),
    ]


def test_the_stored_prediction_follows_from_its_stored_sums(corrected):
    engine, receipt = corrected
    assert load(engine.config, deepcopy(receipt)) == receipt
    entry, measured = stored(receipt), receipt["semantic"]["prediction"]["kinds"]["silent"]
    # The record moved this prediction: the level, the usual forecast and the weight all differ from the guess.
    assert entry["guess"] == pytest.approx(.15) and abs(entry["prediction"] - entry["guess"]) > .005
    assert measured["n"] == pytest.approx(80, rel=1e-4) and 0 < measured["gamma"] < 1
    assert measured == {"n": measured["n"], "high": measured["high"], "low": measured["low"],
                        **forecaster_weight("silent", measured["high"], measured["low"], POLICY)}
    for kind in ("loud", "silent", "seconds", "tokens"):
        entry = stored(receipt, kind)
        sums = [{k: v for k, v in rung.items() if k != "rung"} for rung in entry["rungs"]]
        guess = entry["guess"] if kind in ("loud", "silent") else math.log(entry["guess"])
        again = value(kind, {"guess": guess, "s_stated": entry["s_stated"], "exact_zero": False}, sums,
                      receipt["semantic"]["prediction"]["kinds"][kind]["gamma"], POLICY)
        assert {k: v for k, v in entry.items() if k != "rungs"} == pytest.approx({k: v for k, v in again.items() if k != "rungs"})
    assert stored(receipt, "seconds")["D"] < 1.3


@pytest.mark.parametrize("index", range(len(tamper_cases())))
def test_a_receipt_whose_numbers_do_not_follow_from_its_sums_is_refused(corrected, index):
    engine, receipt = corrected
    receipt = deepcopy(receipt)
    change, match = tamper_cases()[index]
    change(receipt)
    with pytest.raises(HowHardError, match=match):
        load(engine.config, receipt)


def test_v7_to_v10_receipts_load_observe_replay_and_report_unchanged(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=Scripted())
    opus = cell(OPUS)
    history(engine, "invoices", opus, forecast(.20, seconds=1000), silent(4, 30),
            fields=lambda i: {"active_seconds": 500.})
    current = route(engine, "task", TASK, [opus], {OPUS: forecast(.10, seconds=1000)})
    receipt = load(engine.config, current["decision_id"])
    assert receipt["format"] == "howhard-decision-v11" and "prediction" in receipt["semantic"]
    assert stored(receipt)["prediction"] != stored(receipt)["guess"]

    ids = {}
    for name, build in (("v10", to_v10), ("v9", to_v9), ("v8", to_v8), ("v7", to_v7)):
        legacy = build(receipt)
        ids[name] = digest(legacy)
        with locked(engine.config):
            put(engine.config, "decisions", ids[name], canonical(legacy) + b"\n")
        loaded = load(engine.config, ids[name])
        assert loaded == legacy and loaded["format"] == f"howhard-decision-{name}"
        # An older receipt has no corrected prediction and no rung worth, and refuses either.
        assert "prediction" not in loaded["semantic"]
        assert "rung_worth" not in loaded["selection"]["sampling_policy"]
        for change, match in ((lambda r: r["semantic"].update(prediction=receipt["semantic"]["prediction"]), "unknown keys .'prediction'"),
                              (lambda r: r["selection"]["sampling_policy"].update(rung_worth=40.), "unknown keys .'rung_worth'")):
            bad = deepcopy(legacy)
            change(bad)
            with pytest.raises(HowHardError, match=match):
                load(engine.config, bad)

    # Observe through each receipt, replay every forecast and report.
    for name, decision_id in ids.items():
        engine.observe_decision(decision_id, "clean", id=f"from-{name}", ticket=f"from-{name}", active_seconds=10.)
    engine.refresh()
    replayed = engine.memory_beliefs()
    assert replayed["unscored_runs"] == 0
    assert next(e for e in replayed["experts"] if e["key"] == "llm")["evidence_count"] == 34
    stats = report(engine)
    assert stats["unsupported_receipts"] == [] and stats["missing_receipts"] == []
    assert stats["forecast_metrics"]["llm"]["runs"] == 34

    # On another task those four runs enter the ladders under the unnamed host, guessed by the merge rule.
    later = route(engine, "chore", "Write release notes", [opus], {OPUS: forecast(.10, seconds=1000)})
    entry = predicted(later, OPUS)["silent"]
    assert rungs(entry) == pytest.approx([34] * 5, rel=1e-4)
    assert rungs(entry, "guess")[0] == pytest.approx(30 * .20 + 4 * .10, rel=1e-4)
    load(engine.config, later["decision_id"])
