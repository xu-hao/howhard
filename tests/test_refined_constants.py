"""Refined constants: the estimated constants re-estimated from the user's own record (#137).

Every record here is synthetic and owned by the test. The shared record routes one decision a
day for 40 days on tasks that match no prototype, so each run counts only through the ladders
of later decisions. Its loud rate changes abruptly every ten days: four of every five runs are
loud in one stretch and nearly none in the next. Each decision is prepared on a clock the test
sets, and its runs are written to the outcome log an hour later.
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import fixture_filter
from howhard import Engine, capability, load_config, refine, success
from howhard.cli import main
from howhard.context import FORMAT, canonical, digest
from howhard.decisions import load
from howhard.payload_store import put
from howhard.report import report
from howhard.request_state import StaleRequest
from howhard.storage import locked
from howhard.validation import HowHardError
from legacy_receipts import to_v7, to_v8, to_v9, to_v10
from test_capability_filter import filtering, forecasts
from test_evidence_similarity import Scripted

OPUS, SONNET, HAIKU = "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"
CELLS = [{"model": OPUS, "effort": "high", "usd_per_mtok": 1.0}]
DAYS, RUNS, REGIME = 40, 5, 10
SETTINGS = '[task_types]\nY = "standard"\n'


class Clock(datetime):
    """The prepare time of a decision, set by the test."""
    at = None

    @classmethod
    def now(cls, tz=None):
        return cls.at


def route(engine, text, cells=CELLS, *, host=None, seed=3):
    """One routed decision through the calling-agent interface, filtering and forecasting every cell."""
    pending = engine.prepare("Y", text, native_candidates=cells, host=host)
    while "request_id" in pending:
        request = pending["baseline_request"]
        if pending["stage"] == "filter":
            response = filtering(None) if request is None else fixture_filter(request)
        else:
            response = forecasts(request, silent=.02, clean=.68)
        pending = engine.decide(pending["request_id"], response, seed=seed)
    return pending


def row(name, decision, at, outcome, *, ticket=None, active=60.):
    """A finished run of a routed decision, observed at ``at``."""
    return {"id": name, "date": at.date().isoformat(), "ticket": ticket or f"ticket-{name}", "task_type": "Y",
            "risk_class": decision["risk_class"], "model": decision["chosen"]["model"],
            "effort": decision["chosen"]["effort"], "outcome": outcome,
            "cause": None if outcome == "clean" else "unrecorded", "tokens": None, "seconds": None,
            "active_seconds": active, "cost_usd": None, "note": "", "amends": None, "observed_at": at.isoformat(),
            "task_text": decision["context"]["task_text"], "project_context": None, "context_encoder": FORMAT,
            "decision_id": decision["decision_id"]}


def append(engine, rows):
    with engine.config.path(engine.config.data["logs"][0]["path"]).open("a") as handle:
        handle.write("".join(json.dumps(r) + "\n" for r in rows))
    engine.refresh()


def outcome(day, run):
    """Four of five runs are loud in the even ten-day stretches; in the odd ones every third day has one."""
    if (day // REGIME) % 2 == 0:
        return "loud" if run < RUNS - 1 else "clean"
    return "loud" if run == 0 and day % 3 == 0 else "clean"


def active(day, run):
    return 60. * (1 + (day * 7 + run * 3) % 5)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    """The shared record with 198 scored runs: the last decision's final two runs are left to observe."""
    directory = tmp_path_factory.mktemp("refined-record")
    (directory / "howhard.toml").write_text(SETTINGS)
    start = datetime.now(timezone.utc) - timedelta(days=DAYS + 5)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("howhard.evidence.datetime", Clock)
        engine = Engine(load_config(directory / "howhard.toml"), retrieval=Scripted())
        for day in range(DAYS):
            Clock.at = start + timedelta(days=day)
            decision = route(engine, f"Tune cache shard {day}")
            runs = range(RUNS if day < DAYS - 1 else RUNS - 2)
            append(engine, [row(f"run-{day}-{j}", decision, Clock.at + timedelta(hours=1, minutes=j),
                                outcome(day, j), active=active(day, j)) for j in runs])
    return SimpleNamespace(directory=directory, last=decision)


def store(shared, tmp_path, settings=""):
    """A copy of the shared record owned by one test, with ``settings`` added to its configuration."""
    directory = tmp_path / "store"
    shutil.copytree(shared.directory, directory)
    path = directory / "howhard.toml"
    path.write_text(settings + path.read_text())
    return Engine(load_config(path), retrieval=Scripted())


def observe(engine, shared, run):
    """Observe one of the last decision's remaining runs through the public interface."""
    day = DAYS - 1
    engine.observe_decision(shared.last["decision_id"], outcome(day, run), id=f"run-{day}-{run}",
                            ticket=f"ticket-run-{day}-{run}", active_seconds=active(day, run))
    return engine.last_refinement


def cached(engine):
    path = engine.config.state_root / refine.FILE
    return path.read_bytes() if path.exists() else None


# ---------------------------------------------------------------------------------------------
# The worked examples of the scores and the rule


def test_the_worked_examples_of_the_scores_and_the_rule():
    assert refine.outcome_score(.20, .05, "clean") == pytest.approx(-.274, abs=5e-4)
    assert refine.outcome_score(.20, .05, "silent") == pytest.approx(-3.219, abs=5e-4)
    assert refine.outcome_score(.20, .05, "loud") == pytest.approx(-1.609, abs=5e-4)
    # n = 400 differences whose sample standard deviation is exactly 0.10: se = 0.005.
    unit = np.array([1., -1.] * 200) * math.sqrt(399 / 400)
    n, gain, se, moves = refine.rule(.012 + .10 * unit)
    assert (n, gain, se, moves) == (400, pytest.approx(.012), pytest.approx(.005), True)
    assert refine.rule(.008 + .10 * unit)[3] is False
    # The same gain on 199 runs never moves a constant.
    assert refine.rule((.012 + .10 * unit)[:199])[3] is False
    assert refine.measured_miss([1., -1., 2., -2.]) == pytest.approx(1.581, abs=5e-4)
    tau = refine.measured_scatter([.6, .8] * 15)
    assert tau == pytest.approx(.5)
    assert tau ** 2 / refine.measured_miss([1., -1., 2., -2.]) ** 2 == pytest.approx(.1)
    # The run-to-run scatter is offered once 30 pairs exist.
    assert refine.measured_scatter([.6, .8] * 14 + [.6]) is None


# ---------------------------------------------------------------------------------------------
# When a constant moves


def test_the_half_life_shortens_once_200_scored_runs_exist_and_stays_at_14_with_199(shared, tmp_path):
    engine = store(shared, tmp_path)
    status = observe(engine, shared, RUNS - 2)
    assert status == {"status": "not-due", "reason": "fewer than 200 scored runs"}
    assert cached(engine) is None
    # With 199 scored runs the record already favours a shorter half-life, but no constant moves.
    held = refine.estimate(engine)
    assert held["scored_runs"] == 199
    entry = held["constants"]["half_life_days"]
    assert (entry["value"], entry["n"]) == (14., 199) and entry["gain"] > 2 * entry["se"]
    assert all(e["value"] == refine.defaults()[s, k] for (s, k), e in
               ((pair, held["constants"][pair[1]]) for pair in refine.KEYS))
    assert engine.config.data["sampling"] == success.DEFAULTS

    status = observe(engine, shared, RUNS - 1)
    assert status["status"] == "published" and status["scored_runs"] == 200
    assert status["moved"]["half_life_days"] < 14
    stored = json.loads(cached(engine))
    entry = stored["constants"]["half_life_days"]
    assert entry["value"] == status["moved"]["half_life_days"] and entry["n"] == 200
    assert entry["gain"] > 2 * entry["se"] > 0
    assert stored["rows"] == digest(engine.raw_rows) and stored["format"] == refine.FORMAT
    # The engine applies the cached value at refresh; a new engine on the store reads it too.
    assert engine.config.data["sampling"]["half_life_days"] == entry["value"]
    again = Engine(load_config(engine.config.base / "howhard.toml"), retrieval=Scripted())
    assert again.config.data["sampling"]["half_life_days"] == entry["value"]


def test_a_constant_set_in_the_configuration_file_never_moves(shared, tmp_path):
    engine = store(shared, tmp_path, "[sampling]\nhalf_life_days = 21.0\n")
    observe(engine, shared, RUNS - 2)
    status = observe(engine, shared, RUNS - 1)
    assert status["status"] == "published" and "half_life_days" not in status["moved"]
    stored = json.loads(cached(engine))
    assert stored["constants"]["half_life_days"] == {"value": None, "n": None, "gain": None, "se": None}
    assert engine.config.data["sampling"]["half_life_days"] == 21.
    # Every run has its own ticket, so no pair of runs measures the scatter and it keeps its default.
    assert stored["constants"]["scatter_share"] == {"value": .5, "n": 200, "gain": None, "se": None}
    decision = load(engine.config, route(engine, "Tune cache shard pinned")["decision_id"])
    assert decision["selection"]["sampling_policy"]["half_life_days"] == 21.


def test_two_refinements_of_one_record_give_byte_identical_files(shared, tmp_path, monkeypatch):
    engine = store(shared, tmp_path)
    observe(engine, shared, RUNS - 2)
    assert observe(engine, shared, RUNS - 1)["status"] == "published"
    first = cached(engine)
    (engine.config.state_root / refine.FILE).unlink()
    engine.refresh()
    assert engine.config.data["sampling"] == success.DEFAULTS
    # A flush refines again when no cached result names the record; flushing moves no row order.
    assert engine.flush() == [f"run-{DAYS - 1}-{RUNS - 2}", f"run-{DAYS - 1}-{RUNS - 1}"]
    assert engine.last_refinement["status"] == "published"
    assert cached(engine) == first
    engine.flush()
    assert engine.last_refinement == {"status": "not-due", "reason": "the cached result names this record"}

    # A run observed since leaves the cached result in place until 25 more are scored, and a
    # failed refinement after a flush leaves the cached file and the flushed run as they were.
    decision = route(engine, "Tune cache shard extra")
    engine.observe_decision(decision["decision_id"], "clean", id="extra", ticket="extra", active_seconds=60.)
    assert engine.last_refinement["status"] == "not-due"

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(refine, "estimate", broken)
    assert engine.flush() == ["extra"]
    assert engine.last_refinement == {"status": "failed", "error": "RuntimeError: synthetic failure"}
    assert cached(engine) == first
    assert any(r["id"] == "extra" for r in engine.rows)


def test_a_decision_after_a_refinement_stores_the_refined_values_and_an_earlier_request_is_stale(shared, tmp_path):
    engine = store(shared, tmp_path)
    observe(engine, shared, RUNS - 2)
    earlier = engine.prepare("Y", "Tune cache shard early", native_candidates=CELLS)
    observe(engine, shared, RUNS - 1)
    refined = engine.config.data["sampling"]["half_life_days"]
    assert refined < 14
    with pytest.raises(StaleRequest) as raised:
        engine.decide(earlier["request_id"], fixture_filter(earlier["baseline_request"]))
    assert [c["kind"] for c in raised.value.changes] == ["configuration"]

    receipt = load(engine.config, route(engine, "Tune cache shard late")["decision_id"])
    assert receipt["selection"]["sampling_policy"] == engine.config.data["sampling"]
    assert receipt["selection"]["sampling_policy"]["half_life_days"] == refined
    assert receipt["semantic"]["evidence_candidates"]["half_life_days"] == refined
    assert receipt["capability"]["policy"] == engine.config.data["capability"]
    # The receipt re-validates under its own frozen policy.
    assert load(engine.config, deepcopy(receipt)) == receipt


def test_fewer_than_200_scored_runs_move_nothing_and_decisions_keep_the_defaults(shared, tmp_path):
    engine = store(shared, tmp_path)
    assert observe(engine, shared, RUNS - 2)["status"] == "not-due"
    assert cached(engine) is None
    assert engine.config.data["sampling"] == success.DEFAULTS
    assert engine.config.data["capability"] == capability.DEFAULTS
    receipt = load(engine.config, route(engine, "Tune cache shard plain")["decision_id"])
    assert receipt["selection"]["sampling_policy"] == success.DEFAULTS
    assert receipt["capability"]["policy"] == capability.DEFAULTS


# ---------------------------------------------------------------------------------------------
# When it runs


def test_no_decision_command_starts_a_refinement_and_a_failed_one_keeps_the_observation(shared, tmp_path, monkeypatch):
    engine = store(shared, tmp_path)
    # The 199th and 200th runs go straight to the log: nothing has refined the record.
    day = DAYS - 1
    when = datetime.now(timezone.utc)
    append(engine, [row(f"run-{day}-{j}", shared.last, when, outcome(day, j), active=active(day, j))
                    for j in (RUNS - 2, RUNS - 1)])
    calls = []

    def watched(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(refine, "Record", watched)
    monkeypatch.setattr(refine, "estimate", watched)
    engine.summary("Y", "Tune cache shard summary", native_candidates=CELLS)
    pending = engine.prepare("Y", "Tune cache shard context", native_candidates=CELLS)
    pending = engine.retrieve(pending["request_id"], [{"tool": "evidence.search", "query": "cache"}])
    route(engine, "Tune cache shard decide")
    engine.pick("Y", "Tune cache shard pick", native_candidates=CELLS, seed=1)
    assert calls == [] and cached(engine) is None
    assert engine.last_refinement == {"status": "not-requested"}

    decision = route(engine, "Tune cache shard observed")
    observed = engine.observe_decision(decision["decision_id"], "clean", id="observed", ticket="observed")
    assert len(calls) == 1
    assert engine.last_refinement == {"status": "failed", "error": "RuntimeError: synthetic failure"}
    assert observed["id"] == "observed" and any(r["id"] == "observed" for r in engine.rows)
    assert engine.evidence("observed", raw=True)["outcome"] == "clean"
    assert cached(engine) is None


def test_observe_and_flush_report_the_refinement_beside_the_memory_update(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["observe", "new-task", OPUS, "high", "clean", "--id", "first", "--json"]) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["memory_update"]["status"] == "pending"
    assert observed["refinement"] == {"status": "not-due", "reason": "fewer than 200 scored runs"}
    assert main(["flush", "--json"]) == 0
    flushed = json.loads(capsys.readouterr().out)
    assert flushed["ids"] == ["first"] and flushed["refinement"]["status"] == "not-due"


def test_a_malformed_cached_file_is_refused_by_name(shared, tmp_path):
    engine = store(shared, tmp_path)
    observe(engine, shared, RUNS - 2)
    observe(engine, shared, RUNS - 1)
    path = engine.config.state_root / refine.FILE
    value = json.loads(path.read_bytes())
    for change in (lambda v: v.update(format="howhard-refined-v0"),
                   lambda v: v["constants"].pop("rung_worth"),
                   lambda v: v["constants"]["half_life_days"].update(value=-1.),
                   lambda v: v["constants"]["scatter_share"].update(value=2.)):
        bad = deepcopy(value)
        change(bad)
        path.write_bytes(canonical(bad))
        with pytest.raises(HowHardError, match="refined constants"):
            engine.refresh()


# ---------------------------------------------------------------------------------------------
# What a rescoring rebuilds


def labelled(*cells):
    prices = {OPUS: .6, SONNET: .4, HAIKU: .1}
    return [{"model": model, "effort": effort, "usd_per_mtok": prices[model], "agent": agent}
            for model, effort, agent in cells]


def test_a_rescoring_at_the_receipts_own_constants_rebuilds_its_beliefs(tmp_path, monkeypatch):
    """Each scored run is predicted from what was observed before its decision, amendments included.

    The record has two hosts, two agents, similar runs counted at their discounts, a run amended
    between two decisions, and runs under a v10 receipt with imported names, which count as
    history only.
    """
    (tmp_path / "howhard.toml").write_text('[capability]\nreserved_forecast_cells = 0\n' + SETTINGS)
    monkeypatch.setattr("howhard.evidence.datetime", Clock)
    engine = Engine(load_config(tmp_path / "howhard.toml"), retrieval=Scripted())
    inventory = labelled((OPUS, "high", "alpha"), (SONNET, "high", "alpha"), (HAIKU, "low", "beta"))
    start = datetime.now(timezone.utc) - timedelta(days=20)

    def at(days):
        Clock.at = start + timedelta(days=days)
        return Clock.at

    at(0)
    first = route(engine, "Repair rollback ordering", inventory, host="host-one")
    append(engine, [row("a-1", first, at(0) + timedelta(hours=1), "clean", ticket="t-1", active=120.),
                    row("a-2", first, at(0) + timedelta(hours=2), "loud", ticket="t-1", active=300.),
                    row("a-3", first, at(0) + timedelta(hours=3), "silent", ticket="t-2", active=90.)])
    older = to_v10(load(engine.config, first["decision_id"]))
    with locked(engine.config):
        put(engine.config, "decisions", digest(older), canonical(older) + b"\n")
    legacy = {**first, "decision_id": digest(older)}
    append(engine, [row("v10-1", legacy, at(0) + timedelta(hours=4), "clean", active=60.),
                    row("v10-2", legacy, at(0) + timedelta(hours=5), "loud", active=200.)])
    # The v10 receipt names no host or agent: its runs take the names imported for them.
    engine.import_names({"hosts": {digest(older): "host-one"},
                         "agents": {"opus": "alpha", "sonnet": "alpha", "haiku": "beta"}})
    at(1)
    second = route(engine, "Repair rollback retries", inventory, host="host-two")
    append(engine, [row("b-1", second, at(1) + timedelta(hours=1), "silent", ticket="t-3", active=80.),
                    row("b-2", second, at(1) + timedelta(hours=2), "clean", ticket="t-3", active=70.)])
    # The loud run is amended to clean after the second decision and before the third.
    amendment = {**row("a-2-fix", first, at(2), "clean", active=None), "amends": "a-2", "ticket": "t-1"}
    for field in ("task_text", "project_context", "context_encoder", "decision_id", "active_seconds"):
        amendment.pop(field)
    append(engine, [amendment])
    at(3)
    third = route(engine, "Tune cache layout", inventory, host="host-one")
    append(engine, [row("c-1", third, at(3) + timedelta(hours=1), "loud", ticket="t-4", active=400.)])

    record = refine.Record(engine)
    assert record.total == 6
    # Runs under the v10 receipt are history, never scored.
    assert {r["group"][0] for r in record.runs} == {first["decision_id"], second["decision_id"], third["decision_id"]}
    assert {"v10-1", "v10-2"} <= set(record.root_index)
    scorer = refine.Scorer(record)
    policy = engine.config.data["sampling"]
    related = False
    for i, run in enumerate(record.runs):
        receipt = load(engine.config, run["group"][0])
        pair = run["pair"]
        cell = next(c for c in receipt["selection"]["cells"] if (c["family"], c["effort"]) == pair)
        stored = next(c for c in receipt["semantic"]["prediction"]["cells"] if (c["family"], c["effort"]) == pair)
        starts = scorer.starts(i, policy, ("loud", "silent", "seconds"))
        for kind in ("loud", "silent", "seconds"):
            for field in ("prediction", "s", "worth", "N_start", "level", "usual"):
                assert starts[kind][field] == pytest.approx(stored["values"][kind][field], rel=1e-9, abs=1e-12)
        clean, loud, silent = refine.counts(run, policy["half_life_days"], policy)
        expected = cell["outcome_counts"]
        assert (clean, loud, silent) == pytest.approx((expected["n_clean"], expected["n_loud"], expected["n_silent"]),
                                                      rel=1e-9, abs=1e-12)
        assert refine.rate_mean(starts["loud"], loud, clean + silent, {}) == pytest.approx(cell["P_loud"]["mean"], rel=1e-9)
        assert refine.rate_mean(starts["silent"], silent, clean, {}) == pytest.approx(
            cell["P_silent_accepted"]["mean"], rel=1e-9)
        time = success.amount_belief(starts["seconds"], refine.duration_runs(run, policy["half_life_days"]),
                                     policy, "seconds")
        assert (time["centre"], time["spread"]) == pytest.approx(
            (cell["active_time"]["centre"], cell["active_time"]["spread"]), rel=1e-9)
        caught = refine.caught_record(record, run, policy["half_life_days"])
        share = receipt["capability"]["caught_share"]
        assert [caught[k] for k in ("F_all", "C_all", "F_host", "C_host")] == pytest.approx(
            [share[k] for k in ("F_all", "C_all", "F_host", "C_host")], rel=1e-9, abs=1e-12)
        # The receipt lists every candidate, so a run of a related cell, counted at any discount,
        # is weighed again without the reranker when a discount is tried.
        zero = {**policy, "effort_discount": 0., "model_discount": 0., "agent_discount": 0.}
        related |= refine.counts(run, policy["half_life_days"], zero) != (clean, loud, silent)
    assert related
    # The second decision saw the first decision's loud run as loud; the third saw it amended.
    second_run = next(r for r in record.runs if r["group"][0] == second["decision_id"])
    third_run = next(r for r in record.runs if r["group"][0] == third["decision_id"])
    assert refine.caught_record(record, second_run, 14.)["C_all"] > 0
    assert load(engine.config, third["decision_id"])["capability"]["caught_share"]["C_all"] == pytest.approx(
        refine.caught_record(record, third_run, 14.)["C_all"])


def test_v7_to_v10_receipts_load_observe_replay_and_report_beside_a_refinement(shared, tmp_path):
    engine = store(shared, tmp_path)
    observe(engine, shared, RUNS - 2)
    assert observe(engine, shared, RUNS - 1)["status"] == "published"
    current = load(engine.config, shared.last["decision_id"])
    assert refine.receipt_format(engine.config, shared.last["decision_id"]) == "howhard-decision-v11"
    ids = {}
    for name, build in (("v10", to_v10), ("v9", to_v9), ("v8", to_v8), ("v7", to_v7)):
        legacy = build(current)
        ids[name] = digest(legacy)
        with locked(engine.config):
            put(engine.config, "decisions", ids[name], canonical(legacy) + b"\n")
        loaded = load(engine.config, ids[name])
        assert loaded == legacy and loaded["format"] == f"howhard-decision-{name}"
        # The refinement tells the receipts it may score by the format label each one records.
        assert refine.receipt_format(engine.config, ids[name]) == f"howhard-decision-{name}"
        # An older receipt keeps its own frozen policy, which names no refined constant it lacks.
        assert "rung_worth" not in loaded["selection"]["sampling_policy"]
    for name, decision_id in ids.items():
        engine.observe_decision(decision_id, "clean", id=f"from-{name}", ticket=f"from-{name}", active_seconds=10.)
        # Runs under older receipts are not scored, so no refinement becomes due.
        assert engine.last_refinement["status"] == "not-due"
    replayed = engine.memory_beliefs()
    assert replayed["unscored_runs"] == 0
    stats = report(engine)
    assert stats["unsupported_receipts"] == [] and stats["missing_receipts"] == []
    record = refine.Record(engine)
    assert record.total == 200
    assert {f"from-{name}" for name in ids} <= set(record.root_index)
