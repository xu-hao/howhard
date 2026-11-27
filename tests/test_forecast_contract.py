"""Forecast contract v11 at public boundaries (#128).

The host's name on a request, the agent label on a candidate, and stated ranges on active
time and token kinds are validated, stored and reported. The two names never change which
cell is chosen; the ranges are read into the beliefs for active time and tokens (#132).
Every fixture is synthetic and owns its state.
"""

from copy import deepcopy
import json
import math

import pytest

from howhard import Engine, load_config
from howhard.cli import main
from howhard.context import canonical, digest
from howhard.decisions import load
from howhard.payload_store import put
from howhard.report import report
from howhard.storage import locked
from howhard.validation import HowHardError
from test_capability_filter import SimilarityDouble, filtering
from test_execution_accounting import card, counts

TASK = "Repair rollback ordering"
# (model, effort): (agent label, clean, loud, silent, active seconds, token scale)
CELLS = {
    ("fixture/small", "low"): ("agent-a", .80, .15, .05, 120., 1.0),
    ("fixture/small", "high"): ("agent-a", .86, .11, .03, 240., 1.6),
    ("fixture/medium", "low"): ("agent-b", .90, .08, .02, 300., 2.0),
    ("fixture/medium", "high"): ("agent-b", .93, .055, .015, 480., 2.8),
    ("fixture/large", "low"): (None, .95, .04, .01, 600., 3.5),
    ("fixture/large", "high"): (None, .97, .025, .005, 900., 5.0),
}
PRICE = {"fixture/small": 1., "fixture/medium": 3., "fixture/large": 9.}


def candidates(*, agents=False):
    result = []
    for (model, effort), spec in CELLS.items():
        rate = PRICE[model]
        cell = {"model": model, "effort": effort, "rate_card": card(model=model, rates=counts(
            input=rate, output=4 * rate, cache_read=.1 * rate, cache_write=1.25 * rate,
            cache_write_5m=1.25 * rate, cache_write_1h=2 * rate))}
        if agents and spec[0] is not None:
            cell["agent"] = spec[0]
        result.append(cell)
    return result


def forecast(model, effort, *, ranges=False):
    _, clean, loud, silent, seconds, scale = CELLS[model, effort]
    usage = counts(input=4000 * scale, output=1500 * scale, cache_read=20000 * scale, cache_write=500 * scale)
    # The two rate ranges predate this contract and already inform selection.
    row = {"clean": clean, "loud": loud, "silent": silent, "tokens": sum(usage.values()), "seconds": seconds,
           "token_usage": usage, "loud_ci": [loud / 2, min(1., loud * 2)], "silent_ci": [0., min(1., silent * 4)]}
    if ranges:
        row["seconds_ci"] = [seconds / 2, seconds * 3]
        row["token_usage_ci"] = {"input": [usage["input"] / 2, usage["input"] * 2],
                                 "output": [usage["output"] / 4, usage["output"] * 3],
                                 "cache_read": [0, usage["cache_read"] * 4],
                                 "cache_write_5m": [0, 500],   # forecast at zero, yet possibly non-zero
                                 "cache_write_1h": [0, 0]}     # forecast at zero, exactly zero
    return row


class ContractLLM:
    """Deterministic protocol double: one fixed forecast per cell, optionally with ranges."""
    identity = "forecast-contract-double"

    def __init__(self, *, ranges=False):
        self.calls, self.ranges = [], ranges
        self.last_usage = {"seconds": .01, "cost_usd": .001, "tokens": None, "models": ["double"]}

    def complete(self, request):
        self.calls.append(deepcopy(request))
        if request["kind"] == "learn":
            return {"operations": [], "reason": "Synthetic no-op writer"}
        if request["kind"] == "filter":
            if request["phase"] != "baseline":
                return {"memory_filters": [], "reason": "No fixture memory override"}
            keys_by_pair = {(c["model"], c["effort"]): c["key"] for c in request["cells"]}
            order = [keys_by_pair[pair] for pair in CELLS if pair in keys_by_pair]
            result = filtering(request, 0, len(order) - 1, order)
            result.pop("memory_filters")
            return result
        if request["phase"] != "baseline":
            return {"memory_forecasts": [], "reason": "No fixture forecast override"}
        return {"forecasts": [{"family": c["family"], "effort": c["effort"],
                               **forecast(c["model"], c["effort"], ranges=self.ranges)} for c in request["cells"]],
                "reason": "Synthetic contract forecast"}


def scenario(directory, *, host=None, agents=False, ranges=False, seed=20261126):
    """One routed decision over six priced cells with earlier outcomes for two of them.

    The three runs of the first cell are retries of one ticket, so together they count for one run.
    The store forecasts its band alone: with a reserved forecast slot the third model, which has no
    earlier run, would join the forecast cells and the cell on trial would never be drawn.
    """
    from legacy_receipts import without_reserved_slots
    directory.mkdir()
    without_reserved_slots(directory)
    engine = Engine(directory=directory, llm=ContractLLM(ranges=ranges), retrieval=SimilarityDouble())
    history = [("fixture/small", "low", "loud"), ("fixture/small", "low", "clean"), ("fixture/small", "low", "silent"),
               ("fixture/medium", "low", "clean"), ("fixture/medium", "low", "clean"), ("fixture/medium", "low", "loud")]
    for index, (model, effort, outcome) in enumerate(history):
        extra = {} if outcome == "clean" else {"cause": "capability"}
        if model == "fixture/small":
            extra["ticket"] = "T-1"
        engine.observe("task", model, effort, outcome, native=True, task_text=TASK, id=f"earlier-{index}",
                       active_seconds=200. + 40 * index, **extra)
    options = {} if host is None else {"host": host}
    result = engine.pick("task", TASK, native_candidates=candidates(agents=agents), seed=seed, **options)
    return engine, result


def selection(result):
    """The chosen cell, its sampled values, and the admission fields and beliefs of every forecast cell."""
    fields = ("model", "effort", "J", "sampled_J", "T", "admissible", "admission_risk", "pi", "tier", "credit_ok",
              "needed_credit_usd", "credit_variance", "credit_p80_usd", "session_cost_usd", "expected_attempts",
              "active_time", "token_beliefs")
    return {"chosen": result["chosen"], "seed": result["seed"], "cap_met": result["cap_met"],
            "needs_review": result["needs_review"],
            "exploration": {k: result["exploration"][k] for k in ("enabled", "reason", "pool", "mean_best", "eligible_cells")},
            "sampled_expert": result["semantic"].get("sampled_expert"),
            "cells": [{**{k: row[k] for k in fields},
                       **{k: row[k]["mean"] for k in ("P_clean", "P_loud", "P_silent", "P_silent_accepted")}}
                      for row in sorted(result["cells"], key=lambda row: (row["model"], row["effort"]))],
            "sampling": sorted(([c["family"], c["effort"], c["concentration"], c["cost_sd"],
                                 *c["outcome_posterior"]["alpha"]]
                                for c in result["semantic"]["sampling"]["cells"]), key=lambda c: c[:2])}


def close(actual, expected):
    """Recursive equality with a relative tolerance for floating-point values.

    A run's age is measured to the moment its request is prepared, so two runs of the scenario
    differ by the seconds they took: far below this tolerance, far above rounding.
    """
    if isinstance(expected, float) and not isinstance(actual, bool):
        return actual == pytest.approx(expected, rel=1e-3, abs=1e-12)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(close(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(close(a, e) for a, e in zip(actual, expected))
    return actual == expected


def routed(engine, decision_id):
    return next(r for r in report(engine)["routing"] if r["decision_id"] == decision_id)


def write_candidates(directory, cells):
    path = directory / "candidates.json"
    path.write_text(json.dumps(cells))
    return str(path)


# --- 1. The host's name on the request ---------------------------------------------------

def test_host_name_is_stored_in_the_receipt_and_shown_by_report(tmp_path):
    engine, named = scenario(tmp_path / "named", host="host-one")
    assert named["host"] == "host-one"
    receipt = load(engine.config, named["decision_id"])
    assert receipt["host"] == "host-one"
    assert routed(engine, named["decision_id"])["host"] == "host-one"

    other, unnamed = scenario(tmp_path / "unnamed")
    receipt = load(other.config, unnamed["decision_id"])
    assert "host" in receipt and receipt["host"] is None
    assert routed(other, unnamed["decision_id"])["host"] is None


@pytest.mark.parametrize("command", ["pick", "summary", "context"])
def test_cli_accepts_the_host_name_on_every_routing_command(tmp_path, monkeypatch, capsys, command):
    monkeypatch.chdir(tmp_path)
    cells = write_candidates(tmp_path, [{"model": "fixture/worker", "effort": "high", "agent": "agent-a"}])
    assert main([command, "task", "--text", TASK, "--candidates", cells, "--host", "host-one", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["host"] == "host-one"
    if command == "pick":
        assert main(["report", "--json"]) == 0
        shown = json.loads(capsys.readouterr().out)["routing"]
        assert shown == [{"decision_id": result["decision_id"], "host": "host-one",
                          "chosen": {**result["chosen"], "agent": "agent-a"}}]
        # The plain rendering names both labels too.
        assert main(["pick", "task", "--text", TASK, "--candidates", cells, "--host", "host-one", "--seed", "1"]) == 0
        text = capsys.readouterr().out
        assert "Host: host-one" in text and "agent=agent-a" in text


def test_calling_agent_request_carries_the_host_to_the_receipt(tmp_path):
    engine = Engine(directory=tmp_path)
    cells = [{"model": "fixture/worker", "effort": "high"}]
    pending = engine.prepare("task", TASK, native_candidates=cells, host="host-one")
    assert pending["host"] == "host-one"
    pending = engine.decide(pending["request_id"], filtering(pending["baseline_request"]))
    assert pending["host"] == "host-one"
    rows = [{"family": c["family"], "effort": c["effort"], "clean": .9, "loud": .08, "silent": .02,
             "tokens": 1000, "seconds": 60, "seconds_ci": [30, 240]} for c in pending["baseline_request"]["cells"]]
    result = engine.decide(pending["request_id"], {"forecasts": rows, "memory_forecasts": [], "reason": "Synthetic"}, seed=5)
    receipt = load(engine.config, result["decision_id"])
    assert receipt["host"] == "host-one"
    assert receipt["semantic"]["worlds"]["llm"] == rows
    assert engine.prepare("task", "Another task entirely", native_candidates=cells)["host"] is None


@pytest.mark.parametrize("host", ["", "two words", "host/one", "h" * 65])
def test_malformed_host_name_is_refused_before_any_state_exists(tmp_path, monkeypatch, capsys, host):
    monkeypatch.chdir(tmp_path)
    cells = write_candidates(tmp_path, [{"model": "fixture/worker", "effort": "high"}])
    assert main(["pick", "task", "--candidates", cells, "--host", host]) == 2
    assert "host:" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["candidates.json"]
    engine = Engine(directory=tmp_path / "api")
    for method in ("pick", "summary", "prepare"):
        with pytest.raises(HowHardError, match="^host:"):
            getattr(engine, method)("task", native_candidates=json.loads((tmp_path / "candidates.json").read_text()), host=host)
    with pytest.raises(HowHardError, match="^host:"):
        engine.pick("task", native_candidates=[{"model": "fixture/worker", "effort": "high"}], host=7)


# --- 2. The agent label on a candidate ---------------------------------------------------

def test_agent_label_is_stored_with_its_cell_and_shown_for_the_chosen_cell(tmp_path):
    engine, result = scenario(tmp_path / "named", agents=True)
    receipt = load(engine.config, result["decision_id"])
    expected = {pair: spec[0] for pair, spec in CELLS.items()}
    assert {(c["model"], c["effort"]): c.get("agent") for c in receipt["native_candidates"]} == expected
    # A candidate without a label is stored as the one unnamed agent.
    assert all("agent" not in c for c in receipt["native_candidates"] if c["model"] == "fixture/large")
    assert {(c["model"], c["effort"]): c["agent"] for c in receipt["capability"]["cells"]} == expected
    chosen = result["chosen"]
    assert routed(engine, result["decision_id"])["chosen"] == {**chosen, "agent": expected[chosen["model"], chosen["effort"]]}

    other, plain = scenario(tmp_path / "plain")
    receipt = load(other.config, plain["decision_id"])
    assert all("agent" not in c for c in receipt["native_candidates"])
    assert all(c["agent"] is None for c in receipt["capability"]["cells"])
    assert routed(other, plain["decision_id"])["chosen"]["agent"] is None

    # A stored label that disagrees with the inventory is not a valid receipt.
    tampered = load(engine.config, result["decision_id"])
    tampered["capability"]["cells"][0]["agent"] = "agent-z"
    with pytest.raises(HowHardError, match="agent"):
        load(engine.config, tampered)


@pytest.mark.parametrize("label", ["", "two words", "agent/a", 3, None, ["agent-a"], "a" * 65])
def test_malformed_agent_label_is_refused_and_names_the_field(tmp_path, label):
    cells = [{"model": "fixture/worker", "effort": "high", "agent": label}]
    engine = Engine(directory=tmp_path)
    for method in ("pick", "summary", "prepare"):
        with pytest.raises(HowHardError, match=r"^candidate\.agent:"):
            getattr(engine, method)("task", native_candidates=cells)
    assert engine.raw_rows == []
    from howhard.payload_store import decision_ids
    assert decision_ids(engine.config) == []


def test_configured_inventory_accepts_and_validates_the_agent_label(tmp_path, monkeypatch, capsys):
    def configure(label):
        (tmp_path / "howhard.toml").write_text(
            'on_cap_unmet = "flag-review"\n[inventories]\nrouter = [\n'
            f'  {{ model = "fixture/worker", effort = "high", agent = {json.dumps(label)} }},\n'
            '  { model = "fixture/other", effort = "high" },\n]\n')
    configure("agent-a")
    monkeypatch.chdir(tmp_path)
    assert main(["pick", "bookkeeping", "--inventory", "router", "--host", "host-one", "--seed", "3", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    receipt = load(load_config(tmp_path / "howhard.toml"), result["decision_id"])
    assert {c["model"]: c.get("agent") for c in receipt["native_candidates"]} == {"fixture/worker": "agent-a", "fixture/other": None}
    assert receipt["host"] == "host-one"
    configure("not a label")
    with pytest.raises(HowHardError, match=r"candidate\.agent"):
        load_config(tmp_path / "howhard.toml")


# --- 3. Ranges on active time and on each token kind -------------------------------------

def test_forecast_ranges_are_accepted_and_stored_unchanged(tmp_path):
    engine, result = scenario(tmp_path / "ranges", ranges=True)
    baseline = next(c for c in engine.llm.calls if c["kind"] == "predict" and c["phase"] == "baseline")
    assert "seconds_ci" in baseline["contract"] and "token_usage_ci" in baseline["contract"]
    assert baseline["prompt_version"] == "semantic-reader-v8"
    receipt = load(engine.config, result["decision_id"])
    assert (receipt["format"], receipt["semantic"]["format"], receipt["semantic"]["prompt_version"],
            receipt["capability"]["format"]) == ("howhard-decision-v11", "howhard-semantic-v8",
                                                 "semantic-reader-v8", "howhard-capability-v6")
    models = {(c["family"], c["effort"]): c["model"] for c in receipt["selection"]["cells"]}
    stored = receipt["semantic"]["worlds"]["llm"]
    assert len(stored) >= 2
    for row in stored:
        stated = forecast(models[row["family"], row["effort"]], row["effort"], ranges=True)
        assert row == {"family": row["family"], "effort": row["effort"], **stated}
        assert row["token_usage_ci"]["cache_write_1h"] == [0, 0] and "cache_write" not in row["token_usage_ci"]


def decide_with(tmp_path, change, *, rate_card=True):
    """Submit one calling-agent forecast row changed by ``change`` for a single cell."""
    cell = {"model": "fixture/worker", "effort": "high"}
    if rate_card:
        cell["rate_card"] = card(model="fixture/worker")
    engine = Engine(directory=tmp_path)
    pending = engine.prepare("task", TASK, native_candidates=[cell])
    pending = engine.decide(pending["request_id"], filtering(pending["baseline_request"]))
    usage = counts(input=4000, output=1000, cache_read=5000)
    row = {"family": pending["baseline_request"]["cells"][0]["family"], "effort": "high", "clean": .9, "loud": .08,
           "silent": .02, "tokens": 10000, "seconds": 60}
    if rate_card:
        row["token_usage"] = usage
    change(row)
    response = {"forecasts": [row], "memory_forecasts": [], "reason": "Synthetic"}
    return engine, row, lambda: engine.decide(pending["request_id"], response, seed=5)


@pytest.mark.parametrize("change, field", [
    (lambda r: r.update(seconds_ci=[61, 120]), "seconds_ci"),                      # lo above the value
    (lambda r: r.update(seconds_ci=[10, 59.9]), "seconds_ci"),                     # hi below the value
    (lambda r: r.update(seconds_ci=[-1, 120]), "seconds_ci"),
    (lambda r: r.update(seconds_ci=[30]), "seconds_ci"),
    (lambda r: r.update(seconds_ci=[30, 120, 240]), "seconds_ci"),
    (lambda r: r.update(seconds_ci={"lo": 30, "hi": 120}), "seconds_ci"),
    (lambda r: r.update(seconds_ci=[30, "120"]), "seconds_ci"),
    (lambda r: r.update(seconds_ci=[True, 120]), "seconds_ci"),
    (lambda r: r.update(seconds_ci=[30, float("inf")]), "seconds_ci"),
    (lambda r: r.update(token_usage_ci={"input": [4001, 9000]}), r"token_usage_ci\.input"),
    (lambda r: r.update(token_usage_ci={"output": [0, 999]}), r"token_usage_ci\.output"),
    (lambda r: r.update(token_usage_ci={"cache_write": [1, 5]}), r"token_usage_ci\.cache_write"),  # value is zero
    (lambda r: r.update(token_usage_ci={"input": [5000, 3000]}), r"token_usage_ci\.input"),
    (lambda r: r.update(token_usage_ci={"input": [-1, 9000]}), r"token_usage_ci\.input"),
    (lambda r: r.update(token_usage_ci={"input": [4000]}), r"token_usage_ci\.input"),
    (lambda r: r.update(token_usage_ci={"input": None}), r"token_usage_ci\.input"),
    (lambda r: r.update(token_usage_ci={"reasoning": [0, 10]}), "token_usage_ci"),
    (lambda r: r.update(token_usage_ci={"tokens": [0, 20000]}), "token_usage_ci"),   # no range on total tokens
    (lambda r: r.update(token_usage_ci=[[0, 1]]), "token_usage_ci"),
    (lambda r: r.update(tokens_ci=[0, 20000]), "tokens_ci"),
    (lambda r: r.update(credit_ci=[0, 1]), "credit_ci"),
    (lambda r: r.update(loud_ci=[.05, 1.01]), "loud_ci"),                           # a rate range also needs hi <= 1
    (lambda r: r.update(silent_ci=[.03, .5]), "silent_ci"),
])
def test_ranges_that_miss_their_value_or_are_malformed_are_refused_by_field(tmp_path, change, field):
    engine, _, decide = decide_with(tmp_path, change)
    with pytest.raises(HowHardError, match=field):
        decide()
    from howhard.payload_store import decision_ids
    assert decision_ids(engine.config) == []


def test_token_kind_range_needs_a_token_kind_forecast(tmp_path):
    engine, _, decide = decide_with(tmp_path, lambda r: r.update(token_usage_ci={"input": [0, 20000]}), rate_card=False)
    with pytest.raises(HowHardError, match="token_usage_ci: requires token_usage"):
        decide()


@pytest.mark.parametrize("change", [
    lambda r: r.update(seconds_ci=[60, 60]),                                 # a point range at the value
    lambda r: r.update(seconds_ci=[0, 86400 * 30]),                          # no ceiling on a stated width
    lambda r: r.update(token_usage_ci={}),
    lambda r: r.update(token_usage_ci={"cache_write": [0, 0], "cache_write_5m": [0, 2500]}),
    lambda r: r.update(token_usage_ci={k: [0, 10 ** 9] for k in counts()}),
    lambda r: r.update(loud_ci=[0, 1], silent_ci=[.0, .99]),                 # the flat belief; no width ceiling
])
def test_edge_ranges_are_accepted_and_stored_as_stated(tmp_path, change):
    engine, row, decide = decide_with(tmp_path, change)
    assert load(engine.config, decide()["decision_id"])["semantic"]["worlds"]["llm"] == [row]


def test_memory_conditioned_forecasts_accept_and_keep_ranges(tmp_path):
    class Reader(ContractLLM):
        def complete(self, request):
            response = super().complete(request)
            if request["kind"] == "predict" and request["phase"] == "memory":
                row = {"family": request["cells"][0]["family"], "effort": request["cells"][0]["effort"],
                       **forecast(request["cells"][0]["model"], request["cells"][0]["effort"], ranges=True),
                       "seconds_ci": [1, 5000]}
                response["memory_forecasts"] = [{"id": m["id"], "version": m["version"], "forecasts": [row]}
                                                for m in request["memories"] if not m["id"].startswith("execution:")]
            return response

    engine = Engine(directory=tmp_path, llm=Reader(), retrieval=SimilarityDouble())
    memory = engine.memory.add("Rollback", "Rollback ordering tasks can run long.")
    result = engine.pick("task", TASK, native_candidates=candidates(), seed=9)
    receipt = load(engine.config, result["decision_id"])
    world = receipt["semantic"]["worlds"][f"memory:{memory['id']}:{memory['version']}"]
    assert [row["seconds_ci"] for row in world if "seconds_ci" in row] == [[1, 5000]]
    assert all("seconds_ci" not in row for row in receipt["semantic"]["worlds"]["llm"])


# --- 4. Stored prototype forecasts keep what was stated ----------------------------------

def test_stored_prototype_forecast_returns_the_same_ranges_and_host(tmp_path):
    from howhard.prototypes import stored_forecasts
    from test_dynamic_prototypes import TopicSimilarityDouble
    cells = candidates(agents=True)[:4]
    spy = ContractLLM(ranges=True)
    engine = Engine(directory=tmp_path, llm=spy, retrieval=TopicSimilarityDouble())
    first = engine.pick("task", "Repair rollback boundary", native_candidates=cells, seed=7, host="host-one")
    stated = {(c["model"], c["effort"]): forecast(c["model"], c["effort"], ranges=True) for c in cells}

    kept = stored_forecasts(engine.config, task_type="task")
    assert {(f["model"], f["effort"]) for f in kept} == set(stated)
    for item in kept:
        assert item["host"] == "host-one"
        assert item["forecast"] == stated[item["model"], item["effort"]]
    assert stored_forecasts(engine.config, task_type="other") == []

    # A related task reuses the stored forecasts with no reader call, ranges included,
    # under another host: reuse does not depend on the host yet.
    spy.calls.clear()
    second = engine.pick("task", "Repair rollback boundary regression", native_candidates=cells, seed=7, host="host-two")
    assert spy.calls == [] and second["semantic"]["backend"] == "prototype-materialized"
    receipt = load(engine.config, second["decision_id"])
    assert receipt["host"] == "host-two"
    models = {(c["family"], c["effort"]): c["model"] for c in receipt["selection"]["cells"]}
    first_rows = {(r["family"], r["effort"]): r for r in load(engine.config, first["decision_id"])["semantic"]["worlds"]["llm"]}
    for row in receipt["semantic"]["worlds"]["llm"]:
        expected = stated[models[row["family"], row["effort"]], row["effort"]]
        for field in ("loud_ci", "silent_ci", "seconds_ci", "token_usage_ci"):
            assert row[field] == expected[field] == first_rows[row["family"], row["effort"]][field]
        assert row["seconds"] == expected["seconds"]
    assert second["chosen"] == first["chosen"]
    # The stored forecasts still name the host that made them.
    assert {f["host"] for f in stored_forecasts(engine.config)} == {"host-one"}


def test_forecast_stored_without_a_host_or_ranges_reads_as_unnamed(tmp_path):
    from howhard.prototypes import stored_forecasts
    from test_dynamic_prototypes import TopicSimilarityDouble
    engine = Engine(directory=tmp_path, llm=ContractLLM(), retrieval=TopicSimilarityDouble())
    assert stored_forecasts(engine.config) == []
    engine.pick("task", "Repair rollback boundary", native_candidates=candidates()[:2], seed=7)
    kept = stored_forecasts(engine.config)
    assert len(kept) == 2 and all(f["host"] is None for f in kept)
    assert all("seconds_ci" not in f["forecast"] and "token_usage_ci" not in f["forecast"] for f in kept)


# --- 5. Names never change selection; ranges are read into beliefs -------------------------

# The first seed chooses the trusted cell with the lowest mean cost, the second another trusted
# cell by its sampled cost, and the third the cell on trial, which flags the decision for review.
@pytest.mark.parametrize("seed, reason, trusted", [(20261127, "mean-best", True),
                                                   (20261128, "sampled-within-posterior-overlap", True),
                                                   (20261126, "sampled-within-posterior-overlap", False)])
@pytest.mark.parametrize("options", [
    {"host": "host-one"},
    {"agents": True},
    {"host": "host-one", "agents": True},
], ids=["host", "agents", "both"])
def test_host_and_agent_names_do_not_change_selection(tmp_path, options, seed, reason, trusted):
    _, plain = scenario(tmp_path / "plain", seed=seed)
    _, named = scenario(tmp_path / "named", seed=seed, **options)
    expected = selection(plain)
    # The fixture exercises sampled selection over trusted cells and one on trial beside them.
    assert expected["exploration"]["reason"] == reason
    assert expected["cap_met"] is trusted and expected["needs_review"] is not trusted
    assert {c["admissible"] for c in expected["cells"]} == {True, False}
    assert {c["tier"] for c in expected["cells"]} == {"trusted", "on-trial"}
    assert close(json.loads(json.dumps(selection(named))), json.loads(json.dumps(expected))), selection(named)


def test_stated_ranges_widen_the_beliefs_and_leave_their_averages(tmp_path):
    _, plain = scenario(tmp_path / "plain")
    _, ranged = scenario(tmp_path / "ranged", ranges=True)
    stated = {(c["model"], c["effort"]): c for c in ranged["cells"]}
    for row in plain["cells"]:
        other = stated[row["model"], row["effort"]]
        usage = forecast(row["model"], row["effort"])["token_usage"]
        # A range never moves a belief's average: time and each stated kind keep their forecast.
        # (The two scenarios were prepared seconds apart, which ages their earlier runs differently.)
        assert other["T"] == pytest.approx(row["T"], rel=1e-3)
        beliefs = other["token_beliefs"]
        assert beliefs["cache_read"]["expected_tokens"] == pytest.approx(usage["cache_read"])
        # [0, 4x] around x is wider than the minimum spread and is used; [x/2, 2x] is narrower and is not.
        assert beliefs["cache_read"]["spread"] == pytest.approx(math.log(4 * usage["cache_read"]) / 3.92)
        assert beliefs["cache_read"]["spread"] > row["token_beliefs"]["cache_read"]["spread"]
        assert beliefs["input"]["spread"] == pytest.approx(row["token_beliefs"]["input"]["spread"])
        assert beliefs["input"]["stated_spread"] == pytest.approx(math.log(4) / 3.92)
        # A kind forecast at zero is exactly zero unless a non-zero upper end is stated.
        assert row["token_beliefs"]["cache_write_5m"]["expected_tokens"] == 0.
        assert beliefs["cache_write_5m"]["expected_tokens"] == pytest.approx(1.)
        assert beliefs["cache_write_5m"]["spread"] == pytest.approx(math.log(500) / 3.92)
        assert beliefs["cache_write_1h"]["expected_tokens"] == 0. and beliefs["cache_write_1h"]["spread"] == 0.
        assert other["credit_variance"] > row["credit_variance"]
    assert ranged["chosen"] == plain["chosen"]


def test_names_do_not_change_selection_without_a_met_cap_either(tmp_path):
    """Both variants of an inadmissible request agree, including the review flag."""
    def run(directory, **options):
        directory.mkdir()
        cells = [{"model": "fixture/worker", "effort": "high", "usd_per_mtok": 1.0,
                  **({"agent": "agent-a"} if options else {})}]
        engine = Engine(directory=directory)
        pending = engine.prepare("task", TASK, native_candidates=cells, **({"host": "host-one"} if options else {}))
        pending = engine.decide(pending["request_id"], filtering(pending["baseline_request"]))
        row = {"family": pending["baseline_request"]["cells"][0]["family"], "effort": "high", "clean": .5,
               "loud": .2, "silent": .3, "tokens": 1000, "seconds": 60}
        return engine.decide(pending["request_id"], {"forecasts": [row], "memory_forecasts": [], "reason": "Synthetic"}, seed=5)
    plain, named = run(tmp_path / "plain"), run(tmp_path / "named", new=True)
    assert not plain["cap_met"] and plain["needs_review"]
    for field in ("chosen", "cap_met", "needs_review", "exploration", "why"):
        assert plain[field] == named[field]
    assert plain["cells"] == named["cells"]
    assert plain["semantic"]["sampling"] == named["semantic"]["sampling"]


# --- 6. Older receipts ---------------------------------------------------------------------

def test_v7_to_v10_receipts_load_observe_replay_and_report_as_unnamed(tmp_path, monkeypatch, capsys):
    from legacy_receipts import to_v7, to_v8, to_v9, to_v10
    engine = Engine(directory=tmp_path, llm=ContractLLM(ranges=True), retrieval=SimilarityDouble())
    cells = [{**c, "usd_per_mtok": PRICE[c["model"]]} for c in candidates(agents=True)[:2]]
    for cell in cells:
        cell.pop("rate_card")
    current = engine.pick("task", TASK, native_candidates=cells, seed=11, host="host-one")
    receipt = load(engine.config, current["decision_id"])
    assert receipt["format"] == "howhard-decision-v11" and receipt["host"] == "host-one"
    assert any("seconds_ci" in row for row in receipt["semantic"]["worlds"]["llm"])

    formats = {"v10": ("howhard-semantic-v7", "semantic-reader-v7", "howhard-capability-v5"),
               "v9": ("howhard-semantic-v6", "semantic-reader-v7", "howhard-capability-v4"),
               "v8": ("howhard-semantic-v5", "semantic-reader-v6", "howhard-capability-v3"),
               "v7": ("howhard-semantic-v4", "semantic-reader-v5", "howhard-capability-v2")}
    ids = {}
    for name, build in (("v10", to_v10), ("v9", to_v9), ("v8", to_v8), ("v7", to_v7)):
        legacy = build(receipt)
        ids[name] = digest(legacy)
        with locked(engine.config):
            put(engine.config, "decisions", ids[name], canonical(legacy) + b"\n")
        loaded = load(engine.config, ids[name])
        assert loaded == legacy and loaded["format"] == f"howhard-decision-{name}"
        assert (loaded["semantic"]["format"], loaded["semantic"]["prompt_version"], loaded["capability"]["format"]) == formats[name]
        assert "host" not in loaded and all("agent" not in c for c in loaded["native_candidates"])
        assert all("seconds_ci" not in row and "token_usage_ci" not in row
                   for panel in loaded["semantic"]["worlds"].values() for row in panel)

    # Observe through each receipt, by ID and through the CLI.
    monkeypatch.chdir(tmp_path)
    for name in ("v10", "v9", "v8"):
        engine.observe_decision(ids[name], "clean", id=f"from-{name}", active_seconds=10.)
    assert main(["observe", ids["v7"], "clean", "--id", "from-v7", "--json"]) == 0
    capsys.readouterr()
    engine.observe_decision(current["decision_id"], "clean", id="from-v11", active_seconds=10.)
    engine.refresh()

    # Replay: every outcome scores the forecasts frozen in its own receipt.
    beliefs = engine.memory_beliefs()
    assert beliefs["unscored_runs"] == 0
    assert next(e for e in beliefs["experts"] if e["key"] == "llm")["evidence_count"] == 5
    stats = report(engine)
    assert stats["unsupported_receipts"] == [] and stats["missing_receipts"] == []
    assert stats["forecast_metrics"]["llm"]["runs"] == 5
    assert stats["filter_quality"]["decisions"] == 5 and len(stats["filter_quality"]["executions"]) == 5
    shown = {r["decision_id"]: r for r in stats["routing"]}
    chosen = receipt["chosen"]
    assert shown[current["decision_id"]] == {"decision_id": current["decision_id"], "host": "host-one",
                                             "chosen": {**chosen, "agent": "agent-a"}}
    for name in formats:
        assert shown[ids[name]] == {"decision_id": ids[name], "host": None, "chosen": {**chosen, "agent": None}}
    assert main(["report", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["routing"]) == 5

    # A later decision still reads those outcomes as evidence.
    later = engine.pick("task", TASK + " again", native_candidates=cells, seed=12)
    assert len(later["semantic"]["outcome_observations"]) == 5

    # What v11 added is not part of an older receipt, and v11 requires its own shape.
    v10 = load(engine.config, ids["v10"])
    for tamper, match in (
            (lambda r: r.update(host="host-one"), "unknown keys .'host'"),
            (lambda r: r.update(host=None), "unknown keys .'host'"),
            (lambda r: r["native_candidates"][0].update(agent="agent-a"), "unknown keys .'agent'"),
            (lambda r: r["capability"]["cells"][0].update(agent=None), "unknown keys .'agent'"),
            (lambda r: r["semantic"]["worlds"]["llm"][0].update(seconds_ci=[0, 10 ** 6]), "unknown keys .'seconds_ci'"),
            (lambda r: r["semantic"]["worlds"]["llm"][0].update(token_usage_ci={}), "unknown keys .'token_usage_ci'"),
            (lambda r: r["semantic"].update(format="howhard-semantic-v8", prompt_version="semantic-reader-v8"), "forecast format"),
            (lambda r: r["capability"].update(format="howhard-capability-v6"), "capability.format")):
        bad = deepcopy(v10)
        tamper(bad)
        with pytest.raises(HowHardError, match=match):
            load(engine.config, bad)
    for tamper, match in (
            (lambda r: r.pop("host"), "missing keys .'host'"),
            (lambda r: r.update(host="not a name"), "decision.host"),
            (lambda r: r["capability"]["cells"][0].pop("agent"), "missing keys .'agent'"),
            (lambda r: r["capability"].update(format="howhard-capability-v5"), "capability.format"),
            (lambda r: r["semantic"].update(format="howhard-semantic-v7", prompt_version="semantic-reader-v7"), "forecast format"),
            (lambda r: r["semantic"]["worlds"]["llm"][0].update(seconds_ci=[10 ** 6, 10 ** 7]), "seconds_ci")):
        bad = deepcopy(receipt)
        tamper(bad)
        with pytest.raises(HowHardError, match=match):
            load(engine.config, bad)


def test_requests_prepared_before_this_contract_are_not_reused(tmp_path):
    from howhard.payload_store import get
    engine = Engine(directory=tmp_path)
    prepared = engine.prepare("task", TASK, native_candidates=[{"model": "fixture/worker", "effort": "high"}])
    value = json.loads(get(engine.config, "requests", prepared["request_id"]))
    assert value["format"] == "howhard-agent-request-v7" and value["host"] is None
    older = {k: v for k, v in value.items() if k != "host"}
    older["format"] = "howhard-agent-request-v6"
    with locked(engine.config):
        put(engine.config, "requests", digest(older), canonical(older) + b"\n")
    with pytest.raises(HowHardError, match="request"):
        engine.decide(digest(older), filtering(prepared["baseline_request"]))
