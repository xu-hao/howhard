"""Stored forecasts: reuse under the same host, the merge on each value's own scale, no stand-in (#136).

A prototype keeps one Stage 2 forecast per cell and host. A request reuses only the forecasts
its own host made, merges those of several matched prototypes on each value's own scale, and
sends every other cell to the forecaster. Requests are driven as the calling agent, so a
non-null ``baseline_request`` is exactly a call to the forecaster. Prototype matching is not
under test: where a test needs exact match weights it sets them on the real matches.
Every fixture is synthetic and owns its state.
"""

from copy import deepcopy
import json
import math
import sqlite3

import pytest

from conftest import forecast_request
from howhard import Engine, RetrievalPipeline, prototypes
from howhard.context import canonical, digest
from howhard.decisions import load
from howhard.payload_store import put
from howhard.prototypes import merge_stored_forecasts, stored_forecasts
from howhard.report import report
from howhard.storage import locked
from howhard.success import merge_forecasts, rate_guesses
from howhard.validation import HowHardError
from legacy_receipts import to_v7, to_v8, to_v9, to_v10
from test_dynamic_prototypes import ClusterEmbedderDouble, SpyLLM, TopicSimilarityDouble
from test_execution_accounting import card, counts

TOPICS = ("alpha", "beta", "gamma")
WORKER = [{"model": "fixture/worker", "effort": "high", "usd_per_mtok": 1.0}]
BY_KIND = [{"model": "fixture/worker", "effort": "high", "rate_card": card(model="fixture/worker")}]


class TopicDouble:
    """Two task texts are related exactly when they share a topic word."""

    @property
    def reranker(self):
        return self

    def score(self, query, documents):
        topics = {t for t in TOPICS if t in query.casefold()}
        return [8.0 if any(t in doc.casefold() for t in topics) else -18.0 for doc in documents]

    def search(self, records, query, limit=8, **kwargs):
        scores = self.score(query, [r["text"] for r in records])
        paired = sorted(zip(scores, records), key=lambda item: (-item[0], item[1]["key"]))[:limit]
        return {"records": deepcopy([r for _, r in paired]),
                "trace": {"seconds": 0.0, "selected": [{"key": r["key"], "score": float(s)} for s, r in paired]}}


def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


def logit(p):
    return math.log(p / (1.0 - p))


def stated(loud, share, *, loud_spread=None, seconds=600.0, seconds_ci=None, tokens=10000, usage=None, usage_ci=None):
    """A forecast row from a loud rate and an accepted-defect share ``S = silent / (clean + silent)``."""
    silent = (1.0 - loud) * share
    row = {"clean": 1.0 - loud - silent, "loud": loud, "silent": silent, "tokens": tokens, "seconds": seconds}
    if loud_spread is not None:
        row["loud_ci"] = [logistic(logit(loud) - 1.96 * loud_spread), logistic(logit(loud) + 1.96 * loud_spread)]
    if seconds_ci is not None:
        row["seconds_ci"] = list(seconds_ci)
    if usage is not None:
        row.update(token_usage=usage, tokens=sum(usage.values()))
    if usage_ci is not None:
        row["token_usage_ci"] = usage_ci
    return row


def ask(engine, text, cells, *, host=None):
    """Prepare a request and advance it to the forecast stage; ``baseline_request`` lists the cells asked."""
    return forecast_request(engine, engine.prepare("task", text, native_candidates=cells, host=host))


def answer(engine, pending, row=None, *, seed=5):
    """Decide a forecast-stage request, stating ``row`` for every cell the forecaster was asked for."""
    response = {"memory_forecasts": [], "reason": "Synthetic calling-agent forecast"}
    if pending["baseline_request"] is not None:
        response["forecasts"] = [{"family": c["family"], "effort": c["effort"], **deepcopy(row)}
                                 for c in pending["baseline_request"]["cells"]]
    result = engine.decide(pending["request_id"], response, seed=seed)
    return load(engine.config, result["decision_id"])


def route(engine, text, cells, row, *, host=None):
    """One request whose forecaster states ``row``; fails if the forecaster was not asked."""
    pending = ask(engine, text, cells, host=host)
    assert pending["baseline_request"] is not None
    return answer(engine, pending, row)


def weigh(monkeypatch, weights):
    """Set the match weight of each matched prototype by its topic; unlisted prototypes do not match."""
    real = prototypes.match_context

    def matched(engine, task_type, context):
        hits, representative = real(engine, task_type, context)
        chosen = [(proto, score, weights[topic]) for proto, score, _ in hits
                  for topic in weights if topic in proto["representative_task"].casefold()]
        return sorted(chosen, key=lambda hit: -hit[2]), representative

    monkeypatch.setattr(prototypes, "match_context", matched)


def panel_row(receipt):
    (row,) = receipt["semantic"]["worlds"]["llm"]
    return row


def reuse(receipt):
    (entry,) = receipt["semantic"]["stored_forecasts"]
    return entry


def two_prototypes(tmp_path, monkeypatch, first, second, *, cells=WORKER, shares=(0.7, 0.3)):
    """Two prototypes that both match the merged task, each with one forecast made under ``host-one``."""
    engine = Engine(directory=tmp_path, retrieval=TopicDouble())
    route(engine, "Tune alpha queue", cells, first, host="host-one")
    route(engine, "Tune beta queue", cells, second, host="host-one")
    weigh(monkeypatch, dict(zip(TOPICS, shares)))
    return engine


# --- 1. The merge on each value's own scale ------------------------------------------------

def test_two_same_host_forecasts_merge_on_the_log_odds_scale_without_a_forecaster_call(tmp_path, monkeypatch):
    first, second = stated(0.20, 0.10, loud_spread=0.5), stated(0.40, 0.10, loud_spread=0.5)
    engine = two_prototypes(tmp_path, monkeypatch, first, second)

    pending = ask(engine, "Tune alpha beta queue", WORKER, host="host-one")
    assert pending["baseline_request"] is None  # No forecaster call is made.
    receipt = answer(engine, pending)
    row = panel_row(receipt)

    # m = 0.7 * logit(0.2) + 0.3 * logit(0.4) = -1.092, so the row states loud 25.1%.
    assert row["loud"] == pytest.approx(0.2512, abs=5e-5)
    # s^2 = 0.25 + 0.7 * (-1.386 + 1.092)^2 + 0.3 * (-0.405 + 1.092)^2, and the range reads back as s.
    assert row["loud_ci"] == pytest.approx([0.0825, 0.5560], abs=5e-4)
    centre, spread = rate_guesses(row)["loud"]
    assert (centre, spread) == (pytest.approx(-1.09205, abs=5e-5), pytest.approx(0.67233, abs=5e-5))
    # With a merged accepted-defect share of 10% the row is clean 0.674, loud 0.251, silent 0.075.
    assert (row["clean"], row["silent"]) == (pytest.approx(0.6739, abs=5e-5), pytest.approx(0.0749, abs=5e-5))
    assert row["clean"] + row["loud"] + row["silent"] == pytest.approx(1.0)
    assert "silent_ci" not in row and "seconds_ci" not in row  # Equal guesses with no stated range have none.

    entry = reuse(receipt)
    assert [s["share"] for s in entry["sources"]] == pytest.approx([0.7, 0.3])
    assert [s["weight"] for s in entry["sources"]] == pytest.approx([0.7, 0.3])
    assert [s["host"] for s in entry["sources"]] == ["host-one", "host-one"]
    assert [s["forecast"] for s in entry["sources"]] == [first, second]
    assert len({s["prototype_id"] for s in entry["sources"]}) == 2
    assert entry["merged"]["loud"] == {"centre": pytest.approx(centre), "spread": pytest.approx(spread)}
    assert entry["merged"]["silent"] == {"centre": pytest.approx(logit(0.10)), "spread": 0.0}
    # The cell's beliefs start from the merged row exactly as from a forecaster's row.
    (drawn,) = receipt["semantic"]["sampling"]["cells"]
    assert drawn["outcome_beliefs"]["loud"]["prediction"] == pytest.approx(row["loud"])
    assert drawn["outcome_beliefs"]["loud"]["stated_spread"] == pytest.approx(spread)


@pytest.mark.parametrize("guesses, spread", [((0.20, 0.20), 0.5), ((0.20, 0.40), 0.67233), ((0.10, 0.60), 1.29327)])
def test_equal_guesses_keep_the_stated_spread_and_moving_them_apart_widens_it(guesses, spread):
    sources = [{"share": share, "forecast": stated(loud, 0.10, loud_spread=0.5)}
               for share, loud in zip((0.7, 0.3), guesses)]
    row, merged = merge_stored_forecasts(sources, by_kind=False)
    assert merged["loud"]["spread"] == pytest.approx(spread, abs=5e-5)
    assert rate_guesses(row)["loud"][1] == pytest.approx(spread, abs=5e-5)
    assert merged["loud"]["spread"] >= 0.5 - 1e-12


def test_one_usable_forecast_is_reused_unchanged():
    forecast = stated(0.2, 0.1, loud_spread=0.5, seconds=600, seconds_ci=[300, 1200], tokens=12345)
    row, merged = merge_stored_forecasts([{"share": 1.0, "forecast": forecast}], by_kind=False)
    assert row == forecast
    assert merged["loud"] == {"centre": pytest.approx(logit(0.2)), "spread": pytest.approx(0.5)}
    assert merged["seconds"] == {"centre": pytest.approx(math.log(600)), "spread": pytest.approx(0.3536, abs=5e-5)}
    assert merged["tokens"] == {"centre": pytest.approx(math.log(12345)), "spread": 0.0, "share": 1.0}


def test_third_match_with_nothing_usable_leaves_the_shares_of_the_other_two(tmp_path, monkeypatch):
    engine = Engine(directory=tmp_path, retrieval=TopicDouble())
    route(engine, "Tune alpha queue", WORKER, stated(0.2, 0.1, seconds=600.0, seconds_ci=[300.0, 1200.0]), host="host-one")
    route(engine, "Tune beta queue", WORKER, stated(0.2, 0.1, seconds=2400.0), host="host-one")
    # The third prototype stores a forecast for the cell, but another host made it.
    route(engine, "Tune gamma queue", WORKER, stated(0.2, 0.1, seconds=90000.0), host="host-two")
    weigh(monkeypatch, {"alpha": 0.5, "beta": 0.3, "gamma": 0.2})

    pending = ask(engine, "Tune alpha beta gamma queue", WORKER, host="host-one")
    assert pending["baseline_request"] is None
    receipt = answer(engine, pending)
    entry, row = reuse(receipt), panel_row(receipt)
    assert [s["weight"] for s in entry["sources"]] == pytest.approx([0.5, 0.3])
    assert [s["share"] for s in entry["sources"]] == pytest.approx([0.625, 0.375])
    # 0.625 * ln(600) + 0.375 * ln(2400) = ln(1009); the first spread is ln(4) / 3.92 = 0.354.
    assert row["seconds"] == pytest.approx(1009.1, abs=0.05)
    assert merge_forecasts([(1.0, row["seconds"], row["seconds_ci"])]) == (
        pytest.approx(math.log(1009.076), abs=5e-5), pytest.approx(0.72704, abs=5e-5))
    assert entry["merged"]["seconds"]["spread"] == pytest.approx(0.72704, abs=5e-5)
    assert receipt["selection"]["cells"][0]["active_time"]["stated_spread"] == pytest.approx(0.72704, abs=5e-5)


# --- 2. Reuse under the same host only ---------------------------------------------------------

def test_forecast_stored_under_another_host_is_not_reused_and_stays_stored(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=TopicDouble())
    first, second = stated(0.2, 0.1, loud_spread=0.5), stated(0.4, 0.3, seconds=50.0)
    route(engine, "Tune alpha queue", WORKER, first, host="host-one")

    pending = ask(engine, "Tune alpha queue again", WORKER, host="host-two")
    asked = pending["baseline_request"]
    assert asked is not None and [(c["model"], c["effort"]) for c in asked["cells"]] == [("fixture/worker", "high")]
    receipt = answer(engine, pending, second)
    assert receipt["host"] == "host-two" and panel_row(receipt)["loud"] == 0.4
    assert receipt["semantic"]["stored_forecasts"] == []  # The forecaster's own answer has no sources.

    kept = stored_forecasts(engine.config)
    assert [(f["host"], f["forecast"]) for f in kept] == [("host-one", first), ("host-two", second)]
    assert len({f["prototype_id"] for f in kept}) == 1

    # Each host now reuses its own forecast, unchanged, and the unnamed host has none to reuse.
    for host, forecast in (("host-one", first), ("host-two", second)):
        pending = ask(engine, "Tune alpha queue once more", WORKER, host=host)
        assert pending["baseline_request"] is None
        receipt = answer(engine, pending)
        row = panel_row(receipt)
        assert {k: v for k, v in row.items() if k not in ("family", "effort")} == forecast
        assert [s["host"] for s in reuse(receipt)["sources"]] == [host]
    assert ask(engine, "Tune alpha queue once more", WORKER)["baseline_request"] is not None


def test_new_forecast_replaces_the_same_hosts_forecast_whole(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=TopicDouble())
    ranged = stated(0.2, 0.1, loud_spread=0.5, seconds_ci=[300.0, 1200.0])
    route(engine, "Tune alpha queue", WORKER, ranged, host="host-one")
    route(engine, "Tune alpha queue", WORKER, stated(0.3, 0.2), host="host-two")
    # The cell now needs the six token kinds, which host-one's stored forecast does not state.
    usage = counts(input=4000, output=1000, cache_read=20000)
    replacement = stated(0.1, 0.05, usage=usage)
    route(engine, "Tune alpha queue", BY_KIND, replacement, host="host-one")

    kept = {f["host"]: f["forecast"] for f in stored_forecasts(engine.config)}
    assert kept == {"host-one": replacement, "host-two": stated(0.3, 0.2)}
    assert "loud_ci" not in kept["host-one"] and "seconds_ci" not in kept["host-one"]


def test_request_with_no_host_reuses_a_forecast_stored_before_hosts_were_named(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=TopicDouble())
    forecast = stated(0.2, 0.1, loud_spread=0.5, seconds=75.0)
    route(engine, "Tune alpha queue", WORKER, forecast)

    # Rewrite the cell as an older store kept it: the forecast beside the Stage 1 prior, with no host.
    path = engine.config.state_root / "payloads.sqlite3"
    with sqlite3.connect(path) as db:
        ((prior_json,),) = db.execute("SELECT prior_json FROM prototype_cells").fetchall()
        prior = json.loads(prior_json)
        (kept,) = prior.pop("forecasts")
        assert kept.pop("host") is None and kept == forecast
        prior.update(kept)
        db.execute("UPDATE prototype_cells SET prior_json=?", (json.dumps(prior),))
    assert [(f["host"], f["forecast"]) for f in stored_forecasts(engine.config)] == [(None, forecast)]

    pending = ask(engine, "Tune alpha queue again", WORKER)
    assert pending["baseline_request"] is None
    receipt = answer(engine, pending)
    assert receipt["host"] is None and reuse(receipt)["sources"][0]["host"] is None
    assert {k: v for k, v in panel_row(receipt).items() if k not in ("family", "effort")} == forecast

    # A named host does not reuse it; its own answer is stored beside the older forecast.
    named = stated(0.3, 0.2)
    route(engine, "Tune alpha queue again", WORKER, named, host="host-one")
    assert [(f["host"], f["forecast"]) for f in stored_forecasts(engine.config)] == [(None, forecast), ("host-one", named)]
    assert ask(engine, "Tune alpha queue again", WORKER)["baseline_request"] is None


# --- 3. No stand-in forecast --------------------------------------------------------------------

def test_recorded_runs_without_a_stored_forecast_go_to_the_forecaster_once_per_host(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=TopicDouble())
    for index in range(6):
        engine.observe("task", "fixture/worker", "high", "clean", native=True, task_text="Tune alpha queue",
                       tokens=8000, active_seconds=45.0, id=f"run-{index}")

    # Stage 1 is still synthesized from the recorded runs; Stage 2 builds nothing from them.
    first = engine.prepare("task", "Tune alpha queue", native_candidates=WORKER, host="host-one")
    assert first["stage"] == "filter" and first["baseline_request"] is None
    pending = forecast_request(engine, first)
    asked = pending["baseline_request"]
    assert asked is not None and [c["model"] for c in asked["cells"]] == ["fixture/worker"]
    assert "preset_forecasts" not in pending["request"] and "no_memory_forecasts" not in pending["request"]

    forecast = stated(0.31, 0.17, tokens=7777, seconds=55.0)
    receipt = answer(engine, pending, forecast)
    row = panel_row(receipt)
    assert {k: v for k, v in row.items() if k not in ("family", "effort")} == forecast
    assert (row["clean"], row["loud"], row["silent"]) != (0.88, 0.08, 0.04)
    assert receipt["semantic"]["stored_forecasts"] == []
    # The six runs still count as evidence beside the forecaster's row.
    assert receipt["selection"]["cells"][0]["active_time"]["evidence_weight"] > 0

    assert ask(engine, "Tune alpha queue", WORKER, host="host-one")["baseline_request"] is None
    assert ask(engine, "Tune alpha queue", WORKER, host="host-two")["baseline_request"] is not None


def test_explicit_backend_is_asked_for_a_cell_with_runs_and_no_stored_forecast(tmp_path):
    spy = SpyLLM()
    engine = Engine(directory=tmp_path, llm=spy, retrieval=TopicSimilarityDouble())
    for index in range(4):
        engine.observe("task", "fixture/worker", "high", "clean", native=True, task_text="Repair rollback",
                       tokens=8000, active_seconds=45.0, id=f"run-{index}")
    spy.calls.clear()
    result = engine.pick("task", "Repair rollback", native_candidates=WORKER, seed=3)
    # Stage 1 is synthesized from the runs; the forecaster is asked for the Stage 2 row.
    assert [c["kind"] for c in spy.calls if c.get("phase") == "baseline"] == ["predict"]
    (row,) = result["semantic"]["worlds"]["llm"]
    assert (row["clean"], row["loud"], row["silent"], row["tokens"], row["seconds"]) == (0.92, 0.06, 0.02, 10000, 120)
    spy.calls.clear()
    again = engine.pick("task", "Repair rollback again", native_candidates=WORKER, seed=3)
    assert spy.calls == [] and again["semantic"]["backend"] == "prototype-materialized"
    assert len(again["semantic"]["stored_forecasts"]) == 1


# --- 4. Token kinds -------------------------------------------------------------------------------

def test_stored_forecast_without_the_six_token_kinds_is_not_reused_for_a_cell_priced_by_kind(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=TopicDouble())
    route(engine, "Tune alpha queue", WORKER, stated(0.2, 0.1), host="host-one")
    assert ask(engine, "Tune alpha queue", WORKER, host="host-one")["baseline_request"] is None

    pending = ask(engine, "Tune alpha queue", BY_KIND, host="host-one")
    asked = pending["baseline_request"]
    assert asked is not None
    assert [(c["model"], c["forecast_format"]) for c in asked["cells"]] == [("fixture/worker", "category-v1")]


def test_kind_every_source_states_as_exactly_zero_stays_exactly_zero(tmp_path, monkeypatch):
    first = stated(0.2, 0.1, usage=counts(input=4000, output=1000, cache_read=20000, cache_write=500),
                   usage_ci={"input": [2000, 8000], "cache_write_1h": [0, 0]})
    second = stated(0.2, 0.1, usage=counts(input=4000, output=2000, cache_read=30000),
                    usage_ci={"cache_write_5m": [0, 400]})
    engine = two_prototypes(tmp_path, monkeypatch, first, second, cells=BY_KIND)

    pending = ask(engine, "Tune alpha beta queue", BY_KIND, host="host-one")
    assert pending["baseline_request"] is None
    receipt = answer(engine, pending)
    row, merged = panel_row(receipt), reuse(receipt)["merged"]["token_usage"]

    # Every source states the kind as exactly zero: it stays exactly zero, with no range.
    assert row["token_usage"]["cache_write_1h"] == 0 and "cache_write_1h" not in row["token_usage_ci"]
    assert merged["cache_write_1h"] == {"centre": None, "spread": 0.0, "share": 0.0}
    belief = receipt["selection"]["cells"][0]["token_beliefs"]["cache_write_1h"]
    assert (belief["expected_tokens"], belief["spread"]) == (0.0, 0.0)
    # One source states the kind above zero: its value times that source's share.
    assert row["token_usage"]["cache_write"] == pytest.approx(0.7 * 500)
    assert merged["cache_write"] == {"centre": pytest.approx(math.log(500)), "spread": 0.0, "share": pytest.approx(0.7)}
    # A kind at zero with a non-zero upper end is not exactly zero, so it keeps a range.
    assert row["token_usage"]["cache_write_5m"] > 0 and row["token_usage_ci"]["cache_write_5m"][1] > 0
    # Equal guesses keep the one stated spread scaled by its share; different guesses widen.
    assert row["token_usage"]["input"] == pytest.approx(4000)
    assert merged["input"]["spread"] == pytest.approx(math.sqrt(0.7) * math.log(4) / 3.92)
    assert row["token_usage"]["output"] == pytest.approx(math.exp(0.7 * math.log(1000) + 0.3 * math.log(2000)))
    assert merged["output"]["spread"] == pytest.approx(math.sqrt(0.21) * math.log(2))
    assert row["tokens"] == pytest.approx(sum(row["token_usage"].values()))


# --- 5. The receipt ---------------------------------------------------------------------------------

def edit_share(receipt):
    first, second = reuse(receipt)["sources"]
    first["share"], second["share"] = 0.6, 0.4


def edit_weight(receipt):
    reuse(receipt)["sources"][0]["weight"] = 0.6


def edit_source_row(receipt):
    forecast = reuse(receipt)["sources"][0]["forecast"]
    forecast.update(loud=forecast["loud"] + 0.05, clean=forecast["clean"] - 0.05)


def edit_source_range(receipt):
    reuse(receipt)["sources"][1]["forecast"].pop("loud_ci")


def edit_source_host(receipt):
    reuse(receipt)["sources"][1]["host"] = "host-two"


def edit_merged_spread(receipt):
    reuse(receipt)["merged"]["loud"]["spread"] += 0.1


def edit_panel_row(receipt):
    row = panel_row(receipt)
    row.update(loud=row["loud"] - 0.01, clean=row["clean"] + 0.01)


def drop_reuse_record(receipt):
    del receipt["semantic"]["stored_forecasts"]


EDITS = (edit_share, edit_weight, edit_source_row, edit_source_range, edit_source_host, edit_merged_spread,
         edit_panel_row, drop_reuse_record)


@pytest.mark.parametrize("edit", EDITS, ids=lambda edit: edit.__name__)
def test_receipt_with_an_altered_share_source_row_or_source_host_is_refused(tmp_path, monkeypatch, edit):
    engine = two_prototypes(tmp_path, monkeypatch, stated(0.2, 0.1, loud_spread=0.5), stated(0.4, 0.1, loud_spread=0.5))
    receipt = answer(engine, ask(engine, "Tune alpha beta queue", WORKER, host="host-one"))
    assert load(engine.config, deepcopy(receipt)) == receipt
    edit(receipt)
    with pytest.raises(HowHardError):
        load(engine.config, receipt)


def test_receipt_host_differing_from_its_sources_is_refused(tmp_path, monkeypatch):
    engine = two_prototypes(tmp_path, monkeypatch, stated(0.2, 0.1), stated(0.4, 0.1))
    receipt = answer(engine, ask(engine, "Tune alpha beta queue", WORKER, host="host-one"))
    receipt["host"] = "host-two"
    with pytest.raises(HowHardError, match="another host"):
        load(engine.config, receipt)


def test_full_panel_response_cannot_change_a_reused_row(tmp_path):
    cells = WORKER + [{"model": "fixture/other", "effort": "high", "usd_per_mtok": 2.0}]
    engine = Engine(directory=tmp_path, retrieval=TopicDouble())
    forecast = stated(0.2, 0.1)
    route(engine, "Tune alpha queue", WORKER, forecast, host="host-one")

    pending = ask(engine, "Tune alpha queue", cells, host="host-one")
    assert [c["model"] for c in pending["baseline_request"]["cells"]] == ["fixture/other"]
    families = {c["model"]: c["family"] for c in pending["request"]["cells"]}
    panel = [{"family": families[model], "effort": "high", **row}
             for model, row in (("fixture/worker", stated(0.05, 0.01)), ("fixture/other", stated(0.3, 0.2)))]
    with pytest.raises(HowHardError, match="must not be changed"):
        engine.decide(pending["request_id"], {"forecasts": panel, "memory_forecasts": [], "reason": "Rewrite"}, seed=1)
    panel[0] = {"family": families["fixture/worker"], "effort": "high", **forecast}
    result = engine.decide(pending["request_id"], {"forecasts": panel, "memory_forecasts": [], "reason": "Repeat"}, seed=1)
    receipt = load(engine.config, result["decision_id"])
    assert [e["family"] for e in receipt["semantic"]["stored_forecasts"]] == [families["fixture/worker"]]


def test_v7_to_v10_receipts_with_reused_forecasts_load_observe_replay_and_report(tmp_path, monkeypatch):
    engine = two_prototypes(tmp_path, monkeypatch, stated(0.2, 0.1, loud_spread=0.5), stated(0.4, 0.1, loud_spread=0.5))
    current = answer(engine, ask(engine, "Tune alpha beta queue", WORKER, host="host-one"))
    assert len(current["semantic"]["stored_forecasts"]) == 1
    stored = {"v11": digest(current)}
    for name, legacy in (("v10", to_v10(current)), ("v9", to_v9(current)), ("v8", to_v8(current)), ("v7", to_v7(current))):
        stored[name] = digest(legacy)
        with locked(engine.config):
            put(engine.config, "decisions", stored[name], canonical(legacy) + b"\n")
        loaded = load(engine.config, stored[name])
        assert loaded == legacy and loaded["format"] == f"howhard-decision-{name}"
        assert "stored_forecasts" not in loaded["semantic"] and "host" not in loaded
    for index, name in enumerate(stored):
        engine.observe_decision(stored[name], "clean", id=f"from-{name}", seconds=10.0 + index)
    beliefs = engine.memory_beliefs()
    assert beliefs["unscored_runs"] == 0
    assert next(e for e in beliefs["experts"] if e["key"] == "llm")["evidence_count"] == 5
    stats = report(engine)
    assert stats["unsupported_receipts"] == [] and stats["forecast_metrics"]["llm"]["runs"] == 5


# --- 6. What does not change --------------------------------------------------------------------------

def test_stage_one_filter_is_the_same_when_every_run_and_forecast_is_the_unnamed_hosts(tmp_path):
    cells = [{"model": model, "effort": effort, "usd_per_mtok": price}
             for model, price in (("fixture/small", 0.4), ("fixture/large", 1.2)) for effort in ("low", "high")]
    engine = Engine(directory=tmp_path, llm=SpyLLM(), retrieval=TopicSimilarityDouble())
    engine.pick("task", "Repair rollback boundary", native_candidates=cells, seed=7)
    runs = [("fixture/small", "low", "loud"), ("fixture/small", "low", "clean"), ("fixture/small", "high", "clean"),
            ("fixture/large", "low", "clean"), ("fixture/extra", "high", "clean"), ("fixture/extra", "high", "clean"),
            ("fixture/extra", "high", "silent")]
    for index, (model, effort, outcome) in enumerate(runs):
        extra = {} if outcome == "clean" else {"cause": "capability"}
        engine.observe("task", model, effort, outcome, native=True, task_text="Repair rollback boundary",
                       id=f"run-{index}", active_seconds=100.0 + 500 * index, tokens=4000, **extra)
    wider = cells + [{"model": "fixture/extra", "effort": "high", "usd_per_mtok": 0.8}]

    pending = engine.prepare("task", "Repair rollback boundary regression", native_candidates=wider)
    assert pending["stage"] == "filter" and pending["baseline_request"] is None
    names = {c["key"]: (c["model"], c["effort"]) for c in pending["request"]["cells"]}
    synthesized = pending["request"]["no_memory_filter"]
    # The filter this state gave before stored forecasts were kept per host.
    assert [names[key] for key in synthesized["order"]] == [
        ("fixture/large", "low"), ("fixture/large", "high"), ("fixture/small", "low"),
        ("fixture/extra", "high"), ("fixture/small", "high")]
    assert (names[synthesized["lower"]], names[synthesized["upper"]]) == (("fixture/large", "low"), ("fixture/small", "high"))


def cell_stats(engine):
    with sqlite3.connect(engine.config.state_root / "payloads.sqlite3") as db:
        ((stats,),) = db.execute("SELECT stats_json FROM prototype_cells").fetchall()
    return json.loads(stats)


@pytest.mark.parametrize("run_host, timely_share", [("host-one", (1.5 / 10) ** 2), ("host-two", 1.0), (None, 1.0)])
def test_clean_run_is_timely_against_the_forecast_of_its_own_host(tmp_path, run_host, timely_share):
    engine = Engine(directory=tmp_path, retrieval=TopicDouble())
    decisions = {"host-one": route(engine, "Tune alpha queue", WORKER, stated(0.2, 0.1, seconds=100.0), host="host-one"),
                 "host-two": route(engine, "Tune alpha queue", WORKER, stated(0.2, 0.1, seconds=10000.0), host="host-two")}
    # The run took 1,000 active seconds: ten times host-one's forecast and a tenth of host-two's.
    if run_host is None:
        # A run with no decision belongs to the unnamed host, which stored no forecast: no overrun discount.
        engine.observe("task", "fixture/worker", "high", "clean", native=True, task_text="Tune alpha queue",
                       active_seconds=1000.0, id="run")
    else:
        engine.observe_decision(digest(decisions[run_host]), "clean", active_seconds=1000.0, id="run")
    engine.prepare("task", "Tune alpha queue", native_candidates=WORKER, host="host-one")
    stats = cell_stats(engine)
    assert stats["run_count"] == 1 and stats["capability_clean"] > 0
    assert stats["timely_clean"] == pytest.approx(timely_share * stats["capability_clean"])


def test_split_copies_every_hosts_stored_forecast_to_each_child(tmp_path):
    embedder = ClusterEmbedderDouble()
    spy = SpyLLM()
    engine = Engine(directory=tmp_path, llm=spy, retrieval=RetrievalPipeline(tmp_path / ".howhard", embedder=embedder,
                                                                           reranker=embedder))
    engine.observe("task", "fixture/worker", "high", "clean", native=True, task_text="Repair rollback",
                   tokens=3000, active_seconds=120.0, id="seed-broad")
    for host in ("host-one", "host-two"):
        engine.pick("task", "Repair rollback", native_candidates=WORKER, seed=1, host=host)
    (parent,) = {f["prototype_id"] for f in stored_forecasts(engine.config)}
    assert {f["host"] for f in stored_forecasts(engine.config)} == {"host-one", "host-two"}

    for index in range(6):
        engine.observe("task", "fixture/worker", "high", "clean", native=True, tokens=1500, active_seconds=20.0,
                       task_text=f"Repair rollback unit helper case {index}", id=f"fast-{index}")
        engine.observe("task", "fixture/worker", "high", "clean", native=True, tokens=45000, active_seconds=7200.0,
                       task_text=f"Repair rollback cluster migration phase {index}", id=f"slow-{index}")
    spy.calls.clear()
    for host in ("host-one", "host-two"):
        engine.pick("task", "Repair rollback unit helper assertion", native_candidates=WORKER, seed=1, host=host)
        engine.pick("task", "Repair rollback cluster migration cutover", native_candidates=WORKER, seed=1, host=host)
    assert spy.calls == []  # Each child is covered under both hosts.

    kept = stored_forecasts(engine.config)
    children = {f["prototype_id"] for f in kept}
    assert len(children) == 2 and parent not in children
    assert sorted((f["prototype_id"], f["host"]) for f in kept) == sorted(
        (child, host) for child in children for host in ("host-one", "host-two"))
