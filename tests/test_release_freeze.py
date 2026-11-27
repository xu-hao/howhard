"""The frozen v11 formats and the release preparation (#138).

Frozen format labels, older receipts at their public boundaries, the import of names for
runs whose receipt stores none, the notice for retired settings, the comparison bound and
the tolerance for a probability that rounding carried above 1.
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib
import json
from pathlib import Path
import re
import tomllib

import pytest

from conftest import row
from howhard import Engine, capability, context, engine as engine_module, semantic
from howhard.cli import main
from howhard.context import canonical, digest
from howhard.decisions import load
from howhard.payload_store import get, put
from howhard.prototypes import stored_forecasts
from howhard.report import report
from howhard.storage import locked
from howhard.validation import HowHardError
from legacy_receipts import authentic_v10, to_v7, to_v8, to_v9, to_v10
from test_capability_filter import SimilarityDouble, filtering, finish, minute_clock, start  # noqa: F401

REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "reference.md"
BUILDERS = {"v10": lambda receipt: deepcopy(receipt), "v9": to_v9, "v8": to_v8, "v7": to_v7}


def store(engine, receipt):
    """Store a receipt under its digest, as a build of its own format did."""
    decision_id = digest(receipt)
    with locked(engine.config):
        put(engine.config, "decisions", decision_id, canonical(receipt) + b"\n")
    return decision_id


def llm_runs(engine):
    beliefs = engine.memory_beliefs()
    assert beliefs["unscored_runs"] == 0
    return next(e for e in beliefs["experts"] if e["key"] == "llm")["evidence_count"]


# --- 1. Frozen format labels ----------------------------------------------------------------

def frozen_table():
    """The rows of the frozen-format table in the reference: (stored shape, constant, label)."""
    text = REFERENCE.read_text(encoding="utf-8")
    section = text.split("### Frozen formats", 1)[1].split("\n#", 1)[0]
    rows = re.findall(r"^\| ([^|]+?) \| `([^`]+)` \| `([^`]+)` \|$", section, re.M)
    assert rows, "the reference has no frozen-format table"
    return rows


def constant(dotted):
    module, _, attribute = dotted.rpartition(".")
    return getattr(importlib.import_module("howhard." + module), attribute)


def test_every_frozen_label_appears_in_the_reference_and_equals_the_code():
    documented = {name: label for _, name, label in frozen_table()}
    packaged = tomllib.loads((Path(semantic.__file__).parent / "data" / "semantic.toml").read_text())
    in_code = {"context.DECISION_FORMAT": context.DECISION_FORMAT,
               "semantic.FORMAT": packaged["format"], "semantic.PROMPT_VERSION": packaged["prompt_version"],
               "semantic.INTERVAL_KIND": semantic.INTERVAL_KIND, "capability.FORMAT": capability.FORMAT,
               "engine.REQUEST_FORMAT": engine_module.REQUEST_FORMAT}
    assert {name: documented.get(name) for name in in_code} == in_code
    for name, label in documented.items():
        assert constant(name) == label, name
    # A v11 receipt pairs its semantic state and reader prompt with its decision format.
    assert semantic.RECEIPT_SEMANTICS[context.DECISION_FORMAT] == (packaged["format"], packaged["prompt_version"])
    assert (context.DECISION_FORMAT, semantic.INTERVAL_KIND, capability.FORMAT, engine_module.REQUEST_FORMAT) == (
        "howhard-decision-v11", "log-odds-grid", "howhard-capability-v6", "howhard-agent-request-v7")


# --- 2. A v11 receipt of the frozen shape, and earlier shapes -------------------------------

def earlier_shapes():
    """Receipts labelled v11 in the shapes earlier unreleased builds stored."""
    def no_prediction(r):  # before the corrected prediction
        r["semantic"].pop("prediction")

    def one_expert(r):  # before the cost was averaged over the beliefs
        r["semantic"]["sampled_expert"] = "llm"
        for cell in r["semantic"]["sampling"]["cells"]:
            cell.pop("sampled")
            cell["expert"] = "llm"

    def no_host_band(r):  # before the host-aware band
        for field in ("caught_share", "core", "proven_families", "reserved"):
            r["capability"].pop(field)

    def dirichlet(r):  # before the outcome beliefs on the log-odds scale
        r["semantic"]["interval_kind"] = "dirichlet-posterior-mixture"
    return [no_prediction, one_expert, no_host_band, dirichlet]


def test_a_frozen_v11_receipt_loads_revalidates_and_reports_and_earlier_shapes_are_refused(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    result = finish(engine, start(engine, [{"model": "fixture/a", "effort": "high", "usd_per_mtok": 1.0}], host="alpha"))
    receipt = load(engine.config, result["decision_id"])
    assert receipt["format"] == "howhard-decision-v11" and load(engine.config, receipt) == receipt
    stats = report(engine)
    assert stats["receipts"] == {"formats": {"howhard-decision-v11": 1}, "rounded_probabilities": 0}
    assert [r["host"] for r in stats["routing"]] == ["alpha"]
    for change in earlier_shapes():
        older = deepcopy(receipt)
        change(older)
        with pytest.raises(HowHardError):
            load(engine.config, older)
    # Stored, an earlier shape is refused with the ordinary error, and so is a report over it.
    older = deepcopy(receipt)
    earlier_shapes()[0](older)
    store(engine, older)
    with pytest.raises(HowHardError, match="decision.semantic"):
        report(engine)


def test_a_request_prepared_before_the_release_is_refused_with_the_ordinary_errors(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    engine.observe("task", "fixture/a", "high", "clean", native=True, task_text="Repair rollback", id="earlier")
    pending = start(engine, [{"model": "fixture/a", "effort": "high", "usd_per_mtok": 1.0}])
    bundle = json.loads(get(engine.config, "requests", pending["request_id"]))

    def stored(change):
        value = deepcopy(bundle)
        change(value)
        request_id = digest(value)
        with locked(engine.config):
            put(engine.config, "requests", request_id, canonical(value) + b"\n")
        return request_id

    v6 = stored(lambda b: b.update(format="howhard-agent-request-v6"))
    no_record = stored(lambda b: b["learned_state"].pop("prediction"))
    unlabelled = stored(lambda b: [c.pop("agent") for c in b["learned_state"]["evidence_candidates"]["candidates"]])
    response = filtering(pending["baseline_request"])
    with pytest.raises(HowHardError, match="request integrity or format mismatch"):
        engine.decide(v6, response)
    with pytest.raises(HowHardError, match=r"request.learned_state: missing keys \['prediction'\]"):
        engine.decide(no_record, response)
    assert bundle["learned_state"]["evidence_candidates"]["candidates"]
    with pytest.raises(HowHardError, match="request.learned_state.evidence_candidates"):
        engine.decide(unlabelled, response)
    # The request as prepared still decides.
    assert "decision_id" in finish(engine, pending)


# --- 3. v7 to v10 receipts at every boundary ---------------------------------------------------

@pytest.fixture
def authentic(tmp_path):
    """A store laid out as a 5.0.0 build left it, holding its authentic v10 receipt."""
    receipt = authentic_v10(tmp_path)
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    assert len(engine.rows) == 6 and receipt["format"] == "howhard-decision-v10"
    return engine, receipt


@pytest.mark.parametrize("version", ["v10", "v9", "v8", "v7"])
def test_older_receipts_load_observe_replay_and_report(authentic, version):
    engine, v10 = authentic
    receipt = BUILDERS[version](v10)
    decision_id = store(engine, receipt)
    loaded = load(engine.config, decision_id)
    assert loaded == receipt and loaded["format"] == f"howhard-decision-{version}"
    assert loaded["chosen"]["model"] == "fixture/model-a"

    observed = engine.observe_decision(decision_id, "clean", id=f"through-{version}", active_seconds=500.)
    assert observed["decision_id"] == decision_id and observed["task_text"] == receipt["context"]["task_text"]
    assert llm_runs(engine) == 1

    stats = report(engine)
    assert stats["missing_receipts"] == stats["unsupported_receipts"] == []
    assert stats["forecast_metrics"]["llm"]["runs"] == 1
    assert stats["receipts"] == {"formats": {f"howhard-decision-{version}": 1}, "rounded_probabilities": 0}
    assert stats["routing"] == [{"decision_id": decision_id, "host": None,
                                 "chosen": {**receipt["chosen"], "agent": None}}]
    assert [e["execution_id"] for e in stats["filter_quality"]["executions"]] == [f"through-{version}"]
    # Its run is evidence for a later v11 decision on the same task, which validates too.
    later = finish(engine, start(engine, receipt["native_candidates"]))
    assert f"through-{version}" in {c["id"] for c in later["semantic"]["evidence_candidates"]["candidates"]}
    assert load(engine.config, later["decision_id"])["format"] == "howhard-decision-v11"


def test_receipts_older_than_v7_stay_refused_on_load_and_skipped_when_scoring(authentic):
    engine, v10 = authentic
    ancient = {**deepcopy(v10), "format": "howhard-decision-v6"}
    decision_id = store(engine, ancient)
    with pytest.raises(HowHardError, match="decision.format: unsupported version"):
        load(engine.config, decision_id)
    assert load(engine.config, decision_id, for_scoring=True) is None
    stats = report(engine)
    assert stats["receipts"]["formats"] == {"howhard-decision-v6": 1}
    assert stats["routing"] == [] and stats["forecast_metrics"]["llm"]["runs"] == 0


# --- 4. A probability that rounding carried above 1 --------------------------------------------

@pytest.fixture
def saturated(tmp_path, minute_clock):
    """A v11 receipt whose strongest cell holds P_clear = P_timely = 1 after many clean runs."""
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    cells = [{"model": f"fixture/model-{i}", "effort": "high"} for i in range(2)]
    for outcome, count in (("clean", 30), ("loud", 2), ("silent", 1)):
        extra = {} if outcome == "clean" else {"cause": "capability"}
        for i in range(count):
            engine.observe("task", cells[1]["model"], "high", outcome, native=True,
                           task_text="Repair rollback", id=f"{outcome}-{i}", **extra)
    pending = start(engine, cells)
    receipt = load(engine.config, finish(engine, pending, filtering(pending["baseline_request"], 0, 1))["decision_id"])
    top = receipt["capability"]["cells"][-1]
    assert top["P_clear"] == top["P_timely"] == 1.
    return engine, receipt


def rounded_check(engine, receipt):
    """The strongest cell rounded just above 1 loads, observes, replays and reports; more is refused."""
    fields = ("P_clear", "P_timely") if "P_timely" in receipt["capability"]["cells"][-1] else ("P_clear",)

    def rounded(value):
        changed = deepcopy(receipt)
        for field in fields:
            changed["capability"]["cells"][-1][field] = value
        top = changed["capability"]["cells"][-1]["key"]
        for item in [changed["capability"]["executed"], *changed["capability"]["disagreements"]]:
            if item["key"] == top:
                item["P_clear"] = value
        return changed

    for beyond in (1.000001, 1.0000000002):
        with pytest.raises(HowHardError, match="monotone"):
            load(engine.config, rounded(beyond))
    receipt = rounded(1.0000000000000009)
    decision_id = store(engine, receipt)
    assert load(engine.config, decision_id) == receipt
    engine.observe_decision(decision_id, "clean", id="after-rounding", active_seconds=10.)
    assert llm_runs(engine) == 1
    stats = report(engine)
    assert stats["unsupported_receipts"] == [] and stats["forecast_metrics"]["llm"]["runs"] == 1
    assert stats["receipts"]["rounded_probabilities"] == 1
    assert decision_id in {r["decision_id"] for r in stats["routing"]}


def test_a_probability_rounded_above_1_in_a_v11_receipt_is_accepted_and_more_is_refused(saturated):
    engine, receipt = saturated
    rounded_check(engine, receipt)


@pytest.mark.parametrize("version", ["v10", "v9", "v8", "v7"])
def test_a_probability_rounded_above_1_in_an_older_receipt_is_accepted_and_more_is_refused(authentic, version):
    engine, v10 = authentic
    receipt = BUILDERS[version](v10)
    assert receipt["capability"]["cells"][-1]["P_clear"] < 1
    rounded_check(engine, receipt)


# --- 5. Names for runs whose receipt stores none -------------------------------------------------

def names_file(tmp_path, value, name="names.json"):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def import_names(path, capsys):
    capsys.readouterr()
    status = main(["names", "import", "--file", str(path), "--json"])
    captured = capsys.readouterr()
    return status, json.loads(captured.out) if status == 0 else captured.err


def chore(engine, host, cells):
    """A v11 decision on another task type, whose candidates are the fixture's six recorded runs."""
    pending = engine.prepare("chore", "Repair rollback", native_candidates=cells, host=host)
    return load(engine.config, finish(engine, pending, filtering(pending["baseline_request"], 0, len(cells) - 1))["decision_id"])


def loud_rungs(receipt, family):
    entry = next(c["values"]["loud"] for c in receipt["semantic"]["prediction"]["cells"] if c["family"] == family)
    return [rung["weight"] for rung in entry["rungs"]]


def routed(stats, decision_id):
    return next(r for r in stats["routing"] if r["decision_id"] == decision_id)


def test_an_import_names_the_runs_of_an_older_decision_and_the_same_file_again_writes_nothing(authentic, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    engine, v10 = authentic
    older = store(engine, v10)
    families = {c["model"]: c["family"] for c in v10["capability"]["cells"]}
    engine.observe_decision(older, "loud", id="through-v10", cause="capability", active_seconds=600.)
    cells = [{**c, "agent": agent} for c, agent in zip(v10["native_candidates"], ("agent-x", "agent-y", "agent-z"))]
    family_a = families["fixture/model-a"]

    # With no import the older decision and its runs belong to the unnamed host and agent.
    before = chore(engine, "host-a", cells)
    assert routed(report(engine), older) == {"decision_id": older, "host": None, "chosen": {**v10["chosen"], "agent": None}}
    assert {c["agent"] for c in before["semantic"]["evidence_candidates"]["candidates"]} == {None}
    assert before["capability"]["caught_share"]["F_host"] == 0. and before["capability"]["caught_share"]["F_all"] > 0
    assert loud_rungs(before, family_a)[0] > 0 and loud_rungs(before, family_a)[1:] == [0.] * 4

    # Hosts by decision ID, agents by family or by a registered model id; the rest is skipped.
    path = names_file(tmp_path, {
        "hosts": {older: "host-a", digest(before): "host-q", "f" * 64: "host-q", "not-a-decision": "host-q"},
        "agents": {"fixture/model-a": "agent-x", families["fixture/model-b"]: "agent-y", "unregistered": "agent-q"}})
    status, result = import_names(path, capsys)
    assert status == 0
    assert result["hosts"] == {"stored": 1, "unchanged": 0, "replaced": 0, "skipped": 3}
    assert result["agents"] == {"stored": 2, "unchanged": 0, "replaced": 0, "skipped": 1}
    reasons = {(item["table"], item["key"]): item["reason"] for item in result["skipped"]}
    assert reasons[("hosts", digest(before))] == "a howhard-decision-v11 receipt stores its own host"
    assert reasons[("hosts", "f" * 64)] == reasons[("hosts", "not-a-decision")] == "not a stored decision"
    assert reasons[("agents", "unregistered")] == "not a registered family or model"

    # The report shows the names, a v11 decision keeps its own, and no receipt was rewritten.
    engine.refresh()
    stats = report(engine)
    assert routed(stats, older) == {"decision_id": older, "host": "host-a", "chosen": {**v10["chosen"], "agent": "agent-x"}}
    assert routed(stats, digest(before))["host"] == "host-a"
    assert json.loads(get(engine.config, "decisions", older)) == v10

    # The older decision's runs now belong to that host and agent, and runs with no decision to
    # the agent of their family.
    after = chore(engine, "host-a", cells)
    assert after["capability"]["caught_share"]["F_host"] == pytest.approx(after["capability"]["caught_share"]["F_all"])
    assert loud_rungs(after, family_a) == pytest.approx([loud_rungs(after, family_a)[0]] * 5)
    agents = {c["id"]: c["agent"] for c in after["semantic"]["evidence_candidates"]["candidates"]}
    assert agents == {f"run-{i}": "agent-x" if i < 3 else "agent-y" for i in range(6)}
    assert load(engine.config, digest(after)) == after

    # The same file again writes nothing.
    payloads = (tmp_path / ".howhard" / "payloads.sqlite3")
    stamp = payloads.stat().st_mtime_ns, payloads.read_bytes()
    status, again = import_names(path, capsys)
    assert status == 0
    assert again["hosts"] == {"stored": 0, "unchanged": 1, "replaced": 0, "skipped": 3}
    assert again["agents"] == {"stored": 0, "unchanged": 2, "replaced": 0, "skipped": 1}
    assert (payloads.stat().st_mtime_ns, payloads.read_bytes()) == stamp

    # A later file replaces the names an earlier one gave.
    status, later = import_names(names_file(tmp_path, {"hosts": {older: "host-b"}}, "later.json"), capsys)
    assert later["hosts"] == {"stored": 0, "unchanged": 0, "replaced": 1, "skipped": 0}
    assert later["agents"] == {"stored": 0, "unchanged": 0, "replaced": 0, "skipped": 0} and later["skipped"] == []
    engine.refresh()
    assert routed(report(engine), older)["host"] == "host-b"
    assert routed(report(engine), older)["chosen"]["agent"] == "agent-x"


@pytest.mark.parametrize("value,match", [
    ({"hosts": {}, "families": {}}, r"unknown keys \['families'\]"),
    ([], "expected an object"),
    ({"hosts": []}, "names.hosts: expected an object"),
    ({"agents": {"fixture/model-a": "agent x"}}, "names.agents.fixture/model-a"),
    ({"hosts": {"0" * 64: "h" * 65}}, "at most 64 characters"),
    ({"agents": {"fixture/model-a": "one"}, "hosts": {"0" * 64: ""}}, "non-empty string"),
])
def test_a_names_file_of_another_shape_or_with_an_invalid_name_is_refused_whole(authentic, tmp_path, monkeypatch, capsys, value, match):
    monkeypatch.chdir(tmp_path)
    status, error = import_names(names_file(tmp_path, value), capsys)
    assert status == 2 and re.search(match, error)
    engine, _ = authentic
    assert engine.import_names({}) == {"hosts": {"stored": 0, "unchanged": 0, "replaced": 0, "skipped": 0},
                                       "agents": {"stored": 0, "unchanged": 0, "replaced": 0, "skipped": 0},
                                       "skipped": []}


def test_a_model_id_and_its_family_may_not_name_one_family_differently(authentic):
    engine, v10 = authentic
    family = next(c["family"] for c in v10["capability"]["cells"] if c["model"] == "fixture/model-a")
    with pytest.raises(HowHardError, match="name family"):
        engine.import_names({"agents": {"fixture/model-a": "agent-x", family: "agent-y"}})
    assert engine.import_names({"agents": {"fixture/model-a": "agent-x", family: "agent-x"}})["agents"]["stored"] == 1


def test_a_prototype_forecast_stored_before_v11_is_reused_only_without_a_host_name_with_or_without_an_import(tmp_path):
    import sqlite3
    from test_stored_forecast_reuse import TopicDouble, WORKER, answer, ask, route, stated
    engine = Engine(directory=tmp_path, retrieval=TopicDouble())
    forecast = stated(0.2, 0.1, seconds=75.0)
    decision = route(engine, "Tune alpha queue", WORKER, forecast)
    # Rewrite the cell as a store before v11 kept it: the forecast beside the Stage 1 prior, with no host.
    with sqlite3.connect(engine.config.state_root / "payloads.sqlite3") as db:
        ((prior_json,),) = db.execute("SELECT prior_json FROM prototype_cells").fetchall()
        prior = json.loads(prior_json)
        (kept,) = prior.pop("forecasts")
        kept.pop("host")
        prior.update(kept)
        db.execute("UPDATE prototype_cells SET prior_json=?", (json.dumps(prior),))
    older = store(engine, to_v10(decision))

    def reused_by(host):
        return ask(engine, "Tune alpha queue again", WORKER, host=host)["baseline_request"] is None

    assert (reused_by(None), reused_by("host-one")) == (True, False)
    result = engine.import_names({"hosts": {older: "host-one"}, "agents": {"fixture/worker": "agent-x"}})
    assert (result["hosts"]["stored"], result["agents"]["stored"]) == (1, 1)
    assert (reused_by(None), reused_by("host-one")) == (True, False)
    assert [(f["host"], f["forecast"]) for f in stored_forecasts(engine.config)] == [(None, forecast)]
    receipt = answer(engine, ask(engine, "Tune alpha queue again", WORKER))
    assert receipt["host"] is None and receipt["semantic"]["stored_forecasts"][0]["sources"][0]["host"] is None


# --- 6. One notice for retired settings ---------------------------------------------------------

RETIRED = ('on_cap_unmet = "flag-review"\n[sampling]\noutcome_recency = 0.9\nprior_strength = 8.0\n'
           'min_concentration = 2.0\nmax_concentration = 1000000.0\nadmission_quantile = 0.8\n'
           '[capability]\nmin_clear_probability = 0.5\noverpowered_gain_margin = 0.05\n')


def test_a_configuration_that_sets_every_retired_key_loads_and_one_notice_names_them_all(tmp_path, monkeypatch, capsys):
    outputs = {}
    for name, settings in (("plain", ""), ("retired", RETIRED)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "howhard.toml").write_text(settings + '[task_types]\ntask = "standard"\n')
        monkeypatch.chdir(directory)
        capsys.readouterr()
        assert main(["paths", "--config", "howhard.toml", "--json"]) == 0
        captured = capsys.readouterr()
        outputs[name] = json.loads(captured.out), captured.err.splitlines()
        engine = Engine(directory / "howhard.toml")
        outputs[name] += ({k: engine.config.data[k] for k in ("sampling", "capability", "classes", "task_types")},)
    assert outputs["plain"][1] == []
    (notice,) = outputs["retired"][1]
    assert notice == ("howhard: notice: these settings are retired and ignored: on_cap_unmet, sampling.outcome_recency, "
                      "sampling.prior_strength, sampling.min_concentration, sampling.max_concentration")
    assert outputs["retired"][0].keys() == outputs["plain"][0].keys() == {"project_root", "state_root"}
    # The kept keys keep their meaning, at the values they already had.
    assert outputs["retired"][2] == outputs["plain"][2]


# --- 7. The latency bound is a count --------------------------------------------------------------

class Counting:
    """A reranker stub that counts every pair of the new task and a run's own task text it scores.

    Prototype matching finds the runs' prototypes; the reader is shown no precedent, so the
    only task-to-run pairs are the evidence candidates'.
    """

    def __init__(self, texts):
        self.texts, self.pairs = texts, []

    @property
    def reranker(self):
        return self

    @staticmethod
    def relevance(query, document):
        return 8.0 if "rollback" in document and "rollback" in query else -18.0

    def score(self, query, documents):
        self.pairs += [(query, d) for d in documents if d in self.texts]
        return [self.relevance(query, d) for d in documents]

    def search(self, records, query, limit=8, **options):
        shown = [] if query.startswith("Task: ") else records
        ranked = sorted(((self.relevance(query, r["text"]), r) for r in shown), key=lambda item: -item[0])[:limit]
        return {"records": deepcopy([r for _, r in ranked]),
                "trace": {"seconds": 0.0, "selected": [{"key": r["key"], "score": s} for s, r in ranked]}}


def test_one_decision_on_400_candidate_runs_scores_at_most_256_task_to_run_pairs_most_recent_first(project):
    start_at = datetime.now(timezone.utc) - timedelta(days=1)
    rows = [row(i, task="Y", task_text=f"Repair rollback step {i}", seconds=100,
                observed_at=(start_at + timedelta(minutes=i)).isoformat()) for i in range(400)]
    config = project(rows=rows)
    from howhard.retrieval import records
    stub = Counting({doc["text"] for doc in records(rows=rows)})
    engine = Engine(config, retrieval=stub)
    pending = engine.prepare("Y", "Repair rollback", native_candidates=[
        {"model": "claude-opus-5-5", "effort": "high", "usd_per_mtok": 1.0}])
    result = finish(engine, pending, filtering(pending["baseline_request"], 0, 0))
    state = load(engine.config, result["decision_id"])["semantic"]["evidence_candidates"]
    assert state["matched"] == 400 and state["comparison_cap"] == 256
    assert len(stub.pairs) == len(state["candidates"]) == 256
    assert [c["id"] for c in state["candidates"]] == [f"run-{i}" for i in range(399, 143, -1)]
