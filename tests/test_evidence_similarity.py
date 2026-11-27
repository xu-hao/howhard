"""Evidence by direct similarity: one scored candidate set for the three evidence readers (#129)."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import fcntl
import json
import math
import re
import sqlite3

import pytest

from conftest import inventory, row
from howhard import Engine
from howhard.cli import main
from howhard.context import canonical, digest
from howhard.decisions import load
from howhard.payload_store import get, put
from howhard.report import report
from howhard.storage import locked, reading
from howhard.validation import HowHardError
from legacy_receipts import to_v7, to_v8, to_v9, to_v10
from test_capability_filter import filtering, forecasts

TOPICS = {"rollback", "billing"}
OPUS = [("opus", "high")]


def words(value):
    return set(re.findall(r"[a-z]+", value.casefold()))


def unlocked(config):
    """Whether another holder could take both of the store's data locks at this moment."""
    for path in (config.state_root / "storage.lock", config.path(config.data["journal"]).with_suffix(".lock")):
        with open(path, "a+b") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            fcntl.flock(handle, fcntl.LOCK_UN)
    return True


class Scripted:
    """A shared topic word matches a prototype; a run's own task text states its score.

    The new task is compared with a run's own text only through ``reranker.score`` with the
    full task text as the query (it starts with ``Task:``). Prototype matching passes the
    bare task words, and gets ``match`` for a shared topic.
    """

    def __init__(self, *, match=12.0, show=True, store=None):
        self.match, self.show, self.store = match, show, store
        self.compared, self.unlocked = [], []

    @property
    def reranker(self):
        return self

    def relevance(self, query, document):
        if not TOPICS & words(query) & words(document):
            return -18.0
        if not query.startswith("Task: "):
            return self.match
        stated = re.search(r"relevance (-?\d+(?:\.\d+)?)", document)
        return float(stated.group(1)) if stated else 3.0

    def score(self, query, documents):
        if query.startswith("Task: "):
            self.compared.append(list(documents))
            if self.store is not None:
                self.unlocked.append(unlocked(self.store))
        return [self.relevance(query, d) for d in documents]

    def search(self, records, query, limit=8, **options):
        shown = records if self.show or not query.startswith("Task: ") else []
        ranked = sorted(((self.relevance(query, r["text"]), r) for r in shown),
                        key=lambda item: (-item[0], item[1]["key"]))[:limit]
        return {"records": deepcopy([r for _, r in ranked]),
                "trace": {"seconds": 0.0, "selected": [{"key": r["key"], "score": s} for s, r in ranked]}}


def run(name, *, days=0.0, relevance=None, text="Repair rollback", **fields):
    """A synthetic recorded run of task type Y, observed ``days`` before now."""
    observed = datetime.now(timezone.utc) - timedelta(days=days)
    if relevance is not None:
        text = f"{text} relevance {relevance}"
    return row(name, task="Y", task_text=text, observed_at=observed.isoformat(), id=name, **fields)


def logit(p):
    return math.log(p / (1 - p))


def decide(engine, cells, text="Repair rollback", **options):
    """Prepare, filter over the whole inventory and forecast; return the stored receipt."""
    pending = engine.prepare("Y", text, native_candidates=inventory(cells), **options)
    band = engine.decide(pending["request_id"], filtering(pending["baseline_request"], 0, len(cells) - 1))
    result = engine.decide(band["request_id"], forecasts(band["baseline_request"]), seed=3)
    assert "decision_id" in result, result
    return load(engine.config, result["decision_id"])


def scored(receipt):
    return {c["id"]: c for c in receipt["semantic"]["evidence_candidates"]["candidates"]}


def outcome_rows(receipt):
    return {o["id"]: o for o in receipt["semantic"]["outcome_observations"]}


def duration_rows(receipt):
    return {o["id"]: o for o in receipt["semantic"]["duration_observations"]}


def capability_rows(receipt):
    return {o["id"]: o for o in receipt["capability"]["evidence"]}


def readers(receipt):
    return outcome_rows(receipt), duration_rows(receipt), capability_rows(receipt)


def counts(receipt, family="opus", effort="high"):
    cell = next(c for c in receipt["semantic"]["sampling"]["cells"] if (c["family"], c["effort"]) == (family, effort))
    return cell["outcome_posterior"]["counts"]


def test_matched_run_counts_at_similarity_times_age_and_other_runs_count_nowhere(project):
    config = project(rows=[
        run("match", days=7, relevance=1.386, seconds=100),
        run("loud", days=7, relevance=1.386, outcome="loud", seconds=50),
        run("stopped", relevance=5, outcome="interrupted", cause="cancelled"),
        run("elsewhere", relevance=5, text="Tune billing", seconds=100),
    ])
    receipt = decide(Engine(config, retrieval=Scripted()), OPUS)
    state = receipt["semantic"]["evidence_candidates"]
    assert datetime.fromisoformat(state["prepared_at"]).utcoffset() is not None
    assert (state["half_life_days"], state["comparison_cap"], state["matched"]) == (14, 256, 2)
    assert set(scored(receipt)) == {"match", "loud"}
    match = scored(receipt)["match"]
    assert match == {"id": "match", "family": "opus", "effort": "high", "outcome": "clean",
                     "active_seconds": 100.0, "tokens": None, "token_usage": None,
                     "ticket": None, "run_weight": 1.0, "score": 1.386,
                     "similarity": pytest.approx(.8, abs=1e-4), "days": pytest.approx(7, abs=1e-3),
                     "age_factor": pytest.approx(.7071, abs=1e-4)}
    outcomes, durations, capability = readers(receipt)
    for rows in (outcomes, durations, capability):
        assert set(rows) == {"match", "loud"}
    # w = m * sim * delta * a = 1 * 0.8 * 1 * 0.5 ** (7 / 14)
    assert outcomes["match"]["weight"] == pytest.approx(.5657, abs=1e-4)
    assert outcomes["match"] == {"id": "match", "family": "opus", "effort": "high", "outcome": "clean",
                                 "discount": 1.0, "ticket_share": 1.0,
                                 "weight": match["run_weight"] * match["similarity"] * match["age_factor"]}
    assert durations["match"] == {"id": "match", "family": "opus", "effort": "high", "active_seconds": 100.0,
                                  "tokens": None, "token_usage": None, "weight": outcomes["match"]["weight"]}
    assert capability["match"]["strength"] == 1.0
    assert capability["match"]["weight"] == pytest.approx(.5657, abs=1e-4)
    # A loud run with no recorded cause: 0.8 * 0.75 * 0.5 * 0.7071 in the capability reader.
    assert capability["loud"]["strength"] == pytest.approx(.375)
    assert capability["loud"]["weight"] == pytest.approx(.2121, abs=1e-4)
    assert counts(receipt) == pytest.approx([.5657, .5657, 0], abs=1e-4)


def test_no_matched_prototype_leaves_every_reader_empty(project):
    config = project(rows=[run("rollback", relevance=5, seconds=100), run("billing", relevance=5, text="Tune billing")])
    retrieval = Scripted()
    receipt = decide(Engine(config, retrieval=retrieval), OPUS, "Write a poem")
    state = receipt["semantic"]["evidence_candidates"]
    assert (state["matched"], state["candidates"]) == (0, [])
    assert readers(receipt) == ({}, {}, {})
    assert counts(receipt) == [0, 0, 0]
    assert retrieval.compared == []


def test_negative_score_is_counted_at_a_similarity_below_one_half(project):
    config = project(rows=[run("weak", relevance=-1.0, seconds=100)])
    receipt = decide(Engine(config, retrieval=Scripted()), OPUS)
    weak = scored(receipt)["weak"]
    assert weak["score"] == -1.0 and weak["similarity"] == pytest.approx(1 / (1 + math.e)) and weak["similarity"] < .5
    for rows in readers(receipt):
        assert rows["weak"]["weight"] == pytest.approx(weak["similarity"], rel=1e-4) and rows["weak"]["weight"] > 0
    assert counts(receipt)[0] == pytest.approx(weak["similarity"], rel=1e-4)


@pytest.mark.parametrize("settings,days", [("", 14), ("[sampling]\nhalf_life_days = 7\n", 7)])
def test_a_run_one_half_life_old_weighs_half_in_all_three_readers(project, settings, days):
    config = project(settings=settings, rows=[run("old", days=days, relevance=2, seconds=100),
                                              run("new", relevance=2, seconds=100)])
    receipt = decide(Engine(config, retrieval=Scripted()), OPUS)
    assert receipt["semantic"]["evidence_candidates"]["half_life_days"] == days
    assert scored(receipt)["old"]["age_factor"] == pytest.approx(.5, rel=1e-4)
    for rows in readers(receipt):
        assert rows["old"]["weight"] == pytest.approx(rows["new"]["weight"] / 2, rel=1e-4)


def test_runs_of_one_ticket_count_for_at_most_one_run_and_unticketed_runs_are_not_limited(project):
    # The worked example: 0.6, 0.5 and 0.4 of one ticket become 0.4, 0.3333 and 0.2667.
    example = [run("a", relevance=logit(.6), ticket="T-1", outcome="loud"),
               run("b", relevance=logit(.5), ticket="T-1", outcome="loud"),
               run("c", relevance=logit(.4), ticket="T-1"),
               run("d", relevance=logit(.3), ticket="T-2")]
    receipt = decide(Engine(project(rows=example), retrieval=Scripted()), OPUS)
    outcomes = outcome_rows(receipt)
    assert [outcomes[k]["weight"] for k in "abcd"] == pytest.approx([.4, 1 / 3, .8 / 3, .3], abs=1e-4)
    assert [outcomes[k]["ticket_share"] for k in "abcd"] == pytest.approx([1 / 1.5] * 3 + [1], abs=1e-4)
    assert counts(receipt) == pytest.approx([.5667, .7333, 0], abs=1e-4)

    limited = [run(f"t-{i}", relevance=logit(.9), ticket="T-9", seconds=100) for i in range(3)]
    free = [run(f"f-{i}", relevance=logit(.9), seconds=100) for i in range(3)]
    receipt = decide(Engine(project(rows=limited + free), retrieval=Scripted()), OPUS)
    for rows in readers(receipt):
        assert sum(rows[f"t-{i}"]["weight"] for i in range(3)) == pytest.approx(1, abs=1e-3)
        assert sum(rows[f"f-{i}"]["weight"] for i in range(3)) == pytest.approx(2.7, abs=1e-3)
    assert all(capability_rows(receipt)[f"t-{i}"]["ticket_share"] == pytest.approx(1 / 2.7, abs=1e-3) for i in range(3))
    assert all(capability_rows(receipt)[f"f-{i}"]["ticket_share"] == 1 for i in range(3))


def test_only_the_most_recent_comparison_cap_candidates_are_scored_outside_the_storage_lock(project):
    start = datetime.now(timezone.utc) - timedelta(days=1)
    rows = [row(i, task="Y", task_text="Repair rollback", seconds=100,
                observed_at=(start + timedelta(minutes=i)).isoformat()) for i in range(300)]
    config = project(rows=rows)
    # No precedent is shown to the reader, so every surfaced run is a scored candidate.
    retrieval = Scripted(store=config, show=False)
    engine = Engine(config, retrieval=retrieval)
    with locked(config):
        assert not unlocked(config)
    with reading(config):
        assert not unlocked(config)
    receipt = decide(engine, OPUS)
    state = receipt["semantic"]["evidence_candidates"]
    recent, dropped = {f"run-{i}" for i in range(44, 300)}, {f"run-{i}" for i in range(44)}
    assert (state["matched"], len(state["candidates"])) == (300, 256)
    assert [c["id"] for c in state["candidates"]] == [f"run-{i}" for i in range(299, 43, -1)]
    assert sum(len(documents) for documents in retrieval.compared) == 256
    assert retrieval.unlocked and all(retrieval.unlocked)
    for rows in readers(receipt):
        assert set(rows) == recent and not dropped & set(rows)
    assert counts(receipt)[0] == pytest.approx(sum(o["weight"] for o in outcome_rows(receipt).values()))
    assert set(receipt["request_snapshot"]["dependencies"]["evidence"]) == recent


def test_a_run_of_another_cell_adds_nothing_to_a_cells_outcomes_or_active_time(project):
    config = project(rows=[run("opus-run", relevance=-2, seconds=100),
                           run("sonnet-run", relevance=-2, family="sonnet", outcome="silent", seconds=900),
                           run("haiku-run", relevance=4, family="haiku", effort="low", seconds=5)])
    receipt = decide(Engine(config, retrieval=Scripted()), [("opus", "high"), ("sonnet", "high")])
    assert set(scored(receipt)) == {"opus-run", "sonnet-run", "haiku-run"}
    assert {(c["family"], c["effort"]) for c in receipt["semantic"]["sampling"]["cells"]} == {("opus", "high"), ("sonnet", "high")}
    outcomes, durations, capability = readers(receipt)
    for rows in (outcomes, durations):
        assert {(o["id"], o["family"], o["effort"]) for o in rows.values()} == {
            ("opus-run", "opus", "high"), ("sonnet-run", "sonnet", "high")}
    assert set(capability) == {"opus-run", "sonnet-run"}
    assert counts(receipt, "opus") == [outcomes["opus-run"]["weight"], 0, 0]
    assert counts(receipt, "sonnet") == [0, 0, outcomes["sonnet-run"]["weight"]]
    # The same record judged alone gives the same counts for the cell: no other cell's run entered.
    alone = decide(Engine(project(rows=[run("opus-run", relevance=-2, seconds=100)]), retrieval=Scripted()), OPUS)
    assert counts(alone, "opus") == pytest.approx(counts(receipt, "opus"), rel=1e-4)
    assert duration_rows(alone)["opus-run"]["weight"] == pytest.approx(durations["opus-run"]["weight"], rel=1e-4)


def test_prototype_scores_and_membership_weights_enter_no_weight(project):
    rows = [run("first", days=3, relevance=1.5, seconds=100, ticket="T-1"),
            run("second", days=1, relevance=.5, seconds=100, ticket="T-1", outcome="loud"),
            run("third", relevance=-.5, seconds=100)]

    def weights(receipt):
        return [rows[k]["weight"] for rows in readers(receipt) for k in ("first", "second", "third")]

    strong = Engine(project(rows=rows), retrieval=Scripted(match=12.0))
    reference = weights(decide(strong, OPUS))
    # A weaker prototype match stores a lower prototype score and lower membership weights.
    weak = Engine(project(rows=rows), retrieval=Scripted(match=.25))
    assert weights(decide(weak, OPUS)) == pytest.approx(reference, rel=1e-4)
    stored = {}
    for name, engine in (("strong", strong), ("weak", weak)):
        with sqlite3.connect(engine.config.state_root / "payloads.sqlite3") as db:
            stored[name] = sorted(w for (w,) in db.execute("SELECT weight FROM prototype_members"))
    assert stored["weak"] != stored["strong"]
    with sqlite3.connect(strong.config.state_root / "payloads.sqlite3") as db:
        db.execute("UPDATE prototype_members SET weight = 0.0001")
    assert weights(decide(strong, OPUS)) == pytest.approx(reference, rel=1e-4)
    # A run counts the same whichever host asks for the route.
    assert weights(decide(strong, OPUS, host="host-a")) == pytest.approx(reference, rel=1e-4)


def test_every_scored_candidate_is_a_request_dependency(project):
    config = project(rows=[run("precedent", relevance=2, seconds=100), run("other", relevance=2, family="sonnet")])
    # Nothing is shown to the reader, so the candidates are the only surfaced evidence.
    engine = Engine(config, retrieval=Scripted(show=False))
    pending = engine.prepare("Y", "Repair rollback", native_candidates=inventory(OPUS))
    assert pending["request"]["evidence"] == []
    engine.observe("Y", "claude-sonnet-5-5", "high", "silent", amends="other", id="correction")
    with pytest.raises(HowHardError) as caught:
        engine.decide(pending["request_id"], filtering(pending["baseline_request"], 0, 0))
    assert type(caught.value).__name__ == "StaleRequest"
    assert [(c["kind"], c["id"]) for c in caught.value.as_dict()["changes"]] == [("evidence", "other")]
    receipt = decide(engine, OPUS)
    assert set(receipt["request_snapshot"]["dependencies"]["evidence"]) == {"precedent", "other"}


def test_request_stored_without_its_scored_candidates_scores_them_when_it_is_used(project):
    config = project(rows=[run("precedent", relevance=2, seconds=100)])
    retrieval = Scripted(store=config)
    engine = Engine(config, retrieval=retrieval)
    pending = engine.prepare("Y", "Repair rollback", native_candidates=inventory(OPUS))
    bundle = json.loads(get(config, "requests", pending["request_id"]))
    del bundle["learned_state"]
    request_id = digest(bundle)
    with locked(config):
        put(config, "requests", request_id, canonical(bundle) + b"\n")
    retrieval.compared.clear(), retrieval.unlocked.clear()
    band = engine.decide(request_id, filtering(pending["baseline_request"], 0, 0))
    assert [len(documents) for documents in retrieval.compared] == [1] and retrieval.unlocked == [True]
    result = engine.decide(band["request_id"], forecasts(band["baseline_request"]), seed=3)
    assert [len(documents) for documents in retrieval.compared] == [1]
    receipt = load(config, result["decision_id"])
    assert set(scored(receipt)) == {"precedent"}
    for rows in readers(receipt):
        assert set(rows) == {"precedent"}


def test_outcome_recency_loads_with_a_notice_and_the_new_constants_are_validated(project, tmp_path, monkeypatch, capsys):
    capsys.readouterr()
    plain = project()
    assert capsys.readouterr().err == ""
    assert (plain.data["sampling"]["half_life_days"], plain.data["sampling"]["comparison_cap"]) == (14, 256)
    old = project(settings="[sampling]\noutcome_recency = 0.9\n")
    captured = capsys.readouterr()
    assert "outcome_recency" in captured.err and len(captured.err.strip().splitlines()) == 1 and captured.out == ""
    assert old.data["sampling"] == plain.data["sampling"]
    tuned = project(settings="[sampling]\nhalf_life_days = 3.5\ncomparison_cap = 2\n")
    assert (tuned.data["sampling"]["half_life_days"], tuned.data["sampling"]["comparison_cap"]) == (3.5, 2)
    for bad in ("half_life_days = 0", "half_life_days = -1", 'half_life_days = "14"', "comparison_cap = 0",
                "comparison_cap = 2.5", "comparison_cap = true"):
        with pytest.raises(HowHardError, match="sampling"):
            project(settings=f"[sampling]\n{bad}\n")
    # The notice goes to standard error; a command's standard output is unchanged.
    (tmp_path / "cli").mkdir()
    monkeypatch.chdir(tmp_path / "cli")
    path = tmp_path / "cli" / "howhard.toml"
    path.write_text('on_cap_unmet = "flag-review"\n[sampling]\noutcome_recency = 0.9\n')
    capsys.readouterr()
    assert main(["paths", "--config", str(path), "--json"]) == 0
    captured = capsys.readouterr()
    assert set(json.loads(captured.out)) == {"project_root", "state_root"} and "outcome_recency" in captured.err


def evidence_receipt(project, **settings):
    rows = [run("a", days=3, relevance=1.5, seconds=100, ticket="T-1"),
            run("b", days=2, relevance=.5, seconds=200, ticket="T-1", outcome="loud"),
            run("c", days=1, relevance=-.5, seconds=300)]
    engine = Engine(project(rows=rows, **settings), retrieval=Scripted())
    return engine, decide(engine, OPUS)


def edit_score(receipt):
    receipt["semantic"]["evidence_candidates"]["candidates"][0]["score"] += .5


def edit_similarity(receipt):
    receipt["semantic"]["evidence_candidates"]["candidates"][0]["similarity"] *= .9


def edit_age_factor(receipt):
    receipt["semantic"]["evidence_candidates"]["candidates"][0]["age_factor"] *= .9


def edit_days(receipt):
    receipt["semantic"]["evidence_candidates"]["candidates"][0]["days"] += 1


def edit_outcome_weight(receipt):
    receipt["semantic"]["outcome_observations"][0]["weight"] *= 1.01


def edit_outcome_share(receipt):
    receipt["semantic"]["outcome_observations"][-1]["ticket_share"] = 1.0


def edit_duration_weight(receipt):
    receipt["semantic"]["duration_observations"][0]["weight"] *= 1.01


def edit_capability_weight(receipt):
    receipt["capability"]["evidence"][0]["weight"] *= 1.01


def edit_capability_strength(receipt):
    loud = next(o for o in receipt["capability"]["evidence"] if o["outcome"] == "loud")
    loud["strength"] = 1.0


def edit_capability_share(receipt):
    receipt["capability"]["evidence"][-1]["ticket_share"] = .5


def drop_outcome_row(receipt):
    receipt["semantic"]["outcome_observations"].pop()


def edit_run_weight(receipt):
    receipt["semantic"]["evidence_candidates"]["candidates"][0]["run_weight"] = 2.0


def edit_half_life(receipt):
    receipt["semantic"]["evidence_candidates"]["half_life_days"] = 7


EDITS = [edit_score, edit_similarity, edit_age_factor, edit_days, edit_outcome_weight, edit_outcome_share,
         edit_duration_weight, edit_capability_weight, edit_capability_strength, edit_capability_share,
         drop_outcome_row, edit_run_weight, edit_half_life]


def test_stored_v11_decision_revalidates_exactly(project):
    engine, receipt = evidence_receipt(project)
    assert receipt["format"] == "howhard-decision-v11"
    assert set(scored(receipt)) == {"a", "b", "c"} and len(outcome_rows(receipt)) == 3
    assert load(engine.config, deepcopy(receipt)) == receipt
    assert load(engine.config, digest(receipt)) == receipt


@pytest.mark.parametrize("edit", EDITS, ids=lambda edit: edit.__name__)
def test_edited_score_age_factor_or_weight_is_refused(project, edit):
    engine, receipt = evidence_receipt(project)
    edit(receipt)
    with pytest.raises(HowHardError):
        load(engine.config, receipt)


def test_more_than_comparison_cap_candidates_are_refused(project):
    engine, receipt = evidence_receipt(project, settings="[sampling]\ncomparison_cap = 2\n")
    state = receipt["semantic"]["evidence_candidates"]
    assert (state["comparison_cap"], state["matched"]) == (2, 3)
    assert [c["id"] for c in state["candidates"]] == ["c", "b"]
    assert set(outcome_rows(receipt)) == {"c", "b"}
    state["candidates"].append({**state["candidates"][-1], "id": "a"})
    with pytest.raises(HowHardError, match="comparison_cap"):
        load(engine.config, receipt)


def test_v7_to_v10_receipts_with_counted_evidence_load_observe_replay_and_report(project):
    engine, current = evidence_receipt(project)
    stored = {"v11": digest(current)}
    for name, legacy in (("v10", to_v10(current)), ("v9", to_v9(current)), ("v8", to_v8(current)), ("v7", to_v7(current))):
        stored[name] = digest(legacy)
        with locked(engine.config):
            put(engine.config, "decisions", stored[name], canonical(legacy) + b"\n")
        loaded = load(engine.config, stored[name])
        assert loaded == legacy and loaded["format"] == f"howhard-decision-{name}"
        assert "evidence_candidates" not in loaded["semantic"]
    v10 = load(engine.config, stored["v10"])
    assert set(v10["semantic"]["outcome_observations"][0]) == {
        "id", "family", "effort", "outcome", "score", "similarity", "age", "weight"}
    assert "outcome_recency" in v10["selection"]["sampling_policy"]
    for index, name in enumerate(stored):
        engine.observe_decision(stored[name], "clean", id=f"from-{name}", seconds=10. + index)
    beliefs = engine.memory_beliefs()
    assert beliefs["unscored_runs"] == 0
    assert next(e for e in beliefs["experts"] if e["key"] == "llm")["evidence_count"] == 5
    stats = report(engine)
    assert stats["unsupported_receipts"] == [] and stats["forecast_metrics"]["llm"]["runs"] == 5

    # Each format keeps its own shape and the packaged limits its validator checks.
    crossed = deepcopy(v10)
    crossed["semantic"]["outcome_observations"] = deepcopy(current["semantic"]["outcome_observations"])
    with pytest.raises(HowHardError):
        load(engine.config, crossed)
    crossed = deepcopy(current)
    crossed["semantic"]["outcome_observations"] = deepcopy(v10["semantic"]["outcome_observations"])
    with pytest.raises(HowHardError):
        load(engine.config, crossed)
    crowded = deepcopy(v10)
    template = crowded["semantic"]["outcome_observations"][0]
    crowded["semantic"]["outcome_observations"] = [{**template, "id": f"extra-{i}", "weight": 0.0} for i in range(25)]
    with pytest.raises(HowHardError, match="outcome_observation"):
        load(engine.config, crowded)


def test_a_cell_counts_more_runs_than_the_former_per_cell_bound(project):
    rows = [run(f"r{i}", relevance=1, seconds=100) for i in range(30)]
    engine = Engine(project(rows=rows), retrieval=Scripted())
    receipt = decide(engine, OPUS)
    for rows in readers(receipt):
        assert len(rows) == 30
    assert counts(receipt)[0] == pytest.approx(30 * scored(receipt)["r0"]["similarity"], rel=1e-3)
