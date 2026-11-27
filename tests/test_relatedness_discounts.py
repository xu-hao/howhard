"""Relatedness discounts: a recorded run counts as outcome evidence for related cells too (#134)."""

from copy import deepcopy
import json

import pytest

from conftest import OWN_RUNS_ONLY as ZERO, fixture_filter, inventory
from howhard import Engine
from howhard.context import canonical, digest
from howhard.decisions import load
from howhard.payload_store import get, put
from howhard.report import report
from howhard.storage import locked
from howhard.validation import HowHardError
from legacy_receipts import to_v7, to_v8, to_v9, to_v10
from test_capability_filter import filtering, forecasts
from test_evidence_similarity import Scripted, counted_rows as judged, counts, duration_rows, logit, outcome_rows, run, scored

A, B = "agent-a", "agent-b"
DISCOUNTS = ("effort_discount", "model_discount", "agent_discount")


def labelled(*cells):
    """Synthetic candidates ``(family, effort, agent)``; an agent of ``None`` leaves the cell unlabeled."""
    return [{**inventory([(family, effort)])[0], **({} if agent is None else {"agent": agent})}
            for family, effort, agent in cells]


def decide(engine, candidates, text="Repair rollback"):
    """Prepare, keep the whole inventory in the band and forecast; return the stored receipt."""
    pending = engine.prepare("Y", text, native_candidates=candidates)
    request = pending["baseline_request"]
    band = engine.decide(pending["request_id"], filtering(None) if request is None else fixture_filter(request))
    result = engine.decide(band["request_id"], forecasts(band["baseline_request"]), seed=3)
    assert "decision_id" in result, result
    return load(engine.config, result["decision_id"])


def recorded(engine, name, family, effort, agent, *, sim, days=0.0, outcome="clean", decision=True, **fields):
    """Record one synthetic run of a cell whose task reads ``sim`` like the new one.

    With ``decision`` the run names a stored decision whose only candidate, and so its chosen
    cell, carries ``agent``. A run observed now goes through ``observe_decision``; an older
    one, or one with no decision, is written to the test's own log.
    """
    text = f"Repair rollback relevance {logit(sim)}"
    decision_id = digest(decide(engine, labelled((family, effort, agent)), text)) if decision else None
    if decision and not days:
        return engine.observe_decision(decision_id, outcome, id=name, **fields)
    row = run(name, days=days, text=text, family=family, effort=effort, outcome=outcome, **fields)
    if decision:
        row["decision_id"] = decision_id
    with engine.config.path(engine.config.data["logs"][0]["path"]).open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    return row


def example(project, settings=""):
    """The worked example: four runs judged for the cell ``opus`` at ``high`` under agent ``A``."""
    engine = Engine(project(settings=settings), retrieval=Scripted())
    recorded(engine, "run-1", "opus", "high", A, sim=.8, ticket="t1")
    recorded(engine, "run-2", "opus", "low", A, sim=.9, ticket="t1", outcome="loud")
    recorded(engine, "run-3", "sonnet", "high", A, sim=.8, days=14, ticket="t2")
    recorded(engine, "run-4", "haiku", "high", B, sim=.4, ticket="t3", outcome="silent")
    return engine, decide(engine, labelled(("opus", "high", A)))


def test_the_worked_example_counts_related_runs_at_their_discounts_after_the_ticket_limit(project):
    engine, receipt = example(project)
    assert [(c["family"], c["effort"]) for c in receipt["selection"]["cells"]] == [("opus", "high")]
    assert {k: receipt["selection"]["sampling_policy"][k] for k in DISCOUNTS} == {
        "effort_discount": .5, "model_discount": .25, "agent_discount": .25}
    assert receipt["semantic"]["sampling"]["policy"] == receipt["selection"]["sampling_policy"]

    # Every candidate run is stored once with its own cell, agent, ticket and weight before the discount.
    observed = outcome_rows(receipt)
    assert len(receipt["semantic"]["outcome_observations"]) == 4
    assert observed["run-1"] == {"id": "run-1", "family": "opus", "effort": "high", "agent": A, "outcome": "clean",
                                 "ticket": "t1", "weight": pytest.approx(.8, abs=1e-3)}
    assert {k: (o["family"], o["effort"], o["agent"], o["outcome"], o["ticket"]) for k, o in observed.items()} == {
        "run-1": ("opus", "high", A, "clean", "t1"), "run-2": ("opus", "low", A, "loud", "t1"),
        "run-3": ("sonnet", "high", A, "clean", "t2"), "run-4": ("haiku", "high", B, "silent", "t3")}
    assert [observed[f"run-{i}"]["weight"] for i in (1, 2, 3, 4)] == pytest.approx([.8, .9, .4, .4], abs=1e-3)
    assert {k: c["agent"] for k, c in scored(receipt).items()} == {"run-1": A, "run-2": A, "run-3": A, "run-4": B}

    # The cell stores the three sums and the number of runs behind them, never one row per run.
    cell = receipt["selection"]["cells"][0]
    assert cell["outcome_counts"] == {"n_clean": pytest.approx(.74, abs=1e-3), "n_loud": pytest.approx(.36, abs=1e-3),
                                      "n_silent": pytest.approx(.10, abs=1e-3), "runs": 4}
    assert "counted_observations" not in cell and "counted_observations" not in receipt["semantic"]
    # Each run's relation, discount and weight after the limit are computed from the receipt on demand.
    counted = judged(receipt)
    assert all(set(item) == {"id", "relation", "discount", "weight"} for item in counted.values())
    assert [(counted[f"run-{i}"]["relation"], counted[f"run-{i}"]["discount"]) for i in (1, 2, 3, 4)] == [
        ("cell", 1.), ("family", .5), ("agent", .25), ("other", .25)]
    # Ticket t1 carries 0.80 + 0.45 = 1.25, so both of its runs are divided by 1.25.
    assert [counted[f"run-{i}"]["weight"] for i in (1, 2, 3, 4)] == pytest.approx([.64, .36, .10, .10], abs=1e-3)
    assert counts(receipt) == pytest.approx([.74, .36, .10], abs=1e-3)
    assert load(engine.config, deepcopy(receipt)) == receipt and load(engine.config, digest(receipt)) == receipt


def test_with_the_discounts_at_zero_the_example_counts_only_the_cells_own_run(project):
    _, receipt = example(project, ZERO)
    assert set(outcome_rows(receipt)) == {"run-1", "run-2", "run-3", "run-4"}
    assert judged(receipt) == {"run-1": {"id": "run-1", "relation": "cell", "discount": 1.,
                                         "weight": pytest.approx(.8, abs=1e-3)}}
    assert counts(receipt) == pytest.approx([.8, 0, 0], abs=1e-3)


def test_a_run_by_a_cell_of_another_agent_weighing_four_fifths_counts_as_one_fifth(project):
    engine = Engine(project(), retrieval=Scripted())
    recorded(engine, "elsewhere", "haiku", "high", B, sim=.8)
    receipt = decide(engine, labelled(("opus", "high", A)))
    assert outcome_rows(receipt)["elsewhere"]["weight"] == pytest.approx(.8, abs=1e-3)
    assert judged(receipt)["elsewhere"] == {"id": "elsewhere", "relation": "other", "discount": .25,
                                            "weight": pytest.approx(.2, abs=1e-3)}
    assert counts(receipt) == pytest.approx([.2, 0, 0], abs=1e-3)


def test_with_the_three_discounts_at_zero_every_count_is_the_cells_own_runs_limited_per_ticket(project):
    own = [run("own-clean", relevance=logit(.8), ticket="t1"),
           run("own-loud", relevance=logit(.5), ticket="t1", outcome="loud"),
           run("own-free", relevance=logit(.3), outcome="silent")]
    related = [run("effort", relevance=logit(.9), effort="low", ticket="t1"),
               run("family", relevance=logit(.9), family="sonnet", ticket="t1", outcome="silent"),
               run("far", relevance=logit(.9), family="haiku", ticket="t2", outcome="loud")]
    cells = labelled(("opus", "high", None), ("sonnet", "high", None))
    receipt = decide(Engine(project(settings=ZERO, rows=own + related), retrieval=Scripted()), cells)
    assert {(c["family"], c["effort"]) for c in receipt["selection"]["cells"]} == {("opus", "high"), ("sonnet", "high")}
    # Every candidate is still stored once, whatever cell ran it.
    assert set(outcome_rows(receipt)) == {"own-clean", "own-loud", "own-free", "effort", "family", "far"}
    assert {k: v["relation"] for k, v in judged(receipt).items()} == {
        "own-clean": "cell", "own-loud": "cell", "own-free": "cell"}
    assert set(judged(receipt, "sonnet")) == {"family"}
    # The count before relatedness: ticket t1 carries 0.8 + 0.5 = 1.3 of the cell's own runs.
    assert counts(receipt) == pytest.approx([.8 / 1.3, .5 / 1.3, .3], abs=1e-3)
    assert counts(receipt, "sonnet") == pytest.approx([0, 0, .9], abs=1e-3)
    # The same counts as a record that holds the cell's own runs and nothing else, at the defaults.
    alone = decide(Engine(project(rows=own), retrieval=Scripted()), labelled(("opus", "high", None)))
    assert counts(alone) == pytest.approx(counts(receipt), rel=1e-4)
    # With the defaults the related runs do enter.
    default = decide(Engine(project(rows=own + related), retrieval=Scripted()), cells)
    assert set(judged(default)) == set(outcome_rows(default)) and sum(counts(default)) > sum(counts(receipt)) + .1


def test_a_run_with_no_decision_or_no_label_belongs_to_the_unnamed_agent(project):
    # Distinct discounts tell "another family of the same agent" from "another agent".
    config = project(settings="[sampling]\nmodel_discount = 0.3\nagent_discount = 0.1\n")
    engine = Engine(config, retrieval=Scripted())
    recorded(engine, "bare-effort", "opus", "low", None, sim=.8, decision=False)
    recorded(engine, "bare-family", "sonnet", "high", None, sim=.8, decision=False)
    recorded(engine, "unlabeled", "haiku", "high", None, sim=.8)
    recorded(engine, "named", "fable", "high", A, sim=.8)
    # A receipt older than v11 stores no label, whatever its candidates were called.
    text = f"Repair rollback relevance {logit(.8)}"
    older = to_v10(decide(engine, labelled(("sonnet", "low", A)), text))
    with locked(config):
        put(config, "decisions", digest(older), canonical(older) + b"\n")
    engine.observe_decision(digest(older), "clean", id="older")

    for label, relations in ((A, {"bare-effort": "family", "bare-family": "other", "unlabeled": "other",
                                  "named": "agent", "older": "other"}),
                             (None, {"bare-effort": "family", "bare-family": "agent", "unlabeled": "agent",
                                     "named": "other", "older": "agent"})):
        receipt = decide(engine, labelled(("opus", "high", label)))
        assert {k: o["agent"] for k, o in outcome_rows(receipt).items()} == {
            "bare-effort": None, "bare-family": None, "unlabeled": None, "named": A, "older": None}
        counted = judged(receipt)
        assert {k: v["relation"] for k, v in counted.items()} == relations
        # At the same family an unnamed run still gets effort_discount.
        assert counted["bare-effort"]["discount"] == .5
        assert counted["bare-effort"]["weight"] == pytest.approx(.4, abs=1e-3)
        expected = {"family": .5, "agent": .3, "other": .1}
        for name, relation in relations.items():
            assert counted[name]["discount"] == expected[relation]
            assert counted[name]["weight"] == pytest.approx(.8 * expected[relation], abs=1e-3)
        assert counts(receipt)[0] == pytest.approx(sum(.8 * expected[r] for r in relations.values()), abs=1e-3)


def test_a_run_is_stored_once_and_limited_per_judged_cell(project):
    engine = Engine(project(), retrieval=Scripted())
    recorded(engine, "by-a", "opus", "high", A, sim=.9, ticket="t1")
    recorded(engine, "by-b", "sonnet", "high", B, sim=.9, ticket="t1", outcome="silent")
    receipt = decide(engine, labelled(("opus", "high", A), ("sonnet", "high", B)))
    assert [o["id"] for o in receipt["semantic"]["outcome_observations"]].count("by-a") == 1
    assert len(receipt["semantic"]["outcome_observations"]) == 2
    # For each judged cell the ticket carries 0.9 + 0.225 = 1.125, its own run first.
    for family, own, other in (("opus", "by-a", "by-b"), ("sonnet", "by-b", "by-a")):
        counted = judged(receipt, family)
        assert (counted[own]["relation"], counted[other]["relation"]) == ("cell", "other")
        assert (counted[own]["weight"], counted[other]["weight"]) == pytest.approx((.8, .2), abs=1e-3)
    assert counts(receipt, "opus") == pytest.approx([.8, 0, .2], abs=1e-3)
    assert counts(receipt, "sonnet") == pytest.approx([.2, 0, .8], abs=1e-3)


def test_runs_of_other_cells_leave_a_cells_active_time_and_token_evidence_unchanged(project):
    own = [run("own-1", relevance=logit(.9), seconds=100, tokens=1000, ticket="t1"),
           run("own-2", relevance=logit(.6), seconds=400, tokens=4000, ticket="t2", outcome="loud")]
    others = [run("effort", relevance=logit(.9), effort="low", seconds=9000, tokens=90000, ticket="t1"),
              run("family", relevance=logit(.9), family="sonnet", seconds=9000, tokens=90000, ticket="t1"),
              run("far", relevance=logit(.9), family="haiku", seconds=9000, tokens=90000, ticket="t2", outcome="silent")]
    cell = labelled(("opus", "high", None))
    alone = decide(Engine(project(rows=own), retrieval=Scripted()), cell)
    mixed = decide(Engine(project(rows=own + others), retrieval=Scripted()), cell)
    assert set(duration_rows(mixed)) == set(duration_rows(alone)) == {"own-1", "own-2"}
    for name in ("own-1", "own-2"):
        before, after = duration_rows(alone)[name], duration_rows(mixed)[name]
        assert after["weight"] == pytest.approx(before["weight"], rel=1e-4)
        assert {k: v for k, v in after.items() if k != "weight"} == {k: v for k, v in before.items() if k != "weight"}
    assert duration_rows(mixed)["own-1"]["weight"] == pytest.approx(.9, abs=1e-3)

    def beliefs(receipt):
        row = receipt["selection"]["cells"][0]
        values = {("active_time", k): v for k, v in row["active_time"].items()}
        values.update({(kind, k): v for kind, belief in row["token_beliefs"].items() for k, v in belief.items()})
        return {k: v for k, v in values.items() if v is not None}, row["T"]

    assert beliefs(mixed)[0] == pytest.approx(beliefs(alone)[0], rel=1e-4)
    assert beliefs(mixed)[1] == pytest.approx(beliefs(alone)[1], rel=1e-4)
    assert beliefs(mixed)[0]["active_time", "evidence_weight"] == pytest.approx(1.5, abs=1e-3)
    # The same record does move the outcome counts, and the related runs of ticket t1 share its limit there.
    assert set(judged(mixed)) == {"own-1", "own-2", "effort", "family", "far"}
    assert judged(mixed)["own-1"]["weight"] == pytest.approx(.9 / 1.575, abs=1e-3)
    assert counts(mixed) != pytest.approx(counts(alone), rel=1e-2)


@pytest.mark.parametrize("key", DISCOUNTS)
@pytest.mark.parametrize("value", ["1.5", "-0.1", '"0.5"', "true"])
def test_a_discount_outside_zero_to_one_is_refused_with_a_message_that_names_the_key(project, key, value):
    with pytest.raises(HowHardError, match=f"sampling.{key}"):
        project(settings=f"[sampling]\n{key} = {value}\n")


def test_discounts_default_without_configuration_and_accept_both_ends(project):
    plain = project()
    assert {k: plain.data["sampling"][k] for k in DISCOUNTS} == {
        "effort_discount": .5, "model_discount": .25, "agent_discount": .25}
    ends = project(settings="[sampling]\neffort_discount = 1\nmodel_discount = 0\nagent_discount = 0.4\n")
    assert [ends.data["sampling"][k] for k in DISCOUNTS] == [1, 0, .4]


def sums(receipt):
    return receipt["selection"]["cells"][0]["outcome_counts"]


def edit_clean_sum(receipt):
    sums(receipt)["n_clean"] += .01


def edit_loud_sum(receipt):
    sums(receipt)["n_loud"] *= 1.01


def edit_silent_sum(receipt):
    sums(receipt)["n_silent"] = 0.


def edit_run_count(receipt):
    sums(receipt)["runs"] += 1


def drop_sums(receipt):
    del receipt["selection"]["cells"][0]["outcome_counts"]


def store_the_rows(receipt):
    receipt["selection"]["cells"][0]["counted_observations"] = list(judged(receipt).values())


def both(receipt, name):
    """A candidate run as both of its stored rows: the scored candidate and the outcome observation."""
    return [next(o for o in rows if o["id"] == name)
            for rows in (receipt["semantic"]["evidence_candidates"]["candidates"], receipt["semantic"]["outcome_observations"])]


def edit_candidate_agent(receipt):
    # Run 4 relabeled as the judged cell's own agent: another family of the same agent, at 0.25.
    for row in both(receipt, "run-4"):
        row["agent"] = A


def edit_candidate_effort(receipt):
    # Run 2 moved to the judged cell's own effort: it would count in full.
    for row in both(receipt, "run-2"):
        row["effort"] = "high"


def edit_candidate_ticket(receipt):
    # Run 2 moved to a ticket of its own: neither run of t1 would be limited.
    for row in both(receipt, "run-2"):
        row["ticket"] = "t9"


def edit_candidate_score(receipt):
    scored(receipt)["run-3"]["score"] += 1.


def edit_loud_count(receipt):
    receipt["semantic"]["sampling"]["cells"][0]["outcome_beliefs"]["loud"]["counts"][0] += .01


def edit_silent_count(receipt):
    receipt["semantic"]["sampling"]["cells"][0]["outcome_beliefs"]["silent"]["counts"][1] += .01


def edit_run_agent(receipt):
    next(o for o in receipt["semantic"]["outcome_observations"] if o["agent"] == B)["agent"] = A


def edit_run_family(receipt):
    next(o for o in receipt["semantic"]["outcome_observations"] if o["family"] == "sonnet")["family"] = "opus"


def edit_run_effort(receipt):
    next(o for o in receipt["semantic"]["outcome_observations"] if o["effort"] == "low")["effort"] = "high"


def edit_run_ticket(receipt):
    receipt["semantic"]["outcome_observations"][0]["ticket"] = "t9"


def edit_run_weight(receipt):
    receipt["semantic"]["outcome_observations"][0]["weight"] *= 1.01


def drop_run(receipt):
    receipt["semantic"]["outcome_observations"].pop()


def edit_cell_agent(receipt):
    receipt["native_candidates"][0]["agent"] = B


def edit_policy_discount(receipt):
    for policy in (receipt["selection"]["sampling_policy"], receipt["semantic"]["sampling"]["policy"]):
        policy["agent_discount"] = .5


def edit_policy_range(receipt):
    for policy in (receipt["selection"]["sampling_policy"], receipt["semantic"]["sampling"]["policy"]):
        policy["effort_discount"] = 1.5


def drop_policy_discount(receipt):
    for policy in (receipt["selection"]["sampling_policy"], receipt["semantic"]["sampling"]["policy"]):
        del policy["model_discount"]


EDITS = [edit_clean_sum, edit_loud_sum, edit_silent_sum, edit_run_count, drop_sums, store_the_rows,
         edit_candidate_agent, edit_candidate_effort, edit_candidate_ticket, edit_candidate_score,
         edit_loud_count, edit_silent_count, edit_run_agent, edit_run_family, edit_run_effort,
         edit_run_ticket, edit_run_weight, drop_run, edit_cell_agent, edit_policy_discount, edit_policy_range,
         drop_policy_discount]


def test_a_receipt_with_an_altered_candidate_label_discount_or_sum_is_refused(project):
    # Another agent counts at 0.1 here, so that a changed agent label changes a sum.
    engine, receipt = example(project, "[sampling]\nagent_discount = 0.1\n")
    assert sums(receipt) == {"n_clean": pytest.approx(.74, abs=1e-3), "n_loud": pytest.approx(.36, abs=1e-3),
                             "n_silent": pytest.approx(.04, abs=1e-3), "runs": 4}
    for edit in EDITS:
        altered = deepcopy(receipt)
        edit(altered)
        assert altered != receipt, edit.__name__
        try:
            load(engine.config, altered)
        except HowHardError:
            continue
        pytest.fail(f"{edit.__name__}: the altered receipt loaded")
    assert load(engine.config, deepcopy(receipt)) == receipt


def test_v7_to_v10_receipts_load_observe_replay_and_report_beside_related_runs(project):
    engine, current = example(project)
    assert len(judged(current)) == 4
    stored = {"v11": digest(current)}
    for name, legacy in (("v10", to_v10(current)), ("v9", to_v9(current)), ("v8", to_v8(current)), ("v7", to_v7(current))):
        stored[name] = digest(legacy)
        with locked(engine.config):
            put(engine.config, "decisions", stored[name], canonical(legacy) + b"\n")
        loaded = load(engine.config, stored[name])
        assert loaded == legacy and loaded["format"] == f"howhard-decision-{name}"
        assert all("outcome_counts" not in cell for cell in loaded["selection"]["cells"])
        assert not set(DISCOUNTS) & set(loaded["selection"]["sampling_policy"])
    # An older receipt keeps its own shape: only the judged cell's own run, at its weight after the limit.
    v10 = load(engine.config, stored["v10"])
    assert [(o["id"], o["family"], o["effort"]) for o in v10["semantic"]["outcome_observations"]] == [
        ("run-1", "opus", "high")]
    assert set(v10["semantic"]["outcome_observations"][0]) == {
        "id", "family", "effort", "outcome", "score", "similarity", "age", "weight"}
    assert v10["semantic"]["outcome_observations"][0]["weight"] == pytest.approx(.64, abs=1e-3)
    posterior = v10["semantic"]["sampling"]["cells"][0]["outcome_posterior"]
    assert posterior["counts"] == pytest.approx([.64, 0, 0], abs=1e-3)
    before = len(engine.rows)
    for index, name in enumerate(stored):
        engine.observe_decision(stored[name], "clean", id=f"from-{name}", seconds=10. + index)
    assert len(engine.rows) == before + 5
    beliefs = engine.memory_beliefs()
    assert beliefs["unscored_runs"] == 0
    stats = report(engine)
    assert stats["unsupported_receipts"] == []
    assert stats["forecast_metrics"]["llm"]["runs"] == next(
        e for e in beliefs["experts"] if e["key"] == "llm")["evidence_count"]
    # Runs recorded under the older receipts are unnamed; the one under the v11 receipt keeps its label.
    later = decide(engine, labelled(("opus", "high", A)))
    agents = {k: o["agent"] for k, o in outcome_rows(later).items() if k.startswith("from-")}
    assert agents == {"from-v11": A, "from-v10": None, "from-v9": None, "from-v8": None, "from-v7": None}
    # A v10 receipt cannot carry the v11 shape, nor a v11 receipt the v10 one.
    crossed = deepcopy(v10)
    crossed["semantic"]["outcome_observations"] = deepcopy(current["semantic"]["outcome_observations"])
    with pytest.raises(HowHardError):
        load(engine.config, crossed)
    crossed = deepcopy(current)
    crossed["semantic"]["outcome_observations"] = deepcopy(v10["semantic"]["outcome_observations"])
    with pytest.raises(HowHardError):
        load(engine.config, crossed)


def test_a_request_scored_before_candidates_carried_an_agent_reads_its_runs_as_unnamed(project):
    engine = Engine(project(), retrieval=Scripted())
    recorded(engine, "earlier", "haiku", "high", A, sim=.8)
    pending = engine.prepare("Y", "Repair rollback", native_candidates=labelled(("opus", "high", A)))
    bundle = json.loads(get(engine.config, "requests", pending["request_id"]))
    assert [c["agent"] for c in bundle["learned_state"]["evidence_candidates"]["candidates"]] == [A]
    for item in bundle["learned_state"]["evidence_candidates"]["candidates"]:
        del item["agent"]
    request_id = digest(bundle)
    with locked(engine.config):
        put(engine.config, "requests", request_id, canonical(bundle) + b"\n")
    request = pending["baseline_request"]
    band = engine.decide(request_id, filtering(None) if request is None else fixture_filter(request))
    result = engine.decide(band["request_id"], forecasts(band["baseline_request"]), seed=3)
    receipt = load(engine.config, result["decision_id"])
    assert scored(receipt)["earlier"]["agent"] is None and outcome_rows(receipt)["earlier"]["agent"] is None
    # Unnamed, the run is no longer a run of the judged cell's own agent.
    assert judged(receipt)["earlier"]["relation"] == "other"
    assert judged(decide(engine, labelled(("opus", "high", A))))["earlier"]["relation"] == "agent"
