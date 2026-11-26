"""Per-cell Dirichlet outcome posteriors, credible intervals and overlap-gated Thompson sampling (#103)."""

from copy import deepcopy

import numpy as np
import pytest

from howhard import Engine
from howhard.context import canonical, digest
from howhard.decisions import load
from howhard.payload_store import put
from howhard.report import report
from howhard.storage import locked
from howhard.validation import HowHardError
from legacy_receipts import to_v7, to_v8, to_v9, to_v10
from test_capability_filter import SimilarityDouble, filtering

FIELDS = ("P_clean", "P_loud", "P_silent", "P_silent_accepted")


class RecentDouble:
    """Absolute-relevance fixture that retrieves the most recent executions first."""
    def search(self, records, query, limit=8, **kwargs):
        evidence = [r for r in records if r["kind"] == "evidence"]
        # Input order is journal chronology; string IDs would put run-10 before run-2.
        chosen = list(reversed(evidence))[:limit]
        return {"records": deepcopy(chosen), "trace": {"seconds": 0, "selected": [
            {"key": r["key"], "score": 12} for r in chosen]}}


def spec(clean, loud, silent, **extra):
    return {"clean": clean, "loud": loud, "silent": silent, "tokens": 1000, "seconds": 60, **extra}


def decide(engine, candidates, specs, *, seed=1, memory_forecasts=(), task_text="Repair rollback"):
    pending = engine.prepare("task", task_text, native_candidates=candidates)
    if pending["baseline_request"] is None:
        band = engine.decide(pending["request_id"], {"memory_filters": [], "reason": "Materialized prior"})
    else:
        keys_by_pair = {(c["model"], c["effort"]): c["key"] for c in pending["baseline_request"]["cells"]}
        order = [keys_by_pair[c["model"], c["effort"]] for c in candidates]
        band = engine.decide(pending["request_id"], filtering(pending["baseline_request"], 0, len(candidates) - 1, order))
    all_by_cell = {(c["family"], c["effort"]): c["model"] for c in band["request"]["cells"]}
    memory = [{"id": m["id"], "version": m["version"],
               "forecasts": [{"family": f, "effort": e, **s} for (f, e), model in all_by_cell.items()
                             for name, s in m["specs"].items() if name == model]} for m in memory_forecasts]
    if band["baseline_request"] is None:
        return engine.decide(band["request_id"], {"memory_forecasts": memory,
                                                  "reason": "Synthetic posterior fixture"}, seed=seed)
    by_cell = {(c["family"], c["effort"]): c["model"] for c in band["baseline_request"]["cells"]}
    rows = [{"family": f, "effort": e, **specs[m]} for (f, e), m in by_cell.items()]
    return engine.decide(band["request_id"], {"forecasts": rows, "memory_forecasts": memory,
                                              "reason": "Synthetic posterior fixture"}, seed=seed)


def by_model(result):
    return {c["model"]: c for c in result["cells"]}


def calibration(result, model):
    row = by_model(result)[model]
    return next(c for c in result["semantic"]["sampling"]["cells"] if (c["family"], c["effort"]) == (row["family"], row["effort"]))


def observe(engine, model, outcomes, prefix):
    for i, outcome in enumerate(outcomes):
        extra = {} if outcome == "clean" else {"cause": "capability"}
        engine.observe("task", model, "high", outcome, native=True, task_text="Repair rollback", id=f"{prefix}-{i}", **extra)


WORKER = [{"model": "fixture/worker", "effort": "high", "usd_per_mtok": 1.0}]


def test_llm_only_forecasts_report_real_nondegenerate_beta_credible_intervals(tmp_path):
    engine = Engine(directory=tmp_path)
    result = decide(engine, WORKER, {"fixture/worker": spec(.9, .07, .03)}, seed=3)
    assert set(result["semantic"]["worlds"]) == {"llm"}
    assert result["semantic"]["interval_kind"] == "dirichlet-posterior-mixture"
    row = result["cells"][0]
    for field in FIELDS:
        assert row[field]["lower"] < row[field]["mean"] < row[field]["upper"], field
    assert row["P_silent"]["mean"] == pytest.approx(.03)
    assert row["P_silent_accepted"]["mean"] == pytest.approx(.03 / .93)
    # With no observations the posterior is the bounded prior Dirichlet(8 * p0): its marginals are Beta.
    draws = np.random.default_rng(0).dirichlet([8 * .9, 8 * .07, 8 * .03], 400000)
    for i, field in enumerate(("P_clean", "P_loud", "P_silent")):
        assert row[field]["upper"] == pytest.approx(np.quantile(draws[:, i], .975), rel=.02)
    accepted = draws[:, 2] / (draws[:, 0] + draws[:, 2])
    assert row["P_silent_accepted"]["upper"] == pytest.approx(np.quantile(accepted, .975), rel=.02)
    assert row["admission_risk"] == pytest.approx(np.quantile(accepted, .8), rel=.02)
    posterior = calibration(result, "fixture/worker")["outcome_posterior"]
    assert posterior["prior_concentration"] == 8. and posterior["evidence_weight"] == 0.
    receipt = load(engine.config, result["decision_id"])
    assert receipt["format"] == "howhard-decision-v11"
    assert receipt["semantic"]["outcome_observations"] == []


def test_unseen_cell_keeps_prior_variance_despite_global_evidence_while_observed_cell_updates(tmp_path):
    candidates = [{"model": "fixture/unseen", "effort": "high", "usd_per_mtok": 1.0},
                  {"model": "fixture/seen", "effort": "high", "usd_per_mtok": 1.0}]
    specs = {"fixture/seen": spec(.84, .12, .04), "fixture/unseen": spec(.84, .12, .04)}
    fresh = decide(Engine(directory=tmp_path / "fresh", retrieval=SimilarityDouble()), candidates, specs)

    engine = Engine(directory=tmp_path / "history", retrieval=SimilarityDouble())
    observe(engine, "fixture/seen", ["clean"] * 6, "seen")
    # Many scored runs of an unrelated cell raise the llm expert's global evidence count
    # without crowding out fixture/seen's per-cell outcome history.
    other = decide(engine, [{"model": "fixture/other", "effort": "high", "usd_per_mtok": 1.0}],
                   {"fixture/other": spec(.84, .12, .04)})
    for i in range(30):
        engine.observe_decision(other["decision_id"], "clean", id=f"other-{i}")
    assert next(e for e in engine.memory_beliefs()["experts"] if e["key"] == "llm")["evidence_count"] >= 30
    result = decide(engine, candidates, specs)
    observed = result["semantic"]["outcome_observations"]
    assert {o["id"] for o in observed if o["family"] == by_model(result)["fixture/seen"]["family"]} == {f"seen-{i}" for i in range(6)}
    assert not any(o["family"] == by_model(result)["fixture/unseen"]["family"] for o in observed)

    unseen, unseen_fresh = calibration(result, "fixture/unseen"), calibration(fresh, "fixture/unseen")
    assert unseen["outcome_posterior"]["evidence_weight"] == 0.
    assert unseen["concentration"] == unseen_fresh["concentration"] == 8.
    assert unseen["cost_sd"] == pytest.approx(unseen_fresh["cost_sd"])
    assert by_model(result)["fixture/unseen"]["P_clean"] == pytest.approx(by_model(fresh)["fixture/unseen"]["P_clean"])

    seen = calibration(result, "fixture/seen")
    weights = sorted(o["weight"] for o in result["semantic"]["outcome_observations"] if o["id"].startswith("seen-"))
    assert seen["outcome_posterior"]["counts"][0] == pytest.approx(sum(weights)) and sum(weights) > 4
    assert seen["outcome_posterior"]["prior_concentration"] == pytest.approx(1.0)
    assert seen["concentration"] == pytest.approx(1.0 + sum(weights))
    seen_row, unseen_row = by_model(result)["fixture/seen"], by_model(result)["fixture/unseen"]
    assert seen_row["P_clean"]["mean"] > .88 > unseen_row["P_clean"]["mean"]
    width = lambda row, field: row[field]["upper"] - row[field]["lower"]
    assert width(seen_row, "P_silent") < width(unseen_row, "P_silent")
    assert seen["cost_sd"] < unseen["cost_sd"]
    load(engine.config, result["decision_id"])


def test_optional_credible_intervals_validate_and_only_widen_a_bounded_prior(tmp_path):
    bare_eng = Engine(directory=tmp_path / "bare")
    wide_eng = Engine(directory=tmp_path / "wide")
    narrow_eng = Engine(directory=tmp_path / "narrow")
    bare = decide(bare_eng, WORKER, {"fixture/worker": spec(.9, .07, .03)})
    wide = decide(wide_eng, WORKER, {"fixture/worker": spec(.9, .07, .03, silent_ci=[0., .3])})
    narrow = decide(narrow_eng, WORKER, {"fixture/worker": spec(.9, .07, .03, silent_ci=[.0299, .0301], loud_ci=[.0699, .0701])})

    kappa = .03 * .97 / (.3 / 3.92) ** 2 - 1
    assert calibration(wide, "fixture/worker")["outcome_posterior"]["prior_concentration"] == pytest.approx(kappa)
    assert 2. < kappa < 8.
    assert wide["cells"][0]["P_silent"]["upper"] > bare["cells"][0]["P_silent"]["upper"]
    assert wide["cells"][0]["P_silent"]["mean"] == pytest.approx(bare["cells"][0]["P_silent"]["mean"])
    # An overconfident narrow interval cannot mimic more evidence than prior_strength.
    assert calibration(narrow, "fixture/worker")["outcome_posterior"]["prior_concentration"] == 8.
    for field in FIELDS:
        assert narrow["cells"][0][field] == bare["cells"][0][field]
    assert load(wide_eng.config, wide["decision_id"])["semantic"]["worlds"]["llm"][0]["silent_ci"] == [0., .3]

    # A very wide interval stops at min_concentration.
    widest = decide(Engine(directory=tmp_path / "widest"), WORKER, {"fixture/worker": spec(.9, .07, .03, silent_ci=[0., 1.])})
    assert calibration(widest, "fixture/worker")["outcome_posterior"]["prior_concentration"] == 2.

    for idx, bad in enumerate(([.05, .2], [0., 1.2], [.1], "0-0.2", [.2, .01], [0., None])):
        with pytest.raises(HowHardError, match="silent_ci"):
            decide(Engine(directory=tmp_path / f"bad-{idx}"), WORKER, {"fixture/worker": spec(.9, .07, .03, silent_ci=bad)})
    with pytest.raises(HowHardError, match="loud_ci"):
        decide(Engine(directory=tmp_path / "bad-loud"), WORKER, {"fixture/worker": spec(.9, .07, .03, loud_ci=[.08, .1])})

    # Memory experts may supply intervals too; mixtures still report one valid interval.
    mixed_eng = Engine(directory=tmp_path / "mixed")
    memory = mixed_eng.memory.add("Uncertain", "Silent defects are poorly understood for this task")
    mixed = decide(mixed_eng, WORKER, {"fixture/worker": spec(.9, .07, .03)}, memory_forecasts=[
        {**memory, "specs": {"fixture/worker": spec(.85, .07, .08, silent_ci=[.0, .4])}}])
    assert set(mixed["semantic"]["worlds"]) == {"llm", f"memory:{memory['id']}:{memory['version']}"}
    row = mixed["cells"][0]
    assert row["P_silent"]["lower"] < row["P_silent"]["mean"] < row["P_silent"]["upper"]
    load(mixed_eng.config, mixed["decision_id"])


def test_q80_admission_and_posterior_overlap_break_lock_in_but_reject_high_silent_traps(tmp_path):
    candidates = [{"model": "fixture/favored", "effort": "high", "usd_per_mtok": 1.0},
                  {"model": "fixture/native", "effort": "high", "usd_per_mtok": 1.0},
                  {"model": "fixture/trap", "effort": "high", "usd_per_mtok": .01}]
    specs = {"fixture/favored": spec(.94, .05, .01), "fixture/native": spec(.93, .05, .02),
             "fixture/trap": spec(.6, .1, .3)}
    config = tmp_path / "howhard.toml"
    config.write_text('on_cap_unmet = "flag-review"\n[task_types]\ntask = "standard"\n')
    engine = Engine(config)
    first = decide(engine, candidates, specs, seed=0)
    native, trap, favored = (by_model(first)[m] for m in ("fixture/native", "fixture/trap", "fixture/favored"))
    epsilon = first["constants"]["epsilon"]
    # The 97.5% bound of an unseen cell exceeds epsilon, but the q80 admission quantile does not.
    assert native["P_silent_accepted"]["upper"] > epsilon >= native["admission_risk"] and native["admissible"]
    assert trap["admission_risk"] > epsilon and not trap["admissible"]
    assert first["selection"]["sampling_policy"]["admission_quantile"] == .8
    # The point-guess gap exceeds the old 10% mean band but posterior costs overlap.
    assert native["J"] - favored["J"] > .1 * favored["J"]
    assert first["exploration"]["relative_band"] is None
    assert {c["model"] for c in first["exploration"]["eligible_cells"]} == {"fixture/favored", "fixture/native"}

    chosen = [decide(engine, candidates, specs, seed=seed)["chosen"]["model"] for seed in range(150)]
    assert "fixture/trap" not in chosen
    assert 15 <= chosen.count("fixture/native") <= 135
    load(engine.config, first["decision_id"])

    # An explicit risk-class explore_within is still respected.
    capped = tmp_path / "capped"
    capped.mkdir()
    (capped / "howhard.toml").write_text('on_cap_unmet = "flag-review"\n[task_types]\ntask = "standard"\n'
                                         '[classes.standard]\nexplore_within = 0.1\n')
    capped_engine = Engine(capped / "howhard.toml")
    capped_first = decide(capped_engine, candidates, specs, seed=0)
    assert capped_first["exploration"]["relative_band"] == .1
    assert {decide(capped_engine, candidates, specs, seed=s)["chosen"]["model"] for s in range(40)} == {"fixture/favored"}

    # The removed global sampling.explore_within cannot silently change meaning.
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "howhard.toml").write_text('on_cap_unmet = "flag-review"\n[sampling]\nexplore_within = 0.1\n')
    with pytest.raises(HowHardError, match="risk class"):
        Engine(bad / "howhard.toml")


def test_recency_decay_detects_regression_and_recovers_before_undecayed_counts(tmp_path):
    def admission(directory, outcomes, settings=""):
        config = directory / "howhard.toml"
        directory.mkdir()
        config.write_text('on_cap_unmet = "flag-review"\n[task_types]\ntask = "standard"\n' + settings)
        engine = Engine(config, retrieval=RecentDouble())
        observe(engine, "fixture/worker", outcomes, "run")
        result = decide(engine, WORKER, {"fixture/worker": spec(.94, .05, .01)})
        return result["cells"][0], result

    healthy, _ = admission(tmp_path / "healthy", ["clean"] * 5)
    assert healthy["admissible"]
    regressed, result = admission(tmp_path / "regressed", ["clean"] * 5 + ["silent"] * 3)
    assert not regressed["admissible"] and regressed["P_silent"]["mean"] > .15
    ages = {o["id"]: (o["age"], o["weight"]) for o in result["semantic"]["outcome_observations"]}
    assert ages["run-7"][0] == 0 and ages["run-0"][0] == 7
    assert ages["run-0"][1] == pytest.approx(ages["run-7"][1] * .95 ** 7)

    recovering = ["clean"] * 3 + ["silent"] * 2 + ["clean"] * 19
    recovered, recovered_result = admission(tmp_path / "recovered", recovering)
    undecayed, _ = admission(tmp_path / "undecayed", recovering, "[sampling]\noutcome_recency = 1.0\n")
    assert len(recovered_result["semantic"]["outcome_observations"]) == 24
    assert {o["id"] for o in recovered_result["semantic"]["outcome_observations"]} == {f"run-{i}" for i in range(24)}
    rec_ages = {o["id"]: o["age"] for o in recovered_result["semantic"]["outcome_observations"]}
    assert rec_ages["run-23"] == 0 and rec_ages["run-9"] == 14 and rec_ages["run-0"] == 23
    assert recovered["admissible"] and not undecayed["admissible"]
    assert recovered["admission_risk"] < undecayed["admission_risk"]


def test_frozen_v7_to_v10_receipts_load_observe_replay_and_report(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    candidates = [{"model": f"fixture/model-{i}", "effort": "high", "usd_per_mtok": 1.0} for i in range(2)]
    specs = {c["model"]: spec(.95, .04, .01) for c in candidates}
    current_id = decide(engine, candidates, specs, seed=11)["decision_id"]
    current = load(engine.config, current_id)
    assert current["format"] == "howhard-decision-v11"
    ids = {}
    for name, legacy in (("v10", to_v10(current)), ("v9", to_v9(current)), ("v8", to_v8(current)), ("v7", to_v7(current))):
        ids[name] = digest(legacy)
        with locked(engine.config):
            put(engine.config, "decisions", ids[name], canonical(legacy) + b"\n")
        assert load(engine.config, ids[name])["format"] == f"howhard-decision-{name}"
    engine.observe_decision(ids["v7"], "clean", id="from-v7", seconds=12.)
    engine.observe_decision(ids["v8"], "loud", id="from-v8", cause="capability", active_seconds=10.)
    engine.observe_decision(ids["v9"], "clean", id="from-v9", active_seconds=10.)
    engine.observe_decision(ids["v10"], "clean", id="from-v10", active_seconds=10.)
    engine.observe_decision(current_id, "clean", id="from-v11", active_seconds=10.)
    beliefs = engine.memory_beliefs()
    assert beliefs["unscored_runs"] == 0
    assert next(e for e in beliefs["experts"] if e["key"] == "llm")["evidence_count"] == 5
    stats = report(engine)
    assert stats["unsupported_receipts"] == [] and stats["forecast_metrics"]["llm"]["runs"] == 5
    # The new decision sees those outcomes as per-cell evidence.
    later = decide(engine, candidates, specs, seed=12)
    assert {o["outcome"] for o in later["semantic"]["outcome_observations"]} == {"clean", "loud"}

    # Cross-version shapes are rejected in both directions.
    v8 = load(engine.config, ids["v8"])
    for tamper in (lambda r: r["semantic"].update(interval_kind="dirichlet-posterior-mixture"),
                   lambda r: r["semantic"].update(outcome_observations=[]),
                   lambda r: r["semantic"]["worlds"]["llm"][0].update(silent_ci=[0., .5])):
        bad = deepcopy(v8)
        tamper(bad)
        with pytest.raises(HowHardError):
            load(engine.config, bad)
    for tamper in (lambda r: r["semantic"].update(interval_kind="weighted-forecast-envelope"),
                   lambda r: r["semantic"].pop("outcome_observations"),
                   lambda r: r["selection"]["cells"][0].update(admission_risk=r["selection"]["cells"][0]["admission_risk"] + .01),
                   lambda r: r["selection"]["cells"][0]["P_silent"].update(upper=r["selection"]["cells"][0]["P_silent"]["mean"]),
                   lambda r: r["selection"]["sampling_policy"].update(explore_within=.1),
                   lambda r: r["semantic"]["sampling"]["cells"][0]["outcome_posterior"].update(prior_concentration=100.),
                   lambda r: r["semantic"]["sampling"]["cells"][0].update(evidence_count=500)):
        bad = deepcopy(current)
        tamper(bad)
        with pytest.raises(HowHardError):
            load(engine.config, bad)
