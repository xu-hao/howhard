"""Tiered admission at public decision boundaries (#130).

Every forecast cell is sorted by ``pi``, its chance of meeting the silent limit, into
trusted, on trial or excluded. Trusted and on-trial cells compete together, a cell is still
drawn when every cell is excluded, and the two flags describe the chosen cell. Every fixture
is synthetic and owns its state.
"""

from collections import Counter
from copy import deepcopy
import json
import math

import pytest

from howhard import Engine
from howhard.cli import main
from howhard.context import canonical, digest
from howhard.decisions import load
from howhard.payload_store import put
from howhard.report import report
from howhard.storage import locked
from howhard.validation import HowHardError
from legacy_receipts import to_v7, to_v8, to_v9, to_v10
from test_capability_filter import SimilarityDouble, filtering, forecasts, inventory, start
from test_outcome_posterior import by_model, calibration, decide, logit, observe, reference_belief, spec

EPSILON = .10  # The packaged limit of the standard risk class.
TASK = '[task_types]\ntask = "standard"\n'


@pytest.fixture(autouse=True)
def no_model_weights(monkeypatch):
    """Every fixture here is synthetic: no embedding weights may be loaded or fetched."""
    def refuse(*args, **kwargs):
        raise AssertionError("tiered-admission tests must not load embedding weights")
    monkeypatch.setattr("howhard.retrieval.FastEmbedder.encode", refuse)


def configured(directory, settings="", **options):
    directory.mkdir(parents=True)
    path = directory / "howhard.toml"
    path.write_text(TASK + settings)
    return Engine(path, **options)


def priced(*names, **extra):
    return [{"model": f"fixture/{name}", "effort": "high", "usd_per_mtok": 1.0, **extra} for name in names]


def budget(credit):
    return {"available_credit_usd": credit, "session_5h_remaining_credit_usd": credit,
            "weekly_7d_remaining_credit_usd": 5.0}


def with_silent_share(share, loud=.05):
    """A forecast whose guess for the accepted-defect share, silent / (clean + silent), is ``share``."""
    return spec((1 - loud) * (1 - share), loud, (1 - loud) * share)


# Three forecasts of the accepted-defect share with no stated range and no counted run:
# the share, then pi, tier and admission_risk at the limit of 0.10.
TABLE = {"trusted": (.02, .957, "trusted", .021),
         "trial": (.10, .702, "on-trial", .156),
         "excluded": (.50, .011, "excluded", .692)}
TABLE_SPECS = {f"fixture/{name}": with_silent_share(row[0]) for name, row in TABLE.items()}

# Forecasts whose pi is 0.12, 0.08 and 0.05: the shares are solved from the belief's start. The
# third has the lowest mean cost, so only the rule of the second pool keeps it from being chosen.
NEAR_BEST = {"fixture/first": with_silent_share(.3100763259434137, loud=.7),
             "fixture/second": with_silent_share(.34527039736158194, loud=.7),
             "fixture/third": with_silent_share(.38400454829285435, loud=.02)}
# A trusted and an on-trial cell whose costs overlap: the on-trial one is faster and riskier.
OVERLAP = {"fixture/trusted": spec(.93, .05, .02, seconds=3 * 3600), "fixture/trial": spec(.85, .06, .09)}


def test_pi_is_the_share_of_the_belief_within_the_limit_and_sorts_cells_into_three_tiers(tmp_path):
    engine = configured(tmp_path / "table")
    result = decide(engine, priced(*TABLE), TABLE_SPECS, seed=1)
    assert result["constants"]["epsilon"] == EPSILON and set(result["semantic"]["worlds"]) == {"llm"}
    cells = by_model(result)
    for name, (share, pi, tier, risk) in TABLE.items():
        row = cells[f"fixture/{name}"]
        belief = calibration(result, f"fixture/{name}")["outcome_beliefs"]["silent"]
        assert belief["prediction"] == pytest.approx(share) and belief["counts"] == [0., 0.]
        expected = reference_belief(belief["prediction"])
        assert row["pi"] == pytest.approx(pi, abs=5e-4)
        assert row["pi"] == belief["pi"] and row["pi"] == pytest.approx(expected["pi"](EPSILON), abs=1e-9)
        assert row["tier"] == tier and row["admissible"] is (tier == "trusted")
        assert row["admission_risk"] == pytest.approx(risk, abs=5e-4)
        assert row["admission_risk"] == pytest.approx(expected["point"](.8), abs=1e-9)
    policy = result["selection"]["sampling_policy"]
    assert (policy["admission_quantile"], policy["trial_probability"], policy["all_excluded_factor"]) == (.8, .2, .5)
    assert "on_cap_unmet" not in result["selection"]
    assert load(engine.config, result["decision_id"])["selection"]["cells"] == result["selection"]["cells"]

    # A memory expert enters the one forecast of the cell with its weight, on the log-odds scale:
    # pi is the share within the limit of the belief that starts from the merged forecast.
    mixed_engine = configured(tmp_path / "mixed", retrieval=SimilarityDouble())
    memory = mixed_engine.memory.add("Risk", "This worker escapes more defects on rollback repairs")
    mixed = decide(mixed_engine, priced("trusted"), TABLE_SPECS, seed=1, memory_forecasts=[
        {**memory, "specs": {"fixture/trusted": with_silent_share(.10)}}])
    weights = {e["key"]: e["weight"] for e in mixed["semantic"]["beliefs"]["experts"]}
    assert len(weights) == 2 and all(0 < w < 1 for w in weights.values())
    centre = weights["llm"] * logit(.02) + (1 - weights["llm"]) * logit(.10)
    stated = math.sqrt(weights["llm"] * (logit(.02) - centre) ** 2 + (1 - weights["llm"]) * (logit(.10) - centre) ** 2)
    expected = reference_belief(1 / (1 + math.exp(-centre)), stated)
    assert mixed["cells"][0]["pi"] == pytest.approx(expected["pi"](EPSILON), abs=1e-9)
    assert cells["fixture/trial"]["pi"] < mixed["cells"][0]["pi"] < cells["fixture/trusted"]["pi"]
    load(mixed_engine.config, mixed["decision_id"])


def test_pick_and_summary_report_pi_and_tier_for_every_cell(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "candidates.json").write_text(json.dumps(priced("a", "b")))
    arguments = ["task", "--risk", "standard", "--candidates", "candidates.json"]
    assert main(["summary", *arguments, "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert main(["pick", *arguments, "--json", "--seed", "4"]) == 0
    picked = json.loads(capsys.readouterr().out)
    for result in (summary, picked):
        assert len(result["cells"]) == 2
        for row in result["cells"]:
            # The fixture provider forecasts clean 0.5, loud 0.45 and silent 0.05 for every cell.
            assert row["pi"] == pytest.approx(reference_belief(.05 / .55)["pi"](EPSILON), abs=1e-9)
            assert row["tier"] == "on-trial" and row["admissible"] is False
    assert picked["selection"]["cells"] == picked["cells"] and picked["needs_review"] and not picked["cap_met"]
    assert main(["pick", *arguments, "--seed", "4"]) == 0
    text = capsys.readouterr().out
    assert "on-trial (0.734;" in text and "Chosen cell: on-trial" in text and "pool: trusted-and-on-trial" in text
    # Only the loud rate and the accepted-defect share have a belief, so only they print a range.
    header = next(line for line in text.splitlines() if line.startswith("family/effort"))
    assert "P_clean  P_loud [95%]  P_silent  P_silent accepted [95%]" in header
    row = next(line for line in text.splitlines() if "on-trial (0.734;" in line).split("  ")
    assert [field.count("[") for field in row[1:5]] == [0, 1, 0, 1]


def test_a_cell_is_trusted_exactly_when_the_trusted_point_of_its_belief_is_within_the_limit(tmp_path):
    decisions = []
    # A grid of forecast silent shares, read by the forecaster alone and beside a memory expert.
    grid = configured(tmp_path / "grid", retrieval=SimilarityDouble())
    memory = grid.memory.add("Risk", "Silent defects are more common than forecast on rollback repairs")
    names = [f"cell-{i}" for i in range(6)]
    shares = [.005, .02, .04, .06, .12, .3]
    specs = {f"fixture/{name}": spec(.95 - share, .05, share) for name, share in zip(names, shares)}
    overrides = {f"fixture/{name}": spec(.9 - share, .05, share + .05, silent_ci=[0., .6])
                 for name, share in zip(names[::2], shares[::2])}
    for forecast in ((), [{**memory, "specs": overrides}]):
        result = decide(grid, priced(*names), specs, seed=2, memory_forecasts=forecast)
        assert len(result["cells"]) == 6 and len(result["semantic"]["worlds"]) == 1 + bool(forecast)
        decisions.append((grid, result))
    # One cell whose posterior is moved by its own recorded outcomes.
    for index, outcomes in enumerate((["clean"] * 6, ["clean", "silent", "clean", "clean"])):
        history = configured(tmp_path / f"history-{index}", retrieval=SimilarityDouble())
        observe(history, "fixture/worker", outcomes, "run")
        result = decide(history, priced("worker"), {"fixture/worker": spec(.9, .07, .03)}, seed=2)
        assert len(result["semantic"]["outcome_observations"]) == len(outcomes)
        decisions.append((history, result))
    seen = Counter()
    for engine, result in decisions:
        for row in result["cells"]:
            # admission_risk is the admission_quantile point of the same belief that pi is read from.
            within = row["admission_risk"] <= result["constants"]["epsilon"]
            assert (row["tier"] == "trusted") is row["admissible"] is within
            assert (row["pi"] >= .8) is within
            seen[row["tier"]] += 1
        # A v10 receipt keeps its own rule on its own recorded posterior: admissible == (risk <= epsilon).
        legacy = load(engine.config, to_v10(load(engine.config, result["decision_id"])))
        assert all(c["admissible"] is (c["admission_risk"] <= result["constants"]["epsilon"])
                   for c in legacy["selection"]["cells"])
    assert seen["trusted"] >= 3 and seen["on-trial"] >= 3 and seen["excluded"] >= 2


def test_trusted_and_on_trial_cells_compete_and_the_flags_describe_the_chosen_cell(tmp_path):
    engine = configured(tmp_path / "overlap")
    candidates = priced("trusted", "trial")
    first = decide(engine, candidates, OVERLAP, seed=0)
    tiers = {row["model"]: row["tier"] for row in first["cells"]}
    assert tiers == {"fixture/trusted": "trusted", "fixture/trial": "on-trial"}
    assert {c["model"] for c in first["exploration"]["eligible_cells"]} == set(tiers)
    assert first["exploration"]["enabled"] and first["exploration"]["pool"] == "trusted-and-on-trial"
    chosen = Counter()
    for seed in range(120):
        result = decide(engine, candidates, OVERLAP, seed=seed)
        model = result["chosen"]["model"]
        chosen[model] += 1
        trusted = model == "fixture/trusted"
        assert result["cap_met"] is trusted and result["needs_review"] is (not trusted)
        assert result["selection"]["cap_met"] is trusted and result["selection"]["needs_review"] is (not trusted)
        assert result["semantic"]["sampling"]["enabled"] and "sampled_expert" in result["semantic"]
        assert result["exploration"]["reason"] in ("sampled-within-posterior-overlap", "mean-best")
    # Nothing limits how often the on-trial cell is chosen: both win a fair share of the draws.
    assert chosen["fixture/trusted"] >= 15 and chosen["fixture/trial"] >= 15

    # A fixed seed reproduces the choice and its sampled costs in a separate store.
    again = decide(configured(tmp_path / "again"), candidates, OVERLAP, seed=7)
    original = decide(engine, candidates, OVERLAP, seed=7)
    assert again["chosen"] == original["chosen"] and again["selection"] == original["selection"]
    assert again["semantic"]["sampled_expert"] == original["semantic"]["sampled_expert"]


def test_excluded_and_credit_short_cells_never_win_while_a_pickable_cell_has_credit(tmp_path):
    engine = configured(tmp_path / "credit")
    candidates = [{"model": "fixture/excluded", "effort": "high", "usd_per_mtok": 1.0, "credit_budget": budget(1.)},
                  {"model": "fixture/short", "effort": "high", "usd_per_mtok": 5.0, "credit_budget": budget(.001)},
                  {"model": "fixture/trial", "effort": "high", "usd_per_mtok": 10.0, "credit_budget": budget(1.)},
                  {"model": "fixture/trusted", "effort": "high", "usd_per_mtok": 20.0, "credit_budget": budget(1.)}]
    # The excluded cell is quick and the credit-short cell is the best cell on every count.
    specs = {**OVERLAP, "fixture/excluded": spec(.66, .02, .32, seconds=1),
             "fixture/short": spec(.985, .01, .005, seconds=1)}
    first = decide(engine, candidates, specs, seed=0)
    cells = by_model(first)
    assert {m: (c["tier"], c["credit_ok"]) for m, c in cells.items()} == {
        "fixture/excluded": ("excluded", True), "fixture/short": ("trusted", False),
        "fixture/trial": ("on-trial", True), "fixture/trusted": ("trusted", True)}
    assert cells["fixture/short"]["J"] == min(c["J"] for c in cells.values())
    assert {c["model"] for c in first["exploration"]["eligible_cells"]} == {"fixture/trial", "fixture/trusted"}
    why = {w["model"]: w for w in first["why"]}
    assert why["fixture/short"]["reason"] == "insufficient-credit"
    assert why["fixture/excluded"]["reason"] == "silent-cap"
    assert f"pi {cells['fixture/excluded']['pi']:.6g}" in why["fixture/excluded"]["message"]
    assert "trial_probability 0.2" in why["fixture/excluded"]["message"]
    assert all(w["reason"] != "lowest-bound" for w in first["why"])
    chosen = Counter(decide(engine, candidates, specs, seed=seed)["chosen"]["model"] for seed in range(100))
    assert set(chosen) == {"fixture/trial", "fixture/trusted"}

    # Credit, not the tier, takes the best cell out: when it is the only pickable cell, the
    # cells with enough credit compete as the second pool and the decision is flagged.
    alone = decide(engine, candidates[:2] + [{**candidates[0], "model": "fixture/worse"}],
                   {**specs, "fixture/worse": spec(.3, .02, .68, seconds=1)}, seed=0)
    assert alone["chosen"]["model"] == "fixture/excluded" and alone["exploration"]["pool"] == "all-excluded"
    assert alone["needs_review"] and not alone["cap_met"]
    load(engine.config, alone["decision_id"])

    # When no cell has enough credit, selection is refused as before.
    broke = [{**c, "credit_budget": budget(.0001)} for c in candidates]
    with pytest.raises(HowHardError, match="no candidate has sufficient remaining credit"):
        decide(configured(tmp_path / "broke"), broke, specs, seed=0)


def test_when_every_cell_is_excluded_a_sample_is_drawn_among_the_cells_nearest_the_best(tmp_path):
    engine = configured(tmp_path / "excluded")
    candidates = [{"model": "fixture/first", "effort": "high", "usd_per_mtok": 2.0},
                  {"model": "fixture/second", "effort": "high", "usd_per_mtok": 2.0},
                  {"model": "fixture/third", "effort": "high", "usd_per_mtok": 1.0}]
    first = decide(engine, candidates, NEAR_BEST, seed=0)
    cells = by_model(first)
    assert [cells[m]["pi"] for m in NEAR_BEST] == pytest.approx([.12, .08, .05], abs=1e-9)
    assert {c["tier"] for c in cells.values()} == {"excluded"} and all(c["credit_ok"] for c in cells.values())
    # The bar is 0.5 * 0.12 = 0.06: the third cell is left out although its mean cost is the lowest.
    assert cells["fixture/third"]["J"] == min(c["J"] for c in cells.values())
    assert {c["model"] for c in first["exploration"]["eligible_cells"]} == {"fixture/first", "fixture/second"}
    assert first["exploration"]["pool"] == "all-excluded" and first["exploration"]["enabled"]
    assert first["exploration"]["reason"] != "disabled-cap-unmet"
    why = next(w for w in first["why"] if w["model"] == "fixture/third")
    assert why["reason"] == "silent-cap" and "pi 0.05" in why["message"] and "all_excluded_factor 0.5" in why["message"]
    chosen = Counter()
    for seed in range(120):
        result = decide(engine, candidates, NEAR_BEST, seed=seed)
        chosen[result["chosen"]["model"]] += 1
        assert result["needs_review"] and not result["cap_met"]
        assert result["exploration"]["pool"] == "all-excluded"
        # A sample is drawn: the frozen draw covers every forecast cell.
        sampling = result["semantic"]["sampling"]
        assert sampling["enabled"] and len(sampling["cells"]) == 3 and "sampled_expert" in result["semantic"]
    assert set(chosen) == {"fixture/first", "fixture/second"} and min(chosen.values()) >= 15
    assert load(engine.config, first["decision_id"])["selection"]["exploration"]["pool"] == "all-excluded"

    # The factor moves the bar: 1 keeps only the best chance, 0 keeps every cell with credit.
    for factor, pool in ((1., {"fixture/first"}), (0., set(NEAR_BEST))):
        other = configured(tmp_path / f"factor-{factor}", f"[sampling]\nall_excluded_factor = {factor}\n")
        result = decide(other, candidates, NEAR_BEST, seed=0)
        outside = {w["model"] for w in result["why"] if w["reason"] == "silent-cap"}
        eligible = {c["model"] for c in result["exploration"]["eligible_cells"]}
        assert eligible <= pool and result["chosen"]["model"] in pool and not outside & pool
        load(other.config, result["decision_id"])


def test_widening_runs_only_while_every_forecast_cell_is_excluded(tmp_path):
    excluded = lambda request: forecasts(request, silent=.8)
    on_trial = lambda request: forecasts(request, silent=.05, clean=.5)

    # One on-trial cell stops widening, where format v10 kept widening until a cell was admissible.
    engine = Engine(directory=tmp_path / "stops")
    pending = start(engine, inventory(6))
    pending = engine.decide(pending["request_id"], filtering(pending["baseline_request"], 1, 1))
    wider = engine.decide(pending["request_id"], excluded(pending["baseline_request"]))
    assert wider["stage"] == "forecast"
    assert [c["model"] for c in wider["baseline_request"]["cells"]] == ["fixture/model-002"]
    result = engine.decide(wider["request_id"], on_trial(wider["baseline_request"]), seed=3)
    assert {c["model"]: c["tier"] for c in result["cells"]} == {"fixture/model-001": "excluded",
                                                               "fixture/model-002": "on-trial"}
    state = result["capability"]
    assert [w["trigger"] for w in state["widening"]] == ["all-excluded"] and state["widening_stop"] == "pickable"
    assert result["chosen"]["model"] == "fixture/model-002" and result["needs_review"] and not result["cap_met"]
    assert load(engine.config, result["decision_id"])["capability"]["widening_stop"] == "pickable"

    # An on-trial cell in the band means no round at all.
    direct = Engine(directory=tmp_path / "direct")
    pending = start(direct, inventory(6))
    pending = direct.decide(pending["request_id"], filtering(pending["baseline_request"], 1, 1))
    result = direct.decide(pending["request_id"], on_trial(pending["baseline_request"]), seed=3)
    assert result["capability"]["widening"] == [] and result["capability"]["widening_stop"] == "pickable"
    assert result["cells"][0]["tier"] == "on-trial" and not result["cells"][0]["admissible"]

    # Every cell excluded: one stronger cell per round, up to widening_rounds, then a flagged draw.
    bounded = Engine(directory=tmp_path / "bounded")
    pending = start(bounded, inventory(8))
    pending = bounded.decide(pending["request_id"], filtering(pending["baseline_request"], 1, 1))
    requested = []
    while "decision_id" not in pending:
        requested.append([c["model"] for c in pending["baseline_request"]["cells"]])
        pending = bounded.decide(pending["request_id"], excluded(pending["baseline_request"]), seed=1)
    assert requested == [["fixture/model-001"], ["fixture/model-002"], ["fixture/model-003"]]
    state = pending["capability"]
    assert state["policy"]["widening_rounds"] == 2 and state["widening_stop"] == "round-limit"
    assert [(w["round"], w["trigger"]) for w in state["widening"]] == [(1, "all-excluded"), (2, "all-excluded")]
    assert {c["tier"] for c in pending["cells"]} == {"excluded"} and len(pending["cells"]) == 3
    assert pending["exploration"]["pool"] == "all-excluded" and pending["needs_review"] and not pending["cap_met"]
    assert pending["semantic"]["sampling"]["enabled"]
    load(bounded.config, pending["decision_id"])


def test_widening_reads_tiers_and_never_credit(tmp_path):
    engine = Engine(directory=tmp_path)
    cells = [{"model": f"fixture/model-{i:03d}", "effort": "high", "usd_per_mtok": 1.0,
              "credit_budget": budget(.0001 if i == 1 else 1.)} for i in range(5)]
    pending = start(engine, cells)
    pending = engine.decide(pending["request_id"], filtering(pending["baseline_request"], 1, 2))
    by_cell = {c["model"]: c for c in pending["baseline_request"]["cells"]}
    rows = [{"family": c["family"], "effort": c["effort"], "tokens": 1000, "seconds": 10,
             **(dict(clean=.5, loud=.45, silent=.05) if model == "fixture/model-001" else dict(clean=.15, loud=.05, silent=.8))}
            for model, c in by_cell.items()]
    result = engine.decide(pending["request_id"], {"forecasts": rows, "memory_forecasts": [],
                                                   "reason": "Synthetic tier fixture"}, seed=5)
    # The on-trial cell lacks credit, yet it stops widening; the draw then uses the second pool.
    assert "decision_id" in result and result["capability"]["widening"] == []
    assert result["capability"]["widening_stop"] == "pickable"
    cells_by_model = by_model(result)
    assert cells_by_model["fixture/model-001"]["tier"] == "on-trial" and not cells_by_model["fixture/model-001"]["credit_ok"]
    assert result["chosen"]["model"] == "fixture/model-002" and result["exploration"]["pool"] == "all-excluded"
    assert result["needs_review"]
    load(engine.config, result["decision_id"])


@pytest.fixture
def routed(tmp_path, monkeypatch, capsys):
    """Run one seeded CLI pick under a configuration file; return its JSON result and standard error."""
    (tmp_path / "candidates.json").write_text(json.dumps(priced("a", "b", "c")))

    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")

    def run(name, settings):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "howhard.toml").write_text(settings + TASK)
        capsys.readouterr()
        status = main(["pick", "task", "--config", str(directory / "howhard.toml"), "--json", "--seed", "11",
                       "--candidates", str(tmp_path / "candidates.json")])
        captured = capsys.readouterr()
        return status, captured.out, captured.err
    return run


def test_on_cap_unmet_is_accepted_and_ignored_with_one_notice(routed, tmp_path):
    status, out, err = routed("plain", "")
    assert status == 0 and err == ""
    plain = json.loads(out)
    assert "on_cap_unmet" not in plain["selection"]
    compared = ("chosen", "seed", "cap_met", "needs_review", "exploration", "cells", "selection", "why")
    for index, value in enumerate(("flag-review", "lowest-bound")):
        status, out, err = routed(f"old-{index}", f'on_cap_unmet = "{value}"\n')
        assert status == 0
        # Standard output is still exactly one JSON document; the notice is one line on standard error.
        result = json.loads(out)
        notice = err.splitlines()
        assert len(notice) == 1 and "on_cap_unmet" in notice[0] and "retired and ignored" in notice[0]
        assert {k: result[k] for k in compared} == {k: plain[k] for k in compared}
        assert result.keys() == plain.keys() and "on_cap_unmet" not in result["selection"]

    # The library boundary loads the same files, and the packaged default no longer states the key.
    for name in ("plain", "old-0", "old-1"):
        engine = Engine(tmp_path / name / "howhard.toml", directory=tmp_path / "work")
        assert "on_cap_unmet" not in engine.config.data
    assert "on_cap_unmet" not in Engine(directory=tmp_path / "defaults").config.data


def test_trial_probability_zero_excludes_no_cell_and_bad_tier_constants_are_refused(tmp_path):
    engine = configured(tmp_path / "zero", "[sampling]\ntrial_probability = 0.0\n")
    result = decide(engine, priced(*TABLE), TABLE_SPECS, seed=1)
    assert {c["model"]: c["tier"] for c in result["cells"]} == {
        "fixture/trusted": "trusted", "fixture/trial": "on-trial", "fixture/excluded": "on-trial"}
    assert result["exploration"]["pool"] == "trusted-and-on-trial"
    load(engine.config, result["decision_id"])
    # No cell is excluded even with a posterior that almost surely misses the limit, so nothing widens.
    never = configured(tmp_path / "never", "[sampling]\ntrial_probability = 0.0\n")
    pending = start(never, inventory(6))
    pending = never.decide(pending["request_id"], filtering(pending["baseline_request"], 1, 1))
    hopeless = never.decide(pending["request_id"], forecasts(pending["baseline_request"], silent=.8), seed=1)
    assert hopeless["cells"][0]["tier"] == "on-trial" and hopeless["cells"][0]["pi"] < 1e-3
    assert hopeless["capability"]["widening"] == [] and hopeless["exploration"]["pool"] == "trusted-and-on-trial"

    # The on-trial line may reach the trusted line, which leaves no on-trial tier.
    edge = configured(tmp_path / "edge", "[sampling]\ntrial_probability = 0.8\n")
    assert {c["tier"] for c in decide(edge, priced(*TABLE), TABLE_SPECS, seed=1)["cells"]} == {"trusted", "excluded"}

    refused = (("trial_probability = 0.81", "sampling.trial_probability"),
               ("trial_probability = -0.1", "sampling.trial_probability"),
               ('trial_probability = "0.2"', "sampling.trial_probability"),
               ("admission_quantile = 0.6\ntrial_probability = 0.7", "sampling.trial_probability"),
               ("all_excluded_factor = 1.5", "sampling.all_excluded_factor"),
               ("all_excluded_factor = -0.5", "sampling.all_excluded_factor"),
               ("all_excluded_factor = true", "sampling.all_excluded_factor"))
    for index, (setting, key) in enumerate(refused):
        with pytest.raises(HowHardError, match=key):
            configured(tmp_path / f"refused-{index}", f"[sampling]\n{setting}\n")


def executed(receipt, model):
    """Point a receipt at another forecast cell, keeping every other part consistent."""
    row = next(c for c in receipt["selection"]["cells"] if c["model"] == model)
    receipt["chosen"] = {k: row[k] for k in ("model", "family", "effort")}
    cell = next(c for c in receipt["capability"]["cells"] if c["model"] == model)
    receipt["capability"]["executed"] = {k: cell[k] for k in ("key", "classification", "position", "P_clear")}


def test_a_stored_decision_revalidates_and_changed_tier_fields_are_refused(tmp_path):
    engine = configured(tmp_path / "frozen")
    names = ("trusted", "trial", "excluded")
    # Choose the on-trial cell: it is the only cell with enough credit besides the excluded one.
    candidates = [{"model": f"fixture/{name}", "effort": "high", "usd_per_mtok": 1.0,
                   "credit_budget": budget(.00001 if name == "trusted" else 1.)} for name in names]
    result = decide(engine, candidates, TABLE_SPECS, seed=5)
    assert result["chosen"]["model"] == "fixture/trial" and result["needs_review"] and not result["cap_met"]
    receipt = load(engine.config, result["decision_id"])
    assert receipt["format"] == "howhard-decision-v11" and digest(receipt) == result["decision_id"]
    assert receipt["selection"] == result["selection"] and receipt["semantic"]["sampling"]["enabled"]

    def cell(r, name):
        return next(c for c in r["selection"]["cells"] if c["model"] == f"fixture/{name}")

    # Each change is refused by the check that owns the changed field.
    tampers = {
        "pi": (lambda r: cell(r, "trial").update(pi=cell(r, "trial")["pi"] + .001), "pi does not match"),
        "pi across a line": (lambda r: cell(r, "trial").update(pi=.9), "pi does not match"),
        "tier": (lambda r: cell(r, "trial").update(tier="trusted"), "tier/admissible does not match pi"),
        "tier and admissible": (lambda r: cell(r, "trial").update(tier="trusted", admissible=True),
                                "tier/admissible does not match pi"),
        "unknown tier": (lambda r: cell(r, "excluded").update(tier="rejected"), "tier/admissible does not match pi"),
        "admissible": (lambda r: cell(r, "trial").update(admissible=True), "tier/admissible does not match pi"),
        "missing pi": (lambda r: cell(r, "trial").pop("pi"), "missing keys"),
        "cap_met": (lambda r: r["selection"].update(cap_met=True), "inconsistent cap/review state"),
        "needs_review": (lambda r: r["selection"].update(needs_review=False), "inconsistent cap/review state"),
        "both flags": (lambda r: r["selection"].update(cap_met=True, needs_review=False), "inconsistent cap/review state"),
        "flag type": (lambda r: r["selection"].update(cap_met=0, needs_review=1), "inconsistent cap/review state"),
        "pool": (lambda r: r["selection"]["exploration"].update(pool="all-excluded"), "exploration.pool"),
        "missing pool": (lambda r: r["selection"]["exploration"].pop("pool"), "exploration.pool"),
        "chosen outside the pool": (lambda r: executed(r, "fixture/excluded"), "outside the admission pool"),
        "chosen without credit": (lambda r: executed(r, "fixture/trusted"), "not schedulable within credit_budget"),
        "no draw": (lambda r: (r["semantic"]["sampling"].update(enabled=False, reason="cap-unmet", cells=[]),
                               r["semantic"].pop("sampled_expert")), "a sample is always drawn"),
        "retired policy": (lambda r: r["selection"].update(on_cap_unmet="flag-review"), "unknown keys"),
        "trial line": (lambda r: r["selection"]["sampling_policy"].update(trial_probability=.75),
                       "tier/admissible does not match pi"),
        "missing tier constant": (lambda r: r["selection"]["sampling_policy"].pop("all_excluded_factor"), "missing keys"),
        "frozen draw policy": (lambda r: r["semantic"]["sampling"]["policy"].pop("trial_probability"), "missing keys"),
        "widening stop": (lambda r: r["capability"].update(widening_stop="exhausted"),
                          "widening_stop does not match the tiers"),
        "v10 widening stop": (lambda r: r["capability"].update(widening_stop="cap-met"), "widening_stop: expected one of"),
    }
    for name, (tamper, message) in tampers.items():
        bad = deepcopy(receipt)
        tamper(bad)
        with pytest.raises(HowHardError, match=message):
            load(engine.config, bad)
            pytest.fail(f"{name}: accepted")

    # A widened decision freezes the trigger of each round.
    wide = Engine(directory=tmp_path / "wide")
    pending = start(wide, inventory(6))
    pending = wide.decide(pending["request_id"], filtering(pending["baseline_request"], 1, 1))
    while "decision_id" not in pending:
        pending = wide.decide(pending["request_id"], forecasts(pending["baseline_request"], silent=.8), seed=1)
    widened = load(wide.config, pending["decision_id"])
    for tamper, message in ((lambda r: r["capability"]["widening"][0].update(trigger="silent-cap-unmet"), "widening must forecast"),
                            (lambda r: r["capability"].update(widening_stop="pickable"), "widening_stop does not match the tiers")):
        bad = deepcopy(widened)
        tamper(bad)
        with pytest.raises(HowHardError, match=message):
            load(wide.config, bad)


@pytest.mark.parametrize("on_cap_unmet", ["flag-review", "lowest-bound"])
def test_v7_to_v10_receipts_with_a_met_or_unmet_cap_load_observe_replay_and_report(tmp_path, on_cap_unmet):
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())

    def widened(label, excluded_rounds, seed):
        """Forecast one band cell, then each widening cell: excluded for the first rounds, trusted after."""
        cells = [{"model": f"fixture/{label}-{i:03d}", "effort": "high"} for i in range(6)]
        pending = start(engine, cells)
        pending = engine.decide(pending["request_id"], filtering(pending["baseline_request"], 1, 1))
        rounds = 0
        while "decision_id" not in pending:
            share = .8 if rounds < excluded_rounds else .01
            rounds += 1
            pending = engine.decide(pending["request_id"], forecasts(pending["baseline_request"], silent=share), seed=seed)
        return pending

    # One decision that widened to a trusted cell, and one whose every cell stayed excluded.
    met, unmet = widened("met", 1, 3), widened("unmet", 3, 4)
    assert met["cap_met"] and met["capability"]["widening_stop"] == "pickable"
    assert not unmet["cap_met"] and unmet["exploration"]["pool"] == "all-excluded"
    assert unmet["capability"]["widening_stop"] == "round-limit"
    stored = {}
    for label, current in (("met", met), ("unmet", unmet)):
        v10 = to_v10(load(engine.config, current["decision_id"]), on_cap_unmet=on_cap_unmet)
        for version, legacy in (("v10", v10), ("v9", to_v9(v10)), ("v8", to_v8(v10)), ("v7", to_v7(v10))):
            decision_id = digest(legacy)
            with locked(engine.config):
                put(engine.config, "decisions", decision_id, canonical(legacy) + b"\n")
            loaded = load(engine.config, decision_id)
            assert loaded["format"] == f"howhard-decision-{version}" and digest(loaded) == decision_id
            # The older meaning is kept as recorded: a policy, no tiers, and no draw under an unmet cap.
            selection = loaded["selection"]
            assert selection["on_cap_unmet"] == on_cap_unmet and "pool" not in selection["exploration"]
            assert not any({"pi", "tier"} & c.keys() for c in selection["cells"])
            assert "trial_probability" not in selection["sampling_policy"]
            assert loaded["capability"]["widening"] and all(
                w["trigger"] == "silent-cap-unmet" for w in loaded["capability"]["widening"])
            if label == "met":
                assert selection["cap_met"] and loaded["capability"]["widening_stop"] == "cap-met"
                assert loaded["semantic"]["sampling"]["enabled"]
            else:
                assert not selection["cap_met"] and selection["needs_review"]
                assert not loaded["semantic"]["sampling"]["enabled"] and "sampled_expert" not in loaded["semantic"]
                assert selection["exploration"]["reason"] == "disabled-cap-unmet"
            stored[label, version] = decision_id
    for index, ((label, version), decision_id) in enumerate(stored.items()):
        outcome = "clean" if index % 2 else "loud"
        extra = {} if outcome == "clean" else {"cause": "capability"}
        engine.observe_decision(decision_id, outcome, id=f"from-{label}-{version}", active_seconds=10., **extra)
    beliefs = engine.memory_beliefs()
    assert beliefs["unscored_runs"] == 0
    assert next(e for e in beliefs["experts"] if e["key"] == "llm")["evidence_count"] == len(stored)
    stats = report(engine)
    assert stats["unsupported_receipts"] == [] and stats["forecast_metrics"]["llm"]["runs"] == len(stored)
    assert {r["decision_id"] for r in stats["routing"]} >= set(stored.values())
    # A later decision replays those outcomes as evidence and still validates.
    later = widened("met", 1, 6)
    assert {o["outcome"] for o in later["semantic"]["outcome_observations"]} == {"clean", "loud"}
    load(engine.config, later["decision_id"])

    # A v10 receipt cannot borrow the tier fields or drop its own.
    v10 = load(engine.config, stored["unmet", "v10"])
    for tamper in (lambda r: r["selection"]["cells"][0].update(pi=.1, tier="excluded"),
                   lambda r: r["selection"].pop("on_cap_unmet"),
                   lambda r: r["selection"]["sampling_policy"].update(trial_probability=.2, all_excluded_factor=.5),
                   lambda r: r["capability"]["widening"][0].update(trigger="all-excluded")):
        bad = deepcopy(v10)
        tamper(bad)
        with pytest.raises(HowHardError):
            load(engine.config, bad)
