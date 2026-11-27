"""Cost to an accepted result averaged over the beliefs, its exact spread and the full draw (#135).

Every decision here is routed through the public calling-agent interface. Expected numbers
come from the selection contract's formulas written out in plain Python over the beliefs a
decision stores, never from the package's own cost code. Fixtures are synthetic and own
their state: prices, forecasts and earlier runs are test inputs, not provider claims.
"""

from copy import deepcopy
from functools import lru_cache
import math
import statistics

import pytest

from howhard import Engine
from howhard.cli import main
from howhard.context import canonical, digest
from howhard.decisions import load
from howhard.payload_store import put
from howhard.report import report
from howhard.storage import locked
from howhard.success import draw, uniforms
from howhard.validation import HowHardError
from legacy_receipts import to_v7, to_v8, to_v9, to_v10
from test_amount_beliefs import by_kind, decide, stated
from test_evidence_similarity import Scripted
from test_execution_accounting import artifact, card, codex_events, counts

# The packaged standard risk class.
LAMBDA, C_LOUD, C_SILENT = 25., 50., 1000.
RATES = counts(input=2, output=10, cache_read=.2, cache_write=3, cache_write_5m=2.5, cache_write_1h=4)
USAGE = counts(input=600_000, output=200_000, cache_read=400_000)   # 3.28 of credit at RATES
Q = .5                                                               # cost_per_credit of every fixture cell
A, B, C, D = "fixture/a", "fixture/b", "fixture/c", "fixture/d"


def worths(value):
    """A sampling policy in which every prediction is worth ``value / 2`` runs, so beliefs start narrow."""
    return "".join(f"{key} = {value}\n" for key in ("model_worth", "task_worth_silent", "task_worth_loud", "task_worth_amount"))


def engine_at(directory, *, sampling="", classes=""):
    """An engine over its own project; ``sampling`` and ``classes`` are optional configuration text."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "howhard.toml"
    path.write_text('[task_types]\ntask = "standard"\n' + classes + ("[sampling]\n" + sampling if sampling else ""))
    return Engine(path, retrieval=Scripted())


def priced(model, **extra):
    return {"model": model, "effort": "high", "rate_card": card(model=model, rates=RATES), "cost_per_credit": Q, **extra}


def forecast(seconds=1800, clean=.78, loud=.2, silent=.02, **extra):
    return {**by_kind(USAGE, seconds=seconds), "clean": clean, "loud": loud, "silent": silent, **extra}


def route(directory, rows, *, sampling="", classes="", seed=7, candidates=None):
    """One decision over the cells of ``rows`` (model to forecast row); returns engine, request, response, result."""
    engine = engine_at(directory, sampling=sampling, classes=classes)
    cells = candidates or [priced(model) for model in rows]
    request_id, response, result = decide(engine, cells, {(model, "high"): row for model, row in rows.items()}, seed=seed)
    return engine, request_id, response, result


def cell(result, model):
    return next(c for c in result["cells"] if c["model"] == model)


def drawn(result, model):
    row = cell(result, model)
    return next(c for c in result["semantic"]["sampling"]["cells"] if (c["family"], c["effort"]) == (row["family"], row["effort"]))


@lru_cache(maxsize=None)
def grid_belief(prediction, spread, n_yes, n_no):
    """A rate belief on the grid of log-odds ``x_j = -16 + 0.01 j``: ``[(x_j, share of the belief)]``."""
    xs = [-16 + .01 * j for j in range(2501)]
    rates = [1 / (1 + math.exp(-x)) for x in xs]

    def bell(m):
        q = [math.exp(-(x - m) ** 2 / (2 * spread ** 2)) for x in xs]
        total = math.fsum(q)
        return [v / total for v in q]
    lo, hi = -25., 25.
    for _ in range(60):
        mid = (lo + hi) / 2
        if math.fsum(q * r for q, r in zip(bell(mid), rates)) < prediction:
            lo = mid
        else:
            hi = mid
    logs = [math.log(q) + n_yes * math.log(r) + n_no * math.log(1 - r) if q > 0 else -math.inf
            for q, r in zip(bell((lo + hi) / 2), rates)]
    peak = max(logs)
    b = [math.exp(v - peak) for v in logs]
    total = math.fsum(b)
    return tuple((x, v / total) for x, v in zip(xs, b))


def on_grid(belief):
    return grid_belief(belief["prediction"], belief["spread"], *belief["counts"])


def contract_cost(result, model, *, q=Q, rates=RATES):
    """``J``, ``E[R]`` and ``sd`` of one cell from the contract's formulas over its stored beliefs."""
    row, beliefs = cell(result, model), drawn(result, model)["outcome_beliefs"]
    loud, share = on_grid(beliefs["loud"]), on_grid(beliefs["silent"])
    attempts = math.fsum(b * (1 + math.exp(u)) for u, b in loud)                 # E[R]
    attempts_sq = math.fsum(b * (1 + math.exp(u)) ** 2 for u, b in loud)         # E[R^2]
    defect = math.fsum(b / (1 + math.exp(-v)) for v, b in share)                 # E[S]
    defect_var = math.fsum(b / (1 + math.exp(-v)) ** 2 for v, b in share) - defect ** 2
    time = row["active_time"]
    hours = math.exp(time["centre"] + time["spread"] ** 2 / 2) / 3600            # E[T]
    hours_var = hours ** 2 * (math.exp(time["spread"] ** 2) - 1)
    kinds = [(rates[kind] * math.exp(b["centre"] + b["spread"] ** 2 / 2) / 1e6, b["spread"])
             for kind, b in row["token_beliefs"].items() if b["centre"] is not None]
    credit = math.fsum(c for c, _ in kinds)                                      # E[K]
    credit_var = math.fsum(ca * cb * (math.exp(sa * sb) - 1) for ca, sa in kinds for cb, sb in kinds)
    run = q * credit                                                             # C_run
    j = (run + LAMBDA * hours) * attempts + C_LOUD * (attempts - 1) + C_SILENT * defect
    x = run + LAMBDA * hours + C_LOUD                                            # E[X]
    x_sq = q ** 2 * credit_var + LAMBDA ** 2 * hours_var + x ** 2                # E[X^2]
    variance = x_sq * attempts_sq - (x * attempts) ** 2 + C_SILENT ** 2 * defect_var
    return {"J": j, "expected_attempts": attempts, "cost_sd": math.sqrt(variance), "C_run": run, "T": hours}


def point_cost(run, hours, loud, share):
    """The cost at single values: every attempt costs its credit and time, and a loud run is retried."""
    return (run + LAMBDA * hours) / (1 - loud) + C_LOUD * loud / (1 - loud) + C_SILENT * share


# --- 1. The cost averaged over the beliefs and its exact spread -----------------------------

def test_cost_and_its_spread_follow_the_formulas_over_the_cells_beliefs(tmp_path):
    engine = engine_at(tmp_path)
    cells = [priced(A)]
    row = forecast(seconds_ci=[200, 20_000], token_usage_ci={"output": [20_000, 2_000_000]}, loud_ci=[.05, .6])
    decide(engine, cells, {(A, "high"): row})
    # A counted run of the cell reshapes every belief before the cost is read.
    engine.observe("task", A, "high", "clean", native=True, task_text=stated(.6), id="earlier", active_seconds=2400.)
    manifest = artifact(tmp_path, codex_events(), execution_id="earlier", model=A, completed=True)
    engine.attach_usage("earlier", manifest, id="earlier-usage")
    _, _, result = decide(engine, cells)

    stored, expected = cell(result, A), contract_cost(result, A)
    assert stored["active_time"]["evidence_weight"] > .5 and stored["token_beliefs"]["output"]["evidence_weight"] > .5
    assert drawn(result, A)["outcome_beliefs"]["loud"]["counts"][1] > .5
    assert stored["C_run"] == pytest.approx(expected["C_run"]) and stored["T"] == pytest.approx(expected["T"])
    assert stored["J"] == pytest.approx(expected["J"], rel=1e-9)
    assert stored["expected_attempts"] == pytest.approx(expected["expected_attempts"], rel=1e-9)
    assert drawn(result, A)["cost_sd"] == pytest.approx(expected["cost_sd"], rel=1e-9)
    # The retry factor is curved, so the average over the belief charges more than the cost at the averages.
    at_averages = point_cost(stored["C_run"], stored["T"], stored["P_loud"]["mean"], stored["P_silent_accepted"]["mean"])
    assert stored["expected_attempts"] > 1 / (1 - stored["P_loud"]["mean"])
    assert stored["J"] > at_averages
    assert load(engine.config, result["decision_id"])["selection"]["cells"][0]["J"] == stored["J"]


def test_money_enters_at_cost_per_credit_and_not_at_all_when_a_price_is_unknown(tmp_path):
    _, _, _, paid = route(tmp_path / "paid", {A: forecast()})
    _, _, _, free = route(tmp_path / "free", {A: forecast()}, candidates=[priced(A, cost_per_credit=0.)])
    unpriced = [{"model": A, "effort": "high"}]
    engine, _, _, unknown = route(tmp_path / "unknown", {A: forecast()}, candidates=unpriced)
    assert cell(paid, A)["C_run"] == pytest.approx(Q * 3.28)
    assert cell(free, A)["C_run"] == 0. and cell(unknown, A)["C_run"] is None
    assert not unknown["cost_policy"]["worker_money_included"]
    for result, q in ((paid, Q), (free, 0.), (unknown, 0.)):
        expected = contract_cost(result, A, q=q)
        assert cell(result, A)["J"] == pytest.approx(expected["J"], rel=1e-9)
        assert drawn(result, A)["cost_sd"] == pytest.approx(expected["cost_sd"], rel=1e-9)
    # A free or unpriced run costs time and risk only, and its credit adds no spread.
    assert cell(free, A)["J"] == pytest.approx(cell(unknown, A)["J"]) and cell(free, A)["J"] < cell(paid, A)["J"]
    assert drawn(free, A)["cost_sd"] == pytest.approx(drawn(unknown, A)["cost_sd"])
    assert drawn(free, A)["cost_sd"] < drawn(paid, A)["cost_sd"]
    assert drawn(unknown, A)["sampled"]["credit"] is None and drawn(free, A)["sampled"]["credit"] > 0
    load(engine.config, unknown["decision_id"])


def test_as_every_spread_shrinks_the_cost_tends_to_the_cost_at_the_centres_and_its_spread_to_zero(tmp_path):
    at_centres = point_cost(Q * 3.28, .5, .2, .02 / .8)
    gaps, spreads = [], []
    for worth in (5, 100, 10_000, 1_000_000):
        _, _, _, result = route(tmp_path / str(worth), {A: forecast()}, sampling=worths(worth))
        beliefs = drawn(result, A)["outcome_beliefs"]
        widest = max(beliefs["loud"]["spread"], beliefs["silent"]["spread"], cell(result, A)["active_time"]["spread"],
                     *(b["spread"] for b in cell(result, A)["token_beliefs"].values()))
        assert widest < 10 / math.sqrt(worth)
        gaps.append(cell(result, A)["J"] - at_centres)
        spreads.append(drawn(result, A)["cost_sd"])
    assert gaps == sorted(gaps, reverse=True) and spreads == sorted(spreads, reverse=True)
    assert gaps[0] > 10 and 0 <= gaps[-1] < 1e-3 * at_centres
    assert spreads[0] > at_centres and spreads[-1] < .01 * at_centres


def test_wider_loud_belief_at_the_same_average_charges_more_attempts_and_a_higher_cost(tmp_path):
    engine, _, _, result = route(tmp_path, {A: forecast(), B: forecast(loud_ci=[.01, .9])})
    narrow, wide = cell(result, A), cell(result, B)
    assert drawn(result, B)["outcome_beliefs"]["loud"]["spread"] > drawn(result, A)["outcome_beliefs"]["loud"]["spread"]
    assert wide["P_loud"]["mean"] == pytest.approx(.2, abs=1e-9) and narrow["P_loud"]["mean"] == pytest.approx(.2, abs=1e-9)
    assert (wide["C_run"], wide["T"]) == (narrow["C_run"], narrow["T"])
    assert wide["P_silent_accepted"] == narrow["P_silent_accepted"]
    assert wide["expected_attempts"] > narrow["expected_attempts"] > 1 / (1 - .2)
    assert wide["J"] > narrow["J"]
    # The whole difference is the retry factor: (C_run + lambda T + C_loud) more for each extra expected attempt.
    per_attempt = narrow["C_run"] + LAMBDA * narrow["T"] + C_LOUD
    assert wide["J"] - narrow["J"] == pytest.approx(per_attempt * (wide["expected_attempts"] - narrow["expected_attempts"]))
    load(engine.config, result["decision_id"])


@pytest.mark.parametrize("widened, factor", [({"seconds_ci": [20, 200_000]}, 1.5),
                                             ({"token_usage_ci": {"output": [2_000, 20_000_000]}}, 1.03)],
                         ids=["active-time", "token-kind"])
def test_widening_one_cells_stated_range_raises_its_spread_and_no_other_cells_numbers(tmp_path, widened, factor):
    _, _, _, before = route(tmp_path / "before", {A: forecast(), B: forecast(seconds=2400)})
    engine, _, _, after = route(tmp_path / "after", {A: forecast(**widened), B: forecast(seconds=2400)})
    # Credit is a small part of this cell's cost, so its wider belief adds less spread than a wider active time.
    assert drawn(after, A)["cost_sd"] > factor * drawn(before, A)["cost_sd"]
    # A belief averages to its prediction however wide it is, so the cell's cost stays where it was.
    assert cell(after, A)["J"] == pytest.approx(cell(before, A)["J"])
    assert cell(after, B) == cell(before, B) and drawn(after, B) == drawn(before, B)
    assert drawn(after, A)["cost_sd"] == pytest.approx(contract_cost(after, A)["cost_sd"], rel=1e-9)
    load(engine.config, after["decision_id"])


# --- 2. Eligibility -------------------------------------------------------------------------

def competing_rows(**second):
    """Four cells: the best, one a little dearer, one much dearer, and one with too many accepted defects."""
    ordinary = {"clean": .93, "loud": .05, "silent": .02}
    return {A: forecast(3600, **ordinary), B: forecast(4000, **{**ordinary, **second}),
            C: forecast(9000, **ordinary), D: forecast(3600, clean=.5, loud=.05, silent=.45)}


def test_a_cell_is_eligible_exactly_when_its_cost_overlaps_the_best_within_their_spreads(tmp_path):
    engine, _, _, result = route(tmp_path, competing_rows(), sampling=worths(200))
    assert result["sampling_policy"]["uncertainty_multiplier"] == 1 and "explore_within" not in result["constants"]
    assert cell(result, D)["tier"] == "excluded"
    pool = [c for c in result["cells"] if c["tier"] != "excluded"]
    best = min(pool, key=lambda c: c["J"])
    sd = lambda c: drawn(result, c["model"])["cost_sd"]
    overlapping = {c["model"] for c in pool if c["J"] - best["J"] <= sd(best) + sd(c)}
    assert {c["model"] for c in result["exploration"]["eligible_cells"]} == overlapping == {A, B}
    assert result["exploration"]["mean_best"]["model"] == best["model"] == A
    # The third cell is in the pool and loses on the gap alone.
    gap = cell(result, C)["J"] - best["J"]
    assert cell(result, C)["tier"] == "trusted" and gap > sd(best) + sd(cell(result, C))
    assert 0 < cell(result, B)["J"] - best["J"] <= sd(best) + sd(cell(result, B))
    assert result["chosen"]["model"] in overlapping
    load(engine.config, result["decision_id"])


def test_a_class_explore_within_caps_the_gap_at_that_fraction_of_the_best_cost(tmp_path):
    # The cap can only tighten the overlap rule: a generous fraction leaves the overlapping cells as they were.
    for name, fraction, eligible in (("tight", .01, {A}), ("loose", .10, {A, B}), ("generous", 5., {A, B})):
        engine, _, _, result = route(tmp_path / name, competing_rows(), sampling=worths(200),
                                     classes=f"[classes.standard]\nexplore_within = {fraction}\n")
        assert result["constants"]["explore_within"] == fraction == result["exploration"]["relative_band"]
        costs = {model: cell(result, model)["J"] for model in (A, B, C)}
        # B is 5.7% dearer than the best cell and overlaps it; C is 76% dearer and does not.
        assert .05 < costs[B] / costs[A] - 1 < .06 and .7 < costs[C] / costs[A] - 1 < .8
        assert {c["model"] for c in result["exploration"]["eligible_cells"]} == eligible
        assert result["exploration"]["enabled"] == (len(eligible) > 1)
        if len(eligible) == 1:
            assert result["chosen"]["model"] == A
        load(engine.config, result["decision_id"])


# --- 3. The draw ----------------------------------------------------------------------------

def replay(receipt, model, seed, *, position=0, count=1):
    """One cell's draw replayed from a stored receipt with the receipt's own beliefs and prices."""
    row = next(c for c in receipt["selection"]["cells"] if c["model"] == model)
    beliefs = next(c for c in receipt["semantic"]["sampling"]["cells"]
                   if (c["family"], c["effort"]) == (row["family"], row["effort"]))["outcome_beliefs"]
    candidate = next(c for c in receipt["native_candidates"] if c["model"] == model)
    q = candidate["cost_per_credit"] if receipt["cost_policy"]["worker_money_included"] else 0.
    numbers = uniforms(seed, 4 * count)[4 * position:4 * position + 4]
    return draw(numbers, beliefs, row["active_time"], row["token_beliefs"], candidate.get("rate_card"),
                candidate.get("usd_per_mtok"), q, receipt["selection"]["constants"])


def test_uniform_numbers_come_from_raw_pcg64_output_and_stay_strictly_inside_the_unit_interval():
    import numpy
    raw = [int(r) for r in numpy.random.PCG64(2026).random_raw(8)]
    assert uniforms(2026, 8) == [((r >> 12) + .5) / 2 ** 52 for r in raw]
    assert uniforms(2026, 4) == uniforms(2026, 8)[:4] != uniforms(2027, 4)
    assert all(0 < u < 1 for seed in range(50) for u in uniforms(seed, 8))
    # The seed may be any non-negative integer, including one longer than 64 bits.
    assert len(set(uniforms(2 ** 80 + 1, 4))) == 4


def test_each_cell_draws_one_value_of_everything_and_the_sampled_cost_is_the_sum_of_its_parts(tmp_path):
    engine, _, _, result = route(tmp_path, {A: forecast(), B: forecast(seconds=2400)}, seed=11)
    receipt = load(engine.config, result["decision_id"])
    numbers = uniforms(11, 8)
    for position, sampled_cell in enumerate(receipt["semantic"]["sampling"]["cells"]):
        row = next(c for c in receipt["selection"]["cells"] if c["family"] == sampled_cell["family"])
        u_share, u_loud, u_time, u_credit = numbers[4 * position:4 * position + 4]
        sampled, beliefs = sampled_cell["sampled"], sampled_cell["outcome_beliefs"]
        assert set(sampled_cell) == {"family", "effort", "outcome_beliefs", "probabilities", "cost_sd", "sampled"}
        assert set(sampled) == {"S", "u", "seconds", "credit"}

        def first_reaching(belief, level):
            running = 0.
            for x, share in on_grid(belief):
                running += share
                if running >= level:
                    return x
            return 9.
        # S and u are read at the first grid point where the running total of the belief reaches U.
        assert sampled["S"] == pytest.approx(1 / (1 + math.exp(-first_reaching(beliefs["silent"], u_share))))
        assert sampled["u"] == pytest.approx(first_reaching(beliefs["loud"], u_loud))
        z = statistics.NormalDist().inv_cdf
        time = row["active_time"]
        assert sampled["seconds"] == pytest.approx(math.exp(time["centre"] + time["spread"] * z(u_time)))
        # One U_K serves every token kind, and the counts are priced as their averages are.
        tokens = {kind: math.exp(b["centre"] + b["spread"] * z(u_credit)) for kind, b in row["token_beliefs"].items()
                  if b["centre"] is not None}
        assert set(tokens) == {"input", "output", "cache_read"}
        assert sampled["credit"] == pytest.approx(sum(RATES[kind] * n for kind, n in tokens.items()) / 1e6)
        attempts = 1 + math.exp(sampled["u"])
        parts = {"run": Q * sampled["credit"] * attempts, "time": LAMBDA * sampled["seconds"] / 3600 * attempts,
                 "loud": C_LOUD * (attempts - 1), "silent": C_SILENT * sampled["S"]}
        assert row["sampled_J"] == pytest.approx(sum(parts.values()))
        again = replay(receipt, row["model"], 11, position=position, count=2)
        assert again["sampled"] == sampled and again["sampled_J"] == row["sampled_J"]
        assert again["parts"] == pytest.approx(parts)


def test_sampled_cost_over_twenty_thousand_seeds_agrees_with_the_cost_and_its_spread(tmp_path):
    # Each prediction is worth 50 runs, and the stated ranges on active time and output tokens state spreads near 0.5.
    row = forecast(clean=.7, loud=.2, silent=.1, seconds_ci=[700, 4900], token_usage_ci={"output": [80_000, 500_000]})
    engine, request_id, response, result = route(tmp_path, {A: row}, sampling=worths(100), seed=0)
    receipt = load(engine.config, result["decision_id"])
    stored, sampling = receipt["selection"]["cells"][0], receipt["semantic"]["sampling"]["cells"][0]
    spreads = [sampling["outcome_beliefs"]["loud"]["spread"], sampling["outcome_beliefs"]["silent"]["spread"],
               stored["active_time"]["spread"], *(b["spread"] for b in stored["token_beliefs"].values())]
    assert .4 < max(spreads) <= .5 and stored["active_time"]["spread"] > .45
    # The replay over a stored receipt is the draw a decision makes with that seed.
    for seed in (0, 1, 2):
        decided = engine.decide(request_id, response, seed=seed)
        assert replay(receipt, A, seed)["sampled_J"] == decided["cells"][0]["sampled_J"]
        assert replay(receipt, A, seed)["sampled"] == decided["semantic"]["sampling"]["cells"][0]["sampled"]
    costs = [replay(receipt, A, seed)["sampled_J"] for seed in range(20_000)]
    assert statistics.fmean(costs) == pytest.approx(stored["J"], rel=.05)
    assert statistics.pstdev(costs) == pytest.approx(sampling["cost_sd"], rel=.05)
    assert sampling["cost_sd"] > .2 * stored["J"]


def test_same_seed_repeats_the_draw_and_a_thousand_seeds_try_the_wide_cell_and_never_an_ineligible_one(tmp_path):
    engine, request_id, response, result = route(tmp_path, competing_rows(seconds_ci=[40, 400_000]),
                                                 sampling=worths(200), seed=5)
    assert {c["model"] for c in result["exploration"]["eligible_cells"]} == {A, B}
    assert result["exploration"]["mean_best"]["model"] == A
    assert drawn(result, B)["cost_sd"] > 10 * drawn(result, A)["cost_sd"]
    again = engine.decide(request_id, response, seed=5)
    assert again["chosen"] == result["chosen"]
    assert again["semantic"]["sampling"]["cells"] == result["semantic"]["sampling"]["cells"]
    assert [c["sampled_J"] for c in again["cells"]] == [c["sampled_J"] for c in result["cells"]]
    other = engine.decide(request_id, response, seed=6)
    assert [c["sampled"] for c in other["semantic"]["sampling"]["cells"]] != [
        c["sampled"] for c in result["semantic"]["sampling"]["cells"]]

    chosen, lowest = [], []
    for seed in range(1000):
        decided = engine.decide(request_id, response, seed=seed)
        assert decided["seed"] == seed and "sampled_expert" not in decided["semantic"]
        chosen.append(decided["chosen"]["model"])
        lowest.append(min(decided["cells"], key=lambda c: c["sampled_J"])["model"])
    # The wide cell is dearer on average, yet its belief sometimes draws the lower cost.
    assert set(chosen) == {A, B} and chosen.count(B) >= 1 and chosen.count(A) >= 1
    # An ineligible cell draws all the same, and is never chosen even when its sampled cost is the lowest.
    assert C not in chosen and D not in chosen
    assert C in lowest


def test_the_stream_of_numbers_does_not_depend_on_which_cells_compete(tmp_path):
    rows = competing_rows()
    _, _, _, pooled = route(tmp_path / "pooled", rows, sampling=worths(200), seed=9)
    short = [priced(model, credit_budget={"available_credit_usd": .01}) if model == A else priced(model) for model in rows]
    engine, _, _, limited = route(tmp_path / "limited", rows, sampling=worths(200), seed=9, candidates=short)
    assert not cell(limited, A)["credit_ok"] and cell(pooled, A)["credit_ok"]
    assert A not in {c["model"] for c in limited["exploration"]["eligible_cells"]}
    # Every forecast cell draws in the stored order, in the pool or not, so no cell's draw moves.
    for model in rows:
        assert drawn(limited, model)["sampled"] == drawn(pooled, model)["sampled"]
        assert cell(limited, model)["sampled_J"] == cell(pooled, model)["sampled_J"]
    load(engine.config, limited["decision_id"])


def test_cheaper_alternative_is_explained_by_the_four_sampled_parts(tmp_path):
    rows = {A: forecast(3600, clean=.93, loud=.05, silent=.02), B: forecast(3600, clean=.93, loud=.05, silent=.02)}
    cells = [priced(A), priced(B, cost_per_credit=2.)]
    for seed in range(40):
        engine, _, _, result = route(tmp_path / str(seed), rows, sampling=worths(200), seed=seed, candidates=cells)
        if result["chosen"]["model"] == B:
            break
    assert result["chosen"]["model"] == B and result["exploration"]["reason"] == "sampled-within-posterior-overlap"
    why = next(w for w in result["why"] if w["model"] == A)
    assert why["reason"] == "sampled-cost" and set(why["components"]) == {"run", "time", "loud", "silent"}
    receipt = load(engine.config, result["decision_id"])
    order = [c["family"] for c in receipt["semantic"]["sampling"]["cells"]]
    parts = {model: replay(receipt, model, seed, position=order.index(cell(result, model)["family"]), count=2)["parts"]
             for model in (A, B)}
    for name, difference in why["components"].items():
        assert difference == pytest.approx(parts[A][name] - parts[B][name])
    gap = cell(result, A)["sampled_J"] - cell(result, B)["sampled_J"]
    assert gap > 0 and sum(why["components"].values()) == pytest.approx(gap)


def test_summary_makes_no_draw(tmp_path):
    engine = engine_at(tmp_path)
    summary = engine.summary("task", "Repair rollback ordering", native_candidates=[{"model": A, "effort": "high"}])
    assert "seed" not in summary and "sampling" not in summary["semantic"]
    assert all("sampled_J" not in c and c["J"] > 0 and c["expected_attempts"] > 1 for c in summary["cells"])


# --- 4. The receipt ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stored_receipt(tmp_path_factory):
    """One loaded receipt with two eligible cells for every alteration below; each test alters its own copy."""
    engine, _, _, result = route(tmp_path_factory.mktemp("stored"), competing_rows(), sampling=worths(200), seed=3)
    assert len(result["exploration"]["eligible_cells"]) == 2
    return engine, load(engine.config, result["decision_id"])


def test_v11_receipt_draws_no_expert_and_loads_from_its_content(stored_receipt):
    engine, receipt = stored_receipt
    assert receipt["format"] == "howhard-decision-v11" and "sampled_expert" not in receipt["semantic"]
    assert all("expert" not in c for c in receipt["semantic"]["sampling"]["cells"])
    assert load(engine.config, deepcopy(receipt)) == receipt


def swap_chosen(receipt):
    """Name the other eligible cell as chosen, consistently with every field that follows the choice."""
    other = next(c for c in receipt["selection"]["cells"]
                 if c["model"] in (A, B) and c["model"] != receipt["chosen"]["model"])
    receipt["chosen"] = {k: other[k] for k in ("model", "family", "effort")}
    executed = next(c for c in receipt["capability"]["cells"] if c["model"] == other["model"])
    receipt["capability"]["executed"] = {k: executed[k] for k in ("key", "classification", "position", "P_clear")}


def first(receipt):
    return receipt["selection"]["cells"][0]


def sampled(receipt):
    return receipt["semantic"]["sampling"]["cells"][0]["sampled"]


TAMPER = [
    (lambda r: first(r).update(J=first(r)["J"] * 1.001), r"decision\.selection: J does not match"),
    (lambda r: first(r).update(expected_attempts=1 / (1 - first(r)["P_loud"]["mean"])), "expected_attempts"),
    (lambda r: r["semantic"]["sampling"]["cells"][0].update(cost_sd=r["semantic"]["sampling"]["cells"][0]["cost_sd"] * 1.001),
     "cost_sd"),
    (lambda r: sampled(r).update(S=sampled(r)["S"] + .001), r"sampled\.S"),
    (lambda r: sampled(r).update(u=sampled(r)["u"] + .01), r"sampled\.u"),
    (lambda r: sampled(r).update(seconds=sampled(r)["seconds"] * 1.001), r"sampled\.seconds"),
    (lambda r: sampled(r).update(credit=sampled(r)["credit"] * 1.001), r"sampled\.credit"),
    (lambda r: sampled(r).update(credit=None), r"sampled\.credit"),
    (lambda r: sampled(r).pop("u"), "sampled"),
    (lambda r: r["semantic"]["sampling"]["cells"][0].pop("sampled"), "sampled"),
    (lambda r: first(r).update(sampled_J=first(r)["sampled_J"] * 1.001), "sampled_J"),
    (lambda r: r["selection"]["exploration"]["eligible_cells"].pop(), r"exploration\.eligible_cells"),
    (lambda r: r["selection"]["exploration"]["eligible_cells"].append(
        {k: c[k] for c in r["selection"]["cells"] if c["model"] == C for k in ("model", "family", "effort")}),
     r"exploration\.eligible_cells"),
    (swap_chosen, r"decision\.chosen: does not match the draw replayed from seed"),
    # Another seed is another draw, and reordering the drawn cells moves every cell's numbers.
    (lambda r: r.update(seed=r["seed"] + 1), "replayed from seed"),
    (lambda r: r["semantic"]["sampling"]["cells"].reverse(), "replayed from seed"),
    (lambda r: r["semantic"].update(sampled_expert="llm"), "sampled_expert"),
    (lambda r: r["semantic"]["sampling"]["cells"][0].update(expert="llm"), "expert"),
]


@pytest.mark.parametrize("index", range(len(TAMPER)))
def test_receipt_with_an_altered_cost_spread_sampled_value_or_choice_is_refused(stored_receipt, index):
    engine, receipt = stored_receipt
    receipt = deepcopy(receipt)
    change, match = TAMPER[index]
    change(receipt)
    with pytest.raises(HowHardError, match=match):
        load(engine.config, receipt)


# --- 5. Token beliefs count a run with complete usage and no active time ---------------------

def test_run_with_complete_usage_and_no_active_time_moves_the_token_beliefs_and_not_active_time(tmp_path):
    engine = engine_at(tmp_path)
    cells = [priced(A)]
    _, _, before = decide(engine, cells, {(A, "high"): by_kind(counts(input=6000, output=2000, cache_read=4000))})
    # The run records neither an active time nor a total time; its owned usage is complete.
    engine.observe("task", A, "high", "clean", native=True, task_text=stated(.5), id="usage-only")
    manifest = artifact(tmp_path, codex_events(), execution_id="usage-only", model=A, completed=True)
    engine.attach_usage("usage-only", manifest, id="usage-only-usage")
    _, _, after = decide(engine, cells)

    candidate = after["semantic"]["evidence_candidates"]["candidates"][0]
    assert candidate["active_seconds"] is None and candidate["token_usage"] == counts(input=60, output=20, cache_read=40)
    [observed] = after["semantic"]["duration_observations"]
    weight = after["semantic"]["outcome_observations"][0]["weight"]
    assert observed["active_seconds"] is None and observed["token_usage"] == candidate["token_usage"]
    assert observed["weight"] == weight == pytest.approx(.5, rel=1e-4)
    row, start = cell(after, A), cell(before, A)
    assert row["active_time"] == start["active_time"] and row["active_time"]["evidence_weight"] == 0. and row["T"] == start["T"]
    tau = 1.3 * math.sqrt(.5)
    for kind, count in (("input", 60), ("output", 20), ("cache_read", 40)):
        belief = row["token_beliefs"][kind]
        assert belief["evidence_weight"] == weight
        variance = 1 / (1 / belief["prior_spread"] ** 2 + weight / tau ** 2)
        assert belief["spread"] == pytest.approx(math.sqrt(variance)) and belief["spread"] < start["token_beliefs"][kind]["spread"]
        assert belief["centre"] == pytest.approx(
            belief["prior_centre"] + variance * weight * (math.log(count) - belief["prior_centre"]) / tau ** 2)
        assert belief["expected_tokens"] < start["token_beliefs"][kind]["expected_tokens"]
    assert row["needed_credit_usd"] < start["needed_credit_usd"]

    # The receipt validation recomputes the same rows from the scored candidates.
    receipt = load(engine.config, after["decision_id"])
    for change, match in (
            (lambda r: r["semantic"]["duration_observations"].clear(), "duration_observations"),
            (lambda r: r["semantic"]["duration_observations"][0].update(active_seconds=600.), "duration_observations"),
            (lambda r: r["selection"]["cells"][0]["token_beliefs"]["input"].update(evidence_weight=0.), r"token_beliefs\.input")):
        bad = deepcopy(receipt)
        change(bad)
        with pytest.raises(HowHardError, match=match):
            load(engine.config, bad)


def test_run_with_only_a_total_token_count_moves_the_total_tokens_belief_alone(tmp_path):
    engine = engine_at(tmp_path)
    # Priced per total token with no usage by kind: the cell has one token belief, for total tokens.
    cells = [{"model": A, "effort": "high", "usd_per_mtok": 10.}]
    row = {"clean": .98, "loud": .01, "silent": .01, "tokens": 10_000, "seconds": 3600}
    _, _, before = decide(engine, cells, {(A, "high"): row})
    # The run records a total token count, and neither an active time, a total time nor usage by kind.
    engine.observe("task", A, "high", "clean", native=True, task_text=stated(.5), id="total-only", tokens=1000)
    _, _, after = decide(engine, cells)

    candidate = after["semantic"]["evidence_candidates"]["candidates"][0]
    assert (candidate["active_seconds"], candidate["tokens"], candidate["token_usage"]) == (None, 1000, None)
    [observed] = after["semantic"]["duration_observations"]
    weight = after["semantic"]["outcome_observations"][0]["weight"]
    assert (observed["active_seconds"], observed["tokens"], observed["token_usage"]) == (None, 1000, None)
    assert observed["weight"] == weight == pytest.approx(.5, rel=1e-4)
    stored, start = cell(after, A), cell(before, A)
    assert stored["active_time"] == start["active_time"] and stored["active_time"]["evidence_weight"] == 0.
    assert list(stored["token_beliefs"]) == ["tokens"]
    belief, tau = stored["token_beliefs"]["tokens"], 1.3 * math.sqrt(.5)
    assert belief["evidence_weight"] == weight
    variance = 1 / (1 / belief["prior_spread"] ** 2 + weight / tau ** 2)
    assert belief["spread"] == pytest.approx(math.sqrt(variance)) and belief["spread"] < start["token_beliefs"]["tokens"]["spread"]
    assert belief["centre"] == pytest.approx(
        belief["prior_centre"] + variance * weight * (math.log(1000) - belief["prior_centre"]) / tau ** 2)
    assert stored["needed_credit_usd"] < start["needed_credit_usd"] == pytest.approx(.1)

    # The receipt validation recomputes the same row from the scored candidates.
    receipt = load(engine.config, after["decision_id"])
    for change, match in (
            (lambda r: r["semantic"]["duration_observations"].clear(), "duration_observations"),
            (lambda r: r["semantic"]["duration_observations"][0].update(tokens=5000), "duration_observations"),
            (lambda r: r["selection"]["cells"][0]["token_beliefs"]["tokens"].update(evidence_weight=0.), r"token_beliefs\.tokens")):
        bad = deepcopy(receipt)
        change(bad)
        with pytest.raises(HowHardError, match=match):
            load(engine.config, bad)
    # An older receipt read active time alone, so this run has no row there.
    assert load(engine.config, to_v10(receipt))["semantic"]["duration_observations"] == []


def test_total_token_count_moves_no_token_kind_belief_when_usage_is_forecast_by_kind(tmp_path):
    engine = engine_at(tmp_path)
    cells = [priced(A)]
    _, _, before = decide(engine, cells, {(A, "high"): by_kind(counts(input=6000, output=2000, cache_read=4000))})
    engine.observe("task", A, "high", "clean", native=True, task_text=stated(.5), id="total-only", tokens=1000)
    _, _, after = decide(engine, cells)
    # The run has its reader row, and no belief of this cell reads a total token count.
    [observed] = after["semantic"]["duration_observations"]
    assert (observed["active_seconds"], observed["tokens"], observed["token_usage"]) == (None, 1000, None)
    assert set(cell(after, A)["token_beliefs"]) == set(counts())
    assert cell(after, A)["token_beliefs"] == cell(before, A)["token_beliefs"]
    assert cell(after, A)["active_time"] == cell(before, A)["active_time"]
    load(engine.config, after["decision_id"])


def test_run_that_records_no_amount_has_no_reader_row_and_moves_no_amount_belief(tmp_path):
    engine = engine_at(tmp_path)
    cells = [priced(A)]
    _, _, before = decide(engine, cells, {(A, "high"): by_kind(counts(input=6000, output=2000, cache_read=4000))})
    engine.observe("task", A, "high", "clean", native=True, task_text=stated(.5), id="partial")
    manifest = artifact(tmp_path, codex_events(), execution_id="partial", model=A, completed=False)
    engine.attach_usage("partial", manifest, id="partial-usage")
    _, _, after = decide(engine, cells)
    assert after["semantic"]["duration_observations"] == []
    assert after["semantic"]["outcome_observations"][0]["weight"] == pytest.approx(.5, rel=1e-4)
    assert cell(after, A)["active_time"] == cell(before, A)["active_time"]
    assert cell(after, A)["token_beliefs"] == cell(before, A)["token_beliefs"]


# --- 6. Older receipts ------------------------------------------------------------------------

def test_v7_to_v10_receipts_keep_the_cost_at_the_means_and_are_not_replayed(tmp_path, monkeypatch, capsys):
    engine, request_id, response, result = route(tmp_path, {A: forecast(), B: forecast(seconds=2400, loud_ci=[.01, .9])}, seed=4)
    current = load(engine.config, result["decision_id"])
    ids = {}
    for name, build in (("v10", to_v10), ("v9", to_v9), ("v8", to_v8), ("v7", to_v7)):
        legacy = build(current)
        ids[name] = digest(legacy)
        with locked(engine.config):
            put(engine.config, "decisions", ids[name], canonical(legacy) + b"\n")
        loaded = load(engine.config, ids[name])
        assert loaded == legacy and loaded["format"] == f"howhard-decision-{name}"
        assert loaded["semantic"]["sampled_expert"] == "llm"
        for old in loaded["selection"]["cells"]:
            loud, silent = old["P_loud"]["mean"], old["P_silent"]["mean"]
            # The older cost is read at the posterior means, with no average over a belief.
            assert old["expected_attempts"] == pytest.approx(1 / (1 - loud))
            assert old["J"] == pytest.approx((old["C_run"] + LAMBDA * old["T"] + C_LOUD * loud + C_SILENT * silent) / (1 - loud))
        assert all("sampled" not in c and "expert" in c for c in loaded["semantic"]["sampling"]["cells"])
        # An older receipt stores what its own draw gave and is not replayed from its seed.
        moved = deepcopy(legacy)
        moved["seed"] += 1
        moved["selection"]["cells"][0]["sampled_J"] *= 2
        assert load(engine.config, moved)["seed"] == legacy["seed"] + 1
        bad = deepcopy(legacy)
        bad["selection"]["cells"][0]["J"] *= 1.01
        with pytest.raises(HowHardError, match="cost-to-accepted-result mismatch"):
            load(engine.config, bad)
    wide = next(c for c in current["selection"]["cells"] if c["model"] == B)
    assert wide["expected_attempts"] > 1 / (1 - wide["P_loud"]["mean"]) + .1

    # Observe through each receipt, replay every forecast and report.
    monkeypatch.chdir(tmp_path)
    for name in ("v10", "v9", "v8"):
        engine.observe_decision(ids[name], "clean", id=f"from-{name}", active_seconds=10.)
    assert main(["observe", ids["v7"], "clean", "--id", "from-v7", "--json"]) == 0
    capsys.readouterr()
    engine.observe_decision(result["decision_id"], "clean", id="from-v11", active_seconds=10.)
    engine.refresh()
    beliefs = engine.memory_beliefs()
    assert beliefs["unscored_runs"] == 0
    assert next(e for e in beliefs["experts"] if e["key"] == "llm")["evidence_count"] == 5
    stats = report(engine)
    assert stats["unsupported_receipts"] == [] and stats["missing_receipts"] == []
    assert stats["forecast_metrics"]["llm"]["runs"] == 5


def test_command_line_pick_reports_the_sampled_cost_of_its_seed(tmp_path, monkeypatch, capsys):
    import json
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps([{"model": A, "effort": "high", "usd_per_mtok": 10.}]))
    picks = []
    for seed in ("1", "1", "2"):
        assert main(["pick", "task", "--text", "Repair rollback ordering", "--candidates", str(path), "--seed", seed, "--json"]) == 0
        picks.append(json.loads(capsys.readouterr().out))
    assert picks[0]["cells"][0]["sampled_J"] == picks[1]["cells"][0]["sampled_J"] != picks[2]["cells"][0]["sampled_J"]
    assert picks[0]["cells"][0]["J"] == picks[2]["cells"][0]["J"]
    assert "sampled_expert" not in picks[0]["semantic"]
    assert set(picks[0]["semantic"]["sampling"]["cells"][0]["sampled"]) == {"S", "u", "seconds", "credit"}
