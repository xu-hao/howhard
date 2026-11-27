"""The corrected prediction: where every belief of a forecast cell starts.

The record gives the level of a value, and the forecast only says whether this task looks
higher or lower than that forecaster's usual forecast::

    phi(prediction) = phi(level) + gamma * (phi(guess) - phi(usual))

``phi`` is the value's own scale: ``logit`` for the loud rate and the accepted-defect share,
each held in [0.0001, 0.9999], and ``ln`` for active time and each token count. ``guess`` is
the merged forecast of the value for this cell on this task.

``level`` and ``usual`` come down a ladder of five rungs, each leaning on the rung above::

    level(r) = (sum a_i * y_i + kappa_r * level(r+)) / (sum a_i + kappa_r)
    usual(r) = (sum a_i * g_i + kappa_r * usual(r+)) / (sum a_i + kappa_r)

The rungs are all hosts, this host, this agent under it, this family under it and this cell
under it; above all hosts both equal the guess, so an empty record leaves the guess unchanged.
A run ``i`` is a finished routed run on another task: ``y_i`` is what happened, ``g_i`` what
its own decision had guessed, and ``a_i`` its age factor after the per-ticket limit.

``gamma`` is the measured weight on the forecaster's judgment, one per kind of value and
shared by all hosts. The family's own runs set how firmly a prediction is held, and for an
amount the ladder of squared misses gives ``D``, the typical miss left after the correction.
"""

from datetime import datetime
import math

from .accounting import CATEGORIES, usage_by_kind
from .evidence import OUTCOMES, age_factor, ticket_shares
from .storage import effective_active_seconds
from .validation import HowHardError, keys, number, text

RATES = ("loud", "silent")
AMOUNTS = ("seconds", "tokens", *CATEGORIES)
KINDS = (*RATES, *AMOUNTS)
# The task worth of each kind of value: the most a record on other tasks is worth for one task.
TASK_WORTH = {"loud": "task_worth_loud", "silent": "task_worth_silent",
              **{kind: "task_worth_amount" for kind in AMOUNTS}}
RUNGS = ("all-hosts", "host", "agent", "family", "cell")
RATE_HOLD = (.0001, .9999)          # A rate is held in this interval before its log-odds are taken.
MIN_FORECAST_GAP = .05              # Below this gap between the two groups' forecasts, gamma_hat is 1.
GROUP_SUMS = ("weight", "gap", "outcome", "level")
VALUE_FIELDS = ("guess", "rungs", "level", "usual", "prediction", "worth", "N_start", "s_stated", "s")
TOLERANCE = 1e-10


def is_rate(kind):
    return kind in RATES


def scale(kind, value):
    """``phi``: a rate's log-odds, held; an amount is carried as its logarithm already."""
    if not is_rate(kind):
        return value
    from .success import logit
    return logit(min(RATE_HOLD[1], max(RATE_HOLD[0], value)))


def rung_sums(kind):
    """The sums one rung stores: ``sum a``, ``sum a*y``, ``sum a*g`` and, for an amount, ``sum a*e^2``."""
    return ("weight", "outcome", "guess") if is_rate(kind) else ("weight", "outcome", "guess", "miss")


def cell_kinds(entries):
    """The values one forecast cell has: both rates, active time, and its token kinds or their total."""
    return (*RATES, "seconds", *(CATEGORIES if CATEGORIES[0] in entries else ("tokens",)))


def guesses(panels, experts, pair):
    """The merged forecast of every value of one cell, as the experts' panels state it.

    Each entry holds ``guess`` (a probability for a rate, the logarithm of the merged
    forecast for an amount), its stated spread ``s_stated``, ``positive`` (some expert
    forecasts the amount above zero) and ``exact_zero`` (every expert forecasts a token
    count at zero with no non-zero upper end). Total ``tokens`` is always merged; a token
    kind only when every expert states usage by kind.
    """
    from .success import merge_forecasts, merged_forecast
    rows = [(e["weight"], panels[e["key"]][pair]) for e in experts]
    merged = merged_forecast(panels, experts, pair)
    result = {rate: {"guess": merged[rate]["prediction"], "s_stated": merged[rate]["stated_spread"]} for rate in RATES}

    def amount(forecasts, *, zero=True):
        centre, spread = merge_forecasts(forecasts)
        return {"guess": centre, "s_stated": spread, "positive": any(v > 0 for _, v, _ in forecasts),
                "exact_zero": zero and all(v == 0 and (r is None or r[1] == 0) for _, v, r in forecasts)}
    # Only a token count can be exactly zero: active time is held at one second or more.
    result["seconds"] = amount([(w, row["seconds"], row.get("seconds_ci")) for w, row in rows], zero=False)
    result["tokens"] = amount([(w, row["tokens"], None) for w, row in rows])
    if all("token_usage" in row for _, row in rows):
        for kind in CATEGORIES:
            result[kind] = amount([(w, row["token_usage"][kind], row.get("token_usage_ci", {}).get(kind))
                                   for w, row in rows])
    return result


def stored_forecast(receipt):
    """What a stored decision says about its cells: its host, their agents and what it guessed.

    Receipts of every readable format give a guess by the same merge rule; one older than v11
    names neither a host nor an agent. A cell the decision stored no forecast for has no guess.
    """
    from .decisions import host
    from .semantic import panel_lookup
    state = receipt["semantic"]
    panels = {key: panel_lookup(panel) for key, panel in state["worlds"].items()}
    experts = state["beliefs"]["experts"]
    covered = all(e["key"] in panels for e in experts)
    return {"host": host(receipt),
            "agents": {(c["model"], c["effort"]): c.get("agent") for c in receipt["native_candidates"]},
            "guesses": {pair: guesses(panels, experts, pair) for pair in (panels["llm"] if covered else ())
                        if all(pair in panels[e["key"]] for e in experts)}}


def routed_runs(engine, excluded, prepared, half_life):
    """The finished routed runs on other tasks: what each recorded and what its decision had guessed.

    A run enters when its outcome is clean, loud or silent and its decision stored a forecast
    for the cell that ran. Its host and agent are the ones that decision stores. ``excluded``
    holds the evidence candidates of the request, which count through the similar runs
    instead. Each run carries, per kind of value it enters, ``(y, g)``: what happened and
    what had been guessed, on the footing of the ladders.
    """
    from .decisions import exists, load
    from .success import log_amount
    # A receipt is immutable under its ID, so what it guessed is read once per engine.
    known, result = engine.receipt_forecasts, []
    for row in engine.rows:
        decision_id = row.get("decision_id")
        if row["outcome"] not in OUTCOMES or not decision_id or row["id"] in excluded:
            continue
        if decision_id not in known:
            if not exists(engine.config, decision_id):
                continue
            receipt = load(engine.config, decision_id, for_scoring=True)
            known[decision_id] = None if receipt is None else stored_forecast(receipt)
        stored = known[decision_id]
        pair = row["family"], row["resolved_effort"]
        guessed = stored["guesses"].get(pair) if stored is not None else None
        if guessed is None:
            continue
        # The loud rate counts every finished run; the accepted-defect share only accepted results.
        values = {"loud": (1. if row["outcome"] == "loud" else 0., guessed["loud"]["guess"])}
        if row["outcome"] != "loud":
            values["silent"] = (1. if row["outcome"] == "silent" else 0., guessed["silent"]["guess"])
        # An amount counts when both the recorded amount and its guess are above zero.
        usage = usage_by_kind(row.get("execution_usage")) or {}
        recorded = {"seconds": effective_active_seconds(row), "tokens": row.get("tokens"), **usage}
        for kind, amount in recorded.items():
            if amount is not None and amount > 0 and kind in guessed and guessed[kind]["positive"]:
                values[kind] = (log_amount(amount), guessed[kind]["guess"])
        days = max(0., (prepared - datetime.fromisoformat(row["observed_at"])).total_seconds() / 86400.)
        result.append({"host": stored["host"], "agent": stored["agents"].get((row["model"], row["effort"])),
                       "family": pair[0], "effort": pair[1], "ticket": row["ticket"],
                       "age": age_factor(days, half_life), "values": values})
    return result


def ladder(rungs, field, top, policy):
    """The value at the lowest rung: each rung is the weighted average of its runs and the rung above.

    The rung above is worth ``model_worth`` runs at the family rung and ``rung_worth`` runs
    elsewhere. A rung with no run passes the value above through unchanged.
    """
    value = top
    for name, rung in zip(RUNGS, rungs):
        if rung["weight"] > 0:
            kappa = policy["model_worth" if name == "family" else "rung_worth"]
            value = (rung[field] + kappa * value) / (rung["weight"] + kappa)
    return value


def observed_gap(kind, group):
    """``O``: how far one group's results sat from its levels, on the value's own scale."""
    if not is_rate(kind):
        return (group["outcome"] - group["level"]) / group["weight"]
    return scale(kind, (group["outcome"] + .5) / (group["weight"] + 1.)) - scale(
        kind, (group["level"] + .5) / (group["weight"] + 1.))


def forecaster_weight(kind, high, low, policy):
    """``n``, ``gamma_hat`` and ``gamma`` for one kind of value, from its two groups' sums.

    ``gamma_hat`` is the gap that happened between the runs forecast above usual and the rest,
    divided by the gap that was forecast, held in [0, 1]. It is 1 when a group is empty or the
    forecast gap is under 0.05, and ``gamma`` leans on 1 with the worth of one rung.
    """
    n, gamma_hat = high["weight"] + low["weight"], 1.
    if high["weight"] > 0 and low["weight"] > 0:
        forecast = high["gap"] / high["weight"] - low["gap"] / low["weight"]
        if forecast >= MIN_FORECAST_GAP:
            gamma_hat = min(1., max(0., (observed_gap(kind, high) - observed_gap(kind, low)) / forecast))
    kappa = policy["rung_worth"]
    return {"n": n, "gamma_hat": gamma_hat, "gamma": (n * gamma_hat + kappa) / (n + kappa)}


def _path(run):
    return run["host"], run["agent"], run["family"], run["effort"]


def summarize(runs, cells, policy):
    """Reduce the routed runs to the sums a decision stores.

    ``cells`` maps a cell key to its ``(host, agent, family, effort)``. For each kind of value
    the result holds the two groups' sums that measure ``gamma``, and for each cell the sums of
    its five rungs. A kind no run enters is left out; it reads as all zeros.
    """
    result = {"kinds": {}, "cells": {key: {} for key in cells}}
    for kind in KINDS:
        # A run too old to weigh anything is left out with the runs that never entered.
        entries = [(run, *run["values"][kind]) for run in runs if kind in run["values"] and run["age"] > 0]
        if not entries:
            continue
        # All runs of one ticket in this ladder together count for at most one run.
        shares = ticket_shares([(run["ticket"], run["age"]) for run, _, _ in entries])
        weighted = [(_path(run), run["age"] * share, y, g) for (run, y, g), share in zip(entries, shares)]
        nodes = {}
        for path, a, y, g in weighted:
            for depth in range(len(RUNGS)):
                node = nodes.setdefault(path[:depth], {"weight": 0., "outcome": 0., "guess": 0., "miss": 0.})
                node["weight"] += a
                node["outcome"] += a * y
                node["guess"] += a * g
        # Each run's own level and usual: both ladders at its cell over all the runs, with the
        # record's weighted averages above all hosts.
        record, own = nodes[()], {}
        for path in {path for path, _, _, _ in weighted}:
            along = [nodes[path[:depth]] for depth in range(len(RUNGS))]
            own[path] = (ladder(along, "outcome", record["outcome"] / record["weight"], policy),
                         ladder(along, "guess", record["guess"] / record["weight"], policy))
        groups = {name: dict.fromkeys(GROUP_SUMS, 0.) for name in ("high", "low")}
        gaps = []
        for path, a, y, g in weighted:
            level, usual = own[path]
            gap = scale(kind, g) - scale(kind, usual)
            group = groups["high" if gap > 0 else "low"]
            group["weight"] += a
            group["gap"] += a * gap
            group["outcome"] += a * y
            group["level"] += a * level
            gaps.append((level, gap))
        result["kinds"][kind] = groups
        if not is_rate(kind):
            # The miss each past run would have left: its result against its own corrected prediction.
            gamma = forecaster_weight(kind, groups["high"], groups["low"], policy)["gamma"]
            for (path, a, y, _), (level, gap) in zip(weighted, gaps):
                miss = (y - (level + gamma * gap)) ** 2
                for depth in range(len(RUNGS)):
                    nodes[path[:depth]]["miss"] += a * miss
        for key, path in cells.items():
            result["cells"][key][kind] = [{field: nodes[path[:depth]][field] if path[:depth] in nodes else 0.
                                           for field in rung_sums(kind)} for depth in range(len(RUNGS))]
    return result


def collect(engine, bundle, candidates):
    """Read the record for one request, on the caller's captured rows and outside any storage lock."""
    from . import prototypes
    from .capability import cell_key
    from .native import agents
    policy = engine.config.data["sampling"]
    # The evidence candidates of this request are runs on tasks like the new one: they count
    # through their similarity, never in a ladder.
    excluded = prototypes.matched_members(engine, bundle.get("prototype_state"))
    runs = routed_runs(engine, excluded, datetime.fromisoformat(candidates["prepared_at"]), policy["half_life_days"])
    labels = agents(bundle["native_candidates"])
    models = engine.config.data["models"]
    cells = {cell_key(c): (bundle["host"], labels[cell_key(c)], models[c["model"]]["family"], c["effort"])
             for c in bundle["native_candidates"]}
    return summarize(runs, cells, policy)


def empty_rungs(kind):
    return [dict.fromkeys(rung_sums(kind), 0.) for _ in RUNGS]


def _exp(x):
    from .success import EXP_CAP
    return math.exp(min(x, EXP_CAP))


def value(kind, entry, rungs, gamma, policy):
    """One value of one cell: its prediction and how firmly it is held, from the rung sums.

    ``worth = n_family + model_worth`` and ``N_start = 1 / (1/worth + 1/N_task)`` say how many
    runs the prediction is worth. The starting spread is the stated one or
    ``sigma_1 / sqrt(N_start)``, whichever is wider, and at most the flat belief for a rate.
    A rate reports probabilities; an amount reports amounts and adds ``D``.
    """
    from .success import PREDICTION_HOLD, _sigmoid, start_spread
    worth = rungs[RUNGS.index("family")]["weight"] + policy["model_worth"]
    n_start = 1. / (1. / worth + 1. / policy[TASK_WORTH[kind]])
    guess, stated = entry["guess"], entry["s_stated"]
    level, usual = ladder(rungs, "outcome", guess, policy), ladder(rungs, "guess", guess, policy)
    start = scale(kind, guess)
    corrected = scale(kind, level) + gamma * (start - scale(kind, usual))
    record = {"rungs": rungs, "worth": worth, "N_start": n_start, "s_stated": stated}
    if is_rate(kind):
        # A record that moves nothing leaves the guess exactly as it was merged.
        prediction = guess if corrected == start else min(PREDICTION_HOLD[1], max(PREDICTION_HOLD[0], _sigmoid(corrected)))
        return {"guess": guess, "level": level, "usual": usual, "prediction": prediction, **record,
                "s": start_spread(prediction, stated, n_start)}
    # D is the typical miss left after the correction; with no run it is the configured miss.
    miss = policy["duration_miss"]
    if any(rung["weight"] > 0 for rung in rungs):
        miss = math.sqrt(ladder(rungs, "miss", miss ** 2, policy))
    record["D"] = miss
    if entry["exact_zero"]:
        # A kind forecast at zero stays zero and has no spread.
        return {"guess": 0., "level": None, "usual": None, "prediction": 0., **record, "s_stated": 0., "s": 0.}
    return {"guess": _exp(guess), "level": _exp(level), "usual": _exp(usual), "prediction": _exp(corrected), **record,
            "s": max(stated, miss * math.sqrt(policy["scatter_share"]) / math.sqrt(n_start))}


def assemble(learned, panels, experts, cells, policy):
    """The receipt's ``semantic.prediction``: the measured weights and every forecast cell's values.

    ``learned`` is the record read when the request was prepared, and ``cells`` lists
    ``(cell key, (family, effort))`` for the forecast cells.
    """
    empty = dict.fromkeys(GROUP_SUMS, 0.)
    kinds = {}
    for kind in KINDS:
        groups = learned["kinds"].get(kind, {"high": empty, "low": empty})
        high, low = dict(groups["high"]), dict(groups["low"])
        kinds[kind] = {"high": high, "low": low, **forecaster_weight(kind, high, low, policy)}
    rows = []
    for key, pair in cells:
        entries = guesses(panels, experts, pair)
        stored = learned["cells"].get(key, {})
        values = {kind: value(kind, entries[kind], [dict(rung) for rung in stored.get(kind, empty_rungs(kind))],
                              kinds[kind]["gamma"], policy) for kind in cell_kinds(entries)}
        for record in values.values():
            record["rungs"] = [{"rung": name, **rung} for name, rung in zip(RUNGS, record["rungs"])]
        rows.append({"family": pair[0], "effort": pair[1], "values": values})
    return {"kinds": kinds, "cells": rows}


def by_cell(block):
    """Each forecast cell's values from a ``semantic.prediction`` block."""
    return {(cell["family"], cell["effort"]): cell["values"] for cell in block["cells"]}


def _same(actual, expected):
    if expected is None or actual is None:
        return actual is None and expected is None
    return (type(actual) in (int, float) and math.isfinite(actual)
            and abs(actual - expected) <= TOLERANCE * max(1., abs(expected)))


def validate(block, panels, experts, pairs, policy, *, where="decision.semantic.prediction"):
    """Check a stored prediction, recomputing every derived number from its stored sums.

    The guesses are merged again from the frozen forecast panels; the weights, levels, usual
    forecasts, predictions, worths, spreads and misses must follow from the stored sums.
    """
    keys(block, ["kinds", "cells"], where=where)
    keys(block["kinds"], KINDS, where=f"{where}.kinds")
    gammas = {}
    for kind in KINDS:
        stored, spot = block["kinds"][kind], f"{where}.kinds.{kind}"
        keys(stored, ["n", "high", "low", "gamma_hat", "gamma"], where=spot)
        for name in ("high", "low"):
            group = stored[name]
            keys(group, GROUP_SUMS, where=f"{spot}.{name}")
            weight = number(group["weight"], f"{spot}.{name}.weight", nonnegative=True)
            number(group["gap"], f"{spot}.{name}.gap")
            for field in ("outcome", "level"):
                total = number(group[field], f"{spot}.{name}.{field}", nonnegative=True)
                if is_rate(kind) and total > weight * (1 + TOLERANCE) + TOLERANCE:
                    raise HowHardError(f"{spot}.{name}.{field}: a rate's sum cannot exceed its weight")
        expected = forecaster_weight(kind, stored["high"], stored["low"], policy)
        for field, wanted in expected.items():
            if not _same(stored[field], wanted):
                raise HowHardError(f"{spot}.{field}: does not follow from the stored sums")
        gammas[kind] = expected["gamma"]
    if not isinstance(block["cells"], list):
        raise HowHardError(f"{where}.cells: expected array")
    seen = set()
    for cell in block["cells"]:
        keys(cell, ["family", "effort", "values"], where=f"{where}.cell")
        pair = text(cell["family"], f"{where}.cell.family"), text(cell["effort"], f"{where}.cell.effort")
        if pair not in pairs or pair in seen:
            raise HowHardError(f"{where}.cells: unknown or duplicate forecast cell")
        seen.add(pair)
        entries = guesses(panels, experts, pair)
        keys(cell["values"], cell_kinds(entries), where=f"{where}.cell.values")
        for kind in cell_kinds(entries):
            stored, spot = cell["values"][kind], f"{where}.cell.values.{kind}"
            keys(stored, [*VALUE_FIELDS, *(() if is_rate(kind) else ("D",))], where=spot)
            rungs = stored["rungs"]
            if not isinstance(rungs, list) or len(rungs) != len(RUNGS):
                raise HowHardError(f"{spot}.rungs: expected one entry for each of the five rungs")
            above = None
            for name, rung in zip(RUNGS, rungs):
                keys(rung, ["rung", *rung_sums(kind)], where=f"{spot}.rung")
                if rung["rung"] != name:
                    raise HowHardError(f"{spot}.rungs: expected the rungs from all hosts down to the cell")
                weight = number(rung["weight"], f"{spot}.rung.weight", nonnegative=True)
                for field in rung_sums(kind)[1:]:
                    total = number(rung[field], f"{spot}.rung.{field}", nonnegative=True)
                    if is_rate(kind) and total > weight * (1 + TOLERANCE) + TOLERANCE:
                        raise HowHardError(f"{spot}.rung.{field}: a rate's sum cannot exceed its weight")
                # Each rung holds a part of the runs of the rung above it.
                if above is not None and weight > above * (1 + TOLERANCE) + TOLERANCE:
                    raise HowHardError(f"{spot}.rungs: a rung cannot outweigh the rung above it")
                above = weight
            sums = [{field: rung[field] for field in rung_sums(kind)} for rung in rungs]
            expected = value(kind, entries[kind], sums, gammas[kind], policy)
            for field, wanted in expected.items():
                if field != "rungs" and not _same(stored[field], wanted):
                    raise HowHardError(f"{spot}.{field}: does not follow from the stored sums and the frozen forecasts")
    if seen != set(pairs):
        raise HowHardError(f"{where}.cells: must cover the forecast cells")
    return by_cell(block)
