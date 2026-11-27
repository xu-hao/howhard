"""Refined constants: the estimated constants re-estimated from the user's own record.

The work runs after an observation or a flush, never inside a decision, and publishes one
cached file in the state root. A decision only reads it: the engine applies it at refresh,
in the order default, refined, configured. Every score predicts a finished routed run with a
``howhard-decision-v11`` receipt only from the rows and amendments observed before its
decision, rebuilding the cell's beliefs with the value tried::

    outcome:  ln E[theta_l]                        the run ended loud
              ln(1 - E[theta_l]) + ln E[S]         it ended silent
              ln(1 - E[theta_l]) + ln(1 - E[S])    it ended clean
    time:     ln density of the recorded active time under  ln T ~ Normal(m', s'^2 + tau^2)
    caught:   ln d(host)  for a failed run that was loud,   ln(1 - d(host))  for a silent one

The constants are tried one at a time in a fixed order, the earlier ones at their refined
values, and one leaves its default only on a large enough record with a clear gain::

    d_i  = score_i(tried) - score_i(current)
    gain = mean(d_i),   se = sd(d_i) / sqrt(n)
    move <=> n >= 200  and  gain > 2 * se

A receipt stores what each candidate run weighs before the half-life and the discounts
(its similarity, its days, its weight on record, its cell, agent and ticket), so the weights
are recomputed without the reranker; the corrected prediction and the host's caught share
are rebuilt from the rows and from the forecasts and labels in receipts. The rebuilt ladders
leave out only the candidates a receipt stores: when its prototypes matched more than
``comparison_cap`` runs, the extra matched runs count in the rebuilt ladders, although the
decision's own ladders left every matched run out.
"""

from datetime import datetime, timedelta, timezone
import math
import re

import numpy as np

from .context import DECISION_FORMAT, canonical, digest
from .evidence import OUTCOMES
from .validation import HowHardError, keys, number

FORMAT = "howhard-refined-v1"
FILE = "refined-constants.json"
MINIMUM_RUNS = 200      # A constant moves only on at least this many scored runs of its score.
RESCORE_RUNS = 25       # After an observation, a cached result is redone once it covers this many fewer.
MAXIMUM_RUNS = 2000     # At most this many of the most recent scored runs are scored.
MINIMUM_PAIRS = 30      # The measured run-to-run scatter is offered once this many pairs exist.
MEASURED = "measured"
# Each constant in the order it is tried: its section, its key, the values tried (the default
# and the measured value for the two that are measured) and the score that judges it.
CONSTANTS = (
    ("sampling", "half_life_days", (7., 10., 14., 21., 30.), "outcome"),
    ("sampling", "agent_discount", (0., .1, .25, .5), "outcome"),
    ("sampling", "model_discount", (.1, .25, .5), "outcome"),
    ("sampling", "effort_discount", (.25, .5, .75, 1.), "outcome"),
    ("sampling", "rung_worth", (20., 40., 80.), "outcome"),
    ("sampling", "task_worth_silent", (8., 16., 32., 64.), "accepted"),
    ("sampling", "task_worth_loud", (8., 16., 32., 64.), "outcome"),
    ("sampling", "duration_miss", MEASURED, "time"),
    ("sampling", "scatter_share", MEASURED, "time"),
    ("sampling", "task_worth_amount", (.5, 1., 2., 4.), "time"),
    ("capability", "caught_share_worth", (6., 12., 24., 40.), "caught"),
)
KEYS = tuple((section, key) for section, key, _, _ in CONSTANTS)
ENTRY_FIELDS = ("value", "n", "gain", "se")
DISCOUNTS = ("agent_discount", "model_discount", "effort_discount")
DISCOUNT_ORDER = ("cell", "family", "agent", "other")
DAY_US = 86400 * 10 ** 6
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
MICROSECOND = timedelta(microseconds=1)
LATEST = np.iinfo(np.int64).max


def defaults():
    """Each refinable constant's packaged default."""
    from . import capability, success
    tables = {"sampling": success.DEFAULTS, "capability": capability.DEFAULTS}
    return {(section, key): float(tables[section][key]) for section, key in KEYS}


# ---------------------------------------------------------------------------------------------
# The cached file


def path(config):
    return config.state_root / FILE


def validate(value, where="refined constants"):
    """Check a cached result: its format, the record it names and one entry per constant."""
    keys(value, ["format", "rows", "scored_runs", "constants"], where=where)
    if value["format"] != FORMAT:
        raise HowHardError(f"{where}: unsupported format")
    if not isinstance(value["rows"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["rows"]):
        raise HowHardError(f"{where}.rows: expected the SHA-256 digest of the rows read")
    if type(value["scored_runs"]) is not int or value["scored_runs"] < 0:
        raise HowHardError(f"{where}.scored_runs: expected a nonnegative integer")
    keys(value["constants"], [key for _, key in KEYS], where=f"{where}.constants")
    for section, key in KEYS:
        entry = value["constants"][key]
        keys(entry, ENTRY_FIELDS, where=f"{where}.constants.{key}")
        if entry["value"] is not None:
            # A discount may be 0; every other refined constant is positive.
            number(entry["value"], f"{where}.constants.{key}.value", positive=key not in DISCOUNTS, nonnegative=True)
        if entry["n"] is not None and (type(entry["n"]) is not int or entry["n"] < 0):
            raise HowHardError(f"{where}.constants.{key}.n: expected a nonnegative integer or null")
        for field in ("gain", "se"):
            if entry[field] is not None:
                number(entry[field], f"{where}.constants.{key}.{field}")
    return value


def read(config):
    """The cached result in the state root, or ``None`` when there is none."""
    import json
    target = path(config)
    if not target.exists():
        return None
    try:
        value = json.loads(target.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise HowHardError("refined constants: invalid JSON") from exc
    return validate(value)


def apply(config, cached):
    """Set each refinable constant: the default, then the refined value, then the configured one.

    ``config.unrefined`` holds each constant's value before any refinement (the default, or
    what the configuration file sets). A key the configuration file sets keeps its value.
    Nothing changes unless both resulting policies are valid.
    """
    from . import capability, success
    tables = {"sampling": dict(config.data["sampling"]), "capability": dict(config.data["capability"])}
    for section, key in KEYS:
        if (section, key) in config.file_keys:
            continue
        refined = cached["constants"][key]["value"] if cached is not None else None
        tables[section][key] = config.unrefined[section, key] if refined is None else refined
    try:
        success.validate_policy(dict(tables["sampling"]))
        capability.validate_policy(dict(tables["capability"]))
    except HowHardError as exc:
        raise HowHardError(f"refined constants: {exc}") from exc
    for section, values in tables.items():
        config.data[section].update(values)


# ---------------------------------------------------------------------------------------------
# The rule and the scores


def rule(differences):
    """``(n, gain, se, moves)`` for the per-run differences of one value tried against the current one."""
    d = np.asarray(differences, dtype=float)
    n = int(d.size)
    if n == 0:
        return 0, None, None, False
    gain = float(np.mean(d))
    if n < 2:
        return n, gain, None, False
    se = float(np.std(d, ddof=1) / math.sqrt(n))
    return n, gain, se, n >= MINIMUM_RUNS and gain > 2. * se


def outcome_score(loud, share, outcome):
    """The outcome score of one run from ``E[theta_l]``, ``E[S]`` and how it ended."""
    if outcome == "loud":
        return math.log(loud)
    return math.log1p(-loud) + (math.log(share) if outcome == "silent" else math.log1p(-share))


def time_score(y, centre, spread, tau):
    """The log density of ``y = ln T`` under ``Normal(centre, spread^2 + tau^2)``."""
    variance = spread ** 2 + tau ** 2
    return -.5 * math.log(2. * math.pi * variance) - (y - centre) ** 2 / (2. * variance)


def caught_score(d, outcome):
    return math.log(d) if outcome == "loud" else math.log1p(-d)


def measured_miss(misses):
    """``D = sqrt(mean(e_i^2))``; ``None`` with no miss."""
    misses = np.asarray(misses, dtype=float)
    return float(math.sqrt(np.mean(misses ** 2))) if misses.size else None


def measured_scatter(pairs):
    """``tau = sqrt(mean(p_j^2) / 2)``; ``None`` with fewer than ``MINIMUM_PAIRS`` pairs."""
    pairs = np.asarray(pairs, dtype=float)
    return float(math.sqrt(np.mean(pairs ** 2) / 2.)) if pairs.size >= MINIMUM_PAIRS else None


def _micros(stamp):
    """An ISO timestamp as whole microseconds since 1970, so that time differences are exact."""
    return (datetime.fromisoformat(stamp) - EPOCH) // MICROSECOND


# ---------------------------------------------------------------------------------------------
# The record read once: the history of routed runs and the scored runs


class Record:
    """The record as one snapshot holds it, read once for every value tried.

    Each row of ``table`` is one routed run as of one of its amendments: the state that
    applies to a decision is the one whose original and amendments were all observed before
    the decision was prepared. ``runs`` are the scored runs, most recent first, and ``total``
    counts every scored run on record.
    """

    def __init__(self, engine):
        from .decisions import exists, load
        from .prediction import scale, stored_forecast
        from .storage import effective_active_seconds, effective_rows
        from .success import log_amount
        from .names import family_agent, host as named_host
        config = engine.config
        self.rows_digest = digest(engine.raw_rows)
        # Names an operator imported for runs whose receipt stores none, as they stand now.
        names = getattr(engine, "names", None)
        forecasts, hosts, scoring = engine.receipt_forecasts, {}, {}

        def receipt(decision_id):
            """What a stored decision gives: its forecasts, its host and forecast cells, and its scoring inputs."""
            if decision_id not in scoring:
                value = load(config, decision_id, for_scoring=True) if exists(config, decision_id) else None
                scoring[decision_id] = value and _scoring_info(value)
                if decision_id not in forecasts:
                    forecasts[decision_id] = None if value is None else stored_forecast(value)
                hosts[decision_id] = None if value is None else (named_host(value, decision_id, names), frozenset(
                    (c["family"], c["effort"]) for c in value["semantic"]["worlds"]["llm"]))
            return forecasts[decision_id], hosts[decision_id], scoring[decision_id]

        # Each run with its amendments, in the order the record holds them.
        chains, roots = {}, {}
        for row in engine.raw_rows:
            if row["amends"] is None:
                roots[row["id"]] = row["id"]
                chains[row["id"]] = [row]
            else:
                roots[row["id"]] = roots[row["amends"]]
                chains[roots[row["id"]]].append(row)
        codes = [{} for _ in range(5)]   # A code for each path prefix: (), host, agent, family, effort.
        columns = {name: [] for name in ("vf", "vt", "t", "root", "ladder", "caught", "finished", "loud",
                                         "silent", "ticket", "d1", "d2", "d3", "d4", "g_loud", "g_silent",
                                         "sg_loud", "sg_silent", "g_sec", "y_sec", "has_sec")}
        tickets = {}
        self.root_index = {}
        for index, (root_id, chain) in enumerate(chains.items()):
            first = chain[0]
            decision_id = first.get("decision_id")
            if not decision_id:
                continue
            stored, named, _ = receipt(decision_id)
            original = effective_rows(config, chain[:1])[0]
            pair = original["family"], original["resolved_effort"]
            guessed = stored["guesses"].get(pair) if stored is not None else None
            in_caught = named is not None and pair in named[1]
            if guessed is None and not in_caught:
                continue
            self.root_index[root_id] = index
            # A v11 receipt names the run's host and agent; an older one takes the imported names.
            host = named[0]
            if stored is not None and stored["named"]:
                agent = stored["agents"].get((original["model"], original["effort"]))
            else:
                agent = family_agent(pair[0], names)
            path_ = (host, agent, pair[0], pair[1])
            prefix_codes = [codes[d].setdefault(path_[:d], len(codes[d])) for d in range(5)]
            ticket = -1 if original["ticket"] is None else tickets.setdefault(original["ticket"], len(tickets))
            observed = _micros(original["observed_at"])
            valid_from = observed
            starts = []
            for k in range(len(chain)):
                valid_from = max(valid_from, _micros(chain[k]["observed_at"]))
                starts.append(valid_from)
            for k in range(len(chain)):
                state = effective_rows(config, chain[:k + 1])[0] if k else original
                active = effective_active_seconds(state)
                columns["vf"].append(starts[k])
                columns["vt"].append(starts[k + 1] if k + 1 < len(chain) else LATEST)
                columns["t"].append(observed)
                columns["root"].append(index)
                columns["ladder"].append(guessed is not None)
                columns["caught"].append(in_caught)
                columns["finished"].append(state["outcome"] in OUTCOMES)
                columns["loud"].append(state["outcome"] == "loud")
                columns["silent"].append(state["outcome"] == "silent")
                columns["ticket"].append(ticket)
                for d in range(1, 5):
                    columns[f"d{d}"].append(prefix_codes[d])
                if guessed is not None:
                    columns["g_loud"].append(guessed["loud"]["guess"])
                    columns["g_silent"].append(guessed["silent"]["guess"])
                    columns["sg_loud"].append(scale("loud", guessed["loud"]["guess"]))
                    columns["sg_silent"].append(scale("silent", guessed["silent"]["guess"]))
                    columns["g_sec"].append(guessed["seconds"]["guess"])
                    timed = active is not None and active > 0 and guessed["seconds"]["positive"]
                    columns["y_sec"].append(log_amount(active) if timed else 0.)
                    columns["has_sec"].append(timed)
                else:
                    for name in ("g_loud", "g_silent", "sg_loud", "sg_silent", "g_sec", "y_sec"):
                        columns[name].append(0.)
                    columns["has_sec"].append(False)
        ints = ("vf", "vt", "t", "root", "ticket", "d1", "d2", "d3", "d4")
        bools = ("ladder", "caught", "finished", "loud", "silent", "has_sec")
        self.table = {name: np.asarray(values, dtype=np.int64 if name in ints else bool if name in bools else float)
                      for name, values in columns.items()}
        self.codes = codes
        self.prefixes = [max(1, len(c)) for c in codes]
        self.tickets = max(1, len(tickets))
        self.host_codes = codes[1]

        # The scored runs: finished routed runs whose decision is a v11 receipt with a forecast
        # for the cell that ran, most recent first.
        found = []
        for position, row in enumerate(engine.rows):
            decision_id = row.get("decision_id")
            if row["outcome"] not in OUTCOMES or not decision_id:
                continue
            stored, _, info = receipt(decision_id)
            pair = row["family"], row["resolved_effort"]
            if info is None or pair not in stored["guesses"]:
                continue
            found.append((_micros(row["observed_at"]), position, row, stored, info))
        found.sort(key=lambda item: (item[0], item[1]), reverse=True)
        self.total = len(found)
        self.runs = [self._run(observed, row, stored, info) for observed, _, row, stored, info in found[:MAXIMUM_RUNS]]

    def _run(self, observed, row, stored, info):
        """One scored run: what its decision saw for the cell that ran, and what then happened."""
        from .storage import effective_active_seconds
        from .success import log_amount
        pair = row["family"], row["resolved_effort"]
        agent = stored["agents"].get((row["model"], row["effort"]))
        path_ = (stored["host"], agent, *pair)
        cell = (*pair, agent)
        from .evidence import relation
        candidates = info["candidates"]
        local = {}
        active = effective_active_seconds(row)
        own = [c for c in candidates if (c["family"], c["effort"]) == pair]
        return {
            # Runs of one decision and cell share everything that was predicted for them.
            "group": (row["decision_id"], pair, agent),
            "observed": observed, "t": info["prepared_at"], "outcome": row["outcome"], "ticket": row["ticket"],
            "pair": pair, "epsilon": info["epsilon"], "guesses": stored["guesses"][pair],
            "codes": [self.codes[d].get(path_[:d], -1) for d in range(5)],
            "host": self.host_codes.get((stored["host"],), -1),
            "excluded": np.asarray(sorted(self.root_index[c["id"]] for c in candidates if c["id"] in self.root_index),
                                   dtype=np.int64),
            "timed": active is not None and active > 0, "y": log_amount(active) if active else 0.,
            # What each candidate weighs before the half-life and the discounts, and how its cell
            # relates to the cell that ran.
            "base": np.asarray([c["run_weight"] * c["similarity"] for c in candidates], dtype=float),
            "days": np.asarray([c["days"] for c in candidates], dtype=float),
            "relation": np.asarray([DISCOUNT_ORDER.index(relation((c["family"], c["effort"], c["agent"]), cell))
                                    for c in candidates], dtype=np.int64),
            "outcomes": np.asarray([OUTCOMES.index(c["outcome"]) for c in candidates], dtype=np.int64),
            "tickets": np.asarray([-1 if c["ticket"] is None else local.setdefault(c["ticket"], len(local))
                                   for c in candidates], dtype=np.int64),
            "own": [(c["run_weight"] * c["similarity"], c["days"], c["ticket"], c["active_seconds"]) for c in own],
        }


def _scoring_info(receipt):
    """What a v11 receipt gives a rescoring: its prepare time, candidates and risk limit."""
    if receipt["format"] != DECISION_FORMAT:
        return None
    candidates = receipt["semantic"]["evidence_candidates"]
    return {"prepared_at": _micros(candidates["prepared_at"]),
            "candidates": [{field: item[field] for field in ("id", "family", "effort", "agent", "outcome",
                                                             "active_seconds", "ticket", "run_weight",
                                                             "similarity", "days")}
                           for item in candidates["candidates"]],
            "epsilon": float(receipt["selection"]["constants"]["epsilon"])}


# ---------------------------------------------------------------------------------------------
# Rebuilt beliefs


def _shares(keys_, weights, size):
    """``1 / max(1, W_t)`` per entry, ``W_t`` summed over the entries of one ticket; ``1`` with no ticket."""
    ticketed = keys_ >= 0
    shares = np.ones(len(weights))
    if ticketed.any():
        totals = np.bincount(keys_[ticketed], weights=weights[ticketed], minlength=size)
        shares[ticketed] = 1. / np.maximum(1., totals[keys_[ticketed]])
    return shares


def ladders(record, run, half_life, policy):
    """The ladder sums of the run's cell for ``loud``, ``silent`` and ``seconds``, and each kind's gamma.

    The same reduction as ``prediction.summarize``, over the finished routed runs observed
    before the run's decision, as amended then, without its candidates, with ages at
    ``half_life``. Returns ``{kind: (gamma, rungs)}``.
    """
    from .prediction import forecaster_weight, rung_sums
    T, t = record.table, run["t"]
    sel = (T["vf"] < t) & (t <= T["vt"]) & T["ladder"] & T["finished"]
    if run["excluded"].size:
        sel &= ~np.isin(T["root"], run["excluded"])
    idx = np.flatnonzero(sel)
    age = 0.5 ** (np.maximum(0., (t - T["t"][idx]) / DAY_US) / half_life)
    keep = age > 0
    idx, age = idx[keep], age[keep]
    loud = T["loud"][idx]
    timed = T["has_sec"][idx]
    kind = np.concatenate([np.zeros(idx.size, np.int64), np.ones(int((~loud).sum()), np.int64),
                           np.full(int(timed.sum()), 2, np.int64)])
    rows = np.concatenate([idx, idx[~loud], idx[timed]])
    a = np.concatenate([age, age[~loud], age[timed]])
    y = np.concatenate([loud.astype(float), T["silent"][idx][~loud].astype(float), T["y_sec"][idx][timed]])
    g = np.concatenate([T["g_loud"][idx], T["g_silent"][idx][~loud], T["g_sec"][idx][timed]])
    sg = np.concatenate([T["sg_loud"][idx], T["sg_silent"][idx][~loud], T["g_sec"][idx][timed]])
    tickets = T["ticket"][rows]
    a = a * _shares(np.where(tickets >= 0, kind * record.tickets + tickets, -1), a, 3 * record.tickets)
    codes = [np.zeros(rows.size, np.int64)] + [T[f"d{d}"][rows] for d in range(1, 5)]
    keys_ = [kind * record.prefixes[d] + codes[d] for d in range(5)]
    sums = [{field: np.bincount(keys_[d], weights=w, minlength=3 * record.prefixes[d])
             for field, w in (("weight", a), ("outcome", a * y), ("guess", a * g))} for d in range(5)]
    result = {}
    names = ("loud", "silent", "seconds")
    # Each entry's own level and usual: both ladders at its cell, with the record's averages on top.
    totals = sums[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        level = (totals["outcome"] / totals["weight"])[kind]
        usual = (totals["guess"] / totals["weight"])[kind]
    for d in range(5):
        kappa = policy["model_worth"] if d == 3 else policy["rung_worth"]
        weight = sums[d]["weight"][keys_[d]]
        positive = weight > 0
        level = np.where(positive, (sums[d]["outcome"][keys_[d]] + kappa * level) / (weight + kappa), level)
        usual = np.where(positive, (sums[d]["guess"][keys_[d]] + kappa * usual) / (weight + kappa), usual)
    rate = kind < 2
    held = np.clip(usual, 1e-4, 1 - 1e-4)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = np.where(rate, np.log(held) - np.log1p(-held), usual)
    gap = sg - scaled
    high = (gap > 0).astype(np.int64)
    group = kind * 2 + high
    groups = {field: np.bincount(group, weights=w, minlength=6)
              for field, w in (("weight", a), ("gap", a * gap), ("outcome", a * y), ("level", a * level))}
    miss = None
    for k, name in enumerate(names):
        def part(h):
            return {field: float(groups[field][2 * k + h]) for field in groups}
        gamma = forecaster_weight(name, part(1), part(0), policy)["gamma"]
        if name == "seconds":
            # The miss each past run would have left: its result against its own corrected prediction.
            miss = np.where(kind == 2, (y - (level + gamma * gap)) ** 2, 0.)
        rungs = []
        for d in range(5):
            code = run["codes"][d] if d else 0
            entry = dict.fromkeys(rung_sums(name), 0.)
            if code >= 0:
                key = k * record.prefixes[d] + code
                for field in ("weight", "outcome", "guess"):
                    entry[field] = float(sums[d][field][key])
                if name == "seconds":
                    entry["miss"] = float(np.sum((a * miss)[keys_[d] == key]))
            rungs.append(entry)
        result[name] = (gamma, rungs)
    return result


def counts(run, half_life, policy):
    """``(n_c, n_l, n_s)``: the counted weight of each outcome for the run's cell, as ``success.outcome_counts``."""
    if not run["base"].size:
        return 0., 0., 0.
    from .evidence import DISCOUNT_KEYS
    delta = np.asarray([1.] + [float(policy[DISCOUNT_KEYS[r]]) for r in DISCOUNT_ORDER[1:]])[run["relation"]]
    w = run["base"] * 0.5 ** (np.maximum(0., run["days"]) / half_life) * delta
    counted = delta > 0
    w, outcomes, tickets = w[counted], run["outcomes"][counted], run["tickets"][counted]
    w = w * _shares(tickets, w, max(1, int(run["tickets"].max(initial=-1)) + 1))
    totals = np.bincount(outcomes, weights=w, minlength=3)
    return float(totals[0]), float(totals[1]), float(totals[2])


def duration_runs(run, half_life):
    """``[(weight, active seconds)]`` of the same cell's candidates, as ``semantic.duration_observations``."""
    from .evidence import ticket_shares
    own = [(base * 0.5 ** (max(0., days) / half_life), ticket, active) for base, days, ticket, active in run["own"]]
    shares = ticket_shares([(ticket, value) for value, ticket, _ in own])
    return [(value * share, active) for (value, _, active), share in zip(own, shares) if active is not None]


def caught_record(record, run, half_life):
    """``F`` and ``C`` of all hosts and of the run's host, as ``capability.caught_record`` read them."""
    T, t = record.table, run["t"]
    sel = (T["vf"] < t) & (t <= T["vt"]) & T["caught"] & T["finished"] & (T["loud"] | T["silent"])
    idx = np.flatnonzero(sel)
    age = 0.5 ** (np.maximum(0., (t - T["t"][idx]) / DAY_US) / half_life)
    age = age * _shares(T["ticket"][idx], age, record.tickets)
    loud = T["loud"][idx]
    mine = T["d1"][idx] == run["host"]
    return {"host": None, "F_all": float(math.fsum(age)), "C_all": float(math.fsum(age[loud])),
            "F_host": float(math.fsum(age[mine])), "C_host": float(math.fsum(age[mine & loud]))}


def _centre(prediction, spread):
    """The centre of a belief's start: the bell on the grid whose average rate is ``prediction``.

    The equation of ``success._grid_belief``, solved by safeguarded Newton steps.
    """
    from .success import CENTRE_BRACKET, GRID_P, GRID_X
    lo, hi = CENTRE_BRACKET
    c = min(hi, max(lo, (math.log(prediction) - math.log1p(-prediction)) * math.sqrt(1. + math.pi * spread ** 2 / 8.)))
    for _ in range(200):
        z = GRID_X - c
        log_q = -z * z / (2. * spread ** 2)
        q = np.exp(log_q - log_q.max())
        total = float(q.sum())
        mean = float(q @ GRID_P) / total
        f = mean - prediction
        if f == 0.:
            return c
        if f < 0:
            lo = c
        else:
            hi = c
        slope = (float((q * z) @ GRID_P) / total - mean * float(q @ z) / total) / spread ** 2
        step = c - f / slope if slope > 0 else None
        if step is None or not lo < step < hi:
            step = (lo + hi) / 2.
        if abs(step - c) <= 1e-13 * max(1., abs(c)) or hi - lo <= 1e-13 * max(1., abs(c)):
            return step
        c = step
    return c


def rate_mean(start, n_yes, n_no, centres):
    """``E`` of one rate's belief from its start (``prediction.value``) and its counted runs."""
    from .success import GRID_LOG_NO, GRID_LOG_YES, GRID_P, GRID_X
    prediction, spread = start["prediction"], start["s"]
    if n_yes == 0 and n_no == 0:
        # With no counted run the belief is its start, whose average is the prediction.
        return prediction
    key = prediction, spread
    if key not in centres:
        centres[key] = _centre(prediction, spread)
    c = centres[key]
    log_b = -(GRID_X - c) ** 2 / (2. * spread ** 2) + n_yes * GRID_LOG_YES + n_no * GRID_LOG_NO
    b = np.exp(log_b - log_b.max())
    return float(b @ GRID_P) / float(b.sum())


class Scorer:
    """Every scored run's score under one policy, with what does not change between values cached."""

    def __init__(self, record):
        self.record = record
        self.runs = record.runs
        self.ladders, self.counts, self.durations, self.caught, self.centres = {}, {}, {}, {}, {}

    def _ladders(self, i, sampling):
        key = self.runs[i]["group"], sampling["half_life_days"], sampling["rung_worth"], sampling["model_worth"]
        if key not in self.ladders:
            self.ladders[key] = ladders(self.record, self.runs[i], sampling["half_life_days"], sampling)
        return self.ladders[key]

    def starts(self, i, sampling, kinds):
        from .prediction import value
        run, built = self.runs[i], self._ladders(i, sampling)
        return {kind: value(kind, run["guesses"][kind], built[kind][1], built[kind][0], sampling) for kind in kinds}

    def outcome(self, sampling, selected=None):
        result = []
        for i in (range(len(self.runs)) if selected is None else selected):
            run = self.runs[i]
            start = self.starts(i, sampling, ("loud", "silent"))
            key = (run["group"], sampling["half_life_days"],
                   *(sampling[k] for k in ("effort_discount", "model_discount", "agent_discount")))
            if key not in self.counts:
                self.counts[key] = counts(run, sampling["half_life_days"], sampling)
            clean, loud, silent = self.counts[key]
            theta = rate_mean(start["loud"], loud, clean + silent, self.centres)
            share = rate_mean(start["silent"], silent, clean, self.centres)
            result.append(outcome_score(theta, share, run["outcome"]))
        return np.asarray(result)

    def time(self, sampling, selected):
        """Each timed run's score and its miss ``ln(actual / prediction)``."""
        from .success import amount_belief
        scores, misses = [], []
        for i in selected:
            run = self.runs[i]
            start = self.starts(i, sampling, ("seconds",))["seconds"]
            key = run["group"], sampling["half_life_days"]
            if key not in self.durations:
                self.durations[key] = duration_runs(run, sampling["half_life_days"])
            belief = amount_belief(start, self.durations[key], sampling, "seconds")
            tau = start["D"] * math.sqrt(sampling["scatter_share"])
            scores.append(time_score(run["y"], belief["centre"], belief["spread"], tau))
            misses.append(run["y"] - math.log(start["prediction"]))
        return np.asarray(scores), np.asarray(misses)

    def caught_scores(self, sampling, capability, selected):
        from .capability import caught_share
        result = []
        for i in selected:
            run = self.runs[i]
            key = run["group"], sampling["half_life_days"]
            if key not in self.caught:
                self.caught[key] = caught_record(self.record, run, sampling["half_life_days"])
            d = caught_share(self.caught[key], run["epsilon"], capability)["d_host"]
            result.append(caught_score(d, run["outcome"]))
        return np.asarray(result)


def pairs(runs, timed):
    """Differences in log active time between consecutive timed runs of one cell on one ticket."""
    grouped = {}
    for i in timed:
        run = runs[i]
        if run["ticket"] is not None:
            grouped.setdefault((*run["pair"], run["ticket"]), []).append(i)
    result = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda i: (runs[i]["observed"], -i))
        result.extend(runs[b]["y"] - runs[a]["y"] for a, b in zip(ordered, ordered[1:]))
    return result


def _better(candidate, best, default):
    """Whether ``(value, gain, se)`` beats ``best``: a higher gain, or the same gain nearer the default."""
    if best is None or best[1] is None:
        return True
    if candidate[1] is None:
        return False
    return candidate[1] > best[1] or (candidate[1] == best[1] and abs(candidate[0] - default) < abs(best[0] - default))


def estimate(engine, *, record=None):
    """Re-estimate the refinable constants from one snapshot of the record; nothing is written.

    The result is a pure function of the record and the configuration: the 2,000 most recent
    scored runs, each predicted from what was observed before its decision. Each constant
    starts from its value before any refinement (``config.unrefined``).
    """
    record = record if record is not None else Record(engine)
    config = engine.config
    tables = {"sampling": dict(config.data["sampling"]), "capability": dict(config.data["capability"])}
    for section, key in KEYS:
        tables[section][key] = config.unrefined[section, key]
    sampling, capability = tables["sampling"], tables["capability"]
    scorer = Scorer(record)
    runs = record.runs
    everyone = list(range(len(runs)))
    selections = {"outcome": everyone,
                  "accepted": [i for i in everyone if runs[i]["outcome"] != "loud"],
                  "time": [i for i in everyone if runs[i]["timed"]],
                  "caught": [i for i in everyone if runs[i]["outcome"] != "clean"]}
    cache = {}

    def scores(score):
        """The scores of the selected runs under the policy as it stands, and the misses for time."""
        signature = (score, tuple(sorted(sampling.items())), tuple(sorted(capability.items())))
        if signature not in cache:
            chosen = selections[score]
            if score in ("outcome", "accepted"):
                cache[signature] = scorer.outcome(sampling, chosen), None
            elif score == "time":
                cache[signature] = scorer.time(sampling, chosen)
            else:
                cache[signature] = scorer.caught_scores(sampling, capability, chosen), None
        return cache[signature]

    base = defaults()
    entries = {}
    for section, key, values, score in CONSTANTS:
        if (section, key) in config.file_keys:
            # A constant the configuration file sets is never tried.
            entries[key] = dict.fromkeys(ENTRY_FIELDS)
            continue
        table, default = tables[section], base[section, key]
        current, n = table[key], len(selections[score])
        now, misses = scores(score)
        if values == MEASURED:
            # The default and the value the record measures; the miss does not depend on either.
            values = [default]
            miss = measured_miss(misses) if n else None
            measured = miss if key == "duration_miss" else None
            if key == "scatter_share" and miss:
                tau = measured_scatter(pairs(runs, selections["time"]))
                measured = min(1., tau ** 2 / miss ** 2) if tau else None
            if measured is not None and measured > 0:
                values.append(measured)
        best = passing = None
        for tried in values:
            if tried == current:
                continue
            table[key] = tried
            _, gain, se, moves = rule(scores(score)[0] - now)
            table[key] = current
            candidate = tried, gain, se
            if _better(candidate, best, default):
                best = candidate
            if moves and _better(candidate, passing, default):
                passing = candidate
        if passing is not None:
            table[key] = passing[0]
        shown = passing or best
        entries[key] = {"value": float(table[key]), "n": n,
                        "gain": None if shown is None else shown[1], "se": None if shown is None else shown[2]}
    return {"format": FORMAT, "rows": record.rows_digest, "scored_runs": record.total, "constants": entries}


# ---------------------------------------------------------------------------------------------
# When it runs


def receipt_format(config, decision_id):
    """The format label a stored receipt records, read without validating it; ``None`` when unreadable."""
    import json
    from .payload_store import get
    try:
        blob = get(config, "decisions", decision_id)
        value = json.loads(blob) if blob is not None else None
    except (HowHardError, ValueError, UnicodeError):
        return None
    return value.get("format") if isinstance(value, dict) else None


def _possible(engine, below=None):
    """An upper bound on the scored runs: finished runs whose receipt records ``howhard-decision-v11``.

    The format label is read from each named receipt without validating it, once per engine;
    the refinement counts the scored runs exactly. With ``below``, the count stops early,
    without reading a receipt, when even every finished run with a decision falls short of it.
    """
    routed = [row["decision_id"] for row in engine.rows if row["outcome"] in OUTCOMES and row.get("decision_id")]
    if below is not None and len(routed) < below:
        return len(routed)
    formats = engine.receipt_formats
    for decision_id in dict.fromkeys(routed):
        if decision_id not in formats:
            formats[decision_id] = receipt_format(engine.config, decision_id)
    return sum(formats[decision_id] == DECISION_FORMAT for decision_id in routed)


def publish(engine, result):
    """Write the result unless the cached one covers more scored runs; the only admitted write."""
    from . import storage
    from .files import atomic_write
    validate(result)
    with storage.locked(engine.config, operation="refined constants"):
        cached = read(engine.config)
        if cached is not None and result["scored_runs"] < cached["scored_runs"]:
            return {"status": "dropped", "scored_runs": result["scored_runs"], "cached_scored_runs": cached["scored_runs"]}
        atomic_write(path(engine.config), canonical(result) + b"\n")
    engine.refresh()
    moved = {key: result["constants"][key]["value"] for section, key in KEYS
             if result["constants"][key]["value"] not in (None, engine.config.unrefined[section, key])}
    return {"status": "published", "scored_runs": result["scored_runs"], "moved": moved}


def refine(engine, *, after):
    """Run the refinement after ``observe`` or ``flush`` when it is due; never raises.

    The record is read from one snapshot outside the write turn, and only publishing is
    admitted. The status says what happened.
    """
    try:
        from .request_state import capture
        cached = read(engine.config)
        needed = cached["scored_runs"] + RESCORE_RUNS if after == "observe" and cached is not None else 0
        possible = _possible(engine, below=max(MINIMUM_RUNS, needed))
        if possible < MINIMUM_RUNS:
            return {"status": "not-due", "reason": f"fewer than {MINIMUM_RUNS} scored runs"}
        if after == "observe" and cached is not None and possible < cached["scored_runs"] + RESCORE_RUNS:
            return {"status": "not-due", "reason": f"the cached result covers fewer than {RESCORE_RUNS} runs less"}
        if after == "flush" and cached is not None and cached["rows"] == digest(engine.raw_rows):
            return {"status": "not-due", "reason": "the cached result names this record"}
        reader, _ = capture(engine)
        record = Record(reader)
        if record.total < MINIMUM_RUNS:
            return {"status": "not-due", "reason": f"fewer than {MINIMUM_RUNS} scored runs"}
        if after == "observe" and cached is not None and record.total < cached["scored_runs"] + RESCORE_RUNS:
            return {"status": "not-due", "reason": f"the cached result covers fewer than {RESCORE_RUNS} runs less"}
        if after == "flush" and cached is not None and cached["rows"] == record.rows_digest:
            return {"status": "not-due", "reason": "the cached result names this record"}
        return publish(engine, estimate(reader, record=record))
    except Exception as exc:  # A refinement never fails the observation or the flush that started it.
        return {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
