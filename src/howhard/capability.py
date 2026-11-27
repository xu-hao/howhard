"""Task-conditioned ordinal threshold reader over the shared execution evidence."""

from copy import deepcopy
from datetime import datetime
import math
import re

import numpy as np

from .context import DECISION_FORMAT, LEGACY_DECISION_V7, LEGACY_DECISION_V8, LEGACY_DECISION_V9, LEGACY_DECISION_V10, digest
from .llm import LLMError, SETTINGS
from .validation import HowHardError, keys, number, text

FORMAT = "howhard-capability-v6"
LEGACY_FORMAT_V5 = "howhard-capability-v5"
LEGACY_FORMAT_V4 = "howhard-capability-v4"
LEGACY_FORMAT_V3 = "howhard-capability-v3"
LEGACY_FORMAT_V2 = "howhard-capability-v2"
LEGACY_FORMAT_V1 = "howhard-capability-v1"
OVERRUN_GRACE_RATIO = 1.5
# Why a widening round ran, and why widening ended. v11 reads admission tiers; older receipts
# recorded the silent cap. The other three stop labels are shared.
WIDENING_TRIGGER, LEGACY_WIDENING_TRIGGER = "all-excluded", "silent-cap-unmet"
WIDENING_STOPS = ("pickable", "forecast-budget", "round-limit", "exhausted")
EFFORT_ORDER = {e: i for i, e in enumerate(("low", "medium", "high", "xhigh", "max", "ultra"))}
DEFAULTS = {"max_forecast_cells": 12, "widening_rounds": 2,
            "min_clear_probability": .5, "reliability_probability": .9,
            "overpowered_gain_margin": .05, "consistency_margin": .2,
            "observation_noise": .05, "order_penalty": .15,
            "loud_weight": .75, "silent_weight": 1., "unknown_cause_weight": .5,
            "caught_share_worth": 12., "reserved_forecast_cells": 2, "unproven_clear_probability": .2}
# Capability formats before v6 froze their policy without the host-aware band settings.
HOST_AWARE_SETTINGS = ("caught_share_worth", "reserved_forecast_cells", "unproven_clear_probability")
RESERVED_REASONS = ("unproven-family", "missing-agent")


def cell_key(cell):
    return "cell:" + digest([cell["model"], cell["effort"]])[:24]


def validate_policy(policy, *, host_aware=True):
    """Check a capability policy. ``host_aware=False`` reads one frozen before the host-aware band."""
    keys(policy, [k for k in DEFAULTS if host_aware or k not in HOST_AWARE_SETTINGS], where="capability")
    for k, v in policy.items():
        if k in ("max_forecast_cells", "widening_rounds", "reserved_forecast_cells"):
            if type(v) is not int or v < (1 if k == "max_forecast_cells" else 0):
                raise HowHardError(f"capability.{k}: expected {'positive' if k == 'max_forecast_cells' else 'nonnegative'} integer")
        elif k == "caught_share_worth":
            # A count of failed runs' worth, so it may exceed 1.
            number(v, f"capability.{k}", positive=True)
        else:
            number(v, f"capability.{k}", nonnegative=True)
            if k != "order_penalty" and v > 1:
                raise HowHardError(f"capability.{k}: expected probability/weight in [0, 1]")
    if host_aware and policy["reserved_forecast_cells"] >= policy["max_forecast_cells"]:
        raise HowHardError("capability.reserved_forecast_cells: expected an integer below max_forecast_cells")
    if not 0 < policy["min_clear_probability"] < policy["reliability_probability"] < 1:
        raise HowHardError("capability: require 0 < min_clear_probability < reliability_probability < 1")
    if not 0 < policy["observation_noise"] < .5:
        raise HowHardError("capability.observation_noise: expected value in (0, .5)")
    return policy


def request(payload, policy):
    result = deepcopy(payload)
    result["kind"] = "filter"
    result["cells"] = [{**c, "key": cell_key(c)} for c in result["cells"]]
    result["filter_policy"] = deepcopy(policy)
    result["contract"] = {
        "filter": "Return {order: [opaque cell keys], lower: key, upper: key}. Order EVERY supplied key exactly once, weakest to strongest FOR THIS TASK AND PROJECT only. Inclusive lower/upper delimit a nonempty fit band. Known efforts within a model are ordinal low < medium < high < xhigh < max < ultra; keep custom controls in your proposed within-model order. No universal model ranking. Exclude only for capability fit, not price or silent-risk policy. Keep the band within max_forecast_cells where justified; never omit inventory keys.",
        "memory_filters": "Array of up to eight {id,version,filter}: full conditional order/cut-point adjustments for relevant ACTIVE retrieved memories, including read-only execution experts. Interrupted executions (such as usage_limit, provider_error, host_error or cancelled) are operational interruptions, not capability failures. Cite exact id/version. No invented outcomes. Omitted experts have no adjustment.",
        "reason": "Brief explanation of task-specific placement and cut points.",
        "request_more": payload["contract"]["request_more"]}
    return result


def baseline_request(payload):
    if "no_memory_filter" in payload:
        return None
    result = {k: deepcopy(payload[k]) for k in ("kind", "prompt_version", "task_type", "risk_class")}
    result["context"] = deepcopy(payload.get("prototype_context", payload["context"]))
    result.update(phase="baseline", cells=[{k: c[k] for k in ("key", "family", "effort", "model")} for c in payload["cells"]],
                  filter_policy=deepcopy(payload["filter_policy"]),
                  contract="Return ONLY {filter: {order: [key, ...], lower: key, upper: key}, reason: string}. " + payload["contract"]["filter"] + " No memory, evidence, beliefs or retrieval is available in this independent call.")
    if "refilter" in payload:
        result["refilter"] = deepcopy(payload["refilter"])
    return result


def validate_filter(value, cells):
    keys(value, ["order", "lower", "upper"], where="capability filter")
    order = value["order"]
    known = {c["key"]: c for c in cells}
    if (not isinstance(order, list) or any(not isinstance(k, str) or k not in known for k in order)
            or len(order) != len(set(order)) or set(order) != set(known)):
        raise LLMError("capability filter: order must cover every cell exactly once; missing, duplicate or unknown cell key")
    if not isinstance(value["lower"], str) or not isinstance(value["upper"], str) or value["lower"] not in known or value["upper"] not in known or order.index(value["lower"]) > order.index(value["upper"]):
        raise LLMError("capability filter: lower/upper must delimit a nonempty fit band")
    last = {}
    for key in order:
        cell = known[key]
        ordinal = EFFORT_ORDER.get(cell["effort"])
        if ordinal is not None:
            if ordinal < last.get(cell["model"], -1):
                raise LLMError("capability filter: effort order within a model must be ordinal")
            last[cell["model"]] = ordinal
    return value


def validate_response(response, payload, snapshot):
    from .semantic import visible_memories
    if "no_memory_filter" in payload:
        keys(response, ["memory_filters", "reason"], ["filter"], where="capability filter response")
        if "filter" in response and response["filter"] != payload["no_memory_filter"]:
            raise LLMError("Memory reader must not rewrite the isolated baseline filter")
        response = {**deepcopy(response), "filter": deepcopy(payload["no_memory_filter"])}
    else:
        keys(response, ["filter", "memory_filters", "reason"], where="capability filter response")
    text(response["reason"], "capability filter reason")
    validate_filter(response["filter"], payload["cells"])
    items = response["memory_filters"]
    if not isinstance(items, list) or len(items) > SETTINGS["memory_limit"]:
        raise LLMError("memory_filters: expected at most eight adjustments")
    known = {(m["id"], m["version"]): m for m in snapshot["memories"]}
    visible, seen = visible_memories(payload), set()
    for item in items:
        keys(item, ["id", "version", "filter"], where="memory_filter")
        text(item["id"], "memory_filter.id")
        if type(item["version"]) is not int:
            raise LLMError("memory_filter.version: expected integer")
        pair = item["id"], item["version"]
        if pair not in known or pair not in visible or pair in seen or not known[pair]["active"]:
            raise LLMError("memory_filters: unknown, unseen, stale, inactive or duplicate memory")
        seen.add(pair)
        validate_filter(item["filter"], payload["cells"])
    return response


def prior_cdf(value, policy, *, linear_prefix=False, linear_tail=False):
    """Smoothed threshold prior, with legacy variants for frozen receipts."""
    n = len(value["order"])
    lo, hi = value["order"].index(value["lower"]), value["order"].index(value["upper"])
    minimum, reliable = policy["min_clear_probability"], policy["reliability_probability"]
    if linear_prefix:
        before = np.linspace(minimum * .2, minimum * .8, lo).tolist()
    else:
        # Each step below the position just before the lower cut halves its CDF.
        before = (minimum * .8 * 0.5 ** np.arange(lo - 1, -1, -1, dtype=float)).tolist()
    fit = np.linspace(max(minimum, reliable - .2), reliable, hi-lo+1).tolist() if hi > lo else [reliable]
    # Above the upper cut, the maximum possible gain stays below the policy margin.
    # Concentrate the tail geometrically so conditioning cannot stretch it across large inventories.
    tail = min((1-reliable)*.5, policy["overpowered_gain_margin"]*.5)
    after_count = n - 1 - hi
    if linear_tail:
        after = np.linspace(reliable, reliable+tail, n-hi).tolist()[1:]
    elif after_count:
        steps = np.arange(1, after_count + 1, dtype=float)
        after = (reliable + tail * (1.0 - 0.5 ** steps) / (1.0 - 0.5 ** after_count)).tolist()
    else:
        after = []
    return dict(zip(value["order"], before+fit+after))


def _decision_forecast_seconds(engine, row, receipts=None):
    from .decisions import scored_receipt
    from .semantic import panel_lookup
    receipt = scored_receipt(engine.config, row, receipts)
    if receipt is None:
        return None
    pair = row["family"], row["resolved_effort"]
    panels = {k: panel_lookup(p) for k, p in receipt["semantic"]["worlds"].items()}
    if pair not in panels.get("llm", {}):
        return None
    return float(sum(e["weight"] * panels[e["key"]][pair]["seconds"] for e in receipt["semantic"]["beliefs"]["experts"]))


def _strength(row, policy):
    """How strongly a run's outcome bears on the capability threshold, given its recorded cause."""
    if row["outcome"] == "clean":
        return 1.
    strength = policy[row["outcome"]+"_weight"]
    if row.get("cause") != "capability":
        strength *= policy["unknown_cause_weight"]
        if re.search(r'\bcapability_caused\s*[:=]\s*false\b', row.get("note", ""), re.I):
            strength = 0.
    return strength


def observations(engine, payload, candidates, receipts=None):
    """The scored candidates whose cell is in the inventory: ``m * sim * strength * a``, limited per ticket.

    ``receipts`` is the optional cache of loaded decisions that the candidate scoring filled.
    """
    from . import evidence
    known = {cell_key(c) for c in payload["cells"]}
    rows = {r["id"]: r for r in engine.rows}
    policy = payload["filter_policy"]
    counted = []
    for item in candidates["candidates"]:
        row = rows[item["id"]]
        if cell_key(row) not in known:
            continue
        strength = _strength(row, policy)
        counted.append((item, row, strength, evidence.weight(item, strength)))
    # The reader has no relatedness discount; its per-ticket limit runs over its own runs.
    shares = evidence.ticket_shares([(item["ticket"], value) for item, _, _, value in counted])
    return [{"id": row["id"], "key": cell_key(row), "outcome": row["outcome"], "cause": row.get("cause"),
             "active_seconds": item["active_seconds"],
             "forecast_active_seconds": _decision_forecast_seconds(engine, row, receipts),
             "strength": strength, "ticket_share": share, "weight": value * share}
            for (item, row, strength, value), share in zip(counted, shares)]


def caught_record(engine, host, candidates):
    """``F`` and ``C``: the failed routed runs on record, of all hosts and of the asking host.

    A failed run is a finished loud or silent run whose decision receipt holds a forecast
    for the cell that ran. It counts at its age factor ``a = 0.5 ** (days / half_life_days)``,
    limited so that one ticket's failed runs count as one run at most. ``F`` sums ``a`` and
    ``C`` sums ``a`` over the loud runs, which were caught before acceptance. The host is
    the name in the run's receipt; a receipt with no name, or older than v11, belongs to
    the one unnamed host. The age is measured to the request's prepare time.
    """
    from . import evidence
    from .decisions import exists, host as receipt_host, load
    prepared = datetime.fromisoformat(candidates["prepared_at"])
    # A receipt is immutable under its ID, so its host and forecast cells are read once per engine.
    known = engine.receipt_hosts
    failed = []
    for row in engine.rows:
        decision_id = row.get("decision_id")
        if row["outcome"] not in ("loud", "silent") or not decision_id:
            continue
        if decision_id not in known:
            if not exists(engine.config, decision_id):
                continue
            receipt = load(engine.config, decision_id, for_scoring=True)
            known[decision_id] = None if receipt is None else (receipt_host(receipt), frozenset(
                (c["family"], c["effort"]) for c in receipt["semantic"]["worlds"]["llm"]))
        if known[decision_id] is None or (row["family"], row["resolved_effort"]) not in known[decision_id][1]:
            continue
        days = max(0.0, (prepared - datetime.fromisoformat(row["observed_at"])).total_seconds() / 86400.0)
        failed.append((row, known[decision_id][0], evidence.age_factor(days, candidates["half_life_days"])))
    shares = evidence.ticket_shares([(row["ticket"], age) for row, _, age in failed])
    counted = [(row["outcome"] == "loud", name, age * share) for (row, name, age), share in zip(failed, shares)]

    def total(rows):
        return float(math.fsum(rows))
    return {"host": host,
            "F_all": total(a for _, _, a in counted), "C_all": total(a for loud, _, a in counted if loud),
            "F_host": total(a for _, name, a in counted if name == host),
            "C_host": total(a for loud, name, a in counted if loud and name == host)}


def _shrunk(caught, failed, prior, worth):
    # (caught + worth * prior) / (failed + worth), written so that an empty record gives ``prior`` exactly.
    return prior + (caught - failed * prior) / (failed + worth)


def caught_share(record, epsilon, policy):
    """The asking host's caught share ``d`` and the two band edges it sets.

    ``d_0 = (1 - 2 * epsilon) / (1 - epsilon)`` is the share at which both edges equal the
    configured ones. ``d(all)`` shrinks the caught share of all hosts towards ``d_0`` and
    ``d(host)`` shrinks the asking host's towards ``d(all)``, each by ``caught_share_worth``
    failed runs' worth, so an empty record gives ``d_0``.
    """
    worth = policy["caught_share_worth"]
    d_0 = (1.0 - 2.0 * epsilon) / (1.0 - epsilon)
    d_all = _shrunk(record["C_all"], record["F_all"], d_0, worth)
    d_host = _shrunk(record["C_host"], record["F_host"], d_all, worth)
    e, g = band_edges(d_host, epsilon, policy)
    return {**{k: record[k] for k in ("host", "F_all", "C_all", "F_host", "C_host")},
            "d_0": d_0, "d_all": d_all, "d_host": d_host, "e": e, "g": g}


def band_edges(d, epsilon, policy):
    """The lowest ``P_clear`` of a band cell, ``e``, and the upper margin, ``g``, at caught share ``d``.

    A cell too weak for the task fails. With ``q = 1 - P_clear``, ``q * (1 - d)`` of its
    results escape and ``1 - q * d`` are accepted; holding the escaped share at ``epsilon``
    gives ``e``. ``g`` shrinks to keep ``(1 - d) * g`` constant. At ``d >= d_0`` both are the
    configured values, and below it both tighten.
    """
    minimum, margin = policy["min_clear_probability"], policy["overpowered_gain_margin"]
    if d >= (1.0 - 2.0 * epsilon) / (1.0 - epsilon):
        return minimum, margin
    return (max(minimum, 1.0 - epsilon / (1.0 - d + epsilon * d)),
            min(margin, margin * epsilon / ((1.0 - epsilon) * (1.0 - d))))


def _unanchored_reference_seconds(constants, policy):
    rate = constants.get("lambda_per_hour", 0.0) if constants else 0.0
    if rate <= 0:
        return None
    gain_usd = max(1.0, constants.get("C_loud", 0.0)) * policy["overpowered_gain_margin"]
    return max(600.0, 3600.0 * gain_usd / rate)


def _timely_strengths(order, obs, *, fallback_seconds=None):
    positions = {k: i for i, k in enumerate(order)}
    strengths = []
    for item in obs:
        if item["outcome"] != "clean":
            strengths.append(0.0)
            continue
        active = item.get("active_seconds")
        if active is None or active <= 0:
            strengths.append(1.0)
            continue
        pos = positions[item["key"]]
        stronger = [
            r for r in obs
            if r["outcome"] == "clean"
            and positions[r["key"]] > pos
            and r["weight"] > .01
            and r.get("active_seconds") is not None
            and r["active_seconds"] > 0
        ]
        stronger_sec = None
        if stronger:
            w_sum = math.fsum(r["weight"] for r in stronger)
            stronger_sec = math.exp(
                math.fsum(r["weight"] * math.log(max(1.0, r["active_seconds"])) for r in stronger) / w_sum
            )
        forecast_sec = item.get("forecast_active_seconds")
        refs = [v for v in (forecast_sec, stronger_sec) if v is not None and v > 0]
        if not refs:
            if fallback_seconds is None or fallback_seconds <= 0:
                strengths.append(1.0)
                continue
            ref_sec = fallback_seconds
        else:
            ref_sec = min(refs)
        ratio = max(1.0, active) / max(1.0, ref_sec)
        strengths.append(1.0 if ratio <= OVERRUN_GRACE_RATIO else (OVERRUN_GRACE_RATIO / ratio) ** 2)
    return strengths


def _posterior(order, cdf, obs, policy, *, targets=None):
    probabilities = np.asarray([cdf[k] for k in order])
    probabilities = np.maximum.accumulate(probabilities)
    masses = np.diff(np.r_[0., probabilities, 1.])
    logs = np.log(np.maximum(masses, 1e-12))
    positions = {k: i for i, k in enumerate(order)}
    noise = policy["observation_noise"]
    for idx, row in enumerate(obs):
        q = np.where(np.arange(len(order)+1) <= positions[row["key"]], 1-noise, noise)
        z = (1.0 if row["outcome"] == "clean" else 0.0) if targets is None else targets[idx]
        logs += row["weight"] * (z * np.log(q) + (1.0 - z) * np.log(1.0 - q))
    peak = float(logs.max())
    normalizer = peak + math.log(float(np.exp(logs-peak).sum()))
    posterior = np.exp(logs-normalizer)
    # The running sum can round a few units in the last place past 1 once the mass above
    # the strongest cell vanishes; a probability of clearing never exceeds 1.
    cumulative = np.clip(np.cumsum(posterior)[:-1], 0., 1.)
    return dict(zip(order, cumulative.tolist())), normalizer


def _upper_cut(learned_order, probabilities, lo, policy, *, margin=None):
    n = len(learned_order)
    margin = policy["overpowered_gain_margin"] if margin is None else margin
    best = probabilities[learned_order[-1]]
    reliable = next((i for i in range(lo, n) if probabilities[learned_order[i]] >= policy["reliability_probability"]), None)
    anchor = lo if reliable is None else reliable
    return next((i for i in range(anchor, n) if best - probabilities[learned_order[i]] < margin), n - 1)


def _forecast_band(order, clear, timely, policy, *, independent_prior=None, share=None):
    """The fit band. ``share`` moves the clearing interval's two edges; the timely interval keeps the configured ones."""
    minimum = policy["min_clear_probability"]
    e, g = (minimum, policy["overpowered_gain_margin"]) if share is None else (share["e"], share["g"])
    lo_clear = next((i for i, key in enumerate(order) if clear[key] >= e), None)
    if lo_clear is None:
        # No cell reaches the lower edge: start at the first best cell when it reaches the floor.
        best = max(clear[key] for key in order)
        if best < minimum:
            raise HowHardError("capability filter: no cell reaches the required capability probability; supply a more capable inventory")
        lo_clear = next(i for i, key in enumerate(order) if clear[key] >= best)
    if independent_prior is not None and (share is None or share["d_host"] >= share["d_0"]):
        lo = order.index(independent_prior["lower"])
        hi = order.index(independent_prior["upper"])
        return order[lo:hi+1]

    hi_clear = _upper_cut(order, clear, lo_clear, policy, margin=g)

    lo_timely = next((i for i in range(lo_clear, len(order)) if timely[order[i]] >= minimum), None)
    if lo_timely is None:
        # The timely minimum is unattainable. Retain the first near-best timely
        # option among cells that can clear, for the fine reader's time comparison.
        hi_timely = _upper_cut(order, timely, lo_clear, policy)
        lo_timely = hi_timely
    else:
        hi_timely = _upper_cut(order, timely, lo_timely, policy)

    hi = max(hi_clear, hi_timely)
    if hi - lo_clear + 1 <= policy["max_forecast_cells"] or lo_timely <= hi_clear + 1:
        return order[lo_clear:hi+1]
    # Keep both threshold intervals. In the gap, clearing is already saturated
    # and timely clearing falls short of the minimum (or near-best fallback).
    return order[lo_clear:hi_clear+1] + order[lo_timely:hi_timely+1]


def _core(band, policy):
    return band[:policy["max_forecast_cells"] - policy["reserved_forecast_cells"]]


def _reserved(cells, core, share, proven, policy):
    """The cells that take a reserved slot, in slot order.

    A slot buys one forecast for a family the core leaves out, at its cell nearest to the
    core in the order (the stronger one on a tie). Unproven families, with no run among
    the candidate runs, come first, each on its own, at the bar ``unproven_clear_probability``.
    A proven family whose agent has no core cell takes a slot on behalf of that agent, at
    the lower edge ``e``: at most one slot per missing agent, given to its qualifying cell
    nearest to the core. An agent whose unproven family already took a slot is served.
    Nearest first within each group.
    """
    slots, inside = policy["reserved_forecast_cells"], set(core)
    positions = [c["position"] for c in cells if c["key"] in inside]
    families = {c["family"] for c in cells if c["key"] in inside}
    agents = {c["agent"] for c in cells if c["key"] in inside}
    nearest = {}
    for c in cells:
        if not slots or c["family"] in families:
            continue
        rank = min(abs(c["position"] - p) for p in positions), -c["position"]
        if c["family"] not in nearest or rank < nearest[c["family"]][0]:
            nearest[c["family"]] = rank, c
    groups = {reason: [] for reason in RESERVED_REASONS}
    for rank, c in nearest.values():
        if c["family"] not in proven:
            reason, bar = "unproven-family", policy["unproven_clear_probability"]
        elif c["agent"] not in agents:
            reason, bar = "missing-agent", share["e"]
        else:
            continue
        if c["P_clear"] >= bar:
            groups[reason].append((rank, c["agent"], {"key": c["key"], "reason": reason, "bar": bar}))
    taken, served = [], set()
    for reason in RESERVED_REASONS:
        for _, agent, item in sorted(groups[reason], key=lambda entry: entry[0]):
            if len(taken) == slots or (reason == "missing-agent" and agent in served):
                continue
            taken.append(item)
            served.add(agent)
    return taken


def forecast_cells(state):
    """The cells of a new band that get a forecast: the core, the reserved cells, and the
    band's next cells beyond the core for each slot that no family took."""
    taken = [item["key"] for item in state["reserved"]]
    beyond = [k for k in state["band"][len(state["core"]):] if k not in taken]
    return state["core"] + taken + beyond[:max(0, state["policy"]["reserved_forecast_cells"] - len(taken))]


def widening_order(state, forecasted):
    """The stronger cells with no forecast, next first: band cells beyond the core, then overpowered cells."""
    beyond = [k for k in state["band"][len(state["core"]):] if k not in forecasted]
    return beyond + [c["key"] for c in state["cells"] if c["classification"] == "overpowered" and c["key"] not in forecasted]


def _classification(position, lo, hi, in_band):
    if position < lo:
        return "underpowered"
    if position > hi:
        return "overpowered"
    return "fit" if in_band else "intermediate"


def _learn_order(order, cdf, obs, cells, policy):
    """Local constrained posterior search; never a global model leaderboard."""
    initial = list(order)
    known = {c["key"]: c for c in cells}
    # Cross-model comparisons require outcomes at both cells. Unseen placements
    # retain the prior; threshold evidence can still transfer through them.
    observed = {r["key"] for r in obs if r["weight"] > .01}
    def objective(candidate):
        # Prior threshold masses belong to ordinal positions, not model identities.
        prior = dict(zip(candidate, [cdf[k] for k in initial]))
        _, evidence = _posterior(candidate, prior, obs, policy)
        positions = {k: i for i, k in enumerate(initial)}
        distance = sum(abs(i-positions[k]) for i, k in enumerate(candidate)) / (2*max(1, len(candidate)-1))
        return evidence - policy["order_penalty"] * distance
    score = objective(order)
    while True:
        best, best_score = None, score
        indices = [i for i, k in enumerate(order) if k in observed]
        for i, j in zip(indices, indices[1:]):
            a, b = order[i], order[j]
            if known[a]["model"] == known[b]["model"]:
                continue
            trial = list(order)
            trial[i], trial[j] = b, a
            # Swapping observed placements across unseen cells may not reorder any
            # model's effort chain, including custom controls fixed by this prior.
            valid = all([k for k in trial if known[k]["model"] == model] ==
                        [k for k in initial if known[k]["model"] == model]
                        for model in {known[a]["model"], known[b]["model"]})
            if not valid:
                continue
            value = objective(trial)
            if value > best_score + 1e-10:
                best, best_score = trial, value
        if best is None:
            return order
        order, score = best, best_score


def _pav_along(order, raw_cdf):
    blocks = []
    for k in order:
        blocks.append(([k], float(raw_cdf[k])))
        while len(blocks) > 1 and blocks[-2][1] > blocks[-1][1]:
            b, a = blocks.pop(), blocks.pop()
            blocks.append((a[0] + b[0], (len(a[0]) * a[1] + len(b[0]) * b[1]) / (len(a[0]) + len(b[0]))))
    return {k: p for group, p in blocks for k in group}


def aggregate(engine, payload, response, *, observed, covered_priors=None, agents=None, caught=None, candidates=None):
    from .semantic import normalize_support
    policy = payload["filter_policy"]
    baseline = response["filter"]
    base_cdf = prior_cdf(baseline, policy)
    prior_sources = None
    if covered_priors:
        overrides = {k: float(v) for k, v in covered_priors.items() if k in base_cdf}
        if overrides and len(overrides) < len(base_cdf):
            merged_llm = _pav_along(baseline["order"], {**base_cdf, **overrides})
            if merged_llm != base_cdf:
                base_cdf = merged_llm
                prior_sources = overrides
    worlds = {"filter:llm": base_cdf}
    filters = {"filter:llm": deepcopy(baseline)}
    for item in response["memory_filters"]:
        key = f"filter:memory:{item['id']}:{item['version']}"
        filters[key] = deepcopy(item["filter"])
        worlds[key] = prior_cdf(item["filter"], policy)
    belief = deepcopy(payload["beliefs"])
    belief["experts"] = [e for e in belief["experts"] if e["key"] in worlds]
    normalize_support(belief["experts"])
    weights = {e["key"]: e["weight"] for e in belief["experts"]}
    # Consensus positions, constrained by the LLM's within-model effort chains.
    rank = {k: sum(weights[e] * filters[e]["order"].index(k) for e in worlds) for k in baseline["order"]}
    order = sorted(baseline["order"], key=lambda k: (rank[k], baseline["order"].index(k)))
    chains = {}
    for k in baseline["order"]:
        cell = next(c for c in payload["cells"] if c["key"] == k)
        chains.setdefault(cell["model"], []).append(k)
    # Weighted complete orders are consistent for known ordinal efforts; custom
    # controls retain the baseline's within-model order through a topological merge.
    order = []
    while any(chains.values()):
        model = min((m for m in chains if chains[m]), key=lambda m: (rank[chains[m][0]], baseline["order"].index(chains[m][0])))
        order.append(chains[model].pop(0))
    cdf = {k: sum(weights[e]*worlds[e][k] for e in worlds) for k in order}
    # Isotonic CDF projection by pooled adjacent violators.
    blocks = []
    for k in order:
        blocks.append(([k], cdf[k]))
        while len(blocks) > 1 and blocks[-2][1] > blocks[-1][1]:
            b, a = blocks.pop(), blocks.pop()
            blocks.append((a[0]+b[0], (len(a[0])*a[1]+len(b[0])*b[1])/(len(a[0])+len(b[0]))))
    cdf = {k: p for group, p in blocks for k in group}
    obs = deepcopy(observed)
    learned_order = _learn_order(order, cdf, obs, payload["cells"], policy) if obs else order
    cdf = dict(zip(learned_order, [cdf[k] for k in order]))
    probabilities, _ = _posterior(learned_order, cdf, obs, policy)
    fallback_sec = _unanchored_reference_seconds(payload.get("constants"), policy)
    timely_targets = _timely_strengths(learned_order, obs, fallback_seconds=fallback_sec)
    for item, strength in zip(obs, timely_targets):
        item["timely_strength"] = strength
    timely_probabilities, _ = _posterior(learned_order, cdf, obs, policy, targets=timely_targets)
    independent_prior = baseline if not obs and len(worlds) == 1 and prior_sources is None else None
    # Both edges of the clearing interval follow the share of its failures the asking host catches.
    share = caught_share(caught or {"host": None, "F_all": 0., "C_all": 0., "F_host": 0., "C_host": 0.},
                         payload["constants"]["epsilon"], policy)
    band = _forecast_band(learned_order, probabilities, timely_probabilities, policy,
                          independent_prior=independent_prior, share=share)
    if prior_sources is not None and len(band) > policy["max_forecast_cells"]:
        # Do not let smaller-inventory prototype priors inflate a valid larger-inventory filter past budget.
        return aggregate(engine, payload, response, observed=obs, covered_priors=None, agents=agents,
                         caught=caught, candidates=candidates)
    lo, hi = learned_order.index(band[0]), learned_order.index(band[-1])
    band_keys = set(band)
    observed = {r["key"] for r in obs if r["weight"] > .01}
    by_key = {c["key"]: c for c in payload["cells"]}
    agents = agents or {}
    # The agent label is stored with its cell; None is the one unnamed agent. Only the reserved slots read it.
    rows = [{**{name: by_key[k][name] for name in ("key", "model", "family", "effort")},
             "agent": agents.get(k), "position": i, "classification": _classification(i, lo, hi, k in band_keys),
             "P_clear": probabilities[k], "P_timely": timely_probabilities[k],
             "placement": "learned" if k in observed else "prior-only",
             "threshold": "learned" if any(r["weight"] > .01 for r in obs) else "prior-only"}
            for i, k in enumerate(learned_order)]
    core = _core(band, policy)
    # A family is proven for this kind of task when one of its runs is among the candidate runs.
    proven = sorted({c["family"] for c in (candidates or {"candidates": []})["candidates"]})
    out = {"format": FORMAT, "baseline": deepcopy(baseline), "memory_filters": deepcopy(response["memory_filters"]),
           "order": learned_order, "lower": band[0], "upper": band[-1],
           "band": band, "caught_share": share, "core": core, "proven_families": proven,
           "reserved": _reserved(rows, core, share, set(proven), policy), "cells": rows, "worlds": worlds, "beliefs": belief,
           "evidence": obs, "policy": deepcopy(policy), "reason": response["reason"],
           "widening": [], "disagreements": [], "refilter_rounds": 0}
    if prior_sources is not None:
        out["prior_sources"] = prior_sources
    return out


def disagreements(capability, prepared):
    """Compare the two readers' prospective forecasts, before outcome evidence updates either."""
    semantic = prepared[0]["semantic"]
    experts = semantic["beliefs"]["experts"]
    panels = {k: {(c["family"], c["effort"]): c for c in panel} for k, panel in semantic["worlds"].items()}
    forecast = {cell_key(c): sum(e["weight"]*panels[e["key"]][c["family"], c["effort"]]["clean"] for e in experts)
                for c in prepared[0]["cells"]}
    margin = capability["policy"]["consistency_margin"]
    return [{"key": c["key"], "model": c["model"], "effort": c["effort"],
             "P_clear": c["P_clear"], "P_clean": forecast[c["key"]]}
            for c in capability["cells"] if c["key"] in forecast
            and abs(c["P_clear"]-forecast[c["key"]]) > margin]


def _recomputed_share(state, constants, host):
    """Recompute the caught shares and both edges from the recorded ``F`` and ``C``, which are taken as recorded."""
    from .evidence import TOLERANCE, close
    stored = state["caught_share"]
    keys(stored, ["host", "F_all", "C_all", "F_host", "C_host", "d_0", "d_all", "d_host", "e", "g"],
         where="capability caught_share")
    if stored["host"] != host:
        raise HowHardError("decision.capability: caught share does not belong to the host that asked")
    record = {k: number(stored[k], "capability caught_share." + k, nonnegative=True)
              for k in ("F_all", "C_all", "F_host", "C_host")}
    if any(record[part] > record[whole] + TOLERANCE for part, whole in
           (("C_all", "F_all"), ("C_host", "F_host"), ("F_host", "F_all"), ("C_host", "C_all"))):
        raise HowHardError("decision.capability: caught runs must be among the failed runs, and a host's among all hosts'")
    if not isinstance(constants, dict) or type(constants.get("epsilon")) not in (int, float) or not 0 < constants["epsilon"] < 1:
        raise HowHardError("decision.capability: caught share needs the risk class's accepted-defect limit")
    expected = caught_share({"host": host, **record}, constants["epsilon"], state["policy"])
    if any(not close(stored[k], expected[k]) for k in ("d_0", "d_all", "d_host", "e", "g")):
        raise HowHardError("decision.capability: caught shares and band edges do not follow from the recorded failed runs")
    return expected


def validate_receipt(state, inventory, semantic, chosen, *, decision_format=DECISION_FORMAT, constants=None, host=None):
    """Check complete filter coverage and the exact set of prospectively forecast cells."""
    optional_keys = ["prior_sources"] if isinstance(state, dict) and state.get("format") in (FORMAT, LEGACY_FORMAT_V5) else []
    allowed_formats = {
        DECISION_FORMAT: (FORMAT,),
        LEGACY_DECISION_V10: (LEGACY_FORMAT_V5, LEGACY_FORMAT_V4),
        LEGACY_DECISION_V9: (LEGACY_FORMAT_V4, LEGACY_FORMAT_V3),
        LEGACY_DECISION_V8: (LEGACY_FORMAT_V3,),
        LEGACY_DECISION_V7: (LEGACY_FORMAT_V2, LEGACY_FORMAT_V1),
    }
    # The format decides the block's shape, so a block of another decision format is refused by its format.
    if isinstance(state, dict) and "format" in state and state["format"] not in allowed_formats.get(decision_format, ()):
        raise HowHardError("decision.capability.format: unsupported version")
    # v6 adds the host's caught share, the two edges it sets, the core and the reserved cells.
    host_aware = isinstance(state, dict) and state.get("format") == FORMAT
    keys(state, ["format", "baseline", "memory_filters", "order", "lower", "upper", "band", "cells", "worlds", "beliefs",
                 "evidence", "policy", "reason", "widening", "disagreements", "refilter_rounds", "reader_usage", "widening_stop", "executed",
                 *(["caught_share", "core", "proven_families", "reserved"] if host_aware else [])],
         optional_keys, where="decision.capability")
    has_timely = state["format"] in (FORMAT, LEGACY_FORMAT_V5, LEGACY_FORMAT_V4, LEGACY_FORMAT_V3)
    has_split_band = state["format"] in (FORMAT, LEGACY_FORMAT_V5, LEGACY_FORMAT_V4)
    has_prototype_priors = state["format"] in (FORMAT, LEGACY_FORMAT_V5)
    has_agents = state["format"] == FORMAT
    validate_policy(state["policy"], host_aware=host_aware)
    cells = [{**c, "key": cell_key(c)} for c in inventory]
    known = {c["key"]: c for c in cells}
    validate_filter(state["baseline"], cells)
    # Validate the learned order and outer band endpoints with the same effort constraints.
    validate_filter({k: state[k] for k in ("order", "lower", "upper")}, cells)
    lo, hi = state["order"].index(state["lower"]), state["order"].index(state["upper"])
    band = state["band"]
    if not isinstance(state["cells"], list) or len(state["cells"]) != len(cells):
        raise HowHardError("decision.capability: inconsistent cell coverage")
    if (not isinstance(band, list) or not band or band[0] != state["lower"] or band[-1] != state["upper"]
            or any(not isinstance(key, str) for key in band)
            or (not has_split_band and band != state["order"][lo:hi+1])):
        raise HowHardError("decision.capability: inconsistent band")
    banded, forecasted = set(band), set(band)
    cell_fields = ["key", "model", "family", "effort", "position", "classification", "P_clear", "placement", "threshold"]
    if has_timely:
        cell_fields.insert(7, "P_timely")
    if has_agents:
        cell_fields.append("agent")
    previous = -1.
    previous_timely = -1.
    for i, c in enumerate(state["cells"]):
        keys(c, cell_fields, where="capability cell")
        if c["key"] != state["order"][i] or c["key"] != cell_key(c) or c["position"] != i or type(c["position"]) is not int:
            raise HowHardError("decision.capability: inconsistent identity or position")
        if (c["model"], c["effort"]) != (known[c["key"]]["model"], known[c["key"]]["effort"]):
            raise HowHardError("decision.capability: inconsistent inventory identity")
        if has_agents and c["agent"] != known[c["key"]].get("agent"):
            raise HowHardError("decision.capability: cell agent does not match native candidate inventory")
        expected = _classification(i, lo, hi, c["key"] in banded)
        if c["classification"] != expected or c["placement"] not in ("learned", "prior-only") or c["threshold"] not in ("learned", "prior-only"):
            raise HowHardError("decision.capability: inconsistent classification or evidence status")
        value = number(c["P_clear"], "capability.P_clear", nonnegative=True)
        if value > 1 or value < previous - 1e-10:
            raise HowHardError("decision.capability: probabilities must be monotone along this task's order")
        previous = value
        if has_timely:
            timely = number(c["P_timely"], "capability.P_timely", nonnegative=True)
            if timely > 1 or timely < previous_timely - 1e-10 or timely > value + 1e-10:
                raise HowHardError("decision.capability: timely probabilities must be monotone and at most P_clear")
            previous_timely = timely
    filters = {"filter:llm": state["baseline"]}
    memories = {(m["id"], m["version"]) for m in semantic["memories"]}
    if not isinstance(state["memory_filters"], list) or len(state["memory_filters"]) > SETTINGS["memory_limit"]:
        raise HowHardError("decision.capability: invalid memory filters")
    for item in state["memory_filters"]:
        keys(item, ["id", "version", "filter"], where="capability memory")
        key = f"filter:memory:{item['id']}:{item['version']}"
        if (item["id"], item["version"]) not in memories or key in filters:
            raise HowHardError("decision.capability: unknown or duplicate memory")
        validate_filter(item["filter"], cells)
        filters[key] = item["filter"]
    if not isinstance(state["worlds"], dict) or set(state["worlds"]) != set(filters):
        raise HowHardError("decision.capability: inconsistent expert worlds")

    def matches_priors(**options):
        for key, value in filters.items():
            expected_cdf = prior_cdf(value, state["policy"], **options)
            if key == "filter:llm" and "prior_sources" in state and has_prototype_priors:
                ps = state["prior_sources"]
                if not isinstance(ps, dict) or not ps or any(k not in known for k in ps):
                    return False
                for pv in ps.values():
                    number(pv, "capability.prior_sources", nonnegative=True)
                    if pv > 1:
                        return False
                expected_cdf = _pav_along(value["order"], {**expected_cdf, **ps})
            if state["worlds"][key] != expected_cdf:
                return False
        return True

    if not (matches_priors(linear_prefix=not has_split_band)
            or (state["format"] == LEGACY_FORMAT_V1 and matches_priors(linear_prefix=True, linear_tail=True))):
        raise HowHardError("decision.capability: prior probabilities do not match the filter")
    weights, seen = 0., set()
    for e in state["beliefs"]["experts"]:
        keys(e, ["key", "weight", "log_score", "evidence_count"], where="capability expert")
        if e["key"] not in filters or e["key"] in seen or type(e["evidence_count"]) is not int or e["evidence_count"] < 0:
            raise HowHardError("decision.capability: invalid predictive expert")
        seen.add(e["key"])
        weights += number(e["weight"], "capability expert weight", nonnegative=True)
        number(e["log_score"], "capability expert log score")
    if seen != set(filters) or abs(weights-1) > 1e-6:
        raise HowHardError("decision.capability: expert weights must cover priors and sum to one")
    # v6 observations are the request's scored candidates; older formats keep their packaged bound.
    has_candidates = state["format"] == FORMAT
    max_evidence = (
        len(semantic["evidence_candidates"]["candidates"])
        if has_candidates
        else max(SETTINGS["evidence_limit"], len(cells) * SETTINGS["evidence_limit"])
        if has_prototype_priors
        else SETTINGS["evidence_limit"]
    )
    if not isinstance(state["evidence"], list) or len(state["evidence"]) > max_evidence:
        raise HowHardError("decision.capability: invalid precedent count")
    evidence_fields = ["id", "key", "outcome", "cause", "score", "similarity", "weight"]
    if has_timely:
        evidence_fields = ["id", "key", "outcome", "cause", "active_seconds", "forecast_active_seconds", "timely_strength", "score", "similarity", "weight"]
    if has_candidates:
        evidence_fields = ["id", "key", "outcome", "cause", "active_seconds", "forecast_active_seconds", "timely_strength", "strength", "ticket_share", "weight"]
    for item in state["evidence"]:
        keys(item, evidence_fields, where="capability evidence")
        text(item["id"], "capability evidence id")
        if item["key"] not in known or item["outcome"] not in ("clean", "loud", "silent"):
            raise HowHardError("decision.capability: invalid observed cell/outcome")
        for field in ("strength", "ticket_share", "weight") if has_candidates else ("score", "similarity", "weight"):
            number(item[field], "capability evidence "+field, nonnegative=field != "score")
        if has_timely:
            for field in ("active_seconds", "forecast_active_seconds"):
                if item[field] is not None:
                    number(item[field], "capability evidence "+field, nonnegative=True)
            strength = number(item["timely_strength"], "capability evidence timely_strength", nonnegative=True)
            if strength > 1:
                raise HowHardError("decision.capability: timely_strength must be in [0, 1]")
    if has_candidates:
        # Recompute each observation's ticket share and weight from its scored candidate.
        from . import evidence as scoring
        scored = {c["id"]: c for c in semantic["evidence_candidates"]["candidates"]}
        cells_by_key = {c["key"]: c for c in state["cells"]}
        counted, seen_ids = [], set()
        for item in state["evidence"]:
            candidate, cell = scored.get(item["id"]), cells_by_key[item["key"]]
            if candidate is None or item["id"] in seen_ids:
                raise HowHardError("decision.capability: observation is not one scored candidate")
            seen_ids.add(item["id"])
            if ((candidate["family"], candidate["effort"], candidate["outcome"], candidate["active_seconds"])
                    != (cell["family"], cell["effort"], item["outcome"], item["active_seconds"])):
                raise HowHardError("decision.capability: observation does not match its scored candidate")
            strengths = [_strength({"outcome": item["outcome"], "cause": item["cause"]}, state["policy"])]
            if item["outcome"] != "clean" and item["cause"] != "capability":
                strengths.append(0.)  # A capability_caused=false note; the note itself is not stored.
            if not any(scoring.close(item["strength"], value) for value in strengths):
                raise HowHardError("decision.capability: strength does not match the outcome and cause")
            counted.append((item, scoring.weight(candidate, item["strength"])))
        shares = scoring.ticket_shares([(scored[item["id"]]["ticket"], value) for item, value in counted])
        for (item, value), share in zip(counted, shares):
            if not scoring.close(item["ticket_share"], share) or not scoring.close(item["weight"], value * share):
                raise HowHardError("decision.capability: evidence weight does not match its scored candidate")
    if has_timely:
        fallback_sec = _unanchored_reference_seconds(constants, state["policy"])
        expected_strengths = _timely_strengths(state["order"], state["evidence"], fallback_seconds=fallback_sec)
        if any(abs(item["timely_strength"] - exp) > 1e-12 for item, exp in zip(state["evidence"], expected_strengths)):
            raise HowHardError("decision.capability: inconsistent timely_strength")
    if has_split_band:
        independent_prior = (
            state["baseline"]
            if not state["evidence"] and len(filters) == 1 and "prior_sources" not in state
            else None
        )
        clear = {c["key"]: c["P_clear"] for c in state["cells"]}
        timely = {c["key"]: c["P_timely"] for c in state["cells"]}
        share = _recomputed_share(state, constants, host) if host_aware else None
        if band != _forecast_band(state["order"], clear, timely, state["policy"], independent_prior=independent_prior,
                                  share=share):
            raise HowHardError("decision.capability: band does not match the frozen probabilities")
    if host_aware:
        # The core, the reserved cells and the cells that get the first forecasts all follow from the band.
        from .evidence import close
        policy = state["policy"]
        if len(band) > policy["max_forecast_cells"]:
            raise HowHardError("decision.capability: forecast budget exceeded")
        if state["core"] != _core(band, policy):
            raise HowHardError("decision.capability: core does not match the band")
        proven = sorted({c["family"] for c in semantic["evidence_candidates"]["candidates"]})
        if state["proven_families"] != proven:
            raise HowHardError("decision.capability: proven families do not match the scored candidates")
        reserved = _reserved(state["cells"], state["core"], share, set(proven), policy)
        if not isinstance(state["reserved"], list) or len(state["reserved"]) != len(reserved):
            raise HowHardError("decision.capability: reserved cells do not follow from the band and the candidate runs")
        for stored, wanted in zip(state["reserved"], reserved):
            keys(stored, ["key", "reason", "bar"], where="capability reserved cell")
            if (stored["key"], stored["reason"]) != (wanted["key"], wanted["reason"]) or not close(stored["bar"], wanted["bar"]):
                raise HowHardError("decision.capability: reserved cells do not follow from the band and the candidate runs")
        forecasted = set(forecast_cells(state))
    if not isinstance(state["widening"], list) or len(state["widening"]) > state["policy"]["widening_rounds"]:
        raise HowHardError("decision.capability: invalid widening rounds")
    stronger = state["order"][hi+1:]
    tiered = decision_format == DECISION_FORMAT
    trigger = WIDENING_TRIGGER if tiered else LEGACY_WIDENING_TRIGGER
    for i, item in enumerate(state["widening"]):
        keys(item, ["round", "cells", "trigger"], where="capability widening")
        # Older formats widen over the overpowered cells alone, in order.
        following = widening_order(state, forecasted)[:1] if host_aware else stronger[i:i+1]
        if item != {"round": i+1, "cells": following, "trigger": trigger} or not item["cells"]:
            raise HowHardError("decision.capability: widening must forecast the next stronger overpowered cell"
                               if not host_aware else
                               "decision.capability: widening must forecast the next stronger cell with no forecast")
        forecasted.update(item["cells"])
    if tiered and state["widening_stop"] not in WIDENING_STOPS:
        raise HowHardError("decision.capability.widening_stop: expected one of " + ", ".join(WIDENING_STOPS))
    if len(forecasted) > state["policy"]["max_forecast_cells"]:
        raise HowHardError("decision.capability: forecast budget exceeded")
    if type(state["refilter_rounds"]) is not int or state["refilter_rounds"] not in (0, 1):
        raise HowHardError("decision.capability: invalid re-filter count")
    baseline = semantic["worlds"]["llm"]
    actual = {(c["family"], c["effort"]) for c in baseline}
    expected = {(c["family"], c["effort"]) for c in state["cells"] if c["key"] in forecasted}
    if actual != expected:
        raise HowHardError("decision.capability: forecasts must cover exactly the band and forecasted widening cells"
                           if not host_aware else
                           "decision.capability: forecasts must cover exactly the core, the reserved cells, "
                           "the band cells in unfilled slots and the widening cells")
    chosen_cell = next(c for c in state["cells"] if c["key"] == cell_key(chosen))
    if chosen_cell["key"] not in forecasted or state["executed"] != {k: chosen_cell[k] for k in ("key", "classification", "position", "P_clear")}:
        raise HowHardError("decision.capability: executed classification/position must match a forecasted cell")
    # Check the cross-reader probabilities against the frozen forecast ensemble.
    experts = semantic["beliefs"]["experts"]
    panels = {k: {(c["family"], c["effort"]): c for c in panel} for k, panel in semantic["worlds"].items()}
    rows = []
    for c in state["cells"]:
        if c["key"] in forecasted:
            p = sum(e["weight"]*panels[e["key"]][c["family"], c["effort"]]["clean"] for e in experts)
            if abs(c["P_clear"]-p) > state["policy"]["consistency_margin"]:
                rows.append({"key": c["key"], "model": c["model"], "effort": c["effort"], "P_clear": c["P_clear"], "P_clean": p})
    if state["disagreements"] != rows:
        raise HowHardError("decision.capability: inconsistent cross-reader disagreements")
    return state
