"""Task-conditioned ordinal threshold reader over the shared execution evidence."""

from copy import deepcopy
import math
import re

import numpy as np

from .context import digest
from .llm import LLMError, SETTINGS
from .validation import HowHardError, keys, number, text

FORMAT = "howhard-capability-v1"
EFFORT_ORDER = {e: i for i, e in enumerate(("low", "medium", "high", "xhigh", "max", "ultra"))}
DEFAULTS = {"max_forecast_cells": 12, "widening_rounds": 2,
            "min_clear_probability": .5, "reliability_probability": .9,
            "overpowered_gain_margin": .05, "consistency_margin": .2,
            "observation_noise": .05, "order_penalty": .15,
            "loud_weight": .75, "silent_weight": 1., "unknown_cause_weight": .5}


def cell_key(cell):
    return "cell:" + digest([cell["model"], cell["effort"]])[:24]


def validate_policy(policy):
    keys(policy, list(DEFAULTS), where="capability")
    for k, v in policy.items():
        if k in ("max_forecast_cells", "widening_rounds"):
            if type(v) is not int or v < (1 if k == "max_forecast_cells" else 0):
                raise HowHardError(f"capability.{k}: expected {'positive' if k == 'max_forecast_cells' else 'nonnegative'} integer")
        else:
            number(v, f"capability.{k}", nonnegative=True)
            if k != "order_penalty" and v > 1:
                raise HowHardError(f"capability.{k}: expected probability/weight in [0, 1]")
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
        "memory_filters": "Array of up to eight {id,version,filter}: full conditional order/cut-point adjustments for relevant ACTIVE retrieved memories, including read-only execution experts. Cite exact id/version. No invented outcomes. Omitted experts have no adjustment.",
        "reason": "Brief explanation of task-specific placement and cut points.",
        "request_more": payload["contract"]["request_more"]}
    return result


def baseline_request(payload):
    result = {k: deepcopy(payload[k]) for k in ("kind", "prompt_version", "task_type", "context", "risk_class")}
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


def prior_cdf(value, policy):
    """Proper smoothed threshold prior, including mass beyond the strongest cell."""
    n = len(value["order"])
    lo, hi = value["order"].index(value["lower"]), value["order"].index(value["upper"])
    minimum, reliable = policy["min_clear_probability"], policy["reliability_probability"]
    before = np.linspace(minimum * .2, minimum * .8, lo).tolist()
    fit = np.linspace(max(minimum, reliable - .2), reliable, hi-lo+1).tolist() if hi > lo else [reliable]
    # Above the upper cut, the maximum possible gain stays below the policy margin.
    # Concentrate the tail geometrically so conditioning cannot stretch it across large inventories.
    tail = min((1-reliable)*.5, policy["overpowered_gain_margin"]*.5)
    after_count = n - 1 - hi
    if after_count:
        steps = np.arange(1, after_count + 1, dtype=float)
        after = (reliable + tail * (1.0 - 0.5 ** steps) / (1.0 - 0.5 ** after_count)).tolist()
    else:
        after = []
    return dict(zip(value["order"], before+fit+after))


def observations(engine, payload):
    """Retrieve bounded actual observations with absolute, non-normalized relevance."""
    from .retrieval import records, task_text
    evidence = payload["evidence"][:SETTINGS["evidence_limit"]]
    if not evidence:
        return []
    scores = {r["key"]: r["score"] for item in payload["retrieval"]
              if item["request"]["tool"] == "experience.search" for r in item["result"].get("selected", [])}
    missing = [r for r in evidence if scores.get("evidence:" + r["id"]) is None]
    if missing:
        # Small retrieval corpora skip reranking for display; the learned reader needs
        # absolute relevance even for one precedent. Reuse the configured reranker.
        docs = records(rows=missing)
        values = engine.retrieval.reranker.score(task_text(payload["task_type"], payload["context"]), [r["text"] for r in docs])
        values = list(values)
        if len(values) != len(missing) or any(not math.isfinite(float(v)) for v in values):
            raise HowHardError("capability similarity: invalid reranker scores")
        scores.update({"evidence:"+r["id"]: float(v) for r, v in zip(missing, values)})
    known = {cell_key(c) for c in payload["cells"]}
    policy = payload["filter_policy"]
    result = []
    recency_order = {r["id"]: age for age, r in enumerate(reversed(engine.rows))}
    for row in evidence:
        age = recency_order.get(row["id"], 0)
        key = cell_key(row)
        if key not in known:
            continue  # Unknown executed controls are never mapped to guessed cells.
        score = float(scores["evidence:"+row["id"]])
        similarity = 1 / (1 + math.exp(-max(-60, min(60, score))))
        strength = 1.
        if row["outcome"] != "clean":
            strength = policy[row["outcome"]+"_weight"]
            cause = row.get("cause")
            if cause != "capability":
                strength *= policy["unknown_cause_weight"]
                if re.search(r'\bcapability_caused\s*[:=]\s*false\b', row.get("note", ""), re.I):
                    strength = 0.
        result.append({"id": row["id"], "key": key, "outcome": row["outcome"], "cause": row.get("cause"),
                       "score": score, "similarity": similarity,
                       "weight": row["weight"] * similarity * strength * SETTINGS["belief"]["recency"] ** age})
    return result


def _posterior(order, cdf, obs, policy):
    probabilities = np.asarray([cdf[k] for k in order])
    probabilities = np.maximum.accumulate(probabilities)
    masses = np.diff(np.r_[0., probabilities, 1.])
    logs = np.log(np.maximum(masses, 1e-12))
    positions = {k: i for i, k in enumerate(order)}
    noise = policy["observation_noise"]
    for row in obs:
        likelihood = np.where(np.arange(len(order)+1) <= positions[row["key"]], 1-noise, noise)
        if row["outcome"] != "clean":
            likelihood = 1-likelihood
        logs += row["weight"] * np.log(likelihood)
    peak = float(logs.max())
    normalizer = peak + math.log(float(np.exp(logs-peak).sum()))
    posterior = np.exp(logs-normalizer)
    return dict(zip(order, np.cumsum(posterior)[:-1].tolist())), normalizer


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


def aggregate(engine, payload, response, *, observed=None):
    from .semantic import normalize_support
    policy = payload["filter_policy"]
    baseline = response["filter"]
    worlds = {"filter:llm": prior_cdf(baseline, policy)}
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
    obs = observations(engine, payload) if observed is None else deepcopy(observed)
    learned_order = _learn_order(order, cdf, obs, payload["cells"], policy) if obs else order
    cdf = dict(zip(learned_order, [cdf[k] for k in order]))
    probabilities, _ = _posterior(learned_order, cdf, obs, policy)
    n = len(learned_order)
    lo = next((i for i, k in enumerate(learned_order) if probabilities[k] >= policy["min_clear_probability"]), n)
    if lo == n:
        raise HowHardError("capability filter: no cell reaches the required capability probability; supply a more capable inventory")
    # P_clear is monotone along the order, so the strongest cell attains the best probability.
    best = probabilities[learned_order[-1]]
    reliable = next((i for i in range(lo, n) if probabilities[learned_order[i]] >= policy["reliability_probability"]), None)
    # The band reaches the first reliable cell, then stops once the remaining gain is below
    # the margin. With no reliable cell the target is unattainable in this inventory and
    # cannot anchor the band; the gain cut against the best attained probability then
    # applies from the lower cut instead of defaulting to the strongest cell.
    anchor = lo if reliable is None else reliable
    hi = next((i for i in range(anchor, n) if best-probabilities[learned_order[i]] < policy["overpowered_gain_margin"]), n-1)
    # The original LLM cuts are exact when no evidence or memory adjustment exists.
    if not obs and len(worlds) == 1:
        lo, hi = learned_order.index(baseline["lower"]), learned_order.index(baseline["upper"])
    observed = {r["key"] for r in obs if r["weight"] > .01}
    by_key = {c["key"]: c for c in payload["cells"]}
    rows = [{**{name: by_key[k][name] for name in ("key", "model", "family", "effort")},
             "position": i, "classification": "underpowered" if i < lo else "overpowered" if i > hi else "fit",
             "P_clear": probabilities[k], "placement": "learned" if k in observed else "prior-only",
             "threshold": "learned" if any(r["weight"] > .01 for r in obs) else "prior-only"}
            for i, k in enumerate(learned_order)]
    return {"format": FORMAT, "baseline": deepcopy(baseline), "memory_filters": deepcopy(response["memory_filters"]),
            "order": learned_order, "lower": learned_order[lo], "upper": learned_order[hi],
            "band": learned_order[lo:hi+1], "cells": rows, "worlds": worlds, "beliefs": belief,
            "evidence": obs, "policy": deepcopy(policy), "reason": response["reason"],
            "widening": [], "disagreements": [], "refilter_rounds": 0}


def disagreements(capability, prepared):
    indexed = {cell_key(c): c for c in prepared[0]["cells"]}
    margin = capability["policy"]["consistency_margin"]
    return [{"key": c["key"], "model": c["model"], "effort": c["effort"],
             "P_clear": c["P_clear"], "P_clean": indexed[c["key"]]["P_clean"]["mean"]}
            for c in capability["cells"] if c["key"] in indexed
            and abs(c["P_clear"]-indexed[c["key"]]["P_clean"]["mean"]) > margin]


def validate_receipt(state, inventory, semantic, chosen):
    """Check complete filter coverage and the exact set of prospectively forecast cells."""
    keys(state, ["format", "baseline", "memory_filters", "order", "lower", "upper", "band", "cells", "worlds", "beliefs",
                 "evidence", "policy", "reason", "widening", "disagreements", "refilter_rounds", "reader_usage", "widening_stop", "executed"], where="decision.capability")
    if state["format"] != FORMAT:
        raise HowHardError("decision.capability.format: unsupported version")
    validate_policy(state["policy"])
    cells = [{**c, "key": cell_key(c)} for c in inventory]
    known = {c["key"]: c for c in cells}
    validate_filter(state["baseline"], cells)
    # Validate the learned order and nonempty band with the same effort constraints.
    validate_filter({k: state[k] for k in ("order", "lower", "upper")}, cells)
    lo, hi = state["order"].index(state["lower"]), state["order"].index(state["upper"])
    if state["band"] != state["order"][lo:hi+1] or not isinstance(state["cells"], list) or len(state["cells"]) != len(cells):
        raise HowHardError("decision.capability: inconsistent band or cell coverage")
    previous = -1.
    for i, c in enumerate(state["cells"]):
        keys(c, ["key", "model", "family", "effort", "position", "classification", "P_clear", "placement", "threshold"], where="capability cell")
        if c["key"] != state["order"][i] or c["key"] != cell_key(c) or c["position"] != i or type(c["position"]) is not int:
            raise HowHardError("decision.capability: inconsistent identity or position")
        if (c["model"], c["effort"]) != (known[c["key"]]["model"], known[c["key"]]["effort"]):
            raise HowHardError("decision.capability: inconsistent inventory identity")
        expected = "underpowered" if i < lo else "overpowered" if i > hi else "fit"
        if c["classification"] != expected or c["placement"] not in ("learned", "prior-only") or c["threshold"] not in ("learned", "prior-only"):
            raise HowHardError("decision.capability: inconsistent classification or evidence status")
        value = number(c["P_clear"], "capability.P_clear", nonnegative=True)
        if value > 1 or value < previous - 1e-10:
            raise HowHardError("decision.capability: probabilities must be monotone along this task's order")
        previous = value
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
    for key, value in filters.items():
        if state["worlds"][key] != prior_cdf(value, state["policy"]):
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
    if not isinstance(state["evidence"], list) or len(state["evidence"]) > SETTINGS["evidence_limit"]:
        raise HowHardError("decision.capability: invalid precedent count")
    for item in state["evidence"]:
        keys(item, ["id", "key", "outcome", "cause", "score", "similarity", "weight"], where="capability evidence")
        text(item["id"], "capability evidence id")
        if item["key"] not in known or item["outcome"] not in ("clean", "loud", "silent"):
            raise HowHardError("decision.capability: invalid observed cell/outcome")
        for field in ("score", "similarity", "weight"):
            number(item[field], "capability evidence "+field, nonnegative=field != "score")
    forecasted = set(state["band"])
    if not isinstance(state["widening"], list) or len(state["widening"]) > state["policy"]["widening_rounds"]:
        raise HowHardError("decision.capability: invalid widening rounds")
    stronger = state["order"][hi+1:]
    for i, item in enumerate(state["widening"]):
        keys(item, ["round", "cells", "trigger"], where="capability widening")
        if item != {"round": i+1, "cells": stronger[i:i+1], "trigger": "silent-cap-unmet"} or not item["cells"]:
            raise HowHardError("decision.capability: widening must forecast the next stronger overpowered cell")
        forecasted.update(item["cells"])
    if len(forecasted) > state["policy"]["max_forecast_cells"]:
        raise HowHardError("decision.capability: forecast budget exceeded")
    if type(state["refilter_rounds"]) is not int or state["refilter_rounds"] not in (0, 1):
        raise HowHardError("decision.capability: invalid re-filter count")
    baseline = semantic["worlds"]["llm"]
    actual = {(c["family"], c["effort"]) for c in baseline}
    expected = {(c["family"], c["effort"]) for c in state["cells"] if c["key"] in forecasted}
    if actual != expected:
        raise HowHardError("decision.capability: forecasts must cover exactly the band and forecasted widening cells")
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
