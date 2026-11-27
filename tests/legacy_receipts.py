"""Derive authentic frozen legacy receipts from a current receipt's base structure.

``to_v10`` is the base of every older builder: it removes what v11 added (the host's name,
agent labels, ranges on active time and token kinds, the scored evidence candidates, admission
tiers, the outcome beliefs on the log-odds scale, the amount beliefs and credit spread, and the
host-aware band with its core and reserved cells) and restores the v10 format names, evidence
rows, policies, Dirichlet outcome posteriors, meaning of the silent cap, per-expert active-time
calibration, forecast pricing and credit check. A v10 receipt forecasts exactly its band and its
widening cells, so the current receipt it is derived from must have no reserved cell.
"""

from copy import deepcopy
import math

import numpy as np

from howhard.accounting import CATEGORIES, price
from howhard.capability import prior_cdf
from howhard.semantic import OUTCOMES, panel_lookup
from howhard.success import (
    MAX_DURATION_LOG_SD,
    POSTERIOR_FIELDS,
    accepted_risk,
    calibrate_duration,
    cost,
    denominator,
    envelope,
    outcome_posterior,
    posterior_intervals,
)

def without_reserved_slots(directory):
    """Configure the store in ``directory`` to forecast exactly its band, as older receipts did.

    Call it before creating the engine that mints the receipts an older one is derived from, or
    whose fixture is about the band's own cells.
    """
    (directory / "howhard.toml").write_text(
        '[capability]\nreserved_forecast_cells = 0\n[task_types]\ntask = "standard"\n')


LEGACY_SAMPLING = {"min_concentration": 8., "max_concentration": 1000000., "prior_strength": 8.,
                   "explore_within": .1, "uncertainty_multiplier": 1.}
# The three Dirichlet constants a v9 or v10 policy stored, at their packaged values.
DIRICHLET_SAMPLING = {"min_concentration": 2., "max_concentration": 1000000., "prior_strength": 8.}


def legacy_credit(row, candidate):
    """v7 to v10 priced one expert's forecast row directly; ``None`` for an unpriced cell."""
    card = candidate.get("rate_card")
    if card is not None:
        return price(row["token_usage"], card) if all(card["rates"][k] is not None for k in CATEGORIES) else None
    rate = candidate.get("usd_per_mtok")
    return row["tokens"] / 1e6 * rate if rate is not None else None


def legacy_calibration(constants, policy, state, expert, candidate, money_included, pair, *, legacy_v9=False):
    """A v8 to v10 sampling cell: the sampled expert's own calibrated active time and priced forecast.

    Its outcome belief is the sampled expert's Dirichlet posterior: a bounded forecast prior plus
    only this cell's own outcomes, with the retry cost's spread under the Dirichlet covariance.
    """
    def run_cost(row):
        return legacy_credit(row, candidate) * candidate["cost_per_credit"] if money_included else 0.

    experts = state["beliefs"]["experts"]
    panels = {e["key"]: panel_lookup(state["worlds"][e["key"]]) for e in experts}
    row = panels[expert["key"]][pair]
    posterior = outcome_posterior(row, pair, state["outcome_observations"], policy, legacy_v9=legacy_v9)
    probabilities = np.asarray(posterior["mean"])
    concentration = posterior["concentration"]
    durations = {key: calibrate_duration(panel[pair]["seconds"], pair, state["duration_observations"])
                 for key, panel in panels.items()}
    dur = dict(durations[expert["key"]])
    mean_mu = sum(e["weight"] * durations[e["key"]]["mu"] for e in experts)
    between = sum(e["weight"] * (durations[e["key"]]["mu"] - mean_mu) ** 2 for e in experts)
    dur["between_experts_log_variance"] = between
    hours = dur["expected_seconds"] / 3600
    expected = cost(run_cost(row), hours, probabilities[1], probabilities[2], constants)
    d = denominator(probabilities[1])
    gradient = np.array([0., (constants["C_loud"] + expected) / d, constants["C_silent"] / d])
    covariance = (np.diag(probabilities) - np.outer(probabilities, probabilities)) / (concentration + 1)
    sigma_sq = min(MAX_DURATION_LOG_SD ** 2, max(dur["sigma"] ** 2, between))
    variance = max(0., float(gradient @ covariance @ gradient)) + (
        (constants["lambda_per_hour"] / d) ** 2) * (math.exp(sigma_sq) - 1.0) * hours ** 2
    expert_costs = []
    for e in experts:
        mean = outcome_posterior(panels[e["key"]][pair], pair, state["outcome_observations"], policy,
                                 legacy_v9=legacy_v9)["mean"]
        expert_costs.append((e["weight"], cost(run_cost(panels[e["key"]][pair]),
                                               durations[e["key"]]["expected_seconds"] / 3600, mean[1], mean[2], constants)))
    mean_cost = sum(w * j for w, j in expert_costs)
    variance = max(variance, sum(w * (j - mean_cost) ** 2 for w, j in expert_costs))
    return {"family": pair[0], "effort": pair[1], "expert": expert["key"],
            "outcome_posterior": {k: posterior[k] for k in POSTERIOR_FIELDS},
            "probabilities": probabilities.tolist(), "concentration": concentration,
            "duration": dur, "cost_sd": math.sqrt(variance)}


def legacy_sampling_cells(receipt, *, legacy_v9=False):
    """Rebuild every sampled cell of a receipt with the v8 to v10 calibration."""
    sem, selection = receipt["semantic"], receipt["selection"]
    candidates = {(c["model"], c["effort"]): c for c in receipt["native_candidates"]}
    models = {(c["family"], c["effort"]): c["model"] for c in selection["cells"]}
    experts = {e["key"]: e for e in sem["beliefs"]["experts"]}
    return [legacy_calibration(selection["constants"], selection["sampling_policy"], sem, experts[cell["expert"]],
                               candidates[models[cell["family"], cell["effort"]], cell["effort"]],
                               receipt["cost_policy"]["worker_money_included"], (cell["family"], cell["effort"]),
                               legacy_v9=legacy_v9)
            for cell in sem["sampling"]["cells"]]


def restore_dirichlet(receipt, *, legacy_v9=False):
    """Rebuild every outcome-dependent number of a selection cell as a v9 or v10 decision stored it."""
    sem, selection = receipt["semantic"], receipt["selection"]
    panels = {k: panel_lookup(p) for k, p in sem["worlds"].items()}
    weights = {e["key"]: e["weight"] for e in sem["beliefs"]["experts"]}
    constants = selection["constants"]
    for row in selection["cells"]:
        pair = row["family"], row["effort"]
        intervals, _ = posterior_intervals(
            panels, weights, pair, sem["outcome_observations"], selection["sampling_policy"], legacy_v9=legacy_v9
        )
        for field in ("P_clean", "P_loud", "P_silent", "P_silent_accepted"):
            row[field] = intervals[field]
        row["admission_risk"] = intervals["admission_risk"]
        row["admissible"] = row["admission_risk"] <= constants["epsilon"]
        loud, silent = row["P_loud"]["mean"], row["P_silent"]["mean"]
        run = row["session_cost_usd"] if receipt["cost_policy"]["worker_money_included"] else 0.0
        row["expected_attempts"] = 1.0 / denominator(loud)
        row["J"] = cost(run, row["T"], loud, silent, constants)
    if sem["sampling"]["enabled"]:
        sem["sampling"]["cells"] = legacy_sampling_cells(receipt, legacy_v9=legacy_v9)
        sem["sampling"]["reason"] = "per-cell-dirichlet-posterior"


def legacy_capability_prior(flt, policy, *, linear_tail=False):
    order = flt["order"]
    lo, hi = order.index(flt["lower"]), order.index(flt["upper"])
    minimum, reliable = policy["min_clear_probability"], policy["reliability_probability"]
    before = np.linspace(minimum * .2, minimum * .8, lo).tolist()
    fit = np.linspace(max(minimum, reliable - .2), reliable, hi - lo + 1).tolist() if hi > lo else [reliable]
    tail = min((1 - reliable) * .5, policy["overpowered_gain_margin"] * .5)
    after_count = len(order) - 1 - hi
    if linear_tail:
        after = np.linspace(reliable, reliable + tail, after_count + 1).tolist()[1:]
    elif after_count:
        steps = np.arange(1, after_count + 1, dtype=float)
        after = (reliable + tail * (1.0 - 0.5 ** steps) / (1.0 - 0.5 ** after_count)).tolist()
    else:
        after = []
    return dict(zip(order, before + fit + after))


def restore_cap(receipt):
    """The cap before admission tiers: met when some admissible cell has enough credit, else no draw."""
    sem, selection = receipt["semantic"], receipt["selection"]
    cap_met = any(c["admissible"] and c.get("credit_ok", True) for c in selection["cells"])
    assert sem["sampling"]["enabled"] or not cap_met, "a fixture cannot restore a draw that was never made"
    selection["cap_met"] = cap_met
    selection["needs_review"] = not cap_met
    if not cap_met:
        sem["sampling"] = {
            "enabled": False,
            "reason": "cap-unmet",
            "policy": deepcopy(selection["sampling_policy"]),
            "cells": [],
        }
        sem.pop("sampled_expert", None)
        selection["exploration"]["enabled"] = False
        selection["exploration"]["reason"] = "disabled-cap-unmet"


def to_v10(receipt, on_cap_unmet="flag-review"):
    """A v10 receipt: no host name, agent labels, ranges on active time or token kinds, tiers or beliefs."""
    receipt = deepcopy(receipt)
    if receipt["format"] != "howhard-decision-v11":
        return receipt
    # A v10 outcome observation counted for its own forecast cell only, at its weight after the
    # limit. v11 computes those rows on demand, so read them before the labels are removed.
    from howhard.decisions import counted_observations
    own = {(row["family"], row["effort"], item["id"]): item["weight"] for row in receipt["selection"]["cells"]
           for item in counted_observations(receipt, row) if item["relation"] == "cell"}
    receipt["format"] = "howhard-decision-v10"
    receipt.pop("host")
    for candidate in receipt["native_candidates"]:
        candidate.pop("agent", None)
    cap = receipt["capability"]
    cap["format"] = "howhard-capability-v5"
    for cell in cap["cells"]:
        cell.pop("agent")
    for item in cap["widening"]:
        item["trigger"] = "silent-cap-unmet"
    if cap["widening_stop"] == "pickable":
        cap["widening_stop"] = "cap-met"
    assert not cap["reserved"], "derive older receipts from a decision that reserved no forecast cell"
    for field in ("caught_share", "core", "proven_families", "reserved"):
        del cap[field]
    for setting in ("caught_share_worth", "reserved_forecast_cells", "unproven_clear_probability"):
        del cap["policy"][setting]
    sem = receipt["semantic"]
    sem["format"] = "howhard-semantic-v7"
    sem["prompt_version"] = "semantic-reader-v7"
    for panel in sem["worlds"].values():
        for row in panel:
            row.pop("seconds_ci", None)
            row.pop("token_usage_ci", None)
    # v10 kept no record of which memory-free rows came from stored forecasts.
    sem.pop("stored_forecasts")
    # v10 rows carried their own score and similarity, and outcomes a count-based age.
    scored = {c["id"]: c for c in sem.pop("evidence_candidates")["candidates"]}
    ages, outcomes = {}, []
    for obs in sem["outcome_observations"]:
        pair = obs["family"], obs["effort"]
        if (*pair, obs["id"]) not in own:
            continue
        outcomes.append({"id": obs["id"], "family": pair[0], "effort": pair[1], "outcome": obs["outcome"],
                         "score": scored[obs["id"]]["score"], "similarity": scored[obs["id"]]["similarity"],
                         "age": ages.get(pair, 0), "weight": own[(*pair, obs["id"])]})
        ages[pair] = ages.get(pair, 0) + 1
    sem["outcome_observations"] = sorted(outcomes, key=lambda obs: (obs["family"], obs["effort"]))
    for obs in sem["duration_observations"]:
        del obs["tokens"], obs["token_usage"]
        obs.update(score=scored[obs["id"]]["score"], similarity=scored[obs["id"]]["similarity"])
    for item in cap["evidence"]:
        del item["strength"], item["ticket_share"]
        item.update(score=scored[item["id"]]["score"], similarity=scored[item["id"]]["similarity"])

    # v10 has no tiers: a cell is admissible or not, by its own Dirichlet posterior, and the
    # cap-unmet policy is recorded. Its policy aged evidence by count, bounded a Dirichlet
    # concentration, and had neither the two tier constants nor the worths and amount-belief keys.
    selection = receipt["selection"]
    selection["on_cap_unmet"] = on_cap_unmet
    selection["exploration"].pop("pool")
    selection["sampling_policy"] = {**DIRICHLET_SAMPLING, "outcome_recency": .95,
                                    **{k: selection["sampling_policy"][k] for k in ("uncertainty_multiplier", "admission_quantile")}}
    sem["sampling"]["policy"] = deepcopy(selection["sampling_policy"])
    sem["interval_kind"] = "dirichlet-posterior-mixture"
    # v10 calibrated each expert's active time, priced each expert's forecast, and checked the
    # average credit alone.
    panels = {k: panel_lookup(p) for k, p in sem["worlds"].items()}
    experts = sem["beliefs"]["experts"]
    candidates = {(c["model"], c["effort"]): c for c in receipt["native_candidates"]}
    for row in selection["cells"]:
        for field in ("pi", "tier", "outcome_counts", "active_time", "token_beliefs", "credit_variance",
                      "credit_p80_usd"):
            row.pop(field)
        pair = row["family"], row["effort"]
        candidate = candidates[row["model"], row["effort"]]
        row["T"] = sum(e["weight"] * calibrate_duration(panels[e["key"]][pair]["seconds"], pair,
                                                        sem["duration_observations"])["expected_seconds"]
                       for e in experts) / 3600
        credits = [legacy_credit(panels[e["key"]][pair], candidate) for e in experts]
        needed = None if None in credits else sum(e["weight"] * c for e, c in zip(experts, credits))
        row["needed_credit_usd"] = needed
        row["session_cost_usd"] = row["C_run"] = None if needed is None else needed * row["cost_per_credit"]
        # A fixture whose credit check differs between the two rules needs its own builder.
        assert row["credit_ok"] == (row["available_credit_usd"] is None or needed <= row["available_credit_usd"])
    # v10 held a Dirichlet posterior per expert in place of the cell's two beliefs.
    restore_dirichlet(receipt)
    restore_cap(receipt)
    return receipt


def to_v9(receipt):
    """A v9 receipt: undecayed kappa_0 Dirichlet outcome posteriors and capability v4."""
    receipt = to_v10(receipt)
    receipt["format"] = "howhard-decision-v9"
    cap = receipt["capability"]
    cap.pop("prior_sources", None)
    if cap["format"] == "howhard-capability-v5":
        cap["format"] = "howhard-capability-v4"
        cap["worlds"]["filter:llm"] = prior_cdf(cap["baseline"], cap["policy"])
    cap["evidence"] = cap["evidence"][:8]

    sem = receipt["semantic"]
    sem["format"] = "howhard-semantic-v6"
    sem["prompt_version"] = "semantic-reader-v7"
    sem["duration_observations"] = sem["duration_observations"][:8]

    restore_dirichlet(receipt, legacy_v9=True)
    restore_cap(receipt)
    return receipt


def with_capability_v3(receipt):
    """Freeze the prior from before v4 for a fixture with a contiguous band."""
    receipt = to_v9(receipt) if receipt.get("format") in ("howhard-decision-v11", "howhard-decision-v10") else deepcopy(receipt)
    state = receipt["capability"]
    lo, hi = state["order"].index(state["lower"]), state["order"].index(state["upper"])
    assert state["band"] == state["order"][lo:hi+1]
    state["format"] = "howhard-capability-v3"
    filters = {"filter:llm": state["baseline"]}
    filters.update({f"filter:memory:{m['id']}:{m['version']}": m["filter"] for m in state["memory_filters"]})
    state["worlds"] = {key: legacy_capability_prior(flt, state["policy"]) for key, flt in filters.items()}
    return receipt


def to_v8(receipt):
    """A v8 receipt: weighted-forecast envelopes, upper-bound admission, no outcome posteriors."""
    receipt = with_capability_v3(receipt)
    receipt["format"] = "howhard-decision-v8"
    semantic = receipt["semantic"]
    semantic["format"] = "howhard-semantic-v5"
    semantic["prompt_version"] = "semantic-reader-v6"
    semantic["interval_kind"] = "weighted-forecast-envelope"
    semantic.pop("outcome_observations")
    for panel in semantic["worlds"].values():
        for row in panel:
            row.pop("loud_ci", None)
            row.pop("silent_ci", None)
    semantic["sampling"]["policy"] = deepcopy(LEGACY_SAMPLING)
    if semantic["sampling"]["enabled"]:
        semantic["sampling"]["reason"] = "evidence-calibrated-dirichlet"
    for cell in semantic["sampling"]["cells"]:
        cell.pop("outcome_posterior")
        cell["concentration"] = max(8., cell["concentration"])
        cell.update(evidence_count=0, variance_sources={})
    selection = receipt["selection"]
    selection["sampling_policy"] = deepcopy(LEGACY_SAMPLING)
    panels = {k: panel_lookup(p) for k, p in semantic["worlds"].items()}
    experts = semantic["beliefs"]["experts"]
    for row in selection["cells"]:
        pair = row["family"], row["effort"]
        for outcome in OUTCOMES:
            mean = sum(e["weight"] * panels[e["key"]][pair][outcome] for e in experts)
            row["P_" + outcome] = envelope([(panels[e["key"]][pair][outcome], e["weight"]) for e in experts], mean)
        loud, silent = row["P_loud"]["mean"], row["P_silent"]["mean"]
        row["P_silent_accepted"] = envelope(
            [(accepted_risk(panels[e["key"]][pair]["loud"], panels[e["key"]][pair]["silent"]), e["weight"]) for e in experts],
            accepted_risk(loud, silent))
        row.pop("admission_risk")
        row["admissible"] = row["P_silent_accepted"]["upper"] <= selection["constants"]["epsilon"]
        run = row["session_cost_usd"] if receipt["cost_policy"]["worker_money_included"] else 0.
        row["J"] = cost(run, row["T"], loud, silent, selection["constants"])
        row["expected_attempts"] = 1 / denominator(loud)
    restore_cap(receipt)
    return receipt


def to_v7(receipt, capability_format="howhard-capability-v2"):
    """A v7 receipt additionally lacks active-duration calibration and timely capability fields."""
    receipt = to_v8(receipt)
    receipt["format"] = "howhard-decision-v7"
    receipt["semantic"]["format"] = "howhard-semantic-v4"
    receipt["semantic"]["prompt_version"] = "semantic-reader-v5"
    receipt["semantic"].pop("duration_observations", None)
    for cell in receipt["semantic"]["sampling"]["cells"]:
        cell.pop("duration", None)
    panels = {k: panel_lookup(p) for k, p in receipt["semantic"]["worlds"].items()}
    experts = receipt["semantic"]["beliefs"]["experts"]
    for row in receipt["selection"]["cells"]:
        pair = row["family"], row["effort"]
        row["T"] = sum(e["weight"] * panels[e["key"]][pair]["seconds"] for e in experts) / 3600
        run = row["session_cost_usd"] if receipt["cost_policy"]["worker_money_included"] else 0.
        row["J"] = cost(run, row["T"], row["P_loud"]["mean"], row["P_silent"]["mean"], receipt["selection"]["constants"])
    capability = receipt["capability"]
    capability["format"] = capability_format
    for cell in capability["cells"]:
        cell.pop("P_timely", None)
    for item in capability["evidence"]:
        for extra in ("active_seconds", "forecast_active_seconds", "timely_strength"):
            item.pop(extra, None)
    return receipt
