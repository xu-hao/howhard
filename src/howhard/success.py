"""Accepted-result retry costs and evidence-calibrated selection uncertainty."""

from copy import deepcopy
import math

import numpy as np

from .llm import SETTINGS
from .validation import HowHardError, keys, number

DEFAULTS = {"min_concentration": 8., "max_concentration": 1000000., "prior_strength": 8.,
            "explore_within": .1, "uncertainty_multiplier": 1.}


def validate_policy(policy):
    keys(policy, list(DEFAULTS), where="sampling")
    for key, value in policy.items():
        number(value, "sampling."+key, nonnegative=True)
    if not 1 <= policy["min_concentration"] <= policy["max_concentration"] or policy["prior_strength"] <= 0:
        raise HowHardError("sampling: require 1 <= min_concentration <= max_concentration and positive prior_strength")
    if policy["explore_within"] > 1 or policy["uncertainty_multiplier"] <= 0:
        raise HowHardError("sampling: explore_within must be in [0,1] and uncertainty_multiplier positive")
    return policy


def denominator(loud):
    return max(SETTINGS["belief"]["probability_floor"], 1-loud)


def accepted_risk(loud, silent):
    return min(1., silent/denominator(loud))


def cost(run, hours, loud, silent, constants):
    return (run+constants["lambda_per_hour"]*hours+constants["C_loud"]*loud+constants["C_silent"]*silent)/denominator(loud)


def envelope(values, mean):
    """Weighted joint-expert envelope expanded to contain the aggregate ratio."""
    values = sorted(values)
    def quantile(fraction):
        total = 0.
        for value, weight in values:
            total += weight
            if total >= fraction:
                return value
        return values[-1][0]
    return {"mean": mean, "lower": min(mean, quantile(.025)), "upper": max(mean, quantile(.975))}


def calibration(result, state, expert, cell, row):
    """Moment-match Dirichlet variance to evidence precision and expert disagreement."""
    from .semantic import OUTCOMES, panel_lookup, forecast_session_cost
    pair = cell["family"], cell["effort"]
    aggregate = next(c for c in result["cells"] if (c["family"], c["effort"]) == pair)
    probabilities = np.maximum([row[k] for k in OUTCOMES], SETTINGS["belief"]["probability_floor"])
    probabilities /= probabilities.sum()
    policy = result["sampling_policy"]
    count = expert["evidence_count"]
    estimates, concentrations = {}, []
    for i, name in enumerate(OUTCOMES):
        p = float(probabilities[i])
        band = aggregate["P_"+name]
        spread = sum(e["weight"] * (panel_lookup(state["worlds"][e["key"]])[pair][name]-band["mean"])**2
                     for e in state["beliefs"]["experts"])
        width_variance = ((band["upper"]-band["lower"])/3.92)**2
        count_variance = p*(1-p)/(policy["prior_strength"]+count+1)
        variance = max(spread, width_variance, count_variance, 1e-15)
        estimates[name] = {"between_experts": spread, "envelope": width_variance, "evidence": count_variance}
        concentrations.append(p*(1-p)/variance-1)
    concentration = max(policy["min_concentration"], min(policy["max_concentration"], min(concentrations)))
    # Delta-method uncertainty for the same-cell retry objective, including
    # the covariance between loud and silent probabilities in the Dirichlet.
    run = forecast_session_cost(row, cell) if cell["money_included"] else 0.
    constants = result["constants"]
    expected = cost(run, row["seconds"]/3600, probabilities[1], probabilities[2], constants)
    d = denominator(probabilities[1])
    gradient = np.array([0., (constants["C_loud"]+expected)/d, constants["C_silent"]/d])
    covariance = (np.diag(probabilities)-np.outer(probabilities, probabilities))/(concentration+1)
    variance = max(0., float(gradient @ covariance @ gradient))
    expert_costs = []
    for e in state["beliefs"]["experts"]:
        prediction = panel_lookup(state["worlds"][e["key"]])[pair]
        money = forecast_session_cost(prediction, cell) if cell["money_included"] else 0.
        expert_costs.append((e["weight"], cost(money, prediction["seconds"]/3600, prediction["loud"], prediction["silent"], constants)))
    mean_cost = sum(w*j for w, j in expert_costs)
    variance = max(variance, sum(w*(j-mean_cost)**2 for w, j in expert_costs))
    return {"family": cell["family"], "effort": cell["effort"], "expert": expert["key"],
            "evidence_count": count, "probabilities": probabilities.tolist(), "variance_sources": estimates,
            "concentration": concentration, "cost_sd": math.sqrt(variance)}


def validate_selection(selection, semantic, capability, cost_policy):
    """Validate frozen objective/cap meanings without reinterpreting older receipts."""
    keys(selection, ["objective", "retry_policy", "constants", "sampling_policy", "cells", "exploration", "cap_met", "needs_review", "on_cap_unmet"], where="decision.selection")
    if selection["objective"] != "cost-to-accepted-result" or selection["retry_policy"] != "same-cell-until-non-loud":
        raise HowHardError("decision.selection: unsupported objective/retry policy")
    validate_policy(selection["sampling_policy"])
    constants = selection["constants"]
    keys(constants, ["C_loud", "C_silent", "lambda_per_hour", "epsilon"], ["explore_within"], "selection.constants")
    for k, v in constants.items():
        number(v, "selection.constants."+k, nonnegative=True)
    if not 0 < constants["epsilon"] < 1:
        raise HowHardError("decision.selection: invalid accepted-result epsilon")
    from .semantic import OUTCOMES, panel_lookup
    panels = {key: panel_lookup(panel) for key, panel in semantic["worlds"].items()}
    experts = semantic["beliefs"]["experts"]
    known = {(c["family"], c["effort"]): c for c in capability["cells"]}
    seen = set()
    if not isinstance(selection["cells"], list):
        raise HowHardError("decision.selection.cells: expected array")
    for row in selection["cells"]:
        keys(row, ["model", "family", "effort", "P_clean", "P_loud", "P_silent", "C_run", "session_cost_usd", "cost_per_credit", "needed_credit_usd", "cost_basis", "T", "expected_attempts", "J", "P_silent_accepted", "admissible", "sampled_J", "credit_budget", "available_credit_usd", "credit_ok"], where="selection cell")
        for field in ("T", "expected_attempts", "J", "sampled_J"):
            number(row[field], "selection."+field, nonnegative=True)
        cost_per_credit = number(row["cost_per_credit"], "selection.cost_per_credit", nonnegative=True)
        for field in ("C_run", "session_cost_usd", "needed_credit_usd"):
            if row[field] is not None:
                number(row[field], "selection."+field, nonnegative=True)
        needed_credit = row["needed_credit_usd"]
        session_cost = row["session_cost_usd"]
        expected_session_cost = needed_credit * cost_per_credit if needed_credit is not None else None
        if (session_cost is None) != (expected_session_cost is None) or (
            session_cost is not None and abs(session_cost - expected_session_cost) > 1e-10 * max(1., abs(expected_session_cost))
        ):
            raise HowHardError("decision.selection: session_cost_usd must match needed_credit_usd times cost_per_credit")
        if (row["C_run"] is None) != (session_cost is None) or (
            session_cost is not None and abs(row["C_run"] - session_cost) > 1e-10 * max(1., abs(session_cost))
        ):
            raise HowHardError("decision.selection: C_run must match session_cost_usd")
        if type(row["admissible"]) is not bool:
            raise HowHardError("decision.selection: admissibility must be boolean")
        if row["credit_budget"] is not None:
            from .native import credit_budget
            credit_budget(row["credit_budget"])
        expected_available = row["credit_budget"]["available_credit_usd"] if row["credit_budget"] is not None else None
        if row["available_credit_usd"] != expected_available:
            raise HowHardError("decision.selection: available_credit_usd does not match credit_budget")
        expected_ok = True if row["available_credit_usd"] is None else (
            needed_credit is not None and needed_credit <= row["available_credit_usd"]
        )
        if type(row["credit_ok"]) is not bool or row["credit_ok"] != expected_ok:
            raise HowHardError("decision.selection: inconsistent credit_ok status")
        for field in ("P_clean", "P_loud", "P_silent", "P_silent_accepted"):
            keys(row[field], ["mean", "lower", "upper"], where="selection."+field)
            for value in row[field].values():
                number(value, "selection probability", nonnegative=True)
            if not 0 <= row[field]["lower"] <= row[field]["mean"] <= row[field]["upper"] <= 1+1e-10:
                raise HowHardError("decision.selection: inconsistent probability envelope")
        pair = row["family"], row["effort"]
        if pair not in panels["llm"] or pair not in known or pair in seen or row["model"] != known[pair]["model"]:
            raise HowHardError("decision.selection: unknown, duplicate or inconsistent forecasted cell")
        seen.add(pair)
        for outcome in OUTCOMES:
            mean = sum(e["weight"]*panels[e["key"]][pair][outcome] for e in experts)
            expected_envelope = envelope([(panels[e["key"]][pair][outcome], e["weight"]) for e in experts], mean)
            if row["P_"+outcome] != expected_envelope:
                raise HowHardError("decision.selection: probability mean mismatch")
        loud, silent = row["P_loud"]["mean"], row["P_silent"]["mean"]
        expected_risk = envelope([(accepted_risk(panels[e["key"]][pair]["loud"], panels[e["key"]][pair]["silent"]), e["weight"]) for e in experts], accepted_risk(loud, silent))
        if row["P_silent_accepted"] != expected_risk or row["admissible"] != (expected_risk["upper"] <= constants["epsilon"]):
            raise HowHardError("decision.selection: accepted-result risk/cap mismatch")
        expected_hours = sum(e["weight"]*panels[e["key"]][pair]["seconds"] for e in experts)/3600
        if abs(row["T"]-expected_hours) > 1e-10:
            raise HowHardError("decision.selection: expected time mismatch")
        run = row["session_cost_usd"] if cost_policy["worker_money_included"] else 0.
        if run is not None:
            number(run, "selection.session_cost_usd", nonnegative=True)
        expected_cost = cost(run, expected_hours, loud, silent, constants)
        if abs(row["J"]-expected_cost) > 1e-8*max(1, expected_cost) or abs(row["expected_attempts"]-1/denominator(loud)) > 1e-8:
            raise HowHardError("decision.selection: cost-to-accepted-result mismatch")
    if seen != set(panels["llm"]):
        raise HowHardError("decision.selection: missing forecasted cells")
    if not any(c.get("credit_ok", True) for c in selection["cells"]):
        raise HowHardError("decision.selection: no schedulable cells with sufficient credit")
    cap_met = any(c["admissible"] and c.get("credit_ok", True) for c in selection["cells"])
    if semantic["sampling"]["enabled"] != cap_met:
        raise HowHardError("decision.selection: sampling and cap status disagree")
    if type(selection["cap_met"]) is not bool or selection["cap_met"] != cap_met or selection["needs_review"] != (not cap_met):
        raise HowHardError("decision.selection: inconsistent cap/review state")
    exploration = selection["exploration"]
    if type(exploration.get("enabled")) is not bool or (not cap_met and (exploration["enabled"] or exploration["reason"] != "disabled-cap-unmet")):
        raise HowHardError("decision.selection: exploration must be disabled when cap is unmet")
    return selection
