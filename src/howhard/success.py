"""Accepted-result retry costs and per-cell Dirichlet outcome posteriors."""

from copy import deepcopy
import math

import numpy as np

from .context import DECISION_FORMAT, LEGACY_DECISION_V7, LEGACY_DECISION_V8
from .llm import SETTINGS
from .validation import HowHardError, keys, number

DEFAULTS = {"min_concentration": 2., "max_concentration": 1000000., "prior_strength": 8.,
            "uncertainty_multiplier": 1., "admission_quantile": .8, "outcome_recency": .95}
# Frozen v7/v8 receipts recorded this policy shape; it is read-only history.
LEGACY_DEFAULTS = ("min_concentration", "max_concentration", "prior_strength", "explore_within", "uncertainty_multiplier")
INTERVAL = (.025, .975)
CI_Z_WIDTH = 3.92  # A central 95% interval spans about 3.92 standard deviations.
POSTERIOR_FIELDS = ("prior", "prior_concentration", "counts", "evidence_weight", "alpha")
DURATION_PRIOR_STRENGTH = 1.0
MAX_DURATION_LOG_SD = 2.0


def calibrate_duration(forecast_seconds, pair, duration_observations):
    forecast = number(forecast_seconds, "duration.forecast_seconds", nonnegative=True)
    matching = [obs for obs in duration_observations if (obs["family"], obs["effort"]) == pair and obs["weight"] > 0]
    weight = math.fsum(obs["weight"] for obs in matching)
    if weight <= 0:
        mu = math.log(max(1.0, forecast)) if forecast > 0 else 0.0
        return {"forecast_seconds": forecast, "expected_seconds": forecast,
                "mu": mu, "sigma": 0.0, "evidence_weight": 0.0}
    y0 = math.log(max(1.0, forecast))
    ys = [math.log(max(1.0, obs["active_seconds"])) for obs in matching]
    ws = [obs["weight"] for obs in matching]
    y_bar = math.fsum(w * y for w, y in zip(ws, ys)) / weight
    alpha = weight / (weight + DURATION_PRIOR_STRENGTH)
    delta = y_bar - y0
    mu = y0 + alpha * delta
    sample_var = math.fsum(w * (y - y_bar) ** 2 for w, y in zip(ws, ys)) / weight
    sigma_sq = min(MAX_DURATION_LOG_SD ** 2, alpha * sample_var + alpha * (1.0 - alpha) * (delta ** 2))
    sigma = math.sqrt(max(0.0, sigma_sq))
    expected = math.exp(mu + 0.5 * sigma_sq)
    return {"forecast_seconds": forecast, "expected_seconds": expected,
            "mu": mu, "sigma": sigma, "evidence_weight": weight}


def validate_policy(policy, *, legacy=False):
    if not legacy and isinstance(policy, dict) and "explore_within" in policy:
        raise HowHardError("sampling.explore_within: removed; Thompson eligibility uses posterior overlap. "
                           "Set explore_within on a risk class to cap the mean-cost exploration band explicitly")
    keys(policy, list(LEGACY_DEFAULTS if legacy else DEFAULTS), where="sampling")
    for key, value in policy.items():
        number(value, "sampling."+key, nonnegative=True)
    if not 1 <= policy["min_concentration"] <= policy["max_concentration"] or policy["prior_strength"] <= 0:
        raise HowHardError("sampling: require 1 <= min_concentration <= max_concentration and positive prior_strength")
    if policy["uncertainty_multiplier"] <= 0:
        raise HowHardError("sampling: uncertainty_multiplier must be positive")
    if legacy:
        if policy["explore_within"] > 1:
            raise HowHardError("sampling: explore_within must be in [0,1]")
        return policy
    if policy["min_concentration"] > policy["prior_strength"]:
        raise HowHardError("sampling: require min_concentration <= prior_strength (the bounded forecast prior range)")
    if not .5 <= policy["admission_quantile"] < 1:
        raise HowHardError("sampling.admission_quantile: expected a probability in [0.5, 1)")
    if not 0 < policy["outcome_recency"] <= 1:
        raise HowHardError("sampling.outcome_recency: expected a decay factor in (0, 1]")
    return policy


def _beta_continued_fraction(a, b, x):
    tiny, qab, qap, qam = 1e-300, a + b, a + 1., a - 1.
    c, d = 1., 1. - qab * x / qap
    d = 1. / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 501):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1. + aa * d
        d = 1. / (d if abs(d) > tiny else tiny)
        c = 1. + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1. + aa * d
        d = 1. / (d if abs(d) > tiny else tiny)
        c = 1. + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.) < 1e-15:
            break
    return h


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_cdf(x, a, b):
    """Regularized incomplete beta function I_x(a, b) for a, b > 0."""
    if x <= 0.:
        return 0.
    if x >= 1.:
        return 1.
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - _log_beta(a, b))
    if x < (a + 1.) / (a + b + 2.):
        return min(1., front * _beta_continued_fraction(a, b, x) / a)
    return max(0., 1. - front * _beta_continued_fraction(b, a, 1. - x) / b)


def _sigmoid(t):
    return 1. / (1. + math.exp(-t)) if t >= 0 else math.exp(t) / (1. + math.exp(t))


def _beta_density_logit(t, a, b):
    """Beta density with respect to t = logit(x): x^a (1-x)^b / B(a, b)."""
    log_x = -math.log1p(math.exp(-t)) if t > -30 else t
    log_1mx = -math.log1p(math.exp(t)) if t < 30 else -t
    return math.exp(a * log_x + b * log_1mx - _log_beta(a, b))


def mixture_quantile(components, q):
    """Quantile of a weighted Beta mixture [(weight, a, b)], solved by safeguarded Newton in logit space."""
    components = [(w, a, b) for w, a, b in components if w > 0]
    def cdf(t):
        x = _sigmoid(t)
        if x <= 0. or x >= 1.:  # Underflow at the extremes keeps the bracket finite.
            return 0. if x <= 0. else 1.
        return math.fsum(w * beta_cdf(x, a, b) for w, a, b in components)
    lo, hi = -700., 36.
    mean = math.fsum(w * a / (a + b) for w, a, b in components)
    t = min(hi, max(lo, math.log(mean) - math.log1p(-mean) if 0 < mean < 1 else 0.))
    for _ in range(200):
        value = cdf(t) - q
        if value == 0.:
            return _sigmoid(t)
        if value < 0:
            lo = t
        else:
            hi = t
        if hi - lo < 1e-12 * max(1., abs(t)):
            break
        density = math.fsum(w * _beta_density_logit(t, a, b) for w, a, b in components)
        step = t - value / density if density > 0 else None
        t = step if step is not None and lo < step < hi else (lo + hi) / 2.
    return _sigmoid((lo + hi) / 2.)


def mixture_interval(components):
    """Mean and 95% equal-tailed credible interval of a weighted Beta mixture."""
    mean = math.fsum(w * a / (a + b) for w, a, b in components)
    lower, upper = (mixture_quantile(components, q) for q in INTERVAL)
    return {"mean": mean, "lower": min(mean, lower), "upper": max(mean, upper)}


def forecast_probabilities(row):
    from .semantic import OUTCOMES
    probabilities = np.maximum([float(row[k]) for k in OUTCOMES], SETTINGS["belief"]["probability_floor"])
    return probabilities / probabilities.sum()


def prior_concentration(row, policy, probabilities=None):
    """Bounded forecast pseudo-count kappa_0; optional credible intervals can only widen it."""
    p = forecast_probabilities(row) if probabilities is None else probabilities
    kappa = policy["prior_strength"]
    for index, name in ((1, "loud"), (2, "silent")):
        interval = row.get(name + "_ci")
        if interval is None:
            continue
        sd = (interval[1] - interval[0]) / CI_Z_WIDTH
        if sd > 0:
            kappa = min(kappa, p[index] * (1 - p[index]) / sd ** 2 - 1)
    return float(min(policy["prior_strength"], max(policy["min_concentration"], kappa)))


def outcome_counts(pair, outcome_observations):
    from .semantic import OUTCOMES
    counts = [[], [], []]
    for obs in outcome_observations:
        if (obs["family"], obs["effort"]) == pair and obs["weight"] > 0:
            counts[OUTCOMES.index(obs["outcome"])].append(obs["weight"])
    return [math.fsum(c) for c in counts]


def outcome_posterior(row, pair, outcome_observations, policy):
    """Dirichlet posterior alpha = kappa_0 * p_0 + this cell's own weighted outcome counts."""
    p0 = forecast_probabilities(row)
    kappa = prior_concentration(row, policy, p0)
    counts = outcome_counts(pair, outcome_observations)
    alpha = kappa * p0 + np.asarray(counts)
    total = float(alpha.sum())
    if total > policy["max_concentration"]:
        alpha *= policy["max_concentration"] / total
        total = float(alpha.sum())
    return {"prior": p0.tolist(), "prior_concentration": kappa, "counts": counts,
            "evidence_weight": math.fsum(counts), "alpha": alpha.tolist(),
            "concentration": total, "mean": (alpha / total).tolist()}


def posterior_intervals(panels, weights, pair, outcome_observations, policy):
    """Expert-mixture Dirichlet marginals plus the Beta(alpha_silent, alpha_clean) accepted risk."""
    from .semantic import OUTCOMES
    posteriors = {k: outcome_posterior(panels[k][pair], pair, outcome_observations, policy) for k in weights}
    intervals = {}
    for i, name in enumerate(OUTCOMES):
        components = [(weights[k], posteriors[k]["alpha"][i],
                       math.fsum(a for j, a in enumerate(posteriors[k]["alpha"]) if j != i)) for k in weights]
        intervals["P_" + name] = mixture_interval(components)
    accepted = [(weights[k], posteriors[k]["alpha"][2], posteriors[k]["alpha"][0]) for k in weights]
    intervals["P_silent_accepted"] = mixture_interval(accepted)
    intervals["admission_risk"] = mixture_quantile(accepted, policy["admission_quantile"])
    return intervals, posteriors


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
    """Per-cell Dirichlet posterior: a bounded forecast prior plus only this cell's own outcomes."""
    from .semantic import panel_lookup, forecast_session_cost
    pair = cell["family"], cell["effort"]
    policy = result["sampling_policy"]
    outcome_obs = state.get("outcome_observations", [])
    posterior = outcome_posterior(row, pair, outcome_obs, policy)
    probabilities = np.asarray(posterior["mean"])
    concentration = posterior["concentration"]
    duration_obs = state.get("duration_observations", [])
    dur = calibrate_duration(row["seconds"], pair, duration_obs)
    expert_durations = {}
    for e in state["beliefs"]["experts"]:
        prediction = panel_lookup(state["worlds"][e["key"]])[pair]
        expert_durations[e["key"]] = calibrate_duration(prediction["seconds"], pair, duration_obs)
    mean_mu = sum(e["weight"] * expert_durations[e["key"]]["mu"] for e in state["beliefs"]["experts"])
    between_log_var = sum(e["weight"] * (expert_durations[e["key"]]["mu"] - mean_mu) ** 2
                          for e in state["beliefs"]["experts"])
    dur["between_experts_log_variance"] = between_log_var
    # Delta-method uncertainty for the same-cell retry objective, including
    # Dirichlet outcome covariance and log-normal active-duration right-tail variance.
    run = forecast_session_cost(row, cell) if cell["money_included"] else 0.
    constants = result["constants"]
    hours = dur["expected_seconds"] / 3600
    expected = cost(run, hours, probabilities[1], probabilities[2], constants)
    d = denominator(probabilities[1])
    gradient = np.array([0., (constants["C_loud"]+expected)/d, constants["C_silent"]/d])
    covariance = (np.diag(probabilities)-np.outer(probabilities, probabilities))/(concentration+1)
    outcome_variance = max(0., float(gradient @ covariance @ gradient))
    sigma_cost_sq = min(MAX_DURATION_LOG_SD ** 2, max(dur["sigma"] ** 2, between_log_var))
    hours_variance = (math.exp(sigma_cost_sq) - 1.0) * (hours ** 2)
    variance = outcome_variance + ((constants["lambda_per_hour"] / d) ** 2) * hours_variance
    expert_costs = []
    for e in state["beliefs"]["experts"]:
        prediction = panel_lookup(state["worlds"][e["key"]])[pair]
        money = forecast_session_cost(prediction, cell) if cell["money_included"] else 0.
        e_hours = expert_durations[e["key"]]["expected_seconds"] / 3600
        mean = outcome_posterior(prediction, pair, outcome_obs, policy)["mean"]
        expert_costs.append((e["weight"], cost(money, e_hours, mean[1], mean[2], constants)))
    mean_cost = sum(w*j for w, j in expert_costs)
    variance = max(variance, sum(w*(j-mean_cost)**2 for w, j in expert_costs))
    return {"family": cell["family"], "effort": cell["effort"], "expert": expert["key"],
            "outcome_posterior": {k: posterior[k] for k in POSTERIOR_FIELDS},
            "probabilities": probabilities.tolist(), "concentration": concentration,
            "duration": dur, "cost_sd": math.sqrt(variance)}


def validate_selection(selection, semantic, capability, cost_policy, *, decision_format=DECISION_FORMAT):
    """Validate frozen objective/cap meanings without reinterpreting older receipts."""
    keys(selection, ["objective", "retry_policy", "constants", "sampling_policy", "cells", "exploration", "cap_met", "needs_review", "on_cap_unmet"], where="decision.selection")
    if selection["objective"] != "cost-to-accepted-result" or selection["retry_policy"] != "same-cell-until-non-loud":
        raise HowHardError("decision.selection: unsupported objective/retry policy")
    legacy = decision_format in (LEGACY_DECISION_V7, LEGACY_DECISION_V8)
    validate_policy(selection["sampling_policy"], legacy=legacy)
    policy = selection["sampling_policy"]
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
        keys(row, ["model", "family", "effort", "P_clean", "P_loud", "P_silent", "C_run", "session_cost_usd", "cost_per_credit", "needed_credit_usd", "cost_basis", "T", "expected_attempts", "J", "P_silent_accepted", "admissible", "sampled_J", "credit_budget", "available_credit_usd", "credit_ok"]
             + ([] if legacy else ["admission_risk"]), where="selection cell")
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
        if legacy:
            for outcome in OUTCOMES:
                mean = sum(e["weight"]*panels[e["key"]][pair][outcome] for e in experts)
                expected_envelope = envelope([(panels[e["key"]][pair][outcome], e["weight"]) for e in experts], mean)
                if row["P_"+outcome] != expected_envelope:
                    raise HowHardError("decision.selection: probability mean mismatch")
            loud, silent = row["P_loud"]["mean"], row["P_silent"]["mean"]
            expected_risk = envelope([(accepted_risk(panels[e["key"]][pair]["loud"], panels[e["key"]][pair]["silent"]), e["weight"]) for e in experts], accepted_risk(loud, silent))
            if row["P_silent_accepted"] != expected_risk or row["admissible"] != (expected_risk["upper"] <= constants["epsilon"]):
                raise HowHardError("decision.selection: accepted-result risk/cap mismatch")
        else:
            weights = {e["key"]: e["weight"] for e in experts}
            intervals, _ = posterior_intervals(panels, weights, pair, semantic["outcome_observations"], policy)
            for field in ("P_clean", "P_loud", "P_silent", "P_silent_accepted"):
                if any(abs(row[field][k] - intervals[field][k]) > 1e-9 for k in ("mean", "lower", "upper")):
                    raise HowHardError("decision.selection: posterior credible interval mismatch")
            risk = number(row["admission_risk"], "selection.admission_risk", nonnegative=True)
            if abs(risk - intervals["admission_risk"]) > 1e-9 or row["admissible"] != (risk <= constants["epsilon"]):
                raise HowHardError("decision.selection: accepted-result risk/cap mismatch")
            loud, silent = row["P_loud"]["mean"], row["P_silent"]["mean"]
        if decision_format == LEGACY_DECISION_V7:
            expected_hours = sum(e["weight"]*panels[e["key"]][pair]["seconds"] for e in experts)/3600
        else:
            duration_obs = semantic["duration_observations"]
            expected_hours = sum(
                e["weight"] * calibrate_duration(panels[e["key"]][pair]["seconds"], pair, duration_obs)["expected_seconds"]
                for e in experts
            ) / 3600
        if abs(row["T"]-expected_hours) > 1e-10 * max(1., expected_hours):
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
