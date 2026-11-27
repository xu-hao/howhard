"""Accepted-result retry costs, per-cell Dirichlet outcome posteriors and amount beliefs."""

from copy import deepcopy
import math
import sys

import numpy as np

from .context import DECISION_FORMAT, LEGACY_DECISION_V7, LEGACY_DECISION_V8, LEGACY_DECISION_V9, LEGACY_DECISION_V10
from .llm import SETTINGS
from .validation import HowHardError, keys, number

DEFAULTS = {"min_concentration": 2., "max_concentration": 1000000., "prior_strength": 8.,
            "uncertainty_multiplier": 1., "admission_quantile": .8, "trial_probability": .2,
            "all_excluded_factor": .5, "half_life_days": 14., "comparison_cap": 256,
            "model_worth": 5., "task_worth_amount": 1., "duration_miss": 1.3, "scatter_share": .5}
# Frozen v7/v8 receipts recorded this policy shape; it is read-only history.
LEGACY_DEFAULTS = ("min_concentration", "max_concentration", "prior_strength", "explore_within", "uncertainty_multiplier")
# Frozen v9/v10 receipts aged evidence by count (outcome_recency), not by time, and had no
# admission tiers (trial_probability, all_excluded_factor); read-only history.
COUNT_AGED_DEFAULTS = ("min_concentration", "max_concentration", "prior_strength", "uncertainty_multiplier",
                       "admission_quantile", "outcome_recency")
POOLS = ("trusted-and-on-trial", "all-excluded")
INTERVAL = (.025, .975)
CI_Z_WIDTH = 3.92  # A central 95% interval spans about 3.92 standard deviations.
P80_Z = 0.8416  # The 80% point of the standard normal.
# exp() overflows a float just above 709. Only a stated range wider than a factor of e^100
# reaches this; its variance then saturates instead of overflowing.
EXP_CAP = 700.
POSTERIOR_FIELDS = ("prior", "prior_concentration", "counts", "evidence_weight", "alpha")
# Frozen v8 to v10 receipts calibrated active time with these; see calibrate_duration.
DURATION_PRIOR_STRENGTH = 1.0
MAX_DURATION_LOG_SD = 2.0
EMPIRICAL_TAU = 2.0
EMPIRICAL_FLOOR = 1.0
EMPIRICAL_BASE = (0.88, 0.08, 0.04)


def calibrate_duration(forecast_seconds, pair, duration_observations):
    """Active-time calibration of frozen v8 to v10 receipts; v11 uses ``amount_belief``."""
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


def log_amount(value):
    """The log of an amount or a range end, held at 1 or more."""
    return math.log(max(1., value))


def merge_forecasts(forecasts):
    """Merge the experts' forecasts of one value: ``[(weight, forecast, range or None)]``.

    Forecasts are averaged on the log scale and disagreement counts as stated
    uncertainty. Returns the log of the merged forecast and its stated spread.
    """
    logs = [(w, log_amount(v), (log_amount(r[1]) - log_amount(r[0])) / CI_Z_WIDTH if r is not None else 0.)
            for w, v, r in forecasts]
    centre = math.fsum(w * m for w, m, _ in logs)
    return centre, math.sqrt(math.fsum(w * (s ** 2 + (m - centre) ** 2) for w, m, s in logs))


def amount_belief(forecasts, runs, policy, unit, *, zero=False):
    """Log-normal belief for active time or one token count, after same-cell runs.

    ``forecasts`` is ``[(expert weight, forecast, stated range or None)]`` and ``runs``
    is ``[(weight, recorded amount)]``. The belief starts at the merged forecast with
    spread ``max(stated spread, tau / sqrt(N_start))`` and a centre that makes its
    average equal the forecast; each run then pulls the centre toward its own amount
    and narrows the belief by its weight. With ``zero``, a value every expert forecasts
    at zero with no non-zero upper end is exactly zero and has no spread.
    """
    worth, miss = policy["model_worth"], policy["duration_miss"]
    n_start = 1 / (1 / worth + 1 / policy["task_worth_amount"])
    if zero and all(v == 0 and (r is None or r[1] == 0) for _, v, r in forecasts):
        return {"forecast_" + unit: 0., "stated_spread": 0., "worth": worth, "N_start": n_start, "D": miss,
                "prior_centre": None, "prior_spread": 0., "centre": None, "spread": 0.,
                "evidence_weight": 0., "expected_" + unit: 0.}
    tau = miss * math.sqrt(policy["scatter_share"])
    log_forecast, stated = merge_forecasts(forecasts)
    prior_spread = max(stated, tau / math.sqrt(n_start))
    prior_centre = log_forecast - prior_spread ** 2 / 2
    runs = [(w, log_amount(v)) for w, v in runs if w > 0]
    weight = math.fsum(w for w, _ in runs)
    centre, spread = prior_centre, prior_spread
    if runs:
        variance = 1 / (1 / prior_spread ** 2 + weight / tau ** 2)
        centre = prior_centre + variance * math.fsum(w * (y - prior_centre) for w, y in runs) / tau ** 2
        spread = math.sqrt(variance)
    return {"forecast_" + unit: math.exp(log_forecast), "stated_spread": stated, "worth": worth,
            "N_start": n_start, "D": miss, "prior_centre": prior_centre, "prior_spread": prior_spread,
            "centre": centre, "spread": spread, "evidence_weight": weight,
            "expected_" + unit: math.exp(centre + spread ** 2 / 2)}


def cell_beliefs(forecasts, pair, duration_observations, policy):
    """Beliefs for one cell: ``active_time`` and ``token_beliefs``.

    ``forecasts`` is ``[(expert weight, that expert's forecast row for the cell)]``.
    Every token kind has a belief when every expert states usage by kind; otherwise
    total ``tokens`` has the one belief. Runs are the cell's own rows from the
    active-time reader; a kind is updated by the runs that record usage by kind, and
    the total by the runs that record ``tokens``.
    """
    from .accounting import CATEGORIES
    runs = [obs for obs in duration_observations if (obs["family"], obs["effort"]) == pair]
    active = amount_belief([(w, row["seconds"], row.get("seconds_ci")) for w, row in forecasts],
                           [(obs["weight"], obs["active_seconds"]) for obs in runs], policy, "seconds")
    if all("token_usage" in row for _, row in forecasts):
        measured = [obs for obs in runs if obs["token_usage"] is not None]
        tokens = {kind: amount_belief(
            [(w, row["token_usage"][kind], row.get("token_usage_ci", {}).get(kind)) for w, row in forecasts],
            [(obs["weight"], obs["token_usage"][kind]) for obs in measured], policy, "tokens", zero=True)
            for kind in CATEGORIES}
    else:
        tokens = {"tokens": amount_belief(
            [(w, row["tokens"], None) for w, row in forecasts],
            [(obs["weight"], obs["tokens"]) for obs in runs if obs["tokens"] is not None], policy, "tokens", zero=True)}
    return {"active_time": active, "token_beliefs": tokens}


def run_credit(token_beliefs, rate_card, credit_rate):
    """Average, variance and 80% point of the credit one run needs, at the cell's prices.

    Each belief's average is priced at its rate. The kinds move together (``rho = 1``)
    because no run records how they co-vary, and the 80% point is that of the
    log-normal with the same average and variance. An unpriced cell returns ``None``.
    """
    from .accounting import CATEGORIES, price
    expected = {kind: belief["expected_tokens"] for kind, belief in token_beliefs.items()}
    if rate_card is not None:
        if expected.keys() != set(CATEGORIES):
            raise HowHardError("decision.semantic: category rate card requires token_usage forecasts")
        if any(rate_card["rates"][kind] is None for kind in CATEGORIES):
            return None
        costs = price(expected, rate_card, by_kind=True)
    elif credit_rate is None:
        return None
    else:
        costs = {kind: credit_rate * count / 1e6 for kind, count in expected.items()}
    mean = math.fsum(costs.values())
    if mean <= 0:
        return 0., 0., 0.
    shares = [(cost / mean, token_beliefs[kind]["spread"]) for kind, cost in costs.items() if cost > 0]
    # Var(K) / E[K]^2 = sum_a sum_b p_a p_b (exp(s_a s_b) - 1) with p_a = c_a / E[K].
    relative = math.fsum(pa * pb * math.expm1(min(sa * sb, EXP_CAP)) for pa, sa in shares for pb, sb in shares)
    sigma_sq = math.log1p(relative)
    p80 = mean * math.exp(P80_Z * math.sqrt(sigma_sq) - sigma_sq / 2)
    return mean, min(sys.float_info.max, mean * mean * relative), p80


def credit_fits(needed_credit, credit_p80, available_credit):
    """A run must fit the remaining credit four times out of five, not only on average.

    The larger of the two is checked: a very wide belief has its 80% point below its
    average, and the check must not get easier as a belief gets wider.
    """
    if available_credit is None:
        return True
    return needed_credit is not None and max(needed_credit, credit_p80) <= available_credit


def validate_policy(policy, *, legacy=False, count_aged=False):
    """Validate the sampling policy; ``legacy`` reads v7/v8 receipts and ``count_aged`` v9/v10 ones.

    Only the current policy carries the time-based evidence constants and the two tier constants.
    """
    if not legacy and isinstance(policy, dict) and "explore_within" in policy:
        raise HowHardError("sampling.explore_within: removed; Thompson eligibility uses posterior overlap. "
                           "Set explore_within on a risk class to cap the mean-cost exploration band explicitly")
    keys(policy, list(LEGACY_DEFAULTS if legacy else COUNT_AGED_DEFAULTS if count_aged else DEFAULTS), where="sampling")
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
    if count_aged:
        if not 0 < policy["outcome_recency"] <= 1:
            raise HowHardError("sampling.outcome_recency: expected a decay factor in (0, 1]")
        return policy
    if policy["half_life_days"] <= 0:
        raise HowHardError("sampling.half_life_days: expected a positive number of days")
    if type(policy["comparison_cap"]) is not int or policy["comparison_cap"] < 1:
        raise HowHardError("sampling.comparison_cap: expected a positive integer")
    if policy["trial_probability"] > policy["admission_quantile"]:
        raise HowHardError("sampling.trial_probability: expected a probability in [0, admission_quantile]")
    if policy["all_excluded_factor"] > 1:
        raise HowHardError("sampling.all_excluded_factor: expected a factor in [0, 1]")
    if min(policy[k] for k in ("model_worth", "task_worth_amount", "duration_miss")) <= 0:
        raise HowHardError("sampling: model_worth, task_worth_amount and duration_miss must be positive")
    if not 0 < policy["scatter_share"] <= 1:
        raise HowHardError("sampling.scatter_share: expected a share in (0, 1]")
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


def mixture_cdf(components, x):
    """P(X <= x) under a weighted Beta mixture [(weight, a, b)]: sum of w * I_x(a, b)."""
    return math.fsum(w * beta_cdf(x, a, b) for w, a, b in components if w > 0)


def mixture_quantile(components, q):
    """Quantile of a weighted Beta mixture [(weight, a, b)], solved by safeguarded Newton in logit space."""
    components = [(w, a, b) for w, a, b in components if w > 0]
    def cdf(t):
        x = _sigmoid(t)
        if x <= 0. or x >= 1.:  # Underflow at the extremes keeps the bracket finite.
            return 0. if x <= 0. else 1.
        return mixture_cdf(components, x)
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


def outcome_posterior(row, pair, outcome_observations, policy, *, legacy_v9=False):
    """Dirichlet posterior with continuous cold-start prior decay over empirical weight."""
    p0 = forecast_probabilities(row)
    kappa0 = prior_concentration(row, policy, p0)
    counts = outcome_counts(pair, outcome_observations)
    w_emp = math.fsum(counts)
    if legacy_v9:
        kappa = kappa0
        alpha = kappa * p0 + np.asarray(counts)
    else:
        decay = max(0.0, 1.0 - w_emp / EMPIRICAL_TAU)
        kappa_prior = kappa0 * decay
        kappa_base = EMPIRICAL_FLOOR * (1.0 - decay)
        kappa = float(kappa_prior + kappa_base)
        alpha = kappa_prior * p0 + kappa_base * np.asarray(EMPIRICAL_BASE) + np.asarray(counts)
    total = float(alpha.sum())
    if total > policy["max_concentration"]:
        alpha *= policy["max_concentration"] / total
        total = float(alpha.sum())
    return {"prior": p0.tolist(), "prior_concentration": kappa, "counts": counts,
            "evidence_weight": w_emp, "alpha": alpha.tolist(),
            "concentration": total, "mean": (alpha / total).tolist()}


def admission_tier(pi, policy):
    """Admission tier of a cell from its chance ``pi`` of meeting the silent limit."""
    if pi >= policy["admission_quantile"]:
        return "trusted"
    return "on-trial" if pi >= policy["trial_probability"] else "excluded"


def admission_pool(cells, policy):
    """The cells that compete, and which of the two pools they form.

    Trusted and on-trial cells with enough credit compete together. When there is none, the
    cells with enough credit whose ``pi`` is nearest the best one compete instead.
    """
    schedulable = [row for row in cells if row.get("credit_ok", True)]
    pool = [row for row in schedulable if row["tier"] != "excluded"]
    if pool or not schedulable:
        return pool, POOLS[0]
    bar = policy["all_excluded_factor"] * max(row["pi"] for row in schedulable)
    return [row for row in schedulable if row["pi"] >= bar], POOLS[1]


def posterior_intervals(panels, weights, pair, outcome_observations, policy, *, legacy_v9=False, epsilon=None):
    """Expert-mixture Dirichlet marginals plus the Beta(alpha_silent, alpha_clean) accepted risk.

    With ``epsilon``, also ``pi``: the share of that accepted-risk mixture at or below ``epsilon``.
    """
    from .semantic import OUTCOMES
    posteriors = {
        k: outcome_posterior(panels[k][pair], pair, outcome_observations, policy, legacy_v9=legacy_v9)
        for k in weights
    }
    intervals = {}
    for i, name in enumerate(OUTCOMES):
        components = [(weights[k], posteriors[k]["alpha"][i],
                       math.fsum(a for j, a in enumerate(posteriors[k]["alpha"]) if j != i)) for k in weights]
        intervals["P_" + name] = mixture_interval(components)
    accepted = [(weights[k], posteriors[k]["alpha"][2], posteriors[k]["alpha"][0]) for k in weights]
    intervals["P_silent_accepted"] = mixture_interval(accepted)
    intervals["admission_risk"] = mixture_quantile(accepted, policy["admission_quantile"])
    if epsilon is not None:
        intervals["pi"] = min(1., mixture_cdf(accepted, epsilon))
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
    """Per-cell Dirichlet posterior and the first-order spread of the cost to an accepted result.

    Active time and run cost are the cell's own beliefs, already merged over the experts:
    ``T`` and ``C_run`` are their averages, the time term of the spread is
    ``E[T]^2 * (exp(s'^2) - 1)``, and run cost is exact.
    """
    from .semantic import panel_lookup
    pair = cell["family"], cell["effort"]
    policy = result["sampling_policy"]
    outcome_obs = state.get("outcome_observations", [])
    posterior = outcome_posterior(row, pair, outcome_obs, policy)
    probabilities = np.asarray(posterior["mean"])
    concentration = posterior["concentration"]
    summary = panel_lookup(result["cells"])[pair]
    # Delta-method uncertainty for the same-cell retry objective, including
    # Dirichlet outcome covariance and log-normal active-time variance.
    run = summary["session_cost_usd"] if cell["money_included"] else 0.
    constants = result["constants"]
    hours = summary["T"]
    expected = cost(run, hours, probabilities[1], probabilities[2], constants)
    d = denominator(probabilities[1])
    gradient = np.array([0., (constants["C_loud"]+expected)/d, constants["C_silent"]/d])
    covariance = (np.diag(probabilities)-np.outer(probabilities, probabilities))/(concentration+1)
    outcome_variance = max(0., float(gradient @ covariance @ gradient))
    hours_variance = min(sys.float_info.max, hours ** 2 * math.expm1(min(summary["active_time"]["spread"] ** 2, EXP_CAP)))
    variance = outcome_variance + ((constants["lambda_per_hour"] / d) ** 2) * hours_variance
    expert_costs = []
    for e in state["beliefs"]["experts"]:
        prediction = panel_lookup(state["worlds"][e["key"]])[pair]
        mean = outcome_posterior(prediction, pair, outcome_obs, policy)["mean"]
        expert_costs.append((e["weight"], cost(run, hours, mean[1], mean[2], constants)))
    mean_cost = sum(w*j for w, j in expert_costs)
    variance = max(variance, sum(w*(j-mean_cost)**2 for w, j in expert_costs))
    return {"family": cell["family"], "effort": cell["effort"], "expert": expert["key"],
            "outcome_posterior": {k: posterior[k] for k in POSTERIOR_FIELDS},
            "probabilities": probabilities.tolist(), "concentration": concentration,
            "cost_sd": min(sys.float_info.max, math.sqrt(variance))}


def same_number(actual, expected, tolerance=1e-10):
    """Whether a stored number equals its recomputed value; ``None`` matches only ``None``."""
    if actual is None or expected is None:
        return actual is None and expected is None
    return type(actual) in (int, float) and abs(actual - expected) <= tolerance * max(1., abs(expected))


def validate_selection(selection, semantic, capability, cost_policy, *, decision_format=DECISION_FORMAT, chosen=None,
                       inventory=()):
    """Validate frozen objective/cap meanings without reinterpreting older receipts.

    ``inventory`` is the receipt's native candidate list: a v11 receipt's credit numbers
    are recomputed from the token beliefs at those prices.
    """
    # v11 sorts cells into tiers and retires the cap-unmet policy; older receipts keep theirs.
    tiered = decision_format == DECISION_FORMAT
    keys(selection, ["objective", "retry_policy", "constants", "sampling_policy", "cells", "exploration", "cap_met", "needs_review",
                     *(() if tiered else ("on_cap_unmet",))], where="decision.selection")
    if selection["objective"] != "cost-to-accepted-result" or selection["retry_policy"] != "same-cell-until-non-loud":
        raise HowHardError("decision.selection: unsupported objective/retry policy")
    legacy = decision_format in (LEGACY_DECISION_V7, LEGACY_DECISION_V8)
    validate_policy(selection["sampling_policy"], legacy=legacy,
                    count_aged=decision_format in (LEGACY_DECISION_V9, LEGACY_DECISION_V10))
    # v11 gives active time and tokens beliefs, and checks credit on its 80% point.
    amounts = decision_format == DECISION_FORMAT
    policy = selection["sampling_policy"]
    candidates = {(c["model"], c["effort"]): c for c in inventory}
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
             + ([] if legacy else ["admission_risk"]) + (["pi", "tier"] if tiered else [])
             + (["active_time", "token_beliefs", "credit_variance", "credit_p80_usd"] if amounts else []),
             where="selection cell")
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
        if amounts:
            for field in ("credit_variance", "credit_p80_usd"):
                if (row[field] is None) != (needed_credit is None):
                    raise HowHardError(f"decision.selection: {field} must be known exactly when needed_credit_usd is")
                if row[field] is not None:
                    number(row[field], "selection."+field, nonnegative=True)
            expected_ok = credit_fits(needed_credit, row["credit_p80_usd"], row["available_credit_usd"])
        else:
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
            intervals, _ = posterior_intervals(
                panels,
                weights,
                pair,
                semantic["outcome_observations"],
                policy,
                legacy_v9=(decision_format == LEGACY_DECISION_V9),
                epsilon=constants["epsilon"] if tiered else None,
            )
            for field in ("P_clean", "P_loud", "P_silent", "P_silent_accepted"):
                if any(abs(row[field][k] - intervals[field][k]) > 1e-9 for k in ("mean", "lower", "upper")):
                    raise HowHardError("decision.selection: posterior credible interval mismatch")
            risk = number(row["admission_risk"], "selection.admission_risk", nonnegative=True)
            if abs(risk - intervals["admission_risk"]) > 1e-9:
                raise HowHardError("decision.selection: accepted-result risk/cap mismatch")
            if tiered:
                pi = number(row["pi"], "selection.pi", nonnegative=True)
                if abs(pi - intervals["pi"]) > 1e-9:
                    raise HowHardError("decision.selection: pi does not match the frozen forecasts and outcome observations")
                if row["tier"] != admission_tier(pi, policy) or row["admissible"] != (row["tier"] == "trusted"):
                    raise HowHardError("decision.selection: tier/admissible does not match pi")
            elif row["admissible"] != (risk <= constants["epsilon"]):
                raise HowHardError("decision.selection: accepted-result risk/cap mismatch")
            loud, silent = row["P_loud"]["mean"], row["P_silent"]["mean"]
        if amounts:
            beliefs = cell_beliefs([(e["weight"], panels[e["key"]][pair]) for e in experts], pair,
                                   semantic["duration_observations"], policy)
            expected_hours = beliefs["active_time"]["expected_seconds"] / 3600
            candidate = candidates.get((row["model"], row["effort"]))
            if candidate is None:
                raise HowHardError("decision.selection: cell does not match native candidate inventory")
            expected_credit = run_credit(beliefs["token_beliefs"], candidate.get("rate_card"), candidate.get("usd_per_mtok"))
            for field, value in zip(("needed_credit_usd", "credit_variance", "credit_p80_usd"), expected_credit or (None,) * 3):
                if not same_number(row[field], value):
                    raise HowHardError(f"decision.selection: {field} does not match the token beliefs at the cell's prices")
        elif decision_format == LEGACY_DECISION_V7:
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
    exploration = selection["exploration"]
    if tiered:
        if semantic["sampling"]["enabled"] is not True:
            raise HowHardError("decision.selection: a sample is always drawn")
        pool, kind = admission_pool(selection["cells"], policy)
        if not isinstance(exploration, dict) or type(exploration.get("enabled")) is not bool:
            raise HowHardError("decision.selection: invalid exploration state")
        if exploration.get("pool") != kind:
            raise HowHardError("decision.selection: exploration.pool does not match the tiers and credit of the cells")
        picked = (chosen["model"], chosen["effort"]) if isinstance(chosen, dict) else None
        chosen_row = next((c for c in selection["cells"] if (c["model"], c["effort"]) == picked), None)
        if chosen_row is None or not chosen_row["credit_ok"]:
            raise HowHardError("decision.chosen: selected cell is not schedulable within credit_budget")
        if not any(c is chosen_row for c in pool):
            raise HowHardError("decision.selection: the chosen cell is outside the admission pool")
        cap_met = chosen_row["tier"] == "trusted"
        if any(type(selection[flag]) is not bool for flag in ("cap_met", "needs_review")) or (
                selection["cap_met"] != cap_met or selection["needs_review"] != (not cap_met)):
            raise HowHardError("decision.selection: inconsistent cap/review state")
        if (capability["widening_stop"] == "pickable") != any(c["tier"] != "excluded" for c in selection["cells"]):
            raise HowHardError("decision.selection: widening_stop does not match the tiers of the forecast cells")
        return selection
    cap_met = any(c["admissible"] and c.get("credit_ok", True) for c in selection["cells"])
    if semantic["sampling"]["enabled"] != cap_met:
        raise HowHardError("decision.selection: sampling and cap status disagree")
    if type(selection["cap_met"]) is not bool or selection["cap_met"] != cap_met or selection["needs_review"] != (not cap_met):
        raise HowHardError("decision.selection: inconsistent cap/review state")
    if type(exploration.get("enabled")) is not bool or (not cap_met and (exploration["enabled"] or exploration["reason"] != "disabled-cap-unmet")):
        raise HowHardError("decision.selection: exploration must be disabled when cap is unmet")
    return selection
