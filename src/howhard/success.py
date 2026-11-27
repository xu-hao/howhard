"""Accepted-result retry costs, per-cell outcome beliefs on the log-odds scale and amount beliefs.

A cell holds two independent beliefs, one for its loud rate and one for its accepted-defect
share, each on one fixed grid of log-odds. The Dirichlet outcome posterior of decision
formats v9 and v10 stays below as a frozen reader for those receipts.
"""

from copy import deepcopy
from functools import lru_cache
import math
import sys

import numpy as np

from .context import DECISION_FORMAT, LEGACY_DECISION_V7, LEGACY_DECISION_V8, LEGACY_DECISION_V9, LEGACY_DECISION_V10
from .llm import SETTINGS
from .validation import HowHardError, keys, number

DEFAULTS = {"uncertainty_multiplier": 1., "admission_quantile": .8, "trial_probability": .2,
            "all_excluded_factor": .5, "half_life_days": 14., "comparison_cap": 256,
            "model_worth": 5., "task_worth_amount": 1., "duration_miss": 1.3, "scatter_share": .5,
            "task_worth_silent": 32., "task_worth_loud": 16.,
            "effort_discount": .5, "model_discount": .25, "agent_discount": .25}
# How much a run by a related cell counts as outcome evidence, as a fraction of a run by the
# judged cell itself: the same family at another effort, another family of the same agent,
# and another agent.
DISCOUNTS = ("effort_discount", "model_discount", "agent_discount")
# The Dirichlet posterior's three constants. Outcome beliefs on the log-odds scale do not read
# them: a configuration that still sets one loads with a notice, and only v7-v10 receipts store them.
RETIRED = ("prior_strength", "min_concentration", "max_concentration")
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
# The frozen Dirichlet posterior of v9 and v10 receipts.
POSTERIOR_FIELDS = ("prior", "prior_concentration", "counts", "evidence_weight", "alpha")
# Outcome beliefs: one grid of log-odds for every belief, x_j = -16 + 0.01 j for j = 0 .. 2500.
GRID_X = np.arange(2501) * .01 - 16.
GRID_P = 1. / (1. + np.exp(-GRID_X))        # logistic(x_j): the rate each grid point stands for.
GRID_LOG_YES = -np.log1p(np.exp(-GRID_X))   # ln logistic(x_j)
GRID_LOG_NO = -np.log1p(np.exp(GRID_X))     # ln (1 - logistic(x_j))
GRID_ODDS = np.exp(GRID_X)                  # logistic(x_j) / (1 - logistic(x_j))
for _grid in (GRID_X, GRID_P, GRID_LOG_YES, GRID_LOG_NO, GRID_ODDS):
    _grid.setflags(write=False)
FLAT_SPREAD = 1.869                 # The spread of a 95% range of [0.025, 0.975]: the flat belief.
RANGE_HOLD = (.0001, .9999)         # The ends of a stated 95% range are held in this interval.
PREDICTION_HOLD = (.0001, .9998)    # The grid's top value is 0.99988, so every held prediction can be met.
CENTRE_BRACKET, CENTRE_HALVINGS = (-25., 25.), 60
# Each rate's most a record on other tasks is worth for one task, and the fields it adds to a belief.
RATES = {"loud": ("task_worth_loud", ("attempts", "odds")), "silent": ("task_worth_silent", ("pi",))}
BELIEF_FIELDS = ("prediction", "stated_spread", "n_start", "spread", "centre", "counts", "mean", "lower", "upper")
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

    Only the current policy carries the time-based evidence constants, the two tier constants,
    the three worths of the outcome beliefs and the three relatedness discounts; only the
    older ones carry the Dirichlet constants.
    """
    if not legacy and isinstance(policy, dict) and "explore_within" in policy:
        raise HowHardError("sampling.explore_within: removed; Thompson eligibility uses posterior overlap. "
                           "Set explore_within on a risk class to cap the mean-cost exploration band explicitly")
    keys(policy, list(LEGACY_DEFAULTS if legacy else COUNT_AGED_DEFAULTS if count_aged else DEFAULTS), where="sampling")
    for key, value in policy.items():
        number(value, "sampling."+key, nonnegative=True)
    dirichlet = legacy or count_aged
    if dirichlet and (not 1 <= policy["min_concentration"] <= policy["max_concentration"] or policy["prior_strength"] <= 0):
        raise HowHardError("sampling: require 1 <= min_concentration <= max_concentration and positive prior_strength")
    if policy["uncertainty_multiplier"] <= 0:
        raise HowHardError("sampling: uncertainty_multiplier must be positive")
    if legacy:
        if policy["explore_within"] > 1:
            raise HowHardError("sampling: explore_within must be in [0,1]")
        return policy
    if dirichlet and policy["min_concentration"] > policy["prior_strength"]:
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
    for key in ("task_worth_silent", "task_worth_loud"):
        if policy[key] <= 0:
            raise HowHardError(f"sampling.{key}: expected a positive number of runs")
    for key in DISCOUNTS:
        if policy[key] > 1:
            raise HowHardError(f"sampling.{key}: expected a discount in [0, 1]")
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
    """Frozen reader of v9 and v10 receipts: the bounded forecast pseudo-count kappa_0."""
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


def same_cell_counts(pair, outcome_observations):
    """Frozen reader of v9 and v10 receipts: only the judged cell's own observations count."""
    from .semantic import OUTCOMES
    counts = [[], [], []]
    for obs in outcome_observations:
        if (obs["family"], obs["effort"]) == pair and obs["weight"] > 0:
            counts[OUTCOMES.index(obs["outcome"])].append(obs["weight"])
    return [math.fsum(c) for c in counts]


def _counted(cell, outcome_observations, policy):
    """``(observation, relation, discount, weight after the limit)`` for each run counted for ``cell``.

    Each stored observation carries its run's own cell and agent and the weight
    ``m * sim * a``. For the judged cell ``c = (family, effort, agent)``::

        w_i  = m_i * sim_i * a_i * delta(c_i, c)
        w_i <- w_i / max(1, W_t)        W_t: the total w of the runs of ticket t counted for c

    ``delta`` is 1 for the cell's own runs and the policy's discount for a related cell's.
    The limit follows the discount, so it is taken per judged cell. A run whose discount is
    0 is not counted.
    """
    from . import evidence
    counted = []
    for obs in outcome_observations:
        related = evidence.relation((obs["family"], obs["effort"], obs["agent"]), cell)
        delta = evidence.discount(related, policy)
        if delta > 0:
            counted.append((obs, related, delta, obs["weight"] * delta))
    shares = evidence.ticket_shares([(obs["ticket"], value) for obs, _, _, value in counted])
    return [(obs, related, delta, value * share) for (obs, related, delta, value), share in zip(counted, shares)]


def counted_observations(cell, outcome_observations, policy):
    """Each run a judged cell counts: ``id``, ``relation``, ``discount`` and ``weight``.

    ``cell`` is the judged ``(family, effort, agent)`` and ``weight`` the run's weight after the
    per-ticket limit. These rows are computed on demand and never stored: a receipt holds the
    runs, the cell's label and the discounts, and stores only the sums of ``counted_totals``.
    """
    return [{"id": obs["id"], "relation": related, "discount": delta, "weight": value}
            for obs, related, delta, value in _counted(cell, outcome_observations, policy)]


def _totals(cell, outcome_observations, policy):
    """The counted weights of one judged cell by outcome, in the order clean, loud, silent."""
    from .semantic import OUTCOMES
    counts = [[], [], []]
    for obs, _, _, value in _counted(cell, outcome_observations, policy):
        if value > 0:
            counts[OUTCOMES.index(obs["outcome"])].append(value)
    return counts


def outcome_counts(cell, outcome_observations, policy):
    """``[n_c, n_l, n_s]``: the counted clean, loud and silent weight of one judged cell."""
    return [math.fsum(c) for c in _totals(cell, outcome_observations, policy)]


def counted_totals(cell, outcome_observations, policy):
    """What a forecast cell stores of its counted runs: the three sums and how many runs contribute.

    ``n_clean``, ``n_loud`` and ``n_silent`` are the counted weights after the discount and
    the per-ticket limit; ``runs`` is the number of runs with a weight above 0.
    """
    counts = _totals(cell, outcome_observations, policy)
    clean, loud, silent = (math.fsum(c) for c in counts)
    return {"n_clean": clean, "n_loud": loud, "n_silent": silent, "runs": sum(len(c) for c in counts)}


def logit(p):
    return math.log(p) - math.log1p(-p)


def rate_guesses(row):
    """One expert's guess for each rate on the log-odds scale, with its stated spread.

    The loud rate is ``loud``; the accepted-defect share is ``silent / (clean + silent)``, so
    ``silent_ci`` is divided by ``clean + silent``. The spread is 0 when no range is stated.
    """
    clean, loud, silent = (float(p) for p in forecast_probabilities(row))
    accepted = clean + silent

    def spread(interval, scale):
        if interval is None:
            return 0.
        lo, hi = (min(RANGE_HOLD[1], max(RANGE_HOLD[0], float(end) / scale)) for end in interval)
        return max(0., (logit(hi) - logit(lo)) / CI_Z_WIDTH)
    return {"loud": (logit(loud), spread(row.get("loud_ci"), 1.)),
            "silent": (logit(silent / accepted), spread(row.get("silent_ci"), accepted))}


def merged_forecast(panels, experts, pair):
    """One forecast per rate for a cell: the experts' guesses merged on the log-odds scale.

    ``m = sum w_k m_k`` and ``s_stated^2 = sum w_k (s_k^2 + (m_k - m)^2)``, so experts that
    disagree make the forecast less certain. The prediction is ``logistic(m)``, held.
    """
    guesses = [(e["weight"], rate_guesses(panels[e["key"]][pair])) for e in experts]
    merged = {}
    for rate in RATES:
        centre = math.fsum(w * guess[rate][0] for w, guess in guesses)
        variance = math.fsum(w * (guess[rate][1] ** 2 + (guess[rate][0] - centre) ** 2) for w, guess in guesses)
        merged[rate] = {"prediction": min(PREDICTION_HOLD[1], max(PREDICTION_HOLD[0], _sigmoid(centre))),
                        "stated_spread": math.sqrt(max(0., variance))}
    return merged


def start_worth(rate, policy):
    """``N_start``: how many runs the prediction is worth for this rate."""
    return 1. / (1. / policy["model_worth"] + 1. / policy[RATES[rate][0]])


def start_spread(prediction, stated_spread, n_start):
    """The spread at the start: a stated range can widen it, never narrow it, up to the flat belief."""
    one_run = 1. / math.sqrt(prediction * (1. - prediction))
    return min(FLAT_SPREAD, max(stated_spread, one_run / math.sqrt(n_start)))


@lru_cache(maxsize=256)
def _grid_belief(prediction, spread, n_yes, n_no):
    """Centre of the start, and the belief after the counted runs with its running total, on the grid.

    The start is a bell curve whose grid average equals the prediction. Its centre is found by
    halving the bracket, keeping the upper half when the average at the midpoint is too low.
    """
    def bell(centre):
        log_q = -(GRID_X - centre) ** 2 / (2. * spread ** 2)
        return log_q - log_q.max()
    lo, hi = CENTRE_BRACKET
    for _ in range(CENTRE_HALVINGS):
        mid = (lo + hi) / 2.
        q = np.exp(bell(mid))
        if float(q @ GRID_P) / float(q.sum()) < prediction:
            lo = mid
        else:
            hi = mid
    centre = (lo + hi) / 2.
    log_b = bell(centre) + n_yes * GRID_LOG_YES + n_no * GRID_LOG_NO
    b = np.exp(log_b - log_b.max())
    b /= b.sum()
    total = np.cumsum(b)
    for array in (b, total):
        array.setflags(write=False)
    return centre, b, total


def _belief_grid(belief):
    return _grid_belief(belief["prediction"], belief["spread"], *belief["counts"])


def belief_point(belief, q):
    """The rate at the first grid point whose running total of the belief reaches ``q``."""
    total = _belief_grid(belief)[2]
    return float(GRID_P[min(len(GRID_P) - 1, int(np.searchsorted(total, q, side="left")))])


def belief_variance(belief):
    b = _belief_grid(belief)[1]
    return max(0., float(b @ GRID_P ** 2) - float(b @ GRID_P) ** 2)


def rate_belief(rate, forecast, counts, policy, *, epsilon=None):
    """One rate's belief: the start from its merged forecast, reshaped by the counted runs.

    ``counts`` is ``[n_yes, n_no]``. The start counts as ``N_start`` runs and each counted run
    as its weight, so the forecast never fades out. With ``epsilon``, the accepted-defect share
    also carries ``pi``, the share of the belief at or below it.
    """
    n_start = start_worth(rate, policy)
    belief = {"prediction": forecast["prediction"], "stated_spread": forecast["stated_spread"], "n_start": n_start,
              "spread": start_spread(forecast["prediction"], forecast["stated_spread"], n_start),
              "counts": [float(n) for n in counts]}
    centre, b, total = _belief_grid(belief)
    mean = float(b @ GRID_P)
    lower, upper = (belief_point(belief, q) for q in INTERVAL)
    belief.update(centre=centre, mean=mean, lower=min(mean, lower), upper=max(mean, upper))
    if rate == "loud":
        # Expected attempts E[1 / (1 - theta_l)] and expected loud failures per accepted result.
        belief.update(attempts=float(b @ (1. + GRID_ODDS)), odds=float(b @ GRID_ODDS))
    elif epsilon is not None:
        within = int(np.searchsorted(GRID_P, epsilon, side="right"))
        belief["pi"] = min(1., float(total[within - 1])) if within else 0.
    return belief


def outcome_beliefs(panels, experts, pair, outcome_observations, policy, *, agent=None, epsilon=None):
    """The cell's two independent beliefs: ``loud`` for theta_l and ``silent`` for S.

    The accepted-defect share counts silent against clean runs; the loud rate counts loud
    against clean and silent ones. The cell's own runs count in full and a related cell's
    at its discount; ``agent`` is the judged cell's label, ``None`` for the unnamed agent.
    """
    forecast = merged_forecast(panels, experts, pair)
    clean, loud, silent = outcome_counts((*pair, agent), outcome_observations, policy)
    return {"loud": rate_belief("loud", forecast["loud"], (loud, clean + silent), policy),
            "silent": rate_belief("silent", forecast["silent"], (silent, clean), policy, epsilon=epsilon)}


def belief_probabilities(beliefs):
    """``[P_clean, P_loud, P_silent]`` at the averages of the two beliefs."""
    loud, share = beliefs["loud"]["mean"], beliefs["silent"]["mean"]
    return [(1. - loud) * (1. - share), loud, (1. - loud) * share]


def belief_intervals(beliefs, policy):
    """What a selection cell reads from its beliefs.

    ``P_loud`` and ``P_silent_accepted`` are the two beliefs with their 95% points. ``P_clean``
    and ``P_silent`` carry a mean only, because neither has a belief of its own.
    """
    clean, loud, silent = belief_probabilities(beliefs)
    interval = lambda belief: {k: belief[k] for k in ("mean", "lower", "upper")}
    return {"P_clean": {"mean": clean}, "P_loud": interval(beliefs["loud"]), "P_silent": {"mean": silent},
            "P_silent_accepted": interval(beliefs["silent"]),
            "admission_risk": belief_point(beliefs["silent"], policy["admission_quantile"]),
            "pi": beliefs["silent"]["pi"]}


def outcome_posterior(row, pair, outcome_observations, policy, *, legacy_v9=False):
    """Frozen reader of v9 and v10 receipts: the Dirichlet posterior, with v10's cold-start decay."""
    p0 = forecast_probabilities(row)
    kappa0 = prior_concentration(row, policy, p0)
    counts = same_cell_counts(pair, outcome_observations)
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


def posterior_intervals(panels, weights, pair, outcome_observations, policy, *, legacy_v9=False):
    """Frozen reader of v9 and v10 receipts: expert-mixture Dirichlet marginals and accepted risk."""
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


def calibration(result, state, expert, cell):
    """A cell's frozen draw inputs: its two outcome beliefs and the first-order spread of its cost.

    Every input is the cell's own belief, already merged over the experts: the two rates
    from one merged forecast each plus the runs counted for this cell, and active time and
    run cost as their averages ``T`` and ``C_run``. Experts that disagree widen those
    beliefs, so the spread has no separate between-expert term. The time term of the
    spread is ``E[T]^2 * (exp(s'^2) - 1)``, and run cost is exact.
    """
    from .semantic import panel_lookup
    pair = cell["family"], cell["effort"]
    policy = result["sampling_policy"]
    constants = result["constants"]
    panels = {key: panel_lookup(panel) for key, panel in state["worlds"].items()}
    beliefs = outcome_beliefs(panels, state["beliefs"]["experts"], pair, state.get("outcome_observations", []), policy,
                              agent=cell["agent"], epsilon=constants["epsilon"])
    probabilities = belief_probabilities(beliefs)
    loud, silent = probabilities[1], probabilities[2]
    summary = panel_lookup(result["cells"])[pair]
    # Delta-method spread of the same-cell retry objective: the two beliefs are independent, so
    # their variances add, plus the log-normal variance of active time.
    run = summary["session_cost_usd"] if cell["money_included"] else 0.
    hours = summary["T"]
    expected = cost(run, hours, loud, silent, constants)
    d = denominator(loud)
    share = beliefs["silent"]["mean"]
    rates_variance = (((constants["C_loud"] + expected - constants["C_silent"] * share) / d) ** 2
                      * belief_variance(beliefs["loud"])
                      + constants["C_silent"] ** 2 * belief_variance(beliefs["silent"]))
    hours_variance = min(sys.float_info.max, hours ** 2 * math.expm1(min(summary["active_time"]["spread"] ** 2, EXP_CAP)))
    variance = rates_variance + ((constants["lambda_per_hour"] / d) ** 2) * hours_variance
    return {"family": cell["family"], "effort": cell["effort"], "expert": expert["key"],
            "outcome_beliefs": beliefs, "probabilities": probabilities,
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
             + ([] if legacy else ["admission_risk"]) + (["pi", "tier", "outcome_counts"] if tiered else [])
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
            # v11 reads P_loud and P_silent_accepted from a belief each; the other two carry a mean only.
            ranged = not tiered or field in ("P_loud", "P_silent_accepted")
            keys(row[field], ["mean", "lower", "upper"] if ranged else ["mean"], where="selection."+field)
            for value in row[field].values():
                number(value, "selection probability", nonnegative=True)
            if ranged and not 0 <= row[field]["lower"] <= row[field]["mean"] <= row[field]["upper"] <= 1+1e-10:
                raise HowHardError("decision.selection: inconsistent probability envelope")
            if not row[field]["mean"] <= 1+1e-10:
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
            if tiered:
                # Recompute the cell's two beliefs from the frozen forecasts and counted runs. The
                # cell's agent is the label stored with its native candidate.
                if (row["model"], row["effort"]) not in candidates:
                    raise HowHardError("decision.selection: cell does not match native candidate inventory")
                intervals = belief_intervals(outcome_beliefs(
                    panels, experts, pair, semantic["outcome_observations"], policy,
                    agent=candidates[row["model"], row["effort"]].get("agent"), epsilon=constants["epsilon"]), policy)
            else:
                intervals, _ = posterior_intervals(
                    panels,
                    {e["key"]: e["weight"] for e in experts},
                    pair,
                    semantic["outcome_observations"],
                    policy,
                    legacy_v9=(decision_format == LEGACY_DECISION_V9),
                )
            for field in ("P_clean", "P_loud", "P_silent", "P_silent_accepted"):
                if any(abs(row[field][k] - intervals[field][k]) > 1e-9 for k in row[field]):
                    raise HowHardError("decision.selection: posterior credible interval mismatch")
            risk = number(row["admission_risk"], "selection.admission_risk", nonnegative=True)
            if abs(risk - intervals["admission_risk"]) > 1e-9:
                raise HowHardError("decision.selection: accepted-result risk/cap mismatch")
            if tiered:
                pi = number(row["pi"], "selection.pi", nonnegative=True)
                if abs(pi - intervals["pi"]) > 1e-9:
                    raise HowHardError("decision.selection: pi does not match the frozen forecasts and outcome observations")
                drawn = next((c for c in semantic["sampling"]["cells"] if (c["family"], c["effort"]) == pair), None)
                if drawn is None or abs(drawn["outcome_beliefs"]["silent"]["pi"] - intervals["pi"]) > 1e-9:
                    raise HowHardError("decision.selection: pi of the frozen draw does not match the frozen forecasts and outcome observations")
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
