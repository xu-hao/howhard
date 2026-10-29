"""Exact Newton MAP and joint Laplace covariance for weighted multinomial logit."""

import math
import numpy as np

from .gaussian import Gaussian, inverse
from .prior import design
from .validation import HowHardError
from .context import FEATURE_NAMES, row_context


def probabilities(eta):
    logits = np.concatenate((np.zeros((*eta.shape[:-1], 1)), eta), axis=-1)
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def objective(beta, mean, precision, x, counts):
    """Return negative log posterior, gradient and exact Hessian (also used by tests)."""
    eta = (x @ beta).reshape(-1, 2)
    p = probabilities(eta)
    n = counts.sum(axis=1)
    delta = beta - mean
    logits = np.column_stack((np.zeros(len(eta)), eta))
    normalizer = np.logaddexp.reduce(logits, axis=1)
    value = 0.5 * delta @ precision @ delta + np.sum(n * normalizer - (counts[:, 1:] * eta).sum(axis=1))
    gradient = precision @ delta + x.T @ (n[:, None] * p[:, 1:] - counts[:, 1:]).ravel()
    # Each observation contributes a 2x2 covariance between its logits.
    weights = n[:, None, None] * (np.eye(2)[None] * p[:, None, 1:] - p[:, 1:, None] * p[:, None, 1:])
    weighted_x = np.einsum("nij,njk->nik", weights, x.reshape(len(n), 2, -1)).reshape(x.shape)
    hessian = precision + x.T @ weighted_x
    return float(value), gradient, hessian


def fit(prior, rows, start=None):
    """Fit all rows, optionally starting at a labeled Gaussian or a mode array."""
    distribution = prior.outcome
    mean, covariance = distribution.mean, distribution.covariance
    kind = "warm" if start is not None else "full"
    if not rows:
        return Gaussian(distribution.labels, mean.copy(), covariance.copy(), {"kind": kind, "iterations": 0, "final_step": 0.0})
    # Canonical aggregation removes input order from every floating-point operation.
    grouped = {}
    for row in rows:
        context = tuple(row_context(row).values()) if prior.has_context else ()
        key = (row["task_type"], row["family"], row["resolved_effort"], row["project"], context, row["outcome"])
        grouped.setdefault(key, []).append(row["weight"])
    cells = sorted({key[:5] for key in grouped})
    positions = {cell: i for i, cell in enumerate(cells)}
    counts = np.zeros((len(cells), 3))
    for key in sorted(grouped):
        counts[positions[key[:5]], ("clean", "loud", "silent").index(key[5])] = math.fsum(sorted(grouped[key]))
    x = design(distribution, "outcome", [c[:4] for c in cells], [dict(zip(FEATURE_NAMES, c[4])) for c in cells])
    beta = mean.copy()
    if isinstance(start, Gaussian):
        lookup = dict(zip(start.labels, start.mean))
        beta = np.array([lookup.get(label, value) for label, value in zip(distribution.labels, mean)])
    elif start is not None:
        beta = np.array(start, dtype=float, copy=True)
    if beta.shape != mean.shape or not np.all(np.isfinite(beta)):
        raise HowHardError("fit.start: invalid mode shape or values")
    # Integrate inactive coefficients under their joint normal prior. This is an
    # exact marginalization, including imported correlations, not an online fit.
    # It avoids solving for every unused task/model/context interaction on tiny logs.
    original_mean, original_covariance = mean, covariance
    active = np.flatnonzero(np.any(x != 0, axis=0))
    inactive = np.setdiff1d(np.arange(len(mean)), active)
    covariance = covariance[np.ix_(active, active)]
    precision = inverse(covariance)
    mean, beta, x = mean[active], beta[active], x[:, active]
    independent = not np.any(original_covariance[np.ix_(inactive, active)])
    transform = None if independent else original_covariance[:, active] @ precision
    for iteration in range(1, 101):
        value, gradient, hessian = objective(beta, mean, precision, x, counts)
        step = np.linalg.solve(hessian, gradient)
        final_step = float(np.max(np.abs(step if independent else transform @ step)))
        if final_step < 1e-9:
            # Take the converged Newton step, then evaluate curvature at that point.
            beta -= step
            hessian = objective(beta, mean, precision, x, counts)[2]
            active_covariance = inverse(hessian)
            full_mean, full_covariance = original_mean.copy(), original_covariance.copy()
            if independent:
                full_mean[active] = beta
                full_covariance[np.ix_(active, active)] = active_covariance
            else:
                full_mean += transform @ (beta - mean)
                full_covariance += transform @ (active_covariance - covariance) @ transform.T
                full_covariance = (full_covariance + full_covariance.T) / 2
            return Gaussian(distribution.labels, full_mean, full_covariance, {"kind": kind, "iterations": iteration, "final_step": final_step})
        scale = 1.0
        while scale > 2 ** -40:
            candidate = beta - scale * step
            new_value = objective(candidate, mean, precision, x, counts)[0]
            if new_value <= value + 1e-12 * max(1.0, abs(value)):
                beta = candidate
                break
            scale /= 2
        else:
            raise HowHardError("outcome fit: Newton step halving failed")
    raise HowHardError("outcome fit: did not converge within 100 iterations")
