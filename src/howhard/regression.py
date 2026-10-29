"""Exact weighted conjugate updates for log tokens and log seconds."""

import math
import numpy as np

from .gaussian import Gaussian, inverse
from .prior import design
from .context import row_context


def conjugate(prior, x, y, weights, noise_sd):
    if not len(y):
        return Gaussian(prior.labels, prior.mean.copy(), prior.covariance.copy())
    precision = inverse(prior.covariance)
    weighted = np.asarray(weights) / noise_sd ** 2
    post_precision = precision + x.T @ (weighted[:, None] * x)
    covariance = inverse(post_precision)
    mean = np.linalg.solve(post_precision, precision @ prior.mean + x.T @ (weighted * y))
    return Gaussian(prior.labels, mean, covariance)


def fit_regression(prior, rows, block):
    field = "tokens" if block == "tokens" else "seconds"
    selected = sorted((r for r in rows if r[field] is not None),
                      key=lambda r: (r["task_type"], r["family"], r["resolved_effort"], r["project"], tuple(row_context(r).values()), r[field], r["weight"]))
    distribution = getattr(prior, block)
    cells = [(r["task_type"], r["family"], r["resolved_effort"], r["project"]) for r in selected]
    # Zero is a valid recorded measurement. A zero run enters at log(1), documented
    # as the finite one-token / one-second measurement floor.
    y = np.array([math.log(max(1, r[field])) for r in selected])
    return conjugate(distribution, design(distribution, block, cells, [row_context(r) for r in selected]), y,
                     [r["weight"] for r in selected], prior.noise[block])
