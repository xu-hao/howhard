"""Labeled Gaussian distributions and stable linear algebra."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Gaussian:
    labels: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray
    statistics: dict = field(default_factory=dict)

    @property
    def mode(self):
        return self.mean

    def index(self, label):
        return self.labels.index(label)

    def project(self, design):
        return design @ self.mean, design @ self.covariance @ design.T


def inverse(matrix):
    value = np.linalg.solve(matrix, np.eye(len(matrix)))
    return (value + value.T) / 2


def draw(mean, covariance, rng, size):
    # Prediction matrices can be rank deficient; eigendecomposition handles PSD marginals.
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    factor = vectors * np.sqrt(np.maximum(values, 0))
    return mean + rng.standard_normal((size, len(mean))) @ factor.T


def extend(posterior, prior):
    """Add independent, unobserved terms without changing fitted existing terms."""
    result = Gaussian(prior.labels, prior.mean.copy(), prior.covariance.copy(), posterior.statistics.copy())
    old_index = {label: i for i, label in enumerate(posterior.labels)}
    pairs = [(i, old_index[label]) for i, label in enumerate(prior.labels) if label in old_index]
    new, old = zip(*pairs)
    result.mean[list(new)] = posterior.mean[list(old)]
    result.covariance[np.ix_(new, new)] = posterior.covariance[np.ix_(old, old)]
    return result
