"""Validated atomic NPZ caches. A cache is always replaceable, never evidence."""

from dataclasses import dataclass
import hashlib
import io
import json
import zipfile
import numpy as np

from .gaussian import Gaussian
from .transfer import atomic_write

FORMAT = "howhard-cache-v2"


@dataclass
class Posterior:
    outcome: Gaussian
    tokens: Gaussian
    time: Gaussian
    prior: object

    @property
    def statistics(self):
        return self.outcome.statistics


def hashes(config, blobs, imported_bytes):
    def digest(parts):
        h = hashlib.sha256()
        for part in parts:
            h.update(len(part).to_bytes(8, "big"))
            h.update(part)
        return h.hexdigest()
    # Configuration affects weights, class assignments and project identities too.
    from .config import packaged
    prior_hash = digest([FORMAT.encode(), packaged("context.toml"), config.source_bytes, config.prior_bytes,
                         json.dumps(config.data, sort_keys=True).encode(), imported_bytes])
    data_hash = digest([blob for _, _, blob in blobs])
    return prior_hash, data_hash


def save(path, posterior, prior_hash, data_hash):
    values = {"format": np.array(FORMAT), "prior_hash": np.array(prior_hash), "data_hash": np.array(data_hash),
              "statistics": np.array(json.dumps(posterior.statistics))}
    for block in ("outcome", "tokens", "time"):
        source = getattr(posterior, block)
        values[f"{block}_labels"] = np.asarray(source.labels)
        values[f"{block}_mean"] = source.mean
        values[f"{block}_covariance"] = source.covariance
    buffer = io.BytesIO()
    np.savez(buffer, **values)
    atomic_write(path, buffer.getvalue())


def load(path, prior):
    try:
        with np.load(path, allow_pickle=False) as data:
            if data["format"].item() != FORMAT:
                return None
            stats = json.loads(data["statistics"].item())
            if not isinstance(stats, dict) or stats.get("kind") not in ("full", "warm") or type(stats.get("iterations")) is not int:
                return None
            blocks = {}
            for block in ("outcome", "tokens", "time"):
                labels = data[f"{block}_labels"]
                mean, cov = data[f"{block}_mean"], data[f"{block}_covariance"]
                if labels.ndim != 1 or labels.dtype.kind != "U":
                    return None
                labels = tuple(labels.tolist())
                if len(set(labels)) != len(labels) or not set(labels) <= set(getattr(prior, block).labels):
                    return None
                if mean.shape != (len(labels),) or cov.shape != (len(labels), len(labels)):
                    return None
                if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T, rtol=0, atol=1e-12):
                    return None
                np.linalg.cholesky(cov)
                blocks[block] = Gaussian(labels, mean.copy(), cov.copy(), stats.copy() if block == "outcome" else {})
            return Posterior(**blocks, prior=prior), data["prior_hash"].item(), data["data_hash"].item()
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile, np.linalg.LinAlgError):
        return None
