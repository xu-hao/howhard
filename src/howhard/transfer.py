"""Portable shared-term priors containing generic vocabulary and numbers only."""

import json
import os
from pathlib import Path
import tempfile
import numpy as np

from .config import EFFORTS
from .validation import HowHardError, keys, number, text

FORMAT = "howhard-prior-v1"
CONTEXT_FORMAT = "howhard-prior-v2"


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            name = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def shared(label):
    return label.split(".")[1] not in ("pi", "pi_task", "project")


def allowed_labels(vocab):
    families, efforts, tasks = (vocab[k] for k in ("families", "efforts", "task_types"))
    outcome = set()
    for kind in ("loud", "silent"):
        outcome.add(f"{kind}.a")
        outcome.update(f"{kind}.b.{m}" for m in families)
        outcome.update(f"{kind}.c.{e}" for e in efforts)
        outcome.update(f"{kind}.d.{m}.{e}" for m in families for e in efforts)
        outcome.update(f"{kind}.u.{t}" for t in tasks)
        outcome.update(f"{kind}.w.{t}.{m}" for t in tasks for m in families)
        outcome.update(f"{kind}.v.{t}.{e}" for t in tasks for e in efforts)
        for feature in vocab.get("features", []):
            outcome.add(f"{kind}.x.{feature}")
            outcome.update(f"{kind}.x_family.{m}.{feature}" for m in families)
            outcome.update(f"{kind}.x_effort.{e}.{feature}" for e in efforts)
    result = {"outcome": outcome}
    for block in ("tokens", "time"):
        result[block] = ({f"{block}.task.{t}" for t in tasks} |
                         {f"{block}.family.{m}" for m in families} |
                         {f"{block}.effort.{e}" for e in efforts})
        result[block].update(f"{block}.x.{feature}" for feature in vocab.get("features", []))
    return result


def validate_export(value, config):
    from .context import FORMAT as ENCODER, FEATURE_NAMES
    keys(value, ["format", "vocabularies", "blocks", "noise_sds"], ["context_encoder"], where="import")
    if value["format"] not in (FORMAT, CONTEXT_FORMAT):
        raise HowHardError("import.format: unsupported format version")
    contextual = value["format"] == CONTEXT_FORMAT
    if contextual and value.get("context_encoder") != ENCODER:
        raise HowHardError("import.context_encoder: unsupported or missing encoder")
    if not contextual and "context_encoder" in value:
        raise HowHardError("import.context_encoder: unexpected in a legacy prior")
    vocab = value["vocabularies"]
    keys(vocab, ["families", "efforts", "task_types", *(["features"] if contextual else [])], where="import.vocabularies")
    if contextual and vocab["features"] != list(FEATURE_NAMES):
        raise HowHardError("import.vocabularies.features: incompatible encoder features")
    known = {"families": config.prior["families"], "efforts": EFFORTS, "task_types": config.task_types, "features": FEATURE_NAMES}
    for key, names in vocab.items():
        if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names):
            raise HowHardError(f"import.vocabularies.{key}: expected names")
        if len(names) != len(set(names)):
            raise HowHardError(f"import.vocabularies.{key}: duplicate names")
        for name in names:
            text(name, f"import.vocabularies.{key}", label=True)
            if name not in known[key] and not (key == "task_types" and not config.explicit):
                raise HowHardError(f"import.vocabularies.{key}: unknown name {name!r}")
    allowed = allowed_labels(vocab)
    keys(value["blocks"], ["outcome", "tokens", "time"], where="import.blocks")
    for block, spec in value["blocks"].items():
        where = f"import.blocks.{block}"
        keys(spec, ["labels", "mean", "covariance"], where=where)
        labels = spec["labels"]
        if not isinstance(labels, list) or not labels or any(not isinstance(s, str) or s not in allowed[block] for s in labels) or len(set(labels)) != len(labels):
            raise HowHardError(f"{where}.labels: unknown, duplicate or missing label")
        n = len(labels)
        if not isinstance(spec["mean"], list) or len(spec["mean"]) != n:
            raise HowHardError(f"{where}.mean: wrong shape")
        if not isinstance(spec["covariance"], list) or len(spec["covariance"]) != n:
            raise HowHardError(f"{where}.covariance: wrong shape")
        for item in spec["mean"]:
            number(item, f"{where}.mean")
        for row in spec["covariance"]:
            if not isinstance(row, list) or len(row) != n:
                raise HowHardError(f"{where}.covariance: wrong shape")
            for item in row:
                number(item, f"{where}.covariance")
        covariance = np.asarray(spec["covariance"], dtype=float)
        try:
            if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12):
                raise np.linalg.LinAlgError
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise HowHardError(f"{where}.covariance: must be symmetric positive definite") from exc
    keys(value["noise_sds"], ["tokens", "time"], where="import.noise_sds")
    for block, sd in value["noise_sds"].items():
        number(sd, f"import.noise_sds.{block}", positive=True)
    return value


def read_export(path, config):
    try:
        value = json.loads(Path(path).read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise HowHardError(f"import: invalid JSON in {path}") from exc
    return validate_export(value, config)


def export_prior(posterior, config, path):
    inflation = config.data.get("transfer", config.prior["transfer"])["variance_inflation"]
    value = {"format": FORMAT, "vocabularies": {
        "families": list(config.prior["families"]), "efforts": list(config.prior["efforts"]),
        "task_types": sorted(posterior.prior.task_types)}, "blocks": {}, "noise_sds": posterior.prior.noise.copy()}
    if posterior.prior.has_context:
        from .context import FORMAT as ENCODER, FEATURE_NAMES
        value["format"] = CONTEXT_FORMAT
        value["context_encoder"] = ENCODER
        value["vocabularies"]["features"] = list(FEATURE_NAMES)
    for block in ("outcome", "tokens", "time"):
        source = getattr(posterior, block)
        indices = [i for i, label in enumerate(source.labels) if shared(label)]
        value["blocks"][block] = {"labels": [source.labels[i] for i in indices],
            "mean": source.mean[indices].tolist(),
            "covariance": (source.covariance[np.ix_(indices, indices)] * inflation).tolist()}
    atomic_write(path, (json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n").encode())
    return value


def import_prior(config, path):
    if "import" not in config.data["prior"]:
        raise HowHardError('prior.import: add [prior] import = "<path>" to the config')
    value = read_export(path, config)
    destination = config.path(config.data["prior"]["import"])
    atomic_write(destination, (json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n").encode())
    return destination
