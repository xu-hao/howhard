"""Expand compact TOML assumptions into labeled joint normal priors."""

from dataclasses import dataclass
import numpy as np

from .gaussian import Gaussian
from .context import FEATURE_NAMES, SETTINGS


@dataclass
class Prior:
    outcome: Gaussian
    tokens: Gaussian
    time: Gaussian
    noise: dict
    config: object
    task_types: dict
    projects: tuple
    has_context: bool = False


def build_prior(config, rows=(), *, extra_types=None, imported=None, contextual=None):
    p = config.prior
    families, efforts = config.families, config.efforts
    tasks = {**config.task_types, **(extra_types or {})}
    for row in rows:
        tasks.setdefault(row["task_type"], "standard")
    tasks = dict(sorted(tasks.items()))
    projects = tuple(sorted({config.data["project"], *(log["project"] for log in config.data["logs"])}))
    if contextual is None:
        contextual = any("context_encoder" in row for row in rows) or bool(imported and "context_encoder" in imported)
    blocks = {}
    for block in ("outcome", "tokens", "time"):
        labels, means, variances = [], [], []

        def add(label, spec, mean=None):
            labels.append(label)
            means.append(spec["mean"] if mean is None else mean)
            variances.append(spec["sd"] ** 2)

        if block == "outcome":
            for kind in ("loud", "silent"):
                specs = p["outcome"][kind]
                add(f"{kind}.a", specs["a"])
                for i, m in enumerate(families):
                    add(f"{kind}.b.{m}", specs["b"], specs["b"]["means"][i] if i < len(specs["b"]["means"]) else 0.0)
                for i, e in enumerate(efforts):
                    add(f"{kind}.c.{e}", specs["c"], specs["c"]["means"][i] if i < len(specs["c"]["means"]) else 0.0)
                for m in families:
                    for e in efforts:
                        add(f"{kind}.d.{m}.{e}", specs["d"])
                for t, cls in tasks.items():
                    add(f"{kind}.u.{t}", specs["u"], specs["u"]["means"][cls])
                    for m in families:
                        add(f"{kind}.w.{t}.{m}", specs["w"])
                    for e in efforts:
                        add(f"{kind}.v.{t}.{e}", specs["v"])
                for project in projects:
                    add(f"{kind}.pi.{project}", specs["pi"])
                    for t in tasks:
                        add(f"{kind}.pi_task.{project}.{t}", specs["pi_task"])
        else:
            specs = p[block]
            for t, cls in tasks.items():
                add(f"{block}.task.{t}", specs["task"], specs["task"]["means"][cls])
            for i, m in enumerate(families):
                add(f"{block}.family.{m}", specs["family"], specs["family"]["means"][i] if i < len(specs["family"]["means"]) else 0.0)
            for i, e in enumerate(efforts):
                add(f"{block}.effort.{e}", specs["effort"], specs["effort"]["means"][i] if i < len(specs["effort"]["means"]) else 0.0)
            for project in projects:
                add(f"{block}.project.{project}", specs["project"])
        if contextual:
            if block == "outcome":
                for kind in ("loud", "silent"):
                    for feature in FEATURE_NAMES:
                        add(f"{kind}.x.{feature}", SETTINGS["priors"]["silent" if kind == "silent" else "outcome"])
                    for family in families:
                        for feature in FEATURE_NAMES:
                            add(f"{kind}.x_family.{family}.{feature}", SETTINGS["priors"]["silent_interaction" if kind == "silent" else "interaction"])
                    for effort in efforts:
                        for feature in FEATURE_NAMES:
                            add(f"{kind}.x_effort.{effort}.{feature}", SETTINGS["priors"]["silent_interaction" if kind == "silent" else "interaction"])
            else:
                for feature in FEATURE_NAMES:
                    add(f"{block}.x.{feature}", SETTINGS["priors"][block])
        distribution = Gaussian(tuple(labels), np.array(means, dtype=float), np.diag(variances))
        if imported:
            source = imported["blocks"][block]
            positions = {label: i for i, label in enumerate(labels)}
            pairs = [(positions[label], i) for i, label in enumerate(source["labels"]) if label in positions]
            if pairs:
                dest, src = zip(*pairs)
                distribution.mean[list(dest)] = np.asarray(source["mean"])[list(src)]
                distribution.covariance[np.ix_(dest, dest)] = np.asarray(source["covariance"])[np.ix_(src, src)]
        blocks[block] = distribution
    noise = imported["noise_sds"].copy() if imported else {b: p[b]["noise_sd"] for b in ("tokens", "time")}
    return Prior(**blocks, noise=noise, config=config, task_types=tasks, projects=projects, has_context=contextual)


def cell_labels(block, task, family, effort, project):
    if block == "outcome":
        return [[f"{k}.a", f"{k}.b.{family}", f"{k}.c.{effort}", f"{k}.d.{family}.{effort}",
                 f"{k}.u.{task}", f"{k}.w.{task}.{family}", f"{k}.v.{task}.{effort}",
                 f"{k}.pi.{project}", f"{k}.pi_task.{project}.{task}"] for k in ("loud", "silent")]
    return [[f"{block}.task.{task}", f"{block}.family.{family}", f"{block}.effort.{effort}", f"{block}.project.{project}"]]


def design(distribution, block, cells, contexts=None):
    lookup = {label: i for i, label in enumerate(distribution.labels)}
    matrix = np.zeros((len(cells) * (2 if block == "outcome" else 1), len(lookup)))
    for i, cell in enumerate(cells):
        for k, labels in enumerate(cell_labels(block, *cell)):
            matrix[i * (2 if block == "outcome" else 1) + k, [lookup[label] for label in labels]] = 1
        if contexts is not None:
            for feature, value in contexts[i].items():
                if not value:
                    continue
                _, family, effort, _ = cell
                for k, kind in enumerate(("loud", "silent") if block == "outcome" else (block,)):
                    labels = [f"{kind}.x.{feature}"]
                    if block == "outcome":
                        labels += [f"{kind}.x_family.{family}.{feature}", f"{kind}.x_effort.{effort}.{feature}"]
                    for label in labels:
                        if label in lookup:
                            matrix[i * (2 if block == "outcome" else 1) + k, lookup[label]] = value
    return matrix
