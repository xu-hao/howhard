"""Deterministic posterior tables and seeded Thompson selection."""

from copy import deepcopy
import secrets
import numpy as np

from .gaussian import draw, extend
from .outcomes import probabilities
from .prior import build_prior, design
from .validation import HowHardError, text
from .context import encode

SUMMARY_SEED = 20260501
SUMMARY_DRAWS = 4000


def prepare(posterior, config, task, risk=None, context=None):
    text(task, "task_type", label=True)
    cls = risk or config.task_types.get(task)
    if cls is None:
        if config.explicit:
            raise HowHardError(f"task_type {task!r} is unknown; provide --risk")
        cls = "standard"
    if cls not in config.data["classes"]:
        raise HowHardError(f"risk: unknown class {cls!r}")
    # A risk override changes loss constants. Existing task effects retain their
    # learned distribution; only a previously unknown type needs new prior terms.
    distributions = {b: getattr(posterior, b) for b in ("outcome", "tokens", "time")}
    if task not in posterior.prior.task_types or (context is not None and not posterior.prior.has_context):
        prior = build_prior(config, extra_types={**posterior.prior.task_types, task: cls}, contextual=posterior.prior.has_context or context is not None)
        distributions = {b: extend(distributions[b], getattr(prior, b)) for b in distributions}
    cells = []
    for family, family_spec in sorted(config.data["families"].items(), key=lambda pair: (pair[1]["tier"], pair[0])):
        if not family_spec["selectable"]:
            continue
        model = family_spec["current"]
        for effort in config.prior["efforts"]:
            if effort in config.data["models"][model]["efforts"]:
                cells.append(dict(model=model, family=family, effort=effort, price=config.price(model)))
    if not cells:
        raise HowHardError("families: no selectable cells")
    inputs = [(task, c["family"], c["effort"], config.data["project"]) for c in cells]
    features = encode(context["task_text"], context["project_context"]) if context is not None else {}
    marginals = {b: distributions[b].project(design(distributions[b], b, inputs, [features] * len(inputs))) for b in distributions}
    return cls, cells, marginals


def components(marginals, cells, noise, constants, rng, size):
    outcome = draw(*marginals["outcome"], rng, size).reshape(size, len(cells), 2)
    p = probabilities(outcome)
    tokens = np.exp(draw(*marginals["tokens"], rng, size) + noise["tokens"] ** 2 / 2)
    hours = np.exp(draw(*marginals["time"], rng, size) + noise["time"] ** 2 / 2) / 3600
    run = tokens / 1e6 * np.array([c["price"] for c in cells])
    parts = {"run": run, "loud": p[:, :, 1] * constants["C_loud"],
             "silent": p[:, :, 2] * constants["C_silent"], "time": hours * constants["lambda_per_hour"]}
    return p, run, hours, sum(parts.values()), parts


def summarize(posterior, config, task, risk=None, context=None):
    cls, cells, marginals = prepare(posterior, config, task, risk, context)
    constants = config.data["classes"][cls].copy()
    p, run, hours, j, _ = components(marginals, cells, posterior.prior.noise, constants, np.random.default_rng(SUMMARY_SEED), SUMMARY_DRAWS)
    table = []
    for i, cell in enumerate(cells):
        row = {k: v for k, v in cell.items() if k != "price"}
        for k, name in enumerate(("P_clean", "P_loud", "P_silent")):
            row[name] = {"mean": float(p[:, i, k].mean()), "lower": float(np.quantile(p[:, i, k], .025)), "upper": float(np.quantile(p[:, i, k], .975))}
        row.update(C_run=float(run[:, i].mean()), T=float(hours[:, i].mean()), J=float(j[:, i].mean()),
                   admissible=row["P_silent"]["upper"] <= constants["epsilon"])
        table.append(row)
    table.sort(key=lambda row: (row["C_run"], row["family"], row["effort"]))
    result = {"task_type": task, "risk_class": cls, "constants": constants, "cells": table,
              "fit": posterior.statistics.copy()}
    if context is not None:
        result["context"] = deepcopy(context)
    return result, cells, marginals


def choose(prepared, posterior, config, seed=None, candidates=None, sample_parts=None):
    result, cells, marginals = prepared
    result = deepcopy(result)
    if seed is None:
        seed = secrets.randbits(64)
    if type(seed) is not int or seed < 0:
        raise HowHardError("seed: expected a non-negative integer")
    if sample_parts is None:
        _, _, _, j, parts = components(marginals, cells, posterior.prior.noise, result["constants"], np.random.default_rng(seed), 1)
    else:
        parts = sample_parts
        j = sum(parts.values())
    indices = {(cell["family"], cell["effort"]): i for i, cell in enumerate(cells)}
    for row in result["cells"]:
        row["sampled_J"] = float(j[0, indices[row["family"], row["effort"]]])
    available = result["cells"]
    if candidates is not None:
        wanted = set(candidates)
        unknown = wanted - indices.keys()
        if unknown:
            raise HowHardError(f"candidates: unknown cells {sorted(unknown)}")
        available = [row for row in available if (row["family"], row["effort"]) in wanted]
    if not available:
        raise HowHardError("candidates: no selectable cells")
    eligible = [row for row in available if row["admissible"]]
    cap_met = bool(eligible)
    if eligible:
        if "explore_within" in result["constants"]:
            threshold = min(row["J"] for row in eligible) * (1 + result["constants"]["explore_within"])
            eligible = [row for row in eligible if row["J"] <= threshold]
        chosen = min(eligible, key=lambda row: row["sampled_J"])
    elif config.data["on_cap_unmet"] == "lowest-bound":
        chosen = min(available, key=lambda row: (row["P_silent"]["upper"], row["sampled_J"]))
    else:
        chosen = min(available, key=lambda row: row["sampled_J"])
    result.update(chosen={k: chosen[k] for k in ("model", "family", "effort")}, seed=seed, cap_met=cap_met,
                  needs_review=not cap_met and config.data["on_cap_unmet"] == "flag-review", why=[])
    pick_index = indices[chosen["family"], chosen["effort"]]
    for row in result["cells"]:
        if row["C_run"] >= chosen["C_run"]:
            continue
        explanation = {k: row[k] for k in ("model", "family", "effort")}
        if not row["admissible"] and cap_met:
            explanation.update(reason="silent-cap", message=f"excluded: P_silent upper bound {row['P_silent']['upper']:.6g} > epsilon {result['constants']['epsilon']:.6g}")
        elif candidates is not None and (row["family"], row["effort"]) not in wanted:
            explanation.update(reason="candidate-filter", message="excluded: outside the requested candidates")
        elif cap_met and "explore_within" in result["constants"] and row not in eligible:
            explanation.update(reason="explore-within", message="excluded: expected J exceeds explore_within limit")
        elif not cap_met and config.data["on_cap_unmet"] == "lowest-bound":
            explanation.update(reason="lowest-bound", message="higher P_silent upper bound under lowest-bound policy")
        else:
            i = indices[row["family"], row["effort"]]
            differences = {name: float(value[0, i] - value[0, pick_index]) for name, value in parts.items()}
            explanation.update(reason="sampled-cost", components=differences,
                               message=", ".join(f"{name} {delta:+.6f} USD" for name, delta in differences.items()))
        result["why"].append(explanation)
    return result
