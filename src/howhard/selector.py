"""Native candidate policies and selection from LLM and memory forecasts."""

from copy import deepcopy
import secrets

from .validation import HowHardError, text


def prepare(config, task, risk=None, context=None, native_candidates=None):
    text(task, "task_type", label=True)
    cls = risk or config.task_types.get(task)
    if cls is None:
        if config.explicit:
            raise HowHardError(f"task_type {task!r} is unknown; provide --risk")
        cls = "standard"
    if cls not in config.data["classes"]:
        raise HowHardError(f"risk: unknown class {cls!r}")
    from .native import preflight
    inventory = preflight(config, native_candidates)
    cells = [{"model": c["model"], "family": config.data["models"][c["model"]]["family"],
              "effort": c["effort"], "price": c.get("usd_per_mtok")} for c in inventory]
    monetary = all(c["price"] is not None for c in cells)
    for cell in cells:
        cell["comparison_price"] = cell["price"] if monetary else 0.0
    result = {"task_type": task, "risk_class": cls, "constants": deepcopy(config.data["classes"][cls]),
              "cells": [{k: c[k] for k in ("model", "family", "effort")} for c in cells],
              "native_candidates": deepcopy(inventory), "context": deepcopy(context),
              "cost_policy": {"worker_money_included": monetary,
                              "unknown_price_models": sorted({c["model"] for c in cells if c["price"] is None}),
                              "comparison": "full-cost" if monetary else "risk-and-time-excluding-worker-money"},
              "paths": {"project_root": str(config.project_root), "state_root": str(config.state_root)}}
    return result, cells


def choose(prepared, config, seed, parts):
    result, cells = prepared
    result = deepcopy(result)
    if seed is None:
        seed = secrets.randbits(64)
    if type(seed) is not int or seed < 0:
        raise HowHardError("seed: expected a non-negative integer")
    j = sum(parts.values())
    indices = {(cell["family"], cell["effort"]): i for i, cell in enumerate(cells)}
    for row in result["cells"]:
        row["sampled_J"] = float(j[0, indices[row["family"], row["effort"]]])
    available = result["cells"]
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
        if not result["cost_policy"]["worker_money_included"]:
            continue  # Unknown prices do not establish a cheaper monetary alternative.
        if row["C_run"] >= chosen["C_run"]:
            continue
        explanation = {k: row[k] for k in ("model", "family", "effort")}
        if not row["admissible"] and cap_met:
            explanation.update(reason="silent-cap", message=f"excluded: P_silent upper bound {row['P_silent']['upper']:.6g} > epsilon {result['constants']['epsilon']:.6g}")
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
