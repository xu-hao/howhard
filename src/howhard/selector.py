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
              "effort": c["effort"], "credit_rate": c.get("usd_per_mtok"),
              "rate_card": c.get("rate_card"), "cost_per_credit": c["cost_per_credit"],
              "credit_budget": deepcopy(c.get("credit_budget"))} for c in inventory]
    from .accounting import CATEGORIES
    for cell in cells:
        card = cell["rate_card"]
        cell["priced"] = all(card["rates"][k] is not None for k in CATEGORIES) if card else cell["credit_rate"] is not None
        cell["money_basis"] = "estimate:" + card["basis"] if card else "legacy-unspecified" if cell["priced"] else None
    bases = {c["money_basis"] for c in cells if c["priced"]}
    monetary = all(c["priced"] for c in cells) and len(bases) == 1
    if any(c["credit_budget"] is not None for c in cells) and not monetary:
        raise HowHardError("candidates: credit_budget requires uniform pricing across all candidates")
    for cell in cells:
        cell["money_included"] = monetary
    result = {"task_type": task, "risk_class": cls, "constants": deepcopy(config.data["classes"][cls]),
              "cells": [{k: c[k] for k in ("model", "family", "effort")} for c in cells],
              "native_candidates": deepcopy(inventory), "context": deepcopy(context),
              "cost_policy": {"worker_money_included": monetary,
                              "unknown_price_models": sorted({c["model"] for c in cells if not c["priced"]}),
                              "basis": next(iter(bases)) if len(bases) == 1 else "mixed",
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
    schedulable = [row for row in available if row.get("credit_ok", True)]
    if not schedulable:
        min_needed = min(row["needed_credit_usd"] for row in available)
        max_avail = max(row["available_credit_usd"] for row in available)
        raise HowHardError(
            f"candidates: no candidate has sufficient remaining credit to schedule task "
            f"(minimum needed_credit_usd {min_needed:.6g} USD, maximum available_credit_usd {max_avail:.6g} USD)"
        )
    eligible = [row for row in schedulable if row["admissible"]]
    cap_met = bool(eligible)
    best = min(eligible or schedulable, key=lambda row: (row["J"], row["family"], row["effort"]))
    sampling = result["semantic"]["sampling"]
    policy = result["sampling_policy"]
    fraction = min(policy["explore_within"], result["constants"].get("explore_within", policy["explore_within"]))
    if eligible:
        uncertainty = {(c["family"], c["effort"]): c["cost_sd"] for c in sampling["cells"]}
        best_sd = uncertainty[best["family"], best["effort"]]
        eligible = [row for row in eligible if row["J"]-best["J"] <= min(
            abs(best["J"])*fraction,
            policy["uncertainty_multiplier"]*(best_sd+uncertainty[row["family"], row["effort"]])) + 1e-12]
        chosen = min(eligible, key=lambda row: (row["sampled_J"], row["J"], row["family"], row["effort"]))
        reason = "sampled-within-supported-mean-band" if chosen is not best else "mean-best" if len(eligible) > 1 else "disabled-no-supported-alternative"
    elif config.data["on_cap_unmet"] == "lowest-bound":
        chosen = min(schedulable, key=lambda row: (row["P_silent_accepted"]["upper"], row["J"], row["family"], row["effort"]))
        reason = "disabled-cap-unmet"
    else:
        chosen = best
        reason = "disabled-cap-unmet"
    result.update(chosen={k: chosen[k] for k in ("model", "family", "effort")}, seed=seed, cap_met=cap_met,
                  needs_review=not cap_met, why=[])
    result["exploration"] = {"enabled": cap_met and len(eligible) > 1, "reason": reason,
                             "mean_best": {k: best[k] for k in ("model", "family", "effort")},
                             "mean_best_J": best["J"], "chosen_mean_J": chosen["J"],
                             "mean_gap": chosen["J"]-best["J"], "relative_band": fraction,
                             "eligible_cells": [{k: row[k] for k in ("model", "family", "effort")} for row in eligible]}
    result["selection"] = {"objective": "cost-to-accepted-result", "retry_policy": "same-cell-until-non-loud",
                           "constants": deepcopy(result["constants"]), "sampling_policy": deepcopy(policy),
                           "cells": deepcopy(result["cells"]), "exploration": deepcopy(result["exploration"]),
                           "cap_met": cap_met, "needs_review": not cap_met, "on_cap_unmet": config.data["on_cap_unmet"]}
    pick_index = indices[chosen["family"], chosen["effort"]]
    for row in result["cells"]:
        if not result["cost_policy"]["worker_money_included"]:
            continue  # Unknown prices do not establish a cheaper monetary alternative.
        if row["session_cost_usd"] >= chosen["session_cost_usd"]:
            continue
        explanation = {k: row[k] for k in ("model", "family", "effort")}
        if not row.get("credit_ok", True):
            explanation.update(
                reason="insufficient-credit",
                message=f"excluded: needed_credit_usd {row['needed_credit_usd']:.6g} > available_credit_usd {row['available_credit_usd']:.6g}",
            )
        elif not row["admissible"] and cap_met:
            explanation.update(reason="silent-cap", message=f"excluded: P_silent_accepted upper bound {row['P_silent_accepted']['upper']:.6g} > epsilon {result['constants']['epsilon']:.6g}")
        elif cap_met and "explore_within" in result["constants"] and row not in eligible:
            explanation.update(reason="explore-within", message="excluded: mean accepted-result cost exceeds the supported exploration band")
        elif not cap_met and config.data["on_cap_unmet"] == "lowest-bound":
            explanation.update(reason="lowest-bound", message="higher accepted-result silent upper bound under lowest-bound policy")
        else:
            i = indices[row["family"], row["effort"]]
            differences = {name: float(value[0, i] - value[0, pick_index]) for name, value in parts.items()}
            explanation.update(reason="sampled-cost", components=differences,
                               message=", ".join(f"{name} {delta:+.6f} USD" for name, delta in differences.items()))
        result["why"].append(explanation)
    return result
