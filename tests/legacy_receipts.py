"""Derive authentic frozen legacy receipts from a current receipt's base structure."""

from copy import deepcopy

import numpy as np

from howhard.semantic import OUTCOMES, panel_lookup
from howhard.success import accepted_risk, cost, denominator, envelope

LEGACY_SAMPLING = {"min_concentration": 8., "max_concentration": 1000000., "prior_strength": 8.,
                   "explore_within": .1, "uncertainty_multiplier": 1.}


def legacy_capability_prior(flt, policy, *, linear_tail=False):
    order = flt["order"]
    lo, hi = order.index(flt["lower"]), order.index(flt["upper"])
    minimum, reliable = policy["min_clear_probability"], policy["reliability_probability"]
    before = np.linspace(minimum * .2, minimum * .8, lo).tolist()
    fit = np.linspace(max(minimum, reliable - .2), reliable, hi - lo + 1).tolist() if hi > lo else [reliable]
    tail = min((1 - reliable) * .5, policy["overpowered_gain_margin"] * .5)
    after_count = len(order) - 1 - hi
    if linear_tail:
        after = np.linspace(reliable, reliable + tail, after_count + 1).tolist()[1:]
    elif after_count:
        steps = np.arange(1, after_count + 1, dtype=float)
        after = (reliable + tail * (1.0 - 0.5 ** steps) / (1.0 - 0.5 ** after_count)).tolist()
    else:
        after = []
    return dict(zip(order, before + fit + after))


def with_capability_v3(receipt):
    """Freeze the prior from before v4 for a fixture with a contiguous band."""
    receipt = deepcopy(receipt)
    state = receipt["capability"]
    lo, hi = state["order"].index(state["lower"]), state["order"].index(state["upper"])
    assert state["band"] == state["order"][lo:hi+1]
    state["format"] = "howhard-capability-v3"
    filters = {"filter:llm": state["baseline"]}
    filters.update({f"filter:memory:{m['id']}:{m['version']}": m["filter"] for m in state["memory_filters"]})
    state["worlds"] = {key: legacy_capability_prior(flt, state["policy"]) for key, flt in filters.items()}
    return receipt


def to_v8(receipt):
    """A v8 receipt: weighted-forecast envelopes, upper-bound admission, no outcome posteriors."""
    receipt = with_capability_v3(receipt)
    receipt["format"] = "howhard-decision-v8"
    semantic = receipt["semantic"]
    semantic["format"] = "howhard-semantic-v5"
    semantic["prompt_version"] = "semantic-reader-v6"
    semantic["interval_kind"] = "weighted-forecast-envelope"
    semantic.pop("outcome_observations")
    for panel in semantic["worlds"].values():
        for row in panel:
            row.pop("loud_ci", None)
            row.pop("silent_ci", None)
    semantic["sampling"]["policy"] = deepcopy(LEGACY_SAMPLING)
    if semantic["sampling"]["enabled"]:
        semantic["sampling"]["reason"] = "evidence-calibrated-dirichlet"
    for cell in semantic["sampling"]["cells"]:
        cell.pop("outcome_posterior")
        cell["concentration"] = max(8., cell["concentration"])
        cell.update(evidence_count=0, variance_sources={})
    selection = receipt["selection"]
    selection["sampling_policy"] = deepcopy(LEGACY_SAMPLING)
    panels = {k: panel_lookup(p) for k, p in semantic["worlds"].items()}
    experts = semantic["beliefs"]["experts"]
    for row in selection["cells"]:
        pair = row["family"], row["effort"]
        for outcome in OUTCOMES:
            mean = sum(e["weight"] * panels[e["key"]][pair][outcome] for e in experts)
            row["P_" + outcome] = envelope([(panels[e["key"]][pair][outcome], e["weight"]) for e in experts], mean)
        loud, silent = row["P_loud"]["mean"], row["P_silent"]["mean"]
        row["P_silent_accepted"] = envelope(
            [(accepted_risk(panels[e["key"]][pair]["loud"], panels[e["key"]][pair]["silent"]), e["weight"]) for e in experts],
            accepted_risk(loud, silent))
        row.pop("admission_risk")
        row["admissible"] = row["P_silent_accepted"]["upper"] <= selection["constants"]["epsilon"]
        run = row["session_cost_usd"] if receipt["cost_policy"]["worker_money_included"] else 0.
        row["J"] = cost(run, row["T"], loud, silent, selection["constants"])
        row["expected_attempts"] = 1 / denominator(loud)
    return receipt


def to_v7(receipt, capability_format="howhard-capability-v2"):
    """A v7 receipt additionally lacks active-duration calibration and timely capability fields."""
    receipt = to_v8(receipt)
    receipt["format"] = "howhard-decision-v7"
    receipt["semantic"]["format"] = "howhard-semantic-v4"
    receipt["semantic"]["prompt_version"] = "semantic-reader-v5"
    receipt["semantic"].pop("duration_observations", None)
    for cell in receipt["semantic"]["sampling"]["cells"]:
        cell.pop("duration", None)
    panels = {k: panel_lookup(p) for k, p in receipt["semantic"]["worlds"].items()}
    experts = receipt["semantic"]["beliefs"]["experts"]
    for row in receipt["selection"]["cells"]:
        pair = row["family"], row["effort"]
        row["T"] = sum(e["weight"] * panels[e["key"]][pair]["seconds"] for e in experts) / 3600
        run = row["session_cost_usd"] if receipt["cost_policy"]["worker_money_included"] else 0.
        row["J"] = cost(run, row["T"], row["P_loud"]["mean"], row["P_silent"]["mean"], receipt["selection"]["constants"])
    capability = receipt["capability"]
    capability["format"] = capability_format
    for cell in capability["cells"]:
        cell.pop("P_timely", None)
    for item in capability["evidence"]:
        for extra in ("active_seconds", "forecast_active_seconds", "timely_strength"):
            item.pop(extra, None)
    return receipt
