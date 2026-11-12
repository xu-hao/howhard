"""Content-addressed decision receipts bind later outcomes to decision-time context."""

import json
from pathlib import Path
import re

from .context import DECISION_FORMAT, FORMAT, canonical, digest, validate_project, validate_text
from .files import atomic_write
from .validation import HowHardError, keys, text


def receipt_path(config, decision_id):
    return config.state_root / "decisions" / f"{decision_id}.json"


def save(config, result):
    value = {"format": DECISION_FORMAT, "project": config.data["project"],
             "project_identity": config.project_identity,
             **{key: result[key] for key in ("task_type", "risk_class", "chosen", "seed", "context")}}
    value["semantic"] = result["semantic"]
    value["capability"] = result["capability"]
    value["selection"] = result["selection"]
    value["cost_policy"] = result["cost_policy"]
    value["native_candidates"] = result["native_candidates"]
    if "request_snapshot" in result:
        value["request_snapshot"] = result["request_snapshot"]
    decision_id = digest(value)
    from .storage import locked
    with locked(config, operation="receipt commit"):
        atomic_write(receipt_path(config, decision_id), canonical(value) + b"\n")
    return decision_id


def load(config, source, *, for_scoring=False):
    from . import state, native
    state.check(config)
    native.apply(config, native.read(config))
    if isinstance(source, dict):
        if "decision_id" in source:
            source = source["decision_id"]
        else:
            value = json.loads(canonical(source))
    if not isinstance(source, dict):
        if not isinstance(source, (str, Path)):
            raise HowHardError("decision: expected a receipt ID, path or object")
        source = str(source)
        path = receipt_path(config, source) if re.fullmatch(r"[0-9a-f]{64}", source) else Path(source)
        try:
            value = json.loads(path.read_bytes())
        except (OSError, ValueError, UnicodeError) as exc:
            raise HowHardError(f"decision: cannot read valid receipt {path}") from exc
        if re.fullmatch(r"[0-9a-f]{64}", source) and digest(value) != source:
            raise HowHardError("decision: receipt content does not match its ID")
    if for_scoring and isinstance(value, dict) and value.get("format") in {
        "howhard-decision-v1", "howhard-decision-v2", "howhard-decision-v3", "howhard-decision-v4", "howhard-decision-v5"
    }:
        return None  # Preserve old evidence without replaying a removed predictor.
    if not isinstance(value, dict) or value.get("format") != DECISION_FORMAT:
        raise HowHardError("decision.format: unsupported version; prepare a new capability-filtered decision")
    keys(value, ["format", "project", "project_identity", "task_type", "risk_class", "chosen", "seed", "context", "cost_policy", "native_candidates", "semantic", "capability", "selection"], ["request_snapshot"], where="decision")
    if "request_snapshot" in value:
        from .request_state import validate_provenance
        validate_provenance(value["request_snapshot"])
    policy = value["cost_policy"]
    keys(policy, ["worker_money_included", "unknown_price_models", "comparison", "basis"], where="decision.cost_policy")
    if type(policy["worker_money_included"]) is not bool or not isinstance(policy["unknown_price_models"], list):
        raise HowHardError("decision.cost_policy: invalid worker money policy")
    for model_id in policy["unknown_price_models"]:
        text(model_id, "decision.cost_policy.unknown_price_models")
    included = policy["worker_money_included"]
    if (included and policy["unknown_price_models"]) or policy["comparison"] != ("full-cost" if included else "risk-and-time-excluding-worker-money"):
        raise HowHardError("decision.cost_policy: inconsistent comparison")
    from .native import validate
    inventory = validate(value["native_candidates"])
    def priced(c):
        return all(v is not None for v in c["rate_card"]["rates"].values()) if "rate_card" in c else c.get("usd_per_mtok") is not None
    from .semantic import validate_receipt as validate_semantic
    validate_semantic(value["semantic"])
    if "request_snapshot" in value and value["request_snapshot"]["memory_revision"] != value["semantic"]["memory_revision"]:
        raise HowHardError("request_snapshot: memory revision differs from frozen semantic inputs")
    forecast_pairs = {(c["family"], c["effort"]) for c in value["semantic"]["worlds"]["llm"]}
    priced_inventory = [c for c in inventory if (config.data["models"].get(c["model"], {}).get("family"), c["effort"]) in forecast_pairs]
    unknown = sorted({c["model"] for c in priced_inventory if not priced(c)})
    bases = {"estimate:" + c["rate_card"]["basis"] if "rate_card" in c else "legacy-unspecified" for c in priced_inventory if priced(c)}
    if policy["basis"] != (next(iter(bases)) if len(bases) == 1 else "mixed") or included != (not unknown and len(bases) == 1):
        raise HowHardError("decision.cost_policy: inconsistent money basis")
    if unknown != policy["unknown_price_models"]:
        raise HowHardError("decision.cost_policy: does not match its forecasted native inventory")
    if value["project"] != config.data["project"]:
        raise HowHardError("decision.project: does not match this project")
    identity = value["project_identity"]
    if not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{32}", identity) or identity != config.project_identity:
        raise HowHardError("decision.project_identity: does not match this project's identity")
    text(value["task_type"], "decision.task_type", label=True)
    text(value["risk_class"], "decision.risk_class")
    if value["risk_class"] not in config.data["classes"]:
        raise HowHardError("decision.risk_class: unknown class")
    if type(value["seed"]) is not int or value["seed"] < 0:
        raise HowHardError("decision.seed: expected non-negative integer")
    keys(value["chosen"], ["model", "family", "effort"], where="decision.chosen")
    for key, item in value["chosen"].items():
        text(item, f"decision.chosen.{key}")
    if (value["chosen"]["model"], value["chosen"]["effort"]) not in {(c["model"], c["effort"]) for c in value["native_candidates"]}:
        raise HowHardError("decision.chosen: not in its native candidate inventory")
    model = config.data["models"].get(value["chosen"]["model"])
    if model is None or model["family"] != value["chosen"]["family"] or value["chosen"]["effort"] not in model["efforts"]:
        raise HowHardError("decision.chosen: model, family or effort does not match the registry")
    keys(value["context"], ["encoder", "task_text", "project_context"], where="decision.context")
    if value["context"]["encoder"] != FORMAT:
        raise HowHardError("decision.context.encoder: unsupported version")
    validate_text(value["context"]["task_text"])
    validate_project(value["context"]["project_context"])
    if "semantic" in value:
        from .capability import validate_receipt as validate_capability
        validate_capability(value["capability"], inventory, value["semantic"], value["chosen"])
        from .success import validate_selection
        validate_selection(value["selection"], value["semantic"], value["capability"], value["cost_policy"])
        by_pair = {(c["model"], c["effort"]): c for c in inventory}
        for row in value["selection"]["cells"]:
            if "credit_budget" in row and row["credit_budget"] != by_pair.get((row["model"], row["effort"]), {}).get("credit_budget"):
                raise HowHardError("decision.selection: credit_budget does not match native candidate inventory")
        chosen_row = next((c for c in value["selection"]["cells"] if (c["model"], c["effort"]) == (value["chosen"]["model"], value["chosen"]["effort"])), None)
        if chosen_row is None or not chosen_row.get("credit_ok", True):
            raise HowHardError("decision.chosen: selected cell is not schedulable within credit_budget")
        from .semantic import panel_lookup
        for panel in value["semantic"]["worlds"].values():
            lookup = panel_lookup(panel)
            for c in inventory:
                spec = config.data["models"][c["model"]]
                if "rate_card" in c and (spec["family"], c["effort"]) in lookup and "token_usage" not in lookup[spec["family"], c["effort"]]:
                    raise HowHardError("decision.semantic: category rate card requires token_usage forecasts")
    return value
