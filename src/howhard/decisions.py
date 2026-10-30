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
    value["cost_policy"] = result["cost_policy"]
    value["native_candidates"] = result["native_candidates"]
    decision_id = digest(value)
    atomic_write(receipt_path(config, decision_id), canonical(value) + b"\n")
    return decision_id


def load(config, source, *, for_scoring=False):
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
        "howhard-decision-v1", "howhard-decision-v2", "howhard-decision-v3"
    }:
        return None  # Preserve old evidence without replaying a removed predictor.
    keys(value, ["format", "project", "project_identity", "task_type", "risk_class", "chosen", "seed", "context", "cost_policy", "native_candidates", "semantic"], where="decision")
    if value["format"] != DECISION_FORMAT:
        raise HowHardError("decision.format: unsupported version")
    policy = value["cost_policy"]
    keys(policy, ["worker_money_included", "unknown_price_models", "comparison"], where="decision.cost_policy")
    if type(policy["worker_money_included"]) is not bool or not isinstance(policy["unknown_price_models"], list):
        raise HowHardError("decision.cost_policy: invalid worker money policy")
    for model_id in policy["unknown_price_models"]:
        text(model_id, "decision.cost_policy.unknown_price_models")
    included = policy["worker_money_included"]
    if included != (not policy["unknown_price_models"]) or policy["comparison"] != ("full-cost" if included else "risk-and-time-excluding-worker-money"):
        raise HowHardError("decision.cost_policy: inconsistent comparison")
    from .native import validate
    inventory = validate(value["native_candidates"])
    unknown = sorted({c["model"] for c in inventory if c.get("usd_per_mtok") is None})
    if unknown != policy["unknown_price_models"]:
        raise HowHardError("decision.cost_policy: does not match its native inventory")
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
        from .semantic import validate_receipt
        validate_receipt(value["semantic"])
    return value
