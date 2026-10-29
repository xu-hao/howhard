"""Content-addressed decision receipts bind later outcomes to decision-time context."""

import json
from pathlib import Path
import re

from .context import DECISION_FORMAT, FORMAT, canonical, digest, validate_project, validate_text
from .transfer import atomic_write
from .validation import HowHardError, keys, text


def receipt_path(config, decision_id):
    return config.base / ".howhard" / "decisions" / f"{decision_id}.json"


def save(config, result):
    value = {"format": DECISION_FORMAT, "project": config.data["project"],
             **{key: result[key] for key in ("task_type", "risk_class", "chosen", "seed", "context")}}
    decision_id = digest(value)
    atomic_write(receipt_path(config, decision_id), canonical(value) + b"\n")
    return decision_id


def load(config, source):
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
    keys(value, ["format", "project", "task_type", "risk_class", "chosen", "seed", "context"], where="decision")
    if value["format"] != DECISION_FORMAT:
        raise HowHardError("decision.format: unsupported version")
    if value["project"] != config.data["project"]:
        raise HowHardError("decision.project: does not match this project")
    text(value["task_type"], "decision.task_type", label=True)
    text(value["risk_class"], "decision.risk_class")
    if value["risk_class"] not in config.data["classes"]:
        raise HowHardError("decision.risk_class: unknown class")
    if type(value["seed"]) is not int or value["seed"] < 0:
        raise HowHardError("decision.seed: expected non-negative integer")
    keys(value["chosen"], ["model", "family", "effort"], where="decision.chosen")
    for key, item in value["chosen"].items():
        text(item, f"decision.chosen.{key}")
    model = config.data["models"].get(value["chosen"]["model"])
    if model is None or model["family"] != value["chosen"]["family"] or value["chosen"]["effort"] not in model["efforts"]:
        raise HowHardError("decision.chosen: model, family or effort does not match the registry")
    keys(value["context"], ["encoder", "task_text", "project_context"], where="decision.context")
    if value["context"]["encoder"] != FORMAT:
        raise HowHardError("decision.context.encoder: unsupported version")
    validate_text(value["context"]["task_text"])
    validate_project(value["context"]["project_context"])
    return value
