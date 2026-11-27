"""Content-addressed decision receipts bind later outcomes to decision-time context."""

import json
from pathlib import Path
import re

from .context import DECISION_FORMAT, FORMAT, LEGACY_DECISION_V7, LEGACY_DECISION_V8, LEGACY_DECISION_V9, LEGACY_DECISION_V10, canonical, digest, validate_project, validate_text
from .validation import HowHardError, keys, text


def host(receipt):
    """The name of the host that asked for the route; ``None`` is the one unnamed host.

    Receipts older than v11 carry no name and read as unnamed; ``howhard.names.host`` adds
    the name an operator imported for one.
    """
    return receipt.get("host")


def agent(receipt, cell=None):
    """The agent label stored with a cell (the chosen cell by default); ``None`` is unnamed."""
    cell = receipt["chosen"] if cell is None else cell
    pair = cell["model"], cell["effort"]
    return next((c.get("agent") for c in receipt["native_candidates"] if (c["model"], c["effort"]) == pair), None)


def scored_receipt(config, row, receipts=None):
    """The receipt a recorded run's ``decision_id`` names, as feedback scoring reads it.

    ``None`` when the row names no decision, the receipt is not stored, or its format is older
    than the ones scoring replays. ``receipts`` caches each receipt by its ID.
    """
    decision_id = row.get("decision_id")
    if not decision_id:
        return None
    if receipts is not None and decision_id in receipts:
        return receipts[decision_id]
    receipt = load(config, decision_id, for_scoring=True) if exists(config, decision_id) else None
    if receipts is not None:
        receipts[decision_id] = receipt
    return receipt


def run_agent(config, row, receipts=None, names=None):
    """The agent that ran a recorded row: the label its decision stored for the chosen cell.

    A v11 receipt's label, or its unlabeled candidate, is final. A run whose receipt stores no
    agent label (one older than v11, or a run with no decision) takes the agent ``names``
    imported for its family; with none it belongs to the one unnamed agent, ``None``.
    """
    from .names import family_agent
    receipt = scored_receipt(config, row, receipts)
    if receipt is not None and receipt["format"] == DECISION_FORMAT:
        return agent(receipt)
    return family_agent(row["family"], names)


def counted_observations(receipt, cell=None):
    """The runs one forecast cell of a v11 receipt counts as outcome evidence, computed on demand.

    Each row is a run's ``id``, its ``relation`` to the cell (``cell``, ``family``, ``agent`` for
    another family of the same agent, or ``other`` for another agent), its ``discount`` and its
    ``weight`` after the per-ticket limit. A receipt never stores these rows: they follow from
    its outcome observations, the cell's agent label and the discounts in its sampling policy.
    ``cell`` is a selection cell, the chosen one by default.
    """
    if receipt.get("format") != DECISION_FORMAT:
        raise HowHardError("decision: counted observations are computed for howhard-decision-v11 receipts only")
    cell = receipt["chosen"] if cell is None else cell
    from .success import counted_observations as counted
    return counted((cell["family"], cell["effort"], agent(receipt, cell)),
                   receipt["semantic"]["outcome_observations"], receipt["selection"]["sampling_policy"])


def save(config, result):
    value = {"format": DECISION_FORMAT, "project": config.data["project"],
             "project_identity": config.project_identity, "host": result["host"],
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
        from .payload_store import put
        put(config, "decisions", decision_id, canonical(value) + b"\n")
    return decision_id


def exists(config, decision_id):
    from .payload_store import exists as stored
    return stored(config, "decisions", decision_id)


def summaries(config, **filters):
    from .payload_store import decision_summaries
    from .state import check
    check(config)
    return decision_summaries(config, **filters)


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
        is_id = bool(re.fullmatch(r"[0-9a-f]{64}", source))
        try:
            if is_id:
                from .payload_store import get
                blob = get(config, "decisions", source)
                if blob is None:
                    raise HowHardError(f"decision: cannot read valid receipt {source}")
            else:
                blob = Path(source).read_bytes()
            value = json.loads(blob)
        except (OSError, ValueError, UnicodeError) as exc:
            raise HowHardError(f"decision: cannot read valid receipt {source}") from exc
        if is_id and digest(value) != source:
            raise HowHardError("decision: receipt content does not match its ID")
    if for_scoring and isinstance(value, dict) and value.get("format") in {
        "howhard-decision-v1", "howhard-decision-v2", "howhard-decision-v3", "howhard-decision-v4", "howhard-decision-v5", "howhard-decision-v6"
    }:
        return None  # Preserve old evidence without replaying a removed predictor.
    if not isinstance(value, dict) or value.get("format") not in (DECISION_FORMAT, LEGACY_DECISION_V10, LEGACY_DECISION_V9, LEGACY_DECISION_V8, LEGACY_DECISION_V7):
        raise HowHardError("decision.format: unsupported version; prepare a new capability-filtered decision")
    # The host's name and the agent labels start with v11; older receipts read as unnamed.
    named = value["format"] == DECISION_FORMAT
    # A v11 receipt is readable only in its frozen shape, which always records the request it
    # came from; older receipts may lack that provenance.
    keys(value, ["format", "project", "project_identity", "task_type", "risk_class", "chosen", "seed", "context", "cost_policy", "native_candidates", "semantic", "capability", "selection",
                 *(["host", "request_snapshot"] if named else [])], [] if named else ["request_snapshot"], where="decision")
    if named:
        from .native import host_name
        try:
            host_name(value["host"])
        except HowHardError as exc:
            raise HowHardError(f"decision.{exc}") from exc
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
    inventory = validate(value["native_candidates"], labels=named)
    def priced(c):
        return all(v is not None for v in c["rate_card"]["rates"].values()) if "rate_card" in c else c.get("usd_per_mtok") is not None
    from .semantic import validate_receipt as validate_semantic
    validate_semantic(value["semantic"], decision_format=value["format"], selection=value["selection"],
                      host=host(value), inventory=inventory)
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
        validate_capability(value["capability"], inventory, value["semantic"], value["chosen"],
                            decision_format=value["format"], constants=value["selection"].get("constants"),
                            host=host(value))
        from .success import validate_selection
        validate_selection(value["selection"], value["semantic"], value["capability"], value["cost_policy"],
                           decision_format=value["format"], chosen=value["chosen"], inventory=inventory,
                           seed=value["seed"])
        by_pair = {(c["model"], c["effort"]): c for c in inventory}
        for row in value["selection"]["cells"]:
            candidate = by_pair.get((row["model"], row["effort"]))
            if candidate is None:
                raise HowHardError("decision.selection: cell does not match native candidate inventory")
            if row["credit_budget"] != candidate.get("credit_budget"):
                raise HowHardError("decision.selection: credit_budget does not match native candidate inventory")
            if row["cost_per_credit"] != candidate["cost_per_credit"]:
                raise HowHardError("decision.selection: cost_per_credit does not match native candidate inventory")
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
