"""Validated native inventories and backend-owned historical identity metadata."""

from copy import deepcopy
import hashlib
import json

from .validation import HowHardError, keys, number, text

FORMAT = "howhard-native-v1"


def credit_budget(value):
    keys(value, ["available_credit_usd"], ["session_5h_remaining_credit_usd", "weekly_7d_remaining_credit_usd"], "candidate.credit_budget")
    normalized = {key: number(amount, f"candidate.credit_budget.{key}", nonnegative=True) for key, amount in value.items()}
    windows = [normalized[k] for k in ("session_5h_remaining_credit_usd", "weekly_7d_remaining_credit_usd") if k in normalized]
    if windows and abs(normalized["available_credit_usd"] - min(windows)) > 1e-6:
        raise HowHardError("candidate.credit_budget: available_credit_usd must match minimum remaining window credit")
    return normalized


def validate(candidates):
    if not isinstance(candidates, list) or not candidates:
        raise HowHardError("candidates: expected at least one native model/effort cell")
    result, seen = [], set()
    for cell in candidates:
        keys(cell, ["model", "effort"], ["usd_per_mtok", "rate_card", "cost_per_credit", "credit_budget"], "native candidate")
        model = text(cell["model"], "candidate.model")
        effort = text(cell["effort"], "candidate.effort", label=True)
        if len(model) > 256 or any(ord(c) < 32 for c in model) or len(effort) > 64 or effort == "model_default":
            raise HowHardError("candidates: supply actual bounded model and effort/control identities, not model_default")
        pair = model, effort
        if pair in seen:
            raise HowHardError("candidates: duplicate native model/effort cell")
        seen.add(pair)
        if "rate_card" in cell:
            from .accounting import rate_card
            rate_card(cell["rate_card"])
            if "usd_per_mtok" in cell or cell["rate_card"]["model"] != model:
                raise HowHardError("candidate.rate_card: model must match and blended price is exclusive")
        credit_rate = cell.get("usd_per_mtok")
        if credit_rate is not None:
            credit_rate = number(credit_rate, "candidate.usd_per_mtok", nonnegative=True)
        cost_per_credit = number(cell.get("cost_per_credit", 1.0), "candidate.cost_per_credit", nonnegative=True)
        budget = None
        if "credit_budget" in cell:
            budget = credit_budget(cell["credit_budget"])
            if "rate_card" not in cell and credit_rate is None:
                raise HowHardError("candidate.credit_budget: requires rate_card or usd_per_mtok")
        result.append({"model": model, "effort": effort,
                       **({"usd_per_mtok": credit_rate} if "usd_per_mtok" in cell else {}),
                       **({"rate_card": deepcopy(cell["rate_card"])} if "rate_card" in cell else {}),
                       "cost_per_credit": cost_per_credit,
                       **({"credit_budget": deepcopy(budget)} if budget is not None else {})})
    if any("credit_budget" in c for c in result):
        from .accounting import CATEGORIES
        priced = [all(c["rate_card"]["rates"][k] is not None for k in CATEGORIES) if "rate_card" in c else c.get("usd_per_mtok") is not None for c in result]
        bases = {"estimate:" + c["rate_card"]["basis"] if "rate_card" in c else "legacy-unspecified" for c, ok in zip(result, priced) if ok}
        if not all(priced) or len(bases) != 1:
            raise HowHardError("candidates: credit_budget requires uniform pricing across all candidates")
    return sorted(result, key=lambda c: (c["model"], c["effort"]))


def read(config):
    path = config.state_root / "native-models.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise HowHardError("native registry: invalid JSON") from exc
    keys(value, ["format", "models"], where="native registry")
    if value["format"] != FORMAT or not isinstance(value["models"], dict):
        raise HowHardError("native registry: unsupported format or models")
    for model, spec in value["models"].items():
        keys(spec, ["efforts"], where="native model")
        if not isinstance(spec["efforts"], list) or not spec["efforts"]:
            raise HowHardError("native registry: expected efforts")
        for effort in spec["efforts"]:
            validate([{"model": model, "effort": effort}])
        if len(spec["efforts"]) != len(set(spec["efforts"])):
            raise HowHardError("native registry: duplicate efforts")
    return value["models"]


def apply(config, models):
    config.native_models = deepcopy(models)
    for model, spec in sorted(models.items()):
        if model in config.data["models"] and not config.data["models"][model].get("native"):
            raise HowHardError(f"native registry: conflicts with configured model {model}")
        family = "native-" + hashlib.sha256(model.encode()).hexdigest()
        config.data["models"][model] = {"family": family, "efforts": spec["efforts"],
                                        "weight": 1.0, "native": True}


def resolve(config, candidates=None, inventory=None):
    """Use only caller-supplied availability or an explicitly named inventory."""
    if inventory is not None:
        if candidates is not None:
            raise HowHardError("inventory: cannot combine a named inventory with native_candidates")
        text(inventory, "inventory", label=True)
        if inventory not in config.data.get("inventories", {}):
            raise HowHardError(f"inventory: unknown name {inventory!r}")
        candidates = config.data["inventories"][inventory]
    return preflight(config, candidates)


def preflight(config, candidates):
    candidates = validate(candidates)
    # Validate every configured identity/control before persisting anything.
    pairs = set()
    for cell in candidates:
        spec = config.data["models"].get(cell["model"])
        if spec and not spec.get("native") and cell["effort"] not in spec["efforts"]:
            raise HowHardError(f"candidate.effort: unsupported configured control for {cell['model']}")
        family = spec["family"] if spec else "native-" + hashlib.sha256(cell["model"].encode()).hexdigest()
        if (family, cell["effort"]) in pairs:
            raise HowHardError("candidates: registered models share a family/effort cell")
        pairs.add((family, cell["effort"]))
    return candidates


def register(config, candidates):
    from .storage import locked
    candidates = preflight(config, candidates)
    with locked(config, operation="native registration"):
        from .state import check
        check(config)
        models, updated = plan(config, candidates)
        persist(config, models, updated)
    return candidates


def plan(config, candidates):
    candidates = preflight(config, candidates)
    models = read(config)
    updated = deepcopy(models)
    for cell in candidates:
        model, effort = cell["model"], cell["effort"]
        if model in config.data["models"] and not config.data["models"][model].get("native"):
            continue
        efforts = updated.setdefault(model, {"efforts": []})["efforts"]
        if effort not in efforts:
            efforts.append(effort)
            efforts.sort()
    return models, updated


def persist(config, models, updated):
    from .files import atomic_write
    if updated != models:
        atomic_write(config.state_root / "native-models.json", (json.dumps({"format": FORMAT, "models": updated}, sort_keys=True) + "\n").encode())
    apply(config, updated)
