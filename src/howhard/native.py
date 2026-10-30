"""Validated native inventories and backend-owned historical identity metadata."""

from copy import deepcopy
import hashlib
import json

from .validation import HowHardError, keys, number, text

FORMAT = "howhard-native-v1"


def validate(candidates):
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 32:
        raise HowHardError("candidates: expected between 1 and 32 native model/effort cells")
    result, seen = [], set()
    for cell in candidates:
        keys(cell, ["model", "effort"], ["usd_per_mtok"], "native candidate")
        model = text(cell["model"], "candidate.model")
        effort = text(cell["effort"], "candidate.effort", label=True)
        if len(model) > 256 or any(ord(c) < 32 for c in model) or len(effort) > 64 or effort == "model_default":
            raise HowHardError("candidates: supply actual bounded model and effort/control identities, not model_default")
        pair = model, effort
        if pair in seen:
            raise HowHardError("candidates: duplicate native model/effort cell")
        seen.add(pair)
        price = cell.get("usd_per_mtok")
        if price is not None:
            price = number(price, "candidate.usd_per_mtok", nonnegative=True)
        result.append({"model": model, "effort": effort, **({"usd_per_mtok": price} if "usd_per_mtok" in cell else {})})
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
        if family in config.data["families"] and not config.data["models"].get(model, {}).get("native"):
            raise HowHardError("native registry: family identity conflict")
        config.data["families"][family] = {"current": model, "selectable": False, "tier": 0}
        config.data["models"][model] = {"family": family, "efforts": spec["efforts"],
                                        "default_effort": spec["efforts"][0], "weight": 1.0, "native": True}


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
    with locked(config):
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
    from .transfer import atomic_write
    if updated != models:
        atomic_write(config.state_root / "native-models.json", (json.dumps({"format": FORMAT, "models": updated}, sort_keys=True) + "\n").encode())
    apply(config, updated)
