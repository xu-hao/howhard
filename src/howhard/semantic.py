"""LLM reasoning over semantic memories with approximate predictive-evidence weights."""

from copy import deepcopy
import json
import math
from pathlib import Path
import re
import time
from datetime import datetime, timezone

import numpy as np

from .context import canonical, digest, SOURCE_SUFFIXES, MANIFESTS, IGNORED
from .llm import LLMError, SETTINGS
from .files import atomic_write
from .validation import HowHardError, keys, number, text

FORMAT = SETTINGS["format"]
OUTCOMES = ("clean", "loud", "silent")
FIELDS = ("family", "effort", "clean", "loud", "silent", "tokens", "seconds")


def memory_key(entry):
    return f"memory:{entry['id']}:{entry['version']}"


def panel_lookup(panel):
    return {(c["family"], c["effort"]): c for c in panel}


def beliefs(engine, snapshot=None):
    """Replay effective outcomes; only pre-recorded forecasts earn predictive support."""
    from .decisions import load, receipt_path
    snapshot = engine.memory.snapshot() if snapshot is None else snapshot
    priors = SETTINGS["belief"]
    entries = {"llm": priors["llm_prior"],
               **{memory_key(m): priors["memory_prior"] for m in snapshot["memories"]}}
    evidence = {key: [] for key in entries}
    unscored = 0
    latest = {}
    roots = {}
    for index, raw in enumerate(engine.raw_rows):
        roots[raw["id"]] = roots[raw["amends"]] if raw["amends"] else raw["id"]
        latest[roots[raw["id"]]] = index
    for row in sorted(engine.rows, key=lambda r: latest.get(r["id"], -1)):
        decision_id = row.get("decision_id")
        if not decision_id:
            continue
        if not receipt_path(engine.config, decision_id).is_file():
            unscored += 1
            continue
        receipt = load(engine.config, decision_id, for_scoring=True)
        if receipt is None:
            unscored += 1
            continue
        state = receipt["semantic"]
        if (receipt["task_type"], receipt["chosen"]["family"], receipt["chosen"]["effort"]) != (row["task_type"], row["family"], row["resolved_effort"]):
            raise HowHardError(f"semantic evidence: receipt mismatch for {row['id']}")
        cell = row["family"], row["resolved_effort"]
        baseline = panel_lookup(state["worlds"]["llm"])[cell][row["outcome"]]
        floor = priors["probability_floor"]
        for key, panel in state["worlds"].items():
            if key in entries:
                if key.startswith("memory:") and list(cell) not in state["exposures"].get(key, []):
                    continue
                p = panel_lookup(panel)[cell][row["outcome"]]
                score = math.log(max(floor, p)) - math.log(max(floor, baseline))
                reference = panel_lookup(state["worlds"]["llm"])[cell]
                candidate = panel_lookup(panel)[cell]
                for field, measurement in (("tokens", row["tokens"]), ("seconds", row["seconds"])):
                    if measurement is not None:
                        observed = math.log(max(1, measurement))
                        residual = observed - math.log(max(1, candidate[field]))
                        base_residual = observed - math.log(max(1, reference[field]))
                        score += priors["resource_weight"] * (base_residual ** 2 - residual ** 2) / (2 * priors["resource_noise_sd"] ** 2)
                evidence[key].append(row["weight"] * score)
    scores = {key: math.log(prior) + priors["temperature"] * math.fsum(value * priors["recency"] ** age for age, value in enumerate(reversed(evidence[key]))) for key, prior in entries.items()}
    peak = max(scores.values())
    mass = {key: math.exp(score - peak) for key, score in scores.items()}
    total = math.fsum(mass.values())
    return {"revision": snapshot["revision"], "kind": "tempered-predictive-evidence", "unscored_runs": unscored,
            "experts": [{"key": key, "weight": mass[key] / total, "log_score": scores[key], "evidence_count": len(evidence[key])} for key in sorted(entries)]}


def evidence_search(engine, query="", limit=None):
    terms = set(re.findall(r"\w+", query.casefold()))
    def rank(row):
        content = " ".join(str(row.get(k, "")) for k in ("task_type", "task_text", "note", "outcome"))
        return (-len(terms & set(re.findall(r"\w+", content.casefold()))), row["id"])
    return deepcopy(sorted(engine.rows, key=rank)[:limit or SETTINGS["evidence_limit"]])


def tool(engine, request, *, allow_project=True, project_snapshot=None):
    keys(request, ["tool"], ["query", "id", "path"], "LLM retrieval")
    name = request["tool"]
    if name == "memory.search":
        keys(request, ["tool", "query"], where="LLM retrieval")
        return engine.memory.search(text(request["query"], "retrieval.query"))
    if name == "memory.read":
        keys(request, ["tool", "id"], where="LLM retrieval")
        return engine.memory.read(request["id"])
    if name == "evidence.search":
        keys(request, ["tool", "query"], where="LLM retrieval")
        return evidence_search(engine, text(request["query"], "retrieval.query"))
    if name == "evidence.read":
        keys(request, ["tool", "id"], where="LLM retrieval")
        evidence_id = text(request["id"], "retrieval.id")
        raw = {r["id"]: r for r in engine.raw_rows}
        if evidence_id not in raw:
            raise HowHardError(f"retrieval.id: unknown evidence {evidence_id}")
        root = raw[evidence_id]
        while root["amends"] is not None:
            root = raw[root["amends"]]
        return deepcopy(next(r for r in engine.rows if r["id"] == root["id"]))
    if name == "project.read" and allow_project:
        keys(request, ["tool", "path"], where="LLM retrieval")
        relative = Path(text(request["path"], "retrieval.path"))
        path = engine.config.project_root / relative
        resolved = path.resolve()
        if resolved.is_relative_to(engine.config.state_root.resolve()):
            raise HowHardError("retrieval.path: outside project or excluded backend state")
        if relative.is_absolute() or not resolved.is_relative_to(engine.config.project_root) or any(p in IGNORED or p.startswith(".venv") for p in relative.parts):
            raise HowHardError("retrieval.path: outside project or excluded state")
        if not (resolved.suffix.lower() in SOURCE_SUFFIXES | {".md", ".rst", ".txt"} or resolved.name in MANIFESTS):
            raise HowHardError("retrieval.path: only project source, documentation and manifests may be read")
        if not resolved.is_file():
            raise HowHardError("retrieval.path: expected a regular project file")
        if project_snapshot is not None:
            from .context import capture_project
            if capture_project(engine.config, project_snapshot["parameters"], required_path=resolved)["fingerprint"] != project_snapshot["fingerprint"]:
                raise LLMError("Project changed since the decision snapshot; start a fresh decision instead of reading newer context")
        try:
            with resolved.open("rb") as handle:
                data = handle.read(32769)
        except OSError as exc:
            raise HowHardError(f"retrieval.path: {exc}") from exc
        return {"path": str(relative), "text": data[:32768].decode("utf-8", errors="replace"), "truncated": len(data) > 32768}
    raise HowHardError(f"LLM retrieval: unknown or unavailable tool {name!r}")


def request(engine, prepared):
    result = prepared[0]
    snapshot = engine.memory.snapshot()
    state = beliefs(engine, snapshot)
    weights = {e["key"]: e["weight"] for e in state["experts"]}
    # The reader can retrieve other entries before forecasting; this is a bounded
    # initial view, not a fixed-size memory bank.
    memories = sorted(snapshot["memories"], key=lambda m: (-weights[memory_key(m)], m["id"]))[:SETTINGS["memory_limit"]]
    payload = {"kind": "predict", "prompt_version": SETTINGS["prompt_version"], "task_type": result["task_type"],
               "context": deepcopy(result["context"]), "risk_class": result["risk_class"], "constants": result["constants"],
               "cells": deepcopy(result["cells"]), "memory_revision": snapshot["revision"], "memories": memories,
               "beliefs": state, "evidence": evidence_search(engine, result["context"]["task_text"]),
               "retrieval": [], "tools": ["memory.search", "memory.read", "evidence.search", "evidence.read", "project.read"],
               "contract": {"forecasts": "One row for EVERY supplied cell: family, effort, clean, loud, silent, tokens, seconds. Probabilities sum to 1; usage estimates are nonnegative.",
                            "memory_forecasts": "Array of {id, version, forecasts}: predictions conditional on each relevant supplied/read ACTIVE memory. Rows may override a subset of baseline forecasts. No invented memories.",
                            "reason": "Brief explanation grounded in task/project semantics and actual evidence.",
                            "request_more": "Instead of forecasts, optionally return ONLY {request_more: [{tool, query/id/path}]} to retrieve more evidence. At most 3 requests per round."}}
    if "cost_policy" in result:
        payload["cost_policy"] = deepcopy(result["cost_policy"])
    return payload, snapshot


def baseline_request(payload):
    clean = {k: deepcopy(payload[k]) for k in ("kind", "prompt_version", "task_type", "context", "risk_class", "constants")}
    if "cost_policy" in payload:
        clean["cost_policy"] = deepcopy(payload["cost_policy"])
    clean.update(phase="baseline", cells=[{k: c[k] for k in ("family", "effort", "model")} for c in payload["cells"]],
                 contract="Return {forecasts: [{family,effort,clean,loud,silent,tokens,seconds}], reason: string}. Forecast EVERY supplied cell; probabilities sum to 1. No memory, evidence or retrieval is available in this independent call.")
    return clean


def validate_panel(panel, allowed, *, complete=True):
    if not isinstance(panel, list) or not panel:
        raise LLMError("LLM forecasts: expected a nonempty array")
    seen = set()
    for row in panel:
        keys(row, FIELDS, where="LLM forecast")
        text(row["family"], "forecast.family")
        text(row["effort"], "forecast.effort")
        pair = row["family"], row["effort"]
        if pair not in allowed or pair in seen:
            raise LLMError(f"LLM forecasts: unknown or duplicate cell {pair}")
        seen.add(pair)
        ps = [number(row[k], f"forecast.{k}", nonnegative=True) for k in OUTCOMES]
        if any(p > 1 for p in ps) or abs(sum(ps) - 1) > 1e-6:
            raise LLMError("LLM forecasts: probabilities must sum to 1")
        for field in ("tokens", "seconds"):
            number(row[field], f"forecast.{field}", nonnegative=True)
    if complete and seen != set(allowed):
        raise LLMError("LLM forecasts: missing candidate cells")
    return panel


def validate_response(response, payload, snapshot):
    keys(response, ["forecasts", "memory_forecasts", "reason"], where="LLM response")
    text(response["reason"], "LLM response.reason")
    allowed = {(c["family"], c["effort"]) for c in payload["cells"]}
    validate_panel(response["forecasts"], allowed)
    if not isinstance(response["memory_forecasts"], list):
        raise LLMError("memory_forecasts: expected array")
    known = {(m["id"], m["version"]): m for m in snapshot["memories"]}
    visible = {(m["id"], m["version"]) for m in payload["memories"]}
    for item in payload["retrieval"]:
        data = item["result"]
        if item["request"]["tool"].startswith("memory."):
            for entry in data if isinstance(data, list) else [data]:
                if isinstance(entry, dict) and entry.get("active"):
                    visible.add((entry["id"], entry["version"]))
    seen = set()
    for item in response["memory_forecasts"]:
        keys(item, ["id", "version", "forecasts"], where="memory_forecast")
        text(item["id"], "memory_forecast.id")
        if type(item["version"]) is not int:
            raise LLMError("memory_forecast.version: expected integer")
        key = item["id"], item["version"]
        if key not in known or key not in visible or key in seen:
            raise LLMError("memory_forecasts: unknown, unseen, stale or duplicate memory")
        seen.add(key)
        validate_panel(item["forecasts"], allowed, complete=False)
    return response


def call(engine, payload):
    """Persist every actual call, including failed/unknown-cost calls, for field trials."""
    from .storage import append_bytes, locked
    started = time.monotonic()
    error = None
    try:
        response = engine.llm.complete(deepcopy(payload))
    except Exception as exc:
        error = exc
        response = None
    usage = deepcopy(getattr(engine.llm, "last_usage", {})) if error is None else {}
    usage["adapter_seconds"] = usage.get("seconds")
    usage["seconds"] = time.monotonic() - started
    usage.setdefault("cost_usd", None)
    usage.setdefault("tokens", None)
    usage.setdefault("models", [])
    number(usage["seconds"], "LLM usage.seconds", nonnegative=True)
    if usage["cost_usd"] is not None:
        number(usage["cost_usd"], "LLM usage.cost_usd", nonnegative=True)
    record = {"request_hash": digest(payload), "at_utc": datetime.now(timezone.utc).isoformat(), "kind": payload["kind"], "phase": payload.get("phase"),
              "status": "failed" if error else "completed", "backend": str(getattr(engine.llm, "identity", type(engine.llm).__name__)), **usage}
    with locked(engine.config):
        append_bytes(engine.config.state_root / "inference.jsonl", canonical(record) + b"\n")
    engine.last_inference_usage = record
    if error is not None:
        raise LLMError(f"LLM inference failed ({type(error).__name__}): {str(error)[:300]}") from error
    return response, record


def external_submission(engine, payload):
    """A supplied forecast/update is not a measured, free model call."""
    from .storage import append_bytes, locked
    record = {"request_hash": digest(payload), "at_utc": datetime.now(timezone.utc).isoformat(),
              "kind": payload["kind"], "phase": "calling-agent", "status": "completed",
              "backend": "calling-agent", "external": True, "seconds": None,
              "cost_usd": None, "tokens": None, "models": []}
    with locked(engine.config):
        append_bytes(engine.config.state_root / "inference.jsonl", canonical(record) + b"\n")
    return record


def complete(engine, payload, snapshot):
    identity = str(getattr(engine.llm, "identity", type(engine.llm).__name__))
    cache_key = digest({"format": FORMAT, "backend": identity, "settings": SETTINGS, "request": payload})
    path = engine.config.state_root / "llm-cache" / f"{cache_key}.json"
    if path.is_file():
        try:
            cached = json.loads(path.read_bytes())
            integrity = cached.pop("integrity")
            if digest(cached) != integrity:
                raise ValueError("cache integrity mismatch")
            if cached["request"] == payload:
                validate_response(cached["response"], cached["final_request"], snapshot)
                return cached["response"], cached["final_request"], {**cached["usage"], "cached": True, "call_seconds": 0.0, "call_cost_usd": 0.0}
        except (ValueError, KeyError, TypeError, HowHardError):
            pass  # Replace corrupt inference caches; never use them as evidence.
    original, current = deepcopy(payload), deepcopy(payload)
    seconds, cost, cost_known = 0.0, 0.0, True
    calls = []
    # A genuine no-memory forecast must be a separate stateless call, without
    # historical evidence or beliefs.
    baseline_response, usage = call(engine, baseline_request(payload))
    keys(baseline_response, ["forecasts", "reason"], ["memory_forecasts"], "LLM baseline")
    if baseline_response.get("memory_forecasts"):
        raise LLMError("LLM baseline cannot reference memories")
    validate_panel(baseline_response["forecasts"], {(c["family"], c["effort"]) for c in payload["cells"]})
    text(baseline_response["reason"], "LLM baseline.reason")
    current.update(phase="memory", no_memory_forecasts=deepcopy(baseline_response["forecasts"]))
    current["contract"]["forecasts"] = "Omit this field. The isolated no_memory_forecasts are fixed and must not be changed."
    calls.append(usage)
    for round_number in range(SETTINGS["retrieval_rounds"] + 1):
        if snapshot["memories"]:
            response, usage = call(engine, current)
            calls.append(usage)
        else:
            response = {"memory_forecasts": [], "reason": baseline_response["reason"]}
        if isinstance(response, dict) and "request_more" in response:
            keys(response, ["request_more"], where="LLM response")
            more = response["request_more"]
            if round_number == SETTINGS["retrieval_rounds"] or not isinstance(more, list) or not 1 <= len(more) <= 3:
                raise LLMError("LLM retrieval budget exceeded or invalid requests")
            for item in more:
                current["retrieval"].append({"request": deepcopy(item), "result": tool(engine, item, project_snapshot=current["context"]["project_context"])})
            current["tools"] = [] if round_number + 1 == SETTINGS["retrieval_rounds"] else current["tools"]
            continue
        if not isinstance(response, dict):
            raise LLMError("LLM memory response must be an object")
        response = deepcopy(response)
        if "forecasts" in response and response["forecasts"] != current["no_memory_forecasts"]:
            raise LLMError("Memory reader must not rewrite the isolated baseline forecasts")
        response["forecasts"] = deepcopy(current["no_memory_forecasts"])
        response = validate_response(response, current, snapshot)
        if any(item["request"]["tool"] == "project.read" for item in current["retrieval"]):
            from .context import capture_project
            expected = current["context"]["project_context"]
            if capture_project(engine.config, expected["parameters"])["fingerprint"] != expected["fingerprint"]:
                raise LLMError("Project changed during retrieval; start a fresh decision")
        seconds = sum(c["seconds"] for c in calls)
        cost_known = all(c["cost_usd"] is not None for c in calls)
        cost = sum(c["cost_usd"] for c in calls) if cost_known else None
        usage = {"seconds": seconds, "cost_usd": cost, "backend": identity, "calls": calls}
        cached = {"request": original, "final_request": current, "response": response, "usage": usage}
        cached["integrity"] = digest(cached)
        atomic_write(path, canonical(cached) + b"\n")
        return response, current, {**usage, "cached": False, "call_seconds": seconds, "call_cost_usd": usage["cost_usd"]}
    raise LLMError("LLM did not return forecasts")


def augment(engine, prepared, response, payload, snapshot, usage):
    result, cells = deepcopy(prepared[0]), prepared[1]
    baseline = deepcopy(response["forecasts"])
    worlds = {"llm": baseline}
    entries = {(m["id"], m["version"]): m for m in snapshot["memories"]}
    exposures = {}
    llm_indexed = panel_lookup(response["forecasts"])
    for item in response["memory_forecasts"]:
        panel = panel_lookup(deepcopy(baseline))
        changed = [c for c in item["forecasts"] if c != llm_indexed[c["family"], c["effort"]]]
        panel.update(panel_lookup(deepcopy(changed)))
        key = memory_key(entries[item["id"], item["version"]])
        worlds[key] = list(panel.values())
        exposures[key] = [[c["family"], c["effort"]] for c in changed]
    state = payload["beliefs"]
    weights = {e["key"]: e["weight"] for e in state["experts"]}
    # Unretrieved memories are sleeping experts: retain their mass on the independent LLM
    # baseline, and do not award them evidence for forecasts they never made.
    indexed = {key: panel_lookup(panel) for key, panel in worlds.items()}
    indexed.update({key: indexed["llm"] for key in weights if key not in indexed})
    for row in result["cells"]:
        pair = row["family"], row["effort"]
        for outcome in OUTCOMES:
            values = sorted((indexed[k][pair][outcome], weights[k]) for k in weights)
            def quantile(fraction):
                total = 0.0
                for value, mass in values:
                    total += mass
                    if total >= fraction:
                        return value
                return values[-1][0]
            row[f"P_{outcome}"] = {}
            row[f"P_{outcome}"]["mean"] = sum(weights[k] * indexed[k][pair][outcome] for k in weights)
            row[f"P_{outcome}"]["lower"] = min(quantile(.025), row[f"P_{outcome}"]["mean"])
            row[f"P_{outcome}"]["upper"] = max(quantile(.975), row[f"P_{outcome}"]["mean"])
        tokens = sum(weights[k] * indexed[k][pair]["tokens"] for k in weights)
        seconds = sum(weights[k] * indexed[k][pair]["seconds"] for k in weights)
        cell = next(c for c in cells if (c["family"], c["effort"]) == pair)
        row["C_run"] = tokens / 1e6 * cell["price"] if cell["price"] is not None else None
        row["T"] = seconds / 3600
        c = result["constants"]
        row["J"] = tokens / 1e6 * cell["comparison_price"] + c["C_loud"] * row["P_loud"]["mean"] + c["C_silent"] * row["P_silent"]["mean"] + c["lambda_per_hour"] * row["T"]
        row["admissible"] = row["P_silent"]["upper"] <= c["epsilon"]
    result["cells"].sort(key=lambda row: (row["C_run"] if row["C_run"] is not None else float("inf"), row["family"], row["effort"]))
    result["semantic"] = {"format": FORMAT, "prompt_version": SETTINGS["prompt_version"], "backend": usage["backend"],
                          "memory_revision": snapshot["revision"], "memories": deepcopy(snapshot["memories"]),
                          "worlds": worlds, "exposures": exposures, "beliefs": state, "reason": response["reason"], "retrieval": payload["retrieval"],
                          "evidence": deepcopy(payload["evidence"]),
                          "reader_usage": {k: usage[k] for k in ("backend", "seconds", "cost_usd", "calls")}, "interval_kind": "weighted-forecast-envelope"}
    result["backend"] = {"mode": "semantic", "name": usage["backend"], "inference_cached": usage["cached"]}
    result["routing_overhead"] = {"seconds": usage["call_seconds"], "cost_usd": usage["call_cost_usd"]}
    return result, cells


def sample(prepared, seed):
    result, cells = prepared
    state = result["semantic"]
    experts = state["beliefs"]["experts"]
    rng = np.random.default_rng(seed)
    chosen = str(rng.choice([e["key"] for e in experts], p=[e["weight"] for e in experts]))
    state["sampled_expert"] = chosen
    panel = panel_lookup(state["worlds"].get(chosen, state["worlds"]["llm"]))
    c = result["constants"]
    parts = {k: np.zeros((1, len(cells))) for k in ("run", "loud", "silent", "time")}
    for i, cell in enumerate(cells):
        row = panel[cell["family"], cell["effort"]]
        p = rng.dirichlet(np.maximum([row[k] for k in OUTCOMES], SETTINGS["belief"]["probability_floor"]) * SETTINGS["belief"]["sample_concentration"])
        parts["run"][0, i] = row["tokens"] / 1e6 * cell["comparison_price"]
        parts["loud"][0, i] = p[1] * c["C_loud"]
        parts["silent"][0, i] = p[2] * c["C_silent"]
        parts["time"][0, i] = row["seconds"] / 3600 * c["lambda_per_hour"]
    return parts


def validate_receipt(state):
    """Validate portable frozen semantic inputs before they affect feedback weights."""
    keys(state, ["format", "prompt_version", "backend", "memory_revision", "memories", "worlds", "exposures", "beliefs", "reason", "retrieval", "evidence", "reader_usage", "interval_kind"], ["sampled_expert"], "decision.semantic")
    if state["format"] != FORMAT or state["prompt_version"] != SETTINGS["prompt_version"]:
        raise HowHardError("decision.semantic: unsupported forecast format or prompt version")
    if type(state["memory_revision"]) is not int or state["memory_revision"] < 0:
        raise HowHardError("decision.semantic.memory_revision: expected non-negative integer")
    if state["interval_kind"] != "weighted-forecast-envelope":
        raise HowHardError("decision.semantic.interval_kind: unsupported forecast envelope")
    text(state["backend"], "decision.semantic.backend")
    text(state["reason"], "decision.semantic.reason")
    if not isinstance(state["worlds"], dict) or not {"llm"} <= state["worlds"].keys():
        raise HowHardError("decision.semantic.worlds: the llm panel is required")
    baseline = state["worlds"]["llm"]
    if not isinstance(baseline, list) or not baseline:
        raise HowHardError("decision.semantic.worlds: nonempty panels required")
    allowed = set()
    for row in baseline:
        keys(row, FIELDS, where="decision.semantic.forecast")
        text(row["family"], "forecast.family")
        text(row["effort"], "forecast.effort")
        allowed.add((row["family"], row["effort"]))
    if not isinstance(state["memories"], list):
        raise HowHardError("decision.semantic.memories: expected array")
    memory_keys = set()
    for entry in state["memories"]:
        keys(entry, ["id", "version", "active", "title", "body", "scope", "evidence_ids", "parents"], where="decision.semantic.memory")
        for field in ("id", "title", "body"):
            text(entry[field], f"decision.semantic.memory.{field}")
        if type(entry["version"]) is not int or entry["version"] < 1 or type(entry["active"]) is not bool:
            raise HowHardError("decision.semantic.memory: invalid version or active flag")
        key = memory_key(entry)
        if key in memory_keys:
            raise HowHardError("decision.semantic.memory: duplicate memory version")
        memory_keys.add(key)
    for key, panel in state["worlds"].items():
        if key not in {"llm"} | memory_keys:
            raise HowHardError("decision.semantic.worlds: unknown memory version")
        validate_panel(panel, allowed)
    if not isinstance(state["exposures"], dict):
        raise HowHardError("decision.semantic.exposures: expected object")
    for key, pairs in state["exposures"].items():
        if key not in memory_keys or key not in state["worlds"] or not isinstance(pairs, list):
            raise HowHardError("decision.semantic.exposures: unknown memory or invalid cells")
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(v, str) for v in pair) or tuple(pair) not in allowed:
                raise HowHardError("decision.semantic.exposures: unknown cell")
    belief = state["beliefs"]
    keys(belief, ["revision", "kind", "unscored_runs", "experts"], where="decision.semantic.beliefs")
    if type(belief["revision"]) is not int or belief["revision"] != state["memory_revision"] or not isinstance(belief["experts"], list):
        raise HowHardError("decision.semantic.beliefs: inconsistent revision or invalid experts")
    seen, mass = set(), 0.0
    for expert in belief["experts"]:
        keys(expert, ["key", "weight", "log_score", "evidence_count"], where="decision.semantic.expert")
        key = text(expert["key"], "decision.semantic.expert.key")
        if key in seen or key not in {"llm"} | memory_keys:
            raise HowHardError("decision.semantic.expert: duplicate or unknown expert")
        seen.add(key)
        mass += number(expert["weight"], "expert.weight", nonnegative=True)
        number(expert["log_score"], "expert.log_score")
        if type(expert["evidence_count"]) is not int or expert["evidence_count"] < 0:
            raise HowHardError("decision.semantic.expert.evidence_count: expected non-negative integer")
    if not {"llm"} <= seen or abs(mass - 1) > 1e-6 or set(state["worlds"]) - seen:
        raise HowHardError("decision.semantic.beliefs: expert weights must cover forecasts and sum to one")
    if "sampled_expert" in state and state["sampled_expert"] not in seen:
        raise HowHardError("decision.semantic.sampled_expert: unknown expert")
    if not isinstance(state["retrieval"], list) or not isinstance(state["evidence"], list):
        raise HowHardError("decision.semantic: retrieval and evidence must be arrays")
    usage = state["reader_usage"]
    keys(usage, ["backend", "seconds", "cost_usd", "calls"], where="decision.semantic.reader_usage")
    number(usage["seconds"], "reader_usage.seconds", nonnegative=True)
    if usage["cost_usd"] is not None:
        number(usage["cost_usd"], "reader_usage.cost_usd", nonnegative=True)
    if not isinstance(usage["calls"], list):
        raise HowHardError("reader_usage.calls: expected array")
    return state


def learning_event_key(row, raw_rows):
    roots, latest = {}, row["id"]
    for raw in raw_rows:
        root = roots[raw["amends"]] if raw["amends"] else raw["id"]
        roots[raw["id"]] = root
        if root == row["id"]:
            latest = raw["id"]
    payload = {"writer": SETTINGS["writer_version"], "row": row}
    # Keep original learning keys stable; each appended correction is a new event,
    # including a correction that returns to a previously learned classification.
    if latest != row["id"]:
        payload["amendment_id"] = latest
    return digest(payload)


def learning_request(engine, evidence_id):
    row = tool(engine, {"tool": "evidence.read", "id": evidence_id}, allow_project=False)
    event_key = learning_event_key(row, engine.raw_rows)
    snapshot = engine.memory.snapshot(include_removed=True)
    receipt = None
    if row.get("decision_id"):
        from .decisions import load, receipt_path
        if receipt_path(engine.config, row["decision_id"]).is_file():
            receipt = load(engine.config, row["decision_id"], for_scoring=True)
    request = {"kind": "learn", "prompt_version": SETTINGS["writer_version"], "observation": row,
               "decision": receipt, "memory_revision": snapshot["revision"], "memories": snapshot["memories"],
               "beliefs": beliefs(engine), "evidence": evidence_search(engine, row.get("task_text", "")),
               "contract": "Return ONLY {operations: [...], reason: string}. Operations: add(title,body,scope?,evidence_ids?); edit(id,title?/body?/scope?/evidence_ids?,expected_version?); remove(id); restore(id); merge(ids,title,body); split(id,parts:[{title,body}]). Every item includes op. Use actual evidence IDs. Hypotheses are provisional; evidence that inspired a new memory does not validate its forecasts. Prefer targeted useful changes; an empty list is valid. Never delete raw run evidence."}
    roots, history = {}, []
    for raw in engine.raw_rows:
        roots[raw["id"]] = roots[raw["amends"]] if raw["amends"] else raw["id"]
        if roots[raw["id"]] == row["id"]:
            history.append(deepcopy(raw))
    request["outcome_history"] = history
    request["contract"] += " Use observation for current effective outcome and usage. outcome_history is an audit trail: measurement_corrections replace values, never add runs or usage."
    return request, event_key


def apply_learning(engine, evidence_id, response, *, expected_revision, event_key=None, usage=None):
    current, actual_key = learning_request(engine, evidence_id)
    if event_key is not None and actual_key != event_key:
        raise HowHardError("memory learning: evidence changed since the writer request")
    keys(response, ["operations", "reason"], where="LLM memory update")
    text(response["reason"], "LLM memory update.reason")
    if engine.memory.learned(actual_key):
        return {"status": "already-learned", "evidence_id": current["observation"]["id"]}
    if usage is None:
        usage = external_submission(engine, current)
    from .storage import locked, snapshot, effective_rows
    with locked(engine.config):
        current_rows, _ = snapshot(engine.config)
        root_id = current["observation"]["id"]
        latest = next((row for row in effective_rows(engine.config, current_rows) if row["id"] == root_id), None)
        if latest is None or learning_event_key(latest, current_rows) != actual_key:
            raise HowHardError("memory learning: evidence changed during the writer call; refresh and retry")
        results = engine.memory.apply(response["operations"], expected_revision=expected_revision, actor="llm", event_key=actual_key)
    return {"status": "updated", "evidence_id": current["observation"]["id"], "operations": results, "reason": response["reason"], "usage": usage}


def learn(engine, evidence_id):
    if engine.llm is None:
        return {"status": "pending", "evidence_id": evidence_id, "reason": "Supply a calling-agent writer response or run memory learn."}
    request, event_key = learning_request(engine, evidence_id)
    if engine.memory.learned(event_key):
        return {"status": "already-learned", "evidence_id": request["observation"]["id"]}
    response, usage = call(engine, request)
    return apply_learning(engine, evidence_id, response, expected_revision=request["memory_revision"], event_key=event_key, usage=usage)
