"""LLM reasoning over semantic memories with approximate predictive-evidence weights."""

from copy import deepcopy
import json
import math
from pathlib import Path
import re
import time
from datetime import datetime, timezone

import numpy as np

from .context import DECISION_FORMAT, IGNORED, LEGACY_DECISION_V7, LEGACY_DECISION_V8, LEGACY_DECISION_V9, LEGACY_DECISION_V10, MANIFESTS, SOURCE_SUFFIXES, canonical, digest
from .llm import LLMError, SETTINGS
from .files import atomic_write
from .storage import effective_active_seconds
from .validation import HowHardError, keys, number, text

FORMAT = SETTINGS["format"]
LEGACY_FORMAT_V4 = "howhard-semantic-v4"
LEGACY_PROMPT_V5 = "semantic-reader-v5"
# Frozen receipt versions pin the semantic format and reader prompt they were produced with.
RECEIPT_SEMANTICS = {LEGACY_DECISION_V7: (LEGACY_FORMAT_V4, LEGACY_PROMPT_V5),
                     LEGACY_DECISION_V8: ("howhard-semantic-v5", "semantic-reader-v6"),
                     LEGACY_DECISION_V9: ("howhard-semantic-v6", "semantic-reader-v7"),
                     LEGACY_DECISION_V10: ("howhard-semantic-v7", "semantic-reader-v7"),
                     DECISION_FORMAT: (FORMAT, SETTINGS["prompt_version"])}
OUTCOMES = ("clean", "loud", "silent")
FIELDS = ("family", "effort", "clean", "loud", "silent", "tokens", "seconds")
# Stated 95% ranges on the two failure rates, and on active time and each token kind.
INTERVAL_FIELDS = ("loud_ci", "silent_ci")
RANGE_FIELDS = ("seconds_ci", "token_usage_ci")
INTERVAL_KIND = "dirichlet-posterior-mixture"
LEGACY_INTERVAL_KIND = "weighted-forecast-envelope"
INTERVAL_CONTRACT = ("Every range is optional and is a central 95% range [lo, hi] that contains its value with "
                     "0 <= lo <= value <= hi. loud_ci and silent_ci: ranges on those outcome probabilities, also "
                     "with hi <= 1. A range of [0, 1], or one wider than [0.025, 0.975], states a flat belief: no "
                     "knowledge of that probability; there is no other limit on how wide a range may be. Supply them when "
                     "unsure: wider ranges weaken the forecast so observed outcomes dominate sooner; a narrow "
                     "range cannot make a forecast stronger than the bounded prior strength. seconds_ci: range "
                     "on seconds. token_usage_ci: {kind: [lo, hi]} for any of input/output/cache_read/cache_write/"
                     "cache_write_5m/cache_write_1h, only on a row that has token_usage; a kind forecast at 0 is "
                     "exactly 0 unless you state a non-zero hi. There is no range on tokens or on cost. Omit a "
                     "range to state an ordinary-confidence point forecast for that value.")


def memory_key(entry):
    return f"memory:{entry['id']}:{entry['version']}"


def panel_lookup(panel):
    return {(c["family"], c["effort"]): c for c in panel}


def execution_memory(row):
    """A read-only expert backed by an effective observed execution, never an invented run."""
    summary = {k: row.get(k) for k in (
        "task_text", "model", "effort", "outcome", "cause", "tokens", "seconds", "cost_usd", "execution_usage", "note")}
    for extra in ("active_seconds", "queue_wait_seconds"):
        if extra in row:
            summary[extra] = row[extra]
    return {"id": "execution:" + row["id"], "version": int(digest(row)[:12], 16) + 1,
            "active": True, "title": "Observed execution: " + row["task_type"],
            "body": "Observed execution (read-only): " + json.dumps(summary, ensure_ascii=True),
            "scope": "project", "evidence_ids": [row["id"]], "parents": []}


def memory_snapshot(engine):
    snapshot = engine.memory.snapshot()
    snapshot["memories"].extend(execution_memory(row) for row in engine.rows)
    return snapshot


def beliefs(engine, snapshot=None):
    """Replay effective outcomes; only pre-recorded forecasts earn predictive support."""
    from .decisions import load, exists
    snapshot = memory_snapshot(engine) if snapshot is None else snapshot
    priors = SETTINGS["belief"]
    entries = {"llm": priors["llm_prior"],
               **{memory_key(m): priors["memory_prior"] for m in snapshot["memories"]},
               "filter:llm": priors["llm_prior"],
               **{"filter:"+memory_key(m): priors["memory_prior"] for m in snapshot["memories"]}}
    evidence = {key: [] for key in entries}
    unscored = 0
    latest = {}
    roots = {}
    for index, raw in enumerate(engine.raw_rows):
        roots[raw["id"]] = roots[raw["amends"]] if raw["amends"] else raw["id"]
        latest[roots[raw["id"]]] = index
    for row in sorted(engine.rows, key=lambda r: latest.get(r["id"], -1)):
        if row["outcome"] == "interrupted":
            continue
        decision_id = row.get("decision_id")
        if not decision_id:
            continue
        if not exists(engine.config, decision_id):
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
                for field, measurement in (("tokens", row["tokens"]), ("seconds", effective_active_seconds(row))):
                    if measurement is not None:
                        observed = math.log(max(1, measurement))
                        residual = observed - math.log(max(1, candidate[field]))
                        base_residual = observed - math.log(max(1, reference[field]))
                        score += priors["resource_weight"] * (base_residual ** 2 - residual ** 2) / (2 * priors["resource_noise_sd"] ** 2)
                evidence[key].append(row["weight"] * score)
        from .capability import cell_key
        actual = cell_key(row)
        for key, probabilities in receipt["capability"]["worlds"].items():
            if key not in entries:
                continue
            p = probabilities[actual]
            if key != "filter:llm" and p == receipt["capability"]["worlds"]["filter:llm"][actual]:
                continue
            reference = panel_lookup(state["worlds"]["llm"])[cell]["clean"]
            p, reference = (p, reference) if row["outcome"] == "clean" else (1-p, 1-reference)
            evidence[key].append(row["weight"] * (math.log(max(floor, p))-math.log(max(floor, reference))))
    scores = {key: math.log(prior) + priors["temperature"] * math.fsum(value * priors["recency"] ** age for age, value in enumerate(reversed(evidence[key]))) for key, prior in entries.items()}
    peak = max(scores.values())
    mass = {key: math.exp(score - peak) for key, score in scores.items()}
    total = math.fsum(mass.values())
    return {"revision": snapshot["revision"], "kind": "tempered-predictive-evidence", "unscored_runs": unscored,
            "experts": [{"key": key, "weight": mass[key] / total, "log_score": scores[key], "evidence_count": len(evidence[key])} for key in sorted(entries)]}


def evidence_search(engine, query="", limit=None):
    from .retrieval import records
    result = engine.retrieval.search(records(rows=engine.rows), query, limit=limit or SETTINGS["evidence_limit"])
    return [r["value"] for r in result["records"]]


def tool(engine, request, *, allow_project=True, project_snapshot=None):
    keys(request, ["tool"], ["query", "id", "path"], "LLM retrieval")
    name = request["tool"]
    if name == "memory.search":
        keys(request, ["tool", "query"], where="LLM retrieval")
        return engine.memory.search(text(request["query"], "retrieval.query"))
    if name == "memory.read":
        keys(request, ["tool", "id"], where="LLM retrieval")
        for row in engine.rows:
            if request["id"] == "execution:" + row["id"]:
                return execution_memory(row)
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
    snapshot = memory_snapshot(engine)
    state = beliefs(engine, snapshot)
    from .retrieval import records, task_text
    editable = [m for m in snapshot["memories"] if not m["id"].startswith("execution:")]
    retrieved = engine.retrieval.search(records(engine.rows, editable), task_text(result["task_type"], result["context"]))
    engine.last_retrieval_usage = {"seconds": retrieved["trace"].pop("seconds")}
    memories = [r["value"] for r in retrieved["records"] if r["kind"] == "memory"]
    evidence = [r["value"] for r in retrieved["records"] if r["kind"] == "evidence"]
    memories.extend(execution_memory(row) for row in evidence)
    category_cells = {(n["model"], n["effort"]) for n in result["native_candidates"] if "rate_card" in n}
    visible_cells = []
    for cell in result["cells"]:
        visible_cell = deepcopy(cell)
        if (cell["model"], cell["effort"]) in category_cells:
            visible_cell["forecast_format"] = "category-v1"
        visible_cells.append(visible_cell)
    payload = {"kind": "predict", "prompt_version": SETTINGS["prompt_version"], "task_type": result["task_type"],
               "context": deepcopy(result["context"]), "risk_class": result["risk_class"], "constants": result["constants"],
               "cells": visible_cells, "memory_revision": snapshot["revision"], "memories": memories,
               "beliefs": state, "evidence": evidence,
               "retrieval": [{"request": {"tool": "experience.search", "query": task_text(result["task_type"], result["context"])}, "result": retrieved["trace"]}],
               "tools": ["memory.search", "memory.read", "evidence.search", "evidence.read", "project.read"],
               "contract": {"forecasts": "One row for EVERY supplied cell: family, effort, clean, loud, silent, tokens, seconds, plus optional loud_ci/silent_ci/seconds_ci/token_usage_ci (see credible_intervals). Probabilities sum to 1; usage estimates are nonnegative; seconds is expected active execution duration excluding external queue wait. category-v1 cells also REQUIRE token_usage with disjoint input/output/cache_read/cache_write/cache_write_5m/cache_write_1h summing to tokens; total-v1 accepts the scalar tokens forecast.",
                            "credible_intervals": INTERVAL_CONTRACT,
                            "memory_forecasts": "Array of {id, version, forecasts}: predictions conditional on each relevant supplied/read ACTIVE memory. execution: IDs are read-only observed runs; their original outcomes are evidence for related tasks, not counterfactual results. Interrupted runs (such as usage_limit, provider_error, host_error or cancelled) are operational interruptions, not model task failures. Infer model AND reasoning effort jointly. Broader/narrower tasks can transfer imperfectly. Rows may override a subset of baseline forecasts. No invented memories or unrun outcomes.",
                            "reason": "Brief explanation grounded in task/project semantics and actual evidence.",
                            "request_more": "Instead of forecasts, optionally return ONLY {request_more: [{tool, query/id/path}]} to retrieve more evidence. At most 3 requests per round."}}
    if "cost_policy" in result:
        payload["cost_policy"] = deepcopy(result["cost_policy"])
    return payload, snapshot


def visible_request(payload):
    """Keep unselected expert bookkeeping out of the model's context."""
    result = deepcopy(payload)
    for internal in ("prototype_context", "preset_forecasts", "unseen_cells"):
        result.pop(internal, None)
    prefix = "filter:" if payload["kind"] == "filter" else ""
    visible = {prefix+"llm", *(f"{prefix}memory:{id}:{version}" for id, version in visible_memories(payload))}
    result["beliefs"]["experts"] = [e for e in result["beliefs"]["experts"] if e["key"] in visible]
    normalize_support(result["beliefs"]["experts"])
    return result


def visible_memories(payload):
    visible = {(m["id"], m["version"]) for m in payload["memories"]}
    for item in payload["retrieval"]:
        name, data = item["request"]["tool"], item["result"]
        if name.startswith("memory.") or name in ("evidence.search", "evidence.read"):
            for entry in data if isinstance(data, list) else [data]:
                if not isinstance(entry, dict):
                    continue
                if name.startswith("evidence."):
                    entry = execution_memory(entry)
                if entry.get("active"):
                    visible.add((entry["id"], entry["version"]))
    return visible


def normalize_support(experts):
    # Recompute from log scores: globally dominant but irrelevant experts can
    # underflow all participating global weights to zero.
    peak = max(e["log_score"] for e in experts)
    mass = [math.exp(e["log_score"] - peak) for e in experts]
    total = math.fsum(mass)
    for expert, value in zip(experts, mass):
        expert["weight"] = value / total


def baseline_request(payload):
    if payload["kind"] == "filter":
        from .capability import baseline_request as filter_baseline
        return filter_baseline(payload)
    if "no_memory_forecasts" in payload:
        return None
    clean = {k: deepcopy(payload[k]) for k in ("kind", "prompt_version", "task_type", "risk_class", "constants")}
    clean["context"] = deepcopy(payload.get("prototype_context", payload["context"]))
    if "cost_policy" in payload:
        clean["cost_policy"] = deepcopy(payload["cost_policy"])
    source_cells = payload.get("unseen_cells", payload["cells"])
    clean.update(phase="baseline", cells=[{k: c[k] for k in ("family", "effort", "model", "forecast_format") if k in c} for c in source_cells],
                 contract="Return {forecasts: [{family,effort,clean,loud,silent,tokens,seconds,loud_ci?,silent_ci?,seconds_ci?,token_usage_ci?}], reason: string}. Forecast EVERY supplied cell; probabilities sum to 1; seconds is expected active execution duration excluding external queue wait. category-v1 cells require token_usage with disjoint input/output/cache_read/cache_write/cache_write_5m/cache_write_1h summing to tokens; total-v1 accepts scalar tokens. " + INTERVAL_CONTRACT + " No memory, evidence or retrieval is available in this independent call.")
    return clean


def stated_range(value, where):
    if not isinstance(value, list) or len(value) != 2:
        raise LLMError(f"{where}: expected [lo, hi]")
    lo, hi = (number(v, where, nonnegative=True) for v in value)
    return lo, hi


def validate_panel(panel, allowed, *, complete=True, intervals=True, ranges=True):
    if not isinstance(panel, list) or not panel:
        raise LLMError("LLM forecasts: expected a nonempty array")
    seen = set()
    for row in panel:
        keys(row, FIELDS, ["token_usage", *(INTERVAL_FIELDS if intervals else ()), *(RANGE_FIELDS if ranges else ())],
             where="LLM forecast")
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
        for field in INTERVAL_FIELDS:
            if field not in row:
                continue
            lo, hi = stated_range(row[field], f"forecast.{field}")
            value = row[field[:-3]]
            if not lo <= value <= hi <= 1:
                raise LLMError(f"forecast.{field}: require 0 <= lo <= {field[:-3]} <= hi <= 1")
        if "seconds_ci" in row:
            lo, hi = stated_range(row["seconds_ci"], "forecast.seconds_ci")
            if not lo <= row["seconds"] <= hi:
                raise LLMError("forecast.seconds_ci: require 0 <= lo <= seconds <= hi")
        if "token_usage" in row:
            from .accounting import tokens, total
            tokens(row["token_usage"], forecast=True)
            if abs(total(row["token_usage"]) - row["tokens"]) > 1e-6:
                raise LLMError("forecast.token_usage: categories must sum to tokens")
        if "token_usage_ci" in row:
            from .accounting import CATEGORIES
            # Ranges exist per token kind only: total tokens and cost follow from the kinds.
            if "token_usage" not in row:
                raise LLMError("forecast.token_usage_ci: requires token_usage; there is no range on total tokens")
            keys(row["token_usage_ci"], [], CATEGORIES, where="forecast.token_usage_ci")
            for kind, interval in row["token_usage_ci"].items():
                lo, hi = stated_range(interval, f"forecast.token_usage_ci.{kind}")
                if not lo <= row["token_usage"][kind] <= hi:
                    raise LLMError(f"forecast.token_usage_ci.{kind}: require 0 <= lo <= token_usage.{kind} <= hi")
    if complete and seen != set(allowed):
        raise LLMError("LLM forecasts: missing candidate cells")
    return panel


def validate_response(response, payload, snapshot):
    if payload["kind"] == "filter":
        from .capability import validate_response as validate_filter_response
        return validate_filter_response(response, payload, snapshot)
    allowed = {(c["family"], c["effort"]) for c in payload["cells"]}
    category_cells = {(c["family"], c["effort"]) for c in payload["cells"] if c.get("forecast_format") == "category-v1"}
    if "no_memory_forecasts" in payload:
        keys(response, ["memory_forecasts", "reason"], ["forecasts"], where="LLM response")
        if "forecasts" in response and response["forecasts"] != payload["no_memory_forecasts"]:
            raise LLMError("Memory reader must not rewrite the isolated baseline forecasts")
        response = {**deepcopy(response), "forecasts": deepcopy(payload["no_memory_forecasts"])}
        text(response["reason"], "LLM response.reason")
        validate_panel(response["forecasts"], allowed)
    elif "preset_forecasts" in payload and "unseen_cells" in payload:
        keys(response, ["forecasts", "memory_forecasts", "reason"], where="LLM response")
        text(response["reason"], "LLM response.reason")
        unseen_allowed = {(c["family"], c["effort"]) for c in payload["unseen_cells"]}
        unseen_cat = {(c["family"], c["effort"]) for c in payload["unseen_cells"] if c.get("forecast_format") == "category-v1"}
        if {(r.get("family"), r.get("effort")) for r in response["forecasts"] if isinstance(r, dict)} == allowed:
            validate_panel(response["forecasts"], allowed)
            merged = deepcopy(response["forecasts"])
        else:
            validate_panel(response["forecasts"], unseen_allowed)
            if any((r["family"], r["effort"]) in unseen_cat and "token_usage" not in r for r in response["forecasts"]):
                raise LLMError("forecast.token_usage: category rate card requires category forecast")
            merged = deepcopy(payload["preset_forecasts"]) + deepcopy(response["forecasts"])
            validate_panel(merged, allowed)
        response = {**deepcopy(response), "forecasts": merged}
    else:
        keys(response, ["forecasts", "memory_forecasts", "reason"], where="LLM response")
        text(response["reason"], "LLM response.reason")
        validate_panel(response["forecasts"], allowed)
    if any((r["family"], r["effort"]) in category_cells and "token_usage" not in r for r in response["forecasts"]):
        raise LLMError("forecast.token_usage: category rate card requires category forecast")
    if not isinstance(response["memory_forecasts"], list) or len(response["memory_forecasts"]) > SETTINGS["memory_limit"]:
        raise LLMError("memory_forecasts: expected an array with at most eight entries")
    known = {(m["id"], m["version"]): m for m in snapshot["memories"]}
    visible = visible_memories(payload)
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
        if any((r["family"], r["effort"]) in category_cells and "token_usage" not in r for r in item["forecasts"]):
            raise LLMError("forecast.token_usage: category override required")
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
    with locked(engine.config, operation="inference event"):
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
    with locked(engine.config, operation="external inference event"):
        append_bytes(engine.config.state_root / "inference.jsonl", canonical(record) + b"\n")
    return record


def combine_usage(earlier, current):
    """Combine measured reader calls while preserving unknown costs and cache status."""
    if earlier is None:
        return deepcopy(current)
    result = {**current, "calls": earlier["calls"]+current["calls"],
              "cached": earlier["cached"] and current["cached"]}
    for field in ("seconds", "call_seconds", "cost_usd", "call_cost_usd"):
        result[field] = None if earlier[field] is None or current[field] is None else earlier[field]+current[field]
    return result


def _cache_read(path, expected_request, snapshot):
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_bytes())
        integrity = cached.pop("integrity")
        if digest(cached) != integrity:
            raise ValueError("cache integrity mismatch")
        if cached["request"] == expected_request:
            validate_response(cached["response"], cached["final_request"], snapshot)
            return (
                cached["response"],
                cached["final_request"],
                {**cached["usage"], "cached": True, "call_seconds": 0.0, "call_cost_usd": 0.0},
            )
    except (ValueError, KeyError, TypeError, HowHardError):
        pass  # Replace corrupt inference caches; never use them as evidence.
    return None


def _cache_write(engine, path, request_obj, final_request, response, usage):
    from .storage import locked
    cached = {"request": request_obj, "final_request": final_request, "response": response, "usage": usage}
    cached["integrity"] = digest(cached)
    with locked(engine.config, operation="inference cache"):
        atomic_write(path, canonical(cached) + b"\n")


def _replay_signature(payload, snapshot):
    cells_sig = [
        c["key"] if "key" in c else [c["family"], c["effort"], c.get("model"), c.get("forecast_format")]
        for c in payload["cells"]
    ]
    return {
        "kind": payload["kind"],
        "prompt_version": payload.get("prompt_version"),
        "task_type": payload["task_type"],
        "risk_class": payload["risk_class"],
        "context": payload["context"],
        "cells": cells_sig,
        "cost_policy": payload.get("cost_policy"),
        "filter_policy": payload.get("filter_policy"),
        "constants": payload.get("constants"),
        "evidence": payload.get("evidence"),
        "memory_revision": snapshot["revision"],
        "refilter": payload.get("refilter"),
    }


def cached_exact_replay(engine, payload, snapshot):
    llm = engine.llm
    if llm is None:
        import os
        cmd = os.environ.get("HOWHARD_LLM_COMMAND") or engine.config.data.get("llm", {}).get("command")
        if not cmd:
            return None
        try:
            from .llm import CommandLLM
            argv = json.loads(cmd) if isinstance(cmd, str) else cmd
            identity = CommandLLM(argv).identity
        except Exception:
            return None
    else:
        identity = str(getattr(llm, "identity", type(llm).__name__))
    sig = {"replay": _replay_signature(payload, snapshot)}
    ck = digest({"format": FORMAT, "backend": identity, "settings": SETTINGS, "request": sig})
    path = engine.config.state_root / "llm-cache" / f"{ck}.json"
    return _cache_read(path, sig, snapshot)


def complete(engine, payload, snapshot):
    identity = str(getattr(engine.llm, "identity", type(engine.llm).__name__))
    cache_dir = engine.config.state_root / "llm-cache"

    def cache_path(req_obj):
        ck = digest({"format": FORMAT, "backend": identity, "settings": SETTINGS, "request": req_obj})
        return cache_dir / f"{ck}.json"

    hit = _cache_read(cache_path(payload), payload, snapshot)
    if hit is not None:
        return hit

    original, current = deepcopy(payload), deepcopy(payload)
    calls = []
    filtering = payload["kind"] == "filter"
    field, memory_field = ("filter", "memory_filters") if filtering else ("forecasts", "memory_forecasts")
    fixed_field = "no_memory_" + field
    base_req = baseline_request(payload)

    if base_req is not None:
        baseline_response, usage = call(engine, base_req)
        keys(baseline_response, [field, "reason"], [memory_field], "LLM baseline")
        if baseline_response.get(memory_field):
            raise LLMError("LLM baseline cannot reference memories")
        if filtering:
            from .capability import validate_filter
            validate_filter(baseline_response[field], payload["cells"])
            fixed_value = deepcopy(baseline_response[field])
        else:
            unseen_cells = payload.get("unseen_cells", payload["cells"])
            validate_panel(baseline_response[field], {(c["family"], c["effort"]) for c in unseen_cells})
            unseen_cat = {(c["family"], c["effort"]) for c in unseen_cells if c.get("forecast_format") == "category-v1"}
            if any((r["family"], r["effort"]) in unseen_cat and "token_usage" not in r for r in baseline_response[field]):
                raise LLMError("forecast.token_usage: category rate card requires category forecast")
            if "preset_forecasts" in payload:
                fixed_value = deepcopy(payload["preset_forecasts"]) + deepcopy(baseline_response[field])
                current.pop("preset_forecasts", None)
                current.pop("unseen_cells", None)
            else:
                fixed_value = deepcopy(baseline_response[field])
        text(baseline_response["reason"], "LLM baseline.reason")
        current.update(phase="memory", **{fixed_field: fixed_value})
        current["contract"][field] = "Omit this field. The isolated no_memory value is fixed and must not be changed."
        baseline_reason = baseline_response["reason"]
        calls.append(usage)
    else:
        current["phase"] = "memory"
        baseline_reason = "Materialized prototype baseline."
        mem_hit = _cache_read(cache_path(visible_request(current)), visible_request(current), snapshot)
        if mem_hit is not None:
            return mem_hit

    memory_initial = visible_request(current)
    memory_calls = []
    for round_number in range(SETTINGS["retrieval_rounds"] + 1):
        if snapshot["memories"]:
            response, m_usage = call(engine, visible_request(current))
            calls.append(m_usage)
            memory_calls.append(m_usage)
        else:
            response = {memory_field: [], "reason": baseline_reason}
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
        if field in response and response[field] != current[fixed_field]:
            raise LLMError("Memory reader must not rewrite the isolated baseline forecasts")
        response[field] = deepcopy(current[fixed_field])
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
        _cache_write(engine, cache_path(original), original, current, response, usage)
        replay_sig = {"replay": _replay_signature(original, snapshot)}
        _cache_write(engine, cache_path(replay_sig), replay_sig, current, response, usage)
        if snapshot["memories"] and memory_calls:
            m_sec = sum(c["seconds"] for c in memory_calls)
            m_cost = sum(c["cost_usd"] for c in memory_calls) if all(c["cost_usd"] is not None for c in memory_calls) else None
            m_usage_summary = {"seconds": m_sec, "cost_usd": m_cost, "backend": identity, "calls": memory_calls}
            _cache_write(engine, cache_path(memory_initial), memory_initial, current, response, m_usage_summary)
        return response, current, {**usage, "cached": False, "call_seconds": seconds, "call_cost_usd": usage["cost_usd"]}
    raise LLMError("LLM did not return forecasts")


def outcome_observations(candidates, pairs):
    """Each forecast cell's counted runs: ``m * sim * delta * a``, then the per-ticket limit."""
    from . import evidence
    result = []
    for pair in sorted(pairs):
        counted = []
        for item in candidates["candidates"]:
            delta = evidence.discount((item["family"], item["effort"]), pair)
            if delta > 0:
                counted.append((item, delta, evidence.weight(item, delta)))
        shares = evidence.ticket_shares([(item["ticket"], value) for item, _, value in counted])
        for (item, delta, value), share in zip(counted, shares):
            result.append({"id": item["id"], "family": pair[0], "effort": pair[1], "outcome": item["outcome"],
                           "discount": delta, "ticket_share": share, "weight": value * share})
    return result


def duration_observations(candidates, outcome_obs):
    """Runs of the judged cell itself that record an active time, at the outcome reader's weight."""
    scored = {item["id"]: item for item in candidates["candidates"]}
    result = []
    for obs in outcome_obs:
        item = scored[obs["id"]]
        if (item["family"], item["effort"]) == (obs["family"], obs["effort"]) and item["active_seconds"] is not None:
            result.append({"id": obs["id"], "family": obs["family"], "effort": obs["effort"],
                           "active_seconds": float(item["active_seconds"]), "weight": obs["weight"]})
    return result


def augment(engine, prepared, response, payload, snapshot, usage, *, candidates):
    from .success import calibrate_duration, cost, denominator, posterior_intervals
    result, cells = deepcopy(prepared[0]), prepared[1]
    # The readers only weigh the request's scored candidates; nothing is compared again here.
    outcome_obs = outcome_observations(candidates, {(row["family"], row["effort"]) for row in result["cells"]})
    duration_obs = duration_observations(candidates, outcome_obs)
    policy = engine.config.data["sampling"]
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
    state = deepcopy(payload["beliefs"])
    # Relevance gates participation. Unrelated library growth must not dilute
    # the memories that actually produced a prospective forecast for this task.
    state["experts"] = [e for e in state["experts"] if e["key"] in worlds]
    normalize_support(state["experts"])
    weights = {e["key"]: e["weight"] for e in state["experts"]}
    indexed = {key: panel_lookup(panel) for key, panel in worlds.items()}
    for row in result["cells"]:
        pair = row["family"], row["effort"]
        intervals, _ = posterior_intervals(indexed, weights, pair, outcome_obs, policy)
        for field in ("P_clean", "P_loud", "P_silent", "P_silent_accepted"):
            row[field] = intervals[field]
        row["admission_risk"] = intervals["admission_risk"]
        seconds = sum(
            weights[k] * calibrate_duration(indexed[k][pair]["seconds"], pair, duration_obs)["expected_seconds"]
            for k in weights
        )
        cell = next(c for c in cells if (c["family"], c["effort"]) == pair)
        needed_credit = sum(weights[k] * forecast_credit(indexed[k][pair], cell) for k in weights) if cell["priced"] else None
        row["cost_per_credit"] = cell["cost_per_credit"]
        row["needed_credit_usd"] = needed_credit
        row["session_cost_usd"] = needed_credit * cell["cost_per_credit"] if needed_credit is not None else None
        row["C_run"] = row["session_cost_usd"]
        row["cost_basis"] = cell["money_basis"]
        row["credit_budget"] = deepcopy(cell["credit_budget"])
        row["available_credit_usd"] = cell["credit_budget"]["available_credit_usd"] if cell["credit_budget"] is not None else None
        row["credit_ok"] = True if row["available_credit_usd"] is None else (
            row["needed_credit_usd"] is not None and row["needed_credit_usd"] <= row["available_credit_usd"]
        )
        row["T"] = seconds / 3600
        c = result["constants"]
        loud, silent = row["P_loud"]["mean"], row["P_silent"]["mean"]
        row["expected_attempts"] = 1/denominator(loud)
        row["J"] = cost(row["session_cost_usd"] if cell["money_included"] else 0., row["T"], loud, silent, c)
        row["admissible"] = row["admission_risk"] <= c["epsilon"]
    result["cells"].sort(key=lambda row: (row["session_cost_usd"] if row["session_cost_usd"] is not None else float("inf"), row["family"], row["effort"]))
    result["semantic"] = {"format": FORMAT, "prompt_version": SETTINGS["prompt_version"], "backend": usage["backend"],
                          "memory_revision": snapshot["revision"], "memories": deepcopy(snapshot["memories"]),
                          "worlds": worlds, "exposures": exposures, "beliefs": state, "reason": response["reason"], "retrieval": payload["retrieval"],
                          "evidence": deepcopy(payload["evidence"]), "evidence_candidates": deepcopy(candidates),
                          "duration_observations": duration_obs, "outcome_observations": outcome_obs,
                          "reader_usage": {k: usage[k] for k in ("backend", "seconds", "cost_usd", "calls")}, "interval_kind": INTERVAL_KIND}
    result["sampling_policy"] = deepcopy(policy)
    result["backend"] = {"mode": "semantic", "name": usage["backend"], "inference_cached": usage["cached"]}
    result["routing_overhead"] = {"seconds": usage["call_seconds"], "cost_usd": usage["call_cost_usd"]}
    return result, cells


def forecast_credit(row, cell):
    if cell["rate_card"] is not None:
        from .accounting import price
        return price(row["token_usage"], cell["rate_card"])
    return row["tokens"] / 1e6 * cell["credit_rate"] if cell["credit_rate"] is not None else None


def forecast_session_cost(row, cell):
    needed_credit = forecast_credit(row, cell)
    return needed_credit * cell["cost_per_credit"] if needed_credit is not None else None


def sample(prepared, seed):
    from .success import calibration, denominator
    result, cells = prepared
    state = result["semantic"]
    parts = {k: np.zeros((1, len(cells))) for k in ("run", "loud", "silent", "time")}
    constants = result["constants"]
    if not any(c["admissible"] and c.get("credit_ok", True) for c in result["cells"]):
        # No RNG call is made when the accepted-result cap is unmet.
        state["sampling"] = {"enabled": False, "reason": "cap-unmet", "policy": deepcopy(result["sampling_policy"]), "cells": []}
        indexed = panel_lookup(result["cells"])
        for i, cell in enumerate(cells):
            row = indexed[cell["family"], cell["effort"]]
            d = denominator(row["P_loud"]["mean"])
            parts["run"][0, i] = row["session_cost_usd"]/d if cell["money_included"] else 0.
            parts["loud"][0, i] = constants["C_loud"]*row["P_loud"]["mean"]/d
            parts["silent"][0, i] = constants["C_silent"]*row["P_silent"]["mean"]/d
            parts["time"][0, i] = constants["lambda_per_hour"]*row["T"]/d
        return parts
    experts = state["beliefs"]["experts"]
    rng = np.random.default_rng(seed)
    chosen = str(rng.choice([e["key"] for e in experts], p=[e["weight"] for e in experts]))
    state["sampled_expert"] = chosen
    expert = next(e for e in experts if e["key"] == chosen)
    panel = panel_lookup(state["worlds"][chosen])
    calibration_rows = []
    for i, cell in enumerate(cells):
        row = panel[cell["family"], cell["effort"]]
        calibrated = calibration(result, state, expert, cell, row)
        calibration_rows.append(calibrated)
        p = rng.dirichlet(np.asarray(calibrated["outcome_posterior"]["alpha"]))
        dur = calibrated["duration"]
        sampled_seconds = float(rng.lognormal(dur["mu"], dur["sigma"])) if dur["sigma"] > 0 else dur["expected_seconds"]
        d = denominator(p[1])
        parts["run"][0, i] = forecast_session_cost(row, cell)/d if cell["money_included"] else 0.
        parts["loud"][0, i] = p[1]*constants["C_loud"]/d
        parts["silent"][0, i] = p[2]*constants["C_silent"]/d
        parts["time"][0, i] = sampled_seconds/3600*constants["lambda_per_hour"]/d
    state["sampling"] = {"enabled": True, "reason": "per-cell-dirichlet-posterior",
                         "policy": deepcopy(result["sampling_policy"]), "cells": calibration_rows}
    return parts


def validate_receipt(state, *, decision_format=DECISION_FORMAT):
    """Validate portable frozen semantic inputs before they affect feedback weights."""
    legacy = decision_format in (LEGACY_DECISION_V7, LEGACY_DECISION_V8)
    # v10 opened per-cell evidence bounds; v11 adds ranges on active time and token kinds
    # and counts one scored candidate set in place of those bounds.
    per_cell_evidence = decision_format in (DECISION_FORMAT, LEGACY_DECISION_V10)
    ranges = scored = decision_format == DECISION_FORMAT
    required = ["format", "prompt_version", "backend", "memory_revision", "memories", "worlds", "exposures", "beliefs", "reason", "retrieval", "evidence", "reader_usage", "interval_kind", "sampling"]
    if decision_format != LEGACY_DECISION_V7:
        required.append("duration_observations")
    if not legacy:
        required.append("outcome_observations")
    if scored:
        required.append("evidence_candidates")
    keys(state, required, ["sampled_expert"], "decision.semantic")
    if decision_format not in RECEIPT_SEMANTICS or (state["format"], state["prompt_version"]) != RECEIPT_SEMANTICS[decision_format]:
        raise HowHardError("decision.semantic: unsupported forecast format or prompt version")
    if type(state["memory_revision"]) is not int or state["memory_revision"] < 0:
        raise HowHardError("decision.semantic.memory_revision: expected non-negative integer")
    if state["interval_kind"] != (LEGACY_INTERVAL_KIND if legacy else INTERVAL_KIND):
        raise HowHardError("decision.semantic.interval_kind: unsupported credible interval kind")
    text(state["backend"], "decision.semantic.backend")
    text(state["reason"], "decision.semantic.reason")
    if not isinstance(state["worlds"], dict) or not {"llm"} <= state["worlds"].keys():
        raise HowHardError("decision.semantic.worlds: the llm panel is required")
    baseline = state["worlds"]["llm"]
    if not isinstance(baseline, list) or not baseline:
        raise HowHardError("decision.semantic.worlds: nonempty panels required")
    allowed = set()
    for row in baseline:
        keys(row, FIELDS, ["token_usage", *(() if legacy else INTERVAL_FIELDS), *(RANGE_FIELDS if ranges else ())],
             where="decision.semantic.forecast")
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
        validate_panel(panel, allowed, intervals=not legacy, ranges=ranges)
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
    from .success import EMPIRICAL_FLOOR, POSTERIOR_FIELDS, calibrate_duration, outcome_posterior, validate_policy as validate_sampling
    if scored:
        # Recompute what each reader counted from the stored candidates; the counts follow below.
        from . import evidence
        candidates = evidence.validate(state["evidence_candidates"])
        expected = outcome_observations(candidates, allowed)
        evidence.same(state["outcome_observations"], expected, "decision.semantic.outcome_observations")
        evidence.same(state["duration_observations"], duration_observations(candidates, expected),
                      "decision.semantic.duration_observations")
    elif decision_format != LEGACY_DECISION_V7:
        duration_obs = state["duration_observations"]
        max_duration = (
            max(SETTINGS["evidence_limit"], max(1, len(allowed)) * SETTINGS["outcome_evidence_limit"])
            if per_cell_evidence
            else SETTINGS["evidence_limit"]
        )
        if not isinstance(duration_obs, list) or len(duration_obs) > max_duration:
            raise HowHardError("decision.semantic.duration_observations: invalid array")
        for obs in duration_obs:
            keys(obs, ["id", "family", "effort", "active_seconds", "score", "similarity", "weight"], where="decision.semantic.duration_observation")
            for field in ("id", "family", "effort"):
                text(obs[field], "duration_observation." + field)
            number(obs["active_seconds"], "duration_observation.active_seconds", nonnegative=True)
            number(obs["score"], "duration_observation.score")
            for field in ("similarity", "weight"):
                number(obs[field], "duration_observation." + field, nonnegative=True)
    if not legacy and not scored:
        outcome_obs = state["outcome_observations"]
        if not isinstance(outcome_obs, list) or len(outcome_obs) > len(allowed) * SETTINGS["outcome_evidence_limit"]:
            raise HowHardError("decision.semantic.outcome_observations: invalid array")
        counts, seen_ids = {}, set()
        for obs in outcome_obs:
            keys(obs, ["id", "family", "effort", "outcome", "score", "similarity", "age", "weight"], where="decision.semantic.outcome_observation")
            for field in ("id", "family", "effort"):
                text(obs[field], "outcome_observation." + field)
            if obs["outcome"] not in OUTCOMES or type(obs["age"]) is not int or obs["age"] < 0:
                raise HowHardError("decision.semantic.outcome_observation: invalid outcome or age")
            number(obs["score"], "outcome_observation.score")
            for field in ("similarity", "weight"):
                number(obs[field], "outcome_observation." + field, nonnegative=True)
            pair = obs["family"], obs["effort"]
            if pair not in allowed or obs["id"] in seen_ids:
                raise HowHardError("decision.semantic.outcome_observation: unknown cell or duplicate ID")
            seen_ids.add(obs["id"])
            counts[pair] = counts.get(pair, 0) + 1
            if counts[pair] > SETTINGS["outcome_evidence_limit"]:
                raise HowHardError("decision.semantic.outcome_observation: cell limit exceeded")
    sampling = state["sampling"]
    keys(sampling, ["enabled", "reason", "policy", "cells"], where="decision.semantic.sampling")
    validate_sampling(sampling["policy"], legacy=legacy,
                      count_aged=decision_format in (LEGACY_DECISION_V9, LEGACY_DECISION_V10))
    if scored and (candidates["half_life_days"], candidates["comparison_cap"]) != (
            sampling["policy"]["half_life_days"], sampling["policy"]["comparison_cap"]):
        raise HowHardError("decision.semantic.evidence_candidates: constants differ from the sampling policy")
    if type(sampling["enabled"]) is not bool or not isinstance(sampling["cells"], list):
        raise HowHardError("decision.semantic.sampling: invalid calibration")
    if sampling["enabled"]:
        if {(c["family"], c["effort"]) for c in sampling["cells"]} != allowed or len(sampling["cells"]) != len(allowed):
            raise HowHardError("decision.semantic.sampling: calibration must cover forecasted cells")
        panels_by_key = {k: panel_lookup(p) for k, p in state["worlds"].items()}
        min_conc = (
            min(EMPIRICAL_FLOOR, sampling["policy"]["min_concentration"])
            if per_cell_evidence
            else sampling["policy"]["min_concentration"]
        )
        for c in sampling["cells"]:
            number(c["concentration"], "sampling concentration", positive=True)
            number(c["cost_sd"], "sampling cost standard deviation", nonnegative=True)
            if not min_conc <= c["concentration"] <= sampling["policy"]["max_concentration"]:
                raise HowHardError("decision.semantic.sampling: concentration outside bounds")
            if decision_format == LEGACY_DECISION_V7:
                if "duration" in c:
                    raise HowHardError("decision.semantic.sampling: unexpected v8 duration calibration in v7 receipt")
            else:
                if "duration" not in c:
                    raise HowHardError("decision.semantic.sampling: missing duration calibration")
                dur = c["duration"]
                keys(dur, ["forecast_seconds", "expected_seconds", "mu", "sigma", "evidence_weight", "between_experts_log_variance"], where="sampling.duration")
                for field in ("forecast_seconds", "expected_seconds", "sigma", "evidence_weight", "between_experts_log_variance"):
                    number(dur[field], "sampling.duration." + field, nonnegative=True)
                number(dur["mu"], "sampling.duration.mu")
                pair = c["family"], c["effort"]
                expert_key = c.get("expert")
                if expert_key not in panels_by_key:
                    raise HowHardError("decision.semantic.sampling: unknown calibrated expert")
                expected_dur = calibrate_duration(panels_by_key[expert_key][pair]["seconds"], pair, state["duration_observations"])
                expert_durs = {
                    e["key"]: calibrate_duration(panels_by_key[e["key"]][pair]["seconds"], pair, state["duration_observations"])
                    for e in belief["experts"]
                }
                mean_mu = sum(e["weight"] * expert_durs[e["key"]]["mu"] for e in belief["experts"])
                expected_dur["between_experts_log_variance"] = sum(
                    e["weight"] * (expert_durs[e["key"]]["mu"] - mean_mu) ** 2 for e in belief["experts"]
                )
                for field, expected_val in expected_dur.items():
                    if abs(dur[field] - expected_val) > 1e-10 * max(1.0, abs(expected_val)):
                        raise HowHardError("decision.semantic.sampling: inconsistent duration calibration")
            if not legacy:
                keys(c, ["family", "effort", "expert", "outcome_posterior", "probabilities", "concentration", "duration", "cost_sd"], where="decision.semantic.sampling.cell")
                expected = outcome_posterior(
                    panels_by_key[c["expert"]][pair],
                    pair,
                    state["outcome_observations"],
                    sampling["policy"],
                    legacy_v9=(decision_format == LEGACY_DECISION_V9),
                )
                posterior = c["outcome_posterior"]
                keys(posterior, list(POSTERIOR_FIELDS), where="sampling.outcome_posterior")
                if any(not isinstance(posterior[f], list) or len(posterior[f]) != 3 for f in ("prior", "counts", "alpha")) or (
                        not isinstance(c["probabilities"], list) or len(c["probabilities"]) != 3):
                    raise HowHardError("decision.semantic.sampling: invalid per-cell outcome posterior")
                actual = [posterior["prior_concentration"], posterior["evidence_weight"], c["concentration"], *posterior["prior"],
                          *posterior["counts"], *posterior["alpha"], *c["probabilities"]]
                wanted = [expected["prior_concentration"], expected["evidence_weight"], expected["concentration"], *expected["prior"],
                          *expected["counts"], *expected["alpha"], *expected["mean"]]
                if len(actual) != len(wanted) or any(not isinstance(a, (int, float)) or isinstance(a, bool)
                                                     or abs(a - b) > 1e-10 * max(1., abs(b)) for a, b in zip(actual, wanted)):
                    raise HowHardError("decision.semantic.sampling: inconsistent per-cell outcome posterior")
    elif sampling["cells"] or "sampled_expert" in state:
        raise HowHardError("decision.semantic.sampling: disabled sampling cannot claim draws")
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
        from .decisions import load, exists
        if exists(engine.config, row["decision_id"]):
            receipt = load(engine.config, row["decision_id"], for_scoring=True)
    request = {"kind": "learn", "prompt_version": SETTINGS["writer_version"], "observation": row,
               "decision": receipt, "memory_revision": snapshot["revision"], "memories": snapshot["memories"],
               "beliefs": beliefs(engine), "evidence": evidence_search(engine, row.get("task_text", "")),
               "contract": "Return ONLY {operations: [...], reason: string}. Operations: add(title,body,scope?,evidence_ids?); edit(id,title?/body?/scope?/evidence_ids?,expected_version?); remove(id); restore(id); merge(ids,title,body); split(id,parts:[{title,body}]). Every item includes op. Use actual evidence IDs. execution: IDs in decision/beliefs are read-only observed runs, never editable memories. Hypotheses are provisional; evidence that inspired a new memory does not validate its forecasts. Prefer targeted useful changes; an empty list is valid. Never delete raw run evidence."}
    roots, history = {}, []
    for raw in engine.raw_rows:
        roots[raw["id"]] = roots[raw["amends"]] if raw["amends"] else raw["id"]
        if roots[raw["id"]] == row["id"]:
            history.append(deepcopy(raw))
    request["outcome_history"] = history
    request["contract"] += " Use observation for current effective outcome and usage. outcome_history is an audit trail: measurement_corrections replace values, never add runs or usage. Interrupted outcomes (usage_limit, provider_error, host_error, cancelled, unrecorded) represent operational interruptions rather than model task failures."
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
    with locked(engine.config, operation="memory learning"):
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
