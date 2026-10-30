"""Descriptive scores for observed real runs; no counterfactual quality claims."""

from collections import Counter
import json
import math

from .decisions import load, receipt_path
from .semantic import OUTCOMES, SETTINGS, panel_lookup
from .validation import HowHardError


def report(engine):
    outcomes = Counter(row["outcome"] for row in engine.rows)
    scored = {key: [] for key in ("llm", "weighted_memory")}
    missing, unsupported = [], []
    for row in engine.rows:
        decision_id = row.get("decision_id")
        if not decision_id:
            continue
        if not receipt_path(engine.config, decision_id).is_file():
            missing.append(row["id"])
            continue
        decision = load(engine.config, decision_id, for_scoring=True)
        if decision is None:
            unsupported.append(row["id"])
            continue
        state = decision["semantic"]
        pair = row["family"], row["resolved_effort"]
        panels = {key: panel_lookup(panel) for key, panel in state["worlds"].items()}
        if pair not in panels["llm"]:
            raise HowHardError(f"report: receipt has no actual cell for {row['id']}")
        probabilities = {key: [panels[key][pair][outcome] for outcome in OUTCOMES] for key in ("llm",)}
        probabilities["weighted_memory"] = [sum(e["weight"] * panels.get(e["key"], panels["llm"])[pair][outcome] for e in state["beliefs"]["experts"]) for outcome in OUTCOMES]
        target = OUTCOMES.index(row["outcome"])
        for key, values in probabilities.items():
            loss = -math.log(max(SETTINGS["belief"]["probability_floor"], values[target]))
            brier = sum((p - int(i == target)) ** 2 for i, p in enumerate(values))
            scored[key].append((loss, brier))
    metrics = {key: {"runs": len(values), "mean_log_loss": sum(v[0] for v in values) / len(values) if values else None,
                     "mean_brier": sum(v[1] for v in values) / len(values) if values else None} for key, values in scored.items()}
    events = []
    path = engine.config.state_root / "inference.jsonl"
    if path.exists():
        for line, value in enumerate(path.read_text().splitlines(), 1):
            try:
                events.append(json.loads(value))
            except ValueError as exc:
                raise HowHardError(f"{path}: line {line}: invalid inference audit JSON") from exc
    external = [e for e in events if e.get("external")]
    measured = [e for e in events if not e.get("external")]
    unknown_inference = sum(e["cost_usd"] is None for e in measured)
    decisions = []
    decision_path = engine.config.state_root / "decision-events.jsonl"
    if decision_path.exists():
        try:
            decisions = [json.loads(line) for line in decision_path.read_text().splitlines()]
        except ValueError as exc:
            raise HowHardError(f"{decision_path}: invalid decision audit JSON") from exc
    known_inference = sum(e["cost_usd"] for e in events if e["cost_usd"] is not None)
    known_worker = sum(row["cost_usd"] for row in engine.rows if row["cost_usd"] is not None)
    unknown_worker = sum(row["cost_usd"] is None for row in engine.rows)
    return {"project": engine.config.data["project"], "runs": len(engine.rows), "outcomes": {k: outcomes[k] for k in OUTCOMES},
            "forecast_metrics": metrics, "missing_receipts": missing, "unsupported_receipts": unsupported,
            "worker_cost": {"known_usd": known_worker, "unknown_runs": unknown_worker, "total_usd": known_worker if not unknown_worker else None},
            "inference": {"calls": len(measured), "failed_calls": sum(e["status"] == "failed" for e in measured),
                          "seconds": sum(e["seconds"] for e in measured), "known_usd": known_inference,
                          "external_submissions": len(external), "external_cost_and_latency": "unknown" if external else "none",
                          "unknown_cost_calls": unknown_inference, "total_usd": known_inference if not unknown_inference and not external else None,
                          "by_role": dict(Counter(e["kind"] for e in measured))},
            "decisions": {"calls": len(decisions), "failed_calls": sum(e["status"] == "failed" for e in decisions),
                          "end_to_end_seconds": sum(e["seconds"] for e in decisions),
                          "note": "Decision latency includes reader inference; writer inference is reported separately. Action propensities are unknown."},
            "memory": {"active": len(engine.memory.list()), "operations": dict(Counter(e["operation"]["op"] for e in engine.memory.history(None)))},
            "interpretation": "Observed-action descriptive scores only. These logs do not identify unrun alternatives' outcomes, establish routing superiority, or validate a silent-failure cap. Clean outcomes remain subject to later amendments."}
