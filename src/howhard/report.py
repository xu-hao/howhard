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
    from .accounting import summary
    return {"filter_quality": filter_quality(engine), "execution_accounting": summary(engine.rows), "project": engine.config.data["project"], "runs": len(engine.rows), "outcomes": {k: outcomes[k] for k in OUTCOMES},
            "forecast_metrics": metrics, "missing_receipts": missing, "unsupported_receipts": unsupported,
            "worker_cost": {"basis": "legacy-unspecified", "known_usd": known_worker, "unknown_runs": unknown_worker, "total_usd": known_worker if not unknown_worker else None},
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



def filter_quality(engine):
    """Describe actual filter outcomes and frozen disagreements, including pending runs."""
    from .capability import cell_key
    from .retrieval import records, task_text
    from datetime import datetime
    decisions = {}
    path = engine.config.state_root / "decisions"
    for file in sorted(path.glob("*.json")):
        decision = load(engine.config, file.stem, for_scoring=True)
        if decision is not None:
            decisions[file.stem] = decision
    events = engine.config.state_root / "decision-events.jsonl"
    times = {e["decision_id"]: e["at_utc"] for e in (json.loads(line) for line in events.read_text().splitlines())
             if e.get("decision_id")} if events.exists() else {}
    fit = Counter()
    executions, comparisons = [], []
    floor = SETTINGS["belief"]["probability_floor"]
    for row in engine.rows:
        decision = decisions.get(row.get("decision_id"))
        if decision is None:
            continue
        state = decision["capability"]
        actual = cell_key(row)
        c = next(c for c in state["cells"] if c["key"] == actual)
        if c["classification"] == "fit":
            fit[row["outcome"]] += 1
        executions.append({"execution_id": row["id"], "decision_id": row["decision_id"],
                           "classification": c["classification"], "position": c["position"], "outcome": row["outcome"]})
        disagreement = next((d for d in state["disagreements"] if d["key"] == actual), None)
        if disagreement is not None:
            probabilities = [disagreement["P_clear"], disagreement["P_clean"]]
            losses = [-math.log(max(floor, p if row["outcome"] == "clean" else 1-p)) for p in probabilities]
            winner = "tie" if abs(losses[0]-losses[1]) < 1e-12 else "stage1" if losses[0] < losses[1] else "stage2"
            comparisons.append({"execution_id": row["id"], "key": actual, "stage1_log_loss": losses[0],
                                "stage2_log_loss": losses[1], "closer": winner})
    strict = []
    for decision_id, decision in decisions.items():
        excluded = {c["key"] for c in decision["capability"]["cells"] if c["classification"] == "underpowered"}
        created = times.get(decision_id)
        if not excluded or not created:
            continue
        later = [r for r in engine.rows if r["outcome"] == "clean" and cell_key(r) in excluded
                 and datetime.fromisoformat(r["observed_at"]) > datetime.fromisoformat(created)]
        if not later:
            continue
        query = task_text(decision["task_type"], decision["context"])
        result = engine.retrieval.search(records(rows=later), query, limit=SETTINGS["evidence_limit"])
        selected = result["records"]
        scores = {s["key"]: s["score"] for s in result["trace"]["selected"]}
        missing = [r for r in selected if scores.get(r["key"]) is None]
        if missing:
            values = engine.retrieval.reranker.score(query, [r["text"] for r in missing])
            scores.update({r["key"]: float(v) for r, v in zip(missing, values)})
        for item in selected:
            if scores[item["key"]] > 0:
                strict.append({"decision_id": decision_id, "execution_id": item["value"]["id"],
                               "key": cell_key(item["value"]), "similarity_score": scores[item["key"]]})
    n = len(decisions)
    widened = sum(bool(d["capability"]["widening"]) for d in decisions.values())
    total_fit = sum(fit.values())
    return {"decisions": n, "executions": executions,
            "fit_failures": {"executed": total_fit, "loud": fit["loud"], "silent": fit["silent"],
                             "failure_rate": (fit["loud"]+fit["silent"])/total_fit if total_fit else None},
            "widening": {"decisions": widened, "rate": widened/n if n else None},
            "underpowered_later_clean": strict,
            "disagreements": {"count": sum(len(d["capability"]["disagreements"]) for d in decisions.values()),
                              "resolved": len(comparisons), "closer": dict(Counter(c["closer"] for c in comparisons)),
                              "comparisons": comparisons},
            "interpretation": "Task-conditioned observed-action diagnostics; no outcomes are inferred for cells that did not run."}
