"""Behavioral tests for dynamic synthesized (task, context) prototypes and materialized routing (#123)."""

from copy import deepcopy
import hashlib
import json
import sqlite3

import numpy as np
import pytest

from howhard import Engine, RetrievalPipeline
from howhard.cli import main
from howhard.decisions import load
from howhard.llm import LLMError
from howhard.prototypes import synthesize_representative
from test_execution_accounting import artifact, card, codex_events, counts


class SpyLLM:
    identity = "prototype-spy-llm"

    def __init__(self, *, pessimistic_models=()):
        self.calls = []
        self.pessimistic_models = set(pessimistic_models)
        self.last_usage = {"seconds": 0.01, "cost_usd": 0.001, "tokens": None, "models": ["spy"]}

    def complete(self, request):
        self.calls.append(deepcopy(request))
        if request["kind"] == "learn":
            return {"operations": [], "reason": "No-op writer"}
        if request["kind"] == "filter":
            if request.get("phase") == "baseline":
                levels = {e: i for i, e in enumerate(("low", "medium", "high", "xhigh", "max", "ultra"))}
                cells = sorted(request["cells"], key=lambda c: (c["model"], levels.get(c["effort"], 99)))
                order = [c["key"] for c in cells]
                return {
                    "filter": {
                        "order": order,
                        "lower": order[0],
                        "upper": order[min(len(order), request["filter_policy"]["max_forecast_cells"]) - 1],
                    },
                    "reason": "Spy capability baseline",
                }
            return {"memory_filters": [], "reason": "Spy capability memory"}
        if request.get("phase") == "baseline":
            rows = []
            for cell in request["cells"]:
                if cell["model"] in self.pessimistic_models:
                    probs = {"clean": 0.35, "loud": 0.45, "silent": 0.20}
                else:
                    probs = {"clean": 0.92, "loud": 0.06, "silent": 0.02}
                entry = {
                    "family": cell["family"],
                    "effort": cell["effort"],
                    **probs,
                    "tokens": 10000,
                    "seconds": 120,
                }
                if cell.get("forecast_format") == "category-v1":
                    entry["token_usage"] = counts(input=6000, output=2000, cache_read=2000)
                rows.append(entry)
            return {"forecasts": rows, "reason": "Spy forecast baseline"}
        return {"memory_forecasts": [], "reason": "Spy forecast memory"}


class TopicSimilarityDouble:
    """Deterministic topic relevance without pretrained model downloads."""

    @property
    def reranker(self):
        return self

    def score(self, query, documents):
        q = query.casefold()
        scores = []
        for doc in documents:
            d = doc.casefold()
            if "rollback" in q and "rollback" in d:
                if ("cluster migration" in q and "unit helper" in d) or (
                    "unit helper" in q and "cluster migration" in d
                ):
                    scores.append(2.0)
                else:
                    scores.append(12.0)
            elif "sonnet" in q and "sonnet" in d:
                scores.append(11.0)
            else:
                scores.append(-18.0)
        return scores

    def search(self, records, query, limit=8, **kwargs):
        scores = self.score(query, [r["text"] for r in records])
        paired = sorted(zip(scores, records), key=lambda item: (-item[0], item[1]["key"]))[:limit]
        return {
            "records": deepcopy([r for _, r in paired]),
            "trace": {
                "seconds": 0.0,
                "selected": [{"key": r["key"], "score": float(s)} for s, r in paired],
            },
        }


class ClusterEmbedderDouble:
    """Deterministic 16-dim topic embedder for large backfill and split checks."""

    identity = "cluster-embedder-double-v1"
    dimensions = 16

    def encode(self, texts, *, query=False):
        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for idx, raw in enumerate(texts):
            lowered = raw.casefold()
            if "unit helper" in lowered:
                out[idx, 0] = 1.0
                out[idx, 2] = 0.15
            elif "cluster migration" in lowered:
                out[idx, 1] = 1.0
                out[idx, 2] = 0.15
            elif "rollback" in lowered:
                out[idx, 0] = 0.7
                out[idx, 1] = 0.7
            elif "billing" in lowered:
                out[idx, 3] = 1.0
            else:
                digest_bytes = hashlib.sha256(lowered.encode()).digest()[: self.dimensions]
                out[idx] = np.frombuffer(digest_bytes, dtype=np.uint8).astype(np.float32) + 1.0
            # Small deterministic jitter so distinct texts stay distinct while clustering cleanly.
            jitter = hashlib.sha256(raw.encode()).digest()[0] / 25500.0
            out[idx, -1] = 0.01 + jitter
            out[idx] /= np.linalg.norm(out[idx])
        return out

    def score(self, query, documents):
        q_vec = self.encode([query], query=True)[0]
        d_vecs = self.encode(list(documents))
        # Map cosine > 0.5 to positive cross-encoder logits and <= 0.5 to negative logits.
        cos = d_vecs @ q_vec
        return ((cos - 0.5) * 20.0).tolist()


def _db_prototypes(engine):
    path = engine.config.state_root / "payloads.sqlite3"
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        protos = [dict(r) for r in db.execute("SELECT * FROM prototypes ORDER BY created_at, prototype_id")]
        cells = [dict(r) for r in db.execute("SELECT * FROM prototype_cells ORDER BY prototype_id, model, effort")]
        members = [dict(r) for r in db.execute("SELECT * FROM prototype_members ORDER BY prototype_id, evidence_id")]
    return protos, cells, members


def test_cold_start_synthesizes_representative_prototype_and_warm_route_runs_provider_free(
    tmp_path, monkeypatch, capsys
):
    candidates = [
        {"model": "fixture/fast", "effort": "medium", "usd_per_mtok": 0.4},
        {"model": "fixture/strong", "effort": "high", "usd_per_mtok": 1.2},
    ]
    spy = SpyLLM()
    engine = Engine(directory=tmp_path, llm=spy, retrieval=TopicSimilarityDouble())

    noisy_prompt = (
        "You are the assigned leaf worker for issue #123: Repair rollback boundary "
        "after commit dda176a9b2f4e10 in /var/tmp/worktree-123/scratch.json"
    )
    rep = synthesize_representative("task", {"task_text": noisy_prompt, "project_context": None})
    assert "#123" not in rep["representative_task"]
    assert "dda176a9b2f4e10" not in rep["representative_task"]
    assert "/var/tmp/" not in rep["representative_task"]
    assert "assigned leaf worker" not in rep["representative_task"].casefold()
    assert "rollback boundary" in rep["representative_task"].casefold()

    # Cold start runs Stage 1 + Stage 2 baseline once on the synthesized representative task.
    first = engine.pick("task", noisy_prompt, native_candidates=candidates, seed=7)
    assert first["format"] == "howhard-decision-v11" if "format" in first else True
    receipt = load(engine.config, first["decision_id"])
    assert receipt["format"] == "howhard-decision-v11"
    assert receipt["capability"]["format"] == "howhard-capability-v6"
    assert receipt["semantic"]["format"] == "howhard-semantic-v8"
    assert len(spy.calls) == 2
    assert all("#123" not in c["context"]["task_text"] for c in spy.calls)
    assert all("/var/tmp/" not in c["context"]["task_text"] for c in spy.calls)
    # Receipt still preserves the caller's exact original task_text.
    assert receipt["context"]["task_text"] == noisy_prompt

    protos, proto_cells, _ = _db_prototypes(engine)
    assert len(protos) == 1 and protos[0]["active"] == 1
    assert len(proto_cells) == 2
    for row in proto_cells:
        prior = json.loads(row["prior_json"])
        assert "P_clear" in prior and "clean" in prior and "seconds" in prior

    # Second route on a related task with different one-off literals hits the prototype with zero LLM calls.
    spy.calls.clear()
    pending = engine.prepare(
        "task",
        "Repair rollback boundary regression from #987 in /tmp/repro-987.log",
        native_candidates=candidates,
    )
    assert pending["stage"] == "filter"
    assert pending["baseline_request"] is None

    second = engine.pick(
        "task",
        "Repair rollback boundary regression from #987 in /tmp/repro-987.log",
        native_candidates=candidates,
        seed=7,
    )
    assert spy.calls == []
    assert second["chosen"] == first["chosen"]
    assert second["semantic"]["backend"] == "prototype-materialized"
    load(engine.config, second["decision_id"])

    # Warm Engine.pick with llm=None AND no configured CLI provider succeeds completely offline!
    monkeypatch.delenv("HOWHARD_LLM_COMMAND", raising=False)
    monkeypatch.setattr(
        "howhard.llm.ClaudeCLI",
        lambda *a, **kw: (_ for _ in ()).throw(LLMError("No CLI provider allowed on warm fast path")),
    )
    provider_free = Engine(directory=tmp_path, llm=None, retrieval=TopicSimilarityDouble())
    third = provider_free.pick("task", "Repair rollback edge case", native_candidates=candidates, seed=7)
    assert third["chosen"] == first["chosen"]

    # CLI `howhard context --auto-decide` returns a committed decision receipt in one step.
    cand_file = tmp_path / "candidates.json"
    cand_file.write_text(json.dumps(candidates))
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "context",
                "task",
                "--text",
                "Repair rollback handler",
                "--candidates",
                str(cand_file),
                "--auto-decide",
                "--seed",
                "7",
                "--json",
            ]
        )
        == 0
    )
    auto_out = json.loads(capsys.readouterr().out)
    assert "decision_id" in auto_out
    assert auto_out["chosen"] == first["chosen"]


def test_dissimilar_task_misses_existing_prototype_and_partial_unseen_model_forecasts_only_delta(tmp_path):
    base_candidates = [
        {"model": "fixture/model-a", "effort": "high", "usd_per_mtok": 0.5},
        {"model": "fixture/model-b", "effort": "high", "usd_per_mtok": 1.0},
    ]
    spy = SpyLLM()
    engine = Engine(directory=tmp_path, llm=spy, retrieval=TopicSimilarityDouble())

    engine.pick("task", "Repair rollback transaction", native_candidates=base_candidates, seed=1)
    spy.calls.clear()

    # Unrelated task ("Write a sonnet", score <= 0) misses the rollback prototype and emits baseline_request.
    miss_pending = engine.prepare("task", "Write a sonnet", native_candidates=base_candidates)
    assert miss_pending["baseline_request"] is not None
    engine.pick("task", "Write a sonnet", native_candidates=base_candidates, seed=2)
    assert [c["kind"] for c in spy.calls] == ["filter", "predict"]
    protos, _, _ = _db_prototypes(engine)
    assert len([p for p in protos if p["active"] == 1]) == 2

    # Adding a brand-new unseen model to the warm rollback prototype requests Stage 2 forecast ONLY for the new model!
    spy.calls.clear()
    expanded = [*base_candidates, {"model": "fixture/model-new", "effort": "high", "usd_per_mtok": 0.8}]
    stage1 = engine.prepare("task", "Repair rollback retry", native_candidates=expanded)
    assert stage1["baseline_request"] is not None
    assert len(stage1["baseline_request"]["cells"]) == 3

    picked_expanded = engine.pick("task", "Repair rollback retry", native_candidates=expanded, seed=3)
    predict_calls = [c for c in spy.calls if c["kind"] == "predict"]
    assert len(predict_calls) == 1
    assert [c["model"] for c in predict_calls[0]["cells"]] == ["fixture/model-new"]
    assert {c["model"] for c in picked_expanded["cells"]} == {
        "fixture/model-a",
        "fixture/model-b",
        "fixture/model-new",
    }
    load(engine.config, picked_expanded["decision_id"])

    # Subsequent route with the expanded inventory now makes 0 LLM calls.
    spy.calls.clear()
    warm_expanded = engine.pick("task", "Repair rollback retry again", native_candidates=expanded, seed=3)
    assert spy.calls == []
    assert {c["model"] for c in warm_expanded["cells"]} == {
        "fixture/model-a",
        "fixture/model-b",
        "fixture/model-new",
    }


def test_continuous_prior_decay_eliminates_pessimistic_cold_start_bias_at_tau(tmp_path):
    candidates = [
        {"model": "post-cutoff/cheap-winner", "effort": "high", "usd_per_mtok": 0.2},
        {"model": "known/expensive", "effort": "high", "usd_per_mtok": 800.0},
    ]
    spy = SpyLLM(pessimistic_models={"post-cutoff/cheap-winner"})
    engine = Engine(directory=tmp_path, llm=spy, retrieval=TopicSimilarityDouble())

    # Cold start: pessimistic guess (clean=0.35, loud=0.45, silent=0.20) makes cheap-winner inadmissible.
    cold = engine.pick("task", "Repair rollback", native_candidates=candidates, seed=5)
    by_cold = {c["model"]: c for c in cold["cells"]}
    assert not by_cold["post-cutoff/cheap-winner"]["admissible"]
    assert cold["chosen"]["model"] == "known/expensive"

    # Observe 3 clean runs on post-cutoff/cheap-winner (W_emp > 2.0 == EMPIRICAL_TAU).
    for idx in range(3):
        engine.observe(
            "task",
            "post-cutoff/cheap-winner",
            "high",
            "clean",
            native=True,
            task_text="Repair rollback",
            tokens=8000,
            active_seconds=45.0,
            id=f"win-{idx}",
        )

    spy.calls.clear()
    warm = engine.pick("task", "Repair rollback", native_candidates=candidates, seed=5)
    assert spy.calls == []
    by_warm = {c["model"]: c for c in warm["cells"]}
    assert by_warm["post-cutoff/cheap-winner"]["admissible"]
    assert by_warm["post-cutoff/cheap-winner"]["J"] < by_warm["known/expensive"]["J"]
    assert warm["exploration"]["mean_best"]["model"] == "post-cutoff/cheap-winner"
    chosen_models = [
        engine.pick("task", "Repair rollback", native_candidates=candidates, seed=s)["chosen"]["model"]
        for s in range(40)
    ]
    assert spy.calls == []
    assert chosen_models.count("post-cutoff/cheap-winner") >= 30

    winner_family = by_warm["post-cutoff/cheap-winner"]["family"]
    cal = next(c for c in warm["semantic"]["sampling"]["cells"] if c["family"] == winner_family)
    posterior = cal["outcome_posterior"]
    assert posterior["evidence_weight"] >= 2.0
    # Exact 0.0 weight on the pessimistic [0.35, 0.45, 0.20] prior once W_emp >= 2.0!
    assert posterior["prior_concentration"] == pytest.approx(1.0)
    assert posterior["alpha"][0] == pytest.approx(0.88 + posterior["counts"][0])
    assert posterior["alpha"][1] == pytest.approx(0.08 + posterior["counts"][1])
    assert posterior["alpha"][2] == pytest.approx(0.04 + posterior["counts"][2])
    load(engine.config, warm["decision_id"])


def test_incremental_observe_interrupted_exclusion_and_amendment_reversibility(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=TopicSimilarityDouble())
    engine.observe(
        "task",
        "fixture-model",
        "high",
        "clean",
        native=True,
        task_text="Repair rollback",
        tokens=1000,
        active_seconds=60.0,
        id="run-1",
    )
    engine.observe(
        "task",
        "fixture-model",
        "high",
        "clean",
        native=True,
        task_text="Repair rollback",
        tokens=2000,
        active_seconds=120.0,
        id="run-2",
    )

    candidates = [{"model": "fixture-model", "effort": "high", "usd_per_mtok": 1.0}]
    baseline_pick = engine.pick("task", "Repair rollback", native_candidates=candidates, seed=1)
    _, cells_before, _ = _db_prototypes(engine)
    stats_before = json.loads(cells_before[0]["stats_json"])
    assert stats_before["run_count"] == 2
    assert stats_before["counts"][0] > 1.5 and stats_before["counts"][1] == 0.0 and stats_before["counts"][2] == 0.0

    # Interrupted operational runs must be completely excluded from prototype cell stats.
    engine.observe(
        "task",
        "fixture-model",
        "high",
        "interrupted",
        cause="usage_limit",
        task_text="Repair rollback",
        active_seconds=1.0,
        id="run-int",
    )
    after_int_pick = engine.pick("task", "Repair rollback", native_candidates=candidates, seed=1)
    _, cells_int, _ = _db_prototypes(engine)
    assert json.loads(cells_int[0]["stats_json"]) == stats_before
    # Each pick measures the runs' age to its own prepare time, moments apart.
    assert after_int_pick["cells"][0]["J"] == pytest.approx(baseline_pick["cells"][0]["J"], rel=1e-4)

    # Amending run-2 clean -> silent -> clean restores exact cell stats!
    engine.observe("task", "fixture-model", "high", "silent", cause="capability", amends="run-2", id="amend-silent")
    engine.pick("task", "Repair rollback", native_candidates=candidates, seed=1)
    _, cells_silent, _ = _db_prototypes(engine)
    stats_silent = json.loads(cells_silent[0]["stats_json"])
    assert stats_silent["counts"][2] > 0.5
    assert stats_silent["counts"][0] < stats_before["counts"][0]

    engine.observe("task", "fixture-model", "high", "clean", amends="amend-silent", id="amend-clean")
    restored_pick = engine.pick("task", "Repair rollback", native_candidates=candidates, seed=1)
    _, cells_restored, _ = _db_prototypes(engine)
    stats_restored = json.loads(cells_restored[0]["stats_json"])
    assert stats_restored["run_count"] == stats_before["run_count"]
    assert stats_restored["counts"] == pytest.approx(stats_before["counts"])
    assert stats_restored["duration_mean_log"] == pytest.approx(stats_before["duration_mean_log"])
    assert restored_pick["cells"][0]["J"] == pytest.approx(baseline_pick["cells"][0]["J"], rel=1e-4)

    # Attaching complete category token usage updates token_usage_shares for category-v1 candidates.
    manifest = artifact(tmp_path, codex_events(), execution_id="run-1")
    engine.attach_usage("run-1", manifest, id="run-1-usage")
    cat_candidates = [{"model": "fixture-model", "effort": "high", "rate_card": card()}]
    cat_pick = engine.pick("task", "Repair rollback", native_candidates=cat_candidates, seed=1)
    assert cat_pick["cost_policy"]["worker_money_included"]
    assert cat_pick["cells"][0]["needed_credit_usd"] > 0
    load(engine.config, cat_pick["decision_id"])


def test_dynamic_prototype_splits_heterogeneous_cluster_into_two_children(tmp_path):
    embedder = ClusterEmbedderDouble()
    pipeline = RetrievalPipeline(tmp_path / ".howhard", embedder=embedder, reranker=embedder)
    engine = Engine(directory=tmp_path, retrieval=pipeline)

    # Seed an initial broad rollback prototype, then feed 6 fast unit-helper runs and 6 slow cluster-migration runs.
    engine.observe(
        "task",
        "fixture/worker",
        "high",
        "clean",
        native=True,
        task_text="Repair rollback",
        tokens=3000,
        active_seconds=120.0,
        id="seed-broad",
    )
    for i in range(6):
        engine.observe(
            "task",
            "fixture/worker",
            "high",
            "clean",
            native=True,
            task_text=f"Repair rollback unit helper case {i}",
            tokens=1500,
            active_seconds=20.0,
            id=f"fast-{i}",
        )
    for i in range(6):
        engine.observe(
            "task",
            "fixture/worker",
            "high",
            "clean",
            native=True,
            task_text=f"Repair rollback cluster migration phase {i}",
            tokens=45000,
            active_seconds=7200.0,
            id=f"slow-{i}",
        )

    candidates = [{"model": "fixture/worker", "effort": "high", "usd_per_mtok": 1.0}]
    fast_pick = engine.pick("task", "Repair rollback unit helper assertion", native_candidates=candidates, seed=1)
    slow_pick = engine.pick("task", "Repair rollback cluster migration cutover", native_candidates=candidates, seed=1)

    protos, _, _ = _db_prototypes(engine)
    retired = [p for p in protos if p["active"] == 0]
    active = [p for p in protos if p["active"] == 1]
    assert len(retired) >= 1
    assert len(active) == 2
    assert all(p["parent_id"] == retired[0]["prototype_id"] for p in active)
    rep_tasks = " | ".join(p["representative_task"].casefold() for p in active)
    assert "unit helper" in rep_tasks and "cluster migration" in rep_tasks
    assert slow_pick["cells"][0]["T"] > fast_pick["cells"][0]["T"] * 5.0


def test_historical_outcomes_jsonl_backfill_enables_zero_llm_first_warm_pick(project, monkeypatch):
    from conftest import row

    # 40 historical rows (> 32 threshold) across two distinct task clusters, with tokens and seconds recorded.
    rows = [
        row(
            i,
            task="Y",
            family="sonnet",
            effort="high",
            outcome="clean",
            task_text=f"Repair rollback unit helper {i}",
            tokens=2500,
            seconds=45.0,
        )
        for i in range(20)
    ] + [
        row(
            20 + i,
            task="Y",
            family="opus",
            effort="high",
            outcome="clean",
            task_text=f"Reconcile billing invoice ledger {i}",
            tokens=6000,
            seconds=90.0,
        )
        for i in range(20)
    ]
    cfg = project(rows=rows)
    embedder = ClusterEmbedderDouble()
    stages = RetrievalPipeline(cfg.state_root, embedder=embedder, reranker=embedder)

    monkeypatch.delenv("HOWHARD_LLM_COMMAND", raising=False)
    monkeypatch.setattr(
        "howhard.llm.ClaudeCLI",
        lambda *a, **kw: (_ for _ in ()).throw(LLMError("Backfill warm pick must not invoke provider")),
    )
    engine = Engine(cfg, llm=None, retrieval=stages)
    picked = engine.pick(
        "Y",
        "Repair rollback unit helper check",
        native_candidates=[{"model": "claude-sonnet-5-5", "effort": "high", "usd_per_mtok": 0.4}],
        seed=9,
    )
    assert picked["chosen"]["model"] == "claude-sonnet-5-5"
    assert picked["semantic"]["backend"] == "prototype-materialized"
    protos, _, _ = _db_prototypes(engine)
    assert len([p for p in protos if p["active"] == 1]) >= 2
