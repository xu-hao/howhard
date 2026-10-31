"""Public retrieval wiring; model doubles do not establish relevance quality."""

import hashlib
import sqlite3

import numpy as np
import pytest

from conftest import inventory, row
from howhard import Engine
from howhard.retrieval import RetrievalPipeline
from howhard.validation import HowHardError


class ModelDouble:
    identity = "retrieval-protocol-double-v1"
    dimensions = 32

    def __init__(self):
        self.embedded, self.ranked = [], []

    def encode(self, texts, *, query=False):
        self.embedded.extend(texts)
        return np.asarray([np.frombuffer(hashlib.sha256(t.encode()).digest(), dtype=np.uint8)
                           for t in texts], dtype=np.float32) - 127

    def score(self, query, documents):
        self.ranked.append((query, list(documents)))
        terms = set(query.casefold().split())
        return [len(terms & set(d.casefold().split())) for d in documents]


def pipeline(config):
    model = ModelDouble()
    return RetrievalPipeline(config.state_root, embedder=model, reranker=model), model


def test_past_run_retains_joint_action_failure_usage_and_expert_identity(project):
    engine = Engine(project(rows=[row(1, task="Y", task_text="Repair ledger rounding",
                                     outcome="silent", tokens=812, seconds=9, cost_usd=.03)]))
    pending = engine.prepare("Y", "Repair ledger rounding", native_candidates=inventory())
    evidence = pending["request"]["evidence"]
    assert len(evidence) == 1
    assert {k: evidence[0][k] for k in ("model", "effort", "outcome", "tokens", "seconds", "cost_usd")} == {
        "model": "claude-opus-5-5", "effort": "high", "outcome": "silent", "tokens": 812, "seconds": 9, "cost_usd": .03}
    assert evidence[0]["project_context"] is None
    assert pending["request"]["memories"][0]["id"] == "execution:run-1"
    assert not {"evidence", "memories", "retrieval"} & pending["baseline_request"].keys()
    trace = pending["request"]["retrieval"][0]
    assert trace["request"]["tool"] == "experience.search"
    assert trace["result"]["selected"][0]["key"] == "evidence:run-1"


def test_ten_thousand_runs_narrow_to_200_then_eight_and_reuse_index(project):
    config = project(rows=[row(i, task="Y", task_text=f"Maintain service component{i}") for i in range(10001)])
    stages, model = pipeline(config)
    engine = Engine(config, retrieval=stages)
    pending = engine.prepare("Y", "Maintain service component10000", native_candidates=inventory())
    result = pending["request"]["retrieval"][0]["result"]
    assert result["corpus_size"] == 10001
    assert len(result["candidates"]) == 200
    assert len(model.ranked[0][1]) == 200
    assert len(pending["request"]["evidence"]) == 8
    assert "run-10000" in {r["id"] for r in pending["request"]["evidence"]}
    assert len(pending["request"]["beliefs"]["experts"]) == 9
    before = len(model.embedded)
    Engine(config, retrieval=stages).prepare("Y", "Maintain service component10000", native_candidates=inventory())
    assert len(model.embedded) == before + 1


def test_search_refreshes_edit_remove_restore_and_corrupt_graph(project):
    config = project()
    stages, model = pipeline(config)
    engine = Engine(config, retrieval=stages)
    entries = engine.memory.apply([{"op": "add", "title": f"M{i}", "body": f"component{i}"} for i in range(12)])
    target = entries[-1]
    assert target["id"] in {m["id"] for m in engine.memory.search("component11")}
    revised = engine.memory.edit(target["id"], body="newledgerterm")
    assert engine.memory.search("newledgerterm")[0]["version"] == revised["version"]
    engine.memory.remove(target["id"])
    assert target["id"] not in {m["id"] for m in engine.memory.search("newledgerterm")}
    engine.memory.restore(target["id"])
    assert engine.memory.search("newledgerterm")[0]["id"] == target["id"]
    with sqlite3.connect(config.state_root / "retrieval.sqlite3") as db:
        db.execute("UPDATE indexes SET graph = ?", (b"broken",))
    before = engine.memory.snapshot()
    assert engine.memory.search("newledgerterm")[0]["id"] == target["id"]
    assert engine.memory.snapshot() == before


def test_ranker_input_uses_task_and_context_without_preferring_success(project):
    config = project(rows=[row(i, task="Y", task_text=f"broader migration component{i}", outcome="silent") for i in range(12)])
    stages, model = pipeline(config)
    pending = Engine(config, retrieval=stages).prepare("Y", "migration", native_candidates=inventory())
    query, documents = model.ranked[-1]
    assert "migration" in query
    assert all("broader migration" in d for d in documents)
    assert all("silent" not in d and "claude-opus" not in d for d in documents)
    assert all(e["outcome"] == "silent" for e in pending["request"]["evidence"])


def test_amendment_replaces_execution_expert_without_duplicating_run(project):
    engine = Engine(project(rows=[row(1, task="Y", task_text="ledger rounding")]))
    first = engine.prepare("Y", "ledger rounding", native_candidates=inventory())
    engine.observe("Y", "claude-opus-5-5", "high", "silent", amends="run-1", id="correction")
    pending = engine.prepare("Y", "ledger rounding", native_candidates=inventory())
    assert [(e["id"], e["outcome"]) for e in pending["request"]["evidence"]] == [("run-1", "silent")]
    assert pending["request"]["memories"][0]["version"] != first["request"]["memories"][0]["version"]


def test_execution_experts_learn_only_from_later_observed_joint_actions(project):
    from test_semantic import FixtureLLM
    engine = Engine(project(rows=[row(1, task="Y", task_text="Repair transaction ordering")]), llm=FixtureLLM())
    before = next(e for e in engine.memory_beliefs()["experts"] if "execution:run-1:" in e["key"])
    assert before["evidence_count"] == 0
    choice = engine.pick("Y", "Repair transaction ordering", native_candidates=inventory([("opus", "high")]), seed=1)
    engine.observe_decision(choice["decision_id"], "clean", id="future-run")
    after = next(e for e in engine.memory_beliefs()["experts"] if e["key"] == before["key"])
    assert after["evidence_count"] == 1 and after["log_score"] > before["log_score"]
    assert next(e for e in engine.memory_beliefs()["experts"] if "execution:future-run:" in e["key"])["evidence_count"] == 0


def test_unrelated_history_cannot_dilute_selected_memory_support(project):
    from test_semantic import FixtureLLM
    config = project(rows=[row(i, task="Y", task_text="Unrelated task") for i in range(100)])
    stages, _ = pipeline(config)
    engine = Engine(config, retrieval=stages, llm=FixtureLLM())
    memory = engine.memory.add("Relevant lesson", "A useful task precedent")
    stages.fusion = lambda rankings, limit: [f"memory:{memory['id']}:1"]
    choice = engine.pick("Y", "New task", native_candidates=inventory([("opus", "high")]), seed=1)
    experts = {e["key"]: e["weight"] for e in choice["semantic"]["beliefs"]["experts"]}
    assert experts == {"llm": .8, f"memory:{memory['id']}:1": .2}


def test_empty_candidate_pool_and_failed_reranker_never_fabricate_results(project):
    class Empty:
        def search(self, *args, **kwargs):
            return []
    class Broken:
        identity = "failed-reranker"
        def score(self, *args):
            raise RuntimeError("ranking unavailable")
    config = project(rows=[row(i, task="Y") for i in range(12)])
    stages = RetrievalPipeline(config.state_root, lexical=Empty(), dense=Empty(), reranker=Broken())
    pending = Engine(config, retrieval=stages).prepare("Y", "task", native_candidates=inventory())
    assert pending["request"]["evidence"] == []
    stages, _ = pipeline(config)
    stages.reranker = Broken()
    with pytest.raises(HowHardError, match="ranking unavailable"):
        Engine(config, retrieval=stages).prepare("Y", "task", native_candidates=inventory())


def test_all_fit_needs_no_embedding_or_cross_encoder(project):
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError("small corpus needs no models")
    config = project()
    stages = RetrievalPipeline(config.state_root, embedder=Forbidden(), reranker=Forbidden())
    engine = Engine(config, retrieval=stages)
    assert engine.prepare("Y", "task", native_candidates=inventory())["request"]["evidence"] == []
    engine.memory.add("Lesson", "A useful precedent")
    assert engine.prepare("Y", "task", native_candidates=inventory())["request"]["memories"]


def test_retrieval_and_reranker_are_independently_replaceable(project):
    class CustomRetriever:
        def search(self, documents, query, **kwargs):
            return [d["key"] for d in sorted(documents, key=lambda d: int(d["value"]["id"].split("-")[-1]))][-200:]
    class ReverseRanker:
        identity = "custom-ranker"
        def score(self, query, documents):
            return list(range(len(documents)))
    config = project(rows=[row(i, task="Y", task_text=f"Task {i}") for i in range(20)])
    stages = RetrievalPipeline(config.state_root, lexical=CustomRetriever(), dense=CustomRetriever(),
                               reranker=ReverseRanker(), fusion=lambda rankings, limit: rankings[0][:limit])
    pending = Engine(config, retrieval=stages).prepare("Y", "task", native_candidates=inventory())
    assert [r["id"] for r in pending["request"]["evidence"]] == [f"run-{i}" for i in reversed(range(12, 20))]


@pytest.mark.parametrize("limit", [0, 201, True])
def test_search_refuses_invalid_limit(project, limit):
    with pytest.raises(HowHardError):
        Engine(project()).memory.search("query", limit=limit)
