"""Offline benchmark contracts; these doubles establish no retrieval-quality evidence."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from benchmarks.rule_retrieval import Case, Rule, Retriever, load_dataset, metrics

ROOT = Path(__file__).resolve().parents[1]
RULES = (Rule("a", "Condition alpha", "Advice alpha", "Exception alpha"),
         Rule("b", "Condition beta", "Advice beta", "Exception beta"))


class Models:
    def __init__(self):
        self.calls = []

    def documents(self, texts):
        self.calls.append(("documents", texts))
        # Deliberately different whole-rule and condition retrieval, with no gold labels.
        return [[1, 0], [0, 1]] if "Recommendation:" in texts[0] else [[0, 1], [1, 0]]

    def query(self, task):
        self.calls.append(("query", task))
        return [[0, 1]]

    def rerank(self, task, texts):
        self.calls.append(("rerank", task, texts))
        return [3.0 if "Condition alpha" in text else -2.0 for text in texts]


def test_representation_and_reranking_are_distinct_label_blind_stages():
    models = Models()
    retriever = Retriever(RULES, models, retrieve_k=2)
    case = Case("label-hidden", "diagnostic", "A task supplied to models", ("a",), "Never send this rationale to a model")
    whole = retriever.rank(case.task, "whole-rule")
    condition = retriever.rank(case.task, "condition-only")
    reranked = retriever.rank(case.task, "whole-rule-rerank")
    assert whole["ranking"] == ["b", "a"]
    assert condition["ranking"] == ["a", "b"]
    assert reranked["retrieved"] == whole["retrieved"]
    assert reranked["ranking"] == ["a", "b"]
    assert metrics(whole, case.applicable)["top1_correct"] is False
    assert metrics(reranked, case.applicable)["top1_correct"] is True
    assert all(case.rationale not in str(call) and case.id not in str(call) for call in models.calls)
    assert all("Recommendation:" not in text for text in models.calls[1][1])


def test_reranking_cannot_recover_a_rule_missing_from_candidates():
    result = Retriever(RULES, Models(), retrieve_k=1).rank("task", "whole-rule-rerank")
    assert result["ranking"] == ["b"]
    assert metrics(result, ("a",)) == {"candidate_recall": 0, "top1_correct": False,
                                       "reciprocal_rank": 0, "ndcg": 0}


def test_scoring_covers_multiple_applicable_rules_and_rank_positions():
    scores = metrics({"retrieved": ["a", "b", "c"], "ranking": ["b", "c", "a"]}, ("a", "c"))
    assert scores["candidate_recall"] == 1
    assert not scores["top1_correct"]
    assert scores["reciprocal_rank"] == .5
    assert scores["ndcg"] == pytest.approx((1 / math.log2(3) + .5) / (1 + 1 / math.log2(3)))
    partial = metrics({"retrieved": ["a"], "ranking": ["a"]}, ("a", "c"))
    assert partial["candidate_recall"] == .5 and partial["top1_correct"]


@pytest.mark.parametrize("vectors", [[[0, 0], [1, 0]], [[float("nan"), 0], [1, 0]], [[1, 0]], [1, 0]])
def test_invalid_embeddings_are_harness_errors(vectors):
    models = Models()
    models.documents = lambda texts: vectors
    with pytest.raises(ValueError, match="embedding"):
        Retriever(RULES, models)


@pytest.mark.parametrize("scores", [[1], [1, float("inf")], [True, 2], ["wrong", 2]])
def test_invalid_reranker_output_is_a_harness_error(scores):
    models = Models()
    models.rerank = lambda task, texts: scores
    with pytest.raises(ValueError, match="reranker"):
        Retriever(RULES, models).rank("task", "whole-rule-rerank")


def test_corpus_order_does_not_control_ties():
    models = Models()
    models.documents = lambda texts: [[1, 1] for text in texts]
    assert Retriever(tuple(reversed(RULES)), models).rank("task", "whole-rule")["ranking"] == ["a", "b"]


def test_dataset_is_self_contained_and_covers_conditional_failures():
    data = load_dataset()
    assert len(data.cases) >= 20 and len(data.rules) >= 4
    assert {"paraphrase", "action-overlap", "negation", "exception", "missing-prerequisite", "conjunction"} <= {c.category for c in data.cases}
    assert any(len(c.applicable) > 1 for c in data.cases)
    assert len(data.sha256) == 64 and "diagnostic" in data.provenance


@pytest.mark.parametrize("change", ["unknown-rule", "duplicate-case", "missing-rationale", "empty-labels"])
def test_invalid_gold_data_is_refused(tmp_path, change):
    data = json.loads((ROOT / "benchmarks/rule_cases.json").read_text())
    if change == "unknown-rule":
        data["cases"][0]["applicable"] = ["not-a-rule"]
    elif change == "duplicate-case":
        data["cases"].append(data["cases"][0])
    elif change == "missing-rationale":
        del data["cases"][0]["rationale"]
    else:
        data["cases"][0]["applicable"] = []
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="benchmark"):
        load_dataset(path)


def _pytest_env():
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("HOWHARD_BENCH_MODELS", None)
    env["PYTHONPATH"] = str(ROOT)
    return env


def test_explicit_benchmark_collection_is_unarmed_without_model_imports(tmp_path):
    report = tmp_path / "report.json"
    result = subprocess.run([sys.executable, "-c", 'import sys,pytest; sys.modules["fastembed"]=None; raise SystemExit(pytest.main(sys.argv[1:]))',
                             "-p", "pytest_probability.plugin", "benchmarks/bench_rule_retrieval.py", "--prob-runs=2",
                             "--prob-json=" + str(report), "-q"], cwd=ROOT, env=_pytest_env(),
                            capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "320 skipped" in result.stdout
    content = json.loads(report.read_text())
    assert content["records"] == []


def test_probability_reports_wrong_answers_and_errors_separately(tmp_path):
    # Run the real plugin against an isolated tiny benchmark with explicit doubles.
    bench = tmp_path / "bench_contract.py"
    bench.write_text('''import pytest
from benchmarks.rule_retrieval import metrics
@pytest.mark.parametrize("result", ["correct", "incorrect", "broken"])
def bench_grade(result):
    if result == "broken":
        raise RuntimeError("model transport unavailable")
    row = {"retrieved": ["a", "b"], "ranking": ["a", "b"] if result == "correct" else ["b", "a"]}
    assert metrics(row, ("a",))["top1_correct"]
''')
    report = tmp_path / "report.json"
    result = subprocess.run([sys.executable, "-m", "pytest", "-p", "pytest_probability.plugin", str(bench),
                             "--prob-runs=2", "--prob-json=" + str(report), "-q"], cwd=tmp_path, env=_pytest_env(),
                            capture_output=True, text=True, timeout=40)
    assert result.returncode == 1, result.stdout + result.stderr
    rows = json.loads(report.read_text())["rows"]
    assert sorted((row["passes"], row["fails"], row["errors"]) for row in rows) == [(0, 0, 2), (0, 2, 0), (2, 0, 0)]


def test_failed_benchmark_assertion_retains_the_observation(tmp_path, monkeypatch):
    from benchmarks import bench_rule_retrieval as bench
    monkeypatch.setattr(bench, "output", tmp_path)
    monkeypatch.setattr(bench, "retriever", Retriever(RULES, Models()))
    case = Case("wrong-top", "diagnostic", "task", ("a",), "grading only")
    with pytest.raises(AssertionError):
        bench.bench_rule_ranking(case, "whole-rule")
    row = json.loads((tmp_path / "observations.jsonl").read_text())
    assert row["status"] == "fail" and row["ranking"] == ["b", "a"]
    assert row["metrics"]["candidate_recall"] == 1 and not row["metrics"]["top1_correct"]
    assert row["cost_usd"] is None


@pytest.mark.parametrize("error", [RuntimeError("model unavailable"), AssertionError("internal model invariant")])
def test_benchmark_infrastructure_errors_are_retained_as_errors(tmp_path, monkeypatch, error):
    from benchmarks import bench_rule_retrieval as bench
    class Broken:
        def rank(self, task, strategy):
            raise error
    monkeypatch.setattr(bench, "output", tmp_path)
    monkeypatch.setattr(bench, "retriever", Broken())
    with pytest.raises(RuntimeError):
        bench.bench_rule_ranking(Case("broken", "diagnostic", "task", ("a",), "grading only"), "whole-rule")
    row = json.loads((tmp_path / "observations.jsonl").read_text())
    assert row["status"] == "error" and row["error_type"] == type(error).__name__
    assert "metrics" not in row


@pytest.mark.parametrize("setup_fails", [False, True])
def test_benchmark_teardown_removes_only_its_owned_cache(tmp_path, monkeypatch, setup_fails):
    from benchmarks import bench_rule_retrieval as bench
    monkeypatch.setenv("HOWHARD_BENCH_MODELS", "1")
    monkeypatch.setenv("HOWHARD_BENCH_OUTPUT_DIR", str(tmp_path / "results"))
    monkeypatch.delenv("HOWHARD_BENCH_CACHE_DIR", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setattr(bench.importlib.metadata, "version", lambda name: "test-double")
    caches = []
    def create(cache, **kwargs):
        caches.append(Path(cache))
        if setup_fails:
            raise RuntimeError("model initialization failed")
        models = Models()
        models.documents = lambda texts: [[1, 0] for text in texts]
        return models
    monkeypatch.setattr(bench, "LocalModels", create)
    try:
        if setup_fails:
            with pytest.raises(RuntimeError, match="initialization failed"):
                bench.setup()
        else:
            bench.setup()
            assert caches[0].is_dir()
    finally:
        bench.teardown()
    assert not caches[0].exists()
    assert json.loads((tmp_path / "results/manifest.json").read_text())["status"] == ("setup-error" if setup_fails else "ready")
    with pytest.raises(FileExistsError):
        bench.setup()
