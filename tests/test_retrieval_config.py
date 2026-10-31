"""Reranker policy belongs to project/application settings, never an agent tool call."""

import pytest
import numpy as np

from conftest import inventory, row
from howhard import Engine, RetrievalPipeline, load_config
from howhard.memory import MemoryStore
from howhard.retrieval import FastEmbedReranker, QwenReranker
from howhard.validation import HowHardError


@pytest.mark.parametrize("name,expected", [("fast", FastEmbedReranker), ("qwen", QwenReranker)])
def test_project_selects_fixed_reranker_for_engine_and_store(project, name, expected):
    config = project(settings=f'[retrieval]\nreranker = "{name}"\nthreads = 3\nbatch_size = 7\n')
    engine = Engine(config)
    assert isinstance(engine.retrieval.reranker, expected)
    assert engine.retrieval.reranker.threads == 3
    assert engine.retrieval.reranker.batch_size == 7
    assert engine.retrieval.reranker.model is None  # no load or provider just for configuration
    assert isinstance(MemoryStore(config).retrieval.reranker, expected)


def test_model_overrides_require_an_explicit_matching_revision(project):
    revision = "a" * 40
    config = project(settings=f'[retrieval]\nreranker = "fast"\nmodel = "owner/alternative"\nrevision = "{revision}"\nmodel_file = "onnx/alternate.onnx"\n')
    ranker = Engine(config).retrieval.reranker
    assert (ranker.model_name, ranker.revision, ranker.model_file) == ("owner/alternative", revision, "onnx/alternate.onnx")
    config = project(settings=f'[retrieval]\nreranker = "qwen"\nmodel = "owner/alternative"\nrevision = "{revision}"\nmax_length = 2048\ninstruction = "Match transferable project tasks"\n')
    ranker = Engine(config).retrieval.reranker
    assert ranker.max_length == 2048 and ranker.instruction == "Match transferable project tasks"


def test_configured_reranker_is_fixed_across_initial_and_agent_searches(project, monkeypatch):
    from howhard.retrieval import FastEmbedder
    calls = []
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), self.dimensions), dtype=np.float32)
    def score(self, query, documents):
        calls.append((self, len(documents)))
        return list(range(len(documents)))
    monkeypatch.setattr(FastEmbedder, "encode", encode)
    monkeypatch.setattr(QwenReranker, "score", score)
    engine = Engine(project(settings='[retrieval]\nreranker = "qwen"\n',
                            rows=[row(i, task="Y", task_text="Task") for i in range(9)]))
    engine.memory.apply([{"op": "add", "title": f"Lesson {i}", "body": "Task precedent"} for i in range(9)])
    pending = engine.prepare("Y", "Task", native_candidates=inventory())
    engine.retrieve(pending["request_id"], [{"tool": "memory.search", "query": "Task"},
                                             {"tool": "evidence.search", "query": "Task"}])
    assert [n for _, n in calls] == [18, 9, 9]
    assert all(ranker is engine.retrieval.reranker for ranker, _ in calls)


@pytest.mark.parametrize("settings", [
    'reranker = "automatic"', 'reranker = "qwen"\nmodel_file = "onnx/a.onnx"',
    'reranker = "fast"\ninstruction = "Upgrade"', 'reranker = "fast"\nmax_length = 8192',
    'threads = 0', 'threads = true', 'batch_size = -1', 'batch_size = 1.5',
    'model = "owner/model"', 'revision = "main"',
    'model = "owner/model"\nrevision = "main"', 'model_file = "../outside.onnx"',
    'model_file = "/outside.onnx"', 'tier = "slow"', 'allow_agent_upgrade = true',
])
def test_invalid_retrieval_settings_fail_before_state_is_created(tmp_path, settings):
    path = tmp_path / "howhard.toml"
    path.write_text('on_cap_unmet = "flag-review"\n[retrieval]\n' + settings + '\n')
    with pytest.raises(HowHardError, match="retrieval"):
        load_config(path)
    assert not (tmp_path / ".howhard").exists()


@pytest.mark.parametrize("field", ["tier", "reranker", "model", "config"])
def test_agent_cannot_override_reranker_through_retrieval_request(project, field):
    engine = Engine(project())
    pending = engine.prepare("Y", "task", native_candidates=inventory())
    original = engine.retrieval.reranker
    with pytest.raises(HowHardError, match="unknown"):
        engine.retrieve(pending["request_id"], [{"tool": "memory.search", "query": "task", field: "qwen"}])
    assert engine.retrieval.reranker is original


def test_changing_owner_policy_invalidates_prepared_request(project):
    config = project()
    engine = Engine(config)
    pending = engine.prepare("Y", "task", native_candidates=inventory())
    path = config.base / "howhard.toml"
    path.write_text(path.read_text() + '\n[retrieval]\nreranker = "qwen"\n')
    other = Engine(load_config(path))
    with pytest.raises(HowHardError, match="stale"):
        other.retrieve(pending["request_id"], [{"tool": "memory.search", "query": "task"}])


def test_application_pipeline_injection_takes_precedence_over_file(project):
    config = project(settings='[retrieval]\nreranker = "qwen"\n')
    supplied = RetrievalPipeline(config.state_root)
    assert Engine(config, retrieval=supplied).retrieval is supplied
    assert isinstance(supplied.reranker, FastEmbedReranker)
