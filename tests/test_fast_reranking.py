"""Default CPU routing uses a compact pinned ONNX model, without generative inference."""

import builtins
import pytest

from conftest import inventory, row
from howhard import Engine, RetrievalPipeline
from howhard import retrieval
from howhard.validation import HowHardError
from test_retrieval import ModelDouble


def test_default_cpu_reranker_uses_pinned_int8_weights_and_bounded_batches(project, monkeypatch):
    loaded, downloads, registrations, scored = [], [], [], []

    class CrossEncoderDouble:
        @staticmethod
        def list_supported_models():
            return []

        @staticmethod
        def add_custom_model(**kwargs):
            registrations.append(kwargs)

        def __init__(self, **kwargs):
            loaded.append(kwargs)

        def rerank(self, query, documents, batch_size):
            scored.append((query, list(documents), batch_size))
            return range(len(documents))

    def download(model, **kwargs):
        downloads.append((model, kwargs))
        return "/pinned-model-cache"

    def forbidden(*args):
        raise AssertionError("The default CPU path must not load the slow Qwen reranker")

    monkeypatch.setattr(retrieval.QwenReranker, "score", forbidden)
    monkeypatch.setattr("fastembed.rerank.cross_encoder.TextCrossEncoder", CrossEncoderDouble)
    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    config = project(rows=[row(i, task="Y", task_text=f"Task {i}") for i in range(201)])
    pipeline = RetrievalPipeline(config.state_root, embedder=ModelDouble())
    engine = Engine(config, retrieval=pipeline)
    pending = engine.prepare("Y", "Task", native_candidates=inventory())
    trace = pending["request"]["retrieval"][0]["result"]
    assert len(trace["candidates"]) == 200 and len(pending["request"]["evidence"]) == 8
    assert "MiniLM-L-4-v2" in trace["models"]["reranker"] and "model_quantized.onnx" in trace["models"]["reranker"]
    assert downloads[0][1]["revision"] == "e8fdba61d478d042b338f7bcf7ba4e48ed7d46d7"
    assert registrations[0]["model_file"] == "onnx/model_quantized.onnx"
    assert loaded[0]["providers"] == ["CPUExecutionProvider"]
    assert loaded[0]["threads"] == 2 and scored[0][2] == 16
    assert len(scored[0][1]) == 200
    # Model and tokenizer stay resident across requests in a long-running caller.
    engine.prepare("Y", "A different task", native_candidates=inventory())
    assert len(downloads) == len(loaded) == 1


def test_qwen_option_explains_missing_optional_dependencies(tmp_path, monkeypatch):
    original = builtins.__import__
    def without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("optional runtime absent")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_torch)
    with pytest.raises(HowHardError, match=r"howhard\[qwen\]"):
        retrieval.QwenReranker(tmp_path).score("query", ["document"])
