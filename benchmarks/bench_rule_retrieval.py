"""Authored diagnostic cases, not evidence of field routing quality."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import pytest

from benchmarks.rule_retrieval import (
    EMBED_MODEL, RERANK_MODEL, RETRIEVE_K, STRATEGIES, LocalModels, Retriever,
    load_dataset, metrics,
)

DATA = load_dataset()
retriever = None
output = None
owned_cache = None


def setup():
    global retriever, output, owned_cache
    if os.environ.get("HOWHARD_BENCH_MODELS") != "1":
        pytest.skip("opt in with HOWHARD_BENCH_MODELS=1 to download/run local benchmark models")
    if os.environ.get("PYTEST_XDIST_WORKER"):
        raise RuntimeError("run this benchmark serially; each run owns one output directory")
    directory = os.environ.get("HOWHARD_BENCH_OUTPUT_DIR")
    if not directory:
        raise RuntimeError("set HOWHARD_BENCH_OUTPUT_DIR to a new results directory")
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)  # Never mix observations from different sessions.
    cache = os.environ.get("HOWHARD_BENCH_CACHE_DIR")
    if cache is None:
        owned_cache = tempfile.TemporaryDirectory(prefix="howhard-rule-models-")
        cache = owned_cache.name
    threads = int(os.environ.get("AGCOORD_RESOURCE_CPU", "1"))
    if threads < 1:
        raise ValueError("positive CPU allocation required")
    models_config = {"embed_model": os.environ.get("HOWHARD_BENCH_EMBED_MODEL", EMBED_MODEL),
                     "rerank_model": os.environ.get("HOWHARD_BENCH_RERANK_MODEL", RERANK_MODEL)}
    manifest = {"format": "howhard-rule-benchmark-v1", "dataset_sha256": DATA.sha256,
                "provenance": DATA.provenance, "rules": len(DATA.rules), "cases": len(DATA.cases),
                "strategies": STRATEGIES, "retrieve_k": RETRIEVE_K, "threads": threads,
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {name: importlib.metadata.version(name) for name in ("howhard", "fastembed", "onnxruntime", "pytest-probability")},
                "models": models_config, "cost_status": "Local compute cost and tokens are unmeasured. Plugin zero cost means no spend was reported.",
                "timing": "Per-run latency includes query embedding and optional reranking; model loading, warmup and corpus embeddings are setup work.",
                "status": "initializing"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    try:
        models = LocalModels(cache, threads=threads, **models_config)
        models.query("Warm up task retrieval")
        models.rerank("Warm up task retrieval", ["When verifying a change, inspect its behavior."])
        retriever = Retriever(DATA.rules, models)
        manifest["setup_seconds"] = time.monotonic() - started
        files = []
        for path in sorted(Path(cache).rglob("*")):
            if path.is_file():
                sha = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        sha.update(chunk)
                files.append({"path": str(path.relative_to(cache)), "sha256": sha.hexdigest()})
        manifest.update(status="ready", model_files=files)
    except Exception as exc:
        manifest.update(status="setup-error", error_type=type(exc).__name__, setup_seconds=time.monotonic() - started)
        raise
    finally:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if retriever is None and owned_cache is not None:
            owned_cache.cleanup()
            owned_cache = None


def teardown():
    global retriever, owned_cache
    retriever = None
    if owned_cache is not None:
        owned_cache.cleanup()
        owned_cache = None


def _record(row):
    with (output / "observations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("case", [pytest.param(c, id=c.id) for c in DATA.cases])
def bench_rule_ranking(case, strategy):
    started = time.monotonic()
    row = {"case": case.id, "category": case.category, "strategy": strategy,
           "pytest_item": os.environ.get("PYTEST_CURRENT_TEST"), "cost_usd": None}
    try:
        result = retriever.rank(case.task, strategy)
        scores = metrics(result, case.applicable)
    except Exception as exc:
        _record({**row, "status": "error", "error_type": type(exc).__name__, "seconds": time.monotonic() - started})
        if isinstance(exc, AssertionError):
            raise RuntimeError("ranking model failed an internal assertion") from exc
        raise  # Harness/provider errors stay distinct from relevance failures.
    row.update(status="pass" if scores["top1_correct"] and scores["candidate_recall"] == 1 else "fail",
               seconds=time.monotonic() - started, **result, metrics=scores)
    _record(row)  # Keep failures and measurements before asserting.
    assert scores["candidate_recall"] == 1, json.dumps(row)
    assert scores["top1_correct"], json.dumps(row)
