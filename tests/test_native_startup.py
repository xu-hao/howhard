"""Native retrieval stages must compose safely in a fresh interpreter."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("count", [8, 9])
def test_lazy_hybrid_search_survives_native_runtime_initialization(tmp_path, count):
    script = textwrap.dedent('''
        import json
        from pathlib import Path
        import sys

        if sys.platform != "win32":
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        import numpy as np

        from howhard import RetrievalPipeline

        class Embedder:
            identity = "offline-native-startup-v1"
            dimensions = 4

            def encode(self, texts, *, query=False):
                return np.asarray([[1, 1, len(text), 1] for text in texts], dtype=np.float32)

        class Reranker:
            identity = "native-onnx-import-v1"

            def score(self, query, documents):
                # Load the real inference runtime lazily, as the default reranker does.
                import onnxruntime
                assert "CPUExecutionProvider" in onnxruntime.get_available_providers()
                return [float(i) for i in range(len(documents))]

        root, count = Path(sys.argv[1]), int(sys.argv[2])
        pipeline = RetrievalPipeline(root, embedder=Embedder(), reranker=Reranker())
        documents = [{"key": str(i), "text": "task component " + str(i)} for i in range(count)]
        first = pipeline.search(documents, "task component")
        # Repeat with a new pipeline to cover loading the persisted HNSW graph too.
        repeated = RetrievalPipeline(root, embedder=Embedder(), reranker=Reranker()).search(
            documents, "task component")
        print(json.dumps({
            "selected": [d["key"] for d in first["records"]],
            "repeated": [d["key"] for d in repeated["records"]],
            "mode": first["trace"]["mode"],
            "cached_mode": repeated["trace"]["mode"],
            "onnx_loaded": "onnxruntime" in sys.modules,
            "hnsw_loaded": "usearch.index" in sys.modules,
        }))
    ''')
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    # Preserve the source checkout under test without inheriting native modules from pytest.
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", script, str(tmp_path), str(count)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"native search exited {result.returncode}: {result.stderr}"
    output = json.loads(result.stdout)
    assert len(output["selected"]) == 8
    assert output["selected"] == output["repeated"]
    assert output["mode"] == output["cached_mode"] == ("all-fit" if count == 8 else "hybrid-rerank")
    assert output["onnx_loaded"] is (count > 8)
    assert output["hnsw_loaded"] is (count > 8)
