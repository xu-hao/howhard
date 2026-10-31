"""Replaceable retrieval stages over canonical experiences and semantic lessons.

Indexes and embeddings are disposable caches. Ranking never changes outcome evidence.
"""

from copy import deepcopy
import hashlib
import json
import re
import sqlite3
from threading import Lock
from time import monotonic

import numpy as np

from .context import digest
from .validation import HowHardError

CANDIDATES = 200
KEEP = 8
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-4-v2"
RERANK_REVISION = "e8fdba61d478d042b338f7bcf7ba4e48ed7d46d7"
RERANK_FILE = "onnx/model_quantized.onnx"
QWEN_MODEL = "Qwen/Qwen3-Reranker-0.6B"
QWEN_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
_MODEL_REGISTRATION = Lock()
INSTRUCTION = (
    "Rank past project tasks and execution experiences by their usefulness for estimating "
    "model and reasoning effort performance on the new task. Match project constraints, "
    "required capabilities, complexity and scope, including broader and narrower tasks. "
    "Related failed executions are as useful as related successes. Do not prefer records "
    "merely because they mention a particular model or a successful outcome."
)


def task_text(task_type, context):
    project = context.get("project_context") or {}
    return (f"Task: {context.get('task_text', '')}\nTask category: {task_type}\n"
            f"Project context: {project.get('text', '')}\n"
            f"Project characteristics: {json.dumps(project.get('parameters', {}), sort_keys=True)}")


def records(rows=(), memories=()):
    """Keep action/outcome/usage attached, but out of the similarity representation."""
    return ([{"key": "evidence:" + r["id"], "kind": "evidence", "value": r,
              "text": task_text(r["task_type"], r)} for r in rows]
            + [{"key": f"memory:{m['id']}:{m['version']}", "kind": "memory", "value": m,
                "text": m["title"] + "\n" + m["body"]} for m in memories])


def reciprocal_rank_fusion(rankings, *, limit):
    scores = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0.0) + 1 / (60 + rank)
    return sorted(scores, key=lambda k: (-scores[k], k))[:limit]


def validate_ranking(ranking, known, where):
    if (not isinstance(ranking, list) or any(not isinstance(k, str) or k not in known for k in ranking)
            or len(ranking) != len(set(ranking)) or len(ranking) > CANDIDATES):
        raise HowHardError(f"retrieval.{where}: invalid candidate keys")
    return ranking


class FastEmbedder:
    identity = "fastembed-0.9.1:" + EMBED_MODEL
    dimensions = 384

    def __init__(self, cache_directory):
        self.cache_directory, self.model = cache_directory, None

    def encode(self, texts, *, query=False):
        if self.model is None:
            from fastembed import TextEmbedding
            self.model = TextEmbedding(EMBED_MODEL, cache_dir=str(self.cache_directory), threads=1)
        method = self.model.query_embed if query else self.model.passage_embed
        return np.asarray(list(method(texts, batch_size=32)), dtype=np.float32)


class FastEmbedReranker:
    """Compact CPU cross-encoder; frozen INT8 weights, no generative instructions."""

    def __init__(self, cache_directory, *, model=RERANK_MODEL, revision=RERANK_REVISION,
                 model_file=RERANK_FILE, threads=2, batch_size=16):
        self.cache_directory, self.model = cache_directory, None
        self.model_name, self.revision, self.model_file = model, revision, model_file
        self.threads, self.batch_size = threads, batch_size
        fingerprint = digest(["fastembed-0.9.1", model, revision, model_file])
        self.identity = f"{model}@{revision}:{model_file}:{fingerprint}"
        self.alias = "howhard/" + fingerprint

    def score(self, query, documents):
        if not documents:
            return []
        if self.model is None:
            from huggingface_hub import snapshot_download
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            from fastembed.common.model_description import ModelSource
            path = snapshot_download(
                self.model_name, revision=self.revision, cache_dir=str(self.cache_directory),
                allow_patterns=["*.json", "*.txt", self.model_file])
            with _MODEL_REGISTRATION:
                if not any(m["model"] == self.alias for m in TextCrossEncoder.list_supported_models()):
                    TextCrossEncoder.add_custom_model(
                        model=self.alias, sources=ModelSource(hf=self.model_name),
                        model_file=self.model_file, license="apache-2.0")
            self.model = TextCrossEncoder(
                model_name=self.alias, specific_model_path=str(path), threads=self.threads,
                providers=["CPUExecutionProvider"])
        return list(self.model.rerank(query, documents, batch_size=self.batch_size))


class QwenReranker:
    identity = f"{QWEN_MODEL}@{QWEN_REVISION}:routing-v1:8192"

    def __init__(self, cache_directory, *, model=QWEN_MODEL, revision=QWEN_REVISION,
                 instruction=INSTRUCTION, device="cpu", max_length=8192, batch_size=4, threads=1):
        self.cache_directory, self.model = cache_directory, None
        self.model_name, self.revision, self.instruction = model, revision, instruction
        self.device, self.max_length, self.batch_size, self.threads = device, max_length, batch_size, threads
        self.identity = digest([model, revision, instruction, max_length, device, "float32" if device == "cpu" else "auto"]) + ":" + model

    def score(self, query, documents):
        if self.model is None:
            try:
                import torch
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise HowHardError("Qwen reranking requires the optional howhard[qwen] dependencies") from exc
            torch.set_num_threads(self.threads)
            self.model = CrossEncoder(
                self.model_name, revision=self.revision, device=self.device, max_length=self.max_length,
                cache_folder=str(self.cache_directory), trust_remote_code=False,
                model_kwargs={"dtype": torch.float32 if self.device == "cpu" else "auto"},
                prompts={"routing": self.instruction}, default_prompt_name="routing")
        return self.model.predict([(query, d) for d in documents], batch_size=self.batch_size,
                                  show_progress_bar=False).tolist()


class BM25Index:
    """SQLite's standard FTS5 BM25; no application-written lexical ranking model."""
    identity = "sqlite-fts5-bm25-v1"

    def __init__(self, path):
        self.path = path

    def search(self, documents, query, *, limit):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=180) as db:
            db.execute("CREATE TABLE IF NOT EXISTS lexical_meta (fingerprint TEXT)")
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(key UNINDEXED, text)")
            fingerprint = digest([(d["key"], d["text"]) for d in documents])
            previous = db.execute("SELECT fingerprint FROM lexical_meta").fetchone()
            if previous is None or previous[0] != fingerprint:
                db.execute("DELETE FROM search")
                db.executemany("INSERT INTO search(key,text) VALUES (?,?)", [(d["key"], d["text"]) for d in documents])
                db.execute("DELETE FROM lexical_meta")
                db.execute("INSERT INTO lexical_meta VALUES (?)", (fingerprint,))
            # Quote tokens; user input cannot inject FTS syntax. Bound long queries.
            terms = list(dict.fromkeys(re.findall(r"\w+", query.casefold())))[:128]
            if not terms:
                return []
            match = " OR ".join('"' + t + '"' for t in terms)
            return [r[0] for r in db.execute(
                "SELECT key FROM search WHERE search MATCH ? ORDER BY bm25(search),key LIMIT ?", (match, limit))]


class HNSWIndex:
    """USearch HNSW with persistent, content-addressed vectors and atomic graph cache."""
    identity = "usearch-hnsw-cos-f32-v1"

    def __init__(self, path):
        self.path = path

    def search(self, documents, query, *, embedder, limit):
        from usearch.index import Index
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=180) as db:
            db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS indexes (identity TEXT PRIMARY KEY, graph BLOB NOT NULL, checksum TEXT NOT NULL)")
            db.execute("BEGIN IMMEDIATE")
            keys = [digest([embedder.identity, d["text"]]) for d in documents]
            identity = digest([self.identity, embedder.identity, [(d["key"], k) for d, k in zip(documents, keys)]])
            index = Index(ndim=embedder.dimensions, metric="cos", dtype="f32", expansion_search=max(256, limit))
            cached = db.execute("SELECT graph, checksum FROM indexes WHERE identity=?", (identity,)).fetchone()
            loaded = False
            if cached is not None and hashlib.sha256(cached[0]).hexdigest() == cached[1]:
                try:
                    index.load(cached[0])
                    loaded = len(index) == len(documents) and index.ndim == embedder.dimensions
                except (RuntimeError, ValueError):
                    pass
            if not loaded:
                vectors, missing = {}, {}
                for key, document in zip(keys, documents):
                    row = db.execute("SELECT vector FROM vectors WHERE key=?", (key,)).fetchone()
                    vector = np.frombuffer(row[0], dtype=np.float32) if row is not None and len(row[0]) == 4 * embedder.dimensions else None
                    if vector is None or not np.isfinite(vector).all() or not np.linalg.norm(vector):
                        missing[key] = document["text"]
                    else:
                        vectors[key] = vector
                if missing:
                    encoded = _vectors(embedder.encode(list(missing.values())), len(missing), embedder.dimensions)
                    for key, vector in zip(missing, encoded):
                        vectors[key] = vector
                        db.execute("INSERT OR REPLACE INTO vectors VALUES (?,?)", (key, vector.tobytes()))
                matrix = np.asarray([vectors[k] for k in keys], dtype=np.float32)
                index = Index(ndim=embedder.dimensions, metric="cos", dtype="f32", expansion_search=max(256, limit))
                index.add(np.arange(len(documents), dtype=np.uint64), matrix, threads=1)
                graph = index.save()
                # Only one current graph is necessary; corpus vectors survive edits.
                db.execute("DELETE FROM indexes")
                db.execute("INSERT INTO indexes VALUES (?,?,?)", (identity, graph, hashlib.sha256(graph).hexdigest()))
            vector = _vectors(embedder.encode([query], query=True), 1, embedder.dimensions)[0]
            matches = index.search(vector, count=min(limit, len(documents)), threads=1)
            return [documents[int(i)]["key"] for i in matches.keys]


def _vectors(values, count, dimensions):
    values = np.asarray(values, dtype=np.float32)
    if (values.shape != (count, dimensions) or not np.isfinite(values).all()
            or not np.all(np.linalg.norm(values, axis=1) > 0)):
        raise HowHardError("retrieval.embedder: invalid vectors")
    return values


class RetrievalPipeline:
    """Inject any stage without changing stored records or the caller's LLM protocol.

    lexical.search(documents,query,limit=...)->keys
    dense.search(documents,query,embedder=...,limit=...)->keys
    embedder.encode(texts,query=False)->matrix, identity, dimensions
    fusion(rankings,limit=...)->keys
    reranker.score(query,document_texts)->scores, identity
    """

    def __init__(self, state_root, *, embedder=None, lexical=None, dense=None, fusion=None, reranker=None):
        path = state_root / "retrieval.sqlite3"
        self.embedder = embedder if embedder is not None else FastEmbedder(state_root / "models")
        self.lexical = lexical if lexical is not None else BM25Index(path)
        self.dense = dense if dense is not None else HNSWIndex(path)
        self.fusion = fusion if fusion is not None else reciprocal_rank_fusion
        self.reranker = reranker if reranker is not None else FastEmbedReranker(state_root / "models")

    @classmethod
    def from_config(cls, config):
        """Construct fixed application policy; retrieval tool requests cannot change it."""
        from .config import validate_retrieval
        spec = config.data.get("retrieval", {"reranker": "fast"})
        validate_retrieval(spec)
        options = {k: v for k, v in spec.items() if k != "reranker"}
        if spec["reranker"] == "qwen":
            options.setdefault("threads", 2)
            reranker = QwenReranker(config.state_root / "models", **options)
        else:
            reranker = FastEmbedReranker(config.state_root / "models", **options)
        return cls(config.state_root, reranker=reranker)

    def search(self, documents, query, *, limit=KEEP):
        if not isinstance(query, str) or type(limit) is not int or not 1 <= limit <= CANDIDATES:
            raise HowHardError("retrieval.search: expected query string and limit in [1, 200]")
        started = monotonic()
        indexed = {d["key"]: d for d in documents}
        if len(indexed) != len(documents):
            raise HowHardError("retrieval: duplicate document keys")
        trace = {"corpus_size": len(documents), "candidate_limit": CANDIDATES, "keep_limit": limit,
                 "candidates": [], "selected": [], "mode": "all-fit", "models": {}}
        try:
            lexical = validate_ranking(self.lexical.search(documents, query, limit=CANDIDATES), indexed, "lexical") if documents else []
            if len(documents) <= limit:
                candidates = list(dict.fromkeys([*lexical, *sorted(indexed)]))
                scores = None
            else:
                dense = validate_ranking(self.dense.search(documents, query, embedder=self.embedder, limit=CANDIDATES), indexed, "dense")
                candidates = validate_ranking(self.fusion([lexical, dense], limit=CANDIDATES), indexed, "fusion")
                scores = np.asarray(self.reranker.score(query, [indexed[k]["text"] for k in candidates]) if candidates else [], dtype=float)
                if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
                    raise HowHardError("retrieval.reranker: invalid scores")
                trace.update(mode="hybrid-rerank", models={"embedder": self.embedder.identity, "reranker": self.reranker.identity},
                             lexical=lexical, dense=dense)
        except HowHardError:
            raise
        except Exception as exc:
            raise HowHardError(f"retrieval failed ({type(exc).__name__}): {str(exc)[:300]}") from exc
        trace["candidates"] = list(candidates)
        trace["stages"] = {name: getattr(stage, "identity", f"{type(stage).__module__}.{type(stage).__qualname__}")
                           for name, stage in (("lexical", self.lexical), ("dense", self.dense), ("fusion", self.fusion))}
        trace["scores"] = None if scores is None else scores.tolist()
        selected = candidates[:limit] if scores is None else [candidates[i] for i in sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]))[:limit]]
        trace["selected"] = [{"key": k, "score": None if scores is None else float(scores[candidates.index(k)])} for k in selected]
        trace["seconds"] = monotonic() - started
        # Raw scores are ordering signals, not calibrated relevance probabilities.
        return {"records": deepcopy([indexed[k] for k in selected]), "trace": trace}
