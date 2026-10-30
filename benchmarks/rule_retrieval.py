"""Rule-retrieval experiment and label-blind scoring, separate from production state."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

STRATEGIES = ("whole-rule", "condition-only", "whole-rule-rerank", "condition-only-rerank")
DATASET = Path(__file__).with_name("rule_cases.json")
RETRIEVE_K = 4
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class Rule:
    id: str
    applies_when: str
    recommendation: str
    exceptions: str

    def condition_text(self):
        return f"Applies when: {self.applies_when}\nExceptions: {self.exceptions}"

    def full_text(self):
        return self.condition_text() + f"\nRecommendation: {self.recommendation}"


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    task: str
    applicable: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class Dataset:
    rules: tuple[Rule, ...]
    cases: tuple[Case, ...]
    sha256: str
    provenance: str


def load_dataset(path=DATASET):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if set(data) != {"format", "provenance", "rules", "cases"} or data["format"] != "howhard-rule-benchmark-v1":
        raise ValueError("benchmark dataset: unsupported format or fields")
    if not isinstance(data["provenance"], str) or not data["provenance"]:
        raise ValueError("benchmark dataset: provenance required")
    rules, cases = [], []
    for row in data["rules"]:
        if set(row) != {"id", "applies_when", "recommendation", "exceptions"} or any(not isinstance(x, str) or not x for x in row.values()):
            raise ValueError("benchmark rule: expected nonempty text fields")
        rules.append(Rule(**row))
    ids = {r.id for r in rules}
    if not rules or len(ids) != len(rules):
        raise ValueError("benchmark dataset: empty or duplicate rules")
    for row in data["cases"]:
        if set(row) != {"id", "category", "task", "applicable", "rationale"}:
            raise ValueError("benchmark case: unknown or missing fields")
        if any(not isinstance(row[k], str) or not row[k] for k in ("id", "category", "task", "rationale")):
            raise ValueError("benchmark case: expected nonempty text fields")
        labels = row["applicable"]
        if not isinstance(labels, list) or not labels or any(not isinstance(x, str) or x not in ids for x in labels) or len(set(labels)) != len(labels):
            raise ValueError("benchmark case: expected known, distinct applicable rule IDs")
        cases.append(Case(**{**row, "applicable": tuple(labels)}))
    if not cases or len({c.id for c in cases}) != len(cases):
        raise ValueError("benchmark dataset: empty or duplicate cases")
    return Dataset(tuple(rules), tuple(cases), hashlib.sha256(raw).hexdigest(), data["provenance"])


def _unit_vectors(values, count):
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != count or not matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("embedding model: expected finite vectors of one dimension")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("embedding model: zero vector")
    return matrix / norms[:, None]


class Retriever:
    """Exact cosine retrieval isolates representation from ANN approximation errors."""

    def __init__(self, rules, models, *, retrieve_k=RETRIEVE_K):
        if not rules or type(retrieve_k) is not int or retrieve_k < 1:
            raise ValueError("retriever: nonempty rules and positive retrieve_k required")
        if len({r.id for r in rules}) != len(rules):
            raise ValueError("retriever: duplicate rule IDs")
        self.rules = tuple(sorted(rules, key=lambda r: r.id))
        self.models, self.retrieve_k = models, min(retrieve_k, len(rules))
        self.documents = {}
        for representation in ("whole-rule", "condition-only"):
            texts = [r.full_text() if representation == "whole-rule" else r.condition_text() for r in self.rules]
            self.documents[representation] = _unit_vectors(models.documents(texts), len(self.rules))

    def rank(self, task, strategy):
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown benchmark strategy: {strategy}")
        representation = strategy.removesuffix("-rerank")
        query = _unit_vectors(self.models.query(task), 1)[0]
        matrix = self.documents[representation]
        if matrix.shape[1] != query.shape[0]:
            raise ValueError("embedding model: query/document dimension mismatch")
        scores = matrix @ query
        order = sorted(range(len(self.rules)), key=lambda i: (-float(scores[i]), self.rules[i].id))[:self.retrieve_k]
        candidates = [self.rules[i] for i in order]
        retrieved = [r.id for r in candidates]
        rerank_scores = None
        if strategy.endswith("-rerank"):
            rerank_scores = list(self.models.rerank(task, [r.full_text() for r in candidates]))
            if len(rerank_scores) != len(candidates) or any(isinstance(x, bool) or not isinstance(x, (int, float, np.floating)) or not math.isfinite(x) for x in rerank_scores):
                raise ValueError("reranker: expected one finite score per candidate")
            order = sorted(range(len(candidates)), key=lambda i: (-float(rerank_scores[i]), candidates[i].id))
            ranking = [candidates[i].id for i in order]
        else:
            ranking = list(retrieved)
        return {"retrieved": retrieved, "ranking": ranking,
                "retrieval_scores": {self.rules[i].id: float(scores[i]) for i in sorted(range(len(self.rules)))},
                "rerank_scores": dict(zip(retrieved, map(float, rerank_scores))) if rerank_scores is not None else None}


def metrics(observation, applicable):
    """Grade only after inference; labels never enter retrieval or reranking calls."""
    relevant = set(applicable)
    retrieved, ranking = observation["retrieved"], observation["ranking"]
    if not relevant or not ranking or len(retrieved) != len(set(retrieved)) or len(ranking) != len(set(ranking)) or set(ranking) != set(retrieved):
        raise ValueError("metrics: nonempty labels and a permutation of unique candidates required")
    rank = next((i for i, identity in enumerate(ranking, 1) if identity in relevant), None)
    dcg = sum(1 / math.log2(i + 1) for i, identity in enumerate(ranking, 1) if identity in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), len(ranking)) + 1))
    return {"candidate_recall": len(relevant & set(retrieved)) / len(relevant),
            "top1_correct": ranking[0] in relevant,
            "reciprocal_rank": 1 / rank if rank else 0.0,
            "ndcg": dcg / ideal}


class LocalModels:
    """Real CPU models, imported/loaded only by an explicitly armed benchmark."""

    def __init__(self, cache_dir, *, threads=1, embed_model=EMBED_MODEL, rerank_model=RERANK_MODEL):
        from fastembed import TextEmbedding
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.embedding = TextEmbedding(model_name=embed_model, cache_dir=str(Path(cache_dir) / "embedding"),
                                       threads=threads, providers=["CPUExecutionProvider"])
        self.cross_encoder = TextCrossEncoder(model_name=rerank_model, cache_dir=str(Path(cache_dir) / "reranker"),
                                             threads=threads, providers=["CPUExecutionProvider"])

    def documents(self, texts):
        return list(self.embedding.passage_embed(texts))

    def query(self, task):
        return list(self.embedding.query_embed(task))

    def rerank(self, task, texts):
        return list(self.cross_encoder.rerank(task, texts))
