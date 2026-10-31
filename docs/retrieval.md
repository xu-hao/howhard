# Retrieving transferable execution experience

The retrieval query is the new **project context and task**. The primary precedents are
effective observed executions, each retaining its original project snapshot, task,
actual model **and** reasoning effort, clean/loud/silent outcome, tokens, time and
cost. Related failures are useful evidence. A narrower or broader task can be relevant;
similarity does not establish that an outcome transfers to an unrun action.

The default pipeline is:

```text
project context + task
          |
     BM25 + HNSW
          |
  reciprocal rank fusion: up to 200
          |
  Qwen3-Reranker-0.6B: up to 8
          |
  calling agent / configured reader
          |
  forecasts for supplied (model, effort) pairs
```

SQLite FTS5 supplies BM25. FastEmbed's `BAAI/bge-small-en-v1.5` supplies local dense
embeddings; USearch supplies HNSW with cosine distance. The cross-encoder is
[`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B), pinned
to revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`. Its instruction ranks usefulness
for model-and-effort prediction, considering project constraints, capabilities,
complexity, scope, and broader/narrower tasks. It scores query/candidate pairs in
batches; it does not receive 200 candidates in one generative LLM prompt.

Execution ranking text contains the task and historical project text/characteristics,
not the chosen model, outcome or cost. Those fields remain attached as evidence for
the final agent. Written lessons also participate, using their title and body.
Retrieval never prefers successful executions simply because they succeeded.
Neither query nor document includes transient project paths, Git fingerprints or
timestamps as similarity features. Dense embedding length follows the embedding
model's tokenizer limit; the default reranker caps pairs at 8,192 tokens. Truncated
model inputs do not replace the original stored snapshots or observed results.

The eight-result budget is shared by observed executions and written lessons. A
corpus that already fits is passed through without loading embedding or reranking
models. Otherwise BM25 and dense search each return up to 200 hits, RRF fuses them to
at most 200, and the cross-encoder orders the final eight. Scores are ordering signals,
not calibrated relevance probabilities: there is no arbitrary confidence cutoff.
Empty candidate pools return no results. Model/index failures are explicit errors.

## Evidence and weights

Observed executions are exposed to the reader as read-only `execution:RUN_ID` memory
experts, with their actual rows in `evidence`. They are derived from canonical outcome
records, not a second editable memory store. Their content version fingerprints the
effective row. Amendments replace that version; the original run is still counted once.
Read the execution through `evidence RUN_ID` or `memory.read` in the retrieval protocol;
memory editing commands operate only on written lessons.

A new execution starts with no prospective predictive credit. Only its explicit
forecast for a **later** observed model/effort pair can update its support. It does not
validate itself, inherit its source run's success as a weight, or supply counterfactual
outcomes. The same no-retrospective-credit rule applies to written lessons.

The decision normalizes learned support over the independent forecast and the memories
that actually supply overrides. Unrelated library growth cannot dilute their influence.
Unselected memories receive no credit. `memory beliefs` still reports library-wide
support; a decision's receipt records the participating, normalized support.

## Replace stages independently

`RetrievalPipeline` is public and `Engine(retrieval=pipeline)` uses the supplied pipeline
for initial routing context, memory search and evidence search. No agent protocol or
canonical outcome record changes when replacing a stage.

```python
from howhard import Engine, RetrievalPipeline, load_config
from howhard.retrieval import QwenReranker

config = load_config()
reranker = QwenReranker(config.state_root / "models", device="cpu", threads=1)
pipeline = RetrievalPipeline(config.state_root, reranker=reranker)
engine = Engine(config, retrieval=pipeline)
```

Each replacement implements the following interface:

| Stage | Interface |
| --- | --- |
| Embedding | `encode(texts, query=False)` returns a finite nonzero float matrix; `identity`, `dimensions` |
| Lexical retrieval | `search(documents, query, limit=...)` returns ordered document keys |
| Dense retrieval | `search(documents, query, embedder=..., limit=...)` returns ordered document keys |
| Fusion | callable `(rankings, limit=...)` returns distinct existing document keys |
| Reranking | `score(query, document_texts)` returns one finite score per candidate; `identity` |

Pass replacements through `embedder=`, `lexical=`, `dense=`, `fusion=` and `reranker=`.
Documents have `key`, `kind`, ranking `text` and original `value`. Keys distinguish
observed runs from lesson versions. An empty candidate list is valid. Stage identities
must change when their representation/model changes; the dense cache depends on the
embedding identity. `QwenReranker` also accepts `model`, `revision`, `instruction`,
`device`, `max_length` and `batch_size` for another compatible cross-encoder. When
changing a model, supply its matching revision and instruction/template support.

The calling agent remains the final reasoning stage, through the existing
`context`/`retrieve`/`decide` protocol. No extra LLM selection call or rule-condition
schema is required. It can ignore irrelevant precedents and return no overrides;
at most eight memory forecasts are accepted.

## Local operation and validation

No server, container or project configuration is needed. Larger-corpus search loads
the models lazily, downloading weights under the state root's `models/` directory.
The Qwen download is approximately 1.21 GB; inference needs additional RAM and runtime
libraries. Keep the pipeline alive in a long-running Python integration to reuse loaded
models; separate CLI processes reload them from the local cache. CPU float32 is the
default, avoiding software BF16 arithmetic on CPUs without native BF16 support. The
in-memory weights alone therefore need approximately 2.4 GB, before working memory.
Local compute has a cost even though no inference API is called. Decision wall-clock
audit includes retrieval; it does not invent a dollar price for local resources.

`retrieval.sqlite3` holds disposable FTS, content-addressed embeddings, and a checked
HNSW graph. Changes in the indexed content or embedding identity rebuild the current
graph while reusing unchanged vectors. Graph and mapping publication is transactional;
corrupt graph bytes are rebuilt. These caches never replace `memory.sqlite3` or outcome
logs. Removed lessons are excluded unless the caller explicitly requests archives.
Prepared requests retain candidate IDs, scores, stage identities and selected IDs;
receipts freeze the selected evidence and forecast inputs.

Ordinary tests exercise the public pipeline with deterministic model doubles, including
a 10,001-run corpus, bounded 200-to-eight selection, swappable stages, amendments,
graph repair, and prospective weighting. These are behavioral checks, not proof of
ranking quality. The main relevance measure for a labeled evaluation is **hit rate at
eight**: whether at least one useful precedent reaches the agent. Candidate recall at
200 and broader/narrower cases diagnose misses. The earlier
[eight-rule diagnostic](rule-benchmark.md) does not establish production-scale accuracy.
