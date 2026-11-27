# Retrieving transferable execution experience

The retrieval query is the new **project context and task**. The primary precedents are
effective observed executions, each retaining its original project snapshot, task,
actual model **and** reasoning effort, `clean`/`loud`/`silent`/`interrupted` outcome, tokens, time and
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
  MiniLM-L4 INT8 cross-encoder: up to 8
          |
  calling agent / configured reader
          |
  forecasts for supplied (model, effort) pairs
```

SQLite FTS5 supplies BM25. FastEmbed's `BAAI/bge-small-en-v1.5` supplies local dense
embeddings; USearch supplies HNSW with cosine distance. The CPU default cross-encoder
is [`Xenova/ms-marco-MiniLM-L-4-v2`](https://huggingface.co/Xenova/ms-marco-MiniLM-L-4-v2),
pinned to revision `e8fdba61d478d042b338f7bcf7ba4e48ed7d46d7`, using the pretrained
`onnx/model_quantized.onnx` INT8 artifact through FastEmbed. It scores query/candidate
pairs in batches of 16 with two CPU threads and selects up to eight. This compact
search-relevance model has no custom instruction-following capability. The final
agent assesses how broader/narrower task experience transfers to the new task.

The instruction-aware `QwenReranker` remains replaceable through the same interface,
but is optional because its CPU latency is substantially higher. Neither reranker
receives 200 candidates in one generative LLM prompt.

Users/applications can select a fixed profile through
{ref}`[retrieval] configuration <fixed-retrieval-configuration>`:
`reranker = "fast"` or `reranker = "qwen"`, with optional model/revision/runtime
overrides. The agent cannot request a different tier through retrieval tools, and the
backend never upgrades automatically. Changing owner policy requires a fresh request.

Execution ranking text contains the task and historical project text/characteristics,
not the chosen model, outcome or cost. Those fields remain attached as evidence for
the final agent. Written lessons also participate, using their title and body.
Retrieval never prefers successful executions simply because they succeeded.
Neither query nor document includes transient project paths, Git fingerprints or
timestamps as similarity features. Dense embedding length follows the embedding
model's tokenizer limit; the default reranker's **combined query/document window is
512 tokens**. Longer tasks/project descriptions are truncated by the tokenizer. Truncated
model inputs do not replace the original stored snapshots or observed results.

The eight-result budget is shared by observed executions and written lessons. A
corpus that already fits is passed through without loading embedding or reranking
models. Otherwise BM25 and dense search each return up to 200 hits, RRF fuses them to
at most 200, and the cross-encoder orders the final eight. Scores are ordering signals,
not calibrated relevance probabilities: there is no arbitrary confidence cutoff.
Empty candidate pools return no results. Model/index failures are explicit errors.

These eight results are what the reader is shown. Which recorded runs are counted as
evidence, and how much each weighs, is decided separately and does not depend on them:
see {ref}`which recorded runs count <evidence-candidates>`. The same cross-encoder
scores those runs, so a request with matched history loads it even when the corpus fits
the eight-result budget.

Native runtime initialization stays lazy: dense search initializes ONNX Runtime before
loading USearch, whose globally loaded native symbols can otherwise crash a later ONNX
import. This also protects an ONNX reranker paired with a custom embedder. A search whose
corpus fits the eight-result budget loads neither runtime, and callers need no manual
pre-imports. Initialization order does not change the configured stages or ranking.

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

(evidence-candidates)=
### Which recorded runs count, and how much

Three readers learn from recorded runs: the outcome counts of each forecast cell, the
active-time and token beliefs of each forecast cell, and the capability threshold. All
three read one set of candidate runs, chosen and scored once for the decision.

**The candidates.** A candidate is a recorded run with outcome `clean`, `loud` or
`silent` that belongs to at least one prototype the new task matches. Prototypes only
choose the candidates: a prototype's score and a run's membership weight enter no
weight. A task that matches no prototype has no candidate, and nothing is counted;
there is no fallback search. `interrupted` runs are never candidates.

At most `comparison_cap` (256) candidates are scored, most recent first by
`observed_at`; the rest count nowhere. A count keeps the result independent of machine
load and bounds one decision's local model work. Each candidate is compared with the
new task once, when the request is prepared, on the rows captured for that request and
outside the storage lock.

**The weight of one run.**

```text
sim_i = 1 / (1 + exp(-r_i))
a_i   = 0.5^(days_i / h)
w_i   = m_i * sim_i * delta(c_i, c) * a_i
```

- `i`: a candidate run.
- `r_i`: the reranker's score for the new task's text against run `i`'s own task text.
  Both texts are the task, its category and the project text and characteristics, as
  described above.
- `sim_i`: that score on a scale from 0 to 1.
- `days_i`: days from the run's `observed_at` to the time the request was prepared,
  never below 0.
- `h`: the half-life, `half_life_days` = 14.
- `a_i`: the age factor.
- `m_i`: the run's weight on record, the `weight` configured for its model (1 unless set).
- `c_i`, `c`: the cell that ran `i` and the cell being judged.
- `delta(c_i, c)`: the relatedness discount, 1 for the same cell and 0 for any other.
- `w_i`: the run's weight for cell `c` before the per-ticket limit.

In plain words: a run counts more when its task reads like the new one, and half as
much for every 14 days of age. There is no cut-off: a run with a negative score still
counts, at a small weight. A run counts the same whichever host it ran under.

For example, a run of the same cell with `r = 1.386`, recorded 7 days before the
request, with `m = 1`: `sim = 0.8`, `a = 0.5^(7/14) = 0.7071` and
`w = 0.8 * 0.7071 = 0.5657`.

**The per-ticket limit and the counts.**

```text
W_t = sum of w_i over the candidate runs of ticket t
w_i <- w_i / max(1, W_t)
n_k = sum of w_i over the candidate runs with outcome k
```

- `t`: a ticket, the row's `ticket` label. A row with no ticket is not limited.
- `W_t`: the total weight of ticket `t`'s candidate runs for cell `c`.
- `n_k`: the counted amount of outcome `k`, for `clean`, `loud` and `silent`.

In plain words: all runs of one ticket together count for at most one run, because
retries follow failures.

For example, three runs of one ticket weigh 0.6, 0.5 and 0.4. `W_t = 1.5`, so they
become 0.4, 0.3333 and 0.2667, which sum to 1. A run of another ticket weighs 0.3 and
stays 0.3. With outcomes loud, loud, clean and clean, `n_loud = 0.7333` and
`n_clean = 0.5667`.

**The three readers.**

| Reader | Runs it counts | Weight |
| --- | --- | --- |
| Outcomes | The candidates, for each forecast cell | `w_i` after the limit |
| Active time and tokens | The candidates of the same cell that record an active time | The outcome reader's `w_i` for that run |
| Capability | The candidates whose cell is in the inventory | `m_i * sim_i * strength_i * a_i`, then the limit over this reader's runs |

`strength_i` is the capability reader's factor for the outcome and its cause
([capability filtering](capability.md)); that reader has no `delta`. For example, a loud run with no
recorded cause, `sim = 0.8` and `a = 0.7071` weighs
`0.8 * 0.75 * 0.5 * 0.7071 = 0.2121` in the capability reader. The counts feed the
{ref}`outcome posterior <outcome-posterior>`, and the active-time reader's runs and
weights feed the {ref}`beliefs for active time and tokens <amount-beliefs>`: every such
run updates active time, those that record total `tokens` update the total, and those
with a complete execution usage update each token kind.

**What the receipt stores.** `semantic.evidence_candidates` holds the time the request
was prepared (`prepared_at`), both constants (`half_life_days`, `comparison_cap`), the
number of candidates before the cap (`matched`) and each scored candidate once: `id`,
`family`, `effort`, `outcome`, `active_seconds`, `tokens`, `token_usage` (the count of
each token kind, only when the run's execution usage is complete), `ticket`,
`run_weight`, `score`, `similarity`, `days` and `age_factor`. Each row of `semantic.outcome_observations` is
one counted run of one forecast cell with its `discount`, `ticket_share`
(`1 / max(1, W_t)`, or 1 without a ticket) and `weight`; each row of
`semantic.duration_observations` carries the run's active seconds, `tokens`,
`token_usage` and that weight; and
each row of `capability.evidence` carries its `strength`, `ticket_share` and `weight`.
Loading a `howhard-decision-v11` receipt recomputes every similarity, age factor,
ticket share, weight and count from these values and refuses a mismatch, or more than
`comparison_cap` candidates. Every scored candidate is a dependency of the prepared
request, so amending one makes the request stale.

## Replace stages independently

`RetrievalPipeline` is public and `Engine(retrieval=pipeline)` uses the supplied pipeline
for initial routing context, memory search and evidence search, and its
`reranker.score(query, documents)` to score the evidence candidates. No agent protocol or
canonical outcome record changes when replacing a stage.

```python
from howhard import Engine, RetrievalPipeline, load_config
from howhard.retrieval import FastEmbedReranker

config = load_config()
reranker = FastEmbedReranker(config.state_root / "models", threads=2)
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
`FastEmbedReranker` accepts another compatible ONNX model, matching `revision`,
`model_file`, `threads` and `batch_size`.

To opt into instruction-aware Qwen, install `howhard[qwen]` and pass
`reranker=QwenReranker(config.state_root / "models", device="cpu", threads=2)`.
Its frozen default revision is `e61197ed45024b0ed8a2d74b80b4d909f1255473`, its
combined window is capped at 8,192 tokens, and its instruction explicitly considers
project constraints, capabilities, complexity and broader/narrower tasks. It retains
the same calling-agent forecasting and learned-weight interfaces.

The calling agent remains the final reasoning stage, through the existing
`context`/`retrieve`/`decide` protocol. No extra LLM selection call or rule-condition
schema is required. It can ignore irrelevant precedents and return no overrides;
at most eight memory forecasts are accepted.

## Local operation and validation

No server, container or project configuration is needed. Larger-corpus search loads
the models lazily, downloading weights under the state root's `models/` directory.
Keep the pipeline alive in a long-running Python integration to reuse loaded models;
separate CLI processes reload them from the local cache. The ordinary install uses
ONNX Runtime and requires neither PyTorch nor Sentence Transformers. Optional Qwen
uses those libraries, downloads approximately 1.21 GB, and defaults to CPU float32:
its in-memory weights alone need approximately 2.4 GB before working memory.
Local compute has a cost even though no inference API is called. Decision wall-clock
audit includes retrieval; it does not invent a dollar price for local resources.

A local timing check on a Ryzen 9 5900X capped to two CPU threads, batch size 16 and 200 pairs measured
0.32–0.41 seconds for short pairs and 8.34–8.55 seconds for inputs filling the
512-token window. This met the requested under-15-second warm reranking target in
that check. Initial download/model initialization, indexing and final LLM forecasts
are separate. Three repetitions measure timing variation, not independent quality
evidence. A small four-pair sanity check placed both transaction-related precedents
above unrelated prose/CSS work, as Qwen did; this does not establish equivalent
relevance quality. Longer inputs and other hardware need separate validation.

`retrieval.sqlite3` holds up to 16 disposable FTS5 snapshots, content-addressed
embeddings, and up to 64 checked HNSW graphs so experience, memory, and prototype
searches do not thrash one another across stages; `FastEmbedder` also memoizes repeated
query vectors in memory. Dynamic `(task, context)` prototype matching indexes and queries
canonicalized task-only text (`score > 0` cross-encoder match criterion) so shared
project boilerplate never inflates similarity across unrelated tasks. Changes in the indexed content or embedding identity build a new
graph entry while reusing unchanged vectors. Graph and mapping publication is transactional;
corrupt graph bytes are rebuilt. These caches never replace `memory.sqlite3` or outcome
logs. Removed lessons are excluded unless the caller explicitly requests archives.
Cache transactions use the store's [write queue](concurrency.md). Embedding, graph
construction and reranking run outside admission; only cache publication is serialized.
Prepared requests retain candidate IDs, scores, stage identities and selected IDs;
receipts freeze the selected evidence and forecast inputs.

Ordinary tests exercise the public pipeline with deterministic model doubles, including
a 10,001-run corpus, bounded 200-to-eight selection, swappable stages, amendments,
graph repair, and prospective weighting. These are behavioral checks, not proof of
ranking quality. The main relevance measure for a labeled evaluation is **hit rate at
eight**: whether at least one useful precedent reaches the agent. Candidate recall at
200 and broader/narrower cases diagnose misses. The earlier
[eight-rule diagnostic](rule-benchmark.md) does not establish production-scale accuracy.
