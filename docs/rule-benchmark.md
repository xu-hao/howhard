# Rule retrieval benchmark

This opt-in benchmark measures whether a retriever ranks an applicable conditional
memory above plausible distractions. It uses
[pytest-probability 0.3.0](https://github.com/xu-hao/pytest-probability) for ordinary
parametrized assertions, per-case pass fractions, paired strategy comparisons,
latency and a machine-readable report. There are no probability or latency gates.

The corpus contains eight authored rules and forty authored tasks. It is a controlled
diagnostic, not a random sample of project work or evidence of improved routing.
[Real-project evaluation](evaluation.md) remains the field-quality contract. This
change does not modify production memory retrieval, memory schemas or learned weights.

## The comparison

Each rule has free-text applicability conditions, a recommendation and exceptions.
The benchmark compares four strategies on the same corpus and task cases:

| Strategy | Retrieve | Rerank |
| --- | --- | --- |
| `whole-rule` | Embed conditions, exceptions and recommendation | None |
| `condition-only` | Embed conditions and exceptions, omit recommendation | None |
| `whole-rule-rerank` | Same whole-rule retrieval | Cross-encoder over task and full rule |
| `condition-only-rerank` | Same condition retrieval | Cross-encoder over task and full rule |

All retrieval uses exact cosine similarity over the eight rules and retains four
candidates. This isolates representation and reranking from approximate-index errors;
no HNSW or vector-database backend is selected. Both reranking strategies see only the
retrieved candidates. A reranker cannot recover a rule the retrieval stage missed.
Learned reliability weights are not retrieval or relevance signals in this experiment.

The default local models are `BAAI/bge-small-en-v1.5` for embeddings and
`Xenova/ms-marco-MiniLM-L-6-v2` for reranking, run through pinned FastEmbed 0.9.1 on CPU.
They are standard retrieval models, not a proven logical rule-applicability checker.
Failure on a negation or prerequisite is a result to retain. The benchmark calls
no paid LLM API and does not train models or edit production state.

Cases include paraphrased conditions, recommendations containing misleading task
words, explicit negation, exceptions, missing prerequisites and multiple applicable
rules. Labels and rationales are grading-only: neither embedding nor reranking receives
them. Each case has at least one applicable rule. **Abstention when no rule applies is
not measured**; choosing an acceptance threshold requires a separate labeled experiment.
The `clarify` rule covers explicitly underspecified requests, not arbitrary no-match tasks.

## Run from a source checkout

The source distribution includes `benchmarks/`; the backend wheel does not. Create a
benchmark environment in the issue worktree and install its optional dependencies:

```sh
python -m venv .venv-bench
.venv-bench/bin/python -m pip install -e '.[bench]'
```

An explicit opt-in is required before any model import, download or inference. For
example, select a new task-owned output directory and an intentional reusable cache:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
HOWHARD_BENCH_MODELS=1 \
HOWHARD_BENCH_OUTPUT_DIR=/tmp/howhard-rule-run-1 \
HOWHARD_BENCH_CACHE_DIR=/tmp/howhard-rule-models \
agc run --label "rule retrieval benchmark" --resource cpu=1 --resource memory=2147483648 -- \
  .venv-bench/bin/python -m pytest benchmarks/bench_rule_retrieval.py \
  --prob-runs=1 --prob-compare=strategy \
  --prob-json=/tmp/howhard-rule-run-1/probability.json -q
```

Run serially: one session owns one new output directory. Existing output directories
are refused so retries cannot mix independent sessions. If a download or setup fails,
keep its manifest/report, correct the dependency or network problem and retry once
with a fresh output directory; report repeated infrastructure failure instead of
turning it into a relevance score. An explicitly supplied model cache is retained for
reuse; without `HOWHARD_BENCH_CACHE_DIR`, setup owns and removes a temporary cache.
The first real run downloads the model weights. No credentials or service are required.

`HOWHARD_BENCH_EMBED_MODEL` and `HOWHARD_BENCH_RERANK_MODEL` can select other models
supported by FastEmbed. Report their identities and hashes; comparing different model
runs also changes the experiment. Thread counts follow the declared AGCoord CPU claim.
Use `-k` to select cases/strategies for debugging, but label partial reports as partial.

Without `HOWHARD_BENCH_MODELS=1`, collecting the benchmark skips all runs without
importing FastEmbed. Normal `pytest` uses `tests/`; it never collects this benchmark.
The ordinary gate tests the harness and real probability-plugin reporting using
explicit doubles, not downloaded models. pytest-probability is also in the test extra
so those integration contracts are checked in CI; FastEmbed is benchmark-only.

## Results and interpretation

The new output directory retains:

- `manifest.json`: corpus hash/provenance, package/model identities, downloaded model
  file hashes, CPU allocation, setup duration and setup failures.
- `observations.jsonl`: every executed task/strategy, candidate IDs, scores, final order,
  metrics, measured warm duration and pass/fail/error classification. Observations are
  written before assertions, so incorrect rankings remain inspectable.
- `probability.json`: pytest-probability's per-case/run results, latency, aggregates,
  paired comparisons and final exit status. Setup failures remain plugin errors.

Each run records candidate recall, top-1 applicability accuracy, reciprocal rank and
nDCG over the four returned candidates. The assertion passes only when **all** gold
rules were retrieved and the highest-ranked rule is applicable. Multiple gold rules
need not have an invented strict order. Inspect the separate metrics when the combined
assertion fails: missing candidates and poor reranking require different remedies.

Per-run duration includes query embedding and optional reranking. Model download/load,
warmup and both corpus embeddings are setup work, reported separately. CPU compute cost
and token usage are unmeasured; `cost_usd` remains null in observations. The probability
plugin's default zero cost means no spend was reported, not that compute was free.

Start with one run per deterministic local-model case. Repeating identical predictions
does not create independent evidence or additional task coverage. Use more distinct
cases to probe generalization; use repeated runs when comparing a genuinely stochastic
backend and preserve each real inference. No prediction-result cache is used here.

The plugin's intervals and comparisons describe the chosen authored inputs under its
sampling assumptions. They do not establish population performance, causal memory
benefit, routing superiority, rare-failure safety or a calibrated applicability
probability. Freeze cases and labels before comparing models, retain every failure and
avoid tuning a representation or threshold against the same cases used to claim a win.

## Initial diagnostic observation

One CPU run on 2026-10-30 used the frozen forty cases with dataset SHA-256
`6c5b52d7a5c35dc199cd7a1c41b37bbd6c7abed6846798367c7a66aeb459451b`,
the default models, FastEmbed 0.9.1, ONNX Runtime 1.31.1 and pytest-probability 0.3.0.
Each task/strategy ran once; there were 160 observations and no infrastructure errors.

| Strategy | Combined criterion passed | Mean candidate recall | Mean reciprocal rank |
| --- | --- | --- | --- |
| Whole rule | 25/40 | 97.5% | 0.773 |
| Condition only | 24/40 | 100% | 0.754 |
| Whole rule + rerank | 28/40 | 97.5% | 0.819 |
| Condition only + rerank | 28/40 | 100% | 0.829 |

All four strategies ranked an inapplicable rule first on all five documentation-only
cases that mention executable actions without performing them. The four underspecified
requests mentioning billing, deployment, concurrency or dependencies also failed in
all arms. Splitting the condition text was not sufficient to resolve those distinctions.

For either reranking arm versus whole-rule retrieval, the plugin reported an observed
+7.5 percentage-point difference with a paired bootstrap 95% interval of −2.5 to +17.5
points on these authored cases. That comparison is inconclusive; no threshold or case
was changed to manufacture a passing result. The overall benchmark exited 1 because
55 ranking criteria failed, which is distinct from a broken benchmark harness or a
failed ordinary software gate. These observations motivate further applicability
experiments, not selection of a production backend or a claim of routing improvement.
