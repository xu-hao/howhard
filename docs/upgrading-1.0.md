# Upgrading to 1.0

Version 1.0.0 replaces the statistical routing interfaces from the 0.6 series with
LLM forecasts and writable semantic memory. It also adds hybrid execution retrieval,
fixed configurable reranking and the `attribute-failure` skill. This is a breaking
release; the published 0.6.0 and 0.6.1 tags and assets remain available unchanged.

## Install the backend and refresh the skills

Python 3.11+ and a POSIX platform are required. Install the verified wheel from the
[1.0.0 GitHub release](https://github.com/xu-hao/howhard/releases/tag/v1.0.0), or use
the tagged source in an activated virtual environment:

```sh
python -m pip install "git+https://github.com/xu-hao/howhard.git@v1.0.0"
howhard --version
```

The result is `howhard 1.0.0`. For the optional Qwen reranker, install
`howhard[qwen]` from that same source or wheel. The ordinary backend now depends on
FastEmbed and USearch as well as NumPy; ONNX Runtime is installed through FastEmbed.
Larger retrieval searches lazily download pinned embedding and reranker models.
Small collections that already fit the eight-memory budget need neither model load.

Refresh the native plugin separately using its host's
[installation and lifecycle procedure](skills.md). The portable, Claude Code and
Antigravity manifests are also version 1.0.0 and discover both `howhard` and
`attribute-failure`. A Python package upgrade does not refresh a host's cached skills.

## Replace removed routing interfaces

Remove `--llm`, `HOWHARD_LLM`, `--cache`, `Engine(cache_path=...)`, `fit`, posterior/prior
attributes, `export-prior`, `import-prior`, and statistical prior/cache/transfer
configuration. Old family-selection flags, implicit registry-only candidates and
default-effort guessing are unavailable. Invalid configuration fails explicitly.

Follow the [current Quickstart](quickstart.md) and
[native workflow](skills.md#capability-boundaries). Supply the actual host's model
and effort/control inventory through `--candidates FILE` or Python
`native_candidates`. The calling agent prepares context, obtains a fresh independent
baseline, considers retrieved experience, and submits forecasts to `decide`.
Unavailable independent inference stops routing; there is no statistical fallback.
Unknown host charges remain unknown and worker money is omitted uniformly when any
candidate is unpriced. No API price is inferred from a model name.

The default reranker is the compact MiniLM-L4 INT8 ONNX cross-encoder. Select Qwen or
other supported pinned models through the fixed
{ref}`retrieval configuration <fixed-retrieval-configuration>`. The application/user
owns this setting; an agent cannot upgrade its retrieval tier.

## Preserve history and prepare current requests

Before replacing an older installation, retain a backup of its configured state
and project metadata. `howhard paths --json` in that installation reports its roots.
For Git repositories, the current default store is below the common Git directory;
worktrees share that store while capturing their own checkout context. A checkout-local
`.howhard` does not override this default. Explicit configuration and intentional
`--state-dir` selection remain supported; see the [storage contract](reference.md).

Current requests use `howhard-agent-request-v3`, receipts use `howhard-decision-v4`,
and semantic state uses `howhard-semantic-v2`. Earlier requests/receipts are refused
for new execution or feedback, and their forecasts receive no current predictive
credit. Prepare a fresh request and decision for new work. Historical raw observations
and semantic memories remain stored; old posterior/prior artifacts are unused and
left untouched. Unsupported row/context/control formats are refused, not guessed.
If an archived store is incompatible, use an intentional separate fresh store while
retaining the archive. No automatic migration, copying between projects or deletion
is performed.

Keep stable execution IDs and use append-only amendments for later corrections.
The new attribution skill investigates escaped defects using reproductions and Git
history, resolving the original task and execution before amending an outcome.
Blame alone is insufficient and uncertain attribution remains unscored. Retain
base/result commits, PR and acceptance evidence in existing artifacts and observation
notes to support future investigations; see [escaped-defect attribution](skills.md#escaped-defect-attribution).

Published tests and package checks establish implementation behavior and mechanics.
They do not establish comparative routing accuracy or cost savings.
