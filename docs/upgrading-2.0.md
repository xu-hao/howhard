# Upgrading to 2.0

Version 2.0.0 adds portable execution accounting and category pricing. This is a
breaking protocol release: requests use `howhard-agent-request-v4`, receipts use
`howhard-decision-v5`, and semantic state uses `howhard-semantic-v3`. Prepare fresh
requests and decisions. Older receipt forecasts are excluded from predictive replay;
historical raw observations remain stored and their `cost_usd` retains its unspecified
basis. No automatic migration, deletion, provider request or distribution upload occurs.

Back up the configured state and project metadata before upgrading. Before publication,
install from a verified source checkout with `python -m pip install /path/to/howhard`.
Once the `v2.0.0` GitHub tag is published, a pinned backend upgrade is:

```sh
python -m pip install --upgrade "git+https://github.com/xu-hao/howhard.git@v2.0.0"
howhard --version
```

Expect `howhard 2.0.0`. A source version and this guide do not assert that a release
already exists. A GitHub release can supply verified wheel/source assets and checksums;
it does not establish PyPI publication. The GitHub-source command above does not
require a PyPI release. Upgrade the backend and installed native plugin separately,
following the [skill lifecycle](skills.md). For a local installation, load the plugin
from the same verified checkout. All native plugin manifests match the Python version
at 2.0.0. Upgrades from 0.6.x also need the [1.0 compatibility changes](upgrading-1.0.md).

Existing scalar token forecasts and optional blended `usd_per_mtok` candidates remain
available. Their money basis is unspecified. A category `rate_card` instead requires
prospective `token_usage` forecasts and a consistent scalar `tokens` total. Do not
substitute API list prices for unknown subscription or host charges. Unknown prices or
mixed money bases exclude worker money uniformly from the comparison.

An original observation may now include `execution_usage`; a measurement correction
may replace that envelope or clear it with null. Older readers that do not understand
the field will refuse those rows. Keep the outcome durable first, collect from a known
owned artifact, and attach with a stable amendment ID. Repeating the same attachment
is safe; a changed correction requires a new amendment ID. Existing measured scalar
fields remain supported and are not reclassified as billed charges.

Read the {ref}`accounting contract <portable-execution-accounting>` and
[host accounting workflow](https://github.com/xu-hao/howhard/blob/main/skills/howhard/references/accounting.md) before collecting.
No capture hook is installed. Missing exact child/session binding leaves usage unknown;
shared environment values, model IDs or similar timestamps do not establish ownership.
Mechanical fixtures verify these contracts and do not demonstrate routing superiority.
