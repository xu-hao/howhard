# Upgrading to 3.0

3.0 changes the calling-agent protocol and the meaning of selection costs and silent
risk. Update provider adapters and native skill instructions together with the backend.

1. Supply the full nonempty native inventory; the former 32-cell bound is removed.
2. `context` first requests a task-specific capability order and fit cut points.
   Submit `{filter, memory_filters, reason}` to `decide`.
3. Use the returned forecast request ID to obtain calibrated rows only for the fit band.
   Submit `{forecasts, memory_forecasts, reason}` to `decide`. Handle explicit re-filter
   and stronger-only widening requests. `pick` drives these transitions automatically.
4. Compare `J` as expected cost to an accepted result, including repeated loud attempts
   on the same cell. `C_loud` is per-failure overhead in addition to that redo cost.
   Epsilon now caps `P_silent_accepted = P_silent / (1-P_loud)`, rather than the raw
   per-attempt silent probability. Review epsilon and overhead assumptions in an
   existing custom policy.
5. Sampling concentration is derived from forecast spread and recorded expert evidence,
   within the configured bounds. Exploration is restricted by mean cost and uncertainty.
   An unmet cap disables exploration and requires review, including `lowest-bound`.

Receipts use `howhard-decision-v6`, requests `howhard-agent-request-v5`, and semantic
state `howhard-semantic-v4`. New receipts include the capability filter and frozen
accepted-result selection. Earlier receipts are refused for new receipt-bound feedback
and skipped for predictive replay; their raw observation history remains intact.
Reprepare old pending requests. Do not reinterpret old J values or cap results.

The existing observation and memory writer commands remain. A single observed execution
updates both readers; no conversion, second ledger or additional fitness feedback is
needed. Unknown execution usage and worker prices remain unknown.
