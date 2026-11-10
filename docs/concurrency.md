# Concurrent CLI and API clients

Normal clients submit normal operations. Howhard owns write admission, waiting and
release inside the backend; agents need no acquire/release messages, writer windows,
scratch ledger, external broker or daemon. Fresh Git worktrees use the same repository
store automatically. Independently selected stores have independent queues.

## Admission boundaries

Each store has a FIFO queue in `write-queue/`, protected during registration by
`write-queue.lock`. A client holds an OS advisory lock on its ticket while waiting and
writing. Registration under the short queue guard defines order: no later registered
writer overtakes a live earlier ticket. Arrival at the guard itself has no ordering
guarantee. Nested mutations within an admitted operation share that turn. A sequence
number may restart after the queue empties; the random ticket suffix distinguishes requests.

| Operation | Serialized boundary |
| --- | --- |
| First use | Validate evidence, establish project/store identity; initialize memory schema when needed |
| Native inventories | Validate and merge historical identity metadata |
| `context` | Capture one consistent store snapshot; publish the immutable request even if writers ran during retrieval |
| `retrieve` / stage transitions | Recheck surfaced dependencies before publishing the next immutable request |
| `decide` / `pick` | Commit the receipt after checking its forecast inputs again; audit events have separate turns |
| `observe` / amendments | Read current evidence, validate IDs, provenance and corrections, then append/fsync |
| `usage attach` | Collect the owned source outside admission; validate and append the correction inside admission |
| Memory operations / learning | Recheck current evidence, event key, expected revision/version, then commit the SQLite transaction |
| Retrieval / inference caches | Serialize howhard-owned SQLite cache transactions and atomic response-cache publication |
| `flush` | Validate, append/fsync, then truncate/fsync under one turn |

The queue does not reserve a whole workflow. External LLM forecasts, local embedding,
graph construction, reranking, source collection, worker execution, review and gate
waits happen outside the write turn. Retrieval cache publication is separate from its
expensive computation. Pretrained model loading/downloads belong to the inference
libraries, outside store admission. A caller-selected receipt export is an ordinary
output file, outside backend state coordination.

Admitted mutations also hold `storage.lock` and the existing journal advisory lock.
This keeps JSONL snapshots consistent, including clients selecting different journals
inside one store. Evidence readers bypass the FIFO queue and take shared data locks;
they wait only for an active mutation, with the same bounded waiting policy. Memory
reads use SQLite snapshots without schema writes after initialization. First-use
schema initialization is a write. `writes` inspects tickets without joining the queue,
creating a store, or waiting for data locks.

The independent project identity lock is held only while claiming a new store/anchor,
not across every operation. Existing independent stores of the same project therefore
do not serialize their work through that anchor. Identity is checked before admission
and again after waiting. Use the same updated backend for concurrent clients; FIFO
admission applies to clients using this protocol. POSIX advisory locks and a local
filesystem shared by the participating processes are required. This is not a
cross-machine queue or a lease with a forced expiry during a mutation.

## Waiting, cancellation and failures

Every admission has a finite deadline: 30 seconds by default. The deadline covers queue
registration, time behind other writers, and acquisition of data/identity locks. A
multi-step call can have several admissions; this is not a whole-call time limit or a
transaction across the entire call. An admitted mutation is not forcibly interrupted
when its admission deadline passes. Ordinary clients never need manual lock ownership.

```sh
howhard writes --json
howhard observe task actual-model high clean --native --id actual-execution \
  --write-timeout 10 --json
```

`writes` returns the resolved store, ordering policy and live entries, each with a
ticket, PID, operation, enqueue timestamp, waiting seconds, position and state (`head`
or `waiting`). The head may still be acquiring a data lock; it does not imply a mutation
has begun. Status is a nonblocking, observational snapshot, not a claim protocol.
PIDs are diagnostic metadata; ownership and crash recovery use locks, never PID reuse
heuristics. Wait age uses wall-clock time for display; deadlines use a monotonic clock.

`--write-timeout` is available on store commands. On exhaustion, the client withdraws
its ticket and returns `WriteTimeout`; CLI exit status is 2. With `--json`, admission
errors are JSON on stderr under `error`, with `code`, `message`, `store`, `operation`
and `ticket`. Codes are `write_timeout`, `write_cancelled` and
`write_dependency_failed`. Ctrl-C withdraws the ticket and exits 130. Killing a client
releases its kernel locks; the next writer skips/removes abandoned tickets. A stopped
but living client still owns its ticket; other clients terminate at their own deadlines.
Filesystem/lock errors and SQLite dependency failures have a typed terminal outcome
without an automatic retry loop. An out-of-protocol SQLite writer can cause an immediate
dependency failure rather than extending the queue deadline.

```python
from threading import Event
from howhard import Engine, WriteTimeout, WriteCancelled, WriteDependencyError

cancel = Event()
engine = Engine(write_timeout=10, write_cancel=cancel.is_set)
print(engine.write_status())
# Another thread may call cancel.set() while this client waits.
try:
    engine.memory.add("Observed lesson", "A provisional lesson from actual work")
except (WriteTimeout, WriteCancelled, WriteDependencyError):
    pass  # Reconcile durable state and retry through the normal API when appropriate.
```

Cancellation is checked while acquiring admission and data locks. A callback is not a
rollback request for an already executing mutation. API `KeyboardInterrupt` unwinds the
same cleanup. A process crash during a SQLite transaction rolls it back; existing JSONL
crash/flush recovery rules still apply. Queueing does not make a multi-file flush atomic.

## Revalidation and retries

### Prepared request dependencies

Preparation briefly takes shared data locks to capture evidence, native identities and
one SQLite memory snapshot consistently. Retrieval, capability relevance scoring and
model inference then run against those captured values outside the read lock and FIFO
write turn. Writers may continue while preparation runs; their commits cannot cause
preparation itself to fail as stale. A dependency changed during preparation is checked
when the resulting request is used.

New v5 requests persist the read snapshot and a dependency manifest. It contains exact
content/active versions of surfaced memories, effective-row digests of surfaced evidence
(including execution experts and capability inputs), and the bound configuration digest.
Dependencies accumulate across filtering, extra retrieval, forecasting and widening.
Additional memory/evidence retrieval reads the original snapshot. It cannot silently
replace old content with a newer version. Edits to unsurfaced memories, new observations
and unrelated learning commits leave the request valid. Amendments or usage corrections
to a surfaced execution invalidate it when its effective contents change.

The weight policy is **snapshot**. Both readers use the captured belief/expert scores,
evidence counts and capability observation/recency weights throughout the workflow.
They do not replay concurrent outcomes into an in-flight forecast. The receipt's optional
`request_snapshot` records this policy, the dependency manifest, the captured evidence
hash and memory revision. These are decision-time provenance; the global revision is
not a freshness condition. Previously prepared v5 requests without this metadata retain
their conservative whole-store validation. Existing v6 receipts remain readable.

At load and after write admission, howhard checks the dependencies again before a new
stage or receipt is committed. A genuine conflict raises Python `StaleRequest`, exported
from `howhard`. Its `as_dict()` returns `code: "stale_request"`, the original request ID,
`changes` and `refreshed_request`. Memory changes name the ID, old/new versions and active
states; evidence/configuration changes report old/new digests. `decide` and `retrieve`
with `--json` return this object under `error` on stderr with exit status 2.

The refreshed request restarts filtering because a changed precedent can affect the
fit band as well as its forecasts. It preserves the original task/project snapshot and
explicit inventory, or refreshes a named configured inventory. No old response is applied
automatically. `baseline_unchanged` says whether the failed request's independent baseline
is identical to the returned one; only an identical baseline permits reusing that isolated
answer. Re-evaluate affected memory/evidence reasoning. Refresh is attempted once; if
admission or configuration prevents it, `refresh_error` explains the failure and the
original conflict remains visible. There is no unbounded retry or writer-window protocol.
If recording the failed-decision audit also fails admission, `audit_error` carries its
typed diagnostic without replacing the original `stale_request` or changed dependencies.

For example, an observation and a memory update from two worktrees register in one
store queue. The memory update waits its turn, then checks the expected revision
against current memory. A stale revision still returns the existing actionable
conflict; howhard never silently rebases a patch. Learning also checks the effective
evidence event after admission. A prepared request remains bound to its worktree,
actual candidate inventory and surfaced dependencies. A changed dependency during
forecasting, retrieval or a queue wait returns the structured conflict described above.
Unrelated writes and historical identity additions do not invalidate a disjoint request.

A timeout does not execute later in the background. Resubmit a normal call if needed;
no manual queue cleanup is required. Reuse the same stable observation/attachment ID
and identical inputs. Matching retries return the original row; conflicting retries
are refused. Refresh and regenerate stale decisions or writer patches instead of
resubmitting them unchanged. Keep retries bounded and report repeated failures.

Earlier steps may already be durable when a later admission fails. In particular,
an observed outcome survives failed usage collection or learning. Inspect
`evidence ID --raw` after ambiguous feedback and retry the same ID, rather than
minting another original execution. Append-only evidence, immutable receipts and
transactional memory revisions retain their existing contracts.
