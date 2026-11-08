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
| `context` / `retrieve` | Publish the immutable request artifact after checking evidence and memory again |
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

For example, an observation and a memory update from two worktrees register in one
store queue. The memory update waits its turn, then checks the expected revision
against current memory. A stale revision still returns the existing actionable
conflict; howhard never silently rebases a patch. Learning also checks the effective
evidence event after admission. A prepared request remains bound to its worktree,
actual candidate inventory, evidence and memory revision. Evidence or memory changes
during forecasting, retrieval or a queue wait require a new request. Adding an unrelated
historical model identity alone does not invalidate a disjoint inventory.

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
