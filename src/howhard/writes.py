"""Store-local FIFO admission. Kernel locks, never PIDs or leases, own tickets."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import secrets
import time

from .validation import HowHardError


class WriteError(HowHardError):
    code = "write_dependency_failed"

    def __init__(self, store, operation, detail, ticket=None):
        self.store, self.operation, self.ticket = str(store), operation, ticket
        super().__init__(f"{self.code}: {operation} at {store}: {detail}")

    def as_dict(self):
        return dict(code=self.code, message=str(self), store=self.store,
                    operation=self.operation, ticket=self.ticket)


class WriteTimeout(WriteError):
    code = "write_timeout"


class WriteCancelled(WriteError):
    code = "write_cancelled"


class WriteDependencyError(WriteError):
    """A lock/filesystem/SQLite dependency failed; no automatic retry."""


@dataclass(frozen=True)
class WritePolicy:
    timeout: float = 30.0
    cancel: object = None

    def __post_init__(self):
        if type(self.timeout) not in (int, float) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise HowHardError("write_timeout: expected finite positive seconds")
        if self.cancel is not None and not callable(self.cancel):
            raise HowHardError("write_cancel: expected a callable returning a cancellation flag")


class Wait:
    def __init__(self, root, policy, operation):
        self.root, self.policy, self.operation = Path(root), policy, operation
        self.deadline = time.monotonic() + policy.timeout
        self.ticket = None
        self.storage_locks = set()

    def check(self):
        if self.policy.cancel is not None and self.policy.cancel():
            raise WriteCancelled(self.root, self.operation, "cancelled before mutation", self.ticket)
        if time.monotonic() >= self.deadline:
            raise WriteTimeout(self.root, self.operation,
                               f"admission exceeded {self.policy.timeout:g}s; inspect howhard writes", self.ticket)

    def pause(self):
        self.check()
        time.sleep(min(.025, max(0, self.deadline - time.monotonic())))


_held = ContextVar("howhard_write_turns", default={})


def current(root):
    return _held.get().get((os.getpid(), str(Path(root).resolve())))


@contextmanager
def file_lock(path, wait, *, shared=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        while True:
            wait.check()
            try:
                fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                wait.pause()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _live(path):
    """A closed ticket is abandoned even if its PID was reused. Never unlink a live one."""
    try:
        with path.open("r+b") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
    except FileNotFoundError:
        pass
    return False


def status(root):
    """Nonblocking, observational snapshot; does not create a store or reap tickets."""
    root = Path(root).resolve()
    entries = []
    try:
        for path in sorted((root / "write-queue").glob("*.json")):
            if not _live(path):
                continue
            try:
                value = json.loads(path.read_bytes())
            except FileNotFoundError:
                continue
            except ValueError:
                continue  # Registration is still writing its metadata.
            entries.append({**value, "position": len(entries) + 1,
                            "state": "head" if not entries else "waiting",
                            "waiting_seconds": max(0, time.time() - value["enqueued_at"])})
    except OSError as exc:
        raise WriteDependencyError(root, "write status", str(exc)) from exc
    return {"store": str(root), "ordering": "fifo-after-registration", "entries": entries}


@contextmanager
def admit(root, policy=None, *, operation="write"):
    root = Path(root).resolve()
    previous = current(root)
    if previous is not None:
        yield previous
        return
    wait = Wait(root, policy or WritePolicy(), operation)
    directory = root / "write-queue"
    ticket, handle, token = None, None, None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with file_lock(root / "write-queue.lock", wait):
            paths = sorted(directory.glob("*.json"))
            # Registration under this tiny guard defines ordering, independent of
            # clock changes. Sequence numbers may restart when the queue is empty.
            try:
                sequence = max((int(p.name.split("-", 1)[0]) for p in paths), default=0) + 1
            except ValueError as exc:
                raise WriteDependencyError(root, operation, "invalid queue ticket name") from exc
            ticket = directory / f"{sequence:020d}-{secrets.token_hex(12)}.json"
            handle = ticket.open("x+b")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            wait.ticket = ticket.stem
            metadata = dict(ticket=ticket.stem, pid=os.getpid(), operation=operation,
                            enqueued_at=time.time())
            handle.write(json.dumps(metadata).encode())
            handle.flush()
        while True:
            wait.check()
            with file_lock(root / "write-queue.lock", wait):
                ahead = False
                for path in sorted(directory.glob("*.json")):
                    if path == ticket:
                        break
                    if _live(path):
                        ahead = True
                    else:
                        path.unlink(missing_ok=True)
                if not ahead:
                    break
            wait.pause()
        key = os.getpid(), str(root)
        token = _held.set({**_held.get(), key: wait})
        yield wait
    except OSError as exc:
        raise WriteDependencyError(root, operation, str(exc), wait.ticket) from exc
    finally:
        if token is not None:
            _held.reset(token)
        try:
            if ticket is not None:
                ticket.unlink(missing_ok=True)
        finally:
            if handle is not None:
                handle.close()
