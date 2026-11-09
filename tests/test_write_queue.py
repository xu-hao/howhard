"""Concurrent public clients, with only task-owned processes and stores."""

from contextlib import contextmanager, ExitStack
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

import pytest

from howhard import Engine, WriteDependencyError
from howhard.storage import locked
from howhard.validation import HowHardError


INVENTORY = [{"model": "claude-opus-5-5", "effort": "high"}]


def _response(request):
    if request["kind"] == "filter":
        from conftest import fixture_filter
        return fixture_filter(request)
    return {"forecasts": [dict(family=c["family"], effort=c["effort"], clean=.8,
                              loud=.19, silent=.01, tokens=100, seconds=1)
                          for c in request["cells"]], "memory_forecasts": [], "reason": "protocol double"}


def _operation_client(directory, operation, options, connection):
    cancel = options.get("cancel")
    engine = Engine(directory=directory, state_directory=options.get("store"),
                    write_timeout=options.get("timeout", 5),
                    write_cancel=cancel.is_set if cancel is not None else None)
    connection.send("ready")
    connection.recv()
    try:
        if operation == "memory":
            result = engine.memory.apply([dict(op="add", title=options.get("title", "Lesson"), body="Actual lesson")],
                                         expected_revision=options.get("revision"))
        elif operation == "observe":
            result = engine.observe("task", INVENTORY[0]["model"], "high", options.get("outcome", "clean"),
                                    id=options["id"], amends=options.get("amends"))
        elif operation == "decide":
            result = engine.decide(options["request_id"], options["response"], seed=1)
        elif operation == "learn":
            result = engine.apply_memory_learning(options["id"], options["response"],
                        expected_revision=options["revision"], event_key=options["event_key"])
        elif operation == "attach":
            result = engine.attach_usage(options["id"], options["manifest"], id=options["amendment"])
        elif operation == "hold":
            with locked(engine.config, operation="test holder"):
                connection.send("admitted")
                connection.recv()
                if options.get("fail"):
                    raise RuntimeError("owned failure")
            result = "released"
        connection.send({"result": result})
    except BaseException as exc:
        connection.send({"error": type(exc).__name__, "message": str(exc)})


def _forecast_client(directory, connection):
    class Adapter:
        def complete(self, request):
            connection.send("forecast")
            connection.recv()
            return _response(request)
    engine = Engine(directory=directory, llm=Adapter())
    connection.send("ready")
    connection.recv()
    try:
        connection.send({"result": engine.pick("task", native_candidates=INVENTORY)})
    except HowHardError as exc:
        connection.send({"error": str(exc)})


def _embedding_client(directory, connection):
    from howhard.retrieval import RetrievalPipeline
    from test_retrieval import ModelDouble
    class Embedder(ModelDouble):
        blocked = False

        def encode(self, texts, *, query=False):
            if not self.blocked:
                self.blocked = True
                connection.send("embedding")
                connection.recv()
            return super().encode(texts, query=query)
    engine = Engine(directory=directory)
    engine.retrieval = RetrievalPipeline(engine.config.state_root, embedder=Embedder(), reranker=ModelDouble())
    connection.send("ready")
    connection.recv()
    try:
        connection.send({"result": engine.prepare("task", native_candidates=INVENTORY)})
    except HowHardError as exc:
        connection.send({"error": str(exc)})


def _memory_client(directory, connection):
    engine = Engine(directory=directory)
    connection.send("ready")
    connection.recv()
    engine.memory.add("Concurrent lesson", "Must share admission with observations")
    connection.send("written")


@contextmanager
def client(target, *args):
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=target, args=(*args, child))
    process.start()
    child.close()
    try:
        yield process, parent
    finally:
        if process.is_alive():
            process.terminate()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(10)
        parent.close()


def receive(connection):
    assert connection.poll(10), "owned client did not finish"
    return connection.recv()


def queued(engine, count):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        entries = engine.write_status()["entries"]
        if len(entries) == count:
            return entries
        time.sleep(.01)
    pytest.fail(f"expected {count} queue entries; got {entries}")


def test_memory_and_observations_share_write_admission(tmp_path):
    engine = Engine(directory=tmp_path)
    with client(_memory_client, tmp_path) as (process, connection):
        assert connection.poll(10) and connection.recv() == "ready"
        with locked(engine.config):
            connection.send("write")
            assert not connection.poll(.3), "memory bypassed the admitted observation writer"
        assert connection.poll(10) and connection.recv() == "written"
        process.join(10)
        assert process.exitcode == 0
    assert len(engine.memory.list()) == 1


def test_fifo_admission_is_observable_and_readers_bypass_waiters(tmp_path):
    engine = Engine(directory=tmp_path)
    engine.memory.add("Existing", "Visible without a write turn")
    with ExitStack() as stack:
        peers = [stack.enter_context(client(_operation_client, tmp_path, "memory", {"title": str(i)}))
                 for i in range(3)]
        for _, connection in peers:
            assert receive(connection) == "ready"
        with locked(engine.config, operation="test blocker"):
            for i, (_, connection) in enumerate(peers):
                connection.send("go")
                entries = queued(engine, i + 2)
                assert entries[-1]["pid"] == peers[i][0].pid
            # Memory reads make no DDL writes and do not join the queue.
            assert engine.memory.list()[0]["title"] == "Existing"
            assert [e["position"] for e in entries] == [1, 2, 3, 4]
        for process, connection in peers:
            assert "result" in receive(connection)
            process.join(10)
            assert process.exitcode == 0
    assert [e["operation"]["title"] for e in engine.memory.history(None)] == ["Existing", "0", "1", "2"]
    assert engine.write_status()["entries"] == []


@pytest.mark.parametrize("termination", ["timeout", "cancel", "interrupt", "crash"])
def test_waiting_clients_leave_no_blocking_ticket(tmp_path, termination):
    engine = Engine(directory=tmp_path)
    cancel = mp.get_context("spawn").Event()
    options = {"timeout": .2 if termination == "timeout" else 5, "cancel": cancel}
    with client(_operation_client, tmp_path, "memory", options) as (process, connection):
        assert receive(connection) == "ready"
        with locked(engine.config):
            connection.send("go")
            queued(engine, 2)
            if termination == "cancel":
                cancel.set()
            elif termination == "interrupt":
                process_pid = process.pid
                os.kill(process_pid, signal.SIGINT)
            elif termination == "crash":
                process.kill()
                process.join(10)
            if termination != "crash":
                result = receive(connection)
                assert result["error"] == {"timeout": "WriteTimeout", "cancel": "WriteCancelled",
                                            "interrupt": "KeyboardInterrupt"}[termination]
    engine.memory.add("After cancellation", "Queue is usable")
    assert [m["title"] for m in engine.memory.list()] == ["After cancellation"]
    assert engine.write_status()["entries"] == []


@pytest.mark.parametrize("failure", ["crash", "error"])
def test_admitted_client_failure_releases_next_writer(tmp_path, failure):
    engine = Engine(directory=tmp_path)
    with client(_operation_client, tmp_path, "hold", {"fail": True}) as (holder, held):
        assert receive(held) == "ready"
        with client(_operation_client, tmp_path, "memory", {}) as (process, connection):
            assert receive(connection) == "ready"
            held.send("go")
            assert receive(held) == "admitted"
            connection.send("go")
            queued(engine, 2)
            if failure == "crash":
                holder.kill()
                holder.join(10)
            else:
                held.send("fail")
                assert receive(held)["error"] == "RuntimeError"
            assert "result" in receive(connection)
    assert len(engine.memory.list()) == 1


def test_independent_stores_in_one_project_do_not_share_a_write_turn(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    engine = Engine(directory=project, state_directory=tmp_path / "store-a")
    with locked(engine.config):
        with client(_operation_client, project, "memory", {"store": tmp_path / "store-b"}) as (_, connection):
            assert receive(connection) == "ready"
            connection.send("go")
            assert "result" in receive(connection)
    assert not engine.memory.list()


def test_queued_memory_patch_still_rejects_a_stale_revision(tmp_path):
    engine = Engine(directory=tmp_path)
    engine.memory.add("Original", "Revision one")
    with client(_operation_client, tmp_path, "memory", {"revision": 1}) as (_, connection):
        assert receive(connection) == "ready"
        with locked(engine.config):
            connection.send("go")
            queued(engine, 2)
            engine.memory.add("Admitted update", "Revision two")
        result = receive(connection)
        assert result["error"] == "HowHardError" and "stale revision" in result["message"]
    assert len(engine.memory.list()) == 2


def test_queued_decision_revalidates_evidence_before_receipt_commit(tmp_path):
    from howhard.writes import admit
    engine = Engine(directory=tmp_path)
    request = engine.prepare("task", native_candidates=INVENTORY)
    options = {"request_id": request["request_id"], "response": _response(request["baseline_request"])}
    with client(_operation_client, tmp_path, "decide", options) as (_, connection):
        assert receive(connection) == "ready"
        # Hold admission without a JSONL mutation, so the client may read and
        # validate its request before joining the write queue.
        with admit(engine.config.state_root):
            connection.send("go")
            queued(engine, 2)
            engine.observe("task", INVENTORY[0]["model"], "high", "clean", id="new-evidence")
        result = receive(connection)
        assert result["error"] == "HowHardError" and "stale" in result["message"]
    assert not (engine.config.state_root / "decisions").exists()


def test_model_forecasting_does_not_hold_a_write_turn_and_stale_result_is_refused(tmp_path):
    engine = Engine(directory=tmp_path)
    with client(_forecast_client, tmp_path) as (_, connection):
        assert receive(connection) == "ready"
        connection.send("go")
        assert receive(connection) == "forecast"
        assert engine.write_status()["entries"] == []
        engine.memory.add("During forecast", "Independent write completes")
        connection.send("finish")
        assert "stale" in receive(connection)["error"]
    assert not (engine.config.state_root / "decisions").exists()


def test_embedding_does_not_hold_a_write_turn_and_stale_preparation_is_refused(tmp_path):
    engine = Engine(directory=tmp_path)
    engine.memory.apply([dict(op="add", title=f"Lesson {i}", body=f"component{i}") for i in range(9)])
    with client(_embedding_client, tmp_path) as (_, connection):
        assert receive(connection) == "ready"
        connection.send("go")
        assert receive(connection) == "embedding"
        assert not engine.write_status()["entries"]
        engine.memory.add("During embedding", "Independent write completes")
        connection.send("finish")
        assert "stale" in receive(connection)["error"]
    assert not (engine.config.state_root / "requests").exists()


def test_queued_learning_revalidates_the_evidence_event(tmp_path):
    from howhard.writes import admit
    engine = Engine(directory=tmp_path)
    engine.observe("task", INVENTORY[0]["model"], "high", "clean", id="run")
    context = engine.memory_learning_context("run")
    options = {"id": "run", "response": {"operations": [dict(op="add", title="Old", body="Stale lesson")], "reason": "fixture"},
               "revision": context["memory_revision"], "event_key": context["event_key"]}
    with client(_operation_client, tmp_path, "learn", options) as (_, connection):
        assert receive(connection) == "ready"
        with admit(engine.config.state_root):
            connection.send("go")
            queued(engine, 2)
            engine.observe("task", INVENTORY[0]["model"], "high", "silent", amends="run", id="correction")
        result = receive(connection)
        assert "evidence changed" in result["message"]
    assert not engine.memory.list()
    assert len(engine.raw_rows) == 2


def test_external_sqlite_lock_has_typed_terminal_failure_and_preserves_evidence(tmp_path):
    engine = Engine(directory=tmp_path)
    engine.observe("task", INVENTORY[0]["model"], "high", "clean", id="durable")
    engine.memory.add("Keep", "Preserved")
    before = engine.memory.snapshot()
    with sqlite3.connect(engine.memory.path, timeout=0) as db:
        db.execute("BEGIN IMMEDIATE")
        with pytest.raises(WriteDependencyError, match="locked"):
            engine.memory.add("Blocked", "Must fail without endless retries")
    assert engine.memory.snapshot() == before
    assert engine.evidence("durable")["outcome"] == "clean"
    engine.memory.add("After dependency failure", "Released")


def test_simultaneous_observation_retries_append_once_and_conflicts_preserve_history(tmp_path):
    engine = Engine(directory=tmp_path)
    with ExitStack() as stack:
        peers = [stack.enter_context(client(_operation_client, tmp_path, "observe", {"id": "same"})) for _ in range(3)]
        for _, connection in peers:
            assert receive(connection) == "ready"
        for _, connection in peers:
            connection.send("go")
        results = [receive(connection)["result"] for _, connection in peers]
    assert results[0] == results[1] == results[2]
    engine.refresh()
    assert len(engine.raw_rows) == 1
    with pytest.raises(HowHardError, match="conflicting"):
        engine.observe("task", INVENTORY[0]["model"], "high", "silent", id="same")
    assert len(engine.raw_rows) == 1


def test_simultaneous_observation_and_memory_clients_keep_both_writes(tmp_path):
    engine = Engine(directory=tmp_path)
    with client(_operation_client, tmp_path, "observe", {"id": "execution"}) as (_, observation):
        with client(_operation_client, tmp_path, "memory", {}) as (_, memory):
            assert receive(observation) == receive(memory) == "ready"
            observation.send("go")
            memory.send("go")
            assert "result" in receive(observation)
            assert "result" in receive(memory)
    engine.refresh()
    assert len(engine.raw_rows) == len(engine.memory.list()) == 1


def test_queued_usage_attachment_preserves_new_outcome_and_retry_identity(tmp_path):
    from howhard.writes import admit
    from test_execution_accounting import artifact, codex_events
    engine = Engine(directory=tmp_path)
    engine.observe("task", INVENTORY[0]["model"], "high", "clean", id="execution")
    manifest = artifact(tmp_path, codex_events())
    options = {"id": "execution", "manifest": manifest, "amendment": "usage"}
    with client(_operation_client, tmp_path, "attach", options) as (_, connection):
        assert receive(connection) == "ready"
        with admit(engine.config.state_root):
            connection.send("go")
            queued(engine, 2)
            engine.observe("task", INVENTORY[0]["model"], "high", "silent", amends="execution", id="correction")
        recorded = receive(connection)["result"]
    engine.refresh()
    assert engine.evidence("execution")["outcome"] == "silent"
    assert engine.evidence("execution")["tokens"] == 120
    engine.observe("task", INVENTORY[0]["model"], "high", "clean", amends="execution", id="later")
    assert engine.attach_usage("execution", manifest, id="usage") == recorded
    assert len(engine.raw_rows) == 4


@pytest.mark.parametrize("kind", ["guard", "legacy"])
def test_admission_dependencies_share_the_bounded_wait(tmp_path, kind):
    import fcntl
    engine = Engine(directory=tmp_path)
    path = (engine.config.state_root / "write-queue.lock" if kind == "guard"
            else engine.config.path(engine.config.data["journal"]).with_suffix(".lock"))
    with client(_operation_client, tmp_path, "memory", {"timeout": .15}) as (_, connection):
        assert receive(connection) == "ready"
        with path.open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            started = time.monotonic()
            connection.send("go")
            assert receive(connection)["error"] == "WriteTimeout"
            assert time.monotonic() - started < 2
    assert not engine.write_status()["entries"]
    engine.memory.add("After dependency wait", "No stale ticket")


def test_read_snapshot_uses_store_lock_with_different_journals(tmp_path):
    engine = Engine(directory=tmp_path)
    other = Engine(directory=tmp_path, journal_path=engine.config.state_root / "other.jsonl", write_timeout=.1)
    with client(_operation_client, tmp_path, "hold", {}) as (_, connection):
        assert receive(connection) == "ready"
        connection.send("go")
        assert receive(connection) == "admitted"
        from howhard import WriteTimeout
        with pytest.raises(WriteTimeout):
            other.refresh()
        connection.send("release")
        assert receive(connection)["result"] == "released"


def test_queue_status_does_not_initialize_an_empty_store(tmp_path, monkeypatch, capsys):
    from howhard.cli import main
    monkeypatch.chdir(tmp_path)
    assert main(["writes", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["entries"] == []
    assert list(tmp_path.iterdir()) == []


def test_cli_queue_status_and_bounded_json_error(tmp_path):
    engine = Engine(directory=tmp_path)
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    with locked(engine.config):
        status = subprocess.run([sys.executable, "-m", "howhard", "writes", "--json"],
                                cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=10)
        assert status.returncode == 0
        assert len(json.loads(status.stdout)["entries"]) == 1
        result = subprocess.run([sys.executable, "-m", "howhard", "memory", "add", "--title", "T",
                                 "--body", "B", "--write-timeout", ".1", "--json"],
                                 cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=10)
        assert result.returncode == 2
        assert json.loads(result.stderr)["error"]["code"] == "write_timeout"
    assert not engine.memory.list()


def test_git_worktrees_share_write_order_and_keep_request_binding(tmp_path):
    project, sibling = tmp_path / "project", tmp_path / "sibling"
    project.mkdir()
    def git(*args):
        subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True)
    git("init", "-q")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-m", "base")
    git("worktree", "add", "-b", "sibling", str(sibling))
    engine = Engine(directory=project)
    request = engine.prepare("task", native_candidates=INVENTORY)
    with client(_operation_client, sibling, "observe", {"id": "sibling-run"}) as (_, connection):
        assert receive(connection) == "ready"
        with locked(engine.config):
            connection.send("go")
            queued(engine, 2)
        assert "result" in receive(connection)
    engine.refresh()
    assert engine.evidence("sibling-run")["project_context"]["revision"]
    with pytest.raises(HowHardError, match="another worktree"):
        Engine(directory=sibling).decide(request["request_id"], _response(request["baseline_request"]))
