"""Current public contracts reject superseded routing and persistence inputs."""

import json
import subprocess

import pytest

from howhard import Engine
from howhard.cli import main
from howhard.decisions import load
from howhard.validation import HowHardError


INVENTORY = [{"model": "claude-opus-5-5", "effort": "high"}]


@pytest.mark.parametrize("method", ["pick", "summary", "prepare"])
def test_routing_requires_an_explicit_host_inventory(tmp_path, method):
    engine = Engine(directory=tmp_path)
    with pytest.raises(HowHardError, match="candidates"):
        getattr(engine, method)("task")
    assert not (engine.config.state_root / "decisions").exists()


@pytest.mark.parametrize("command", ["pick", "summary", "context"])
def test_cli_requires_inventory_before_creating_state(tmp_path, monkeypatch, command):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        main([command, "task"])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_git_ignores_checkout_local_store(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    old = tmp_path / ".howhard"
    old.mkdir()
    marker = old / "outcomes.jsonl"
    marker.write_bytes(b"old data remains untouched\n")
    engine = Engine(directory=tmp_path)
    assert engine.config.state_root == tmp_path / ".git/howhard"
    assert marker.read_bytes() == b"old data remains untouched\n"


@pytest.mark.parametrize("missing", ["project_identity", "native_candidates", "cost_policy"])
def test_receipts_require_identity_inventory_and_cost_policy(tmp_path, missing):
    engine = Engine(directory=tmp_path)
    decision = engine.pick("task", native_candidates=INVENTORY, seed=1)
    value = load(engine.config, decision["decision_id"])
    value.pop(missing)
    with pytest.raises(HowHardError, match=missing):
        load(engine.config, value)


@pytest.mark.parametrize("version", ["howhard-decision-v1", "howhard-decision-v2", "howhard-decision-v3"])
def test_superseded_receipt_formats_are_refused(tmp_path, version):
    engine = Engine(directory=tmp_path)
    decision = engine.pick("task", native_candidates=INVENTORY, seed=1)
    value = load(engine.config, decision["decision_id"])
    value["format"] = version
    with pytest.raises(HowHardError, match="format"):
        load(engine.config, value)


@pytest.mark.parametrize("missing", ["observed_at", "task_text", "project_context", "context_encoder"])
def test_original_records_require_timestamp_and_context(tmp_path, missing):
    engine = Engine(directory=tmp_path)
    observed = engine.observe("task", "claude-opus-5-5", "high", "clean", id="run")
    observed.pop(missing)
    engine.config.path(engine.config.data["journal"]).write_text(json.dumps(observed) + "\n")
    with pytest.raises(HowHardError, match=missing):
        Engine(directory=tmp_path)


def test_feedback_requires_resolved_effort(tmp_path):
    engine = Engine(directory=tmp_path)
    with pytest.raises(HowHardError, match="effort"):
        engine.observe("task", "claude-opus-5-5", "model_default", "clean")
    assert engine.raw_rows == []


def test_unclaimed_memory_is_refused_even_with_recognizable_schema(tmp_path):
    engine = Engine(directory=tmp_path)
    engine.memory.add("Keep", "An existing memory must not be adopted by schema inspection")
    store = engine.config.state_root
    (store / "store.json").unlink()
    before = {p.name: p.read_bytes() for p in store.iterdir() if p.is_file()}
    with pytest.raises(HowHardError, match="unclaimed memory"):
        Engine(directory=tmp_path)
    assert before == {p.name: p.read_bytes() for p in store.iterdir() if p.is_file()}


def test_prepared_requests_require_current_format_and_worktree_binding(tmp_path):
    from howhard.context import canonical, digest
    engine = Engine(directory=tmp_path)
    prepared = engine.prepare("task", native_candidates=INVENTORY)
    path = engine.config.state_root / "requests" / f"{prepared['request_id']}.json"
    value = json.loads(path.read_bytes())
    assert value["format"] == "howhard-agent-request-v3"
    for changed, match in (({**value, "format": "howhard-agent-request-v1"}, "format"),
                           ({k: v for k, v in value.items() if k != "project_root"}, "project_root")):
        request_id = digest(changed)
        (path.parent / f"{request_id}.json").write_bytes(canonical(changed))
        with pytest.raises(HowHardError, match=match):
            engine.retrieve(request_id, [{"tool": "memory.search", "query": "rollback"}])
