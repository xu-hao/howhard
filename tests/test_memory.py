from copy import deepcopy
import pytest

from howhard import Engine
from howhard.validation import HowHardError


def test_add_edit_remove_restore_preserves_versions_and_history(project):
    engine = Engine(project())
    memory = engine.memory.add("Migration depth", "Cross-service migrations need more reasoning.")
    assert engine.memory.read(memory["id"])["body"] == memory["body"]
    revised = engine.memory.edit(memory["id"], body="Cross-service migrations without integration tests need more reasoning.")
    assert revised["version"] == 2
    assert engine.memory.read(memory["id"], version=1)["body"] == memory["body"]
    engine.memory.remove(memory["id"])
    assert engine.memory.list() == []
    assert engine.memory.read(memory["id"])["active"] is False
    restored = engine.memory.restore(memory["id"])
    assert restored["version"] == 2 and restored["active"]
    assert len(engine.memory.history(memory["id"])) == 4
    assert Engine(engine.config).memory.read(memory["id"])["body"] == revised["body"]


def test_merge_split_and_search_are_real_operations(project):
    engine = Engine(project())
    a = engine.memory.add("Contracts", "Contract changes across services need boundary tests.")
    b = engine.memory.add("Migrations", "Database migrations need rollback tests.")
    merged = engine.memory.merge([a["id"], b["id"]], title="Boundary validation", body="Contract changes and migrations need integration and rollback checks.")
    assert {m["id"] for m in engine.memory.list()} == {merged["id"]}
    assert set(merged["parents"]) == {a["id"], b["id"]}
    parts = engine.memory.split(merged["id"], [{"title": "Contracts", "body": "Cross-service contract changes need integration tests."}, {"title": "Migrations", "body": "Database migrations need rollback tests."}])
    assert len(parts) == 2
    assert engine.memory.read(merged["id"])["active"] is False
    assert engine.memory.search("rollback")[0]["id"] == parts[1]["id"]


def test_memory_batches_are_atomic_and_reject_stale_writes(project):
    engine = Engine(project())
    before = engine.memory.snapshot()
    with pytest.raises(HowHardError):
        engine.memory.apply([{"op": "add", "title": "Valid", "body": "valid"}, {"op": "remove", "id": "missing"}])
    assert engine.memory.snapshot() == before
    engine.memory.add("Another", "Concurrent update")
    with pytest.raises(HowHardError, match="stale"):
        engine.memory.apply([{"op": "add", "title": "Old view", "body": "stale"}], expected_revision=before["revision"])
    assert len(engine.memory.list()) == 1


def test_memory_evidence_is_validated_and_survives_removal(project):
    engine = Engine(project())
    with pytest.raises(HowHardError, match="evidence"):
        engine.memory.add("Invented", "unsupported", evidence_ids=["unknown-run"])
    engine.observe("Y", "claude-opus-5-5", "high", "loud", id="real-run", note="boundary failure")
    entry = engine.memory.add("Boundary", "Boundary failures require additional reasoning.", evidence_ids=["real-run"])
    engine.memory.remove(entry["id"])
    assert engine.memory.read(entry["id"])["evidence_ids"] == ["real-run"]
    assert engine.rows[0]["note"] == "boundary failure"
