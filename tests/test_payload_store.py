"""Observable persistence, migration, and summary behavior."""

import json
import sqlite3

import pytest

from howhard import Engine
from howhard.context import canonical, digest
from howhard.decisions import load, summaries
from howhard.payload_store import get
from howhard.validation import HowHardError


INVENTORY = [{"model": "claude-opus-5-5", "effort": "high"}]


def test_new_payloads_are_compressed_and_summaries_are_queryable(tmp_path, monkeypatch):
    engine = Engine(directory=tmp_path)
    request = engine.prepare("task", "preserve evidence", native_candidates=INVENTORY)
    decision = engine.pick("task", "preserve evidence", native_candidates=INVENTORY, seed=3)
    root = engine.config.state_root
    assert not (root / "requests").exists()
    assert not (root / "decisions").exists()
    assert json.loads(get(engine.config, "requests", request["request_id"]))["format"] == "howhard-agent-request-v5"
    assert json.loads(get(engine.config, "decisions", decision["decision_id"]))["semantic"] == load(engine.config, decision["decision_id"])["semantic"]
    rows = summaries(engine.config, task_type="task", needs_review=decision["needs_review"])
    assert len(rows) == 1
    assert rows[0]["decision_id"] == decision["decision_id"]
    assert rows[0]["chosen"] == decision["chosen"]
    assert rows[0]["context_summary"]["task_text"] == "preserve evidence"
    assert rows[0]["cap_met"] == decision["cap_met"]
    assert summaries(engine.config, task_type="other") == []
    with sqlite3.connect(root / "payloads.sqlite3") as db:
        assert db.execute("SELECT length(payload) FROM decisions").fetchone()[0] < len(get(engine.config, "decisions", decision["decision_id"]))
    monkeypatch.setattr("howhard.payload_store.zstd.ZstdDecompressor",
                        lambda: pytest.fail("summary query decompressed a receipt"))
    assert summaries(engine.config, task_type="task")[0]["decision_id"] == decision["decision_id"]


def test_claimed_store_migrates_exact_legacy_bytes_and_removes_loose_files(tmp_path):
    engine = Engine(directory=tmp_path)
    request = engine.prepare("task", "migrate", native_candidates=INVENTORY)
    decision = engine.pick("task", "migrate", native_candidates=INVENTORY, seed=4)
    root = engine.config.state_root
    originals = {}
    with sqlite3.connect(root / "payloads.sqlite3") as db:
        for kind, identifier in (("requests", request["request_id"]), ("decisions", decision["decision_id"])):
            raw = get(engine.config, kind, identifier)
            # Preserve extra whitespace and newline as part of the migrated bytes.
            raw = b"  " + raw + b"\n"
            directory = root / kind
            directory.mkdir()
            (directory / f"{identifier}.json").write_bytes(raw)
            db.execute(f"DELETE FROM {kind} WHERE {kind[:-1]}_id=?", (identifier,))
            originals[kind] = raw
    reopened = Engine(directory=tmp_path)
    for kind, identifier in (("requests", request["request_id"]), ("decisions", decision["decision_id"])):
        assert get(reopened.config, kind, identifier) == originals[kind]
        assert not (root / kind).exists()
    assert reopened._load_request(request["request_id"])["format"] == "howhard-agent-request-v5"
    assert load(reopened.config, decision["decision_id"])["format"] == "howhard-decision-v9"


def test_invalid_legacy_payload_remains_unlinked(tmp_path):
    engine = Engine(directory=tmp_path)
    directory = engine.config.state_root / "decisions"
    directory.mkdir()
    path = directory / ("0" * 64 + ".json")
    path.write_bytes(canonical({"format": "historical"}))
    with pytest.raises(HowHardError, match="content does not match ID"):
        Engine(directory=tmp_path)
    assert path.is_file()


def test_legacy_receipt_versions_and_duplicate_id_bytes_are_preserved(tmp_path):
    engine = Engine(directory=tmp_path)
    directory = engine.config.state_root / "decisions"
    directory.mkdir()
    old = {"format": "howhard-decision-v1", "semantic": {"historical": [1, 2, 3]}}
    identifier = digest(old)
    path = directory / f"{identifier}.json"
    raw = b"\n" + canonical(old) + b"\n"
    path.write_bytes(raw)
    reopened = Engine(directory=tmp_path)
    assert get(reopened.config, "decisions", identifier) == raw
    assert summaries(reopened.config)[0]["decision_id"] == identifier
    assert not path.exists()
    # A duplicate ID with different JSON whitespace keeps both byte streams.
    directory.mkdir()
    path.write_bytes(canonical(old))
    Engine(directory=tmp_path)
    assert not path.exists()
    assert get(reopened.config, "decisions", identifier) == canonical(old)
    with sqlite3.connect(engine.config.state_root / "payloads.sqlite3") as db:
        variant = db.execute("SELECT payload FROM payload_variants WHERE kind='decisions' AND payload_id=?",
                             (identifier,)).fetchone()[0]
    import zstandard as zstd
    assert zstd.ZstdDecompressor().decompress(variant) == raw
