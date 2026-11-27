"""Observable persistence, runtime isolation from loose JSON, and standalone migration script behavior."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest
import zstandard as zstd

from howhard import Engine, decisions, payload_store
from howhard.cli import main
from howhard.context import canonical, digest
from howhard.decisions import exists, load, summaries
from howhard.payload_store import get
from howhard.report import report
from howhard.state import validate_store
from howhard.validation import HowHardError
from legacy_receipts import to_v7, to_v8
from test_capability_filter import ProtocolDouble


INVENTORY = [{"model": "claude-opus-5-5", "effort": "high"}]
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate-payloads.py"


def _run_migrate(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_new_payloads_are_compressed_and_summaries_are_queryable(tmp_path, monkeypatch):
    engine = Engine(directory=tmp_path)
    request = engine.prepare("task", "preserve evidence", native_candidates=INVENTORY)
    decision = engine.pick("task", "preserve evidence", native_candidates=INVENTORY, seed=3)
    root = engine.config.state_root
    assert not (root / "requests").exists()
    assert not (root / "decisions").exists()
    assert json.loads(get(engine.config, "requests", request["request_id"]))["format"] == "howhard-agent-request-v7"
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


def test_production_runtime_has_zero_loose_json_awareness_and_no_migrate_cli(tmp_path, monkeypatch):
    assert not hasattr(decisions, "receipt_path")
    assert not hasattr(payload_store, "migrate")

    engine = Engine(directory=tmp_path)
    request = engine.prepare("task", "ignore loose files", native_candidates=INVENTORY)
    decision = engine.pick("task", "ignore loose files", native_candidates=INVENTORY, seed=4)
    engine.observe_decision(decision["decision_id"], "clean", id="run-1")
    root = engine.config.state_root

    req_id = request["request_id"]
    dec_id = decision["decision_id"]
    req_raw = b"  " + get(engine.config, "requests", req_id) + b"\n"
    dec_raw = b"  " + get(engine.config, "decisions", dec_id) + b"\n"

    req_dir = root / "requests"
    dec_dir = root / "decisions"
    req_dir.mkdir()
    dec_dir.mkdir()
    req_file = req_dir / f"{req_id}.json"
    dec_file = dec_dir / f"{dec_id}.json"
    bad_file = dec_dir / ("0" * 64 + ".json")
    req_file.write_bytes(req_raw)
    dec_file.write_bytes(dec_raw)
    bad_file.write_bytes(b"{not valid json")

    with sqlite3.connect(root / "payloads.sqlite3") as db:
        db.execute("DELETE FROM requests WHERE request_id=?", (req_id,))
        db.execute("DELETE FROM decisions WHERE decision_id=?", (dec_id,))

    # Reopening Engine and refreshing does not read, migrate, validate, or unlink loose files.
    reopened = Engine(directory=tmp_path)
    reopened.refresh()
    assert req_file.read_bytes() == req_raw
    assert dec_file.read_bytes() == dec_raw
    assert bad_file.read_bytes() == b"{not valid json"
    assert get(reopened.config, "requests", req_id) is None
    assert get(reopened.config, "decisions", dec_id) is None
    assert not exists(reopened.config, dec_id)

    with pytest.raises(HowHardError, match=f"decision: cannot read valid receipt {dec_id}"):
        load(reopened.config, dec_id)
    with pytest.raises(HowHardError, match="request_id: unavailable or invalid agent request"):
        reopened.retrieve(req_id, [{"tool": "memory.search", "query": "loose"}])

    stats = report(reopened)
    assert stats["missing_receipts"] == ["run-1"]
    assert stats["filter_quality"]["decisions"] == 0

    # A directory with only legacy loose directories is not recognized as a howhard store.
    legacy_only = tmp_path / "legacy-only"
    (legacy_only / "decisions").mkdir(parents=True)
    (legacy_only / "requests").mkdir(parents=True)
    with pytest.raises(HowHardError, match="refusing unrelated existing directory"):
        validate_store(legacy_only)

    # No `howhard migrate` CLI subcommand exists.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["migrate"])
    assert exc.value.code == 2


def test_standalone_migrate_payloads_script_migrates_losslessly_and_makes_payloads_readable(tmp_path):
    engine = Engine(directory=tmp_path, llm=ProtocolDouble())
    request = engine.prepare("task", "migrate", native_candidates=INVENTORY)
    decision = engine.pick("task", "migrate", native_candidates=INVENTORY, seed=4)
    engine.observe_decision(decision["decision_id"], "clean", id="run-1")
    root = engine.config.state_root

    req_id = request["request_id"]
    dec_id = decision["decision_id"]
    current_receipt = load(engine.config, dec_id)
    v8_receipt = to_v8(current_receipt)
    v7_receipt = to_v7(current_receipt)
    v1_receipt = {"format": "howhard-decision-v1", "semantic": {"historical": [1, 2, 3]}}
    v8_id, v7_id, v1_id = digest(v8_receipt), digest(v7_receipt), digest(v1_receipt)

    req_raw = b"  " + get(engine.config, "requests", req_id) + b"\n"
    dec_raw = b"  " + get(engine.config, "decisions", dec_id) + b"\n"
    v8_raw = canonical(v8_receipt) + b"\n"
    v7_raw = canonical(v7_receipt) + b"\n"
    v1_raw = b"\n" + canonical(v1_receipt) + b"\n"

    (root / "requests").mkdir()
    (root / "decisions").mkdir()
    (root / "requests" / f"{req_id}.json").write_bytes(req_raw)
    (root / "decisions" / f"{dec_id}.json").write_bytes(dec_raw)
    (root / "decisions" / f"{v8_id}.json").write_bytes(v8_raw)
    (root / "decisions" / f"{v7_id}.json").write_bytes(v7_raw)
    v1_path = root / "decisions" / f"{v1_id}.json"
    v1_path.write_bytes(v1_raw)
    fixed_mtime = 1700000000
    os.utime(v1_path, (fixed_mtime, fixed_mtime))

    event_time = "2026-11-20T12:34:56+00:00"
    (root / "decision-events.jsonl").write_text(
        json.dumps({"decision_id": dec_id, "at_utc": event_time, "status": "completed", "seconds": 0.1}) + "\n"
    )

    with sqlite3.connect(root / "payloads.sqlite3") as db:
        db.execute("DELETE FROM requests")
        db.execute("DELETE FROM decisions")

    res = _run_migrate("--directory", tmp_path, "--batch-size", 1, "--json")
    assert res.returncode == 0, res.stderr
    summary = json.loads(res.stdout)
    expected_raw = len(req_raw) + len(dec_raw) + len(v8_raw) + len(v7_raw) + len(v1_raw)
    assert summary["store"] == str(root)
    assert summary["requests"] == 1
    assert summary["decisions"] == 4
    assert summary["raw_bytes"] == expected_raw
    assert 0 < summary["compressed_bytes"] < expected_raw

    assert not (root / "requests").exists()
    assert not (root / "decisions").exists()

    reopened = Engine(directory=tmp_path)
    assert get(reopened.config, "requests", req_id) == req_raw
    assert get(reopened.config, "decisions", dec_id) == dec_raw
    assert get(reopened.config, "decisions", v8_id) == v8_raw
    assert get(reopened.config, "decisions", v7_id) == v7_raw
    assert get(reopened.config, "decisions", v1_id) == v1_raw

    assert reopened._load_request(req_id)["format"] == "howhard-agent-request-v7"
    assert exists(reopened.config, dec_id)
    assert load(reopened.config, dec_id)["format"] == "howhard-decision-v11"
    assert load(reopened.config, v8_id)["format"] == "howhard-decision-v8"
    assert load(reopened.config, v7_id)["format"] == "howhard-decision-v7"

    by_id = {row["decision_id"]: row for row in summaries(reopened.config)}
    assert by_id[dec_id]["created_at"] == event_time
    assert by_id[v1_id]["created_at"] == datetime.fromtimestamp(fixed_mtime, timezone.utc).isoformat()

    stats = report(reopened)
    assert stats["missing_receipts"] == []
    assert stats["filter_quality"]["decisions"] == 3

    # A duplicate ID with different JSON whitespace preserves both exact byte streams in payload_variants.
    (root / "decisions").mkdir()
    v1_path.write_bytes(canonical(v1_receipt))
    res2 = _run_migrate("--directory", tmp_path, "--state-dir", root)
    assert res2.returncode == 0, res2.stderr
    assert "decisions=1" in res2.stdout or "1 decision" in res2.stdout
    assert not (root / "decisions").exists()
    assert get(reopened.config, "decisions", v1_id) == canonical(v1_receipt)
    with sqlite3.connect(root / "payloads.sqlite3") as db:
        variant = db.execute(
            "SELECT payload FROM payload_variants WHERE kind='decisions' AND payload_id=?",
            (v1_id,),
        ).fetchone()[0]
    assert zstd.ZstdDecompressor().decompress(variant) == v1_raw


def test_standalone_migrate_payloads_refuses_invalid_symlink_unclaimed_and_foreign_stores(tmp_path):
    # 1. Unclaimed store with outcomes.jsonl + decisions/*.json but no store.json is refused without mutation.
    unclaimed_proj = tmp_path / "unclaimed"
    unclaimed_store = unclaimed_proj / ".howhard"
    (unclaimed_store / "decisions").mkdir(parents=True)
    (unclaimed_store / "outcomes.jsonl").write_text("")
    v1 = {"format": "howhard-decision-v1", "semantic": {}}
    v1_id = digest(v1)
    unclaimed_file = unclaimed_store / "decisions" / f"{v1_id}.json"
    unclaimed_file.write_bytes(canonical(v1))
    res_unclaimed = _run_migrate("--directory", unclaimed_proj, "--json")
    assert res_unclaimed.returncode == 2
    assert unclaimed_file.is_file()
    assert not (unclaimed_store / "payloads.sqlite3").exists()

    # 2. Foreign store (store.json belongs to a different project identity) is refused without mutation.
    foreign_proj = tmp_path / "foreign"
    engine_foreign = Engine(directory=foreign_proj)
    foreign_store = engine_foreign.config.state_root
    (foreign_store / "store.json").write_text(
        json.dumps({"format": "howhard-store-v2", "project": "f" * 32}) + "\n"
    )
    (foreign_store / "decisions").mkdir()
    foreign_file = foreign_store / "decisions" / f"{v1_id}.json"
    foreign_file.write_bytes(canonical(v1))
    res_foreign = _run_migrate("--directory", foreign_proj)
    assert res_foreign.returncode == 2
    assert foreign_file.is_file()

    # 3. Corrupt / mismatched content ID file in a claimed store is refused and left unlinked.
    claimed_proj = tmp_path / "claimed"
    engine = Engine(directory=claimed_proj)
    root = engine.config.state_root
    dec_dir = root / "decisions"
    dec_dir.mkdir()
    bad_path = dec_dir / ("0" * 64 + ".json")
    bad_path.write_bytes(canonical({"format": "historical"}))
    res_bad = _run_migrate("--directory", claimed_proj)
    assert res_bad.returncode == 2
    assert "content does not match ID" in res_bad.stderr
    assert bad_path.is_file()
    bad_path.unlink()

    # 4. Symlinked payload file is refused and left unlinked.
    target = tmp_path / "external.json"
    target.write_bytes(canonical(v1))
    sym_file = dec_dir / f"{v1_id}.json"
    sym_file.symlink_to(target)
    res_sym_file = _run_migrate("--directory", claimed_proj)
    assert res_sym_file.returncode == 2
    assert "non-regular payload" in res_sym_file.stderr
    assert sym_file.is_symlink() and target.is_file()
    sym_file.unlink()
    dec_dir.rmdir()

    # 5. Symlinked legacy directory is refused without mutation.
    ext_dir = tmp_path / "ext-decisions"
    ext_dir.mkdir()
    (root / "decisions").symlink_to(ext_dir)
    res_sym_dir = _run_migrate("--directory", claimed_proj)
    assert res_sym_dir.returncode == 2
    assert "refusing legacy directory symlink" in res_sym_dir.stderr
    assert (root / "decisions").is_symlink()
