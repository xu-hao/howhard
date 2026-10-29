from copy import deepcopy
import json
import pytest

from howhard import Engine
from howhard.storage import flush
from howhard.validation import HowHardError
from conftest import row


def test_observe_round_trip_and_two_observes_flush_in_order(project):
    config = project()
    engine = Engine(config)
    log = config.path(config.data["logs"][0]["path"])
    journal = config.path(config.data["journal"])
    before = log.read_bytes()
    a = engine.observe("X", "claude-opus-5-5", "high", "clean", id="a", tokens=45, seconds=1.2, cost_usd=.01, note="note", ticket="ticket")
    b = engine.observe("Y", "claude-sonnet-5-5", "model_default", "loud", cause="depth", id="b")
    assert log.read_bytes() == before
    assert [json.loads(line) for line in journal.read_text().splitlines()] == [a, b]
    assert b["default_effort"] == "high"
    assert engine.flush() == ["a", "b"]
    assert journal.read_bytes() == b""
    assert [json.loads(line) for line in log.read_text().splitlines()] == [a, b]
    assert engine.flush() == []


@pytest.mark.parametrize("change,match", [
    ({"extra": 1}, "extra"), ({"id": ""}, "id"), ({"date": "2026-02-30"}, "date"),
    ({"ticket": 3}, "ticket"), ({"outcome": "maybe"}, "outcome"),
    ({"tokens": 1.2}, "tokens"), ({"tokens": True}, "tokens"),
    ({"seconds": -1}, "seconds"), ({"cost_usd": float("nan")}, "cost_usd"),
    ({"note": None}, "note"), ({"model": "unknown-model"}, "registry"),
    ({"effort": "huge"}, "effort"), ({"default_effort": "high"}, "default_effort"),
    ({"effort": "model_default"}, "default_effort"),
    ({"risk_class": "unknown"}, "risk_class"), ({"task_type": "unknown"}, "task_type"),
    ({"outcome": "loud"}, "cause"), ({"cause": "depth"}, "cause"),
    ({"amends": "later"}, "amends"),
])
def test_malformed_rows_report_line_and_key(project, change, match):
    config = project(rows=[row(0), {**row(1), **change}])
    with pytest.raises(HowHardError, match=match) as error:
        Engine(config)
    assert "line 2" in str(error.value)


def test_missing_key_and_invalid_json_report_line(project):
    config = project()
    path = config.path(config.data["logs"][0]["path"])
    invalid = row(0)
    invalid.pop("tokens")
    path.write_text(json.dumps(invalid) + "\n")
    with pytest.raises(HowHardError, match="tokens"):
        Engine(config)
    path.write_text("{\n")
    with pytest.raises(HowHardError, match="line 1.*JSON"):
        Engine(config)


def test_flush_duplicates_refuses_all_and_preserves_both_files(project):
    config = project(rows=[row(0), row(1)])
    journal = config.path(config.data["journal"])
    journal.write_text("".join(json.dumps(r) + "\n" for r in [row(0), row(1), row(1)]))
    log = config.path(config.data["logs"][0]["path"])
    before = log.read_bytes(), journal.read_bytes()
    with pytest.raises(HowHardError, match="run-0, run-1"):
        flush(config)
    assert (log.read_bytes(), journal.read_bytes()) == before


def test_amendment_chain_retains_original_cell_project_and_cost(project):
    original = row(0, tokens=10000, seconds=4.0, cost_usd=.5)
    amendment = row(1, task="Y", family="haiku", outcome="silent", amends="run-0")
    second = row(2, outcome="silent", amends="run-1")
    config = project(rows=[original, amendment, second])
    engine = Engine(config)
    assert len(engine.rows) == 1
    effective = engine.rows[0]
    assert effective["outcome"] == "silent"
    assert effective["task_type"] == "X" and effective["family"] == "opus"
    assert effective["tokens"] == 10000 and effective["cost_usd"] == .5
    assert effective["project"] == "local"
    assert not (engine.posterior.tokens.mean == engine.prior.tokens.mean).all()


def test_journal_amendment_can_target_an_existing_log_row(project):
    config = project(rows=[row(0, tokens=100)])
    engine = Engine(config)
    engine.observe("X", "claude-opus-5-5", "high", "silent", amends="run-0", id="amend")
    assert len(engine.rows) == 1 and engine.rows[0]["outcome"] == "silent"
    assert engine.flush() == ["amend"]
    assert len(engine.rows) == 1 and engine.rows[0]["tokens"] == 100


def test_flush_amendment_of_a_run_in_an_additional_log(project):
    config = project(settings='[[logs]]\npath = "own.jsonl"\nproject = "local"\n[[logs]]\npath = "other.jsonl"\nproject = "other"\n')
    other = config.path("other.jsonl")
    other.write_text(json.dumps(row(0, tokens=150)) + "\n")
    engine = Engine(config)
    engine.observe("X", "claude-opus-5-5", "high", "silent", amends="run-0", id="amend")
    engine.flush()
    assert len(engine.rows) == 1
    assert engine.rows[0]["project"] == "other"
    assert engine.rows[0]["outcome"] == "silent" and engine.rows[0]["tokens"] == 150


def test_later_amendment_in_same_log_is_refused(project):
    config = project(rows=[row(1, outcome="silent", amends="run-0"), row(0)])
    with pytest.raises(HowHardError, match="later"):
        Engine(config)


def test_amendment_refuses_unknown_later_or_measurements(project):
    for changes in ({"amends": "unknown", "outcome": "silent", "cause": "depth"},
                    {"amends": "run-0", "outcome": "silent", "cause": "depth", "tokens": 1}):
        config = project(rows=[row(0), {**row(1), **changes}])
        with pytest.raises(HowHardError, match="amends"):
            Engine(config)


def test_model_default_historical_resolution_and_new_version(project):
    config = project(rows=[row(0, effort="model_default", default_effort="high")])
    config.data["models"]["new-opus"] = deepcopy(config.data["models"]["claude-opus-5-5"])
    config.data["models"]["claude-opus-5-5"]["default_effort"] = "low"
    engine = Engine(config)
    assert engine.rows[0]["resolved_effort"] == "high"
    before = engine.summary("X")
    config.data["families"]["opus"]["current"] = "new-opus"
    after = Engine(config).summary("X")
    a = {(r["family"], r["effort"]): r for r in before["cells"]}
    for b in after["cells"]:
        assert b["P_loud"] == a[b["family"], b["effort"]]["P_loud"]
