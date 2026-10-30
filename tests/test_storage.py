from copy import deepcopy
import json
import numpy as np
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


@pytest.mark.parametrize("initial,corrections", [
    ("clean", ["loud"]), ("loud", ["clean"]),
    ("clean", ["loud", "silent", "clean"]),
])
def test_outcome_corrections_replace_original_evidence_once(project, initial, corrections):
    original = row(0, effort="model_default", default_effort="high", outcome=initial,
                   tokens=10000, seconds=4.0, cost_usd=.5)
    config = project(rows=[original])
    config.data["models"][original["model"]]["weight"] = .4
    engine = Engine(config)
    original_effective = deepcopy(engine.rows[0])
    target = original["id"]
    for index, outcome in enumerate(corrections):
        correction = engine.observe("Y", "claude-sonnet-5-5", "low", outcome,
                                    amends=target, id=f"correction-{index}")
        target = correction["id"]
        cause = None if outcome == "clean" else "unrecorded"
        assert engine.rows == [{**original_effective, "outcome": outcome, "cause": cause}]
        assert engine.raw_rows[0] == {**original, "project": "local"}
        expected_config = project(rows=[{**original, "outcome": outcome, "cause": cause}])
        expected_config.data["models"][original["model"]]["weight"] = .4
        expected = Engine(expected_config)
        for block in ("outcome", "tokens", "time"):
            actual_fit, expected_fit = getattr(engine.posterior, block), getattr(expected.posterior, block)
            assert actual_fit.labels == expected_fit.labels
            np.testing.assert_allclose(actual_fit.mean, expected_fit.mean, atol=1e-6, rtol=0)
            np.testing.assert_allclose(actual_fit.covariance, expected_fit.covariance, atol=1e-6, rtol=0)
    engine.flush()
    assert Engine(config).rows == engine.rows


def test_journal_amendment_can_target_an_existing_log_row(project):
    config = project(rows=[row(0, tokens=100)])
    engine = Engine(config)
    engine.observe("X", "claude-opus-5-5", "high", "silent", amends="run-0", id="amend")
    assert len(engine.rows) == 1 and engine.rows[0]["outcome"] == "silent"
    assert engine.flush() == ["amend"]
    assert len(engine.rows) == 1 and engine.rows[0]["tokens"] == 100


@pytest.mark.parametrize("outcome", ["clean", "loud", "silent"])
def test_flush_amendment_of_a_run_in_an_additional_log(project, outcome):
    config = project(settings='[[logs]]\npath = "own.jsonl"\nproject = "local"\n[[logs]]\npath = "other.jsonl"\nproject = "other"\n')
    other = config.path("other.jsonl")
    other.write_text(json.dumps(row(0, tokens=150)) + "\n")
    engine = Engine(config)
    engine.observe("X", "claude-opus-5-5", "high", outcome, amends="run-0", id="amend")
    engine.flush()
    assert len(engine.rows) == 1
    assert engine.rows[0]["project"] == "other"
    assert engine.rows[0]["outcome"] == outcome and engine.rows[0]["tokens"] == 150


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


@pytest.mark.parametrize("outcome", ["clean", "loud", "silent"])
@pytest.mark.parametrize("field", ["tokens", "seconds", "cost_usd"])
def test_corrections_refuse_measurements_without_appending(project, outcome, field):
    config = project(rows=[row(0)])
    engine = Engine(config)
    with pytest.raises(HowHardError, match="amends"):
        engine.observe("X", "claude-opus-5-5", "high", outcome,
                       amends="run-0", **{field: 1})
    assert len(Engine(config).raw_rows) == 1


@pytest.mark.parametrize("outcome,cause", [("clean", "depth"), ("loud", None), ("silent", "unknown")])
def test_corrections_preserve_strict_cause_validation(project, outcome, cause):
    engine = Engine(project(rows=[row(0)]))
    with pytest.raises(HowHardError, match="cause"):
        engine.observe("X", "claude-opus-5-5", "high", outcome, cause=cause, amends="run-0")


def test_cli_appends_clean_and_loud_corrections(tmp_path, monkeypatch, capsys):
    from howhard.cli import main
    monkeypatch.chdir(tmp_path)
    arguments = ["bookkeeping", "claude-opus-5-5", "high"]
    assert main(["observe", *arguments, "clean", "--id", "original", "--tokens", "100", "--json"]) == 0
    capsys.readouterr()
    for outcome, target in (("loud", "original"), ("clean", "loud")):
        assert main(["observe", *arguments, outcome, "--id", outcome, "--amends", target, "--json"]) == 0
        correction = json.loads(capsys.readouterr().out)["observed"]
        assert correction["outcome"] == outcome and correction["amends"] == target
    engine = Engine(directory=tmp_path)
    assert len(engine.rows) == 1
    assert engine.rows[0]["outcome"] == "clean" and engine.rows[0]["tokens"] == 100


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
