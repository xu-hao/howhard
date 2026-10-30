"""Append-only measurement mechanics; these fixtures do not measure routing quality."""

from copy import deepcopy
import json
import math

import numpy as np
import pytest

from howhard import Engine
from howhard.cli import main
from howhard.report import report
from howhard.validation import HowHardError
from test_semantic import FixtureLLM, expert


CANDIDATES = [{"model": "claude-opus-5-5", "effort": "high", "usd_per_mtok": .6}]


def test_corrections_replace_only_explicit_measurements_and_refit_one_run(project):
    config = project()
    config.data["models"]["claude-opus-5-5"]["weight"] = .4
    engine = Engine(config, llm="off")
    choice = engine.pick("Y", "Measure completed work", native_candidates=CANDIDATES, seed=3)
    engine.observe_decision(choice["decision_id"], "clean", id="original", tokens=200, seconds=8, cost_usd=.4)
    original = deepcopy(engine.evidence("original", raw=True))
    journal = config.path(config.data["journal"])
    original_bytes = journal.read_bytes()
    corrected = engine.observe("Y", "claude-opus-5-5", "high", "clean", amends="original", id="corrected",
                               measurement_corrections={"tokens": 100})
    assert all(corrected[k] is None for k in ("tokens", "seconds", "cost_usd"))
    assert journal.read_bytes().startswith(original_bytes)
    assert engine.evidence("original", raw=True) == original
    effective = engine.evidence("corrected")
    assert len(engine.rows) == 1
    assert (effective["tokens"], effective["seconds"], effective["cost_usd"]) == (100, 8, .4)
    for key in ("id", "decision_id", "model", "effort", "task_text", "project_context"):
        assert effective[key] == original[key]

    expected_config = project()
    expected_config.data["models"]["claude-opus-5-5"]["weight"] = .4
    expected = Engine(expected_config, llm="off")
    expected.observe("Y", "claude-opus-5-5", "high", "clean", id="original", tokens=100, seconds=8, cost_usd=.4,
                     task_text=original["task_text"], project_context=original["project_context"])
    for block in ("outcome", "tokens", "time"):
        actual_fit, expected_fit = getattr(engine.posterior, block), getattr(expected.posterior, block)
        np.testing.assert_allclose(actual_fit.mean, expected_fit.mean, atol=1e-6, rtol=0)
        np.testing.assert_allclose(actual_fit.covariance, expected_fit.covariance, atol=1e-6, rtol=0)

    engine.observe("Y", "claude-opus-5-5", "high", "loud", amends="corrected", id="cleared",
                   measurement_corrections={"seconds": None, "cost_usd": None})
    effective = engine.evidence("cleared")
    assert (effective["tokens"], effective["seconds"], effective["cost_usd"], effective["outcome"]) == (100, None, None, "loud")
    assert report(engine)["worker_cost"] == {"known_usd": 0, "unknown_runs": 1, "total_usd": None}
    np.testing.assert_allclose(engine.posterior.time.mean, engine.prior.time.mean)
    engine.observe("Y", "claude-opus-5-5", "high", "clean", id="other", cost_usd=.2)
    assert report(engine)["worker_cost"] == {"known_usd": .2, "unknown_runs": 1, "total_usd": None}
    engine.flush()
    assert Engine(config, llm="off").evidence("cleared") == effective


@pytest.mark.parametrize("corrections", [None, {}, [], {"other": 1}, {"tokens": True}, {"tokens": 1.2},
                                        {"tokens": -1}, {"seconds": True}, {"seconds": -1},
                                        {"cost_usd": float("nan")}, {"cost_usd": float("inf")}])
def test_invalid_measurement_corrections_refused_without_append(project, corrections):
    engine = Engine(project(), llm="off")
    engine.observe("Y", "claude-opus-5-5", "high", "clean", id="original", tokens=200)
    journal = engine.config.path(engine.config.data["journal"])
    before = journal.read_bytes()
    with pytest.raises(HowHardError, match="measurement_corrections"):
        engine.observe("Y", "claude-opus-5-5", "high", "clean", amends="original", id="invalid",
                       measurement_corrections=corrections)
    assert journal.read_bytes() == before


def test_original_refuses_corrections_and_stable_correction_retry_is_strict(project):
    engine = Engine(project(), llm="off")
    journal = engine.config.path(engine.config.data["journal"])
    with pytest.raises(HowHardError, match="amend"):
        engine.observe("Y", "claude-opus-5-5", "high", "clean", id="original",
                       measurement_corrections={"tokens": 100})
    assert not journal.exists()
    engine.observe("Y", "claude-opus-5-5", "high", "clean", id="original", tokens=200)
    arguments = {"amends": "original", "id": "corrected", "measurement_corrections": {"tokens": 100}}
    first = engine.observe("Y", "claude-opus-5-5", "high", "clean", **arguments)
    before = journal.read_bytes()
    assert engine.observe("Y", "claude-opus-5-5", "high", "clean", **arguments) == first
    for corrections in ({"tokens": 101}, {"tokens": None}, {"tokens": 100, "seconds": None}):
        with pytest.raises(HowHardError, match="conflict"):
            engine.observe("Y", "claude-opus-5-5", "high", "clean", **{**arguments, "measurement_corrections": corrections})
    assert journal.read_bytes() == before


def test_corrected_usage_reaches_writer_and_scores_original_forecast_once(project):
    from howhard.llm import SETTINGS
    provider = FixtureLLM()
    engine = Engine(project(), llm=provider)
    memory = engine.memory.add("Usage", "A provisional resource forecast")
    choice = engine.pick("Y", "Measure completed work", native_candidates=CANDIDATES, seed=3)
    engine.observe_decision(choice["decision_id"], "clean", id="original", tokens=40000, seconds=120, cost_usd=.4)
    old_context = engine.memory_learning_context("original")
    before_score = expert(engine, memory)["log_score"]
    engine.observe("Y", "claude-opus-5-5", "high", "clean", amends="original", id="corrected",
                   measurement_corrections={"tokens": 10000, "seconds": 30, "cost_usd": .1})
    writer = provider.calls[-1]
    assert writer["observation"]["tokens"] == 10000 and writer["observation"]["cost_usd"] == .1
    assert [r["tokens"] for r in writer["outcome_history"]] == [40000, None]
    assert writer["outcome_history"][1]["measurement_corrections"]["tokens"] == 10000
    assert expert(engine, memory)["evidence_count"] == 1
    assert expert(engine, memory)["log_score"] != before_score
    assert report(engine)["worker_cost"] == {"known_usd": .1, "unknown_runs": 0, "total_usd": .1}
    assert all(m["runs"] == 1 for m in report(engine)["forecast_metrics"].values())
    with pytest.raises(HowHardError, match="evidence changed"):
        engine.apply_memory_learning("original", {"operations": [], "reason": "stale"},
                                     expected_revision=old_context["memory_revision"], event_key=old_context["event_key"])
    engine.observe("Y", "claude-opus-5-5", "high", "clean", amends="corrected", id="unknown",
                   measurement_corrections={"tokens": None, "seconds": None, "cost_usd": None})
    writer = provider.calls[-1]
    assert all(writer["observation"][k] is None for k in ("tokens", "seconds", "cost_usd"))
    assert [r["tokens"] for r in writer["outcome_history"]] == [40000, None, None]
    worlds = choice["semantic"]["worlds"]
    key = f"memory:{memory['id']}:{memory['version']}"
    expected = math.log(SETTINGS["belief"]["memory_prior"]) + SETTINGS["belief"]["temperature"] * math.log(
        worlds[key][0]["clean"] / worlds["statistical"][0]["clean"])
    assert expert(engine, memory)["log_score"] == pytest.approx(expected)
    assert expert(engine, memory)["evidence_count"] == 1
    for block in ("tokens", "time"):
        np.testing.assert_allclose(getattr(engine.posterior, block).mean, getattr(engine.prior, block).mean)
    assert report(engine)["runs"] == 1 and report(engine)["worker_cost"]["total_usd"] is None
    assert engine.memory_learn("unknown")["status"] == "already-learned"


def test_cli_measurement_corrections_file_and_null_clear(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    args = ["observe", "task", "actual-model", "high", "clean", "--llm", "off", "--json"]
    assert main([*args, "--native", "--id", "original", "--tokens", "200", "--cost-usd", ".4"]) == 0
    capsys.readouterr()
    path = tmp_path / "corrections.json"
    path.write_text(json.dumps({"tokens": 100, "cost_usd": None}))
    command = [*args, "--amends", "original", "--id", "corrected", "--measurement-corrections", str(path)]
    assert main(command) == 0
    first = json.loads(capsys.readouterr().out)["observed"]
    assert first["measurement_corrections"] == {"tokens": 100, "cost_usd": None}
    assert main(command) == 0
    assert json.loads(capsys.readouterr().out)["observed"] == first
    assert main(["evidence", "corrected", "--llm", "off", "--json"]) == 0
    effective = json.loads(capsys.readouterr().out)
    assert effective["tokens"] == 100 and effective["cost_usd"] is None
    engine = Engine(directory=tmp_path, llm="off")
    journal = engine.config.path(engine.config.data["journal"])
    before = journal.read_bytes()
    path.write_text('{"tokens": 1.5}')
    assert main([*args, "--amends", "corrected", "--id", "invalid", "--measurement-corrections", str(path)]) == 2
    assert "measurement_corrections" in capsys.readouterr().err
    assert journal.read_bytes() == before
