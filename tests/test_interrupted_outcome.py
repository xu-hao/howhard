"""Interrupted operational outcomes (usage limits, provider/host errors, cancellations) (#104)."""

import json
import math
import pytest

from conftest import fixture_filter, inventory, row
from howhard import Engine
from howhard.cli import main
from howhard.decisions import load
from howhard.report import report
from howhard.validation import HowHardError
from test_capability_filter import SimilarityDouble, filtering, forecasts
from test_execution_accounting import artifact
from test_semantic import FixtureLLM, expert


@pytest.mark.parametrize("cause", ["usage_limit", "provider_error", "host_error", "cancelled", "unrecorded"])
def test_observe_accepts_interrupted_causes_and_cli_round_trip(tmp_path, monkeypatch, capsys, cause):
    monkeypatch.chdir(tmp_path)
    cmd = [
        "observe", "task", "actual-worker", "high", "interrupted", "--native",
        "--cause", cause, "--id", f"run-{cause}", "--seconds", "4.5", "--json",
    ]
    assert main(cmd) == 0
    observed = json.loads(capsys.readouterr().out)["observed"]
    assert observed["outcome"] == "interrupted"
    assert observed["cause"] == cause
    assert observed["seconds"] == pytest.approx(4.5)

    # Idempotent stable-ID retry returns the same row; conflicting cause is rejected.
    assert main(cmd) == 0
    assert json.loads(capsys.readouterr().out)["observed"] == observed
    other_cause = "cancelled" if cause != "cancelled" else "usage_limit"
    assert main([
        "observe", "task", "actual-worker", "high", "interrupted", "--native",
        "--cause", other_cause, "--id", f"run-{cause}", "--seconds", "4.5", "--json",
    ]) == 2
    assert "conflict" in capsys.readouterr().err


def test_interrupted_default_cause_and_strict_cause_matrix(project, tmp_path, monkeypatch, capsys):
    engine = Engine(project())
    defaulted = engine.observe("X", "claude-opus-5-5", "high", "interrupted", id="default-cause")
    assert defaulted["outcome"] == "interrupted"
    assert defaulted["cause"] == "unrecorded"

    for outcome, bad_cause in (
        ("interrupted", None),
        ("interrupted", "capability"),
        ("interrupted", "depth"),
        ("interrupted", "unknown"),
        ("loud", "usage_limit"),
        ("loud", "provider_error"),
        ("loud", "host_error"),
        ("loud", "cancelled"),
        ("silent", "usage_limit"),
        ("clean", "usage_limit"),
    ):
        with pytest.raises(HowHardError, match="cause"):
            engine.observe("X", "claude-opus-5-5", "high", outcome, cause=bad_cause, id=f"bad-{outcome}-{bad_cause}")

    # Direct JSONL validation enforces the same strict matrix on load.
    bad_row = {**row(1, outcome="interrupted"), "cause": "capability"}
    with pytest.raises(HowHardError, match="cause"):
        Engine(project(rows=[row(0), bad_row]))

    bad_loud = {**row(1, outcome="loud"), "cause": "usage_limit"}
    with pytest.raises(HowHardError, match="cause"):
        Engine(project(rows=[row(0), bad_loud]))

    # CLI rejects invalid cause for loud with exit code 2.
    monkeypatch.chdir(tmp_path)
    assert main(["observe", "task", "worker", "high", "loud", "--native", "--cause", "usage_limit", "--id", "bad-cli"]) == 2
    assert "cause" in capsys.readouterr().err
    assert main(["observe", "task", "worker", "high", "interrupted", "--native", "--cause", "capability", "--id", "bad-cli-2"]) == 2
    assert "cause" in capsys.readouterr().err


def test_interrupted_decision_excluded_from_beliefs_and_amendment_reverses_loud_penalty(project, monkeypatch, capsys):
    provider = FixtureLLM()
    engine = Engine(project(), llm=provider)
    memory = engine.memory.add("Migration", "Transaction tasks need careful reasoning.")
    before_beliefs = engine.memory_beliefs()

    pick = engine.pick("Y", "migration", seed=1, native_candidates=inventory([("opus", "high")]))

    # Record via CLI decision-ID shorthand as interrupted/usage_limit with measured partial usage.
    monkeypatch.chdir(engine.config.base)
    assert main([
        "observe", pick["decision_id"], "interrupted",
        "--cause", "usage_limit", "--id", "quota-hit",
        "--tokens", "500", "--active-seconds", "3.0", "--queue-wait-seconds", "2.0",
        "--note", "5h quota window exhausted mid-run", "--json",
    ]) == 0
    observed = json.loads(capsys.readouterr().out)["observed"]
    assert observed["outcome"] == "interrupted"
    assert observed["cause"] == "usage_limit"
    assert observed["decision_id"] == pick["decision_id"]

    engine.refresh()
    after_interrupted = engine.memory_beliefs()
    assert after_interrupted["unscored_runs"] == before_beliefs["unscored_runs"]
    before_mem = next(e for e in before_beliefs["experts"] if e["key"] == f"memory:{memory['id']}:{memory['version']}")
    after_mem = expert(engine, memory)
    assert after_mem["evidence_count"] == before_mem["evidence_count"] == 0
    assert after_mem["log_score"] == pytest.approx(before_mem["log_score"])
    for before_e in before_beliefs["experts"]:
        after_e = next(e for e in after_interrupted["experts"] if e["key"] == before_e["key"])
        assert after_e["evidence_count"] == before_e["evidence_count"] == 0
        assert after_e["log_score"] == pytest.approx(before_e["log_score"])

    # Writer sees the interrupted outcome/cause and full outcome_history.
    ctx = engine.memory_learning_context("quota-hit")
    assert ctx["request"]["observation"]["outcome"] == "interrupted"
    assert ctx["request"]["observation"]["cause"] == "usage_limit"
    assert len(ctx["request"]["outcome_history"]) == 1

    # Amending interrupted -> loud scores the frozen forecast once.
    engine.observe("Y", "claude-opus-5-5", "high", "loud", cause="capability", amends="quota-hit", id="reclassify-loud")
    after_loud = expert(engine, memory)
    assert after_loud["evidence_count"] == 1
    assert after_loud["log_score"] < next(
        e for e in before_beliefs["experts"] if e["key"] == after_loud["key"]
    )["log_score"]

    # Amending loud -> interrupted (usage_limit) cleanly removes the spurious penalty.
    engine.observe("Y", "claude-opus-5-5", "high", "interrupted", cause="usage_limit", amends="reclassify-loud", id="fix-quota")
    restored = engine.memory_beliefs()
    for before_e in before_beliefs["experts"]:
        restored_e = next(e for e in restored["experts"] if e["key"] == before_e["key"])
        assert restored_e["evidence_count"] == 0
        assert restored_e["log_score"] == pytest.approx(before_e["log_score"])

    # Stale writer patch captured before amending to interrupted is rejected.
    with pytest.raises(HowHardError, match="evidence changed"):
        engine.apply_memory_learning(
            "quota-hit",
            {"operations": [], "reason": "stale"},
            expected_revision=ctx["memory_revision"],
            event_key=ctx["event_key"],
        )


def test_interrupted_excluded_from_capability_threshold_and_duration_calibration(tmp_path):
    candidates = [
        {"model": "fixture/weak", "effort": "high", "usd_per_mtok": 1.0},
        {"model": "fixture/strong", "effort": "high", "usd_per_mtok": 5.0},
    ]

    def run_decision(eng):
        pending = eng.prepare("task", "Repair rollback", native_candidates=candidates)
        band = eng.decide(pending["request_id"], filtering(pending["baseline_request"], 0, 1))
        rows = [
            {"family": c["family"], "effort": c["effort"], "clean": 0.9, "loud": 0.09, "silent": 0.01, "tokens": 5000, "seconds": 900}
            for c in band["baseline_request"]["cells"]
        ]
        return eng.decide(band["request_id"], {"forecasts": rows, "memory_forecasts": [], "reason": "Fixed forecast"}, seed=11)

    # Baseline engine with one clean observation (active_seconds=1800).
    base_engine = Engine(directory=tmp_path / "base", retrieval=SimilarityDouble())
    base_engine.observe("task", "fixture/weak", "high", "clean", native=True, id="clean-1", active_seconds=1800.0)
    base_decision = run_decision(base_engine)

    # Engine with the same clean observation PLUS multiple interrupted usage-limit runs with truncated 2s duration.
    int_engine = Engine(directory=tmp_path / "interrupted", retrieval=SimilarityDouble())
    int_engine.observe("task", "fixture/weak", "high", "clean", native=True, id="clean-1", active_seconds=1800.0)
    for idx in range(3):
        int_engine.observe(
            "task", "fixture/weak", "high", "interrupted", cause="usage_limit", native=True,
            id=f"limit-{idx}", active_seconds=2.0, note="5h usage limit hit immediately",
        )
    int_decision = run_decision(int_engine)

    # Receipt loads and validates cleanly even with interrupted runs in evidence/memories.
    receipt = load(int_engine.config, int_decision["decision_id"])
    assert receipt["capability"]["evidence"] == base_decision["capability"]["evidence"]
    assert receipt["semantic"]["duration_observations"] == base_decision["semantic"]["duration_observations"]

    # Capability P_clear, P_timely, band, and calibrated T/J are completely invariant to interrupted runs.
    base_cap = {c["model"]: c for c in base_decision["capability"]["cells"]}
    int_cap = {c["model"]: c for c in int_decision["capability"]["cells"]}
    for model in ("fixture/weak", "fixture/strong"):
        assert int_cap[model]["P_clear"] == pytest.approx(base_cap[model]["P_clear"])
        assert int_cap[model]["P_timely"] == pytest.approx(base_cap[model]["P_timely"])
        assert int_cap[model]["classification"] == base_cap[model]["classification"]
    base_cells = {c["model"]: c for c in base_decision["cells"]}
    int_cells = {c["model"]: c for c in int_decision["cells"]}
    assert int_cells["fixture/weak"]["T"] == pytest.approx(base_cells["fixture/weak"]["T"])
    assert int_cells["fixture/weak"]["J"] == pytest.approx(base_cells["fixture/weak"]["J"])

    # Contrast: recording the same failures as loud DOES enter capability evidence and lowers P_clear.
    loud_engine = Engine(directory=tmp_path / "loud", retrieval=SimilarityDouble())
    loud_engine.observe("task", "fixture/weak", "high", "clean", native=True, id="clean-1", active_seconds=1800.0)
    for idx in range(3):
        loud_engine.observe(
            "task", "fixture/weak", "high", "loud", cause="capability", native=True,
            id=f"loud-{idx}", active_seconds=2.0,
        )
    loud_decision = run_decision(loud_engine)
    loud_cap = {c["model"]: c for c in loud_decision["capability"]["cells"]}
    assert len(loud_decision["capability"]["evidence"]) == 4
    assert loud_cap["fixture/weak"]["P_clear"] < base_cap["fixture/weak"]["P_clear"]


def test_report_separates_interrupted_from_model_failures_while_preserving_accounting(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    candidates = [{"model": "fixture/worker", "effort": "high", "usd_per_mtok": 1.0}]

    # Decision 1: create an intentional stage1-vs-stage2 disagreement (P_clear=0.9 vs P_clean=0.5).
    p1 = engine.prepare("task", "Task one", native_candidates=candidates)
    b1 = engine.decide(p1["request_id"], filtering(p1["baseline_request"], 0, 0))
    cell = b1["baseline_request"]["cells"][0]
    d1 = engine.decide(
        b1["request_id"],
        {"forecasts": [{"family": cell["family"], "effort": cell["effort"], "clean": 0.5, "loud": 0.45, "silent": 0.05, "tokens": 1000, "seconds": 60}],
         "memory_forecasts": [], "reason": "Disagreeing forecast"},
        seed=1,
    )
    assert d1["capability"]["disagreements"]
    engine.observe_decision(d1["decision_id"], "interrupted", cause="usage_limit", id="exec-limit", cost_usd=0.05)

    # Attach partial execution usage with duration to the interrupted run; outcome and cause remain preserved.
    manifest = artifact(
        tmp_path,
        [{"type": "thread.started", "thread_id": "sess-1"}, {"type": "turn.failed"}],
        execution_id="exec-limit", session_id="sess-1", completed=False,
        active_seconds=12.0, queue_wait_seconds=3.0,
    )
    engine.attach_usage("exec-limit", manifest, id="exec-limit-usage")
    ev = engine.evidence("exec-limit")
    assert ev["outcome"] == "interrupted"
    assert ev["cause"] == "usage_limit"
    assert ev["seconds"] == pytest.approx(15.0)
    assert ev["active_seconds"] == pytest.approx(12.0)

    # Decision 2: clean execution on the same cell.
    p2 = engine.prepare("task", "Task two", native_candidates=candidates)
    b2 = engine.decide(p2["request_id"], filtering(p2["baseline_request"], 0, 0))
    d2 = engine.decide(
        b2["request_id"],
        {"forecasts": [{"family": cell["family"], "effort": cell["effort"], "clean": 0.9, "loud": 0.08, "silent": 0.02, "tokens": 1000, "seconds": 60}],
         "memory_forecasts": [], "reason": "Clean forecast"},
        seed=2,
    )
    engine.observe_decision(d2["decision_id"], "clean", id="exec-clean", seconds=40.0, cost_usd=0.20)

    stats = report(engine)
    assert stats["runs"] == 2
    assert stats["outcomes"] == {"clean": 1, "loud": 0, "silent": 0, "interrupted": 1}
    assert stats["interrupted_causes"] == {
        "usage_limit": 1, "provider_error": 0, "host_error": 0, "cancelled": 0, "unrecorded": 0,
    }
    # Only the non-interrupted run scores forecast metrics.
    assert stats["forecast_metrics"]["llm"]["runs"] == 1
    assert stats["forecast_metrics"]["weighted_memory"]["runs"] == 1
    assert stats["forecast_metrics"]["llm"]["mean_log_loss"] == pytest.approx(-math.log(0.9))
    # Accounting totals still include both runs.
    assert stats["worker_cost"]["known_usd"] == pytest.approx(0.25)
    assert stats["worker_cost"]["total_usd"] == pytest.approx(0.25)
    assert stats["worker_time"]["seconds"]["known_seconds"] == pytest.approx(55.0)
    assert stats["execution_accounting"]["runs"] == 2
    # filter_quality separates interrupted from fit_failures and does not resolve d1's disagreement.
    fq = stats["filter_quality"]
    assert fq["fit_failures"] == {"executed": 1, "loud": 0, "silent": 0, "interrupted": 1, "failure_rate": 0.0}
    assert fq["disagreements"]["count"] == 1
    assert fq["disagreements"]["resolved"] == 0
    assert len(fq["executions"]) == 2
    interrupted_exec = next(e for e in fq["executions"] if e["execution_id"] == "exec-limit")
    assert interrupted_exec["outcome"] == "interrupted"
    assert interrupted_exec["cause"] == "usage_limit"
