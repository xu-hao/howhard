"""Active vs. queue-wait duration, log-normal J calibration, and timely capability saturation (#99)."""

from copy import deepcopy
import json
import math

import pytest

from howhard import Engine
from howhard.accounting import collect, validate
from howhard.cli import main
from howhard.context import canonical, digest
from howhard.decisions import load
from howhard.payload_store import put
from howhard.report import report
from howhard.storage import locked
from howhard.validation import HowHardError
from test_capability_filter import SimilarityDouble, filtering, forecasts, inventory as cap_inventory
from test_execution_accounting import artifact, codex_events


def test_observe_derives_and_validates_active_and_queue_wait_seconds_and_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cmd = [
        "observe", "task", "actual-worker", "high", "clean", "--native",
        "--id", "exec-derived", "--active-seconds", "120.5", "--queue-wait-seconds", "299.5", "--json",
    ]
    assert main(cmd) == 0
    observed = json.loads(capsys.readouterr().out)["observed"]
    assert observed["seconds"] == pytest.approx(420.0)
    assert observed["active_seconds"] == pytest.approx(120.5)
    assert observed["queue_wait_seconds"] == pytest.approx(299.5)

    # Idempotent retry returns the same row; conflicting breakdown is rejected.
    assert main(cmd) == 0
    assert json.loads(capsys.readouterr().out)["observed"] == observed
    conflict = [*cmd[:-1], "--active-seconds", "121.0", "--json"]
    assert main(conflict) == 2
    assert "conflict" in capsys.readouterr().err

    # Explicit matching wall-clock seconds within 1e-6 succeeds.
    assert main([
        "observe", "task", "actual-worker", "high", "clean",
        "--id", "exec-explicit", "--seconds", "420.0000005",
        "--active-seconds", "120.5", "--queue-wait-seconds", "299.5", "--json",
    ]) == 0
    capsys.readouterr()

    assert main(["evidence", "exec-derived", "--json"]) == 0
    effective = json.loads(capsys.readouterr().out)
    assert effective["seconds"] == pytest.approx(420.0)
    assert effective["active_seconds"] == pytest.approx(120.5)
    assert effective["queue_wait_seconds"] == pytest.approx(299.5)

    engine = Engine(directory=tmp_path)
    journal = engine.config.path(engine.config.data["journal"])
    before = journal.read_bytes()

    invalid_cases = [
        {"id": "bad-sum", "seconds": 421.0, "active_seconds": 120.0, "queue_wait_seconds": 300.0},
        {"id": "bad-active-gt", "seconds": 100.0, "active_seconds": 101.0},
        {"id": "bad-queue-gt", "seconds": 100.0, "queue_wait_seconds": 101.0},
        {"id": "bad-neg-active", "active_seconds": -0.1},
        {"id": "bad-neg-queue", "queue_wait_seconds": -1.0},
        {"id": "bad-bool-active", "active_seconds": True},
        {"id": "bad-bool-queue", "queue_wait_seconds": False},
        {"id": "bad-nan", "active_seconds": float("nan")},
        {"id": "bad-inf", "queue_wait_seconds": float("inf")},
    ]
    for case in invalid_cases:
        with pytest.raises(HowHardError):
            engine.observe("task", "actual-worker", "high", "clean", **case)

    # Amendments must carry null direct measurement fields and use measurement_corrections instead.
    for field in ("active_seconds", "queue_wait_seconds"):
        with pytest.raises(HowHardError, match="amends"):
            engine.observe("task", "actual-worker", "high", "loud", amends="exec-derived", id=f"bad-{field}", **{field: 10.0})

    assert journal.read_bytes() == before


def test_usage_collect_attach_measurement_corrections_and_report_time_coverage(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    # Unavailable token telemetry still preserves active/queue-wait duration on execution_usage.
    manifest = artifact(
        tmp_path, [{"type": "future.unsupported"}],
        execution_id="exec-1", active_seconds=90.0, queue_wait_seconds=210.0,
    )
    usage = collect(manifest)
    assert usage["status"] == "unavailable"
    assert usage["seconds"] == pytest.approx(300.0)
    assert usage["active_seconds"] == pytest.approx(90.0)
    assert usage["queue_wait_seconds"] == pytest.approx(210.0)

    for bad in (
        {**manifest, "seconds": 301.0},
        {**manifest, "active_seconds": -1.0},
        {**manifest, "queue_wait_seconds": True},
    ):
        with pytest.raises(HowHardError):
            collect(bad)

    tampered = deepcopy(usage)
    tampered["seconds"] = 305.0
    with pytest.raises(HowHardError):
        validate(tampered)

    engine = Engine(directory=tmp_path)
    engine.observe("task", "fixture-model", "high", "clean", native=True, id="exec-1")
    engine.attach_usage("exec-1", manifest, id="exec-1-usage")
    ev1 = engine.evidence("exec-1")
    assert ev1["seconds"] == pytest.approx(300.0)
    assert ev1["active_seconds"] == pytest.approx(90.0)
    assert ev1["queue_wait_seconds"] == pytest.approx(210.0)

    # Measurement corrections derive seconds when active+queue_wait are replaced together, and reject inconsistent sums.
    engine.observe(
        "task", "fixture-model", "high", "clean", amends="exec-1-usage", id="exec-1-fix",
        measurement_corrections={"active_seconds": 150.0, "queue_wait_seconds": 50.0},
    )
    ev1_fixed = engine.evidence("exec-1")
    assert ev1_fixed["seconds"] == pytest.approx(200.0)
    assert ev1_fixed["active_seconds"] == pytest.approx(150.0)
    assert ev1_fixed["queue_wait_seconds"] == pytest.approx(50.0)

    with pytest.raises(HowHardError):
        engine.observe(
            "task", "fixture-model", "high", "clean", amends="exec-1-fix", id="exec-1-bad",
            measurement_corrections={"seconds": 999.0},
        )

    # Scalar-only historical style row and unmeasured row keep unknown coverage explicit in report().
    engine.observe("task", "fixture-model", "high", "clean", id="exec-scalar", seconds=60.0)
    engine.observe("task", "fixture-model", "high", "clean", id="exec-null")

    stats = report(engine)
    wt = stats["worker_time"]
    assert wt["seconds"] == {"known_seconds": pytest.approx(260.0), "unknown_runs": 1, "total_seconds": None}
    assert wt["active_seconds"] == {"known_seconds": pytest.approx(150.0), "unknown_runs": 2, "total_seconds": None}
    assert wt["queue_wait_seconds"] == {"known_seconds": pytest.approx(50.0), "unknown_runs": 2, "total_seconds": None}
    assert wt["effective_active_seconds"] == {"known_seconds": pytest.approx(210.0), "unknown_runs": 1, "total_seconds": None}


def test_beliefs_scores_forecasts_against_effective_active_seconds_excluding_queue_wait(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    memory = engine.memory.add("Active time", "Task takes 2 minutes active execution; external queue wait is separate.")
    candidates = [{"model": "fixture/worker", "effort": "high", "usd_per_mtok": 1.0}]
    pending = engine.prepare("task", "Repair rollback", native_candidates=candidates)
    band = engine.decide(pending["request_id"], filtering(pending["baseline_request"], 0, 0))
    cell = band["baseline_request"]["cells"][0]
    # Baseline predicts 3720s (confounded by queue wait); memory accurately predicts 120s active duration.
    base_row = {"family": cell["family"], "effort": cell["effort"], "clean": 0.9, "loud": 0.09, "silent": 0.01, "tokens": 1000, "seconds": 3720}
    mem_row = {**base_row, "seconds": 120}
    decision = engine.decide(
        band["request_id"],
        {"forecasts": [base_row], "memory_forecasts": [{"id": memory["id"], "version": memory["version"], "forecasts": [mem_row]}], "reason": "Active vs queue forecast"},
        seed=7,
    )
    before = next(e for e in engine.memory_beliefs()["experts"] if e["key"] == f"memory:{memory['id']}:{memory['version']}")
    engine.observe_decision(
        decision["decision_id"], "clean", id="queued-run",
        tokens=1000, active_seconds=120.0, queue_wait_seconds=3600.0,
    )
    after = next(e for e in engine.memory_beliefs()["experts"] if e["key"] == f"memory:{memory['id']}:{memory['version']}")
    assert after["evidence_count"] == 1
    assert after["log_score"] > before["log_score"]


def test_empirical_duration_calibration_and_right_tail_variance_price_out_slow_cheap_cell(tmp_path):
    candidates = [
        {"model": "fixture/cheap-thrasher", "effort": "high", "usd_per_mtok": 0.1},
        {"model": "fixture/fast-strong", "effort": "high", "usd_per_mtok": 10.0},
    ]

    def prepare_band(eng):
        pending = eng.prepare("task", "Repair rollback", native_candidates=candidates)
        band = eng.decide(pending["request_id"], filtering(pending["baseline_request"], 0, 1))
        specs = {
            "fixture/cheap-thrasher": {"clean": 0.92, "loud": 0.07, "silent": 0.01, "tokens": 5000, "seconds": 900},
            "fixture/fast-strong": {"clean": 0.94, "loud": 0.05, "silent": 0.01, "tokens": 100000, "seconds": 900},
        }
        # Stateless forecast optimistically guesses 900s (15m) for each retained band cell.
        rows = [
            {"family": c["family"], "effort": c["effort"], **specs[c["model"]]}
            for c in band["baseline_request"]["cells"]
        ]
        return band["request_id"], {"forecasts": rows, "memory_forecasts": [], "reason": "Optimistic stateless duration"}

    def decide_once(eng, seed=42):
        req_id, resp = prepare_band(eng)
        return eng.decide(req_id, resp, seed=seed)

    # 1. Zero duration evidence preserves the expert's 900s (0.25h) forecast and picks the cheap cell.
    zero_engine = Engine(directory=tmp_path / "zero", retrieval=SimilarityDouble())
    zero_req, zero_resp = prepare_band(zero_engine)
    zero_result = zero_engine.decide(zero_req, zero_resp, seed=1)
    cells_zero = {c["model"]: c for c in zero_result["cells"]}
    assert cells_zero["fixture/cheap-thrasher"]["T"] == pytest.approx(0.25)
    assert zero_result["chosen"]["model"] == "fixture/cheap-thrasher"

    # 2. Multi-hour active runtime reality (with right-tail variance) calibrates E[T] = exp(mu + sigma^2/2),
    #    raises cost_sd, and prices out the slow cheap cell in J and Thompson sampling.
    thrash_engine = Engine(directory=tmp_path / "thrash", retrieval=SimilarityDouble())
    for i, active in enumerate((7200.0, 10800.0, 14400.0, 10800.0)):
        thrash_engine.observe(
            "task", "fixture/cheap-thrasher", "high", "clean", native=True,
            task_text="Repair rollback", id=f"thrash-{i}",
            active_seconds=active, queue_wait_seconds=60.0,
        )
    thrash_req, thrash_resp = prepare_band(thrash_engine)
    calibrated_result = thrash_engine.decide(thrash_req, thrash_resp, seed=1)
    cells_cal = {c["model"]: c for c in calibrated_result["cells"]}
    # Geometric mean of [7200, 10800, 14400, 10800] is ~10392s; shrunk + exp(sigma^2/2) exceeds 2.5 hours (> $62 time cost).
    assert cells_cal["fixture/cheap-thrasher"]["T"] > 2.5
    assert cells_cal["fixture/fast-strong"]["T"] == pytest.approx(0.25)
    assert cells_cal["fixture/cheap-thrasher"]["J"] > cells_cal["fixture/fast-strong"]["J"] + 50.0

    cal_by_family = {c["family"]: c for c in calibrated_result["semantic"]["sampling"]["cells"]}
    zero_by_family = {c["family"]: c for c in zero_result["semantic"]["sampling"]["cells"]}
    weak_family = cells_cal["fixture/cheap-thrasher"]["family"]
    assert cal_by_family[weak_family]["duration"]["sigma"] > 0
    assert cal_by_family[weak_family]["cost_sd"] > zero_by_family[weak_family]["cost_sd"]

    # Across 200 Thompson seeds, multi-hour active duration shifts selection strongly to the fast strong cell
    # (unseen fast-strong retains bounded Dirichlet outcome uncertainty, so rare high-silent draws still explore).
    zero_wins = [zero_engine.decide(zero_req, zero_resp, seed=s)["chosen"]["model"] for s in range(200)].count("fixture/fast-strong")
    thrash_wins = [thrash_engine.decide(thrash_req, thrash_resp, seed=s)["chosen"]["model"] for s in range(200)].count("fixture/fast-strong")
    assert thrash_wins >= 165 and thrash_wins > zero_wins + 50
    load(thrash_engine.config, calibrated_result["decision_id"])

    # 3. Passive external queue wait (900s active + 9900s queue wait = 10800s wall-clock) does NOT penalize the cell.
    queued_engine = Engine(directory=tmp_path / "queued", retrieval=SimilarityDouble())
    for i in range(4):
        queued_engine.observe(
            "task", "fixture/cheap-thrasher", "high", "clean", native=True,
            task_text="Repair rollback", id=f"queued-{i}",
            active_seconds=900.0, queue_wait_seconds=9900.0,
        )
    queued_result = decide_once(queued_engine, seed=1)
    cells_queued = {c["model"]: c for c in queued_result["cells"]}
    assert cells_queued["fixture/cheap-thrasher"]["T"] == pytest.approx(0.25)
    assert queued_result["chosen"]["model"] == "fixture/cheap-thrasher"


def test_between_expert_duration_disagreement_increases_cost_sd(tmp_path):
    candidates = [{"model": "fixture/worker", "effort": "high", "usd_per_mtok": 1.0}]

    # Disagreeing experts: 600s vs 14400s
    disagree_engine = Engine(directory=tmp_path / "disagree")
    mem = disagree_engine.memory.add("Duration warning", "May thrash for 4 hours on boundary edge cases.")
    p1 = disagree_engine.prepare("task", "Repair rollback", native_candidates=candidates)
    b1 = disagree_engine.decide(p1["request_id"], filtering(p1["baseline_request"], 0, 0))
    c1 = b1["baseline_request"]["cells"][0]
    base = {"family": c1["family"], "effort": c1["effort"], "clean": 0.95, "loud": 0.04, "silent": 0.01, "tokens": 1000, "seconds": 600}
    over = {**base, "seconds": 14400}
    res_disagree = disagree_engine.decide(
        b1["request_id"],
        {"forecasts": [base], "memory_forecasts": [{"id": mem["id"], "version": mem["version"], "forecasts": [over]}], "reason": "Disagreeing active duration"},
        seed=5,
    )

    # Agreeing experts with the same weighted mean seconds
    mean_seconds = res_disagree["cells"][0]["T"] * 3600
    agree_engine = Engine(directory=tmp_path / "agree")
    p2 = agree_engine.prepare("task", "Repair rollback", native_candidates=candidates)
    b2 = agree_engine.decide(p2["request_id"], filtering(p2["baseline_request"], 0, 0))
    c2 = b2["baseline_request"]["cells"][0]
    same = {"family": c2["family"], "effort": c2["effort"], "clean": 0.95, "loud": 0.04, "silent": 0.01, "tokens": 1000, "seconds": mean_seconds}
    res_agree = agree_engine.decide(b2["request_id"], {"forecasts": [same], "memory_forecasts": [], "reason": "Agreeing active duration"}, seed=5)

    cal_disagree = res_disagree["semantic"]["sampling"]["cells"][0]
    cal_agree = res_agree["semantic"]["sampling"]["cells"][0]
    assert cal_disagree["duration"]["between_experts_log_variance"] > 0
    assert cal_disagree["cost_sd"] > cal_agree["cost_sd"]


def test_slow_clean_runs_do_not_prune_faster_stronger_cells_above_upper(tmp_path):
    # Case A: Weak cell W clears ~90% in 3 hours while stronger cell S clears in 15 minutes (raw native precedents).
    cells = [
        {"model": f"fixture/model-{i:03d}", "effort": "high", "usd_per_mtok": 0.1 + i * 0.5}
        for i in range(6)
    ]
    engine = Engine(directory=tmp_path / "cross-cell", retrieval=SimilarityDouble())
    for i in range(5):
        engine.observe(
            "task", cells[0]["model"], "high", "clean", native=True,
            task_text="Repair rollback", id=f"weak-slow-{i}", active_seconds=10800.0,
        )
    engine.observe(
        "task", cells[0]["model"], "high", "loud", native=True,
        task_text="Repair rollback", cause="capability", id="weak-loud", active_seconds=7200.0,
    )
    for i in range(2):
        engine.observe(
            "task", cells[2]["model"], "high", "clean", native=True,
            task_text="Repair rollback", id=f"strong-fast-{i}", active_seconds=900.0,
        )

    pending = engine.prepare("task", "Repair rollback", native_candidates=cells)
    # Even if baseline filter tries to cap upper at model-000 or model-002, slow clean runs on model-000 must not prune model-002.
    stage2 = engine.decide(pending["request_id"], filtering(pending["baseline_request"], 0, 2))
    cap = stage2["capability"]
    by_model = {c["model"]: c for c in cap["cells"]}
    assert by_model["fixture/model-000"]["classification"] == "fit"
    assert by_model["fixture/model-002"]["classification"] == "fit"
    assert by_model["fixture/model-002"]["P_timely"] > by_model["fixture/model-000"]["P_timely"]
    # Cells well above fast-clearing model-002 still saturate and remain overpowered.
    assert by_model["fixture/model-005"]["classification"] == "overpowered"
    # Verify no internal failure counter was added to policy, cells, or evidence.
    assert not any("fail" in k for k in cap["policy"])
    assert not any("fail" in k for item in cap["evidence"] for k in item)

    # Complete Stage 2 and verify the faster stronger cell (model-002) beats both slow model-000 and slower/riskier model-001.
    stage2_specs = {
        "fixture/model-000": {"clean": 0.91, "loud": 0.08, "silent": 0.01, "tokens": 5000, "seconds": 900},
        "fixture/model-001": {"clean": 0.92, "loud": 0.07, "silent": 0.01, "tokens": 15000, "seconds": 3600},
        "fixture/model-002": {"clean": 0.95, "loud": 0.04, "silent": 0.01, "tokens": 20000, "seconds": 900},
    }
    rows = [
        {"family": c["family"], "effort": c["effort"], **stage2_specs.get(c["model"], stage2_specs["fixture/model-002"])}
        for c in stage2["baseline_request"]["cells"]
    ]
    final = engine.decide(stage2["request_id"], {"forecasts": rows, "memory_forecasts": [], "reason": "Stage 2 comparison"}, seed=42)
    assert final["chosen"]["model"] == "fixture/model-002"
    load(engine.config, final["decision_id"])

    # Case B: Unseen stronger cells when weak cell W overran its own decision receipt's forecasted active duration,
    # even on a 50-cell inventory where baseline put upper=0.
    large_cells = cap_inventory(50)
    large_engine = Engine(directory=tmp_path / "receipt-overrun", retrieval=SimilarityDouble())
    first_req = large_engine.prepare("task", "Repair rollback", native_candidates=large_cells)
    first_band = large_engine.decide(first_req["request_id"], filtering(first_req["baseline_request"], 0, 0))
    first_decision = large_engine.decide(first_band["request_id"], forecasts(first_band["baseline_request"]), seed=1)
    # Forecast was 10s, actual active runtime was 10800s (3 hours) — severe blowup on clean run.
    large_engine.observe_decision(first_decision["decision_id"], "clean", id="blowup-1", active_seconds=10800.0)

    second_req = large_engine.prepare("task", "Repair rollback", native_candidates=large_cells)
    second_band = large_engine.decide(second_req["request_id"], filtering(second_req["baseline_request"], 0, 0))
    assert second_band["stage"] == "forecast"
    second_by_model = {c["model"]: c for c in second_band["capability"]["cells"]}
    assert second_by_model["fixture/model-001"]["classification"] == "fit"
    assert len(second_band["capability"]["band"]) <= second_band["capability"]["policy"]["max_forecast_cells"]


def test_frozen_v7_receipts_and_historical_journals_load_validate_observe_replay_and_report(tmp_path):
    engine = Engine(directory=tmp_path, retrieval=SimilarityDouble())
    cells = [{"model": f"fixture/model-{i:03d}", "effort": "high", "usd_per_mtok": 1.0} for i in range(4)]

    # Mint a current receipt and derive authentic frozen v8 and v7 receipts from its base structure.
    from legacy_receipts import to_v7, to_v8
    p = engine.prepare("task", "Repair rollback", native_candidates=cells)
    b = engine.decide(p["request_id"], filtering(p["baseline_request"], 0, 1))
    current = load(engine.config, engine.decide(b["request_id"], forecasts(b["baseline_request"]), seed=11)["decision_id"])
    assert current["format"] == "howhard-decision-v9"
    assert current["semantic"]["format"] == "howhard-semantic-v6"
    assert current["capability"]["format"] == "howhard-capability-v4"
    v8_receipt = to_v8(current)
    v8_id = digest(v8_receipt)
    with locked(engine.config):
        put(engine.config, "decisions", v8_id, canonical(v8_receipt) + b"\n")
    v8_loaded = load(engine.config, v8_id)
    assert v8_loaded["format"] == "howhard-decision-v8"
    assert v8_loaded["capability"]["format"] == "howhard-capability-v3"
    assert v8_loaded["semantic"]["format"] == "howhard-semantic-v5"

    v7_receipt = to_v7(current)
    v7_id = digest(v7_receipt)
    with locked(engine.config):
        put(engine.config, "decisions", v7_id, canonical(v7_receipt) + b"\n")

    # Frozen v7 receipt loads, validates, accepts observe_decision, replays in beliefs(), and scores in report().
    loaded_v7 = load(engine.config, v7_id)
    assert loaded_v7["format"] == "howhard-decision-v7"
    engine.observe_decision(v7_id, "clean", id="obs-from-v7", seconds=12.0)
    engine.observe_decision(v8_id, "clean", id="obs-from-v8", active_seconds=10.0, queue_wait_seconds=5.0)

    beliefs = engine.memory_beliefs()
    assert beliefs["unscored_runs"] == 0
    llm_expert = next(e for e in beliefs["experts"] if e["key"] == "llm")
    assert llm_expert["evidence_count"] == 2

    rep = report(engine)
    assert rep["unsupported_receipts"] == []
    assert rep["forecast_metrics"]["llm"]["runs"] == 2

    # Cross-version tampering is rejected in both directions.
    bad_v7 = deepcopy(loaded_v7)
    bad_v7["semantic"]["format"] = "howhard-semantic-v5"
    with pytest.raises(HowHardError):
        load(engine.config, bad_v7)

    bad_v8 = deepcopy(v8_loaded)
    bad_v8["semantic"]["format"] = "howhard-semantic-v4"
    with pytest.raises(HowHardError):
        load(engine.config, bad_v8)
