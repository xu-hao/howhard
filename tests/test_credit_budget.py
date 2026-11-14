"""Candidate quota-credit validation and actual session-cost accounting.

Issue #75 tracks hard credit-budget feasibility; issue #95 separates credit burn from
actual session money cost.
"""

from copy import deepcopy
import json

import pytest

from conftest import forecast_request
from howhard import Engine, load_config
from howhard.cli import main, render
from howhard.decisions import load
from howhard.native import validate
from howhard.report import report
from howhard.validation import HowHardError
from test_execution_accounting import card, counts
from test_semantic import FixtureLLM


def decide_with_forecasts(engine, candidates, rows, *, seed=7):
    pending = forecast_request(engine, engine.prepare("task", "Repair rollback", native_candidates=candidates))
    by_model = {(c["model"], c["effort"]): c["family"] for c in pending["baseline_request"]["cells"]}
    forecasts = []
    for model, effort, clean, loud, silent, tokens, seconds, usage in rows:
        item = {
            "family": by_model[model, effort],
            "effort": effort,
            "clean": clean,
            "loud": loud,
            "silent": silent,
            "tokens": tokens,
            "seconds": seconds,
        }
        if usage is not None:
            item["token_usage"] = usage
        forecasts.append(item)
    return engine.decide(pending["request_id"], {"forecasts": forecasts, "memory_forecasts": [], "reason": "Synthetic credit fixture"}, seed=seed)


def test_credit_budget_validates_windows_and_preserves_copy_without_persisting_in_registry(tmp_path):
    candidates = [
        {
            "model": "route-a/model",
            "effort": "low",
            "usd_per_mtok": 10.0,
            "credit_budget": {
                "available_credit_usd": 0.25,
                "session_5h_remaining_credit_usd": 0.25,
                "weekly_7d_remaining_credit_usd": 4.0,
            },
        },
        {
            "model": "route-b/model",
            "effort": "high",
            "usd_per_mtok": 20.0,
            "credit_budget": {
                "available_credit_usd": 0.5,
                "session_5h_remaining_credit_usd": 2.0,
                "weekly_7d_remaining_credit_usd": 0.5,
            },
        },
    ]
    validated = validate(candidates)
    assert validated[0]["credit_budget"] == {
        "available_credit_usd": 0.25,
        "session_5h_remaining_credit_usd": 0.25,
        "weekly_7d_remaining_credit_usd": 4.0,
    }
    assert validated[0]["credit_budget"] is not candidates[0]["credit_budget"]

    engine = Engine(directory=tmp_path, llm=FixtureLLM())
    engine.prepare("task", "Check registry persistence", native_candidates=candidates)
    registry = json.loads((engine.config.state_root / "native-models.json").read_text())
    assert registry["models"] == {
        "route-a/model": {"efforts": ["low"]},
        "route-b/model": {"efforts": ["high"]},
    }


def test_cost_per_credit_defaults_to_one_and_allows_a_free_tier():
    assert validate([{"model": "route-a/model", "effort": "low"}])[0]["cost_per_credit"] == 1.0
    free = validate([{"model": "route-a/model", "effort": "low", "cost_per_credit": 0.0}])
    assert free[0]["cost_per_credit"] == 0.0


@pytest.mark.parametrize("factor", [-0.01, True, None, float("inf")])
def test_invalid_cost_per_credit_is_rejected(factor):
    with pytest.raises(HowHardError, match="cost_per_credit"):
        validate([{"model": "route-a/model", "effort": "low", "cost_per_credit": factor}])


def test_credit_budget_uses_only_explicit_credit_amount_keys():
    with pytest.raises(HowHardError, match="credit_budget"):
        validate([{"model": "route-a/model", "effort": "low", "usd_per_mtok": 5.0,
                   "credit_budget": {"available_usd": 1.0}}])


@pytest.mark.parametrize(
    "budget,match",
    [
        (None, "credit_budget"),
        ("invalid", "credit_budget"),
        ({}, "missing keys"),
        ({"available_credit_usd": 1.0, "unknown_window": 1.0}, "unknown keys"),
        ({"available_credit_usd": None}, "finite number"),
        ({"available_credit_usd": True}, "finite number"),
        ({"available_credit_usd": False}, "finite number"),
        ({"available_credit_usd": -0.01}, "non-negative"),
        ({"available_credit_usd": float("inf")}, "finite number"),
        ({"available_credit_usd": 1.0, "session_5h_remaining_credit_usd": -1.0}, "non-negative"),
        ({"available_credit_usd": 1.0, "weekly_7d_remaining_credit_usd": True}, "finite number"),
        ({"available_credit_usd": 1.0, "session_5h_remaining_credit_usd": None}, "finite number"),
        (
            {"available_credit_usd": 1.0, "session_5h_remaining_credit_usd": 0.4, "weekly_7d_remaining_credit_usd": 1.0},
            "available_credit_usd must match minimum remaining window credit",
        ),
        (
            {"available_credit_usd": 1.0, "session_5h_remaining_credit_usd": 1.0, "weekly_7d_remaining_credit_usd": 0.6},
            "available_credit_usd must match minimum remaining window credit",
        ),
        (
            {"available_credit_usd": 0.5, "session_5h_remaining_credit_usd": 0.8},
            "available_credit_usd must match minimum remaining window credit",
        ),
        (
            {"available_credit_usd": 0.5, "weekly_7d_remaining_credit_usd": 0.2},
            "available_credit_usd must match minimum remaining window credit",
        ),
    ],
)
def test_invalid_credit_budget_values_and_window_mismatch_rejected(budget, match):
    with pytest.raises(HowHardError, match=match):
        validate([{"model": "route-a/model", "effort": "high", "usd_per_mtok": 5.0, "credit_budget": budget}])


def test_credit_budget_requires_pricing_and_uniform_basis(tmp_path):
    with pytest.raises(HowHardError, match="requires rate_card or usd_per_mtok"):
        validate([{"model": "route-a/model", "effort": "high", "credit_budget": {"available_credit_usd": 1.0}}])

    with pytest.raises(HowHardError, match="requires rate_card or usd_per_mtok"):
        validate([{"model": "route-a/model", "effort": "high", "usd_per_mtok": None, "credit_budget": {"available_credit_usd": 1.0}}])

    for second in (
        {"model": "route-b/model", "effort": "high"},
        {"model": "route-b/model", "effort": "high", "rate_card": card(model="route-b/model")},
    ):
        with pytest.raises(HowHardError, match="credit_budget requires uniform pricing across all candidates"):
            validate(
                [
                    {"model": "route-a/model", "effort": "high", "usd_per_mtok": 5.0, "credit_budget": {"available_credit_usd": 1.0}},
                    second,
                ]
            )

    with pytest.raises(HowHardError, match="credit_budget requires uniform pricing across all candidates"):
        validate(
            [
                {
                    "model": "fixture-model",
                    "effort": "high",
                    "rate_card": card(rates=counts(input=2, output=8, cache_read=0.2, cache_write=None, cache_write_5m=2.5, cache_write_1h=4)),
                    "credit_budget": {"available_credit_usd": 1.0},
                }
            ]
        )


def test_toml_inventories_load_and_validate_credit_budget(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "howhard.toml"
    config_path.write_text(
        'on_cap_unmet = "flag-review"\n'
        "[inventories]\n"
        "router = [\n"
        '  { model = "route-a/model", effort = "low", usd_per_mtok = 10.0, '
        "credit_budget = { available_credit_usd = 0.05, session_5h_remaining_credit_usd = 0.05, weekly_7d_remaining_credit_usd = 2.0 } },\n"
        '  { model = "route-b/model", effort = "high", usd_per_mtok = 20.0, '
        "credit_budget = { available_credit_usd = 1.0, session_5h_remaining_credit_usd = 1.0, weekly_7d_remaining_credit_usd = 5.0 } },\n"
        "]\n"
    )
    loaded = load_config(config_path)
    assert loaded.data["inventories"]["router"][0]["credit_budget"]["available_credit_usd"] == pytest.approx(0.05)

    monkeypatch.chdir(tmp_path)
    assert main(["pick", "bookkeeping", "--inventory", "router", "--seed", "3", "--json"]) == 0
    decision = json.loads(capsys.readouterr().out)
    assert decision["chosen"]["model"] == "route-b/model"
    assert decision["why"][0]["reason"] == "insufficient-credit"
    assert {"session_cost_usd", "needed_credit_usd"} <= decision["cells"][0].keys()
    assert decision["native_candidates"][0]["cost_per_credit"] == 1.0

    bad_path = tmp_path / "bad.toml"
    bad_path.write_text(
        'on_cap_unmet = "flag-review"\n'
        "[inventories]\n"
        'router = [{ model = "route-a/model", effort = "low", credit_budget = { available_credit_usd = 1.0 } }]\n'
    )
    with pytest.raises(HowHardError, match="requires rate_card or usd_per_mtok"):
        load_config(bad_path)


@pytest.mark.parametrize(
    "budget",
    [
        {"available_credit_usd": 0.05, "session_5h_remaining_credit_usd": 0.05, "weekly_7d_remaining_credit_usd": 2.0},
        {"available_credit_usd": 0.05, "session_5h_remaining_credit_usd": 2.0, "weekly_7d_remaining_credit_usd": 0.05},
    ],
)
def test_selection_excludes_cheaper_cell_with_insufficient_5h_or_7d_credit(tmp_path, budget):
    engine = Engine(directory=tmp_path)
    candidates = [
        {"model": "route-cheap/model", "effort": "low", "usd_per_mtok": 10.0, "credit_budget": budget},
        {
            "model": "route-ok/model",
            "effort": "high",
            "usd_per_mtok": 20.0,
            "credit_budget": {"available_credit_usd": 0.5, "session_5h_remaining_credit_usd": 0.5, "weekly_7d_remaining_credit_usd": 1.0},
        },
    ]
    # 10,000 tokens -> cheap needs 0.1 USD (> 0.05 available); ok needs 0.2 USD (<= 0.5 available).
    result = decide_with_forecasts(
        engine,
        candidates,
        [
            ("route-cheap/model", "low", 0.99, 0.005, 0.005, 10000, 10, None),
            ("route-ok/model", "high", 0.95, 0.04, 0.01, 10000, 20, None),
        ],
    )
    assert result["chosen"]["model"] == "route-ok/model"
    assert result["cap_met"] and not result["needs_review"]
    by_model = {c["model"]: c for c in result["cells"]}
    assert by_model["route-cheap/model"]["needed_credit_usd"] == pytest.approx(0.1)
    assert by_model["route-cheap/model"]["available_credit_usd"] == pytest.approx(0.05)
    assert not by_model["route-cheap/model"]["credit_ok"]
    assert by_model["route-cheap/model"]["admissible"]
    assert by_model["route-ok/model"]["needed_credit_usd"] == pytest.approx(0.2)
    assert by_model["route-ok/model"]["available_credit_usd"] == pytest.approx(0.5)
    assert by_model["route-ok/model"]["credit_ok"]
    assert result["why"] == [
        {
            "model": "route-cheap/model",
            "family": by_model["route-cheap/model"]["family"],
            "effort": "low",
            "reason": "insufficient-credit",
            "message": "excluded: needed_credit_usd 0.1 > available_credit_usd 0.05",
        }
    ]
    receipt = load(engine.config, result["decision_id"])
    assert receipt["chosen"]["model"] == "route-ok/model"


def test_rate_card_candidates_enforce_credit_budget_and_exact_equality_is_schedulable(tmp_path):
    engine = Engine(directory=tmp_path)
    usage = counts(input=100000, output=50000, cache_read=0, cache_write=0, cache_write_5m=0, cache_write_1h=0)
    # input and output rates yield needed_credit_usd = 0.6; the default multiplier makes session_cost_usd equal.
    candidates = [
        {
            "model": "route-tight",
            "effort": "low",
            "rate_card": card(model="route-tight"),
            "credit_budget": {"available_credit_usd": 0.59, "session_5h_remaining_credit_usd": 0.59, "weekly_7d_remaining_credit_usd": 3.0},
        },
        {
            "model": "route-exact",
            "effort": "high",
            "rate_card": card(model="route-exact", rates=counts(input=4, output=8, cache_read=0.2, cache_write=3, cache_write_5m=2.5, cache_write_1h=4)),
            "credit_budget": {"available_credit_usd": 0.8, "session_5h_remaining_credit_usd": 0.8, "weekly_7d_remaining_credit_usd": 2.0},
        },
    ]
    result = decide_with_forecasts(
        engine,
        candidates,
        [
            ("route-tight", "low", 0.98, 0.01, 0.01, 150000, 10, usage),
            ("route-exact", "high", 0.98, 0.01, 0.01, 150000, 15, usage),
        ],
    )
    assert result["chosen"]["model"] == "route-exact"
    by_model = {c["model"]: c for c in result["cells"]}
    assert not by_model["route-tight"]["credit_ok"]
    assert by_model["route-exact"]["needed_credit_usd"] == pytest.approx(0.8)
    assert by_model["route-exact"]["credit_ok"]
    assert result["why"][0]["reason"] == "insufficient-credit"


@pytest.mark.parametrize("policy", ["flag-review", "lowest-bound"])
def test_on_cap_unmet_never_falls_back_to_unschedulable_cell_lacking_credit(tmp_path, policy):
    config_path = tmp_path / "howhard.toml"
    config_path.write_text(f'on_cap_unmet = "{policy}"\n[task_types]\ntask = "standard"\n')
    engine = Engine(config_path)
    candidates = [
        {
            "model": "route-unschedulable/model",
            "effort": "low",
            "usd_per_mtok": 5.0,
            "credit_budget": {"available_credit_usd": 0.01, "session_5h_remaining_credit_usd": 0.01, "weekly_7d_remaining_credit_usd": 1.0},
        },
        {
            "model": "route-schedulable-a/model",
            "effort": "high",
            "usd_per_mtok": 10.0,
            "credit_budget": {"available_credit_usd": 1.0, "session_5h_remaining_credit_usd": 1.0, "weekly_7d_remaining_credit_usd": 5.0},
        },
        {
            "model": "route-schedulable-b/model",
            "effort": "high",
            "usd_per_mtok": 20.0,
            "credit_budget": {"available_credit_usd": 1.0, "session_5h_remaining_credit_usd": 1.0, "weekly_7d_remaining_credit_usd": 5.0},
        },
    ]
    # Unschedulable cell has lower silent risk (even admissible) and lower J, but lacks credit (needs 0.05 > 0.01).
    # Both schedulable cells fail the standard epsilon=0.10 silent cap:
    # schedulable-a has lower J (silent=0.30, J cheaper) while schedulable-b has lower silent risk (silent=0.20) but higher J.
    result = decide_with_forecasts(
        engine,
        candidates,
        [
            ("route-unschedulable/model", "low", 0.99, 0.005, 0.005, 10000, 10, None),
            ("route-schedulable-a/model", "high", 0.65, 0.05, 0.30, 10000, 10, None),
            ("route-schedulable-b/model", "high", 0.75, 0.05, 0.20, 10000, 36000, None),
        ],
    )
    assert not result["cap_met"] and result["needs_review"]
    assert result["chosen"]["model"] != "route-unschedulable/model"
    expected_model = "route-schedulable-a/model" if policy == "flag-review" else "route-schedulable-b/model"
    assert result["chosen"]["model"] == expected_model
    assert result["why"][0]["model"] == "route-unschedulable/model"
    assert result["why"][0]["reason"] == "insufficient-credit"

    receipt = load(engine.config, result["decision_id"])
    assert receipt["chosen"]["model"] == expected_model
    tampered = deepcopy(receipt)
    unsched = next(c for c in tampered["selection"]["cells"] if c["model"] == "route-unschedulable/model")
    tampered["chosen"] = {k: unsched[k] for k in ("model", "family", "effort")}
    tampered["capability"]["executed"] = {
        k: next(c for c in tampered["capability"]["cells"] if c["model"] == "route-unschedulable/model")[k]
        for k in ("key", "classification", "position", "P_clear")
    }
    with pytest.raises(HowHardError, match="not schedulable within credit_budget"):
        load(engine.config, tampered)


def test_raises_howhard_error_when_no_candidate_has_sufficient_credit(tmp_path):
    engine = Engine(directory=tmp_path)
    candidates = [
        {
            "model": "route-a/model",
            "effort": "low",
            "usd_per_mtok": 10.0,
            "credit_budget": {"available_credit_usd": 0.04, "session_5h_remaining_credit_usd": 0.04, "weekly_7d_remaining_credit_usd": 1.0},
        },
        {
            "model": "route-b/model",
            "effort": "high",
            "usd_per_mtok": 25.0,
            "credit_budget": {"available_credit_usd": 0.08, "session_5h_remaining_credit_usd": 0.5, "weekly_7d_remaining_credit_usd": 0.08},
        },
    ]
    # 10,000 tokens -> route-a needs 0.1 USD (> 0.04), route-b needs 0.25 USD (> 0.08).
    with pytest.raises(
        HowHardError,
        match=r"candidates: no candidate has sufficient remaining credit to schedule task \(minimum needed_credit_usd 0\.1 USD, maximum available_credit_usd 0\.08 USD\)",
    ):
        decide_with_forecasts(
            engine,
            candidates,
            [
                ("route-a/model", "low", 0.95, 0.04, 0.01, 10000, 10, None),
                ("route-b/model", "high", 0.99, 0.005, 0.005, 10000, 20, None),
            ],
        )


def test_omitted_credit_budget_preserves_existing_selection_and_reporting(tmp_path):
    engine = Engine(directory=tmp_path)
    candidates = [
        {"model": "route-a/model", "effort": "low", "usd_per_mtok": 10.0},
        {"model": "route-b/model", "effort": "high", "usd_per_mtok": 20.0},
    ]
    result = decide_with_forecasts(
        engine,
        candidates,
        [
            ("route-a/model", "low", 0.98, 0.01, 0.01, 10000, 10, None),
            ("route-b/model", "high", 0.98, 0.01, 0.01, 10000, 3600, None),
        ],
    )
    assert result["chosen"]["model"] == "route-a/model"
    for row in result["cells"]:
        assert row["credit_budget"] is None
        assert row["available_credit_usd"] is None
        assert row["credit_ok"] is True
        assert row["session_cost_usd"] == pytest.approx(row["C_run"])
        assert row["needed_credit_usd"] == pytest.approx(row["session_cost_usd"])
    engine.observe_decision(result["decision_id"], "clean", id="exec-1")
    assert report(engine)["runs"] == 1


def test_session_cost_drives_sorting_and_selection_while_credit_stays_explicit(tmp_path, monkeypatch):
    from howhard import semantic

    config_path = tmp_path / "howhard.toml"
    config_path.write_text(
        'on_cap_unmet = "flag-review"\n'
        '[sampling]\nexplore_within = 0.0\n'
        '[task_types]\ntask = "standard"\n'
    )
    engine = Engine(config_path)
    sampled_runs = {}
    original_sample = semantic.sample

    def capture_sample(prepared, seed):
        parts = original_sample(prepared, seed)
        sampled_runs.update({cell["model"]: float(parts["run"][0, i]) for i, cell in enumerate(prepared[1])})
        return parts

    monkeypatch.setattr(semantic, "sample", capture_sample)
    candidates = [
        {"model": "route-high-credit/model", "effort": "low",
         "rate_card": card(model="route-high-credit/model", rates=counts(input=20, output=0, cache_read=0, cache_write=0, cache_write_5m=0, cache_write_1h=0)),
         "cost_per_credit": 0.1,
         "credit_budget": {"available_credit_usd": 0.25,
                           "session_5h_remaining_credit_usd": 0.25,
                           "weekly_7d_remaining_credit_usd": 1.0}},
        {"model": "route-low-credit/model", "effort": "low",
         "rate_card": card(model="route-low-credit/model", rates=counts(input=5, output=0, cache_read=0, cache_write=0, cache_write_5m=0, cache_write_1h=0)),
         "cost_per_credit": 1.0,
         "credit_budget": {"available_credit_usd": 0.25,
                           "session_5h_remaining_credit_usd": 0.25,
                           "weekly_7d_remaining_credit_usd": 1.0}},
    ]
    result = decide_with_forecasts(
        engine,
        candidates,
        [
            ("route-high-credit/model", "low", 0.98, 0.01, 0.01, 10000, 10,
             counts(input=10000, output=0, cache_read=0, cache_write=0, cache_write_5m=0, cache_write_1h=0)),
            ("route-low-credit/model", "low", 0.98, 0.01, 0.01, 10000, 10,
             counts(input=10000, output=0, cache_read=0, cache_write=0, cache_write_5m=0, cache_write_1h=0)),
        ],
    )
    assert [row["model"] for row in result["cells"]] == [
        "route-high-credit/model", "route-low-credit/model"
    ]
    by_model = {row["model"]: row for row in result["cells"]}
    assert by_model["route-high-credit/model"]["needed_credit_usd"] == pytest.approx(0.2)
    assert by_model["route-high-credit/model"]["session_cost_usd"] == pytest.approx(0.02)
    assert by_model["route-high-credit/model"]["C_run"] == pytest.approx(0.02)
    assert by_model["route-low-credit/model"]["needed_credit_usd"] == pytest.approx(0.05)
    assert by_model["route-low-credit/model"]["session_cost_usd"] == pytest.approx(0.05)
    assert all(row["credit_ok"] for row in result["cells"])
    assert by_model["route-high-credit/model"]["J"] < by_model["route-low-credit/model"]["J"]
    assert result["chosen"]["model"] == "route-high-credit/model"
    assert sampled_runs["route-high-credit/model"] < sampled_runs["route-low-credit/model"]
    displayed = render(result)
    assert "session_cost_usd (cost)" in displayed
    assert "needed_credit_usd (credit)" in displayed
    assert "0.0200  0.2000" in displayed
    assert "0.0500  0.0500" in displayed


def test_credit_budget_compares_list_price_credit_not_scaled_session_cost(tmp_path):
    engine = Engine(directory=tmp_path)
    candidates = [
        {"model": "route-cheap-cost/model", "effort": "low", "usd_per_mtok": 20.0,
         "cost_per_credit": 0.1,
         "credit_budget": {"available_credit_usd": 0.15}},
        {"model": "route-available/model", "effort": "low", "usd_per_mtok": 5.0,
         "cost_per_credit": 1.0,
         "credit_budget": {"available_credit_usd": 0.05}},
    ]
    result = decide_with_forecasts(
        engine,
        candidates,
        [
            ("route-cheap-cost/model", "low", 0.98, 0.01, 0.01, 10000, 10, None),
            ("route-available/model", "low", 0.98, 0.01, 0.01, 10000, 10, None),
        ],
    )
    by_model = {row["model"]: row for row in result["cells"]}
    assert by_model["route-cheap-cost/model"]["session_cost_usd"] < by_model["route-available/model"]["session_cost_usd"]
    assert by_model["route-cheap-cost/model"]["needed_credit_usd"] > by_model["route-cheap-cost/model"]["available_credit_usd"]
    assert not by_model["route-cheap-cost/model"]["credit_ok"]
    assert by_model["route-available/model"]["credit_ok"]
    assert result["chosen"]["model"] == "route-available/model"


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("session_cost_usd", 0.03, "session_cost_usd must match"),
        ("C_run", 0.03, "C_run must match session_cost_usd"),
    ],
)
def test_receipts_verify_session_cost_credit_and_legacy_run_cost_agree(tmp_path, field, value, message):
    engine = Engine(directory=tmp_path)
    result = decide_with_forecasts(
        engine,
        [{"model": "route-a/model", "effort": "low", "usd_per_mtok": 20.0, "cost_per_credit": 0.1}],
        [("route-a/model", "low", 0.98, 0.01, 0.01, 10000, 10, None)],
    )
    receipt = load(engine.config, result["decision_id"])
    receipt["selection"]["cells"][0][field] = value
    with pytest.raises(HowHardError, match=message):
        load(engine.config, receipt)
