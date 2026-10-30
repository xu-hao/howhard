"""Public behavior after removal of statistical routing."""

import json

import pytest

from howhard import Engine
from howhard.cli import main
from howhard.llm import LLMError
from howhard.report import report
from test_semantic import FixtureLLM

INVENTORY = [{"model": "host-model", "effort": "ultra"}]


def test_llm_switch_is_rejected_before_state_creation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(["paths", "--llm", "off"])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_removed_python_mode_is_rejected(tmp_path):
    with pytest.raises(LLMError, match="adapter"):
        Engine(directory=tmp_path, llm="off")
    assert list(tmp_path.iterdir()) == []


def test_llm_only_forecasts_determine_cap_without_statistical_floor(tmp_path):
    class Confident(FixtureLLM):
        def complete(self, request):
            result = super().complete(request)
            if request.get("phase") == "baseline":
                for row in result["forecasts"]:
                    row.update(clean=.999, loud=.0009, silent=.0001)
            return result
    engine = Engine(directory=tmp_path, llm=Confident())
    choice = engine.pick("task", native_candidates=INVENTORY, seed=1)
    assert choice["cap_met"] and not choice["needs_review"]
    assert choice["cells"][0]["P_silent"]["upper"] == pytest.approx(.0001)
    assert set(choice["semantic"]["worlds"]) == {"llm"}
    assert "statistical" not in {e["key"] for e in engine.memory_beliefs()["experts"]}
    assert not (engine.config.state_root / "posterior.npz").exists()
    engine.observe_decision(choice["decision_id"], "clean", id="run")
    assert set(report(engine)["forecast_metrics"]) == {"llm", "weighted_memory"}


def test_no_provider_stops_routing_but_calling_agent_protocol_works(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise LLMError("No model available; use the calling-agent protocol")
    monkeypatch.delenv("HOWHARD_LLM_COMMAND", raising=False)
    monkeypatch.setattr("howhard.llm.ClaudeCLI", unavailable)
    engine = Engine(directory=tmp_path)
    with pytest.raises(LLMError, match="calling-agent"):
        engine.pick("task", native_candidates=INVENTORY, seed=1)
    assert not (engine.config.state_root / "decisions").exists()
    pending = engine.prepare("task", native_candidates=INVENTORY)
    response = FixtureLLM().complete(pending["baseline_request"])
    response["memory_forecasts"] = []
    choice = engine.decide(pending["request_id"], response, seed=1)
    assert choice["semantic"]["backend"] == "calling-agent"
    engine.observe_decision(choice["decision_id"], "clean", id="actual")
    assert engine.evidence("actual")["outcome"] == "clean"


@pytest.mark.parametrize("command", ["export-prior", "import-prior"])
def test_prior_commands_are_removed_before_state_creation(tmp_path, monkeypatch, command):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        main([command])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("setting", ['cache = "posterior.npz"', '[prior]\nimport = "prior.json"',
                                     '[transfer]\nvariance_inflation = 2.0'])
def test_removed_statistical_configuration_is_rejected(tmp_path, setting):
    from howhard.validation import HowHardError
    path = tmp_path / "howhard.toml"
    path.write_text('on_cap_unmet = "flag-review"\n' + setting + '\n')
    with pytest.raises(HowHardError, match="unknown"):
        Engine(path)
    assert not (tmp_path / ".howhard").exists()


def test_legacy_receipts_do_not_block_evidence_memory_or_reports(tmp_path):
    from howhard.context import canonical, digest
    from howhard.decisions import load
    from howhard.validation import HowHardError
    engine = Engine(directory=tmp_path, llm=FixtureLLM())
    choice = engine.pick("task", native_candidates=INVENTORY, seed=1)
    legacy = load(engine.config, choice["decision_id"])
    legacy["format"] = "howhard-decision-v3"
    legacy.pop("semantic")
    legacy_id = digest(legacy)
    legacy_path = engine.config.state_root / "decisions" / f"{legacy_id}.json"
    legacy_path.write_bytes(canonical(legacy))
    row = engine.observe_decision(choice["decision_id"], "clean", id="historical")
    journal = engine.config.path(engine.config.data["journal"])
    row["decision_id"] = legacy_id
    journal.write_bytes(canonical(row) + b"\n")
    before = journal.read_bytes(), legacy_path.read_bytes()
    engine.refresh()
    assert engine.evidence("historical")["outcome"] == "clean"
    assert engine.memory_beliefs()["unscored_runs"] == 1
    assert report(engine)["unsupported_receipts"] == ["historical"]
    assert engine.memory_learning_context("historical")["request"]["decision"] is None
    with pytest.raises(HowHardError):
        engine.observe_decision(legacy_id, "clean", id="unrun")
    assert (journal.read_bytes(), legacy_path.read_bytes()) == before


def test_memory_forecast_can_raise_risk_and_require_review(tmp_path):
    from conftest import inventory
    engine = Engine(directory=tmp_path)
    memory = engine.memory.add("Boundary", "Cross-service changes can escape local tests")
    pending = engine.prepare("task", native_candidates=inventory([("opus", "high")]))
    cell = pending["baseline_request"]["cells"][0]
    reference = {"family": cell["family"], "effort": cell["effort"], "clean": .98,
                 "loud": .015, "silent": .005, "tokens": 1000, "seconds": 10}
    response = {"forecasts": [reference], "memory_forecasts": [{"id": memory["id"],
                "version": memory["version"], "forecasts": [{**reference, "clean": .5, "loud": .1, "silent": .4}]}],
                "reason": "The observed boundary makes this candidate risky"}
    choice = engine.decide(pending["request_id"], response, seed=1)
    assert not choice["cap_met"] and choice["needs_review"]
    assert choice["cells"][0]["P_silent"]["upper"] == .4


def test_external_observation_invalidates_prepared_request(tmp_path):
    from howhard.validation import HowHardError
    engine = Engine(directory=tmp_path)
    pending = engine.prepare("task", native_candidates=INVENTORY)
    response = FixtureLLM().complete(pending["baseline_request"])
    response["memory_forecasts"] = []
    other = Engine(directory=tmp_path)
    other.observe("task", "host-model", "ultra", "clean", id="new-evidence")
    with pytest.raises(HowHardError, match="stale"):
        engine.decide(pending["request_id"], response, seed=1)


@pytest.mark.parametrize("mode", ["off", "auto", "claude", "command"])
def test_removed_environment_modes_cannot_restore_routing(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("HOWHARD_LLM", mode)
    engine = Engine(directory=tmp_path)
    with pytest.raises(LLMError, match="removed"):
        engine.pick("task", native_candidates=INVENTORY)
    assert not (engine.config.state_root / "decisions").exists()
