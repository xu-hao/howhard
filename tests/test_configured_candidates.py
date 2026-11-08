"""External-router configuration and feedback mechanics, without provider calls."""

from copy import deepcopy
import json

import pytest

from howhard import Engine, load_config
from howhard.cli import main
from howhard.decisions import load
from howhard.validation import HowHardError
from test_semantic import FixtureLLM


INVENTORY = [
    {"model": "provider-a/shared-model", "effort": "high"},
    {"model": "provider-b/shared-model", "effort": "high"},
]


def configure(directory, candidates=INVENTORY):
    path = directory / "howhard.toml"
    lines = ['on_cap_unmet = "flag-review"']
    if not candidates:
        lines.append("candidates = []")
    for cell in candidates:
        lines.append("\n[[candidates]]")
        lines.extend(f"{key} = {json.dumps(value)}" for key, value in cell.items())
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize("method", ["prepare", "pick", "summary"])
def test_python_selection_uses_configured_cross_provider_list(tmp_path, method):
    configure(tmp_path)
    engine = Engine(directory=tmp_path, llm=FixtureLLM())
    result = getattr(engine, method)("bookkeeping", "Repair transaction ordering")
    cells = result["baseline_request"]["cells"] if method == "prepare" else result["cells"]
    assert {c["model"] for c in cells} == {c["model"] for c in INVENTORY}
    assert len({c["family"] for c in cells}) == 2
    if method == "pick":
        assert result["chosen"]["model"] in {c["model"] for c in INVENTORY}
        assert load(engine.config, result["decision_id"])["native_candidates"] == INVENTORY


@pytest.mark.parametrize("command", ["context", "pick", "summary"])
def test_cli_uses_configured_list_and_file_replaces_it(tmp_path, monkeypatch, capsys, command):
    configure(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main([command, "bookkeeping", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    cells = result["baseline_request"]["cells"] if command == "context" else result["cells"]
    assert {c["model"] for c in cells} == {c["model"] for c in INVENTORY}
    available = [INVENTORY[1]]
    path = tmp_path / "available.json"
    path.write_text(json.dumps(available))
    assert main([command, "bookkeeping", "--candidates", str(path), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    cells = result["baseline_request"]["cells"] if command == "context" else result["cells"]
    assert [c["model"] for c in cells] == [available[0]["model"]]


def test_availability_override_is_complete_and_receipt_feedback_survives_config_change(tmp_path):
    path = configure(tmp_path)
    first = Engine(directory=tmp_path, llm=FixtureLLM())
    choice = first.pick("bookkeeping", native_candidates=[INVENTORY[0]], seed=3)
    configure(tmp_path, [INVENTORY[1]])
    later = Engine(path, llm=FixtureLLM())
    result = later.pick("bookkeeping", seed=3)
    assert result["chosen"]["model"] == INVENTORY[1]["model"]
    assert result["native_candidates"] == [INVENTORY[1]]
    observed = later.observe_decision(choice["decision_id"], "clean", id="earlier-run")
    assert observed["model"] == INVENTORY[0]["model"]
    before = deepcopy(later.raw_rows)
    with pytest.raises(HowHardError, match="candidates"):
        later.pick("bookkeeping", native_candidates=[])
    assert later.raw_rows == before


@pytest.mark.parametrize("candidates", [[], [INVENTORY[0], INVENTORY[0]],
                                      [{"model": "provider-a/model", "effort": "model_default"}],
                                      [{"model": "provider-a/model", "effort": "high", "usd_per_mtok": -1}]])
def test_invalid_configured_inventory_is_rejected_before_state_creation(tmp_path, candidates):
    path = configure(tmp_path, candidates)
    with pytest.raises(HowHardError):
        load_config(path)
    assert not (tmp_path / ".howhard").exists()


def test_explicit_empty_cli_list_does_not_fall_back_or_create_state(tmp_path, monkeypatch, capsys):
    configure(tmp_path)
    path = tmp_path / "empty.json"
    path.write_text("[]")
    monkeypatch.chdir(tmp_path)
    assert main(["context", "bookkeeping", "--candidates", str(path)]) == 2
    assert "candidates" in capsys.readouterr().err
    assert not (tmp_path / ".howhard").exists()


def test_changing_configured_availability_invalidates_pending_request(tmp_path):
    configure(tmp_path)
    first = Engine(directory=tmp_path)
    pending = first.prepare("bookkeeping")
    response = FixtureLLM().complete(pending["baseline_request"])
    response["memory_forecasts"] = []
    configure(tmp_path, [INVENTORY[1]])
    later = Engine(directory=tmp_path)
    with pytest.raises(HowHardError, match="stale"):
        later.decide(pending["request_id"], response, seed=3)


def test_one_provider_experience_informs_another_without_counterfactual_credit(tmp_path):
    configure(tmp_path)
    source = Engine(directory=tmp_path)
    source.observe("bookkeeping", INVENTORY[0]["model"], "high", "loud", native=True,
                   task_text="Repair transaction ordering", id="provider-a-run")
    pending = source.memory_learning_context("provider-a-run")
    update = source.apply_memory_learning("provider-a-run", {
        "operations": [{"op": "add", "title": "Transaction ordering", "body": "Provisional lesson from the observed failure.",
                        "evidence_ids": ["provider-a-run"]}], "reason": "Synthetic writer response"},
        expected_revision=pending["memory_revision"], event_key=pending["event_key"])
    lesson = update["operations"][0]

    class Reader(FixtureLLM):
        def complete(self, request):
            if request.get("phase") != "memory":
                return super().complete(request)
            forecasts = [{"family": c["family"], "effort": c["effort"], "clean": .98,
                          "loud": .015, "silent": .005, "tokens": 5000., "seconds": 15.}
                         for c in request["cells"]]
            assert lesson["id"] in {m["id"] for m in request["memories"]}
            assert request["evidence"][0]["model"] == INVENTORY[0]["model"]
            return {"memory_forecasts": [{"id": lesson["id"], "version": lesson["version"], "forecasts": forecasts}],
                    "reason": "Synthetic cross-provider forecast; not routing-quality evidence"}

    router = Engine(directory=tmp_path, llm=Reader())
    key = f"memory:{lesson['id']}:{lesson['version']}"
    assert next(e for e in router.memory_beliefs()["experts"] if e["key"] == key)["evidence_count"] == 0
    choice = router.pick("bookkeeping", "Repair transaction ordering", native_candidates=[INVENTORY[1]], seed=3)
    assert choice["chosen"]["model"] == INVENTORY[1]["model"]
    assert choice["semantic"]["worlds"][key][0]["clean"] > choice["semantic"]["worlds"]["llm"][0]["clean"]
    router.observe_decision(choice["decision_id"], "clean", id="provider-b-run")
    belief = next(e for e in router.memory_beliefs()["experts"] if e["key"] == key)
    assert belief["evidence_count"] == 1
    assert {r["model"]: r["outcome"] for r in router.rows} == {
        INVENTORY[0]["model"]: "loud", INVENTORY[1]["model"]: "clean"}
