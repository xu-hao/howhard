"""CLI/adapter protocol checks only, not an evaluation of model quality."""

import json
import sys
import pytest

from conftest import inventory
from howhard import Engine
from howhard.cli import main
from howhard.llm import CommandLLM, LLMError
from howhard.report import report
from howhard.validation import HowHardError
from test_semantic import FixtureLLM


def test_memory_cli_full_lifecycle(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    def run(*args):
        assert main(["memory", *args, "--json"]) == 0
        return json.loads(capsys.readouterr().out)
    a = run("add", "--title", "A", "--body", "Migration boundary")
    b = run("add", "--title", "B", "--body", "Rollback validation")
    assert run("edit", a["id"], "--body", "Revised migration boundary")["version"] == 2
    assert run("read", a["id"], "--version", "1")["body"] == "Migration boundary"
    assert run("remove", a["id"])["active"] is False
    assert len(run("list")["memories"]) == 1
    assert run("restore", a["id"])["active"]
    merged = run("merge", a["id"], b["id"], "--title", "AB", "--body", "Combined lesson")
    parts = tmp_path / "parts.json"
    parts.write_text(json.dumps([{"title": "X", "body": "X lesson"}, {"title": "Y", "body": "Y lesson"}]))
    split = run("split", merged["id"], "--file", str(parts))
    assert len(split["memories"]) == 2
    assert len(run("history")["history"]) == 7
    assert run("search", "lesson")["memories"]
    assert run("beliefs")["kind"] == "tempered-predictive-evidence"


def test_calling_agent_request_retrieval_decision_and_writer(project):
    engine = Engine(project(), llm="off")
    entry = engine.memory.add("Migration", "Use boundary validation")
    request = engine.prepare("Y", "Repair transaction ordering", native_candidates=inventory())
    assert "memories" not in request["baseline_request"]
    looked_up = engine.retrieve(request["request_id"], [{"tool": "memory.read", "id": entry["id"]}])
    fixture = FixtureLLM()
    response = fixture.complete(looked_up["baseline_request"])
    response["memory_forecasts"] = []
    picked = engine.decide(looked_up["request_id"], response, seed=4)
    assert picked["semantic"]["backend"] == "calling-agent"
    engine.observe_decision(picked["decision_id"], "loud", id="real-reference")
    context = engine.memory_learning_context("real-reference")
    result = engine.apply_memory_learning("real-reference", {"operations": [{"op": "remove", "id": entry["id"]}], "reason": "Retire this hypothesis"}, expected_revision=context["memory_revision"], event_key=context["event_key"])
    assert result["status"] == "updated" and not engine.memory.list()
    with pytest.raises(HowHardError, match="stale"):
        engine.decide(looked_up["request_id"], response, seed=4)


def test_command_adapter_and_cli_integration(tmp_path, monkeypatch, capsys, candidate_file):
    script = tmp_path / "fixture_model.py"
    script.write_text('''import json,sys
r=json.load(sys.stdin)["request"]
if r["kind"]=="learn":
 print(json.dumps({"operations":[{"op":"add","title":"Learned","body":"Provisional lesson","evidence_ids":[r["observation"]["id"]]}],"reason":"recorded outcome"}))
elif r.get("phase")=="baseline":
 print(json.dumps({"forecasts":[{"family":c["family"],"effort":c["effort"],"clean":0.8,"loud":0.15,"silent":0.05,"tokens":1000,"seconds":10} for c in r["cells"]],"reason":"protocol fixture"}))
else:
 print(json.dumps({"memory_forecasts":[],"reason":"no additional forecast"}))
''')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOWHARD_LLM_COMMAND", json.dumps([sys.executable, str(script)]))
    assert main(["pick", "bookkeeping", "--text", "Track issue", "--llm", "command", "--json", "--seed", "8", "--candidates", str(candidate_file)]) == 0
    picked = json.loads(capsys.readouterr().out)
    assert picked["backend"]["mode"] == "semantic"
    assert main(["observe", picked["decision_id"], "clean", "--llm", "command", "--json", "--id", "run"]) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["memory_update"]["status"] == "updated"
    assert main(["memory", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["memories"][0]["title"] == "Learned"
    assert main(["report", "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["forecast_metrics"]["weighted_memory"]["runs"] == 1
    assert stats["inference"]["total_usd"] is None


def test_command_invalid_json_timeout_and_secret_free_identity(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text('print("not JSON")')
    with pytest.raises(LLMError, match="JSON"):
        CommandLLM([sys.executable, str(bad)]).complete({})
    slow = tmp_path / "slow.py"
    slow.write_text('import time; time.sleep(10)')
    with pytest.raises(LLMError, match="timed out"):
        CommandLLM([sys.executable, str(slow)], timeout=.05).complete({})
    assert "secret-token" not in CommandLLM(["client", "--key", "secret-token"]).identity


def test_report_scores_observed_actions_and_keeps_unknown_costs(project):
    engine = Engine(project(), llm=FixtureLLM())
    picked = engine.pick("Y", "Task", seed=1, native_candidates=inventory([("opus", "high")]))
    engine.observe_decision(picked["decision_id"], "clean", id="one", cost_usd=.2)
    stats = report(engine)
    assert stats["runs"] == 1 and stats["outcomes"]["clean"] == 1
    assert stats["forecast_metrics"]["llm"]["mean_log_loss"] == pytest.approx(-__import__("math").log(.5))
    assert stats["worker_cost"]["total_usd"] == .2
    assert stats["inference"]["calls"] == 2  # reader + writer, no memories yet
    assert stats["inference"]["total_usd"] == .002


def test_calling_agent_costs_and_decisions_are_not_reported_as_free(project):
    engine = Engine(project(), llm="off")
    pending = engine.prepare("Y", "Task", native_candidates=inventory())
    response = FixtureLLM().complete(pending["baseline_request"])
    response["memory_forecasts"] = []
    engine.decide(pending["request_id"], response, seed=1)
    stats = report(engine)
    assert stats["inference"]["calls"] == 0
    assert stats["inference"]["external_submissions"] == 1
    assert stats["inference"]["total_usd"] is None
    assert stats["decisions"]["calls"] == 1
