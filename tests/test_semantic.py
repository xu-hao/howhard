"""Mechanical integration checks; these fixtures do not evaluate routing quality."""

from copy import deepcopy
import json
import math
import pytest

from conftest import inventory
from howhard import Engine
from howhard.decisions import load
from howhard.llm import LLMError
from howhard.report import report


class FixtureLLM:
    identity = "test-protocol-v1"

    def __init__(self):
        self.calls = []
        self.last_usage = {"seconds": .01, "cost_usd": .001, "tokens": {"input_tokens": 10, "output_tokens": 10}, "models": ["fixture"]}
        self.writer_operations = []
        self.retrieve = False
        self.fail_writer = False

    def complete(self, request):
        self.calls.append(deepcopy(request))
        if request["kind"] == "learn":
            if self.fail_writer:
                raise LLMError("writer temporarily unavailable")
            operations = deepcopy(self.writer_operations)
            self.writer_operations = []
            return {"operations": operations, "reason": "Protocol fixture operation"}
        if request["phase"] == "baseline":
            return {"forecasts": [{"family": c["family"], "effort": c["effort"], "clean": .5, "loud": .45, "silent": .05, "tokens": 10000., "seconds": 30.} for c in request["cells"]], "reason": "Independent baseline fixture"}
        if self.retrieve and not request["retrieval"]:
            return {"request_more": [{"tool": "memory.search", "query": "migration"}, {"tool": "project.read", "path": "README.md"}]}
        forecasts = []
        for memory in request["memories"]:
            if not any(c["family"] == "opus" and c["effort"] == "high" for c in request["cells"]):
                continue
            forecasts.append({"id": memory["id"], "version": memory["version"], "forecasts": [{"family": "opus", "effort": "high", "clean": .98, "loud": .015, "silent": .005, "tokens": 5000., "seconds": 15.}]})
        return {"memory_forecasts": forecasts, "reason": "Memory-conditioned fixture"}


def expert(engine, memory):
    return next(e for e in engine.memory_beliefs()["experts"] if e["key"] == f"memory:{memory['id']}:{memory['version']}")


def test_independent_baseline_and_original_semantic_inputs_are_preserved(project):
    provider = FixtureLLM()
    engine = Engine(project(), llm=provider)
    memory = engine.memory.add("Migration", "Preserve transaction ordering across service boundaries.")
    result = engine.pick("Y", "Preserve τ-ordering; never commit a partial distributed transaction", seed=42, native_candidates=inventory())
    baseline, reader = provider.calls
    assert baseline["phase"] == "baseline"
    assert not {"memories", "beliefs", "evidence", "retrieval"} & baseline.keys()
    assert set(baseline["cells"][0]) == {"model", "family", "effort"}
    assert reader["context"]["task_text"] == result["context"]["task_text"]
    assert reader["memories"][0]["body"] == memory["body"]
    receipt = load(engine.config, result["decision_id"])
    assert receipt["format"] == "howhard-decision-v4"
    assert receipt["semantic"]["memories"][0]["body"] == memory["body"]
    again = engine.pick("Y", result["context"]["task_text"], seed=42, native_candidates=inventory())
    assert again["chosen"] == result["chosen"]
    assert again["decision_id"] == result["decision_id"]
    assert again["backend"]["inference_cached"]
    assert len(provider.calls) == 2


def test_feedback_changes_weight_amendment_replaces_it_and_edit_resets(project):
    provider = FixtureLLM()
    engine = Engine(project(), llm=provider)
    memory = engine.memory.add("Migration", "Transaction tasks need careful reasoning.")
    before = expert(engine, memory)
    pick = engine.pick("Y", "migration", seed=1, native_candidates=inventory([("opus", "high")]))
    engine.observe_decision(pick["decision_id"], "clean", id="first")
    after = expert(engine, memory)
    assert after["log_score"] > before["log_score"]
    assert after["evidence_count"] == 1
    engine.observe("Y", "claude-opus-5-5", "high", "silent", amends="first", id="late")
    amended = expert(engine, memory)
    assert amended["evidence_count"] == 1
    assert amended["log_score"] < before["log_score"]
    revised = engine.memory.edit(memory["id"], body="Revised condition after the escaped failure.")
    assert expert(engine, revised)["evidence_count"] == 0
    assert expert(engine, revised)["log_score"] == math.log(.25)
    assert load(engine.config, pick["decision_id"])["semantic"]["memories"][0]["body"] == memory["body"]


def test_correction_chain_replays_forecasts_and_writer_events_once(project):
    from howhard.llm import SETTINGS
    provider = FixtureLLM()
    engine = Engine(project(), llm=provider)
    model = "claude-opus-5-5"
    engine.config.data["models"][model]["weight"] = .4
    memory = engine.memory.add("Review", "Review can revise an earlier outcome judgment.")
    choice = engine.pick("Y", "review", seed=1, native_candidates=inventory([("opus", "high")]))
    engine.observe_decision(choice["decision_id"], "clean", id="original", tokens=10000, seconds=30)
    initial = expert(engine, memory)
    first_key = engine.memory_learning_context("original")["event_key"]
    target = "original"
    keys = {first_key}
    for index, outcome in enumerate(("loud", "silent", "clean")):
        amendment = f"correction-{index}"
        provider.fail_writer = index == 0
        engine.observe("X", "claude-sonnet-5-5", "low", outcome, amends=target, id=amendment)
        if provider.fail_writer:
            assert engine.last_memory_update["status"] == "failed"
            provider.fail_writer = False
            assert engine.memory_learn(amendment)["status"] == "updated"
        assert engine.last_memory_update["status"] == "updated"
        writer = provider.calls[-1]
        assert writer["kind"] == "learn"
        assert writer["observation"]["id"] == "original"
        assert writer["observation"]["outcome"] == outcome
        assert writer["observation"]["decision_id"] == choice["decision_id"]
        assert writer["observation"]["project_context"] == choice["context"]["project_context"]
        current = expert(engine, memory)
        assert current["evidence_count"] == 1
        worlds = choice["semantic"]["worlds"]
        key = current["key"]
        forecasts = {name: next(c for c in panel if (c["family"], c["effort"]) == ("opus", "high"))
                     for name, panel in worlds.items()}
        # Resource evidence is unchanged; only the outcome likelihood ratio changes.
        expected = initial["log_score"] + SETTINGS["belief"]["temperature"] * .4 * math.log(
            (forecasts[key][outcome] / forecasts["llm"][outcome]) /
            (forecasts[key]["clean"] / forecasts["llm"]["clean"]))
        assert current["log_score"] == pytest.approx(expected)
        metrics_report = report(engine)
        assert metrics_report["runs"] == 1 and metrics_report["outcomes"][outcome] == 1
        for metrics in metrics_report["forecast_metrics"].values():
            assert metrics["runs"] == 1
        assert metrics_report["forecast_metrics"]["llm"]["mean_log_loss"] == pytest.approx(
            -math.log(forecasts["llm"][outcome]))
        event_key = engine.memory_learning_context(amendment)["event_key"]
        assert event_key not in keys
        keys.add(event_key)
        calls = len(provider.calls)
        assert engine.memory_learn("original")["status"] == "already-learned"
        assert engine.memory_learn(amendment)["status"] == "already-learned"
        assert len(provider.calls) == calls
        target = amendment
    assert expert(engine, memory)["log_score"] == pytest.approx(initial["log_score"])


def test_unrelated_cell_does_not_credit_memory_and_removal_stops_influence(project):
    provider = FixtureLLM()
    engine = Engine(project(), llm=provider)
    memory = engine.memory.add("Migration", "Transaction boundary guidance")
    candidates = inventory([("sonnet", "high"), ("opus", "high")])
    candidates[1]["usd_per_mtok"] = 1e9
    pick = engine.pick("Y", "migration", seed=2, native_candidates=candidates)
    assert pick["chosen"]["family"] == "sonnet"
    assert pick["semantic"]["exposures"][f"memory:{memory['id']}:{memory['version']}"] == [["opus", "high"]]
    engine.observe_decision(pick["decision_id"], "clean", id="unrelated")
    assert expert(engine, memory)["evidence_count"] == 0
    engine.memory.remove(memory["id"])
    result = engine.pick("Y", "migration", seed=2, native_candidates=inventory())
    assert result["semantic"]["memories"] == []
    assert set(result["semantic"]["worlds"]) == {"llm"}
    engine.memory.restore(memory["id"])
    assert engine.pick("Y", "migration", seed=2, native_candidates=inventory())["semantic"]["memories"][0]["id"] == memory["id"]


def test_writer_can_add_remove_and_split_with_recoverable_failure(project):
    provider = FixtureLLM()
    engine = Engine(project(), llm=provider)
    provider.writer_operations = [{"op": "add", "title": "Learned", "body": "A provisional lesson", "evidence_ids": ["run"]}]
    engine.observe("Y", "claude-opus-5-5", "high", "loud", id="run")
    created = engine.memory.list()[0]
    assert engine.last_memory_update["status"] == "updated"
    calls = len(provider.calls)
    assert engine.memory_learn("run")["status"] == "already-learned"
    assert len(provider.calls) == calls
    provider.writer_operations = [{"op": "split", "id": created["id"], "parts": [{"title": "A", "body": "Condition A"}, {"title": "B", "body": "Condition B"}]}]
    engine.observe("Y", "claude-opus-5-5", "high", "clean", id="second")
    assert len(engine.memory.list()) == 2
    provider.fail_writer = True
    engine.observe("Y", "claude-opus-5-5", "high", "clean", id="durable")
    assert engine.last_memory_update["status"] == "failed"
    assert any(r["id"] == "durable" for r in engine.rows)
    provider.fail_writer = False
    provider.writer_operations = [{"op": "remove", "id": engine.memory.list()[0]["id"]}]
    engine.memory_learn("durable")
    assert len(engine.memory.list()) == 1


def test_retrieval_results_frozen_and_changed_memory_invalidates_cache(project):
    provider = FixtureLLM()
    provider.retrieve = True
    config = project()
    (config.base / "README.md").write_text("Project semantics before the run")
    engine = Engine(config, llm=provider)
    memory = engine.memory.add("Migration", "Ordering constraints")
    result = engine.pick("Y", "migration", seed=4, native_candidates=inventory())
    assert any(r["request"]["tool"] == "project.read" for r in result["semantic"]["retrieval"])
    (config.base / "README.md").write_text("A different project state after the run")
    engine.observe_decision(result["decision_id"], "clean", id="done")
    receipt = load(config, result["decision_id"])
    assert "before" in receipt["semantic"]["retrieval"][1]["result"]["text"]
    before = len(provider.calls)
    engine.memory.edit(memory["id"], body="New semantics")
    engine.pick("Y", "migration", seed=4, native_candidates=inventory())
    assert len(provider.calls) > before


def test_provider_failure_never_falls_back(project):
    class Broken:
        identity = "broken"
        def complete(self, request):
            raise LLMError("unavailable")
    engine = Engine(project(), llm=Broken())
    with pytest.raises(LLMError):
        engine.pick("Y", "task", seed=1, native_candidates=inventory())
    engine.backend["mode"] = "auto"
    with pytest.raises(LLMError):
        engine.pick("Y", "task", seed=1, native_candidates=inventory())
    events = [json.loads(line) for line in (engine.config.base / ".howhard/inference.jsonl").read_text().splitlines()]
    assert len(events) == 2 and all(e["status"] == "failed" for e in events)
    assert all(e["cost_usd"] is None for e in events)


@pytest.mark.parametrize("corrections", [["loud"], ["silent"], ["loud", "clean"]])
@pytest.mark.parametrize("refresh", [False, True])
def test_stale_writer_outcome_is_rejected_without_memory_mutation(project, corrections, refresh):
    from howhard.validation import HowHardError
    config = project()
    engine = Engine(config)
    engine.observe("Y", "claude-opus-5-5", "high", "clean", id="original")
    pending = engine.memory_learning_context("original")
    other = Engine(config)
    target = "original"
    for index, outcome in enumerate(corrections):
        target = other.observe("Y", "claude-opus-5-5", "high", outcome,
                               amends=target, id=f"correction-{index}")["id"]
    if refresh:
        engine.refresh()
    with pytest.raises(HowHardError, match="evidence changed"):
        engine.apply_memory_learning("original", {"operations": [{"op": "add", "title": "Stale", "body": "Old hypothesis"}], "reason": "Old feedback"}, expected_revision=pending["memory_revision"], event_key=pending["event_key"])
    assert engine.memory.list() == []


def test_project_retrieval_rejects_changed_context_and_private_state(project):
    from howhard.validation import HowHardError
    config = project()
    readme = config.base / "README.md"
    readme.write_text("Original project")
    engine = Engine(config)
    pending = engine.prepare("Y", "Task", native_candidates=inventory())
    readme.write_text("Changed project")
    with pytest.raises(HowHardError, match="Project changed"):
        engine.retrieve(pending["request_id"], [{"tool": "project.read", "path": "README.md"}])
    for path in ("../outside.py", ".howhard/decisions/file.json", ".git/config"):
        with pytest.raises(HowHardError, match="outside project or excluded"):
            engine.retrieve(pending["request_id"], [{"tool": "project.read", "path": path}])


def test_corrupted_inference_cache_is_recomputed(project):
    provider = FixtureLLM()
    engine = Engine(project(), llm=provider)
    first = engine.pick("Y", "Task", seed=3, native_candidates=inventory())
    cache = next((engine.config.base / ".howhard/llm-cache").glob("*.json"))
    data = json.loads(cache.read_bytes())
    data["response"]["reason"] = "Tampered result"
    cache.write_text(json.dumps(data))
    second = engine.pick("Y", "Task", seed=3, native_candidates=inventory())
    assert len(provider.calls) == 2
    assert second["semantic"]["reason"] == first["semantic"]["reason"]
    assert not second["backend"]["inference_cached"]


def test_invalid_seed_does_not_spend_inference(project):
    from howhard.validation import HowHardError
    provider = FixtureLLM()
    engine = Engine(project(), llm=provider)
    with pytest.raises(HowHardError, match="seed"):
        engine.pick("Y", "Task", seed=-1, native_candidates=inventory())
    assert provider.calls == []


def test_project_retrieval_does_not_read_files_omitted_from_git_snapshot(project):
    import subprocess
    from howhard.validation import HowHardError
    config = project()
    subprocess.run(["git", "init", "--quiet", str(config.base)], check=True)
    (config.base / ".gitignore").write_text("ignored.py\n")
    (config.base / "README.md").write_text("A public project description")
    (config.base / "ignored.py").write_text("excluded from decision-time project capture")
    engine = Engine(config)
    pending = engine.prepare("Y", "Task", native_candidates=inventory())
    # Internal requests/memory state must not change the project fingerprint.
    engine.retrieve(pending["request_id"], [{"tool": "project.read", "path": "README.md"}])
    with pytest.raises(HowHardError, match="outside the captured snapshot"):
        engine.retrieve(pending["request_id"], [{"tool": "project.read", "path": "ignored.py"}])
