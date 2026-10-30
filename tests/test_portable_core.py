"""Portable public-boundary mechanics, not routing-quality evidence."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from conftest import inventory
from howhard import Engine
from howhard.cli import main
from howhard.validation import HowHardError
from test_semantic import FixtureLLM


def native(model="host-model-a", effort="high", **options):
    return {"model": model, "effort": effort, **options}


def test_native_candidates_precede_baseline_and_survive_feedback_reload(tmp_path):
    engine = Engine(directory=tmp_path, llm="off")
    inventory = [native(), native("host-model-b", "ultra")]
    request = engine.prepare("task", "Repair rollback", native_candidates=inventory)
    cells = request["baseline_request"]["cells"]
    assert {(c["model"], c["effort"]) for c in cells} == {("host-model-a", "high"), ("host-model-b", "ultra")}
    assert len({c["family"] for c in cells}) == 2
    assert not {"haiku", "sonnet", "opus", "fable"} & {c["family"] for c in cells}
    response = FixtureLLM().complete(request["baseline_request"])
    response["memory_forecasts"] = []
    choice = engine.decide(request["request_id"], response, seed=3)
    assert choice["chosen"]["model"] in {c["model"] for c in inventory}
    fresh = Engine(directory=tmp_path, llm="off")
    observed = fresh.observe_decision(choice["decision_id"], "loud", id="execution", tokens=100, cost_usd=.2)
    assert observed["model"] == choice["chosen"]["model"]
    assert fresh.evidence("execution")["decision_id"] == choice["decision_id"]
    assert fresh.memory_learning_context("execution")["request"]["observation"]["outcome"] == "loud"


def test_any_unpriced_candidate_omits_worker_money_for_every_candidate(tmp_path):
    engine = Engine(directory=tmp_path, llm="off")
    a = engine.pick("task", native_candidates=[native(usd_per_mtok=1), native("host-model-b")], seed=3)
    b = engine.pick("task", native_candidates=[native(usd_per_mtok=1e9), native("host-model-b")], seed=3)
    assert a["chosen"] == b["chosen"]
    assert a["cost_policy"]["worker_money_included"] is False
    assert [c["J"] for c in sorted(a["cells"], key=lambda c: c["model"])] == pytest.approx(
        [c["J"] for c in sorted(b["cells"], key=lambda c: c["model"])])
    assert next(c for c in a["cells"] if c["model"] == "host-model-b")["C_run"] is None
    assert next(c for c in a["cells"] if c["model"] == "host-model-a")["C_run"] > 0
    priced = engine.pick("task", native_candidates=[native(usd_per_mtok=1), native("host-model-b", usd_per_mtok=2)], seed=3)
    assert priced["cost_policy"]["worker_money_included"] is True
    assert all(c["C_run"] is not None for c in priced["cells"])
    from howhard.decisions import load
    assert load(engine.config, priced["decision_id"])["native_candidates"] == priced["native_candidates"]
    hosted = engine.pick("task", native_candidates=[native("claude-opus-5-5")], seed=3)
    assert hosted["cells"][0]["C_run"] is None
    assert hosted["cost_policy"]["worker_money_included"] is False


def test_matching_retry_is_idempotent_conflict_is_refused_and_correction_writer_replays(tmp_path):
    engine = Engine(directory=tmp_path, llm="off")
    choice = engine.pick("task", "Repair rollback", native_candidates=[native()], seed=3)
    first = engine.observe_decision(choice["decision_id"], "clean", id="execution", tokens=100)
    snapshot = deepcopy(engine.raw_rows)
    assert engine.observe_decision(choice["decision_id"], "clean", id="execution", tokens=100) == first
    assert engine.raw_rows == snapshot
    with pytest.raises(HowHardError, match="conflict"):
        engine.observe_decision(choice["decision_id"], "loud", id="execution", tokens=100)
    assert engine.raw_rows == snapshot
    pending = engine.memory_learning_context("execution")
    engine.observe("task", "host-model-a", "high", "loud", amends="execution", id="correction")
    current = engine.memory_learning_context("correction")
    assert current["event_key"] != pending["event_key"]
    assert current["request"]["observation"]["tokens"] == 100
    with pytest.raises(HowHardError, match="evidence changed"):
        engine.apply_memory_learning("execution", {"operations": [], "reason": "old"},
                                     expected_revision=pending["memory_revision"], event_key=pending["event_key"])
    applied = engine.apply_memory_learning("execution", {"operations": [{"op": "add", "title": "Review", "body": "Provisional rollback lesson", "evidence_ids": ["execution"]}], "reason": "actual feedback"},
                                           expected_revision=current["memory_revision"], event_key=current["event_key"])
    assert applied["status"] == "updated"
    assert engine.apply_memory_learning("correction", {"operations": [], "reason": "retry"},
                                        expected_revision=current["memory_revision"], event_key=current["event_key"])["status"] == "already-learned"
    entry = engine.memory.list()[0]
    engine.memory.edit(entry["id"], body="Revised lesson")
    engine.memory.remove(entry["id"])
    assert not engine.memory.list()
    engine.memory.restore(entry["id"])
    assert len(engine.rows) == 1 and engine.evidence("correction")["outcome"] == "loud"


def test_raw_retry_preserves_first_auto_snapshot_after_project_changes(tmp_path):
    (tmp_path / "README.md").write_text("Before execution")
    engine = Engine(directory=tmp_path, llm="off")
    engine.pick("task", native_candidates=[native()], seed=3)
    first = engine.observe("task", "host-model-a", "high", "clean", task_text="Repair rollback", id="execution", tokens=100, note="checked")
    before = deepcopy(engine.raw_rows)
    (tmp_path / "README.md").write_text("After execution, changed source")
    assert engine.observe("task", "host-model-a", "high", "clean", task_text="Repair rollback", id="execution", tokens=100, note="checked") == first
    assert engine.raw_rows == before

    for changes in ({"task_text": "Different task"}, {"project_context": {"files": 123}}, {"tokens": 101}, {"note": "different"}):
        options = {"task_text": "Repair rollback", "id": "execution", "tokens": 100, "note": "checked", **changes}
        with pytest.raises(HowHardError, match="conflict"):
            engine.observe("task", "host-model-a", "high", "clean", **options)
    assert engine.raw_rows == before


def test_raw_retry_retains_derived_risk_and_exact_effort(project):
    engine = Engine(project(), llm="off")
    first = engine.observe("Y", "claude-opus-5-5", "high", "clean", id="execution")
    engine.config.data["task_types"]["Y"] = "critical"
    assert engine.observe("Y", "claude-opus-5-5", "high", "clean", id="execution") == first
    with pytest.raises(HowHardError, match="conflict"):
        engine.observe("Y", "claude-opus-5-5", "high", "clean", id="execution", risk_class="critical")
    with pytest.raises(HowHardError, match="conflict"):
        engine.observe("Y", "claude-opus-5-5", "low", "clean", id="execution")


def test_cli_native_input_evidence_and_paths_without_project_adapter(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps([native()]))
    assert main(["pick", "task", "--candidates", str(inventory), "--llm", "off", "--json"]) == 0
    decision = json.loads(capsys.readouterr().out)
    assert main(["observe", decision["decision_id"], "clean", "--id", "execution", "--llm", "off", "--json"]) == 0
    capsys.readouterr()
    assert main(["evidence", "execution", "--raw", "--llm", "off", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "execution"
    assert main(["paths", "--llm", "off", "--json"]) == 0
    paths = json.loads(capsys.readouterr().out)
    assert paths["project_root"] == str(tmp_path)
    assert paths["state_root"] == str(tmp_path / ".howhard")
    assert not (tmp_path / "howhard.toml").exists()


def test_git_worktrees_share_store_capture_current_tree_and_isolate_repositories(tmp_path):
    def git(directory, *args):
        return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "README.md").write_text("Original checkout")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-qm", "initial")
    worktrees = [tmp_path / "one", tmp_path / "two"]
    try:
        for index, tree in enumerate(worktrees):
            git(repo, "worktree", "add", "-qb", f"task-{index}", str(tree))
        first = Engine(directory=worktrees[0], llm="off")
        request = first.prepare("task", "Repair rollback", native_candidates=[native()])
        chosen = first.pick("task", "Repair rollback", native_candidates=[native()], seed=3)
        first.observe_decision(chosen["decision_id"], "clean", id="execution")
        first.memory.add("Review", "A provisional lesson")
        (worktrees[1] / "README.md").write_text("Different current worktree")
        second = Engine(directory=worktrees[1], llm="off")
        assert second.config.state_root == first.config.state_root
        assert len(second.rows) == 1 and len(second.memory.list()) == 1
        response = FixtureLLM().complete(request["baseline_request"])
        response["memory_forecasts"] = []
        with pytest.raises(HowHardError, match="another worktree"):
            second.decide(request["request_id"], response, seed=3)
        with pytest.raises(HowHardError, match="another worktree"):
            second.retrieve(request["request_id"], [{"tool": "project.read", "path": "README.md"}])
        retry = second.observe_decision(chosen["decision_id"], "clean", id="execution")
        assert retry["project_context"] == chosen["context"]["project_context"]
        next_choice = second.pick("task", "Repair rollback", native_candidates=[native()], seed=3)
        assert next_choice["context"]["project_context"]["text"] != chosen["context"]["project_context"]["text"]
        assert next_choice["paths"]["project_root"] == str(worktrees[1])
        other = tmp_path / "other"
        other.mkdir()
        git(other, "init", "-q")
        isolated = Engine(directory=other, llm="off")
        assert isolated.config.state_root != first.config.state_root
        assert isolated.rows == [] and isolated.memory.list() == []
    finally:
        for tree in worktrees:
            if tree.exists():
                git(repo, "worktree", "remove", "--force", str(tree))


def test_native_models_have_distinct_neutral_family_terms_and_no_identity_transfer(tmp_path):
    engine = Engine(directory=tmp_path, llm="off")
    choice = engine.pick("task", native_candidates=[native(), native("host-model-b")], seed=3)
    families = {c["model"]: c["family"] for c in choice["cells"]}
    for family in families.values():
        assert engine.prior.outcome.mean[engine.prior.outcome.index(f"loud.b.{family}")] == 0
    engine.observe("task", "host-model-a", "high", "loud", id="execution")
    assert engine.posterior.outcome.mean[engine.posterior.outcome.index(f"loud.b.{families['host-model-a']}")] > 0
    assert engine.posterior.outcome.mean[engine.posterior.outcome.index(f"loud.b.{families['host-model-b']}")] == 0
    exported = engine.export_prior(tmp_path / "prior.json")
    assert not any(family in json.dumps(exported) for family in families.values())


@pytest.mark.parametrize("candidate", [native(effort=None), native(effort="model_default"), native(usd_per_mtok=-1)])
def test_invalid_native_inventory_does_not_register_or_observe(tmp_path, candidate):
    engine = Engine(directory=tmp_path, llm="off")
    with pytest.raises(HowHardError):
        engine.prepare("task", native_candidates=[candidate])
    assert not (engine.config.state_root / "native-models.json").exists()
    assert engine.raw_rows == []


def test_foreign_store_refusal_preserves_all_contents(tmp_path):
    first_dir, other_dir = tmp_path / "first", tmp_path / "other"
    first_dir.mkdir()
    other_dir.mkdir()
    first = Engine(directory=first_dir, llm="off")
    first.observe("task", "claude-opus-5-5", "high", "clean", id="execution")
    first.memory.add("Review", "Provisional lesson")
    store = first.config.state_root
    before = {p.relative_to(store): p.read_bytes() for p in store.rglob("*") if p.is_file()}
    with pytest.raises(HowHardError, match="another project"):
        Engine(directory=other_dir, state_directory=store, llm="off")
    assert before == {p.relative_to(store): p.read_bytes() for p in store.rglob("*") if p.is_file()}
    assert not (other_dir / ".howhard").exists()


def test_unclaimed_foreign_memory_database_is_not_adopted(tmp_path):
    import sqlite3
    store = tmp_path / ".howhard"
    store.mkdir()
    database = store / "memory.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
        connection.execute("INSERT INTO unrelated VALUES ('keep')")
    before = {p.name: p.read_bytes() for p in store.iterdir()}
    with pytest.raises(HowHardError, match="memory"):
        Engine(directory=tmp_path, llm="off")
    assert before == {p.name: p.read_bytes() for p in store.iterdir()}


def test_invalid_cli_inventory_does_not_create_state(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps([native(), native("host-b", effort=None)]))
    assert main(["context", "task", "--candidates", str(inventory), "--llm", "off", "--json"]) == 2
    assert "effort" in capsys.readouterr().err
    assert not (tmp_path / ".howhard").exists()


def test_native_raw_actual_identity_admission_retry_and_conflict(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("Original context")
    arguments = ["observe", "task", "actual-host-model", "ultra", "loud", "--native", "--id", "execution", "--text", "Repair rollback", "--note", "Actual execution differed from recommendation", "--llm", "off", "--json"]
    assert main(arguments) == 0
    original = json.loads(capsys.readouterr().out)["observed"]
    assert original["model"] == "actual-host-model" and original["effort"] == "ultra"
    assert "decision_id" not in original
    (tmp_path / "README.md").write_text("Later current context")
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["observed"] == original
    engine = Engine(directory=tmp_path, llm="off")
    registry = (engine.config.state_root / "native-models.json").read_bytes()
    rows = deepcopy(engine.raw_rows)
    for change in ({"id": "execution"}, {"id": "bad-measurement", "tokens": -1}):
        with pytest.raises(HowHardError):
            engine.observe("task", "different-new-model", "ultra", "loud", native=True, **change)
        assert (engine.config.state_root / "native-models.json").read_bytes() == registry
        assert engine.raw_rows == rows
    assert Engine(directory=tmp_path, llm="off").evidence("execution")["model"] == "actual-host-model"


def test_concurrent_native_inventory_admission_preserves_both_identities(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    def prepare(model):
        return Engine(directory=tmp_path, llm="off").prepare("task", native_candidates=[native(model)])
    with ThreadPoolExecutor(max_workers=2) as pool:
        requests = list(pool.map(prepare, ["host-a", "host-b"]))
    models = Engine(directory=tmp_path, llm="off").config.native_models
    assert set(models) == {"host-a", "host-b"}
    assert all(len(request["baseline_request"]["cells"]) == 1 for request in requests)


def test_git_ignores_old_store_and_preserves_explicit_config(tmp_path):
    repo, tree = tmp_path / "repo", tmp_path / "tree"
    repo.mkdir()
    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    git("init", "-q")
    (repo / "README.md").write_text("Current project")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-qm", "initial")
    old_store = repo / ".howhard"
    old_store.mkdir()
    (old_store / "outcomes.jsonl").write_bytes(b"")
    try:
        git("worktree", "add", "-qb", "work", str(tree))
        engine = Engine(directory=tree, llm="off")
        assert engine.config.state_root == repo / ".git/howhard"
        assert engine.config.project_root == tree
        assert (old_store / "outcomes.jsonl").read_bytes() == b""
        assert (repo / ".git/howhard").exists()
        (tree / "howhard.toml").write_text('on_cap_unmet = "flag-review"\n[prior.overrides.outcome.loud.a]\nmean = -1.2\nsd = 0.5\n')
        configured = Engine(directory=tree, llm="off")
        assert configured.config.state_root == tree / ".howhard"
        assert configured.config.prior["outcome"]["loud"]["a"]["mean"] == -1.2
        assert engine.config.state_root == repo / ".git/howhard"
    finally:
        if tree.exists():
            git("worktree", "remove", "--force", str(tree))


def test_writer_context_contains_original_and_all_correction_notes(tmp_path):
    from howhard.report import report
    engine = Engine(directory=tmp_path, llm="off")
    engine.observe("task", "actual-host-model", "high", "clean", native=True, id="execution",
                   note="Initial assessment", tokens=100, cost_usd=.2)
    original_key = engine.memory_learning_context("execution")["event_key"]
    engine.observe("task", "actual-host-model", "high", "loud", amends="execution", id="caught",
                   note="Later review caught a failed boundary")
    caught_key = engine.memory_learning_context("caught")["event_key"]
    engine.observe("task", "actual-host-model", "high", "clean", amends="caught", id="reviewed",
                   note="The failure report was a mistaken judgment")
    context = engine.memory_learning_context("reviewed")
    history = context["request"]["outcome_history"]
    assert [row["note"] for row in history] == ["Initial assessment", "Later review caught a failed boundary", "The failure report was a mistaken judgment"]
    assert [row["outcome"] for row in history] == ["clean", "loud", "clean"]
    assert [row["tokens"] for row in history] == [100, None, None]
    assert context["request"]["observation"]["outcome"] == "clean"
    assert context["event_key"] not in (original_key, caught_key)
    assert len(engine.rows) == 1
    assert report(engine)["worker_cost"]["known_usd"] == .2


@pytest.mark.parametrize("git_project", [False, True])
def test_initialized_project_and_state_survive_directory_move(tmp_path, git_project):
    project = tmp_path / "original"
    project.mkdir()
    (project / "README.md").write_text("Project before moving")
    if git_project:
        subprocess.run(["git", "-C", str(project), "init", "-q"], check=True)
    engine = Engine(directory=project, llm="off")
    decision = engine.pick("task", native_candidates=[native("actual-model")], seed=3)
    engine.observe("task", "actual-model", "high", "clean", native=True, id="execution")
    entry = engine.memory.add("Review", "A real retained provisional lesson")
    moved = tmp_path / "moved"
    project.rename(moved)
    restored = Engine(directory=moved, llm="off")
    assert restored.evidence("execution")["model"] == "actual-model"
    assert restored.memory.read(entry["id"])["body"] == entry["body"]
    assert restored.config.project_identity == engine.config.project_identity
    restored.observe_decision(decision["decision_id"], "clean", id="after-move")


def test_new_receipts_refuse_different_repository_with_same_project_label(tmp_path):
    projects = [tmp_path / "first", tmp_path / "other"]
    for project in projects:
        project.mkdir()
        subprocess.run(["git", "-C", str(project), "init", "-q"], check=True)
    first, other = (Engine(directory=p, llm="off") for p in projects)
    choice = first.pick("task", seed=3, native_candidates=inventory())
    path = first.config.state_root / "decisions" / f"{choice['decision_id']}.json"
    assert first.config.data["project"] == other.config.data["project"] == "local"
    with pytest.raises(HowHardError, match="identity"):
        other.observe_decision(path, "clean", id="foreign")
    assert other.raw_rows == []


@pytest.mark.parametrize("ancestor", [False, True])
def test_state_directory_cannot_replace_or_contain_capture_root(tmp_path, ancestor):
    project = tmp_path / "project"
    project.mkdir()
    (project / "README.md").write_text("Keep this project source")
    with pytest.raises(HowHardError, match="capture root"):
        Engine(directory=project, state_directory=tmp_path if ancestor else project, llm="off")
    assert sorted(p.name for p in project.iterdir()) == ["README.md"]


def test_default_state_symlink_cannot_resolve_to_capture_root(tmp_path):
    (tmp_path / "README.md").write_text("Keep this project source")
    (tmp_path / "pending.jsonl").write_bytes(b"")
    (tmp_path / ".howhard").symlink_to(tmp_path, target_is_directory=True)
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(HowHardError, match="capture root"):
        Engine(directory=tmp_path, llm="off")
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_custom_nested_state_is_excluded_from_capture_and_retrieval(tmp_path):
    (tmp_path / "README.md").write_text("Current source")
    (tmp_path / "main.py").write_text("def rollback(): pass\n")
    state = tmp_path / "custom-learning-store"
    engine = Engine(directory=tmp_path, state_directory=state, llm="off")
    first = engine.pick("task", native_candidates=[native()], seed=3)
    engine.memory.add("Lesson", "Learning state is not source context")
    (state / "lesson.md").write_text("Private learned notes")
    second = engine.pick("task", native_candidates=[native()], seed=3)
    assert first["context"] == second["context"]
    assert second["context"]["project_context"]["parameters"]["files"] == 2
    request = engine.prepare("task", native_candidates=[native()])
    for path in ("custom-learning-store/lesson.md", "state-alias/lesson.md"):
        if not (tmp_path / "state-alias").exists():
            (tmp_path / "state-alias").symlink_to(state, target_is_directory=True)
        with pytest.raises(HowHardError, match="state|excluded|outside"):
            engine.retrieve(request["request_id"], [{"tool": "project.read", "path": path}])
    (tmp_path / "README.md").write_text("Source has actually changed")
    changed = engine.pick("task", native_candidates=[native()], seed=3)
    assert changed["context"] != first["context"]
