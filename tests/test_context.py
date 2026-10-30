"""Public-boundary regressions for task text and time-varying project state."""

from copy import deepcopy
import json
import pytest

from howhard import Engine
from howhard.validation import HowHardError
from conftest import inventory, MODELS, cell
from howhard.context import capture_project, FORMAT
from howhard.storage import observe


def test_changed_project_is_recaptured_and_old_rows_are_unchanged(tmp_path):
    (tmp_path / "README.md").write_text("small isolated library")
    engine = Engine(directory=tmp_path)
    a = engine.pick("bookkeeping", task_text="update the documentation", seed=42, native_candidates=inventory())
    (tmp_path / "README.md").write_text("large distributed system with queues transactions and concurrency")
    (tmp_path / "new.py").write_text("def public_api(): pass\n")
    b = engine.pick("bookkeeping", task_text="update the documentation", seed=42, native_candidates=inventory())
    assert a["context"]["project_context"]["fingerprint"] != b["context"]["project_context"]["fingerprint"]
    chosen = a["chosen"]
    observed = engine.observe("bookkeeping", chosen["model"], chosen["effort"], "clean", decision=a["decision_id"], id="original")
    assert observed["project_context"] == a["context"]["project_context"]
    assert observed["task_text"] == "update the documentation"
    engine.observe("bookkeeping", chosen["model"], chosen["effort"], "silent", amends="original", id="amend")
    assert engine.rows[0]["project_context"] == a["context"]["project_context"]
    engine.refresh()
    assert engine.rows[0]["project_context"] == a["context"]["project_context"]


def test_context_reproducible_and_memoized_by_actual_context(project):
    engine = Engine(project())
    args = dict(task_text="repair transaction rollback", project_context={"files": 42, "coverage": .8}, seed=17)
    a = engine.pick("Y", **args, native_candidates=inventory())
    assert a["decision_id"] == engine.pick("Y", **args, native_candidates=inventory())["decision_id"]
    b = engine.pick("Y", **{**args, "task_text": "repair spelling"}, native_candidates=inventory())
    assert a["decision_id"] != b["decision_id"]
    assert len(engine._summaries) == 2


@pytest.mark.parametrize("context", [{"files": -1}, {"coverage": 1.1}, {"unknown": 3}, {"files": True}])
def test_invalid_project_parameters_are_refused(project, context):
    with pytest.raises(HowHardError):
        Engine(project()).pick("Y", task_text="test", project_context=context, seed=1, native_candidates=inventory())


def test_snapshot_sees_staged_changes_and_respects_gitignore(tmp_path):
    import subprocess
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / ".gitignore").write_text("ignored/\n.howhard/\n")
    source = tmp_path / "main.py"
    source.write_text("def spelling(): pass\n")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-qm", "initial")
    from howhard import load_config
    config = load_config(directory=tmp_path)
    before = capture_project(config)
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored/secret.py").write_text("not part of the project")
    assert capture_project(config) == before
    source.write_text("def transaction_locking(): pass\n")
    git("add", "main.py")
    after = capture_project(config)
    assert after["parameters"]["changed_files"] == 1
    assert after["fingerprint"] != before["fingerprint"]
    assert after["text"] != before["text"]


def test_virtual_environment_does_not_change_project_snapshot(tmp_path):
    from howhard import load_config
    (tmp_path / "README.md").write_text("small library")
    config = load_config(directory=tmp_path)
    before = capture_project(config)
    (tmp_path / "venv/lib").mkdir(parents=True)
    (tmp_path / "venv/lib/dependency.py").write_text("third party code")
    assert capture_project(config) == before


def test_receipt_cli_round_trip_after_project_change(tmp_path, monkeypatch, capsys, candidate_file):
    from howhard.cli import main
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("original description")
    (tmp_path / "task.txt").write_text("repair the transaction rollback")
    assert main(["pick", "specified-implementation", "--text-file", "task.txt", "--save-decision", "receipt.json", "--json", "--seed", "42", "--candidates", str(candidate_file)]) == 0
    choice = json.loads(capsys.readouterr().out)
    (tmp_path / "README.md").write_text("completely different repository now")
    assert main(["observe", "receipt.json", "clean", "--json", "--id", "from-receipt"]) == 0
    recorded = json.loads(capsys.readouterr().out)["observed"]
    assert recorded["task_text"] == choice["context"]["task_text"]
    assert recorded["project_context"] == choice["context"]["project_context"]
    assert recorded["decision_id"] == choice["decision_id"]


def test_receipt_mismatch_and_partial_row_context_refused(project):
    config = project()
    engine = Engine(config)
    picked = engine.pick("Y", "fix spelling", seed=3, native_candidates=inventory())
    with pytest.raises(HowHardError, match="match"):
        engine.observe("X", picked["chosen"]["model"], picked["chosen"]["effort"], "clean", decision=picked)
    log = config.path(config.data["logs"][0]["path"])
    from conftest import row
    log.write_text(json.dumps({k: v for k, v in row(1).items() if k != "context_encoder"}) + "\n")
    with pytest.raises(HowHardError, match="occur together"):
        Engine(config)


def test_context_encoder_mismatch_and_corrupt_receipt_refused(project, tmp_path):
    from howhard.decisions import receipt_path
    engine = Engine(project())
    choice = engine.pick("Y", "private task", seed=2, native_candidates=inventory())
    receipt = receipt_path(engine.config, choice["decision_id"])
    data = json.loads(receipt.read_bytes())
    data["context"]["task_text"] = "tampered"
    receipt.write_text(json.dumps(data))
    with pytest.raises(HowHardError, match="match its ID"):
        engine.observe_decision(choice["decision_id"], "clean")
    engine.observe("Y", "claude-opus-5-5", "high", "clean", task_text="task", project_context={"files": 1})
