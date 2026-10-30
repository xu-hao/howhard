"""Public-boundary regressions for task text and time-varying project state."""

from copy import deepcopy
import json
import numpy as np
import pytest

from howhard import Engine
from howhard.validation import HowHardError
from conftest import inventory, MODELS, cell
from howhard.context import capture_project, lexical, FEATURE_NAMES, FORMAT
from howhard.storage import observe


def test_text_changes_predictions_within_one_type(project):
    engine = Engine(project())
    from howhard.storage import observe
    for i in range(10):
        observe(engine.config, "Y", "claude-opus-5-5", "high", "loud", task_text="database migration transaction locking", project_context={"files": 100}, id=f"hard-{i}")
        observe(engine.config, "Y", "claude-opus-5-5", "high", "clean", task_text="spelling punctuation typo correction", project_context={"files": 100}, id=f"easy-{i}")
    engine.refresh()
    hard = cell(engine.summary("Y", task_text="database migration transaction locking", project_context={"files": 100}, native_candidates=inventory()), "opus", "high")
    easy = cell(engine.summary("Y", task_text="spelling punctuation typo correction", project_context={"files": 100}, native_candidates=inventory()), "opus", "high")
    assert hard["P_loud"]["mean"] > easy["P_loud"]["mean"] + .1


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
    engine.refresh(full=True)
    assert engine.rows[0]["project_context"] == a["context"]["project_context"]


def test_context_reproducible_and_memoized_by_actual_context(project):
    engine = Engine(project())
    args = dict(task_text="repair transaction rollback", project_context={"files": 42, "coverage": .8}, seed=17)
    a = engine.pick("Y", **args, native_candidates=inventory())
    assert a == engine.pick("Y", **args, native_candidates=inventory())
    b = engine.pick("Y", **{**args, "task_text": "repair spelling"}, native_candidates=inventory())
    assert a["decision_id"] != b["decision_id"]
    assert len(engine._summaries) == 2


@pytest.mark.parametrize("context", [{"files": -1}, {"coverage": 1.1}, {"unknown": 3}, {"files": True}])
def test_invalid_project_parameters_are_refused(project, context):
    with pytest.raises(HowHardError):
        Engine(project()).pick("Y", task_text="test", project_context=context, seed=1, native_candidates=inventory())


def test_task_text_can_reverse_family_and_effort_preferences(project):
    config = project()
    snapshot = capture_project(config, {"files": 500})
    for name in ("critical", "standard"):
        config.data["classes"][name]["epsilon"] = .99
    for i in range(24):
        # Crossed outcomes keep type/global effects balanced; text must interact
        # with the candidate to distinguish the two tasks.
        for family, effort, description, outcome in (
            ("sonnet", "high", "spelling punctuation correction", "clean"),
            ("opus", "xhigh", "spelling punctuation correction", "loud"),
            ("sonnet", "high", "distributed transaction concurrency invariants", "loud"),
            ("opus", "xhigh", "distributed transaction concurrency invariants", "clean"),
        ):
            model = MODELS[family]
            observe(config, "Y", model, effort, outcome, task_text=description, project_context=snapshot, id=f"{i}-{family}-{description[:3]}")
    engine = Engine(config)
    a = engine.summary("Y", "spelling punctuation correction", project_context=snapshot, native_candidates=inventory())
    b = engine.summary("Y", "distributed transaction concurrency invariants", project_context=snapshot, native_candidates=inventory())
    assert cell(a, "sonnet", "high")["P_clean"]["mean"] > cell(a, "opus", "xhigh")["P_clean"]["mean"]
    assert cell(b, "sonnet", "high")["P_clean"]["mean"] < cell(b, "opus", "xhigh")["P_clean"]["mean"]


def test_task_text_can_reverse_effort_within_same_family(project):
    config = project()
    snapshot = capture_project(config)
    for i in range(24):
        for effort, description, outcome in (
            ("high", "spelling punctuation correction", "clean"),
            ("xhigh", "spelling punctuation correction", "loud"),
            ("high", "distributed transaction concurrency invariants", "loud"),
            ("xhigh", "distributed transaction concurrency invariants", "clean"),
        ):
            observe(config, "Y", "claude-opus-5-5", effort, outcome, task_text=description, project_context=snapshot, id=f"{i}-{effort}-{description[:3]}")
    engine = Engine(config)
    a = engine.summary("Y", "spelling punctuation correction", project_context=snapshot, native_candidates=inventory())
    b = engine.summary("Y", "distributed transaction concurrency invariants", project_context=snapshot, native_candidates=inventory())
    assert cell(a, "opus", "high")["P_clean"]["mean"] > cell(a, "opus", "xhigh")["P_clean"]["mean"]
    assert cell(b, "opus", "high")["P_clean"]["mean"] < cell(b, "opus", "xhigh")["P_clean"]["mean"]


def test_project_parameters_change_outcomes_tokens_and_time(project):
    config = project()
    small = capture_project(config, {"files": 0, "coverage": 1.0})
    large = capture_project(config, {"files": 10000, "coverage": 0.0})
    for i in range(30):
        observe(config, "Y", "claude-opus-5-5", "high", "clean", task_text="same task", project_context=small, tokens=1000, seconds=10, id=f"small-{i}")
        observe(config, "Y", "claude-opus-5-5", "high", "loud", task_text="same task", project_context=large, tokens=100000, seconds=1000, id=f"large-{i}")
    engine = Engine(config)
    a = cell(engine.summary("Y", "same task", project_context=small, native_candidates=inventory()), "opus", "high")
    b = cell(engine.summary("Y", "same task", project_context=large, native_candidates=inventory()), "opus", "high")
    assert a["P_loud"]["mean"] + .1 < b["P_loud"]["mean"]
    assert a["C_run"] < b["C_run"] / 2
    assert a["T"] < b["T"] / 2


def test_changing_project_parameters_can_reverse_family_preference(project):
    config = project()
    small = capture_project(config, {"files": 0, "coverage": 1.0})
    large = capture_project(config, {"files": 10000, "coverage": 0.0})
    for i in range(24):
        for family, context, outcome in (("sonnet", small, "clean"), ("opus", small, "loud"), ("sonnet", large, "loud"), ("opus", large, "clean")):
            observe(config, "Y", MODELS[family], "high", outcome, task_text="same description", project_context=context, id=f"{i}-{family}-{context['parameters']['files']}")
    engine = Engine(config)
    a = engine.summary("Y", "same description", project_context=small, native_candidates=inventory())
    b = engine.summary("Y", "same description", project_context=large, native_candidates=inventory())
    assert cell(a, "sonnet", "high")["P_clean"]["mean"] > cell(a, "opus", "high")["P_clean"]["mean"]
    assert cell(b, "sonnet", "high")["P_clean"]["mean"] < cell(b, "opus", "high")["P_clean"]["mean"]


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


def test_context_fit_warm_full_order_and_hessian_agree(project):
    config = project()
    for i in range(8):
        observe(config, "Y", "claude-opus-5-5", "high", "loud" if i % 2 else "clean", task_text="database lock" if i % 2 else "fix spelling", project_context={"files": 100 + i}, id=f"row-{i}")
    engine = Engine(config)
    engine.observe("Y", "claude-opus-5-5", "high", "clean", task_text="fix spelling", project_context={"files": 100}, id="new")
    warm = deepcopy(engine.posterior)
    assert warm.statistics["kind"] == "warm"
    engine.refresh(full=True)
    for block in ("outcome", "tokens", "time"):
        np.testing.assert_allclose(getattr(warm, block).mean, getattr(engine.posterior, block).mean, atol=1e-6, rtol=0)
        np.testing.assert_allclose(getattr(warm, block).covariance, getattr(engine.posterior, block).covariance, atol=1e-6, rtol=0)
    from howhard.outcomes import fit
    shuffled = list(reversed(engine.rows))
    reordered = fit(engine.prior, shuffled)
    np.testing.assert_allclose(reordered.mean, engine.posterior.outcome.mean, atol=1e-9, rtol=0)


def test_context_transfer_excludes_text_and_snapshots(project, tmp_path):
    config = project()
    engine = Engine(config)
    engine.observe("Y", "claude-opus-5-5", "high", "loud", task_text="private-task-phrase", project_context={"files": 1234}, id="private-row-id", note="private-note")
    exported = engine.export_prior(tmp_path / "shared.json")
    assert exported["context_encoder"] == FORMAT
    assert exported["vocabularies"]["features"] == list(FEATURE_NAMES)
    text = (tmp_path / "shared.json").read_text()
    for forbidden in ("private-task-phrase", "private-row-id", "private-note", "project_context", "fingerprint", str(config.base)):
        assert forbidden not in text
    b = Engine(project())
    b.import_prior(tmp_path / "shared.json")
    assert b.prior.has_context
    # Shared contextual effects survived transfer; the new project intercept did not.
    label = "loud.x.project_known"
    assert b.posterior.outcome.mean[b.posterior.outcome.index(label)] == engine.posterior.outcome.mean[engine.posterior.outcome.index(label)]
    assert b.posterior.outcome.mean[b.posterior.outcome.index("loud.pi.local")] == 0


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


def test_hashing_and_receipt_reproducible_across_processes(project):
    import os
    import subprocess
    import sys
    config = project()
    # A path-stable snapshot avoids inspecting process-local temporary artifacts.
    script = 'from howhard import Engine; import json; e=Engine("howhard.toml"); print(json.dumps(e.pick("Y", "Unicode CAFÉ rollback transaction", seed=42, native_candidates=[{"model":"claude-opus-5-5","effort":"high"}]), sort_keys=True))'
    outputs = []
    for seed in ("1", "999"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outputs.append(json.loads(subprocess.check_output([sys.executable, "-c", script], cwd=config.base, env=env, text=True)))
    # Cache statistics report the work of each call, independently of predictions.
    outputs[0].pop("fit")
    outputs[1].pop("fit")
    assert outputs[0] == outputs[1]


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
    exported = engine.export_prior(tmp_path / "shared.json")
    exported["context_encoder"] = "unknown-encoder"
    (tmp_path / "shared.json").write_text(json.dumps(exported))
    with pytest.raises(HowHardError, match="context_encoder"):
        engine.import_prior(tmp_path / "shared.json")


def test_marginal_fit_equals_full_joint_newton_with_correlated_inactive_terms():
    from types import SimpleNamespace
    from howhard.gaussian import Gaussian
    from howhard.outcomes import fit, objective
    from howhard.prior import cell_labels, design
    rng = np.random.default_rng(29)
    cell_tuple = ("task", "family", "high", "project")
    labels = sorted({label for group in cell_labels("outcome", *cell_tuple) for label in group})
    labels += ["loud.unused", "silent.unused"]
    a = rng.normal(size=(len(labels), len(labels)))
    covariance = (a @ a.T) / len(labels) + np.eye(len(labels))
    mean = rng.normal(size=len(labels))
    prior = SimpleNamespace(outcome=Gaussian(tuple(labels), mean, covariance), has_context=False)
    rows = [{"task_type": "task", "family": "family", "resolved_effort": "high", "project": "project", "weight": 1.0, "outcome": outcome}
            for outcome in ("clean", "clean", "loud", "silent")]
    actual = fit(prior, rows)
    x = design(prior.outcome, "outcome", [cell_tuple])
    counts = np.array([[2., 1., 1.]])
    precision = np.linalg.inv(covariance)
    beta = mean.copy()
    for _ in range(50):
        _, gradient, hessian = objective(beta, mean, precision, x, counts)
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if max(abs(step)) < 1e-11:
            break
    np.testing.assert_allclose(actual.mean, beta, atol=1e-9, rtol=0)
    hessian = objective(beta, mean, precision, x, counts)[2]
    np.testing.assert_allclose(actual.covariance, np.linalg.inv(hessian), atol=1e-9, rtol=0)
