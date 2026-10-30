"""Public workflow checks; these fixtures do not measure routing quality."""

import json
from pathlib import Path

import pytest

from howhard.cli import main


def test_retired_skill_copier_is_rejected_before_project_state(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOWHARD_LLM", "unavailable-backend")
    (tmp_path / "howhard.toml").write_text("invalid config")
    with pytest.raises(SystemExit) as error:
        main(["skills", "install", "--host", "codex"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["howhard.toml"]


def test_canonical_native_example_preserves_supplied_identity(tmp_path, monkeypatch, capsys):
    from howhard import Engine
    monkeypatch.chdir(tmp_path)
    (tmp_path / "task.txt").write_text("Verify rollback at the public boundary")
    (tmp_path / "host-candidates.json").write_text(json.dumps([
        {"model": "actual-native-model", "effort": "ultra"},
    ]))
    workflow = (Path(__file__).resolve().parents[1] / "skills/howhard/references/workflow.md").read_text()
    example = workflow.split("```python\n", 1)[1].split("```", 1)[0]
    exec(compile(example, "canonical-workflow-example", "exec"), {})
    decision = json.loads(capsys.readouterr().out)
    assert decision["chosen"]["model"] == "actual-native-model"
    assert decision["chosen"]["effort"] == "ultra"
    assert decision["cells"][0]["C_run"] is None
    assert Engine(directory=tmp_path).raw_rows == []
