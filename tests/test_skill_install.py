"""Offline public CLI checks in isolated projects and homes; no host calls."""

from importlib.resources import files
import json
from pathlib import Path

import pytest

from howhard import __version__
from howhard.cli import main


@pytest.mark.parametrize("scope,host,relative", [
    ("project", "codex", [".agents/skills/howhard"]),
    ("project", "claude-code", [".claude/skills/howhard"]),
    ("project", "antigravity", [".agents/skills/howhard"]),
    ("project", "all", [".agents/skills/howhard", ".claude/skills/howhard"]),
    ("user", "codex", [".agents/skills/howhard"]),
    ("user", "claude-code", [".claude/skills/howhard"]),
    ("user", "antigravity", [".gemini/config/skills/howhard", ".gemini/antigravity-cli/skills/howhard"]),
    ("user", "all", [".agents/skills/howhard", ".claude/skills/howhard", ".gemini/config/skills/howhard", ".gemini/antigravity-cli/skills/howhard"]),
])
def test_install_hosts_scopes_copy_complete_bundle_and_repeat(tmp_path, monkeypatch, capsys, scope, host, relative):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOWHARD_LLM", "unavailable-backend")
    monkeypatch.chdir(project)
    # Installation does not parse project settings or initialize learning state.
    (project / "howhard.toml").write_text("invalid config")
    arguments = ["skills", "install", "--host", host, "--scope", scope, "--json"]
    base = home if scope == "user" else project
    expected = [str(base / location) for location in relative]
    assert main(arguments) == 0
    installed = json.loads(capsys.readouterr().out)
    assert installed["version"] == __version__ and installed["scope"] == scope
    assert [entry["path"] for entry in installed["locations"]] == expected
    assert all(entry["status"] == "installed" for entry in installed["locations"])
    source = files("howhard").joinpath("skills/howhard")
    for target in map(Path, expected):
        assert not target.is_symlink()
        assert (target / "SKILL.md").read_bytes() == source.joinpath("SKILL.md").read_bytes()
        for reference in source.joinpath("references").iterdir():
            assert (target / "references" / reference.name).read_bytes() == reference.read_bytes()
        assert json.loads((target / "howhard-install.json").read_text())["version"] == __version__
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in base.rglob("*") if p.is_file()}
    assert main(arguments) == 0
    assert all(entry["status"] == "unchanged" for entry in json.loads(capsys.readouterr().out)["locations"])
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in base.rglob("*") if p.is_file()}
    assert not (project / ".howhard").exists()
    if scope == "project" and host == "all":
        assert installed["locations"][0]["hosts"] == ["codex", "antigravity"]


def test_project_directory_target_and_unrelated_siblings_are_preserved(tmp_path, monkeypatch, capsys):
    elsewhere, project = tmp_path / "elsewhere", tmp_path / "plain directory"
    elsewhere.mkdir()
    project.mkdir()
    sibling = project / ".agents/skills/other/SKILL.md"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("another skill")
    instructions = project / ".agents/AGENTS.md"
    instructions.symlink_to("../AGENTS.md")
    (project / "AGENTS.md").write_text("project instructions")
    monkeypatch.chdir(elsewhere)
    assert main(["skills", "install", "--host", "codex", "--directory", str(project), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["locations"][0]["path"] == str(project / ".agents/skills/howhard")
    assert sibling.read_text() == "another skill"
    assert instructions.is_symlink() and instructions.read_text() == "project instructions"
    assert not (elsewhere / ".agents").exists()


@pytest.mark.parametrize("conflict", ["file", "empty", "content", "extra", "symlink", "ancestor-symlink", "nested-symlink"])
def test_conflicts_refuse_all_destinations_and_preserve_content(tmp_path, monkeypatch, capsys, conflict):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / ".claude/skills/howhard"
    target.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("untouched")
    if conflict == "file":
        target.write_text("unrelated file")
    elif conflict == "symlink":
        target.symlink_to(outside, target_is_directory=True)
    elif conflict == "ancestor-symlink":
        target.parent.rmdir()
        target.parent.symlink_to(outside, target_is_directory=True)
    elif conflict == "extra":
        assert main(["skills", "install", "--host", "claude-code", "--json"]) == 0
        capsys.readouterr()
        (target / "custom.md").write_text("local edit")
    else:
        target.mkdir()
        if conflict == "content":
            (target / "SKILL.md").write_text("local edit")
        elif conflict == "nested-symlink":
            (target / "references").symlink_to(outside, target_is_directory=True)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file() and not p.is_symlink()}
    assert main(["skills", "install", "--host", "all", "--json"]) == 2
    assert "conflicting" in capsys.readouterr().err
    assert not (tmp_path / ".agents").exists()
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file() and not p.is_symlink()}


def test_invalid_directory_scope_arguments_do_not_install(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    for arguments in (["--directory", str(tmp_path / "missing")], ["--scope", "user", "--directory", str(tmp_path)]):
        assert main(["skills", "install", "--host", "codex", *arguments]) == 2
        assert "skills.directory" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_packaged_candidate_example_executes_only_supplied_cells(tmp_path, monkeypatch, capsys):
    from howhard import Engine
    monkeypatch.chdir(tmp_path)
    (tmp_path / "task.txt").write_text("Verify rollback at the public boundary")
    (tmp_path / "host-cells.json").write_text(json.dumps([
        {"model": "claude-opus-5-5", "effort": "high"},
        {"model": "unregistered-host-model", "effort": "high"},
    ]))
    workflow = files("howhard").joinpath("skills/howhard/references/workflow.md").read_text()
    example = workflow.split("```python\n", 1)[1].split("```", 1)[0]
    exec(compile(example, "packaged-workflow-example", "exec"), {})
    decision = json.loads(capsys.readouterr().out)
    assert decision["chosen"]["model"] == "claude-opus-5-5"
    assert decision["chosen"]["effort"] == "high"
    assert Engine(directory=tmp_path).raw_rows == []  # A recommendation is not execution evidence.
    (tmp_path / "host-cells.json").write_text('[{"model": "unregistered-host-model", "effort": "high"}]')
    with pytest.raises(SystemExit, match="Capability gap"):
        exec(compile(example, "packaged-workflow-example", "exec"), {})
