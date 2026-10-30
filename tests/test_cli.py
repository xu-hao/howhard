from importlib.resources import files
import json
import subprocess
import sys
import pytest

from conftest import inventory
from howhard import Engine, __version__, load_config
from howhard.cli import main


def test_help_for_every_command():
    for args in (["--help"], *([name, "--help"] for name in ("pick", "summary", "observe", "flush", "context", "decide", "paths", "report"))):
        with pytest.raises(SystemExit) as exit:
            main(args)
        assert exit.value.code == 0


def test_packaged_defaults_available_through_resources():
    for name in ("default_config.toml", "semantic.toml", "context.toml"):
        assert files("howhard").joinpath("data", name).read_bytes()


def test_zero_configuration_end_to_end(tmp_path, monkeypatch, capsys, candidate_file):
    monkeypatch.chdir(tmp_path)
    assert main(["pick", "new-task", "--json", "--seed", "42", "--candidates", str(candidate_file)]) == 0
    picked = json.loads(capsys.readouterr().out)
    assert picked["risk_class"] == "standard"
    assert picked["seed"] == 42
    assert set(picked["chosen"]) == {"model", "family", "effort"}
    chosen = picked["chosen"]
    assert main(["observe", "new-task", chosen["model"], chosen["effort"], "clean", "--id", "first", "--json"]) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["observed"]["id"] == "first"
    assert observed["memory_update"]["status"] == "pending"
    assert not (tmp_path / ".howhard/outcomes.jsonl").exists()
    assert main(["flush", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ids"] == ["first"]
    assert (tmp_path / ".howhard/pending.jsonl").read_bytes() == b""
    assert main(["summary", "new-task", "--json", "--candidates", str(candidate_file)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert "chosen" not in summary and "seed" not in summary
    assert all("sampled_J" not in c for c in summary["cells"])
    assert not (tmp_path / "howhard.toml").exists()


def test_journal_override_cannot_alias_tracked_log(project):
    from howhard.validation import HowHardError
    config = project()
    log = config.path(config.data["logs"][0]["path"])
    with pytest.raises(HowHardError, match="distinct"):
        Engine(config, journal_path=log)


def test_explicit_config_required_policy_and_unknown_type_risk(tmp_path, capsys, candidate_file):
    path = tmp_path / "config.toml"
    path.write_text('project = "local"\n')
    assert main(["pick", "bookkeeping", "--config", str(path), "--candidates", str(candidate_file)]) == 2
    assert "on_cap_unmet" in capsys.readouterr().err
    path.write_text('on_cap_unmet = "flag-review"\n')
    assert main(["pick", "unknown", "--config", str(path), "--candidates", str(candidate_file)]) == 2
    assert "--risk" in capsys.readouterr().err
    assert main(["pick", "unknown", "--risk", "docs", "--config", str(path), "--json", "--candidates", str(candidate_file)]) == 0
    assert json.loads(capsys.readouterr().out)["risk_class"] == "docs"


def test_bad_seed_and_malformed_observe_are_diagnostics(tmp_path, monkeypatch, capsys, candidate_file):
    monkeypatch.chdir(tmp_path)
    assert main(["pick", "bookkeeping", "--seed", "-1", "--candidates", str(candidate_file)]) == 2
    assert "seed" in capsys.readouterr().err
    assert main(["observe", "bookkeeping", "unknown", "high", "clean"]) == 2
    assert "registry" in capsys.readouterr().err


def test_module_entrypoint_version_and_pick(tmp_path, candidate_file):
    version = subprocess.run([sys.executable, "-m", "howhard", "--version"], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert version.stdout.strip() == f"howhard {__version__}"
    picked = subprocess.run([sys.executable, "-m", "howhard", "pick", "bookkeeping", "--json", "--seed", "12", "--candidates", str(candidate_file)], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(picked.stdout)["seed"] == 12
