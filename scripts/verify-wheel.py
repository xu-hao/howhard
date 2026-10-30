"""Check built artifacts at their packaging boundary, including the version source."""

from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile

expected = version("howhard")
wheel = Path("dist") / f"howhard-{expected}-py3-none-any.whl"
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    for resource in ("default_prior.toml", "default_config.toml", "context.toml", "semantic.toml"):
        assert f"howhard/data/{resource}" in names, resource
    skill_names = ["SKILL.md", "references/hosts.md", "references/workflow.md"]
    skill_bundle = {name: archive.read(f"howhard/skills/howhard/{name}") for name in skill_names}
    metadata = archive.read(f"howhard-{expected}.dist-info/METADATA").decode()
    assert f"Version: {expected}\n" in metadata
    entrypoints = archive.read(f"howhard-{expected}.dist-info/entry_points.txt").decode()
    assert "howhard = howhard.cli:main" in entrypoints
sdist = Path("dist") / f"howhard-{expected}.tar.gz"
with tarfile.open(sdist) as archive:
    assert f"howhard-{expected}/tests/conftest.py" in archive.getnames()

# Exercise an installed wheel outside the source checkout. No editable install or
# inherited PYTHONPATH can supply its resources or entry point.
with tempfile.TemporaryDirectory(prefix="howhard-wheel-") as directory:
    root = Path(directory)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["HOWHARD_LLM"] = "off"  # Packaging checks never spend inference tokens.
    subprocess.run([sys.executable, "-m", "venv", str(root / "venv")], check=True)
    python = root / "venv/bin/python"
    subprocess.run([str(python), "-m", "pip", "install", "--quiet", str(wheel.resolve())],
                   cwd=root, env=environment, check=True)
    code = 'from importlib.resources import files; import howhard; assert files("howhard").joinpath("data/default_prior.toml").is_file(); print(howhard.__version__)'
    installed_version = subprocess.check_output([str(python), "-c", code], cwd=root, env=environment, text=True).strip()
    assert installed_version == expected
    command = root / "venv/bin/howhard"
    assert subprocess.check_output([str(command), "--version"], cwd=root, env=environment, text=True).strip() == f"howhard {expected}"
    picked = json.loads(subprocess.check_output([str(command), "pick", "bookkeeping", "--text", "repair spelling", "--seed", "42", "--json"], cwd=root, env=environment, text=True))
    assert picked["seed"] == 42 and picked["chosen"]["model"]
    observed = json.loads(subprocess.check_output([str(command), "observe", picked["decision_id"], "clean", "--json"], cwd=root, env=environment, text=True))
    assert observed["observed"]["task_text"] == "repair spelling"
    assert observed["observed"]["project_context"] == picked["context"]["project_context"]
    entry = json.loads(subprocess.check_output([str(command), "memory", "add", "--title", "Installed memory", "--body", "An editable semantic memory", "--json"], cwd=root, env=environment, text=True))
    removed = json.loads(subprocess.check_output([str(command), "memory", "remove", entry["id"], "--json"], cwd=root, env=environment, text=True))
    assert removed["active"] is False
    project = root / "skill-project"
    project.mkdir()
    (project / "howhard.toml").write_text("invalid config")
    install_environment = {**environment, "HOWHARD_LLM": "unavailable-backend"}
    installed = json.loads(subprocess.check_output(
        [str(command), "skills", "install", "--host", "all", "--directory", str(project), "--json"],
        cwd=root, env=install_environment, text=True))
    assert installed["version"] == expected and len(installed["locations"]) == 2
    for location in installed["locations"]:
        assert location["status"] == "installed"
        target = Path(location["path"])
        for name, data in skill_bundle.items():
            assert (target / name).read_bytes() == data
    repeated = json.loads(subprocess.check_output(
        [str(command), "skills", "install", "--host", "all", "--directory", str(project), "--json"],
        cwd=root, env=install_environment, text=True))
    assert all(location["status"] == "unchanged" for location in repeated["locations"])
    assert not (project / ".howhard").exists()
    # Inject a temporary home in-process: never touch the user's host installations.
    home = root / "skill-home"
    home.mkdir()
    user_code = 'from pathlib import Path; import sys; from howhard.cli import main; Path.home = classmethod(lambda cls: Path(sys.argv[1])); raise SystemExit(main(["skills", "install", "--host", "all", "--scope", "user", "--json"]))'
    user_install = json.loads(subprocess.check_output([str(python), "-c", user_code, str(home)],
                                                    cwd=root, env=install_environment, text=True))
    assert len(user_install["locations"]) == 4
    for location in user_install["locations"]:
        assert Path(location["path"]).is_relative_to(home)
        assert (Path(location["path"]) / "SKILL.md").read_bytes() == skill_bundle["SKILL.md"]
print(f"Verified howhard {expected} source distribution and clean installed wheel")
