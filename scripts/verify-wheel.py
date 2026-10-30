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
    for resource in ("default_config.toml", "context.toml", "semantic.toml"):
        assert f"howhard/data/{resource}" in names, resource
    assert not any(name.startswith("howhard/skills/") for name in names)
    for retired in ("cache", "outcomes", "prior", "regression", "gaussian", "transfer"):
        assert f"howhard/{retired}.py" not in names
    assert "howhard/data/default_prior.toml" not in names
    metadata = archive.read(f"howhard-{expected}.dist-info/METADATA").decode()
    assert f"Version: {expected}\n" in metadata
    entrypoints = archive.read(f"howhard-{expected}.dist-info/entry_points.txt").decode()
    assert "howhard = howhard.cli:main" in entrypoints
sdist = Path("dist") / f"howhard-{expected}.tar.gz"
with tarfile.open(sdist) as archive:
    assert f"howhard-{expected}/tests/conftest.py" in archive.getnames()
    for name in ("SKILL.md", "references/hosts.md", "references/workflow.md"):
        assert f"howhard-{expected}/skills/howhard/{name}" in archive.getnames()
    for name in ("plugin.json", ".claude-plugin/plugin.json", ".antigravity-plugin/plugin.json"):
        manifest = json.load(archive.extractfile(f"howhard-{expected}/{name}"))
        assert manifest["name"] == "howhard" and manifest["version"] == expected
    for name in (".agents/plugins/marketplace.json", ".claude-plugin/marketplace.json"):
        marketplace = json.load(archive.extractfile(f"howhard-{expected}/{name}"))
        assert marketplace["name"] == "howhard"
        assert [plugin["name"] for plugin in marketplace["plugins"]] == ["howhard"]

# Exercise an installed wheel outside the source checkout. No editable install or
# inherited PYTHONPATH can supply its resources or entry point.
with tempfile.TemporaryDirectory(prefix="howhard-wheel-") as directory:
    root = Path(directory)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("HOWHARD_LLM", None)
    environment.pop("HOWHARD_LLM_COMMAND", None)
    subprocess.run([sys.executable, "-m", "venv", str(root / "venv")], check=True)
    python = root / "venv/bin/python"
    subprocess.run([str(python), "-m", "pip", "install", "--quiet", str(wheel.resolve())],
                   cwd=root, env=environment, check=True)
    code = 'from importlib.resources import files; import howhard; assert files("howhard").joinpath("data/semantic.toml").is_file(); print(howhard.__version__)'
    installed_version = subprocess.check_output([str(python), "-c", code], cwd=root, env=environment, text=True).strip()
    assert installed_version == expected
    command = root / "venv/bin/howhard"
    assert subprocess.check_output([str(command), "--version"], cwd=root, env=environment, text=True).strip() == f"howhard {expected}"
    inventory = root / "candidates.json"
    inventory.write_text(json.dumps([{"model": "wheel-native-model", "effort": "ultra"}]))
    prepared = json.loads(subprocess.check_output([str(command), "context", "bookkeeping", "--candidates", str(inventory), "--text", "repair spelling", "--json"], cwd=root, env=environment, text=True))
    forecasts = [{"family": c["family"], "effort": c["effort"], "clean": .9, "loud": .09, "silent": .01, "tokens": 1000, "seconds": 10} for c in prepared["baseline_request"]["cells"]]
    response = root / "forecasts.json"
    response.write_text(json.dumps({"forecasts": forecasts, "memory_forecasts": [], "reason": "installed protocol fixture"}))
    picked = json.loads(subprocess.check_output([str(command), "decide", prepared["request_id"], "--file", str(response), "--seed", "42", "--json"], cwd=root, env=environment, text=True))
    assert picked["seed"] == 42 and picked["chosen"]["model"] == "wheel-native-model"
    assert set(picked["semantic"]["worlds"]) == {"llm"}
    assert not (root / ".howhard/posterior.npz").exists()
    observed = json.loads(subprocess.check_output([str(command), "observe", picked["decision_id"], "clean", "--json"], cwd=root, env=environment, text=True))
    assert observed["observed"]["task_text"] == "repair spelling"
    assert observed["observed"]["project_context"] == picked["context"]["project_context"]
    entry = json.loads(subprocess.check_output([str(command), "memory", "add", "--title", "Installed memory", "--body", "An editable semantic memory", "--json"], cwd=root, env=environment, text=True))
    removed = json.loads(subprocess.check_output([str(command), "memory", "remove", entry["id"], "--json"], cwd=root, env=environment, text=True))
    assert removed["active"] is False
    native_choice = picked
    observation = [str(command), "observe", native_choice["decision_id"], "clean", "--id", "wheel-execution", "--json"]
    first = json.loads(subprocess.check_output(observation, cwd=root, env=environment, text=True))["observed"]
    assert json.loads(subprocess.check_output(observation, cwd=root, env=environment, text=True))["observed"] == first
    evidence = json.loads(subprocess.check_output([str(command), "evidence", "wheel-execution", "--json"], cwd=root, env=environment, text=True))
    assert evidence["model"] == "wheel-native-model"
print(f"Verified howhard {expected} source distribution and clean installed wheel")
