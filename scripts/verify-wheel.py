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
    assert not any(name.startswith("howhard/skills/") for name in names)
    metadata = archive.read(f"howhard-{expected}.dist-info/METADATA").decode()
    assert f"Version: {expected}\n" in metadata
    entrypoints = archive.read(f"howhard-{expected}.dist-info/entry_points.txt").decode()
    assert "howhard = howhard.cli:main" in entrypoints
sdist = Path("dist") / f"howhard-{expected}.tar.gz"
with tarfile.open(sdist) as archive:
    assert f"howhard-{expected}/tests/conftest.py" in archive.getnames()
    for name in ("SKILL.md", "references/hosts.md", "references/workflow.md"):
        assert f"howhard-{expected}/skills/howhard/{name}" in archive.getnames()

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
print(f"Verified howhard {expected} source distribution and clean installed wheel")
