"""The root AGENTS.snippet.md at its packaging boundaries (#164).

Builds and renders run on disposable copies under tmp_path, with no network, and doubles
stand in for the native plugin clients.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from xml.etree import ElementTree

import pytest

ROOT = Path(__file__).resolve().parents[1]
SNIPPET = "AGENTS.snippet.md"
SDIST_INPUTS = ("pyproject.toml", "MANIFEST.in", "README.md", SNIPPET, "docs/skills.md")


def isolated_environment(tmp_path, **values):
    """An environment whose home and temporary files stay under tmp_path."""
    home, temporary = tmp_path / "home", tmp_path / "tmp"
    home.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)
    environment = {key: value for key, value in os.environ.items()
                   if key != "PYTHONPATH" and not key.startswith("XDG_")}
    return {**environment, "HOME": str(home), "TMPDIR": str(temporary), **values}


def run(tmp_path, *command, **values):
    result = subprocess.run(command, cwd=tmp_path, env=isolated_environment(tmp_path, **values),
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def checkout_copy(tmp_path):
    """The packaging inputs alone, so that a build writes nothing into the checkout."""
    source = tmp_path / "source"
    shutil.copytree(ROOT / "src/howhard", source / "src/howhard",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for name in SDIST_INPUTS:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, source / name)
    return source


def unpacked_sdist(tmp_path, source):
    output, unpacked = tmp_path / "dist", tmp_path / "unpacked"
    run(tmp_path, sys.executable, "-m", "build", "--sdist", "--no-isolation",
        "--outdir", str(output), str(source))
    [sdist] = output.glob("howhard-*.tar.gz")
    with tarfile.open(sdist) as archive:
        archive.extractall(unpacked, filter="data")
    return unpacked / sdist.name.removesuffix(".tar.gz")


def test_source_distribution_ships_the_snippet(tmp_path):
    distribution = unpacked_sdist(tmp_path, checkout_copy(tmp_path))
    assert (distribution / SNIPPET).read_bytes() == (ROOT / SNIPPET).read_bytes()


def test_skills_page_from_the_source_distribution_renders_the_one_snippet(tmp_path):
    pytest.importorskip("sphinx")
    pytest.importorskip("myst_parser")
    source = checkout_copy(tmp_path)
    # A line that exists only in this copy: the page can show it only by reading the file.
    edited = (source / SNIPPET).read_text(encoding="utf-8") + "- Line added by this test.\n"
    (source / SNIPPET).write_text(edited, encoding="utf-8")
    distribution = unpacked_sdist(tmp_path, source)
    configuration, rendered = tmp_path / "configuration", tmp_path / "rendered"
    configuration.mkdir()
    (configuration / "conf.py").write_text(
        'extensions = ["myst_parser"]\nsource_suffix = {".md": "markdown"}\nmaster_doc = "skills"\n')
    run(tmp_path, sys.executable, "-m", "sphinx", "-b", "xml", "-c", str(configuration),
        str(distribution / "docs"), str(rendered))
    blocks = [block.text for block in ElementTree.parse(rendered / "skills.xml").iter("literal_block")
              if Path(block.get("source", "")).name == SNIPPET]
    assert [block.rstrip("\n") for block in blocks] == [edited.rstrip("\n")]


def test_strict_antigravity_payload_ships_the_snippet(tmp_path):
    clients, recorded = tmp_path / "bin", tmp_path / "strict-payload.json"
    clients.mkdir()
    # Both doubles accept every validation; the strict JSON one records the payload it saw.
    double = f"""#!{sys.executable}
import json, sys
from pathlib import Path
if "--strict" in sys.argv and "--json" in sys.argv:
    payload = Path(sys.argv[3])
    Path({str(recorded)!r}).write_text(json.dumps({{
        path.relative_to(payload).as_posix(): path.read_text(encoding="utf-8")
        for path in payload.rglob("*") if path.is_file()}}))
print(json.dumps({{"findings": [], "components": {{"skills": 2}}}}))
"""
    for name in ("claude", "agy"):
        (clients / name).write_text(double)
        (clients / name).chmod(0o755)
    run(tmp_path, sys.executable, str(ROOT / "scripts/check-plugins.py"), PATH=str(clients))
    payload = json.loads(recorded.read_text())
    assert payload[SNIPPET] == (ROOT / SNIPPET).read_text(encoding="utf-8")
    assert {".antigravity-plugin/plugin.json", "skills/howhard/SKILL.md",
            "skills/attribute-failure/SKILL.md"} <= set(payload)
    # The payload is the host's own: contributor rules and the portable manifest stay out.
    assert not {"AGENTS.md", "plugin.json"} & set(payload)
