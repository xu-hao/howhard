"""Explicit network-enabled integration smoke for the standard skills CLI.

Run through AGCoord with a task-owned npm_config_cache. This is not part of the
ordinary Python gate and never installs into global agent directories.
"""

import os
from pathlib import Path
import signal
import subprocess
import tempfile


source = Path(__file__).resolve().parents[1]
agents = ["codex", "claude-code", "antigravity", "antigravity-cli"]
environment = {**os.environ, "DO_NOT_TRACK": "1"}
if not environment.get("npm_config_cache"):
    raise SystemExit("Set npm_config_cache to a task-owned temporary directory")


def run(project, *arguments):
    command = ["npx", "--yes", "skills@1.7.4", *arguments]
    print("$ " + " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=project, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        output, _ = process.communicate(timeout=180)
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise
    print(output, flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)
    return output


with tempfile.TemporaryDirectory(prefix="howhard-standard-skills-") as temporary:
    project = Path(temporary)
    assert "howhard" in run(project, "add", str(source), "--list")
    run(project, "add", str(source), "--skill", "howhard", "--agent", *agents, "--yes")
    installed = project / ".agents/skills/howhard"
    for original in (source / "skills/howhard").rglob("*"):
        if original.is_file():
            relative = original.relative_to(source / "skills/howhard")
            assert (installed / relative).read_bytes() == original.read_bytes()
    assert (project / ".claude/skills/howhard").is_symlink()
    for agent in agents:
        assert "howhard" in run(project, "list", "--agent", agent)
    run(project, "add", str(source), "--skill", "howhard", "--agent", *agents, "--yes")
    assert (installed / "SKILL.md").read_bytes() == (source / "skills/howhard/SKILL.md").read_bytes()
    run(project, "remove", "howhard", "--yes")
    assert not installed.exists()
    assert not (project / ".claude/skills/howhard").exists()
    assert "howhard" not in run(project, "list", "--agent", *agents)
print("Verified standard skills discovery, all four agent IDs, references, repeat and removal")
