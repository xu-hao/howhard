"""Copy the packaged howhard skill into explicit host discovery locations."""

from importlib.resources import files
import json
from pathlib import Path

from . import __version__
from .validation import HowHardError

HOSTS = ("codex", "claude-code", "antigravity")
PROJECT_PATHS = {
    "codex": (".agents/skills/howhard",),
    "claude-code": (".claude/skills/howhard",),
    "antigravity": (".agents/skills/howhard",),
}
USER_PATHS = {
    "codex": (".agents/skills/howhard",),
    "claude-code": (".claude/skills/howhard",),
    "antigravity": (".gemini/config/skills/howhard", ".gemini/antigravity-cli/skills/howhard"),
}


def _bundle():
    result = {}

    def read(directory, prefix=""):
        for entry in directory.iterdir():
            name = prefix + entry.name
            if entry.is_dir():
                read(entry, name + "/")
            else:
                result[name] = entry.read_bytes()

    read(files("howhard").joinpath("skills/howhard"))
    result["howhard-install.json"] = (json.dumps({"skill": "howhard", "version": __version__}, sort_keys=True) + "\n").encode()
    return result


def _matching(target, bundle):
    if not target.is_dir():
        return False
    actual, directories = {}, set()
    for entry in target.rglob("*"):
        if entry.is_symlink():
            return False
        relative = entry.relative_to(target).as_posix()
        if entry.is_dir():
            directories.add(relative)
        elif entry.is_file():
            actual[relative] = entry.read_bytes()
        else:
            return False
    expected_directories = {parent.as_posix() for name in bundle for parent in Path(name).parents if parent != Path(".")}
    return actual == bundle and directories == expected_directories


def install_skills(host, *, scope="project", directory=None):
    """Install copied resources, accepting only byte-identical existing bundles.

    Preflight every destination before writing any of them. Existing conflicting
    content and symlinked discovery paths are never replaced.
    """
    if host not in (*HOSTS, "all"):
        raise HowHardError("skills.host: expected codex, claude-code, antigravity or all")
    if scope not in ("project", "user"):
        raise HowHardError("skills.scope: expected project or user")
    if scope == "user" and directory is not None:
        raise HowHardError("skills.directory: only applies to project scope")
    base = (Path.home() if scope == "user" else Path(directory or Path.cwd()).expanduser()).resolve()
    if not base.is_dir():
        raise HowHardError(f"skills.directory: expected an existing directory: {base}")
    destinations = {}
    paths = USER_PATHS if scope == "user" else PROJECT_PATHS
    for selected in HOSTS if host == "all" else (host,):
        for relative in paths[selected]:
            destinations.setdefault(base / relative, []).append(selected)
    bundle = _bundle()
    locations = []
    for target, hosts in destinations.items():
        for entry in (target, *target.parents):
            if entry == base:
                break
            if entry.is_symlink() or (entry.exists() and not entry.is_dir()):
                raise HowHardError(f"skills: conflicting path {entry}; preserve it and choose another location")
        exists = target.exists()
        if exists and not _matching(target, bundle):
            raise HowHardError(f"skills: conflicting content at {target}; archive it outside skill discovery before installing")
        locations.append({"path": str(target), "hosts": hosts, "status": "unchanged" if exists else "installed"})
    for location in locations:
        if location["status"] == "unchanged":
            continue
        target = Path(location["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir()  # Exclusive creation refuses a destination added after preflight.
        for name, data in bundle.items():
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(data)
    return {"skill": "howhard", "version": __version__, "host": host, "scope": scope, "locations": locations}
