"""Project capture and durable project state have independent roots."""

import json
import fcntl
from contextlib import contextmanager
import re
import secrets
from pathlib import Path
import subprocess

from .validation import HowHardError


def git_path(root, option):
    try:
        result = subprocess.run(["git", "--no-optional-locks", "-C", str(root), "rev-parse", option],
                                capture_output=True, text=True, timeout=5)
        return (root / result.stdout.strip()).resolve() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def validate_store(path):
    if not path.exists():
        return
    if not path.is_dir():
        raise HowHardError(f"state: expected a directory at {path}")
    names = {entry.name for entry in path.iterdir()}
    known = {"pending.jsonl", "pending.lock", "outcomes.jsonl", "posterior.npz", "memory.sqlite3",
             "decisions", "requests", "llm-cache", "inference.jsonl", "decision-events.jsonl",
             "imported-prior.json", "native-models.json", "store.json", "project.json", "project.lock"}
    if names and not names & known:
        raise HowHardError(f"state: refusing unrelated existing directory {path}")
    if "memory.sqlite3" in names and "store.json" not in names:
        raise HowHardError(f"state: refusing unclaimed memory database at {path}")


def read_identity(path):
    if not path.exists():
        return ""
    try:
        value = json.loads(path.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise HowHardError(f"state: invalid project identity at {path}") from exc
    if not isinstance(value, dict) or set(value) != {"format", "id"} or value["format"] != "howhard-project-v1" or not isinstance(value["id"], str) or not re.fullmatch(r"[0-9a-f]{32}", value["id"]):
        raise HowHardError(f"state: invalid project identity at {path}")
    return value["id"]


def roots(project, *, explicit=False, state_directory=None):
    project = Path(project).resolve()
    common = git_path(project, "--git-common-dir")
    if not explicit:
        checkout = git_path(project, "--show-toplevel")
        if checkout:
            project = checkout
    identity_path = common / "howhard-project.json" if common else project / ".howhard/project.json"
    if state_directory is not None:
        store = Path(state_directory).expanduser().resolve()
    elif explicit:
        store = project / ".howhard"
    else:
        store = common / "howhard" if common else project / ".howhard"
    store = store.resolve()
    if project.is_relative_to(store):
        raise HowHardError("state: directory must not equal or contain the project capture root")
    if (not explicit or state_directory is not None) and store.exists():
        validate_store(store)
    return project, store, identity_path, read_identity(identity_path)


def check(config):
    identity = read_identity(config.identity_path)
    if config.project_identity and config.project_identity != identity:
        raise HowHardError("state: this project's identity changed or is missing")
    config.project_identity = identity
    path = config.state_root / "store.json"
    if path.exists():
        try:
            existing = json.loads(path.read_bytes())
        except (ValueError, UnicodeError) as exc:
            raise HowHardError(f"state: invalid store identity at {path}") from exc
        if not identity or existing != {"format": "howhard-store-v2", "project": identity}:
            raise HowHardError(f"state: store belongs to another project or lacks this project's identity: {config.state_root}")


@contextmanager
def identity_locked(config):
    # A project anchor is independent of custom stores. The shared identity lock
    # serializes first use even when callers choose different new state directories.
    check(config)
    path = config.identity_path.with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            check(config)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def claim(config):
    """Called after data validation, under both project identity and storage locks."""
    from .files import atomic_write
    check(config)
    if not config.project_identity:
        config.project_identity = secrets.token_hex(16)
        atomic_write(config.identity_path, (json.dumps({"format": "howhard-project-v1", "id": config.project_identity}, sort_keys=True) + "\n").encode())
    path = config.state_root / "store.json"
    if not path.exists():
        atomic_write(path, (json.dumps({"format": "howhard-store-v2", "project": config.project_identity}, sort_keys=True) + "\n").encode())
