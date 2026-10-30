"""Packaged defaults and optional, strictly validated project overrides."""

from copy import deepcopy
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
import tomllib

from .validation import HowHardError, keys, number, table, text


def merge(base, changes):
    result = deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def packaged(name):
    return files("howhard").joinpath("data", name).read_bytes()


def parse_toml(data, where):
    try:
        return tomllib.loads(data.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise HowHardError(f"{where}: {exc}") from exc


@dataclass
class Config:
    data: dict
    base: Path
    explicit: bool
    project_root: Path | None = None
    state_root: Path | None = None
    identity_path: Path | None = None
    project_identity: str = ""
    native_models: dict = field(default_factory=dict)

    def path(self, value):
        relative = Path(text(value, "path"))
        if relative.parts[:1] == (".howhard",) and self.state_root is not None:
            return self.state_root.joinpath(*relative.parts[1:])
        return self.base / relative

    @property
    def task_types(self):
        return self.data.get("task_types", {})

    def validate_paths(self):
        names = [log["path"] for log in self.data["logs"]]
        names.append(self.data["journal"])
        names.append(str(self.path(self.data["journal"]).with_suffix(".lock")))
        paths = [self.path(name).resolve() for name in names]
        if len(set(paths)) != len(paths):
            raise HowHardError("config paths: logs, journal and lock files must be distinct")


def load_config(path=None, *, directory=None, state_directory=None):
    base = Path(directory or Path.cwd()).resolve()
    if path is None and (base / "howhard.toml").is_file():
        path = base / "howhard.toml"
    default = parse_toml(packaged("default_config.toml"), "default config")
    raw = b""
    if path is not None:
        path = Path(path).resolve()
        base = path.parent
        raw = path.read_bytes()
        overrides = parse_toml(raw, str(path))
        # An explicitly supplied policy must make its fallback intentional.
        if "on_cap_unmet" not in overrides:
            raise HowHardError("config.on_cap_unmet: required; set 'flag-review' or 'lowest-bound'")
        data = merge(default, overrides)
    else:
        data = default
    keys(data, ["project", "on_cap_unmet", "journal", "logs", "classes", "models"],
         ["task_types"], "config")
    for key in ("classes", "models", "task_types"):
        table(data.get(key, {}), f"config.{key}")
    text(data["project"], "config.project", label=True)
    if data["on_cap_unmet"] not in ("flag-review", "lowest-bound"):
        raise HowHardError("config.on_cap_unmet: unknown policy")
    text(data["journal"], "config.journal")
    if not isinstance(data["logs"], list) or not data["logs"]:
        raise HowHardError("config.logs: at least one log is required")
    paths = []
    for log in data["logs"]:
        keys(log, ["path", "project"], where="config.logs")
        text(log["path"], "logs.path")
        text(log["project"], "logs.project", label=True)
        paths.append((base / log["path"]).resolve())
    if data["logs"][0]["project"] != data["project"]:
        raise HowHardError("logs[0].project must match config.project (flush destination)")
    paths.append((base / data["journal"]).resolve())
    if len(set(paths)) != len(paths):
        raise HowHardError("config paths: logs and journal must be distinct")
    for cls, spec in data["classes"].items():
        text(cls, "risk class", label=True)
        keys(spec, ["C_loud", "C_silent", "lambda_per_hour", "epsilon"], ["explore_within"], f"classes.{cls}")
        for key, value in spec.items():
            number(value, f"classes.{cls}.{key}", nonnegative=True)
        if not 0 < spec["epsilon"] < 1:
            raise HowHardError(f"classes.{cls}.epsilon: expected probability in (0, 1)")
    for model, spec in data["models"].items():
        text(model, "model id")
        keys(spec, ["family", "efforts"], ["weight"], f"models.{model}")
        text(spec["family"], f"models.{model}.family")
        efforts = spec["efforts"]
        if not isinstance(efforts, list) or not efforts or any(not isinstance(e, str) or not e or e == "model_default" for e in efforts) or len(set(efforts)) != len(efforts):
            raise HowHardError(f"models.{model}.efforts: unknown, duplicate or missing effort")
        number(spec.get("weight", 1.0), f"models.{model}.weight", positive=True)
    for task, cls in data.get("task_types", {}).items():
        text(task, "task_type", label=True)
        text(cls, f"task_types.{task}")
        if cls not in data["classes"]:
            raise HowHardError(f"task_types.{task}: unknown class {cls!r}")
    config = Config(data, base, path is not None)
    from .state import roots
    project = Path(directory).resolve() if directory is not None else base
    # Explicit configuration owns its relative data paths and supplemental store.
    selected_state = state_directory if state_directory is not None else base / ".howhard" if path is not None else None
    config.project_root, config.state_root, config.identity_path, config.project_identity = roots(project, explicit=path is not None, state_directory=selected_state)
    config.validate_paths()
    return config
