"""Packaged defaults and optional, strictly validated project overrides."""

from copy import deepcopy
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
import tomllib

from .validation import HowHardError, keys, number, table, text

EFFORTS = ("low", "medium", "high", "xhigh", "max")


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


def validate_prior(p):
    template = parse_toml(packaged("default_prior.toml"), "default prior")
    keys(p, template, where="prior")
    if type(p["format"]) is not int or p["format"] != 1:
        raise HowHardError("prior.format: unsupported version")
    for vocab in ("families", "efforts"):
        if p[vocab] != template[vocab]:
            raise HowHardError(f"prior.{vocab}: unknown or missing family/effort")
    keys(p["transfer"], ["variance_inflation"], where="prior.transfer")
    number(p["transfer"]["variance_inflation"], "transfer.variance_inflation", positive=True)
    keys(p["outcome"], ["loud", "silent"], where="prior.outcome")
    for block in ("loud", "silent", "tokens", "time"):
        data = p["outcome"][block] if block in ("loud", "silent") else p[block]
        schema = template["outcome"][block] if block in ("loud", "silent") else template[block]
        keys(data, schema, where=f"prior.{block}")
        for term, spec in data.items():
            where = f"prior.{block}.{term}"
            if term == "noise_sd":
                number(spec, where, positive=True)
                continue
            keys(spec, schema[term], where=where)
            number(spec["sd"], f"{where}.sd", positive=True)
            field = "means" if "means" in spec else "mean"
            means = spec[field]
            if isinstance(schema[term][field], dict):
                keys(means, schema[term][field], where=where)
                values = means.values()
            elif isinstance(schema[term][field], list):
                if not isinstance(means, list) or len(means) != len(schema[term][field]):
                    raise HowHardError(f"{where}.means: wrong vocabulary length")
                values = means
            else:
                values = [means]
            for value in values:
                number(value, f"{where}.{field}")
    if not isinstance(p["taxonomy"], dict) or not p["taxonomy"]:
        raise HowHardError("prior.taxonomy: expected task types")
    for name, spec in p["taxonomy"].items():
        text(name, "taxonomy task type", label=True)
        keys(spec, ["definition", "class"], where=f"taxonomy.{name}")
        text(spec["definition"], f"taxonomy.{name}.definition")
        text(spec["class"], f"taxonomy.{name}.class")
        if spec["class"] not in template["tokens"]["task"]["means"]:
            raise HowHardError(f"taxonomy.{name}.class: unknown class {spec['class']!r}")
    return p


def load_prior(path=None, overrides=None):
    raw = Path(path).read_bytes() if path is not None else packaged("default_prior.toml")
    return validate_prior(merge(parse_toml(raw, "prior"), overrides or {}))


@dataclass
class Config:
    data: dict
    prior: dict
    base: Path
    source_bytes: bytes
    prior_bytes: bytes
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
    def families(self):
        return [*self.prior["families"], *(self.data["models"][model]["family"] for model in sorted(self.native_models))]

    @property
    def efforts(self):
        return [*self.prior["efforts"], *sorted({e for spec in self.native_models.values() for e in spec["efforts"]} - set(self.prior["efforts"]))]

    @property
    def task_types(self):
        return {**{t: s["class"] for t, s in self.prior["taxonomy"].items()},
                **self.data.get("task_types", {})}

    def validate_paths(self):
        names = [log["path"] for log in self.data["logs"]]
        names += [self.data[k] for k in ("journal", "cache")]
        names += [self.data["prior"][k] for k in ("path", "import") if k in self.data["prior"]]
        names.append(str(self.path(self.data["journal"]).with_suffix(".lock")))
        paths = [self.path(name).resolve() for name in names]
        if len(set(paths)) != len(paths):
            raise HowHardError("config paths: logs, journal, lock, cache and prior files must be distinct")


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
    keys(data, ["project", "on_cap_unmet", "journal", "cache", "logs", "prior", "classes", "models"],
         ["task_types", "transfer"], "config")
    for key in ("classes", "models", "task_types"):
        table(data.get(key, {}), f"config.{key}")
    text(data["project"], "config.project", label=True)
    if data["on_cap_unmet"] not in ("flag-review", "lowest-bound"):
        raise HowHardError("config.on_cap_unmet: unknown policy")
    for key in ("journal", "cache"):
        text(data[key], f"config.{key}")
    keys(data["prior"], [], ["path", "import", "overrides"], "config.prior")
    if "path" in data["prior"]:
        text(data["prior"]["path"], "config.prior.path")
    if "overrides" in data["prior"]:
        table(data["prior"]["overrides"], "config.prior.overrides")
    prior_path = base / data["prior"]["path"] if "path" in data["prior"] else None
    prior_raw = prior_path.read_bytes() if prior_path else packaged("default_prior.toml")
    prior = validate_prior(merge(parse_toml(prior_raw, "prior"), data["prior"].get("overrides", {})))
    if "transfer" in data:
        keys(data["transfer"], ["variance_inflation"], where="config.transfer")
        number(data["transfer"]["variance_inflation"], "config.transfer.variance_inflation", positive=True)
    if "import" in data["prior"]:
        text(data["prior"]["import"], "config.prior.import")
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
    paths += [(base / data[k]).resolve() for k in ("journal", "cache")]
    if len(set(paths)) != len(paths):
        raise HowHardError("config paths: logs, journal and cache must be distinct")
    for cls, spec in data["classes"].items():
        if cls not in prior["tokens"]["task"]["means"]:
            raise HowHardError(f"classes.{cls}: unknown class")
        keys(spec, ["C_loud", "C_silent", "lambda_per_hour", "epsilon"], ["explore_within"], f"classes.{cls}")
        for key, value in spec.items():
            number(value, f"classes.{cls}.{key}", nonnegative=True)
        if not 0 < spec["epsilon"] < 1:
            raise HowHardError(f"classes.{cls}.epsilon: expected probability in (0, 1)")
    for model, spec in data["models"].items():
        text(model, "model id")
        keys(spec, ["family", "efforts"], ["weight"], f"models.{model}")
        text(spec["family"], f"models.{model}.family")
        if spec["family"] not in prior["families"]:
            raise HowHardError(f"models.{model}.family: unknown family")
        efforts = spec["efforts"]
        if not isinstance(efforts, list) or not efforts or any(e not in EFFORTS for e in efforts) or len(set(efforts)) != len(efforts):
            raise HowHardError(f"models.{model}.efforts: unknown, duplicate or missing effort")
        number(spec.get("weight", 1.0), f"models.{model}.weight", positive=True)
    for task, cls in data.get("task_types", {}).items():
        text(task, "task_type", label=True)
        text(cls, f"task_types.{task}")
        if cls not in data["classes"]:
            raise HowHardError(f"task_types.{task}: unknown class {cls!r}")
    config = Config(data, prior, base, packaged("default_config.toml") + raw, prior_raw, path is not None)
    from .state import roots
    project = Path(directory).resolve() if directory is not None else base
    # Explicit configuration owns its relative data paths and supplemental store.
    selected_state = state_directory if state_directory is not None else base / ".howhard" if path is not None else None
    config.project_root, config.state_root, config.identity_path, config.project_identity = roots(project, explicit=path is not None, state_directory=selected_state)
    config.validate_paths()
    return config
