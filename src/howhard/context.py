"""Versioned local features and immutable snapshots of changing project state."""

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tomllib
import unicodedata
from stat import S_ISREG

import numpy as np

from .config import packaged
from .validation import HowHardError, keys, number

SETTINGS = tomllib.loads(packaged("context.toml").decode())
FORMAT = SETTINGS["format"]
PROJECT_FORMAT = "howhard-project-v1"
DECISION_FORMAT = "howhard-decision-v1"
COUNTS = ("files", "source_bytes", "languages", "test_files", "manifests", "changed_files")
FLAGS = ("has_ci", "has_lockfile", "has_typing")
PARAMETERS = (*COUNTS, *FLAGS, "coverage")
FEATURE_NAMES = tuple([f"task-text-{i:02d}" for i in range(SETTINGS["task_buckets"])] +
                      [f"project-text-{i:02d}" for i in range(SETTINGS["project_buckets"])] +
                      ["task_length", "task_lines", *COUNTS, *FLAGS, "coverage", "coverage_known", "project_known"])
IGNORED = {".git", ".howhard", "venv", "env", ".env", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", "dist", "build", "_build", "htmlcov", "vendor", "target"}
SOURCE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".java", ".kt", ".swift", ".rb", ".php", ".cs", ".sh", ".sql", ".ex", ".exs", ".scala", ".vue", ".svelte"}
MANIFESTS = {"pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Gemfile", "pom.xml", "build.gradle", "composer.json"}
LOCKFILES = {"uv.lock", "poetry.lock", "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "Cargo.lock", "go.sum", "Gemfile.lock"}
TYPING = {"pyrightconfig.json", "mypy.ini", ".mypy.ini", "tsconfig.json"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def validate_text(value, where="task_text"):
    if not isinstance(value, str):
        raise HowHardError(f"{where}: expected string")
    return value


def validate_parameters(value, *, complete=False):
    keys(value, PARAMETERS if complete else [], [] if complete else PARAMETERS, "project_context.parameters")
    for key, item in value.items():
        where = f"project_context.parameters.{key}"
        if key in FLAGS:
            if type(item) is not bool:
                raise HowHardError(f"{where}: expected boolean")
        elif key == "coverage":
            if item is not None and not 0 <= number(item, where) <= 1:
                raise HowHardError(f"{where}: expected fraction in [0, 1] or null")
        elif type(item) is not int or item < 0:
            raise HowHardError(f"{where}: expected non-negative integer")


def validate_project(value):
    keys(value, ["format", "parameters", "text", "revision", "fingerprint", "truncated"], where="project_context")
    if value["format"] != PROJECT_FORMAT:
        raise HowHardError("project_context.format: unsupported version")
    validate_parameters(value["parameters"], complete=True)
    validate_text(value["text"], "project_context.text")
    if len(value["text"]) > SETTINGS["text_limit"]:
        raise HowHardError("project_context.text: exceeds snapshot text limit")
    revision = value["revision"]
    if revision is not None and (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision)):
        raise HowHardError("project_context.revision: expected Git object id or null")
    if not isinstance(value["fingerprint"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["fingerprint"]):
        raise HowHardError("project_context.fingerprint: expected SHA-256")
    if type(value["truncated"]) is not bool:
        raise HowHardError("project_context.truncated: expected boolean")
    return value


def git(directory, *args):
    try:
        result = subprocess.run(["git", "--no-optional-locks", "-C", str(directory), *args], capture_output=True, timeout=5)
        return result.stdout if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def capture_project(config, overrides=None, *, required_path=None):
    """Read the current working tree; never revisit it when fitting historical rows."""
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise HowHardError("project_context: expected snapshot or parameter object")
        if "format" in overrides:
            return json.loads(canonical(validate_project(overrides)))
        validate_parameters(overrides)
    root = config.project_root
    state_root = config.state_root.resolve()
    excluded = {config.path(s["path"]).resolve() for s in config.data["logs"]}
    excluded.update(config.path(config.data[k]).resolve() for k in ("journal", "cache"))
    listing = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    included = {os.fsdecode(p) for p in listing.split(b"\0") if p} if listing is not None else None
    def candidates():
        if included is not None:
            for name in sorted(included):
                yield Path(name), root / name
        else:
            for current, dirs, files in os.walk(root, followlinks=False):
                dirs[:] = sorted(d for d in dirs if d not in IGNORED and not d.startswith(".venv") and not d.endswith(".egg-info") and not (Path(current) / d).is_symlink() and not (Path(current) / d).resolve().is_relative_to(state_root))
                for name in sorted(files):
                    path = Path(current) / name
                    yield path.relative_to(root), path

    paths, truncated = [], False
    for relative, path in candidates():
        if any(part in IGNORED or part.startswith(".venv") or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink() or path.resolve() in excluded or path.resolve().is_relative_to(state_root) or not path.resolve().is_relative_to(root):
            continue
        paths.append((relative, path))
        if len(paths) >= SETTINGS["file_limit"]:
            truncated = True
            break
    if required_path is not None and required_path.resolve() not in {path.resolve() for _, path in paths}:
        raise HowHardError("retrieval.path: file is outside the captured snapshot (ignored, excluded or beyond the scan limit)")
    params = {key: 0 for key in COUNTS}
    params.update({key: False for key in FLAGS}, coverage=None)
    languages, descriptors, stamps, sources = set(), [], [], []
    for relative, path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        if not S_ISREG(stat.st_mode):
            continue
        params["files"] += 1
        stamps.append([relative.as_posix(), stat.st_size, stat.st_mtime_ns])
        name, suffix = path.name, path.suffix.lower()
        if suffix in SOURCE_SUFFIXES:
            languages.add(suffix)
            params["source_bytes"] += stat.st_size
            sources.append((hashlib.sha256(os.fsencode(relative.as_posix())).digest(), path))
        if any(p.lower() in ("test", "tests", "__tests__", "spec") for p in relative.parts[:-1]) or name.startswith("test_") or ".test." in name or name.endswith("_test.go"):
            params["test_files"] += 1
        params["has_ci"] |= relative.parts[:2] == (".github", "workflows") or name in (".gitlab-ci.yml", "Jenkinsfile", ".travis.yml")
        params["has_lockfile"] |= name in LOCKFILES
        params["has_typing"] |= name in TYPING or suffix in (".ts", ".tsx")
        params["manifests"] += name in MANIFESTS
        if name.lower().startswith("readme") or name in MANIFESTS:
            try:
                content = path.open("rb")
                with content:
                    descriptor = content.read(SETTINGS["text_limit"]).decode("utf-8", errors="replace")
                descriptors.append((relative.as_posix(), descriptor))
                if name == "pyproject.toml":
                    params["has_typing"] |= "[tool.mypy" in descriptor or "[tool.pyright" in descriptor
            except OSError:
                continue
    params["languages"] = len(languages)
    # Use a stable path set, not status text: filenames with spaces/newlines remain intact.
    changed_paths = set()
    changed = git(root, "ls-files", "-z", "--modified", "--deleted", "--others", "--exclude-standard")
    if changed is not None:
        changed_paths = {os.fsdecode(p) for p in changed.split(b"\0") if p}
        staged = git(root, "diff", "--cached", "--name-only", "-z", "--no-renames")
        if staged is not None:
            changed_paths.update(os.fsdecode(p) for p in staged.split(b"\0") if p)
        changed_paths = {p for p in changed_paths if (root / p).resolve() not in excluded and not (root / p).resolve().is_relative_to(state_root) and
                         not any(part in IGNORED or part.startswith(".venv") or part.endswith(".egg-info") for part in Path(p).parts)}
        params["changed_files"] = len(changed_paths)
    revision = git(root, "rev-parse", "--verify", "HEAD")
    revision = revision.decode().strip() if revision else None
    snippets = []
    for _, path in sorted(sources)[:SETTINGS["source_files"]]:
        try:
            with path.open("rb") as handle:
                snippets.append(handle.read(SETTINGS["source_chars"]).decode("utf-8", errors="replace"))
        except OSError:
            continue
    budget = SETTINGS["text_limit"] // 2
    documentation = "\n".join(content for _, content in sorted(descriptors))[:budget]
    source_text = "\n".join(snippets)[:budget - 1]
    text = "\n".join(part for part in (documentation, source_text) if part)
    params.update(overrides or {})
    value = {"format": PROJECT_FORMAT, "parameters": params, "text": text, "revision": revision,
             "fingerprint": digest([params, text, revision, stamps, sorted(changed_paths)]), "truncated": truncated}
    return validate_project(value)


def lexical(value, buckets):
    value = unicodedata.normalize("NFKC", value[:SETTINGS["text_limit"]]).casefold()
    words = re.findall(r"\w+", value)[:SETTINGS["word_limit"]]
    terms = Counter([f"u:{w}" for w in words] + [f"b:{a} {b}" for a, b in zip(words, words[1:])])
    result = np.zeros(buckets)
    for term, count in sorted(terms.items()):
        hashed = hashlib.blake2b(term.encode(), digest_size=8, person=b"howhard-text-v1").digest()
        result[int.from_bytes(hashed[:4], "big") % buckets] += (1 if hashed[4] & 1 else -1) * math.log1p(count)
    norm = np.linalg.norm(result)
    return result / norm if norm else result


def scaled(value, scale):
    if value >= scale:
        return 1.0
    return min(1.0, math.log1p(value) / math.log1p(scale))


def encode(task_text, project_context):
    validate_text(task_text)
    project_text = "" if project_context is None else validate_project(project_context)["text"]
    result = dict(zip(FEATURE_NAMES, np.zeros(len(FEATURE_NAMES))))
    for prefix, value, buckets in (("task-text", task_text, SETTINGS["task_buckets"]), ("project-text", project_text, SETTINGS["project_buckets"])):
        result.update({f"{prefix}-{i:02d}": float(x) for i, x in enumerate(lexical(value, buckets))})
    result["task_length"] = scaled(len(task_text), SETTINGS["scales"]["task_length"])
    result["task_lines"] = scaled(task_text.count("\n") + bool(task_text), SETTINGS["scales"]["task_lines"])
    if project_context is not None:
        params = project_context["parameters"]
        for key in COUNTS:
            result[key] = scaled(params[key], SETTINGS["scales"][key])
        result.update({key: float(params[key]) for key in FLAGS})
        result["coverage_known"] = float(params["coverage"] is not None)
        result["coverage"] = params["coverage"] or 0.0
        result["project_known"] = 1.0
    return result


def row_context(row):
    return encode(row.get("task_text", ""), row.get("project_context"))


def validate_row_context(row, where):
    present = {"task_text", "project_context", "context_encoder"} & row.keys()
    if present and present != {"task_text", "project_context", "context_encoder"}:
        raise HowHardError(f"{where}: task_text, project_context and context_encoder must occur together")
    if present:
        if row["context_encoder"] != FORMAT:
            raise HowHardError(f"{where}.context_encoder: unsupported version")
        validate_text(row["task_text"], f"{where}.task_text")
        if row["project_context"] is not None:
            try:
                validate_project(row["project_context"])
            except HowHardError as exc:
                raise HowHardError(f"{where}.{exc}") from exc
