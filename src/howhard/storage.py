"""Strict append-only JSONL, amendments and a durable pending journal."""

from contextlib import contextmanager
from datetime import date
import fcntl
import json
import os
from pathlib import Path
import secrets

from .config import EFFORTS
from .validation import HowHardError, keys, number, text
from .context import FORMAT as CONTEXT_FORMAT, capture_project, validate_row_context

ROW_KEYS = ("id", "date", "ticket", "task_type", "risk_class", "model", "effort",
            "default_effort", "outcome", "cause", "tokens", "seconds", "cost_usd", "note", "amends")


@contextmanager
def locked(config):
    path = config.path(config.data["journal"]).with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_bytes(path):
    try:
        return Path(path).read_bytes()
    except FileNotFoundError:
        return b""


def validate_row(row, config, where):
    keys(row, ROW_KEYS, ["task_text", "project_context", "context_encoder", "decision_id"], where=where)
    validate_row_context(row, where)
    if "decision_id" in row:
        text(row["decision_id"], f"{where}.decision_id")
    for key in ("id", "task_type", "risk_class", "model", "effort", "outcome"):
        text(row[key], f"{where}.{key}")
    text(row["task_type"], f"{where}.task_type", label=True)
    try:
        if not isinstance(row["date"], str) or date.fromisoformat(row["date"]).isoformat() != row["date"]:
            raise ValueError
    except ValueError as exc:
        raise HowHardError(f"{where}.date: expected YYYY-MM-DD") from exc
    for key in ("ticket", "amends"):
        if row[key] is not None and not isinstance(row[key], str):
            raise HowHardError(f"{where}.{key}: expected string or null")
    if row["amends"] == "":
        raise HowHardError(f"{where}.amends: expected a non-empty id")
    if not isinstance(row["note"], str):
        raise HowHardError(f"{where}.note: expected string")
    if row["risk_class"] not in config.data["classes"]:
        raise HowHardError(f"{where}.risk_class: unknown class {row['risk_class']!r}")
    if row["task_type"] not in config.task_types and config.explicit:
        raise HowHardError(f"{where}.task_type: unknown task type {row['task_type']!r}; add it to task_types")
    model = config.data["models"].get(row["model"])
    if model is None:
        raise HowHardError(f"{where}.model: unknown model id {row['model']!r}; add it to the model registry")
    if row["effort"] == "model_default":
        if row["default_effort"] not in EFFORTS:
            raise HowHardError(f"{where}.default_effort: required for model_default")
    elif row["effort"] not in EFFORTS or row["default_effort"] is not None:
        raise HowHardError(f"{where}.effort/default_effort: invalid effort or unexpected default_effort")
    resolved = row["default_effort"] if row["effort"] == "model_default" else row["effort"]
    if resolved not in model["efforts"]:
        raise HowHardError(f"{where}.effort: unsupported by models.{row['model']}")
    if row["outcome"] not in ("clean", "loud", "silent"):
        raise HowHardError(f"{where}.outcome: unknown outcome")
    if row["outcome"] == "clean":
        if row["cause"] is not None:
            raise HowHardError(f"{where}.cause: must be null for clean")
    elif row["cause"] not in ("depth", "capability", "unrecorded"):
        raise HowHardError(f"{where}.cause: expected depth, capability or unrecorded")
    for key in ("tokens", "seconds", "cost_usd"):
        if row[key] is not None:
            number(row[key], f"{where}.{key}", nonnegative=True)
            if key == "tokens" and type(row[key]) is not int:
                raise HowHardError(f"{where}.tokens: expected integer or null")
    if row["amends"] is not None:
        if row["outcome"] != "silent" or any(row[k] is not None for k in ("tokens", "seconds", "cost_usd")):
            raise HowHardError(f"{where}.amends: amendments must be silent with null tokens, seconds and cost_usd")


def parse_rows(blobs, config):
    rows, seen, locations, duplicates = [], {}, {}, set()
    for path, project, blob in blobs:
        try:
            lines = blob.decode("utf-8").splitlines()
        except UnicodeError as exc:
            raise HowHardError(f"{path}: line 1: invalid UTF-8") from exc
        for line, value in enumerate(lines, 1):
            where = f"{path}: line {line}"
            try:
                row = json.loads(value)
            except (ValueError, TypeError) as exc:
                raise HowHardError(f"{where}: invalid JSON object") from exc
            validate_row(row, config, where)
            if row["id"] in seen:
                duplicates.add(row["id"])
            locations[row["id"]] = path, line
            entry = {**row, "project": project}
            seen[row["id"]] = entry
            rows.append(entry)
    if duplicates:
        locations_text = ", ".join(f"{locations[k][0]}: line {locations[k][1]}" for k in sorted(duplicates))
        raise HowHardError(f"{locations_text}.id: duplicate ids: {', '.join(sorted(duplicates))}")
    for row in rows:
        target = row["amends"]
        if target is None:
            continue
        path, line = locations[row["id"]]
        target_location = locations.get(target)
        # Within a file the append order defines 'earlier'. Across files the target
        # must exist, not be future-dated, and form an acyclic dependency. This keeps
        # valid amendments readable when flush moves them before another log.
        if target_location is None or (target_location[0] == path and target_location[1] >= line) or seen[target]["date"] > row["date"]:
            raise HowHardError(f"{path}: line {line}.amends: unknown or later id {target!r}")
    ordered, done = [], set()
    for row in rows:
        chain, visiting = [], set()
        current = row
        while current["id"] not in done:
            if current["id"] in visiting:
                path, line = locations[current["id"]]
                raise HowHardError(f"{path}: line {line}.amends: amendment cycle")
            visiting.add(current["id"])
            chain.append(current)
            if current["amends"] is None:
                break
            current = seen[current["amends"]]
        for item in reversed(chain):
            ordered.append(item)
            done.add(item["id"])
    return ordered


def snapshot(config):
    specs = [(config.path(log["path"]), log["project"]) for log in config.data["logs"]]
    specs.append((config.path(config.data["journal"]), config.data["project"]))
    blobs = [(path, project, read_bytes(path)) for path, project in specs]
    return parse_rows(blobs, config), blobs


def effective_rows(config, rows):
    originals, roots = {}, {}
    for row in rows:
        if row["amends"] is None:
            roots[row["id"]] = row["id"]
            model = config.data["models"][row["model"]]
            originals[row["id"]] = {**row, "family": model["family"], "weight": model.get("weight", 1.0),
                                     "resolved_effort": row["default_effort"] if row["effort"] == "model_default" else row["effort"]}
        else:
            root = roots[row["amends"]]
            roots[row["id"]] = root
            originals[root]["outcome"] = row["outcome"]
            originals[root]["cause"] = row["cause"]
    return [originals[key] for key in sorted(originals)]


def append_bytes(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell():
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())


def observe(config, task_type, model, effort, outcome, **options):
    allowed = {"cause", "tokens", "seconds", "cost_usd", "ticket", "note", "amends", "date", "id", "task_text", "project_context", "decision_id", "risk_class"}
    unknown = options.keys() - allowed
    if unknown:
        raise HowHardError(f"observe: unknown keys {sorted(unknown)}")
    registry = config.data["models"].get(model)
    if registry is None:
        raise HowHardError(f"model: unknown model id {model!r}; add it to the model registry")
    row = dict(id=options.get("id", secrets.token_hex(6)), date=options.get("date", date.today().isoformat()),
               ticket=options.get("ticket"), task_type=task_type, risk_class=options.get("risk_class", config.task_types.get(task_type, "standard")),
               model=model, effort=effort, default_effort=registry["default_effort"] if effort == "model_default" else None,
               outcome=outcome, cause=options.get("cause", None if outcome == "clean" else "unrecorded"),
               tokens=options.get("tokens"), seconds=options.get("seconds"), cost_usd=options.get("cost_usd"),
               note=options.get("note", ""), amends=options.get("amends"))
    if "task_text" in options or "project_context" in options:
        row.update(task_text=options.get("task_text", ""), project_context=capture_project(config, options.get("project_context")), context_encoder=CONTEXT_FORMAT)
    if "decision_id" in options:
        row["decision_id"] = options["decision_id"]
    with locked(config):
        _, blobs = snapshot(config)
        path, project, pending = blobs[-1]
        line = len(pending.splitlines()) + 1
        validate_row(row, config, f"{path}: line {line}")
        blob = (json.dumps(row, allow_nan=False) + "\n").encode()
        parse_rows([*blobs[:-1], (path, project, pending + (b"\n" if pending and not pending.endswith(b"\n") else b"") + blob)], config)
        append_bytes(path, blob)
    return row


def flush(config):
    with locked(config):
        _, blobs = snapshot(config)  # Validate everything before either file is changed.
        journal, project, pending = blobs[-1]
        # Amendment references can point into the logs, so take ids directly from the
        # already-validated pending bytes rather than validating a disconnected journal.
        moved = [json.loads(line)["id"] for line in pending.splitlines()]
        if pending:
            log_path, log_project, log_bytes = blobs[0]
            separator = b"\n" if log_bytes and not log_bytes.endswith(b"\n") else b""
            proposed = [(log_path, log_project, log_bytes + separator + pending), *blobs[1:-1], (journal, project, b"")]
            parse_rows(proposed, config)
            append_bytes(config.path(config.data["logs"][0]["path"]), pending if pending.endswith(b"\n") else pending + b"\n")
            with journal.open("wb") as handle:
                handle.flush()
                os.fsync(handle.fileno())
    return moved
