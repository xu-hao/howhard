"""Strict append-only JSONL, amendments and a durable pending journal."""

from contextlib import contextmanager, ExitStack
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import secrets

from .validation import HowHardError, keys, number, text
from .context import FORMAT as CONTEXT_FORMAT, capture_project, validate_row_context

ROW_KEYS = ("id", "date", "ticket", "task_type", "risk_class", "model", "effort",
            "outcome", "cause", "tokens", "seconds", "cost_usd", "note", "amends", "observed_at")


@contextmanager
def locked(config, *, operation="storage write"):
    from .state import check
    from .writes import admit, current, file_lock
    check(config)
    paths = {config.state_root / "storage.lock", config.path(config.data["journal"]).with_suffix(".lock").resolve()}
    paths = sorted(paths, key=lambda p: (p != config.state_root / "storage.lock", str(p)))
    previous = current(config.state_root)
    if previous is not None and all(p in previous.storage_locks for p in paths):
        yield
        return
    with admit(config.state_root, config.write_policy, operation=operation) as wait:
        with ExitStack() as stack:
            acquired = []
            for path in paths:
                if path not in wait.storage_locks:
                    stack.enter_context(file_lock(path, wait))
                    acquired.append(path)
            check(config)
            wait.storage_locks.update(acquired)
            try:
                yield
            finally:
                wait.storage_locks.difference_update(acquired)


@contextmanager
def reading(config):
    """Consistent JSONL snapshots bypass queued writers, waiting only on mutation."""
    from .state import check
    from .writes import current, file_lock, Wait, WriteDependencyError
    check(config)
    previous = current(config.state_root)
    paths = [config.state_root / "storage.lock", config.path(config.data["journal"]).with_suffix(".lock").resolve()]
    if previous is not None and all(p in previous.storage_locks for p in paths):
        check(config)
        yield
        return
    wait = Wait(config.state_root, config.write_policy, "storage read")
    try:
        with ExitStack() as stack:
            for path in dict.fromkeys(paths):
                if previous is None or path not in previous.storage_locks:
                    stack.enter_context(file_lock(path, wait, shared=True))
            check(config)
            yield
    except OSError as exc:
        raise WriteDependencyError(config.state_root, "storage read", str(exc)) from exc


def read_bytes(path):
    try:
        return Path(path).read_bytes()
    except FileNotFoundError:
        return b""


def validate_measurement(value, key, where):
    if value is not None:
        number(value, f"{where}.{key}", nonnegative=True)
        if key == "tokens" and type(value) is not int:
            raise HowHardError(f"{where}.{key}: expected integer or null")


def validate_row(row, config, where):
    keys(row, ROW_KEYS, ["task_text", "project_context", "context_encoder", "decision_id", "measurement_corrections", "execution_usage"], where=where)
    try:
        if not isinstance(row["observed_at"], str) or datetime.fromisoformat(row["observed_at"]).utcoffset() is None:
            raise ValueError
    except ValueError as exc:
        raise HowHardError(f"{where}.observed_at: expected ISO timestamp with timezone") from exc
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
    if row["effort"] not in model["efforts"]:
        raise HowHardError(f"{where}.effort: unsupported by models.{row['model']}")
    if row["outcome"] not in ("clean", "loud", "silent"):
        raise HowHardError(f"{where}.outcome: unknown outcome")
    if row["outcome"] == "clean":
        if row["cause"] is not None:
            raise HowHardError(f"{where}.cause: must be null for clean")
    elif row["cause"] not in ("depth", "capability", "unrecorded"):
        raise HowHardError(f"{where}.cause: expected depth, capability or unrecorded")
    for key in ("tokens", "seconds", "cost_usd"):
        validate_measurement(row[key], key, where)
    if "execution_usage" in row:
        if row["amends"] is not None:
            raise HowHardError("execution_usage: amendments use measurement_corrections")
        if row["execution_usage"] is not None:
            from .accounting import validate
            validate(row["execution_usage"])
            if row["execution_usage"]["execution_id"] != row["id"]:
                raise HowHardError("execution_usage.execution_id: must match original row ID")
    if "measurement_corrections" in row:
        corrections = row["measurement_corrections"]
        location = f"{where}.measurement_corrections"
        if row["amends"] is None:
            raise HowHardError(f"{location}: only permitted on amendments")
        keys(corrections, [], ["tokens", "seconds", "cost_usd", "execution_usage"], where=location)
        if not corrections:
            raise HowHardError(f"{location}: expected at least one corrected measurement")
        for key, value in corrections.items():
            if key == "execution_usage":
                if value is not None:
                    from .accounting import validate
                    validate(value)
            else:
                validate_measurement(value, key, location)
    if row["amends"] is not None:
        if any(row[k] is not None for k in ("tokens", "seconds", "cost_usd")):
            raise HowHardError(f"{where}.amends: amendments require null tokens, seconds and cost_usd")


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
    from .accounting import duplicates
    duplicates(effective_rows(config, ordered))
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
                                     "resolved_effort": row["effort"]}
        else:
            root = roots[row["amends"]]
            roots[row["id"]] = root
            originals[root]["outcome"] = row["outcome"]
            originals[root]["cause"] = row["cause"]
            originals[root].update(row.get("measurement_corrections", {}))
    for row in originals.values():
        usage = row.get("execution_usage")
        if usage is not None and usage["provenance"].get("completed") and usage["segments"] and all(s.get("reported_tokens") is not None for s in usage["segments"]):
            row["tokens"] = sum(s["reported_tokens"] for s in usage["segments"])
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


def observe(config, task_type, model, effort, outcome, *, _automatic_context=(), native=False, _preserve_outcome=False, **options):
    allowed = {"cause", "tokens", "seconds", "cost_usd", "ticket", "note", "amends", "date", "id", "task_text", "project_context", "decision_id", "risk_class", "measurement_corrections", "execution_usage"}
    unknown = options.keys() - allowed
    if unknown:
        raise HowHardError(f"observe: unknown keys {sorted(unknown)}")
    if native and (options.get("amends") is not None or options.get("decision_id") is not None):
        raise HowHardError("native: actual identity admission is only for original raw feedback without a decision")
    with locked(config, operation="observe"):
        from . import native as native_registry
        from .state import check
        from copy import deepcopy
        check(config)
        native_registry.apply(config, native_registry.read(config))
        current_rows, blobs = snapshot(config)
        validation_config = config
        if native:
            old_models, updated = native_registry.plan(config, [{"model": model, "effort": effort}])
            validation_config = deepcopy(config)
            native_registry.apply(validation_config, updated)
        if _preserve_outcome:
            target = next((r for r in effective_rows(config, current_rows) if r["id"] == options.get("amends")), None)
            if target is None:
                raise HowHardError("usage attach: target original is unavailable")
            outcome = target["outcome"]
            options["cause"] = target["cause"]
        registry = validation_config.data["models"].get(model)
        if registry is None:
            raise HowHardError(f"model: unknown model id {model!r}; add it to the registry or use native actual feedback")
        row = dict(id=options.get("id", secrets.token_hex(6)), date=options.get("date", date.today().isoformat()),
                   observed_at=datetime.now(timezone.utc).isoformat(),
                   ticket=options.get("ticket"), task_type=task_type, risk_class=options.get("risk_class", config.task_types.get(task_type, "standard")),
                   model=model, effort=effort,
                   outcome=outcome, cause=options.get("cause", None if outcome == "clean" else "unrecorded"),
                   tokens=options.get("tokens"), seconds=options.get("seconds"), cost_usd=options.get("cost_usd"),
                   note=options.get("note", ""), amends=options.get("amends"))
        if "execution_usage" in options:
            row["execution_usage"] = deepcopy(options["execution_usage"])
        if "measurement_corrections" in options:
            row["measurement_corrections"] = deepcopy(options["measurement_corrections"])
        if row["amends"] is None:
            row.update(task_text=options.get("task_text", ""), project_context=capture_project(config, options.get("project_context")), context_encoder=CONTEXT_FORMAT)
        if "decision_id" in options:
            row["decision_id"] = options["decision_id"]
        path, project, pending = blobs[-1]
        line = len(pending.splitlines()) + 1
        validate_row(row, validation_config, f"{path}: line {line}")
        existing = next((r for r in current_rows if r["id"] == row["id"]), None)
        if existing is not None:
            if existing["project"] != project:
                raise HowHardError(f"observe.id: conflicting project for {row['id']!r}")
            recorded = {k: v for k, v in existing.items() if k != "project"}
            comparable = dict(row)
            if _preserve_outcome:
                comparable.update(outcome=recorded["outcome"], cause=recorded["cause"])
            for field in _automatic_context:
                comparable[field] = recorded[field]
            comparable["observed_at"] = recorded["observed_at"]
            if "date" not in options:
                comparable["date"] = recorded["date"]
            if "risk_class" not in options:
                comparable["risk_class"] = recorded["risk_class"]
            if comparable != recorded:
                raise HowHardError(f"observe.id: conflicting feedback for {row['id']!r}")
            return recorded
        blob = (json.dumps(row, allow_nan=False) + "\n").encode()
        parse_rows([*blobs[:-1], (path, project, pending + (b"\n" if pending and not pending.endswith(b"\n") else b"") + blob)], validation_config)
        if native:
            native_registry.persist(config, old_models, updated)
        append_bytes(path, blob)
    return row


def flush(config):
    with locked(config, operation="flush"):
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
