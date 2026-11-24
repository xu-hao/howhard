"""Lossless, content-addressed request and receipt storage."""

from datetime import datetime, timezone
from hashlib import sha256
import json
import re
import sqlite3

import zstandard as zstd

from .context import digest
from .validation import HowHardError

NAME = "payloads.sqlite3"
ID = re.compile(r"[0-9a-f]{64}\Z")
TABLES = {"requests", "decisions"}
DECISION_COLUMNS = ("decision_id", "created_at", "chosen", "context_summary", "selection",
                    "task_type", "risk_class", "cap_met", "needs_review")


def _path(config):
    return config.state_root / NAME


def _connect(config, *, write=False):
    path = _path(config)
    if path.is_symlink():
        raise HowHardError(f"payload store: refusing symlink at {path}")
    if not write and not path.is_file():
        return None
    db = None
    try:
        db = sqlite3.connect(path if write else path.as_uri() + "?mode=ro", uri=not write, timeout=30)
        if write:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("CREATE TABLE IF NOT EXISTS requests (request_id TEXT PRIMARY KEY, payload BLOB NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS payload_variants (
                kind TEXT NOT NULL, payload_id TEXT NOT NULL, raw_sha256 TEXT NOT NULL,
                payload BLOB NOT NULL, PRIMARY KEY(kind, payload_id, raw_sha256))""")
            db.execute("""CREATE TABLE IF NOT EXISTS decisions (
                decision_id TEXT PRIMARY KEY, payload BLOB NOT NULL, created_at TEXT NOT NULL,
                chosen TEXT, context_summary TEXT, selection TEXT, task_type TEXT, risk_class TEXT,
                cap_met INTEGER, needs_review INTEGER)""")
            db.execute("CREATE INDEX IF NOT EXISTS decisions_created_at ON decisions(created_at)")
            db.execute("CREATE INDEX IF NOT EXISTS decisions_task_risk ON decisions(task_type, risk_class)")
            db.execute("CREATE INDEX IF NOT EXISTS decisions_chosen ON decisions(chosen)")
            db.execute("CREATE INDEX IF NOT EXISTS decisions_review ON decisions(needs_review, cap_met)")
            db.execute("""CREATE TABLE IF NOT EXISTS prototypes (
                prototype_id TEXT PRIMARY KEY, task_type TEXT NOT NULL, risk_class TEXT NOT NULL,
                representative_task TEXT NOT NULL, representative_context TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1, parent_id TEXT,
                member_count INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS prototypes_active_task ON prototypes(active, task_type)")
            db.execute("""CREATE TABLE IF NOT EXISTS prototype_members (
                evidence_id TEXT NOT NULL, prototype_id TEXT NOT NULL, weight REAL NOT NULL,
                PRIMARY KEY(evidence_id, prototype_id))""")
            db.execute("CREATE INDEX IF NOT EXISTS prototype_members_proto ON prototype_members(prototype_id)")
            db.execute("""CREATE TABLE IF NOT EXISTS prototype_cells (
                prototype_id TEXT NOT NULL, model TEXT NOT NULL, family TEXT NOT NULL, effort TEXT NOT NULL,
                prior_json TEXT, stats_json TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(prototype_id, model, effort))""")
            db.execute("CREATE TABLE IF NOT EXISTS prototype_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS prototype_row_state (
                evidence_id TEXT PRIMARY KEY, row_digest TEXT NOT NULL, position INTEGER NOT NULL)""")
    except sqlite3.Error as exc:
        if db is not None:
            db.close()
        if write:
            from .writes import WriteDependencyError
            raise WriteDependencyError(config.state_root, "payload database", str(exc)) from exc
        raise HowHardError(f"payload store: cannot read {path}: {exc}") from exc
    return db


def _summary(value, created_at):
    context = value.get("context")
    context = context if isinstance(context, dict) else {}
    project = context.get("project_context")
    project = project if isinstance(project, dict) else {}
    selection = value.get("selection")
    selection = selection if isinstance(selection, dict) else {}
    chosen = value.get("chosen")
    chosen = chosen if isinstance(chosen, dict) else None
    small_selection = {key: selection[key] for key in ("objective", "retry_policy", "on_cap_unmet") if key in selection}
    return (created_at,
            json.dumps(chosen, sort_keys=True, separators=(",", ":")) if chosen is not None else None,
            json.dumps({"task_text": context.get("task_text"), "project_fingerprint": project.get("fingerprint")},
                       sort_keys=True, separators=(",", ":")),
            json.dumps(small_selection, sort_keys=True, separators=(",", ":")),
            value.get("task_type") if isinstance(value.get("task_type"), str) else None,
            value.get("risk_class") if isinstance(value.get("risk_class"), str) else None,
            int(selection["cap_met"]) if type(selection.get("cap_met")) is bool else None,
            int(selection["needs_review"]) if type(selection.get("needs_review")) is bool else None)


def _validate_payload(kind, identifier, blob):
    if kind not in TABLES or not ID.fullmatch(identifier):
        raise HowHardError(f"{kind}: invalid payload ID")
    try:
        value = json.loads(blob)
    except (ValueError, UnicodeError) as exc:
        raise HowHardError(f"{kind}: invalid JSON for {identifier}") from exc
    if not isinstance(value, dict) or digest(value) != identifier:
        raise HowHardError(f"{kind}: content does not match ID {identifier}")
    return value


def _insert(db, kind, identifier, blob, *, created_at=None):
    value = _validate_payload(kind, identifier, blob)
    existing = db.execute(f"SELECT payload FROM {kind} WHERE {kind[:-1]}_id=?", (identifier,)).fetchone()
    if existing is not None:
        old = zstd.ZstdDecompressor().decompress(existing[0])
        if old != blob:
            # An ID hashes the JSON value, so whitespace variants share the ID.
            # Keep both exact byte streams when two commits for one ID differ bytewise.
            db.execute("INSERT OR IGNORE INTO payload_variants VALUES (?, ?, ?, ?)",
                       (kind, identifier, sha256(old).hexdigest(), existing[0]))
            db.execute(f"UPDATE {kind} SET payload=? WHERE {kind[:-1]}_id=?",
                       (zstd.ZstdCompressor(level=3).compress(blob), identifier))
        return
    compressed = zstd.ZstdCompressor(level=3).compress(blob)
    if kind == "requests":
        db.execute("INSERT INTO requests VALUES (?, ?)", (identifier, compressed))
    else:
        created_at = created_at or datetime.now(timezone.utc).isoformat()
        db.execute("INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (identifier, compressed, *_summary(value, created_at)))


def put(config, kind, identifier, blob):
    """Commit an immutable payload; caller must hold storage.locked admission."""
    db = _connect(config, write=True)
    try:
        with db:
            _insert(db, kind, identifier, blob)
    except sqlite3.Error as exc:
        from .writes import WriteDependencyError
        raise WriteDependencyError(config.state_root, "payload database", str(exc)) from exc
    finally:
        db.close()


def get(config, kind, identifier):
    if kind not in TABLES or not ID.fullmatch(identifier):
        return None
    db = _connect(config)
    if db is None:
        return None
    try:
        row = db.execute(f"SELECT payload FROM {kind} WHERE {kind[:-1]}_id=?", (identifier,)).fetchone()
        return zstd.ZstdDecompressor().decompress(row[0]) if row else None
    except (sqlite3.Error, zstd.ZstdError) as exc:
        raise HowHardError(f"{kind}: corrupt payload store") from exc
    finally:
        db.close()


def exists(config, kind, identifier):
    if kind not in TABLES or not ID.fullmatch(identifier):
        return False
    db = _connect(config)
    if db is None:
        return False
    try:
        return db.execute(f"SELECT 1 FROM {kind} WHERE {kind[:-1]}_id=?", (identifier,)).fetchone() is not None
    finally:
        db.close()


def decision_summaries(config, *, task_type=None, risk_class=None, cap_met=None, needs_review=None, limit=None):
    """Read indexed decision metadata without inflating receipt payloads."""
    db = _connect(config)
    if db is None:
        return []
    where, values = [], []
    for column, value in (("task_type", task_type), ("risk_class", risk_class),
                          ("cap_met", cap_met), ("needs_review", needs_review)):
        if value is not None:
            where.append(f"{column}=?")
            values.append(int(value) if column in ("cap_met", "needs_review") else value)
    query = f"SELECT {', '.join(DECISION_COLUMNS)} FROM decisions"
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY created_at DESC, decision_id DESC"
    if limit is not None:
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        query += " LIMIT ?"
        values.append(limit)
    try:
        rows = db.execute(query, values).fetchall()
        return [{key: json.loads(value) if key in ("chosen", "context_summary", "selection") and value is not None
                 else bool(value) if key in ("cap_met", "needs_review") and value is not None else value
                 for key, value in zip(DECISION_COLUMNS, row)} for row in rows]
    finally:
        db.close()


def decision_ids(config):
    db = _connect(config)
    if db is None:
        return []
    try:
        return [row[0] for row in db.execute("SELECT decision_id FROM decisions ORDER BY decision_id")]
    finally:
        db.close()
