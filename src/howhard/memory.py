"""Transactional, versioned semantic memory. Removing a memory preserves evidence."""

from contextlib import contextmanager
import json
import secrets
import sqlite3

from .validation import HowHardError, keys, text


def strings(value, where):
    if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
        raise HowHardError(f"{where}: expected a list of non-empty strings")
    if len(value) != len(set(value)):
        raise HowHardError(f"{where}: duplicate values")
    return value


class MemoryStore:
    def __init__(self, config, evidence=None, retrieval=None):
        self.path = config.state_root / "memory.sqlite3"
        self._evidence = evidence or (lambda: set())
        if retrieval is None:
            from .retrieval import RetrievalPipeline
            retrieval = RetrievalPipeline(config.state_root)
        self.retrieval = retrieval

    @contextmanager
    def _db(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            if db.execute("PRAGMA user_version").fetchone()[0] not in (0, 1):
                raise HowHardError("memory database: unsupported schema version")
            db.executescript("""
                PRAGMA foreign_keys = ON;
                PRAGMA user_version = 1;
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY, version INTEGER NOT NULL, active INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS versions (
                    id TEXT NOT NULL, version INTEGER NOT NULL, title TEXT NOT NULL,
                    body TEXT NOT NULL, scope TEXT NOT NULL, evidence TEXT NOT NULL,
                    parents TEXT NOT NULL, PRIMARY KEY(id, version));
                CREATE TABLE IF NOT EXISTS events (
                    revision INTEGER PRIMARY KEY AUTOINCREMENT, actor TEXT NOT NULL,
                    targets TEXT NOT NULL, operation TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS learned (
                    event_key TEXT PRIMARY KEY, revision INTEGER NOT NULL);
            """)
            yield db
        except sqlite3.Error as exc:
            db.rollback()
            raise HowHardError(f"memory database: {exc}") from exc
        finally:
            db.close()

    @staticmethod
    def _revision(db):
        return db.execute("SELECT COALESCE(MAX(revision), 0) FROM events").fetchone()[0]

    @staticmethod
    def _read(db, memory_id, version=None):
        text(memory_id, "memory.id")
        current = db.execute("SELECT version, active FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if current is None:
            raise HowHardError(f"memory.id: unknown memory {memory_id!r}")
        version = current["version"] if version is None else version
        if type(version) is not int or version < 1:
            raise HowHardError("memory.version: expected positive integer")
        row = db.execute("SELECT * FROM versions WHERE id = ? AND version = ?", (memory_id, version)).fetchone()
        if row is None:
            raise HowHardError(f"memory.version: unknown version {version} for {memory_id}")
        return {"id": memory_id, "version": version, "active": bool(current["active"]),
                "title": row["title"], "body": row["body"], "scope": row["scope"],
                "evidence_ids": json.loads(row["evidence"]), "parents": json.loads(row["parents"])}

    def read(self, memory_id, *, version=None):
        with self._db() as db:
            return self._read(db, memory_id, version)

    def snapshot(self, *, include_removed=False):
        with self._db() as db:
            db.execute("BEGIN")
            clause = "" if include_removed else " WHERE active = 1"
            ids = [row[0] for row in db.execute("SELECT id FROM memories" + clause + " ORDER BY id")]
            return {"revision": self._revision(db), "memories": [self._read(db, id) for id in ids]}

    def list(self, *, include_removed=False):
        return self.snapshot(include_removed=include_removed)["memories"]

    def history(self, memory_id):
        with self._db() as db:
            if memory_id is not None:
                self._read(db, memory_id)
            return [{"revision": row["revision"], "actor": row["actor"], "operation": json.loads(row["operation"])}
                    for row in db.execute("SELECT * FROM events ORDER BY revision") if memory_id is None or memory_id in json.loads(row["targets"])]

    def search(self, query, *, limit=8, include_removed=False):
        from .retrieval import records
        result = self.retrieval.search(records(memories=self.list(include_removed=include_removed)), query, limit=limit)
        return [r["value"] for r in result["records"]]

    def learned(self, event_key):
        with self._db() as db:
            return db.execute("SELECT 1 FROM learned WHERE event_key = ?", (event_key,)).fetchone() is not None

    def apply(self, operations, *, expected_revision=None, actor="user", event_key=None):
        if not isinstance(operations, list):
            raise HowHardError("memory.operations: expected a list")
        text(actor, "memory.actor")
        if expected_revision is not None and (type(expected_revision) is not int or expected_revision < 0):
            raise HowHardError("memory.expected_revision: expected non-negative integer")
        with self._db() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                revision = self._revision(db)
                if event_key is not None and db.execute("SELECT 1 FROM learned WHERE event_key = ?", (event_key,)).fetchone():
                    return []
                if expected_revision is not None and expected_revision != revision:
                    raise HowHardError("memory: stale revision; read current memory before editing")
                results = [self._apply_one(db, operation, actor) for operation in operations]
                if event_key is not None:
                    text(event_key, "memory.event_key")
                    db.execute("INSERT INTO learned VALUES (?, ?)", (event_key, self._revision(db)))
                db.commit()
                return results
            except BaseException:
                db.rollback()
                raise

    def _content(self, title, body, scope, evidence_ids):
        text(title, "memory.title")
        text(body, "memory.body")
        if scope not in ("project", "shared"):
            raise HowHardError("memory.scope: expected project or shared")
        evidence_ids = strings(evidence_ids, "memory.evidence_ids")
        unknown = set(evidence_ids) - set(self._evidence())
        if unknown:
            raise HowHardError(f"memory.evidence_ids: unknown evidence {sorted(unknown)}")

    def _new(self, db, title, body, scope="project", evidence_ids=None, parents=None):
        evidence_ids, parents = [] if evidence_ids is None else evidence_ids, [] if parents is None else parents
        self._content(title, body, scope, evidence_ids)
        memory_id = secrets.token_hex(6)
        db.execute("INSERT INTO memories VALUES (?, 1, 1)", (memory_id,))
        db.execute("INSERT INTO versions VALUES (?, 1, ?, ?, ?, ?, ?)",
                   (memory_id, title, body, scope, json.dumps(evidence_ids), json.dumps(parents)))
        return self._read(db, memory_id)

    def _apply_one(self, db, operation, actor):
        keys(operation, ["op"], ["id", "ids", "title", "body", "scope", "evidence_ids", "expected_version", "parts", "reason"], "memory.operation")
        op = operation["op"]
        if op not in ("add", "edit", "remove", "restore", "merge", "split"):
            raise HowHardError(f"memory.operation.op: unknown operation {op!r}")
        required = {"add": ["title", "body"], "edit": ["id"], "remove": ["id"], "restore": ["id"], "merge": ["ids", "title", "body"], "split": ["id", "parts"]}[op]
        optional = {"add": ["scope", "evidence_ids"], "edit": ["title", "body", "scope", "evidence_ids", "expected_version"], "remove": ["expected_version"], "restore": ["expected_version"], "merge": ["scope", "evidence_ids"], "split": ["expected_version"]}[op]
        keys(operation, ["op", *required], ["reason", *optional], "memory.operation")
        if "reason" in operation:
            text(operation["reason"], "memory.reason")
        targets = []
        if op == "add":
            result = self._new(db, **{k: operation[k] for k in ("title", "body", "scope", "evidence_ids") if k in operation})
            targets = [result["id"]]
        elif op == "merge":
            ids = strings(operation["ids"], "memory.ids")
            if len(ids) < 2:
                raise HowHardError("memory.merge: at least two distinct memories are required")
            entries = [self._read(db, id) for id in ids]
            if not all(e["active"] for e in entries):
                raise HowHardError("memory.merge: source memories must be active")
            evidence = operation.get("evidence_ids", sorted({id for e in entries for id in e["evidence_ids"]}))
            result = self._new(db, operation["title"], operation["body"], operation.get("scope", entries[0]["scope"]), evidence, ids)
            db.executemany("UPDATE memories SET active = 0 WHERE id = ?", [(id,) for id in ids])
            targets = [*ids, result["id"]]
        else:
            entry = self._read(db, operation["id"])
            if "expected_version" in operation and (type(operation["expected_version"]) is not int or operation["expected_version"] != entry["version"]):
                raise HowHardError("memory: stale content version")
            targets = [entry["id"]]
            if op == "edit":
                if not any(k in operation for k in ("title", "body", "scope", "evidence_ids")):
                    raise HowHardError("memory.edit: provide a content field")
                values = {k: operation.get(k, entry[k]) for k in ("title", "body", "scope", "evidence_ids")}
                self._content(**values)
                db.execute("INSERT INTO versions VALUES (?, ?, ?, ?, ?, ?, ?)", (entry["id"], entry["version"] + 1, values["title"], values["body"], values["scope"], json.dumps(values["evidence_ids"]), json.dumps(entry["parents"])))
                db.execute("UPDATE memories SET version = version + 1 WHERE id = ?", (entry["id"],))
                result = self._read(db, entry["id"])
            elif op in ("remove", "restore"):
                db.execute("UPDATE memories SET active = ? WHERE id = ?", (int(op == "restore"), entry["id"]))
                result = self._read(db, entry["id"])
            else:
                parts = operation["parts"]
                if not entry["active"] or not isinstance(parts, list) or len(parts) < 2:
                    raise HowHardError("memory.split: active source and at least two parts required")
                result = []
                for part in parts:
                    keys(part, ["title", "body"], ["scope", "evidence_ids"], "memory.split.part")
                    result.append(self._new(db, part["title"], part["body"], part.get("scope", entry["scope"]), part.get("evidence_ids", entry["evidence_ids"]), [entry["id"]]))
                db.execute("UPDATE memories SET active = 0 WHERE id = ?", (entry["id"],))
                targets += [e["id"] for e in result]
        db.execute("INSERT INTO events(actor, targets, operation) VALUES (?, ?, ?)", (actor, json.dumps(targets), json.dumps(operation, allow_nan=False)))
        return result

    def add(self, title, body, *, scope="project", evidence_ids=None):
        return self.apply([dict(op="add", title=title, body=body, scope=scope, evidence_ids=[] if evidence_ids is None else evidence_ids)])[0]

    def edit(self, memory_id, **changes):
        return self.apply([dict(op="edit", id=memory_id, **changes)])[0]

    def remove(self, memory_id, **options):
        return self.apply([dict(op="remove", id=memory_id, **options)])[0]

    def restore(self, memory_id, **options):
        return self.apply([dict(op="restore", id=memory_id, **options)])[0]

    def merge(self, ids, *, title, body, **options):
        return self.apply([dict(op="merge", ids=ids, title=title, body=body, **options)])[0]

    def split(self, memory_id, parts, **options):
        return self.apply([dict(op="split", id=memory_id, parts=parts, **options)])[0]
