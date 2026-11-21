#!/usr/bin/env python3
"""Standalone one-time offline migration of pre-4.0 loose requests/*.json and decisions/*.json into payloads.sqlite3."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

import zstandard as zstd

from howhard.config import load_config
from howhard.payload_store import ID, TABLES, _connect, _insert, _validate_payload, get
from howhard.state import check
from howhard.storage import locked
from howhard.validation import HowHardError
from howhard.writes import WriteCancelled, WriteDependencyError, WriteError, WritePolicy


def parser():
    root = argparse.ArgumentParser(
        prog="migrate-payloads.py",
        description="Migrate pre-4.0 loose requests/*.json and decisions/*.json into payloads.sqlite3.",
    )
    root.add_argument("--config", metavar="PATH", help="optional advanced project settings")
    root.add_argument("--directory", metavar="PATH", default=Path.cwd(), type=Path,
                      help="project directory (default: current working directory)")
    root.add_argument("--state-dir", metavar="PATH", help="explicit durable state directory")
    root.add_argument("--write-timeout", type=float, default=30.0,
                      help="maximum seconds to wait per store admission (default: 30)")
    root.add_argument("--batch-size", type=int, default=64,
                      help="files per write-admitted SQLite transaction batch (default: 64)")
    root.add_argument("--json", action="store_true", help="emit JSON migration summary")
    return root


def _decision_timestamps(state_root):
    created = {}
    events = state_root / "decision-events.jsonl"
    if events.is_file():
        for line in events.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if (isinstance(event, dict)
                    and ID.fullmatch(str(event.get("decision_id", "")))
                    and isinstance(event.get("at_utc"), str)):
                created.setdefault(event["decision_id"], event["at_utc"])
    return created


def _check_claimed(config):
    check(config)
    if not (config.state_root / "store.json").is_file():
        raise HowHardError(f"state: refusing unclaimed store at {config.state_root}")
    for kind in TABLES:
        if (config.state_root / kind).is_symlink():
            raise HowHardError(f"{kind}: refusing legacy directory symlink")


def migrate_store(config, *, batch_size=64):
    if type(batch_size) is not int or batch_size < 1:
        raise HowHardError("batch_size: expected a positive integer")
    _check_claimed(config)
    created = _decision_timestamps(config.state_root)
    counts = {"requests": 0, "decisions": 0}
    raw_bytes = 0
    compressed_bytes = 0

    for kind in ("requests", "decisions"):
        directory = config.state_root / kind
        if not directory.is_dir():
            continue
        files = sorted(directory.glob("*.json"))
        if not files:
            with locked(config, operation="payload migration"):
                _check_claimed(config)
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
            continue
        for start in range(0, len(files), batch_size):
            chunk = files[start:start + batch_size]
            with locked(config, operation="payload migration"):
                _check_claimed(config)
                batch = []
                for path in chunk:
                    if path.is_symlink() or not path.is_file():
                        raise HowHardError(f"{kind}: refusing non-regular payload {path}")
                    blob = path.read_bytes()
                    _validate_payload(kind, path.stem, blob)
                    timestamp = (
                        created.get(path.stem)
                        or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
                    )
                    batch.append((path, blob, timestamp))
                db = _connect(config, write=True)
                compressed_lengths = {}
                try:
                    with db:
                        for path, blob, timestamp in batch:
                            _insert(db, kind, path.stem, blob, created_at=timestamp)
                        for path, blob, _ in batch:
                            row = db.execute(
                                f"SELECT payload FROM {kind} WHERE {kind[:-1]}_id=?",
                                (path.stem,),
                            ).fetchone()
                            if row is None or zstd.ZstdDecompressor().decompress(row[0]) != blob:
                                raise HowHardError(f"{kind}: round-trip mismatch for {path}")
                            compressed_lengths[path] = len(row[0])
                except sqlite3.Error as exc:
                    raise WriteDependencyError(config.state_root, "payload migration", str(exc)) from exc
                finally:
                    db.close()
                for path, blob, _ in batch:
                    if get(config, kind, path.stem) != blob or path.read_bytes() != blob:
                        raise HowHardError(f"{kind}: post-commit mismatch for {path}")
                    path.unlink()
                    counts[kind] += 1
                    raw_bytes += len(blob)
                    compressed_bytes += compressed_lengths[path]
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()

    return {
        "store": str(config.state_root),
        "requests": counts["requests"],
        "decisions": counts["decisions"],
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
    }


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config, directory=args.directory, state_directory=args.state_dir)
        config.write_policy = WritePolicy(args.write_timeout)
        summary = migrate_store(config, batch_size=args.batch_size)
        if args.json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print(
                f"Migrated {summary['store']}: requests={summary['requests']} "
                f"decisions={summary['decisions']} raw_bytes={summary['raw_bytes']} "
                f"compressed_bytes={summary['compressed_bytes']}"
            )
        return 0
    except KeyboardInterrupt:
        error = WriteCancelled(args.state_dir or args.directory, "payload migration", "interrupted")
        print(json.dumps({"error": error.as_dict()}) if args.json else f"migrate-payloads: {error}", file=sys.stderr)
        return 130
    except (HowHardError, OSError) as exc:
        if args.json and isinstance(exc, WriteError):
            print(json.dumps({"error": exc.as_dict()}), file=sys.stderr)
            return 2
        print(f"migrate-payloads: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
