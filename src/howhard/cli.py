"""Command line entry point. The ordinary workflow needs no configuration."""

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .engine import Engine
from .config import load_config
from .validation import HowHardError


def common(command):
    command.add_argument("--config", metavar="PATH", help="optional advanced project settings")
    command.add_argument("--journal", metavar="PATH", help=argparse.SUPPRESS)
    command.add_argument("--state-dir", metavar="PATH", help="explicit durable state directory; current project capture stays separate")
    command.add_argument("--json", action="store_true")


def parser():
    root = argparse.ArgumentParser(prog="howhard", description="Choose a model and effort, then learn from the outcome. No setup required.")
    root.add_argument("--version", action="version", version=f"howhard {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("pick", "summary", "observe", "flush", "context", "decide", "retrieve", "report", "evidence", "paths"):
        command = commands.add_parser(name)
        common(command)
        if name in ("pick", "summary", "observe", "context"):
            command.add_argument("task_type")
            text_options = command.add_mutually_exclusive_group()
            text_options.add_argument("--text", help="task description")
            text_options.add_argument("--text-file", metavar="PATH", help="read the task description from a UTF-8 file")
            command.add_argument("--project-context", metavar="PATH", help="JSON snapshot or project parameter overrides")
        if name in ("pick", "summary", "context"):
            command.add_argument("--risk", help="optional risk policy override")
            command.add_argument("--candidates", required=True, metavar="FILE", help="JSON native model/effort inventory, with optional usd_per_mtok")
        if name == "evidence":
            command.add_argument("id")
            command.add_argument("--raw", action="store_true", help="return the exact recorded row rather than current effective evidence")
        if name == "pick":
            command.add_argument("--seed", type=int)
            command.add_argument("--save-decision", metavar="PATH", help="copy the decision receipt to a file")
        if name == "observe":
            command.add_argument("--native", action="store_true", help="admit a known actual native model/control for raw feedback without a decision")
            command.add_argument("model", nargs="?", help="model ID, or outcome when the first argument is a decision ID")
            command.add_argument("effort", nargs="?", help="exact registered effort/control identity")
            command.add_argument("outcome", nargs="?", choices=("clean", "loud", "silent"))
            command.add_argument("--decision", metavar="ID_OR_PATH", help="use the original task text and project snapshot")
            command.add_argument("--cause", choices=("depth", "capability", "unrecorded"))
            command.add_argument("--measurement-corrections", metavar="FILE", help="amendment-only JSON map of tokens/seconds/cost_usd replacements; null clears to unknown")
            for option, kind in (("tokens", int), ("seconds", float), ("cost-usd", float), ("ticket", str), ("note", str), ("amends", str), ("date", str), ("id", str)):
                command.add_argument(f"--{option}", type=kind)
        if name in ("decide", "retrieve"):
            command.add_argument("request_id")
            command.add_argument("--file", required=True, help="response/queries JSON file")
        if name == "decide":
            command.add_argument("--seed", type=int)
    memory = commands.add_parser("memory", help="inspect and manage persistent semantic memory")
    operations = memory.add_subparsers(dest="memory_command", required=True)
    for name in ("list", "read", "search", "add", "edit", "remove", "restore", "merge", "split", "history", "apply", "beliefs", "learn", "learn-context"):
        command = operations.add_parser(name)
        common(command)
        if name in ("read", "edit", "remove", "restore", "split", "learn", "learn-context"):
            command.add_argument("id")
        if name == "history":
            command.add_argument("id", nargs="?")
        if name == "read":
            command.add_argument("--version", type=int)
        if name in ("list", "search"):
            command.add_argument("--all", action="store_true", help="include removed memories")
        if name == "search":
            command.add_argument("query")
        if name == "merge":
            command.add_argument("ids", nargs="+")
        if name in ("add", "edit", "merge"):
            command.add_argument("--title", required=name != "edit")
            group = command.add_mutually_exclusive_group(required=name != "edit")
            group.add_argument("--body")
            group.add_argument("--body-file")
            command.add_argument("--scope", choices=("project", "shared"))
            command.add_argument("--evidence-id", action="append")
        if name in ("edit", "remove", "restore", "split"):
            command.add_argument("--expected-version", type=int)
        if name in ("split", "apply"):
            command.add_argument("--file", required=True)
        if name == "apply":
            command.add_argument("--expected-revision", type=int)
            command.add_argument("--actor", default="user")
        if name == "learn":
            command.add_argument("--file", help="apply a calling-agent writer response")
            command.add_argument("--expected-revision", type=int)
            command.add_argument("--event-key")
    return root


def render(result):
    if "cells" not in result:
        return json.dumps(result, indent=2)
    lines = []
    if "backend" in result:
        backend = result["backend"]
        lines.append(f"Backend: {backend.get('name', backend['mode'])}" + (f" ({backend['reason']})" if "reason" in backend else ""))
    if "semantic" in result:
        lines.append(result["semantic"]["reason"])
        lines.append("Intervals: approximate weighted LLM forecast envelope")
    if not result["cost_policy"]["worker_money_included"]:
        lines.append("Worker monetary cost is unknown for at least one candidate and omitted uniformly from comparison; J includes risk and time only.")
    if "chosen" in result:
        chosen = result["chosen"]
        lines.append(f"{chosen['model']} ({chosen['family']}) effort={chosen['effort']} seed={result['seed']}")
        lines.append(f"Silent cap: {'met' if result['cap_met'] else 'UNMET'}")
        lines.append(f"Decision: {result['decision_id']} (record with: howhard observe {result['decision_id']} clean)")
        if result["needs_review"]:
            lines.append("Independent review is required before release.")
    lines.append(f"Task: {result['task_type']}  Risk: {result['risk_class']}  Constants: {json.dumps(result['constants'], sort_keys=True)}")
    lines.append("family/effort  P_clean [95%]  P_loud [95%]  P_silent [95%]  run USD  time h  J USD  sampled J  admissible")
    for row in result["cells"]:
        fields = [f"{row['family']}/{row['effort']}"]
        for name in ("P_clean", "P_loud", "P_silent"):
            p = row[name]
            fields.append(f"{p['mean']:.3f} [{p['lower']:.3f}, {p['upper']:.3f}]")
        fields.extend((f"{row['C_run']:.4f}" if row["C_run"] is not None else "unknown", f"{row['T']:.4f}", f"{row['J']:.4f}",
                       f"{row['sampled_J']:.4f}" if "sampled_J" in row else "—", str(row["admissible"])))
        lines.append("  ".join(fields))
    for reason in result.get("why", []):
        lines.append(f"Why {reason['family']}/{reason['effort']} lost: {reason['message']}")
    return "\n".join(lines)


def read_json(path):
    try:
        return json.loads(Path(path).read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise HowHardError(f"{path}: expected valid JSON") from exc


def memory_command(engine, args):
    name, store = args.memory_command, engine.memory
    if name == "list":
        return {"memories": store.list(include_removed=args.all)}
    if name == "read":
        return store.read(args.id, version=args.version)
    if name == "search":
        return {"memories": store.search(args.query, include_removed=args.all)}
    if name == "history":
        return {"history": store.history(args.id)}
    if name == "beliefs":
        return engine.memory_beliefs()
    if name == "learn-context":
        return engine.memory_learning_context(args.id)
    if name == "learn":
        if args.file is None:
            return engine.memory_learn(args.id)
        if args.expected_revision is None or args.event_key is None:
            raise HowHardError("memory learn --file requires --expected-revision and --event-key from learn-context")
        return engine.apply_memory_learning(args.id, read_json(args.file), expected_revision=args.expected_revision, event_key=args.event_key)
    if name == "apply":
        return {"results": store.apply(read_json(args.file), expected_revision=args.expected_revision, actor=args.actor)}
    changes = {}
    if name in ("add", "edit", "merge"):
        if args.title is not None:
            changes["title"] = args.title
        if args.body is not None or args.body_file is not None:
            changes["body"] = args.body if args.body is not None else Path(args.body_file).read_text(encoding="utf-8")
        if args.scope is not None:
            changes["scope"] = args.scope
        if args.evidence_id is not None:
            changes["evidence_ids"] = args.evidence_id
    if name in ("edit", "remove", "restore", "split") and args.expected_version is not None:
        changes["expected_version"] = args.expected_version
    if name == "add":
        return store.add(**changes)
    if name == "merge":
        return store.merge(args.ids, **changes)
    if name == "split":
        return {"memories": store.split(args.id, read_json(args.file), **changes)}
    return getattr(store, name)(args.id, **changes)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config, directory=Path.cwd(), state_directory=args.state_dir)
        native_candidates = None
        if args.command in ("pick", "summary", "context"):
            from .native import preflight
            native_candidates = preflight(config, read_json(args.candidates))
        engine = Engine(config, journal_path=args.journal)
        context_options = {}
        if args.command in ("pick", "summary", "observe", "context"):
            if args.text_file is not None:
                context_options["task_text"] = Path(args.text_file).read_text(encoding="utf-8")
            elif args.text is not None:
                context_options["task_text"] = args.text
            if args.project_context is not None:
                try:
                    context_options["project_context"] = json.loads(Path(args.project_context).read_bytes())
                except (ValueError, UnicodeError) as exc:
                    raise HowHardError("project_context: expected valid JSON") from exc
        if args.command == "pick":
            result = engine.pick(args.task_type, risk=args.risk, seed=args.seed, native_candidates=native_candidates, **context_options)
            if args.save_decision is not None:
                engine.save_decision(result["decision_id"], args.save_decision)
        elif args.command == "summary":
            result = engine.summary(args.task_type, risk=args.risk, native_candidates=native_candidates, **context_options)
        elif args.command == "context":
            result = engine.prepare(args.task_type, risk=args.risk, native_candidates=native_candidates, **context_options)
        elif args.command == "evidence":
            result = engine.evidence(args.id, raw=args.raw)
        elif args.command == "paths":
            result = engine.paths
        elif args.command == "decide":
            result = engine.decide(args.request_id, read_json(args.file), seed=args.seed)
        elif args.command == "retrieve":
            result = engine.retrieve(args.request_id, read_json(args.file))
        elif args.command == "memory":
            result = memory_command(engine, args)
        elif args.command == "report":
            from .report import report
            result = report(engine)
        elif args.command == "observe":
            options = {key: getattr(args, key) for key in ("cause", "tokens", "seconds", "cost_usd", "ticket", "note", "amends", "date", "id") if getattr(args, key) is not None}
            if args.measurement_corrections is not None:
                options["measurement_corrections"] = read_json(args.measurement_corrections)
            if args.effort is None and args.outcome is None:
                if args.model not in ("clean", "loud", "silent") or args.decision is not None or context_options:
                    raise HowHardError("observe: expected DECISION_ID OUTCOME, or TASK_TYPE MODEL EFFORT OUTCOME")
                observed = engine.observe_decision(args.task_type, args.model, native=args.native, **options)
            elif args.model is not None and args.effort is not None and args.outcome is not None:
                observed = engine.observe(args.task_type, args.model, args.effort, args.outcome, decision=args.decision, native=args.native, **context_options, **options)
            else:
                raise HowHardError("observe: expected DECISION_ID OUTCOME, or TASK_TYPE MODEL EFFORT OUTCOME")
            result = {"observed": observed, "memory_update": engine.last_memory_update}
        elif args.command == "flush":
            ids = engine.flush()
            result = {"moved": len(ids), "ids": ids}
        print(json.dumps(result, allow_nan=False, sort_keys=True) if args.json else render(result))
        return 0
    except (HowHardError, OSError) as exc:
        print(f"howhard: {exc}", file=sys.stderr)
        return 2
