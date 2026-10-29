"""Command line entry point. The ordinary workflow needs no configuration."""

import argparse
import json
import sys

from . import __version__
from .config import EFFORTS
from .engine import Engine
from .validation import HowHardError


def parser():
    root = argparse.ArgumentParser(prog="howhard", description="Choose a model and effort, then learn from the outcome. No setup required.")
    root.add_argument("--version", action="version", version=f"howhard {__version__}")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("pick", "summary", "observe", "flush", "export-prior", "import-prior"):
        command = commands.add_parser(name)
        command.add_argument("--config", metavar="PATH", help="optional advanced project settings")
        command.add_argument("--cache", metavar="PATH", help=argparse.SUPPRESS)
        command.add_argument("--journal", metavar="PATH", help=argparse.SUPPRESS)
        command.add_argument("--json", action="store_true")
        if name in ("pick", "summary", "observe"):
            command.add_argument("task_type")
        if name in ("pick", "summary"):
            command.add_argument("--risk", help="optional risk policy override")
        if name == "pick":
            command.add_argument("--seed", type=int)
        if name == "observe":
            command.add_argument("model")
            command.add_argument("effort", choices=(*EFFORTS, "model_default"))
            command.add_argument("outcome", choices=("clean", "loud", "silent"))
            command.add_argument("--cause", choices=("depth", "capability", "unrecorded"))
            for option, kind in (("tokens", int), ("seconds", float), ("cost-usd", float), ("ticket", str), ("note", str), ("amends", str), ("date", str), ("id", str)):
                command.add_argument(f"--{option}", type=kind)
        if name == "export-prior":
            command.add_argument("--out", required=True, metavar="FILE")
        if name == "import-prior":
            command.add_argument("file")
    return root


def render(result):
    if "cells" not in result:
        return json.dumps(result, indent=2)
    lines = []
    if "chosen" in result:
        chosen = result["chosen"]
        lines.append(f"{chosen['model']} ({chosen['family']}) effort={chosen['effort']} seed={result['seed']}")
        lines.append(f"Silent cap: {'met' if result['cap_met'] else 'UNMET'}")
        if result["needs_review"]:
            lines.append("Independent review is required before release.")
    lines.append(f"Task: {result['task_type']}  Risk: {result['risk_class']}  Constants: {json.dumps(result['constants'], sort_keys=True)}")
    lines.append("family/effort  P_clean [95%]  P_loud [95%]  P_silent [95%]  run USD  time h  J USD  sampled J  admissible")
    for row in result["cells"]:
        fields = [f"{row['family']}/{row['effort']}"]
        for name in ("P_clean", "P_loud", "P_silent"):
            p = row[name]
            fields.append(f"{p['mean']:.3f} [{p['lower']:.3f}, {p['upper']:.3f}]")
        fields.extend((f"{row['C_run']:.4f}", f"{row['T']:.4f}", f"{row['J']:.4f}",
                       f"{row['sampled_J']:.4f}" if "sampled_J" in row else "—", str(row["admissible"])))
        lines.append("  ".join(fields))
    for reason in result.get("why", []):
        lines.append(f"Why {reason['family']}/{reason['effort']} lost: {reason['message']}")
    stats = result["fit"]
    lines.append(f"Fit: {stats['kind']}, {stats['iterations']} Newton iterations")
    return "\n".join(lines)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        engine = Engine(args.config, cache_path=args.cache, journal_path=args.journal)
        if args.command == "pick":
            result = engine.pick(args.task_type, risk=args.risk, seed=args.seed)
        elif args.command == "summary":
            result = engine.summary(args.task_type, risk=args.risk)
        elif args.command == "observe":
            options = {key: getattr(args, key) for key in ("cause", "tokens", "seconds", "cost_usd", "ticket", "note", "amends", "date", "id") if getattr(args, key) is not None}
            result = {"observed": engine.observe(args.task_type, args.model, args.effort, args.outcome, **options), "fit": engine.posterior.statistics}
        elif args.command == "flush":
            ids = engine.flush()
            result = {"moved": len(ids), "ids": ids, "fit": engine.posterior.statistics}
        elif args.command == "export-prior":
            engine.export_prior(args.out)
            result = {"exported": args.out}
        else:
            result = {"imported": str(engine.import_prior(args.file))}
        print(json.dumps(result, allow_nan=False, sort_keys=True) if args.json else render(result))
        return 0
    except (HowHardError, OSError) as exc:
        print(f"howhard: {exc}", file=sys.stderr)
        return 2
