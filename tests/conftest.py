import json

import pytest

from howhard import load_config
from howhard.storage import effective_rows


@pytest.fixture
def project(tmp_path):
    count = 0

    def create(*, settings="", rows=(), name=None):
        nonlocal count
        count += 1
        directory = tmp_path / (name or f"project-{count}")
        directory.mkdir()
        path = directory / "howhard.toml"
        # Synthetic economics: rework dominates token cost, while escaped defects
        # cost twice as much. The cheap class permits no expected-cost exploration.
        policy = '\n[classes.critical]\nC_loud = 1000.0\nC_silent = 2000.0\nlambda_per_hour = 0.0\n[classes.standard]\nC_loud = 1000.0\nC_silent = 2000.0\nlambda_per_hour = 0.0\nepsilon = 0.5\n[classes.routine]\nexplore_within = 0.0\n'
        path.write_text('on_cap_unmet = "flag-review"\n' + settings + policy + '\n[task_types]\nX = "critical"\nY = "standard"\nZ = "critical"\ncheap = "routine"\n')
        config = load_config(path)
        log = config.path(config.data["logs"][0]["path"])
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return config
    return create


def row(i, *, task="X", family="opus", effort="high", outcome="clean", **kwargs):
    return {"id": f"run-{i}", "date": "2026-10-29", "ticket": None, "task_type": task,
            "risk_class": "critical" if task in ("X", "Z") else "routine" if task == "cheap" else "standard",
            "model": {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5", "fable": "claude-fable-5-1"}[family],
            "effort": effort, "default_effort": None, "outcome": outcome,
            "cause": None if outcome == "clean" else "unrecorded", "tokens": None,
            "seconds": None, "cost_usd": None, "note": "", "amends": None, **kwargs}


def fitted_rows(config, rows):
    return effective_rows(config, [{**r, "project": config.data["project"]} for r in rows])


def cell(summary, family, effort):
    return next(c for c in summary["cells"] if (c["family"], c["effort"]) == (family, effort))


def frequency(engine, task, target, candidates):
    return sum((p["chosen"]["family"], p["chosen"]["effort"]) == target
               for p in (engine.pick(task, seed=s, candidates=candidates) for s in range(200))) / 200
