"""Write the authentic howhard-decision-v10 fixture beside this file with a 5.0.0 build.

The fixture is a receipt that a 5.0.0 build stored for a synthetic task, the recorded runs
it counted, and the native registry of its candidates. Nothing here is read by the tests;
it records how the fixture was made. Run it against a checkout of the v5.0.0 tag::

    PYTHONPATH=<v5.0.0 checkout>/src python make_fixture.py

Every input is synthetic: two fixture models, made-up runs, forecasts and prices, a fixed
project identity and a project directory that holds nothing but the store.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile

import howhard
from howhard import Engine
from howhard.context import canonical
from howhard.decisions import load

assert howhard.__version__ == "5.0.0", "make the v10 fixture with a 5.0.0 build"

HERE = Path(__file__).resolve().parent
IDENTITY = "0123456789abcdef0123456789abcdef"
TASK = "Repair rollback"
CANDIDATES = [{"model": "fixture/model-a", "effort": "high", "usd_per_mtok": 1.0},
              {"model": "fixture/model-b", "effort": "high", "usd_per_mtok": 3.0},
              {"model": "fixture/model-c", "effort": "high", "usd_per_mtok": 9.0}]
FORECASTS = {"fixture/model-a": {"clean": .90, "loud": .07, "silent": .03, "tokens": 20000, "seconds": 600,
                                 "loud_ci": [.03, .15], "silent_ci": [.01, .08]},
             "fixture/model-b": {"clean": .93, "loud": .05, "silent": .02, "tokens": 15000, "seconds": 400},
             "fixture/model-c": {"clean": .96, "loud": .03, "silent": .01, "tokens": 12000, "seconds": 300}}
RUNS = [("fixture/model-a", "clean", 500.), ("fixture/model-a", "loud", 700.), ("fixture/model-a", "loud", 650.),
        ("fixture/model-b", "clean", 380.), ("fixture/model-b", "silent", 420.), ("fixture/model-b", "clean", 300.)]


class Similar:
    """Every recorded run is a close precedent of the task; no model is loaded."""

    class reranker:
        @staticmethod
        def score(query, documents):
            return [8.] * len(documents)

    def search(self, records, query, limit=8, **kwargs):
        chosen = records[:limit]
        return {"records": deepcopy(chosen),
                "trace": {"seconds": 0, "selected": [{"key": r["key"], "score": 8.} for r in chosen]}}


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "project"
        (root / ".howhard").mkdir(parents=True)
        (root / ".howhard" / "project.json").write_text(json.dumps({"format": "howhard-project-v1", "id": IDENTITY}) + "\n")
        engine = Engine(directory=root, retrieval=Similar())
        for i, (model, outcome, seconds) in enumerate(RUNS):
            extra = {} if outcome == "clean" else {"cause": "capability"}
            engine.observe("chore", model, "high", outcome, native=True, task_text=TASK, id=f"run-{i}",
                           ticket=f"ticket-{i}", active_seconds=seconds, tokens=18000, **extra)
        engine.flush()
        pending = engine.prepare("task", TASK, native_candidates=CANDIDATES)
        cells = pending["baseline_request"]["cells"]
        order = [c["key"] for c in sorted(cells, key=lambda c: c["model"])]
        band = engine.decide(pending["request_id"], {"filter": {"order": order, "lower": order[0], "upper": order[-1]},
                                                     "memory_filters": [], "reason": "Synthetic fit band"})
        by_cell = {(c["family"], c["effort"]): c["model"] for c in band["baseline_request"]["cells"]}
        rows = [{"family": f, "effort": e, **FORECASTS[m]} for (f, e), m in by_cell.items()]
        result = engine.decide(band["request_id"], {"forecasts": rows, "memory_forecasts": [],
                                                    "reason": "Synthetic forecasts"}, seed=7)
        receipt = load(engine.config, result["decision_id"])
        assert receipt["format"] == "howhard-decision-v10" and result["semantic"]["sampling"]["enabled"]
        (HERE / "receipt.json").write_bytes(canonical(receipt) + b"\n")
        state = engine.config.state_root
        (HERE / "outcomes.jsonl").write_bytes((state / "outcomes.jsonl").read_bytes())
        (HERE / "native-models.json").write_bytes((state / "native-models.json").read_bytes())
        print(result["decision_id"], result["chosen"]["model"], file=sys.stderr)


if __name__ == "__main__":
    main()
