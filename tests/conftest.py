import json

import pytest

from howhard import load_config
from howhard.storage import effective_rows
from howhard.context import FORMAT


@pytest.fixture(autouse=True)
def offline_llm_by_default(monkeypatch, tmp_path_factory):
    # No test invokes an authenticated model. The subprocess adapter is a protocol double.
    import sys
    monkeypatch.delenv("HOWHARD_LLM", raising=False)
    script = tmp_path_factory.mktemp("fixture-model") / "provider.py"
    script.write_text('''import json,sys
r=json.load(sys.stdin)["request"]
if r["kind"]=="learn":
 print(json.dumps({"operations":[],"reason":"fixture"}))
elif r["kind"]=="filter":
 if r.get("phase")=="baseline":
  levels={e:i for i,e in enumerate(["low","medium","high","xhigh","max","ultra"])}
  cells=sorted(r["cells"],key=lambda c:(c["model"],levels.get(c["effort"],99)))
  order=[c["key"] for c in cells]
  print(json.dumps({"filter":{"order":order,"lower":order[0],"upper":order[min(len(order),r["filter_policy"]["max_forecast_cells"])-1]},"reason":"fixture filter"}))
 else:
  print(json.dumps({"memory_filters":[],"reason":"fixture"}))
elif r.get("phase")=="baseline":
 print(json.dumps({"forecasts":[{"family":c["family"],"effort":c["effort"],"clean":.5,"loud":.45,"silent":.05,"tokens":10000,"seconds":30} for c in r["cells"]],"reason":"fixture"}))
else:
 print(json.dumps({"memory_forecasts":[],"reason":"fixture"}))
''')
    monkeypatch.setenv("HOWHARD_LLM_COMMAND", json.dumps([sys.executable, str(script)]))


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
            "effort": effort, "outcome": outcome,
            "observed_at": "2026-10-29T12:00:00+00:00", "task_text": "",
            "project_context": None, "context_encoder": FORMAT,
            "cause": None if outcome == "clean" else "unrecorded", "tokens": None,
            "seconds": None, "cost_usd": None, "note": "", "amends": None, **kwargs}


def cell(summary, family, effort):
    return next(c for c in summary["cells"] if (c["family"], c["effort"]) == (family, effort))


# Synthetic host inventory and prices; these are fixture inputs, not provider claims.
MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5",
          "opus": "claude-opus-5-5", "fable": "claude-fable-5-1"}


def inventory(cells=None):
    cells = cells if cells is not None else [(family, effort) for family in MODELS
              for effort in ("low", "medium", "high", "xhigh", "max")]
    prices = {"haiku": .1, "sonnet": .4, "opus": .6, "fable": 1.5}
    return [{"model": MODELS[family], "effort": effort, "usd_per_mtok": prices[family]}
            for family, effort in cells]


@pytest.fixture
def candidate_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("host-inventory") / "candidates.json"
    path.write_text(json.dumps(inventory()))
    return path



def forecast_request(engine, pending):
    """Advance an explicit synthetic filter before testing calibrated forecasting."""
    if pending["stage"] == "forecast":
        return pending
    return engine.decide(pending["request_id"], fixture_filter(pending["baseline_request"]))


def fixture_filter(request):
    from howhard.capability import EFFORT_ORDER
    cells = sorted(request["cells"], key=lambda c: (c["model"], EFFORT_ORDER.get(c["effort"], 99)))
    order = [c["key"] for c in cells]
    return {"filter": {"order": order, "lower": order[0],
                       "upper": order[min(len(order), request["filter_policy"]["max_forecast_cells"])-1]},
            "memory_filters": [], "reason": "Synthetic task-local ordinal prior"}


@pytest.fixture(autouse=True)
def offline_capability_relevance(request, monkeypatch):
    # Existing tests without a supplied retrieval double must not download weights
    # for the learned reader's new absolute relevance pass on small corpora.
    if request.path.name not in {"test_fast_reranking.py", "test_retrieval_config.py", "test_retrieval.py", "test_native_startup.py"}:
        monkeypatch.setattr("howhard.retrieval.FastEmbedReranker.score", lambda self, query, documents: [8.] * len(documents))
