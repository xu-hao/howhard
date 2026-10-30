"""The ten named behavioral acceptance tests from issue #9."""

from copy import deepcopy
import json
from time import perf_counter
import numpy as np
import pytest

from howhard import Engine
from howhard.outcomes import fit
from howhard.prior import build_prior
from howhard.context import FORMAT as ENCODER
from howhard.transfer import FORMAT, allowed_labels, shared
from conftest import inventory, cell, fitted_rows, frequency, row


def test_effort_sensitive_type_prefers_higher_effort_same_model(project):
    rows = [row(i, outcome="loud") for i in range(6)]
    rows += [row(i + 6, family="fable", outcome="loud") for i in range(6)]
    rows += [row(i + 12, effort="xhigh") for i in range(6)]
    engine = Engine(project(rows=rows))
    posterior, prior = engine.posterior.outcome, engine.prior.outcome
    def mean(label):
        return posterior.mean[posterior.index(label)]
    assert mean("loud.u.X") > prior.mean[prior.index("loud.u.X")]
    assert abs(mean("loud.w.X.fable")) <= .3
    assert mean("loud.v.X.high") > 0
    summary = engine.summary("X", native_candidates=inventory())
    assert cell(summary, "opus", "xhigh")["J"] < cell(summary, "fable", "high")["J"]
    candidates = [("opus", "xhigh"), ("fable", "high")]
    assert frequency(engine, "X", candidates[0], candidates) >= .8


def test_interaction_lower_model_at_higher_effort_can_win(project):
    rows = [row(i, task="Y", family="sonnet", effort="xhigh") for i in range(10)]
    rows += [row(i + 10, task="Y", effort="medium", outcome="loud") for i in range(10)]
    engine = Engine(project(rows=rows))
    summary = engine.summary("Y", native_candidates=inventory())
    a, b = cell(summary, "sonnet", "xhigh"), cell(summary, "opus", "medium")
    assert a["J"] < b["J"]
    assert a["P_clean"]["mean"] > b["P_clean"]["mean"]
    candidates = [("sonnet", "xhigh"), ("opus", "medium")]
    assert frequency(engine, "Y", candidates[0], candidates) >= .8


def test_silent_cap_excludes_a_cell(project):
    rows = [row(i, task="Z", family="haiku", outcome="silent" if i < 3 else "clean") for i in range(10)]
    config = project(rows=rows)
    engine = Engine(config)
    assert not cell(engine.summary("Z", native_candidates=inventory()), "haiku", "high")["admissible"]
    # Add evidence in batches until the other cell satisfies the cap.
    log = config.path(config.data["logs"][0]["path"])
    for n in (20, 50, 100, 200):
        amended = rows + [row(i + 10, task="Z") for i in range(n)]
        log.write_text("".join(json.dumps(r) + "\n" for r in amended))
        engine.refresh()
        if cell(engine.summary("Z", native_candidates=inventory()), "opus", "high")["admissible"]:
            break
    assert cell(engine.summary("Z", native_candidates=inventory()), "opus", "high")["admissible"]
    assert not cell(engine.summary("Z", native_candidates=inventory()), "haiku", "high")["admissible"]
    candidates = [("haiku", "high"), ("opus", "high")]
    assert frequency(engine, "Z", candidates[1], candidates) == 1


def test_a_cheap_cell_wins_when_adequate(project):
    rows = [row(i, task="cheap", family="haiku", effort="low") for i in range(30)]
    engine = Engine(project(rows=rows))
    assert min(engine.summary("cheap", native_candidates=inventory())["cells"], key=lambda c: c["J"])["family"] == "haiku"
    assert min(engine.summary("cheap", native_candidates=inventory())["cells"], key=lambda c: c["J"])["effort"] == "low"
    picks = [engine.pick("cheap", seed=s, native_candidates=inventory())["chosen"] for s in range(200)]
    assert sum((p["family"], p["effort"]) == ("haiku", "low") for p in picks) > 100


def test_the_posterior_updates(project):
    engine = Engine(project())
    before = cell(engine.summary("X", native_candidates=inventory()), "opus", "high")["P_loud"]["mean"]
    for i in range(4):
        engine.observe("X", "claude-opus-5-5", "high", "loud", id=f"loud-{i}")
    loud = cell(engine.summary("X", native_candidates=inventory()), "opus", "high")["P_loud"]["mean"]
    assert loud > before
    for i in range(8):
        engine.observe("X", "claude-opus-5-5", "high", "clean", id=f"clean-{i}")
    assert cell(engine.summary("X", native_candidates=inventory()), "opus", "high")["P_loud"]["mean"] < loud


def test_thompson_draws_reproducible_under_seed(project):
    engine = Engine(project())
    assert engine.pick("Y", seed=37, native_candidates=inventory()) == engine.pick("Y", seed=37, native_candidates=inventory())
    picks = {(p["chosen"]["family"], p["chosen"]["effort"])
             for p in (engine.pick("Y", seed=s, native_candidates=inventory()) for s in range(100))}
    assert len(picks) > 1
    fresh = engine.pick("Y", native_candidates=inventory())
    assert engine.pick("Y", seed=fresh["seed"], native_candidates=inventory()) == fresh


def test_transfer_from_synthetic_project_a_to_b(project, tmp_path):
    rows = [row(i, family="sonnet", ticket="private-ticket", note="private-note") for i in range(60)]
    rows += [row(i + 60, outcome="loud") for i in range(60)]
    a = Engine(project(rows=rows, settings='project = "secret-project-a"\n[[logs]]\npath = "runs.jsonl"\nproject = "secret-project-a"\n'))
    path = tmp_path / "export.json"
    a.export_prior(path)
    b = Engine(project(settings='project = "project-b"\n[[logs]]\npath = "runs.jsonl"\nproject = "project-b"\n'))
    before = b.summary("X", native_candidates=inventory())
    assert cell(before, "sonnet", "high")["P_loud"]["mean"] > cell(before, "opus", "high")["P_loud"]["mean"]
    b.import_prior(path)
    after = b.summary("X", native_candidates=inventory())
    assert cell(after, "sonnet", "high")["P_loud"]["mean"] < cell(after, "opus", "high")["P_loud"]["mean"]
    candidates = [("sonnet", "high"), ("opus", "high")]
    assert frequency(b, "X", candidates[0], candidates) >= .8
    for kind in ("loud", "silent"):
        for label in (f"{kind}.pi.project-b", f"{kind}.pi_task.project-b.X"):
            i = b.posterior.outcome.index(label)
            assert b.posterior.outcome.mean[i] == 0
            assert b.posterior.outcome.covariance[i, i] == pytest.approx(b.prior.outcome.covariance[i, i], abs=1e-12)


def test_export_contains_generic_labels_and_numbers_only(project, tmp_path):
    engine = Engine(project(rows=[row("private-row", ticket="private-ticket", note="private-note")],
                            settings='project = "secret-project"\n[[logs]]\npath = "private-path.jsonl"\nproject = "secret-project"\n'))
    path = tmp_path / "export.json"
    exported = engine.export_prior(path)
    labels = allowed_labels(exported["vocabularies"])
    allowed = {FORMAT, ENCODER, "features", "context_encoder", "format", "vocabularies", "blocks", "noise_sds", "families", "efforts", "task_types", "labels", "mean", "covariance", "outcome", "tokens", "time"}
    allowed.update(v for names in exported["vocabularies"].values() for v in names)
    allowed.update(label for group in labels.values() for label in group)
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert key in allowed
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            assert value in allowed
        else:
            assert type(value) in (int, float)
    walk(exported)
    raw = path.read_text()
    for forbidden in ("secret-project", "private-row", "private-ticket", "private-note", "private-path", "claude-opus-5-5", "2026-10-29"):
        assert forbidden not in raw


def test_cache_zero_fitting_warm_equivalence_and_prior_invalidation(project, monkeypatch):
    config = project(rows=[row(i, outcome="loud" if i < 4 else "clean") for i in range(20)])
    full = Engine(config)
    from howhard import engine as module
    original = module.fit
    def forbidden(*args, **kwargs):
        pytest.fail("a cache hit fitted the outcome model")
    monkeypatch.setattr(module, "fit", forbidden)
    cached = Engine(config)
    cached.pick("X", seed=42, native_candidates=inventory())
    assert cached.posterior.statistics == {"kind": "cached", "iterations": 0, "final_step": 0.0}
    monkeypatch.setattr(module, "fit", original)
    cached.observe("X", "claude-opus-5-5", "high", "clean")
    warm = deepcopy(cached.posterior)
    assert warm.statistics["kind"] == "warm"
    cached.refresh(full=True)
    for block in ("outcome", "tokens", "time"):
        assert np.allclose(getattr(warm, block).mean, getattr(cached.posterior, block).mean, atol=1e-6, rtol=0)
        assert np.allclose(getattr(warm, block).covariance, getattr(cached.posterior, block).covariance, atol=1e-6, rtol=0)
    assert warm.statistics["iterations"] < cached.posterior.statistics["iterations"]
    config.prior["outcome"]["loud"]["a"]["mean"] += .01
    # In real use the changed bytes are reloaded; include this synthetic effective override.
    config.data["prior"]["overrides"] = {"outcome": {"loud": {"a": config.prior["outcome"]["loud"]["a"]}}}
    assert Engine(config).posterior.statistics["kind"] == "full"


def test_refit_1000_rows_under_two_seconds(project):
    config = project()
    rng = np.random.default_rng(101)
    types = list(config.prior["taxonomy"])
    families = config.prior["families"]
    efforts = config.prior["efforts"]
    rows = [row(i, task=types[i % 16], family=families[int(rng.integers(4))],
                effort=efforts[int(rng.integers(5))], outcome=str(rng.choice(["clean", "loud", "silent"], p=[.65, .3, .05]))) for i in range(1000)]
    effective = fitted_rows(config, rows)
    prior = build_prior(config, effective)
    start = perf_counter()
    posterior = fit(prior, effective)
    elapsed = perf_counter() - start
    assert posterior.statistics["iterations"] < 100
    assert elapsed < 2, f"1,000-row fit took {elapsed:.3f}s"
