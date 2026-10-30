from copy import deepcopy
import json
import numpy as np
import pytest

from howhard import Engine
from howhard.transfer import import_prior, shared
from howhard.validation import HowHardError
from conftest import inventory, cell, row


def test_both_cap_fallbacks_and_why_components(project):
    config = project()
    config.data["classes"]["standard"]["epsilon"] = 1e-9
    engine = Engine(config)
    flagged = engine.pick("Y", seed=42, native_candidates=inventory())
    assert not flagged["cap_met"] and flagged["needs_review"]
    chosen = cell(flagged, flagged["chosen"]["family"], flagged["chosen"]["effort"])
    for why in flagged["why"]:
        cheaper = cell(flagged, why["family"], why["effort"])
        assert sum(why["components"].values()) == pytest.approx(cheaper["sampled_J"] - chosen["sampled_J"], abs=1e-9)
    config.data["on_cap_unmet"] = "lowest-bound"
    other = Engine(config).pick("Y", seed=42, native_candidates=inventory())
    assert not other["cap_met"] and not other["needs_review"]
    picked = cell(other, other["chosen"]["family"], other["chosen"]["effort"])
    assert picked["P_silent"]["upper"] == min(c["P_silent"]["upper"] for c in other["cells"])


def test_explore_within_limits_expected_cost_and_explains_filter(project):
    config = project()
    config.data["classes"]["standard"]["epsilon"] = .99
    config.data["classes"]["standard"]["explore_within"] = 0
    engine = Engine(config)
    summary = engine.summary("Y", native_candidates=inventory())
    minimum = min(c["J"] for c in summary["cells"])
    for seed in range(50):
        result = engine.pick("Y", seed=seed, native_candidates=inventory())
        assert cell(result, result["chosen"]["family"], result["chosen"]["effort"])["J"] == minimum
    assert any(w["reason"] == "explore-within" for w in result["why"])


def test_cap_explanations_and_json_schema(project):
    config = project(rows=[row(i, task="cheap", family="haiku", effort="low") for i in range(30)])
    result = Engine(config).pick("cheap", seed=1, native_candidates=inventory())
    assert set(result) == {"task_type", "risk_class", "constants", "cells", "fit", "chosen", "seed", "cap_met", "needs_review", "why", "context", "decision_id", "native_candidates", "cost_policy", "paths"}
    assert result["cells"] == sorted(result["cells"], key=lambda c: (c["C_run"], c["family"], c["effort"]))
    assert all(set(c) == {"model", "family", "effort", "P_clean", "P_loud", "P_silent", "C_run", "T", "J", "sampled_J", "admissible"} for c in result["cells"])


def test_corrupt_cache_delete_reproduce_and_row_order(project):
    config = project(rows=[row(i, outcome="clean" if i % 3 else "loud", tokens=10000+i) for i in range(20)])
    engine = Engine(config)
    original = deepcopy(engine.posterior)
    path = config.path(config.data["cache"])
    path.write_bytes(b"garbage truncated archive")
    rebuilt = Engine(config)
    assert rebuilt.posterior.statistics["kind"] == "full"
    assert path.read_bytes().startswith(b"PK")
    path.unlink()
    reproduced = Engine(config)
    for block in ("outcome", "tokens", "time"):
        np.testing.assert_allclose(getattr(original, block).mean, getattr(reproduced.posterior, block).mean, atol=1e-9, rtol=0)
        np.testing.assert_allclose(getattr(original, block).covariance, getattr(reproduced.posterior, block).covariance, atol=1e-9, rtol=0)
    with np.load(path, allow_pickle=False) as data:
        before_hash = data["data_hash"].item()
    log = config.path(config.data["logs"][0]["path"])
    log.write_bytes(b"\n".join(reversed(log.read_bytes().splitlines())) + b"\n")
    reordered = Engine(config)
    with np.load(path, allow_pickle=False) as data:
        assert data["data_hash"].item() != before_hash
    np.testing.assert_allclose(reordered.posterior.outcome.mean, original.outcome.mean, atol=1e-9, rtol=0)
    np.testing.assert_allclose(reordered.posterior.tokens.mean, original.tokens.mean, atol=1e-9, rtol=0)


def test_real_prior_file_change_invalidates_cache(project):
    from howhard.config import packaged
    config = project()
    path = config.base / "prior.toml"
    path.write_bytes(packaged("default_prior.toml"))
    source = config.base / "howhard.toml"
    source.write_text(source.read_text() + '\n[prior]\npath = "prior.toml"\n')
    assert Engine(source).posterior.statistics["kind"] == "full"
    assert Engine(source).posterior.statistics["kind"] == "cached"
    path.write_bytes(path.read_bytes().replace(b"sd = 0.5", b"sd = 0.51", 1))
    assert Engine(source).posterior.statistics["kind"] == "full"


def test_export_inflation_preserves_means_and_cross_covariances(project, tmp_path):
    config = project(rows=[row(i, outcome="loud") for i in range(6)])
    engine = Engine(config)
    value = engine.export_prior(tmp_path / "prior.json")
    for block in ("outcome", "tokens", "time"):
        source = getattr(engine.posterior, block)
        indices = [i for i, label in enumerate(source.labels) if shared(label)]
        exported = value["blocks"][block]
        positions = [exported["labels"].index(source.labels[i]) for i in indices]
        np.testing.assert_array_equal(np.asarray(exported["mean"])[positions], source.mean[indices])
        np.testing.assert_array_equal(np.asarray(exported["covariance"])[np.ix_(positions, positions)], source.covariance[np.ix_(indices, indices)] * 2)


@pytest.mark.parametrize("malformation,match", [
    ("version", "format"), ("negative", "positive definite"), ("asymmetric", "positive definite"),
    ("unknown_label", "labels"), ("unknown_family", "families"), ("shape", "shape"),
    ("nan", "finite"), ("bool", "finite"), ("extra", "unknown"),
])
def test_import_validation_and_no_write_on_error(project, tmp_path, malformation, match):
    config = project()
    engine = Engine(config)
    path = tmp_path / "export.json"
    value = engine.export_prior(path)
    block = value["blocks"]["outcome"]
    if malformation == "version":
        value["format"] = "unknown"
    elif malformation == "negative":
        block["covariance"][0][0] = -1
    elif malformation == "asymmetric":
        block["covariance"][0][1] += .1
    elif malformation == "unknown_label":
        block["labels"][0] = "loud.pi.secret-project"
    elif malformation == "unknown_family":
        value["vocabularies"]["families"].append("unknown")
    elif malformation == "shape":
        block["mean"].pop()
    elif malformation == "nan":
        block["mean"][0] = float("nan")
    elif malformation == "bool":
        block["mean"][0] = True
    else:
        value["private"] = "secret"
    path.write_text(json.dumps(value))
    destination = config.path(config.data["prior"]["import"])
    with pytest.raises(HowHardError, match=match):
        import_prior(config, path)
    assert not destination.exists()


def test_missing_import_destination_refused_and_missing_terms_handset(project, tmp_path):
    config = project()
    engine = Engine(config)
    path = tmp_path / "export.json"
    exported = engine.export_prior(path)
    config.data["prior"].pop("import")
    with pytest.raises(HowHardError, match="prior.import"):
        import_prior(config, path)
    config.data["prior"]["import"] = "imported.json"
    # Partial shared blocks are legal; terms absent from transfer keep hand-set priors.
    for block in exported["blocks"].values():
        block["labels"] = block["labels"][:1]
        block["mean"] = block["mean"][:1]
        block["covariance"] = [[block["covariance"][0][0]]]
    path.write_text(json.dumps(exported))
    engine.import_prior(path)
    index = engine.posterior.outcome.index("loud.w.X.opus")
    assert engine.posterior.outcome.mean[index] == 0
    assert engine.posterior.outcome.covariance[index, index] == .4 ** 2
