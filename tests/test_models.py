import numpy as np
import pytest

from howhard import Engine
from howhard.config import load_prior, packaged
from howhard.gaussian import Gaussian
from howhard.outcomes import fit, objective, probabilities
from howhard.prior import build_prior, design
from howhard.regression import conjugate, fit_regression
from howhard.validation import HowHardError
from conftest import cell, fitted_rows, row


def test_no_data_identity(project):
    engine = Engine(project())
    for block in ("outcome", "tokens", "time"):
        posterior, prior = getattr(engine.posterior, block), getattr(engine.prior, block)
        np.testing.assert_allclose(posterior.mean, prior.mean, atol=1e-9, rtol=0)
        np.testing.assert_allclose(posterior.covariance, prior.covariance, atol=1e-9, rtol=0)


def test_prior_mode_calibration(project):
    config = project()
    prior = build_prior(config)
    x = design(prior.outcome, "outcome", [("contract-change", "opus", "high", "local")])
    np.testing.assert_allclose(probabilities((x @ prior.outcome.mean).reshape(1, 2))[0], [.6, .35, .05], atol=1e-12)


@pytest.mark.parametrize("old,new,key", [
    ('sd = 0.5', 'sd = 0.0', 'sd'),
    ('sd = 0.5', 'spread = 0.5', 'sd'),
    ('sd = 0.5', 'sd = 0.5, typo = 1.0', 'typo'),
    ('"haiku", "sonnet", "opus", "fable"', '"unknown", "sonnet", "opus", "fable"', 'families'),
    ('"low", "medium", "high", "xhigh", "max"', '"low", "medium", "high", "xhigh", "unknown"', 'efforts'),
    ('class = "contract"', 'class = "unknown"', 'class'),
])
def test_strict_prior_file(project, old, new, key):
    config = project()
    path = config.base / "prior.toml"
    path.write_bytes(packaged("default_prior.toml").replace(old.encode(), new.encode(), 1))
    with pytest.raises(HowHardError, match=key):
        load_prior(path)


def test_gradient_hessian_against_central_differences():
    rng = np.random.default_rng(41)
    n, d = 5, 7
    x = rng.normal(size=(2 * n, d))
    counts = rng.uniform(.1, 3, size=(n, 3))
    mean, beta = rng.normal(size=(2, d))
    a = rng.normal(size=(d, d))
    precision = a @ a.T + np.eye(d)
    _, gradient, hessian = objective(beta, mean, precision, x, counts)
    numerical_gradient = np.zeros(d)
    numerical_hessian = np.zeros((d, d))
    h = 1e-5
    for i in range(d):
        direction = np.eye(d)[i] * h
        hi = objective(beta + direction, mean, precision, x, counts)
        lo = objective(beta - direction, mean, precision, x, counts)
        numerical_gradient[i] = (hi[0] - lo[0]) / (2 * h)
        numerical_hessian[:, i] = (hi[1] - lo[1]) / (2 * h)
    np.testing.assert_allclose(gradient, numerical_gradient, atol=1e-7, rtol=0)
    np.testing.assert_allclose(hessian, numerical_hessian, atol=1e-7, rtol=0)


def test_row_order_invariance_and_weight_equivalence(project):
    config = project()
    rows = fitted_rows(config, [row(i, effort="high" if i % 2 else "xhigh", outcome="clean" if i % 3 else "loud") for i in range(30)])
    prior = build_prior(config, rows)
    a = fit(prior, rows)
    np.random.default_rng(45).shuffle(rows)
    b = fit(prior, rows)
    np.testing.assert_allclose(a.mean, b.mean, atol=1e-9, rtol=0)
    np.testing.assert_allclose(a.covariance, b.covariance, atol=1e-9, rtol=0)
    half = [{**r, "weight": .5} for r in rows for _ in range(2)]
    c = fit(prior, half)
    np.testing.assert_allclose(a.mean, c.mean, atol=1e-9, rtol=0)
    np.testing.assert_allclose(a.covariance, c.covariance, atol=1e-9, rtol=0)


def test_conjugate_regression_matches_closed_form_random_problem():
    rng = np.random.default_rng(44)
    x, y, mean = rng.normal(size=(20, 6)), rng.normal(size=20), rng.normal(size=6)
    a = rng.normal(size=(6, 6))
    cov = a @ a.T + np.eye(6)
    weights = rng.uniform(.2, 1, size=20)
    prior = Gaussian(tuple(map(str, range(6))), mean, cov)
    posterior = conjugate(prior, x, y, weights, 1.2)
    precision = np.linalg.inv(cov)
    expected_cov = np.linalg.inv(precision + x.T @ np.diag(weights / 1.2**2) @ x)
    expected_mean = expected_cov @ (precision @ mean + x.T @ (weights / 1.2**2 * y))
    np.testing.assert_allclose(posterior.mean, expected_mean, atol=1e-9, rtol=0)
    np.testing.assert_allclose(posterior.covariance, expected_cov, atol=1e-9, rtol=0)


def test_xhigh_token_effect_learns_twofold_volume(project):
    config = project()
    rows = fitted_rows(config, [row(i, effort="high" if i % 2 else "xhigh", tokens=100000 if i % 2 else 200000) for i in range(200)])
    prior = build_prior(config, rows)
    posterior = fit_regression(prior, rows, "tokens")
    delta = posterior.mean[posterior.index("tokens.effort.xhigh")] - posterior.mean[posterior.index("tokens.effort.high")]
    assert .2 < delta < np.log(2)


def test_null_measurement_ignored_but_outcomes_counted(project):
    engine = Engine(project(rows=[row(i, outcome="loud") for i in range(8)]))
    np.testing.assert_array_equal(engine.posterior.tokens.mean, engine.prior.tokens.mean)
    assert cell(engine.summary("X"), "opus", "high")["P_loud"]["mean"] > .5


def test_price_changes_only_its_own_running_cost(project):
    config = project()
    a = Engine(config).summary("Y")
    model = config.data["families"]["opus"]["current"]
    config.data["models"][model] = {**config.data["models"][model], "usd_per_mtok": 2 * config.price(model)}
    config.data["models"][model].pop("prices")
    b = Engine(config).summary("Y")
    for before in a["cells"]:
        after = cell(b, before["family"], before["effort"])
        factor = 2 if before["family"] == "opus" else 1
        assert after["C_run"] == pytest.approx(factor * before["C_run"], rel=1e-12)


def test_blended_prices_and_malformed_prices(project):
    config = project()
    model = config.data["families"]["opus"]["current"]
    assert config.price(model) == pytest.approx(4 * .04 + 20 * .01 + 5 * .01 + 8 * 0 + .2 * .94)


@pytest.mark.parametrize("settings,match", [
    ('[token_mix]\ninput = 0.8', 'token_mix'),
    ('[models.claude-opus-5-5.prices]\ninput = -1.0', 'input'),
    ('[models.claude-opus-5-5.prices]\ntypo = 3.0', 'typo'),
    ('[models.claude-opus-5-5]\nusd_per_mtok = "cheap"', 'usd_per_mtok'),
    ('[models.claude-opus-5-5]\nweight = false', 'weight'),
    ('[classes.standard]\nepsilon = 1.0', 'epsilon'),
    ('[task_types]\nX = "unknown"', 'task_types'),
    ('classes = []', 'classes'),
    ('families = []', 'families'),
    ('models = []', 'models'),
    ('task_types = []', 'task_types'),
    ('[models.claude-opus-5-5]\nfamily = []', 'family'),
    ('[task_types]\nX = []', 'task_types'),
])
def test_config_refuses_invalid_settings(project, tmp_path, settings, match):
    from howhard import load_config
    path = tmp_path / "invalid.toml"
    path.write_text('on_cap_unmet = "flag-review"\n' + settings)
    with pytest.raises(HowHardError, match=match):
        load_config(path)
