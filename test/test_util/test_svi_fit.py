"""Raw-SVI slice fit tests (spec WP4.2)."""
import numpy as np
import pytest

from quantark.param.vol.svi import SVIParams, fit_svi_slice
from quantark.util.exceptions import NumericalError

TRUE = SVIParams(a=0.02, b=0.4, rho=-0.3, m=0.05, sigma=0.2)
Y = np.linspace(-0.6, 0.6, 15)


def test_exact_recovery():
    w = TRUE.total_variance(Y)
    fit = fit_svi_slice(Y, w, None, expiry_t=0.5)
    assert fit.rmse < 1e-6
    assert fit.params.a == pytest.approx(TRUE.a, abs=2e-3)
    assert fit.params.b == pytest.approx(TRUE.b, abs=2e-3)
    assert fit.params.rho == pytest.approx(TRUE.rho, abs=5e-3)
    assert fit.params.m == pytest.approx(TRUE.m, abs=5e-3)
    assert fit.params.sigma == pytest.approx(TRUE.sigma, abs=5e-3)
    assert not fit.refit_applied


def test_butterfly_positive_on_dense_grid():
    fit = fit_svi_slice(Y, TRUE.total_variance(Y), None, expiry_t=0.5)
    dense = np.arange(-1.5, 1.5, 0.01)
    assert float(np.min(fit.params.g(dense))) > -1e-8


def test_noisy_fit():
    rng = np.random.default_rng(7)
    w = TRUE.total_variance(Y) + rng.normal(0.0, 1e-4, Y.size)
    fit = fit_svi_slice(Y, w, None, expiry_t=0.5)
    assert fit.rmse < 5e-4


def test_lee_wing_bound_enforced():
    # target data with wing slopes well beyond the Lee bound (b=3): the
    # optimizer must return a constrained fit or fail loudly - never emit
    # params violating b(1+|rho|) <= 2
    steep = SVIParams(a=0.001, b=3.0, rho=0.0, m=0.0, sigma=0.05)
    try:
        fit = fit_svi_slice(Y, steep.total_variance(Y), None, expiry_t=0.5)
        assert fit.params.b * (1 + abs(fit.params.rho)) <= 2.0 + 1e-9
    except NumericalError:
        pass  # loud failure is an acceptable outcome per the no-arb policy


def test_analytic_g_matches_finite_differences():
    p = TRUE
    y = np.linspace(-0.8, 0.8, 33)
    h = 1e-5
    w = p.total_variance(y)
    w1_fd = (p.total_variance(y + h) - p.total_variance(y - h)) / (2 * h)
    w2_fd = (
        p.total_variance(y + h) - 2 * w + p.total_variance(y - h)
    ) / (h * h)
    term = 1.0 - y * w1_fd / (2.0 * w)
    g_fd = term ** 2 - (w1_fd ** 2 / 4.0) * (1.0 / w + 0.25) + w2_fd / 2.0
    assert np.allclose(p.g(y), g_fd, atol=1e-6)


def test_deterministic():
    w = TRUE.total_variance(Y)
    f1 = fit_svi_slice(Y, w, None, expiry_t=0.5)
    f2 = fit_svi_slice(Y, w, None, expiry_t=0.5)
    assert f1.params == f2.params


def test_to_dict_json_safe():
    import json
    fit = fit_svi_slice(Y, TRUE.total_variance(Y), None, expiry_t=0.5)
    json.dumps(fit.to_dict())
