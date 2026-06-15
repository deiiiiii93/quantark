import numpy as np
import pytest
from quantark.util.enum.engine_enums import ADIScheme
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams, heston_call_price
from quantark.volmodels.heston.pde_kernel import price_european_heston_pde


P = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)


def test_pde_matches_analytic_call():
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    price = price_european_heston_pde(s0, k, True, T, P, r, q, n_x=200, n_v=80, n_t=100)
    assert price == pytest.approx(heston_call_price(s0, k, T, P, r, q), abs=0.3)


def test_pde_matches_analytic_put():
    s0, k, T, r, q = 100.0, 95.0, 1.0, 0.03, 0.0
    from quantark.volmodels.heston.analytical_kernel import heston_put_price
    price = price_european_heston_pde(s0, k, False, T, P, r, q, n_x=200, n_v=80, n_t=100)
    assert price == pytest.approx(heston_put_price(s0, k, T, P, r, q), abs=0.3)


def test_douglas_and_cs_agree():
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    d = price_european_heston_pde(s0, k, True, T, P, r, q, n_x=200, n_v=80, n_t=100,
                                  scheme=ADIScheme.DOUGLAS)
    c = price_european_heston_pde(s0, k, True, T, P, r, q, n_x=200, n_v=80, n_t=100,
                                  scheme=ADIScheme.CRAIG_SNEYD)
    assert abs(d - c) < 0.15


def test_sparse_matches_dense():
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    dense = price_european_heston_pde(s0, k, True, T, P, r, q, n_x=150, n_v=60, n_t=60, use_sparse=False)
    sparse = price_european_heston_pde(s0, k, True, T, P, r, q, n_x=150, n_v=60, n_t=60, use_sparse=True)
    assert sparse == pytest.approx(dense, abs=1e-9)


def test_pde_refinement_converges():
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    analytic = heston_call_price(s0, k, T, P, r, q)
    coarse = abs(price_european_heston_pde(s0, k, True, T, P, r, q, n_x=80, n_v=40, n_t=40) - analytic)
    fine = abs(price_european_heston_pde(s0, k, True, T, P, r, q, n_x=300, n_v=120, n_t=160) - analytic)
    assert fine < coarse


def test_invalid_inputs():
    with pytest.raises(ValidationError):
        price_european_heston_pde(100.0, 100.0, True, 1.0, P, 0.03, 0.01, n_x=2)
    with pytest.raises(ValidationError):
        price_european_heston_pde(-1.0, 100.0, True, 1.0, P, 0.03, 0.01)


def test_mcs_rejected():
    with pytest.raises(ValidationError):
        price_european_heston_pde(100.0, 100.0, True, 1.0, P, 0.03, 0.01, scheme=ADIScheme.MCS)


def test_low_volvol_matches_analytic_deterministic_limit():
    from quantark.volmodels.black_scholes import bs_call_price
    # sigma below threshold -> deterministic-variance BS limit.
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=1e-6, rho=-0.5)
    price = price_european_heston_pde(100.0, 100.0, True, 1.0, p, 0.03, 0.01)
    assert price == pytest.approx(bs_call_price(100.0, 100.0, 1.0, 0.2, 0.03, 0.01), abs=1e-6)


def test_grid_greeks_gamma_matches_analytic():
    from quantark.volmodels.heston.pde_kernel import price_delta_gamma_heston_pde
    from quantark.volmodels.heston import heston_call_price
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    _, delta, gamma = price_delta_gamma_heston_pde(s0, k, True, T, P, r, q, n_x=400, n_v=140, n_t=160)
    h = 0.5
    ana_gamma = (heston_call_price(s0 + h, k, T, P, r, q) - 2 * heston_call_price(s0, k, T, P, r, q)
                 + heston_call_price(s0 - h, k, T, P, r, q)) / (h * h)
    ana_delta = (heston_call_price(s0 + h, k, T, P, r, q) - heston_call_price(s0 - h, k, T, P, r, q)) / (2 * h)
    assert delta == pytest.approx(ana_delta, abs=0.02)
    assert gamma == pytest.approx(ana_gamma, abs=4e-3)  # positive, right magnitude
