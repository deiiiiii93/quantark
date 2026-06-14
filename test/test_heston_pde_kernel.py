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
