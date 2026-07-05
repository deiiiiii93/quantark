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


# --- WS-B2: pinned regressions for the batched-tridiagonal swap (bit-identity gate) ---

_REG_PARAMS = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.5, rho=-0.7)


@pytest.mark.parametrize("scheme,is_call,pinned", [
    (ADIScheme.CRAIG_SNEYD, True, 8.882379432432987),
    (ADIScheme.CRAIG_SNEYD, False, 6.9082903890508),
    (ADIScheme.DOUGLAS, True, 8.884504985480513),
])
def test_adi_price_regression_pinned_for_batch_swap(scheme, is_call, pinned):
    # Regression anchor for the current scheme. WS-C1 (2026-07-05) deliberately moved these
    # values: the Craig-Sneyd corrector fix (base Y0, not Y2) + implicit -rU restore
    # 2nd-order-in-time; prices moved within discretization tolerance.
    p = price_european_heston_pde(100.0, 100.0, is_call, 1.0, _REG_PARAMS, 0.03, 0.01,
                                  n_x=120, n_v=60, n_t=60, scheme=scheme)
    assert np.isclose(p, pinned, rtol=1e-13)


def test_heston_pde_cs_reference_values():
    # Regression anchor via the unified HestonSLVADICore (dense CS, sparse Douglas,
    # grid_spot delta/gamma). Values reflect the WS-C1 corrected-CS + implicit-rU scheme.
    from quantark.util.enum.engine_enums import ADIScheme
    from quantark.volmodels.heston.params import HestonParams
    from quantark.volmodels.heston.pde_kernel import (
        price_european_heston_pde, price_delta_gamma_heston_pde)
    P = HestonParams(v0=0.04, kappa=1.5, theta=0.05, sigma=0.4, rho=-0.6)
    a = price_european_heston_pde(100,100,True,1.0,P,0.03,0.01, n_x=80,n_v=48,n_t=40, scheme=ADIScheme.CRAIG_SNEYD)
    b = price_european_heston_pde(100,110,False,0.7,P,0.05,0.0, n_x=80,n_v=48,n_t=40, scheme=ADIScheme.DOUGLAS, use_sparse=True)
    c = price_delta_gamma_heston_pde(100,100,True,1.0,P,0.03,0.01, n_x=80,n_v=48,n_t=40, grid_spot=100.0)
    PINS = (9.19048150136939, 10.189989602224074,
            (9.174386304782422, 0.6232661496627931, 0.018848069046049663))
    assert abs(a - PINS[0]) <= 1e-12 * max(1.0, abs(PINS[0]))
    assert abs(b - PINS[1]) <= 1e-12 * max(1.0, abs(PINS[1]))
    for got, ref in zip(c, PINS[2]):
        assert abs(got - ref) <= 1e-12 * max(1.0, abs(ref))
