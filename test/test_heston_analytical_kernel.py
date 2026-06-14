import numpy as np
import pytest
from quantark.util.exceptions import ValidationError
from quantark.volmodels.black_scholes import bs_call_price
from quantark.volmodels.heston import (
    HestonParams, MarketOption, calibrate_heston,
    heston_call_price, price_european_gatheral, price_european_lewis, price_european_weber,
)
from quantark.volmodels.heston.analytical_kernel import heston_put_price


P = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)


def test_three_methods_agree():
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    g = price_european_gatheral(s0, r, q, P, k, T)
    le = price_european_lewis(s0, r, q, P, k, T)
    we = price_european_weber(s0, r, q, P, k, T)
    assert le == pytest.approx(g, abs=1e-3)
    assert we == pytest.approx(g, abs=1e-3)


def test_three_methods_agree_otm_and_itm():
    s0, T, r, q = 100.0, 0.75, 0.02, 0.0
    for k in (80.0, 120.0):
        g = price_european_gatheral(s0, r, q, P, k, T)
        le = price_european_lewis(s0, r, q, P, k, T)
        we = price_european_weber(s0, r, q, P, k, T)
        assert le == pytest.approx(g, abs=2e-3)
        assert we == pytest.approx(g, abs=2e-3)


def test_heston_reduces_to_black_scholes_as_volvol_vanishes():
    # sigma -> 0, v0 = theta = const => deterministic variance => BS with vol=sqrt(v0).
    flat = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=1e-4, rho=0.0)
    s0, k, T, r, q = 100.0, 105.0, 1.0, 0.03, 0.01
    heston = heston_call_price(s0, k, T, flat, r, q)
    bs = bs_call_price(s0, k, T, np.sqrt(0.04), r, q)
    assert heston == pytest.approx(bs, abs=1e-2)


def test_put_call_parity():
    s0, k, T, r, q = 100.0, 110.0, 1.0, 0.03, 0.01
    c = heston_call_price(s0, k, T, P, r, q)
    p = heston_put_price(s0, k, T, P, r, q)
    parity = s0 * np.exp(-q * T) - k * np.exp(-r * T)
    assert (c - p) == pytest.approx(parity, abs=1e-6)


def test_unknown_method_raises():
    with pytest.raises(ValidationError):
        heston_call_price(100.0, 100.0, 1.0, P, 0.03, 0.0, method="bogus")


def test_calibration_recovers_known_params():
    s0, r, q = 100.0, 0.02, 0.0
    true = HestonParams(v0=0.05, kappa=1.5, theta=0.05, sigma=0.3, rho=-0.5)
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    mats = [0.5, 1.0, 1.5]
    opts = [MarketOption(K=k, T=t, price=heston_call_price(s0, k, t, true, r, q))
            for t in mats for k in strikes]
    init = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.2)
    res = calibrate_heston(s0, opts, r, q, init, target="price", regularize_feller=0.0)
    assert res.success
    # recover params to a loose tolerance (calibration is well-posed with this grid)
    assert res.params.v0 == pytest.approx(true.v0, abs=5e-3)
    assert res.params.theta == pytest.approx(true.theta, abs=1e-2)
    assert res.params.rho == pytest.approx(true.rho, abs=5e-2)


def test_deterministic_limit_sigma_zero_matches_bs():
    # sigma exactly 0: exact deterministic-variance limit (no ZeroDivisionError).
    flat = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.0, rho=0.0)
    s0, k, T, r, q = 100.0, 105.0, 1.0, 0.03, 0.01
    price = heston_call_price(s0, k, T, flat, r, q)
    assert price == pytest.approx(bs_call_price(s0, k, T, np.sqrt(0.04), r, q), abs=1e-9)


def test_deterministic_limit_term_structure_variance():
    # v0 != theta, sigma=0 => integrated CIR variance, not flat v0.
    p = HestonParams(v0=0.09, kappa=3.0, theta=0.01, sigma=0.0, rho=0.0)
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.0, 0.0
    integrated = 0.01 * T + (0.09 - 0.01) * (1 - np.exp(-3.0 * T)) / 3.0
    vol_eff = np.sqrt(integrated / T)
    assert heston_call_price(s0, k, T, p, r, q) == pytest.approx(
        bs_call_price(s0, k, T, vol_eff, r, q), abs=1e-9)


def test_extreme_params_raise_rather_than_return_garbage():
    from quantark.util.exceptions import NumericalError
    # ultra-short maturity + far OTM: CF quadrature is unreliable -> must raise, not
    # silently return a negative/huge price.
    p = HestonParams(v0=0.04, kappa=0.3, theta=0.04, sigma=2.0, rho=-0.99)
    with pytest.raises(NumericalError):
        heston_call_price(100.0, 10000.0, 1e-8, p, 0.0, 0.0)


def test_calibration_default_feller_penalty_runs():
    # Default regularize_feller>0 must not crash when params cross the Feller boundary.
    s0, r, q = 100.0, 0.02, 0.0
    true = HestonParams(v0=0.05, kappa=1.5, theta=0.05, sigma=0.3, rho=-0.5)
    opts = [MarketOption(K=k, T=t, price=heston_call_price(s0, k, t, true, r, q))
            for t in (0.5, 1.0) for k in (90.0, 100.0, 110.0)]
    init = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.9, rho=-0.2)  # Feller violated
    res = calibrate_heston(s0, opts, r, q, init, target="price")  # default penalty on
    assert res.success


def test_calibration_rejects_quote_without_price_or_iv():
    with pytest.raises(ValidationError):
        calibrate_heston(100.0, [MarketOption(K=100.0, T=1.0)], 0.02, 0.0,
                         HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.3, rho=-0.2))
