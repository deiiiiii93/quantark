import numpy as np
import pytest
from quantark.util.enum.engine_enums import HestonMCScheme
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams, heston_call_price
from quantark.volmodels.heston.mc_kernel import price_european_heston_mc


P = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)


def _const(T, M, r, q):
    return np.full(M, T / M), np.full(M, r), np.full(M, q)


@pytest.mark.parametrize("scheme", [HestonMCScheme.EULER, HestonMCScheme.EULERLOG, HestonMCScheme.QUADEXP])
def test_mc_converges_to_analytic(scheme):
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    dt, rf, cf = _const(T, 100, r, q)
    price, se = price_european_heston_mc(
        s0, k, True, P, dt, rf, cf, disc_factor=float(np.exp(-r * T)),
        scheme=scheme, num_paths=200_000, seed=7, use_antithetic=True, return_stderr=True,
    )
    analytic = heston_call_price(s0, k, T, P, r, q)
    assert abs(price - analytic) < 4 * se + 0.05


def test_quadexp_accurate_few_steps():
    # QE is accurate even with coarse time stepping (its design goal).
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    dt, rf, cf = _const(T, 12, r, q)
    price = price_european_heston_mc(
        s0, k, True, P, dt, rf, cf, disc_factor=float(np.exp(-r * T)),
        scheme=HestonMCScheme.QUADEXP, num_paths=300_000, seed=3, use_antithetic=True,
    )
    assert price == pytest.approx(heston_call_price(s0, k, T, P, r, q), abs=0.1)


def test_put_via_parity_mc():
    s0, k, T, r, q = 100.0, 105.0, 1.0, 0.02, 0.0
    dt, rf, cf = _const(T, 100, r, q)
    kw = dict(s0=s0, strike=k, params=P, step_dt=dt, r_fwd=rf, carry_fwd=cf,
              disc_factor=float(np.exp(-r * T)), scheme=HestonMCScheme.QUADEXP,
              num_paths=200_000, seed=11, use_antithetic=True)
    c = price_european_heston_mc(is_call=True, **kw)
    p = price_european_heston_mc(is_call=False, **kw)
    parity = s0 * np.exp(-q * T) - k * np.exp(-r * T)
    assert (c - p) == pytest.approx(parity, abs=0.1)


def test_reproducible_and_validation():
    dt, rf, cf = _const(1.0, 50, 0.0, 0.0)
    kw = dict(s0=100.0, strike=100.0, is_call=True, params=P, step_dt=dt, r_fwd=rf,
              carry_fwd=cf, disc_factor=1.0, scheme=HestonMCScheme.QUADEXP,
              num_paths=10_000, seed=42)
    assert price_european_heston_mc(**kw) == price_european_heston_mc(**kw)
    with pytest.raises(ValidationError):
        price_european_heston_mc(s0=-1.0, strike=100.0, is_call=True, params=P, step_dt=dt,
                                 r_fwd=rf, carry_fwd=cf, disc_factor=1.0, num_paths=100)


def test_quadexp_deterministic_vol_is_rho_independent():
    # sigma->0: variance deterministic => price independent of rho, equals BS.
    from quantark.volmodels.black_scholes import bs_call_price
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.0, 0.0
    dt, rf, cf = _const(T, 100, r, q)
    bs = bs_call_price(s0, k, T, np.sqrt(0.04), r, q)
    for rho in (0.0, -0.7, -0.99):
        p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=1e-10, rho=rho)
        price, se = price_european_heston_mc(
            s0, k, True, p, dt, rf, cf, disc_factor=1.0, scheme=HestonMCScheme.QUADEXP,
            num_paths=200_000, seed=4, use_antithetic=True, return_stderr=True)
        assert abs(price - bs) < 4 * se + 0.02


def test_quadexp_zero_kappa_matches_analytic():
    # kappa=0, sigma>0: variance is a driftless CIR; QE moments use the k->0 limit.
    p = HestonParams(v0=0.04, kappa=0.0, theta=0.04, sigma=0.3, rho=-0.5)
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.02, 0.0
    dt, rf, cf = _const(T, 200, r, q)
    price, se = price_european_heston_mc(
        s0, k, True, p, dt, rf, cf, disc_factor=float(np.exp(-r * T)),
        scheme=HestonMCScheme.QUADEXP, num_paths=300_000, seed=8,
        use_antithetic=True, return_stderr=True)
    analytic = heston_call_price(s0, k, T, p, r, q)
    assert abs(price - analytic) < 4 * se + 0.05
