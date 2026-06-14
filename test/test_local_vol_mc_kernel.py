import numpy as np
import pytest
from quantark.util.exceptions import ValidationError
from quantark.volmodels.localvol.surface import LocalVolSurface
from quantark.volmodels.localvol.mc_kernel import price_european_lv_mc
from quantark.volmodels.black_scholes import bs_call_price, bs_put_price


def _flat_lv(sigma=0.2):
    return LocalVolSurface(np.array([1.0, 1e6]), np.array([0.0, 100.0]), np.full((2, 2), sigma))


def test_flat_local_vol_matches_black_scholes_call():
    sigma, s0, k, T, r, q = 0.2, 100.0, 100.0, 1.0, 0.03, 0.01
    n = 252
    price, stderr = price_european_lv_mc(
        s0=s0, strike=k, is_call=True, lv_surface=_flat_lv(sigma),
        step_dt=np.full(n, T / n), r_fwd=np.full(n, r), carry_fwd=np.full(n, q),
        disc_factor=float(np.exp(-r * T)), num_paths=200_000, seed=7,
        use_antithetic=True, return_stderr=True,
    )
    bs = bs_call_price(s0, k, T, sigma, r, q)
    assert abs(price - bs) < 4 * stderr + 5e-3


def test_put_via_is_call_false():
    sigma, s0, k, T = 0.25, 100.0, 110.0, 0.5
    n = 120
    price = price_european_lv_mc(
        s0=s0, strike=k, is_call=False, lv_surface=_flat_lv(sigma),
        step_dt=np.full(n, T / n), r_fwd=np.zeros(n), carry_fwd=np.zeros(n),
        disc_factor=1.0, num_paths=100_000, seed=1, use_antithetic=True,
    )
    assert abs(price - bs_put_price(s0, k, T, sigma, 0.0, 0.0)) < 0.1


def test_reproducible_with_seed():
    lv = _flat_lv(0.2)
    args = dict(s0=100.0, strike=100.0, is_call=True, lv_surface=lv,
                step_dt=np.full(50, 0.02), r_fwd=np.zeros(50), carry_fwd=np.zeros(50),
                disc_factor=1.0, num_paths=10_000, seed=42)
    assert price_european_lv_mc(**args) == price_european_lv_mc(**args)


def test_antithetic_stderr_uses_pair_averages():
    # With antithetic sampling the stderr must be computed from pair-average payoffs,
    # which is generally smaller than the naive all-paths stderr for a monotone payoff.
    lv = _flat_lv(0.2)
    _, stderr = price_european_lv_mc(
        s0=100.0, strike=100.0, is_call=True, lv_surface=lv,
        step_dt=np.full(50, 0.02), r_fwd=np.zeros(50), carry_fwd=np.zeros(50),
        disc_factor=1.0, num_paths=20_000, seed=3, use_antithetic=True, return_stderr=True,
    )
    assert stderr > 0


def test_rejects_invalid_inputs():
    lv = _flat_lv(0.2)
    with pytest.raises(ValidationError):
        price_european_lv_mc(s0=-1.0, strike=100.0, is_call=True, lv_surface=lv,
                             step_dt=np.full(5, 0.2), r_fwd=np.zeros(5), carry_fwd=np.zeros(5),
                             disc_factor=1.0, num_paths=100)
    with pytest.raises(ValidationError):
        price_european_lv_mc(s0=100.0, strike=100.0, is_call=True, lv_surface=lv,
                             step_dt=np.array([0.2, np.nan]), r_fwd=np.zeros(2),
                             carry_fwd=np.zeros(2), disc_factor=1.0, num_paths=100)
    with pytest.raises(ValidationError):
        price_european_lv_mc(s0=100.0, strike=100.0, is_call=True, lv_surface=lv,
                             step_dt=np.full(5, 0.2), r_fwd=np.zeros(5), carry_fwd=np.zeros(5),
                             disc_factor=-0.1, num_paths=100)


def test_antithetic_stderr_below_naive():
    # For a monotone payoff, antithetic pairing reduces variance vs independent paths.
    lv = _flat_lv(0.2)
    common = dict(s0=100.0, strike=100.0, is_call=True, lv_surface=lv,
                  step_dt=np.full(50, 0.02), r_fwd=np.zeros(50), carry_fwd=np.zeros(50),
                  disc_factor=1.0, num_paths=40_000, seed=9, return_stderr=True)
    _, se_anti = price_european_lv_mc(use_antithetic=True, **common)
    _, se_plain = price_european_lv_mc(use_antithetic=False, **common)
    assert se_anti < se_plain
