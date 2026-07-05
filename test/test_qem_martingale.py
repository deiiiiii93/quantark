"""WS-C5: Andersen QE-M martingale correction (HestonMCScheme.QUADEXP_M)."""
import numpy as np
import pytest

from quantark.util.enum.engine_enums import HestonMCScheme
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.heston.mc_kernel import price_european_heston_mc
from quantark.volmodels.heston.analytical_kernel import heston_call_price


def _run(scheme, s0=100.0, T=1.0, r=0.0, q=0.0, steps=1, num_paths=500_000, seed=7):
    # Coarse-step, high vol-of-vol, strong negative correlation: the regime where the
    # standard QUADEXP drift approximation leaves a MEASURABLE martingale bias (~ -7.8
    # stderr here) that QE-M's exact K0* removes (~ +0.6 stderr). At the fine 8-steps/yr
    # grid QE's bias is already sub-noise, so a coarse grid is needed to expose it.
    params = HestonParams(kappa=1.0, theta=0.16, sigma=1.5, rho=-0.9, v0=0.16)
    dt = np.full(steps, T / steps)
    rf = np.full(steps, r)
    cf = np.full(steps, q)
    df = np.exp(-r * T)
    # Near-zero strike -> call payoff == S_T, so discounted price == discounted E[S_T];
    # under an exact martingale that equals s0.
    k = 1e-6
    price, stderr = price_european_heston_mc(
        s0, k, True, params, dt, rf, cf, df, scheme=scheme,
        num_paths=num_paths, seed=seed, return_stderr=True,
    )
    return price, stderr


def test_qem_removes_martingale_bias_where_qe_shows_it():
    s0 = 100.0
    p_qe, se_qe = _run(HestonMCScheme.QUADEXP)
    p_qem, se_qem = _run(HestonMCScheme.QUADEXP_M)
    # QE-M within 3 stderr of the exact forward (discounted E[S_T] == s0).
    assert abs(p_qem - s0) <= 3.0 * se_qem
    # And QE-M is at least as unbiased as QE in this regime.
    assert abs(p_qem - s0) <= abs(p_qe - s0) + 1e-9


def test_qem_reprices_european_within_mc_error_vs_analytical():
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.0
    params = HestonParams(kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04)
    dt = np.full(24, T / 24); rf = np.full(24, r); cf = np.full(24, q)
    df = np.exp(-r * T)
    analytic = heston_call_price(s0, k, T, params, r, q)
    price, stderr = price_european_heston_mc(
        s0, k, True, params, dt, rf, cf, df, scheme=HestonMCScheme.QUADEXP_M,
        num_paths=200_000, seed=11, return_stderr=True,
    )
    assert abs(price - analytic) <= 4.0 * stderr + 1e-3


def test_quadexp_output_unchanged_by_qem_addition():
    # Guard: QUADEXP must be byte-identical to its pre-QE-M behavior (same seed/paths).
    s0, k, T, r, q = 100.0, 105.0, 1.0, 0.02, 0.01
    params = HestonParams(kappa=1.5, theta=0.05, sigma=0.6, rho=-0.5, v0=0.05)
    dt = np.full(12, T / 12); rf = np.full(12, r); cf = np.full(12, q)
    df = np.exp(-r * T)
    price = price_european_heston_mc(
        s0, k, True, params, dt, rf, cf, df, scheme=HestonMCScheme.QUADEXP,
        num_paths=50_000, seed=42,
    )
    assert price == pytest.approx(5.956183946331143, abs=0.0, rel=0.0)
