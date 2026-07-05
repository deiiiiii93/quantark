"""WS-C7: LV Crank-Nicolson Rannacher start-up + strike mid-cell grid."""
import numpy as np

from quantark.volmodels.localvol.pde_kernel import (
    _solve_lv_pde, price_european_lv_pde, price_delta_gamma_european_lv_pde,
)
from quantark.volmodels.localvol.surface import LocalVolSurface
from quantark.volmodels.black_scholes import bs_call_price, bs_put_price


def _flat_surface(vol=0.2):
    return LocalVolSurface(
        strike_grid=np.array([1.0, 1.0e6]),
        time_grid=np.array([0.0, 100.0]),
        lv_grid=np.full((2, 2), vol),
    )


def test_rannacher_default_is_on():
    import inspect
    for fn in (price_european_lv_pde, price_delta_gamma_european_lv_pde):
        assert inspect.signature(fn).parameters["rannacher"].default is True


def test_default_gamma_free_of_cn_oscillation_near_strike():
    # WS-C7 acceptance: the default (mid-cell grid + Rannacher) gives an oscillation-free
    # near-strike gamma profile. CN oscillation on an un-averaged kink would drive gamma
    # negative or zig-zag across the strike cells; the mid-cell placement (primary cure)
    # plus Rannacher start-up (robustness, matching the ADI convention) keeps gamma the
    # smooth, strictly non-negative bell it should be for a call. Tested across the
    # short-dated / coarse-time regimes where CN rings worst.
    s0, k, r, q = 100.0, 100.0, 0.03, 0.0
    surface = _flat_surface(0.2)
    for T, nt in [(0.05, 3), (0.02, 2), (0.1, 4), (0.5, 8)]:
        dt = np.full(nt, T / nt); rf = np.full(nt, r); cf = np.full(nt, q)
        s_grid, v = _solve_lv_pde(s0, k, True, T, surface, dt, rf, cf, n_s=201)
        gamma = np.gradient(np.gradient(v, s_grid), s_grid)
        window = (s_grid > 0.85 * k) & (s_grid < 1.15 * k)
        assert gamma[window].min() >= -1e-9, f"CN oscillation (negative gamma) at T={T},nt={nt}"


def test_rannacher_is_l_stable_at_ultra_coarse_steps():
    # Rannacher's two implicit half-steps are L-stable, so even a single ultra-coarse
    # backward step reprices near BS without CN's transient ringing.
    s0, k, T, r, q, vol = 100.0, 100.0, 0.25, 0.03, 0.0, 0.2
    surface = _flat_surface(vol)
    dt = np.array([T]); rf = np.array([r]); cf = np.array([q])   # ONE coarse step
    price = price_european_lv_pde(s0, k, True, T, surface, dt, rf, cf, n_s=400)
    # L-stability (not accuracy): a single step of first-order implicit-Euler half-steps
    # carries a real O(dt) error (~5% here), but the solution stays finite and ring-free.
    assert abs(price - bs_call_price(s0, k, T, vol, r, q)) < 0.35


def test_price_still_matches_bs_within_tolerance():
    s0, k, T, r, q, vol = 100.0, 105.0, 1.0, 0.03, 0.01, 0.2
    surface = _flat_surface(vol)
    dt = np.full(50, T / 50); rf = np.full(50, r); cf = np.full(50, q)
    price = price_european_lv_pde(s0, k, True, T, surface, dt, rf, cf, n_s=400)
    assert abs(price - bs_call_price(s0, k, T, vol, r, q)) < 0.05


def test_strike_is_mid_cell_and_boundary_nodes_preserved():
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.0
    surface = _flat_surface(0.2)
    dt = np.full(10, T / 10); rf = np.full(10, r); cf = np.full(10, q)
    s_grid, _ = _solve_lv_pde(s0, k, True, T, surface, dt, rf, cf, n_s=101)
    ds = s_grid[1] - s_grid[0]
    d = np.min(np.abs(s_grid - k))
    assert abs(d - ds / 2.0) < 0.02 * ds        # K exactly mid-cell
    assert s_grid[0] == 0.0                       # lower boundary node stays at S=0
    assert np.isclose(s_grid[-1], ds * (len(s_grid) - 1))


def test_put_and_low_spot_prices_match_bs():
    # The mid-cell change must not corrupt puts or low-spot cases (boundary economics).
    T, r, q, vol = 1.0, 0.03, 0.01, 0.2
    surface = _flat_surface(vol)
    dt = np.full(50, T / 50); rf = np.full(50, r); cf = np.full(50, q)
    for s0, k in [(100.0, 100.0), (60.0, 100.0), (100.0, 140.0)]:
        p_put = price_european_lv_pde(s0, k, False, T, surface, dt, rf, cf, n_s=400)
        assert abs(p_put - bs_put_price(s0, k, T, vol, r, q)) < 0.06
