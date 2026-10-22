"""Tests for the LV MC time-sampling modes ("left" | "mid" | "integrated").

Covers the surface's exact time-averaged variance, the kernel-level modes, and the
LocalVolSnowballMCEngine wiring. The "left" default must stay bitwise-identical to
the historical scheme; "integrated" must be exact (level-independent) on a surface
whose vol depends on t only.
"""

import numpy as np
import pytest

from quantark.util.exceptions import ValidationError
from quantark.volmodels.localvol.mc_kernel import (
    price_barrier_lv_mc,
    price_european_lv_mc,
)
from quantark.volmodels.localvol.surface import LocalVolSurface

STRIKES = np.geomspace(40.0, 250.0, 31)
TIMES = np.array([0.0, 1 / 52, 1 / 12, 0.25, 0.5, 0.75, 1.0])


def _steep_surface():
    term = 0.22 + 0.12 * np.exp(-3.5 * TIMES)
    skew = np.clip((STRIKES / 100.0) ** (-0.9), 0.45, 2.2)
    return LocalVolSurface(STRIKES.copy(), TIMES.copy(), term[:, None] * skew[None, :])


def _time_only_surface():
    term = 0.22 + 0.12 * np.exp(-3.5 * TIMES)
    return LocalVolSurface(
        STRIKES.copy(), TIMES.copy(), np.repeat(term[:, None], STRIKES.size, axis=1)
    )


def _flat_surface(sigma=0.25):
    return LocalVolSurface(
        STRIKES.copy(), TIMES.copy(), np.full((TIMES.size, STRIKES.size), sigma)
    )


# ---------------------------------------------------------------------------
# LocalVolSurface.time_avg_var
# ---------------------------------------------------------------------------


def test_time_avg_var_matches_brute_force_quadrature():
    surf = _steep_surface()
    spots = np.array([55.0, 80.0, 100.0, 130.0, 220.0])
    for t0, t1 in [(0.0, 1.0), (0.01, 0.13), (0.3, 0.31), (0.9, 1.4)]:
        exact = surf.time_avg_var(spots, t0, t1)
        grid = np.linspace(t0, t1, 20_001)
        vals = np.stack([np.asarray(surf.local_vol(spots, u)) ** 2 for u in grid])
        brute = np.trapezoid(vals, grid, axis=0) / (t1 - t0)
        np.testing.assert_allclose(exact, brute, rtol=1e-7)


def test_time_avg_var_scalar_spot_returns_float():
    surf = _steep_surface()
    out = surf.time_avg_var(100.0, 0.0, 0.5)
    assert isinstance(out, float) and out > 0


def test_time_avg_var_flat_surface_is_constant_variance():
    surf = _flat_surface(0.31)
    out = surf.time_avg_var(np.array([70.0, 100.0]), 0.05, 0.62)
    np.testing.assert_allclose(out, 0.31**2, rtol=1e-14)


def test_time_avg_var_rejects_bad_interval():
    surf = _flat_surface()
    with pytest.raises(ValidationError):
        surf.time_avg_var(100.0, 0.5, 0.5)


# ---------------------------------------------------------------------------
# Kernel modes
# ---------------------------------------------------------------------------

KERNEL_ARGS = dict(
    s0=100.0, strike=100.0, is_call=True, disc_factor=1.0,
    num_paths=40_000, seed=11,
)


def _euro(surf, n, time_sampling, **overrides):
    args = {**KERNEL_ARGS, **overrides}
    return price_european_lv_mc(
        lv_surface=surf, step_dt=np.full(n, 1.0 / n), r_fwd=np.zeros(n),
        carry_fwd=np.zeros(n), time_sampling=time_sampling, **args,
    )


def test_kernel_default_is_left_bitwise():
    surf = _steep_surface()
    assert _euro(surf, 24, "left") == price_european_lv_mc(
        lv_surface=surf, step_dt=np.full(24, 1.0 / 24), r_fwd=np.zeros(24),
        carry_fwd=np.zeros(24), **KERNEL_ARGS,
    )


def test_kernel_flat_surface_modes_agree():
    surf = _flat_surface(0.27)
    p_left = _euro(surf, 12, "left")
    assert _euro(surf, 12, "mid") == p_left
    assert abs(_euro(surf, 12, "integrated") - p_left) < 1e-10


def test_kernel_integrated_exact_on_time_only_surface():
    """On a time-only surface "integrated" is exact: coarse == fine within MC noise,
    while "left" at the same coarse step carries a visible O(h) bias."""
    surf = _time_only_surface()
    paths = 400_000
    p_int_4, se_int_4 = _euro(surf, 4, "integrated", num_paths=paths, return_stderr=True)
    p_int_96, se_int_96 = _euro(surf, 96, "integrated", num_paths=paths, return_stderr=True)
    p_left_4, se_left_4 = _euro(surf, 4, "left", num_paths=paths, return_stderr=True)
    tol = 4.0 * np.hypot(se_int_4, se_int_96)
    assert abs(p_int_4 - p_int_96) < tol
    assert abs(p_left_4 - p_int_96) > 4.0 * np.hypot(se_left_4, se_int_96)


def test_kernel_rejects_unknown_mode():
    surf = _flat_surface()
    with pytest.raises(ValidationError):
        _euro(surf, 4, "midpoint")


def test_barrier_kernel_time_sampling_smoke():
    """Barrier kernel accepts the mode and the bridge stays consistent: flat-surface
    "integrated" equals "left" to float tolerance (identical variance per step)."""
    surf = _flat_surface(0.3)
    common = dict(
        s0=100.0, strike=100.0, is_call=False, lv_surface=surf,
        step_dt=np.full(24, 1.0 / 24), r_fwd=np.zeros(24), carry_fwd=np.zeros(24),
        disc_factor=1.0, barrier=75.0, is_up=False, is_out=False,
        continuous=True, num_paths=30_000, seed=5,
    )
    p_left = price_barrier_lv_mc(**common)
    p_int = price_barrier_lv_mc(time_sampling="integrated", **common)
    assert abs(p_int - p_left) < 1e-9


# ---------------------------------------------------------------------------
# LocalVolSnowballMCEngine wiring
# ---------------------------------------------------------------------------


def _snowball_setup():
    from datetime import datetime

    from quantark.asset.equity.product.option import SnowballOption
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.param.div import ContinuousDividendYield
    from quantark.priceenv import PricingEnvironment

    barrier_config = BarrierConfig(
        ko_barrier=103.0, ko_rate=0.15,
        ko_observation_dates=[i / 12 for i in range(1, 13)],
        ki_barrier=75.0, ki_continuous=True,
    )
    product = SnowballOption(
        initial_price=100.0, strike=100.0, barrier_config=barrier_config,
        contract_multiplier=1.0, maturity=1.0,
    )
    env = PricingEnvironment(
        rate_curve=FlatRateCurve(0.0),
        valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(0.25),
        div_yield=ContinuousDividendYield(0.0),
    )
    return product, env


def _engine(surface, time_sampling="left", substeps=1, num_paths=60_000):
    from quantark.asset.equity.engine.mc import LocalVolSnowballMCEngine
    from quantark.asset.equity.param import MCParams

    return LocalVolSnowballMCEngine(
        params=MCParams(num_paths=num_paths, seed=42),
        local_vol_surface=surface,
        lv_time_sampling=time_sampling,
        substeps_per_interval=substeps,
    )


def test_engine_default_mode_bitwise_and_validation():
    product, env = _snowball_setup()
    surf = _steep_surface()
    p_default = _engine(surf).price(product, env)
    p_left = _engine(surf, "left").price(product, env)
    assert p_default == p_left
    with pytest.raises(ValidationError):
        _engine(surf, "midpoint")


def test_engine_integrated_invariant_on_time_only_surface():
    """Snowball PV under "integrated" must be substep-invariant on a time-only
    surface (the scheme's marginal law is exact there, so any level dependence
    is a wiring bug). The bias contrast against "left" stepping is asserted at
    kernel level, where the payoff makes it visible above MC noise."""
    product, env = _snowball_setup()
    surf = _time_only_surface()
    e1 = _engine(surf, "integrated", substeps=1, num_paths=150_000)
    p1 = e1.price(product, env)
    e8 = _engine(surf, "integrated", substeps=8, num_paths=150_000)
    p8 = e8.price(product, env)
    tol = 5.0 * float(np.hypot(e1.get_last_std_error(), e8.get_last_std_error()))
    assert abs(p1 - p8) < tol
