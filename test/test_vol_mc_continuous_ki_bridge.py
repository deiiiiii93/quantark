"""The continuous-KI Brownian bridge under Local Vol / Heston / SLV paths.

``SnowballMCEngine._check_ki_barriers_continuous_with_bridge`` estimates
whether a path touched the barrier BETWEEN two recorded nodes, using the
Brownian-bridge crossing probability ``exp(-2 ln(s0/B) ln(s1/B) / h^2)``.
``h^2`` is the log-variance the path actually accumulated over the interval,
and the vol-model engines inherited a version that took one scalar --
``pricing_env.get_vol(product.strike, T)``, the implied vol at the strike --
for every path and every step.  Their paths do not have that variance.

These tests need no benchmark.  When the simulated paths are GBM the
probability of ever touching the barrier is a closed form (reflection
principle), so the engine can be checked against arithmetic:

    P(min S <= B) = Phi((a - mu T)/(sig sqrt(T)))
                    + exp(2 mu a / sig^2) Phi((a + mu T)/(sig sqrt(T)))

with a = ln(B/S0) and mu = r - q - sig^2/2.  Degenerate Heston (zero
vol-of-vol, v0 = theta) is GBM; so is a constant local-vol surface, and so is
constant leverage on degenerate Heston.  Setting the ENVIRONMENT's implied vol
to a different number then makes the scalar the bridge used provably wrong.
"""

from datetime import datetime

import numpy as np
import pytest
from scipy.stats import norm

from quantark.asset.equity.engine.mc import (
    HestonSLVSnowballMCEngine,
    HestonSnowballMCEngine,
    LocalVolSnowballMCEngine,
    SnowballMCEngine,
)
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, GridVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import LocalVolSurface
from quantark.volmodels.slv.leverage import LeverageSurface

S0, BARRIER, T, RATE, DIV = 100.0, 75.0, 1.0, 0.03, 0.01
IMPLIED_VOL = 0.20  # what the environment quotes
PATH_VOL = 0.35     # what the paths actually do
NUM_PATHS = 200_000


def _env(vol=IMPLIED_VOL):
    strikes = list(S0 * np.exp(np.linspace(-0.5, 0.5, 9)))
    maturities = list(np.linspace(0.25, 1.0, 4))
    return PricingEnvironment(
        rate_curve=FlatRateCurve(RATE),
        valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=S0),
        vol_surface=GridVolSurface(
            strikes, maturities, np.full((len(maturities), len(strikes)), vol)
        ),
        div_yield=ContinuousDividendYield(vol * 0 + DIV),
    )


def _knock_in_only_snowball():
    """KO barrier out of reach, so ``ki_probability`` IS P(ever touch B)."""
    return SnowballOption(
        initial_price=S0, strike=S0, maturity=T, contract_multiplier=1.0,
        barrier_config=BarrierConfig(
            ko_barrier=1.0e6, ko_rate=0.12,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=BARRIER,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
    )


def _analytic_first_passage(sigma):
    a = np.log(BARRIER / S0)
    mu = RATE - DIV - 0.5 * sigma * sigma
    s = sigma * np.sqrt(T)
    return float(
        norm.cdf((a - mu * T) / s)
        + np.exp(2.0 * mu * a / (sigma * sigma)) * norm.cdf((a + mu * T) / s)
    )


def _params():
    return MCParams(num_paths=NUM_PATHS, time_steps=52, seed=7)


def _standard_error(p):
    return float(np.sqrt(p * (1.0 - p) / NUM_PATHS))


def _degenerate_heston(vol):
    return HestonParams(v0=vol * vol, kappa=2.0, theta=vol * vol, sigma=1e-9, rho=0.0)


def _assert_first_passage(engine, path_vol, tolerance_se=4.0):
    exact = _analytic_first_passage(path_vol)
    stats = engine.calculate_event_stats(_knock_in_only_snowball(), _env())
    measured = float(stats.ki_probability)
    z = (measured - exact) / _standard_error(exact)
    assert abs(z) < tolerance_se, (
        f"P(KI) {measured:.6f} vs analytic {exact:.6f} at vol {path_vol}: "
        f"{z:+.1f} standard errors"
    )


def test_the_constant_vol_bridge_still_matches_the_analytic_probability():
    """The control. Here the environment's vol IS the path vol, so the scalar
    the bridge takes is already the right one and this engine must not move."""
    engine = SnowballMCEngine(params=_params())
    stats = engine.calculate_event_stats(_knock_in_only_snowball(), _env(PATH_VOL))
    exact = _analytic_first_passage(PATH_VOL)
    z = (float(stats.ki_probability) - exact) / _standard_error(exact)
    assert abs(z) < 4.0


def test_heston_bridge_uses_the_simulated_variance_not_the_quoted_vol():
    engine = HestonSnowballMCEngine(
        model_params=_degenerate_heston(PATH_VOL), params=_params()
    )
    _assert_first_passage(engine, PATH_VOL)


def test_local_vol_bridge_uses_the_surface_not_the_quoted_vol():
    surface = LocalVolSurface(
        strike_grid=np.array([1.0, 1.0e6]),
        time_grid=np.array([0.0, 10.0]),
        lv_grid=np.full((2, 2), PATH_VOL),
    )
    engine = LocalVolSnowballMCEngine(params=_params(), local_vol_surface=surface)
    _assert_first_passage(engine, PATH_VOL)


def test_slv_bridge_carries_the_leverage():
    """Leverage 2 on variance (0.175)^2 is GBM at 35%, and only reaches the
    bridge if the leverage multiplies the variance there too."""
    leverage = LeverageSurface(
        time_grid=np.linspace(0.0, 1.0, 4),
        strike_grid=np.array(list(S0 * np.exp(np.linspace(-1.2, 1.2, 11)))),
        leverage_grid=np.full((4, 11), 2.0),
    )
    engine = HestonSLVSnowballMCEngine(
        model_params=_degenerate_heston(PATH_VOL / 2.0),
        params=_params(),
        leverage_surface=leverage,
    )
    _assert_first_passage(engine, PATH_VOL)


# ---------------------------------------------------------------------------
# Phoenix carries its own copy of the bridge and its own vol-model engines.
# ---------------------------------------------------------------------------


def _knock_in_only_phoenix():
    from quantark.asset.equity.product.option.phoenix_config import (
        CouponBarrierConfig,
    )
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
    from quantark.asset.equity.product.option.snowball_config import PayoffConfig
    from quantark.util.calendar.day_counter import DayCountConvention
    from quantark.util.enum import CouponPayType

    return PhoenixOption(
        initial_price=S0, strike=S0, maturity=T, contract_multiplier=1.0,
        is_reverse=False,
        barrier_config=BarrierConfig(
            ko_barrier=1.0e6, ko_rate=0.12,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=BARRIER,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=1.0e6, coupon_rate=0.02,
            coupon_pay_type=CouponPayType.INSTANT,
            day_count_convention=DayCountConvention.ACT_365,
            memory_coupon=False,
        ),
        payoff_config=PayoffConfig(include_principal=True),
    )


def _assert_phoenix_first_passage(engine, path_vol):
    exact = _analytic_first_passage(path_vol)
    stats = engine.calculate_event_stats(_knock_in_only_phoenix(), _env())
    z = (float(stats.ki_probability) - exact) / _standard_error(exact)
    assert abs(z) < 4.0, (
        f"P(KI) {stats.ki_probability:.6f} vs analytic {exact:.6f}: {z:+.1f} SE"
    )


def test_heston_phoenix_bridge_uses_the_simulated_variance():
    from quantark.asset.equity.engine.mc import HestonPhoenixMCEngine

    _assert_phoenix_first_passage(
        HestonPhoenixMCEngine(
            model_params=_degenerate_heston(PATH_VOL), params=_params()
        ),
        PATH_VOL,
    )


def test_local_vol_phoenix_bridge_uses_the_surface():
    from quantark.asset.equity.engine.mc import LocalVolPhoenixMCEngine

    surface = LocalVolSurface(
        strike_grid=np.array([1.0, 1.0e6]),
        time_grid=np.array([0.0, 10.0]),
        lv_grid=np.full((2, 2), PATH_VOL),
    )
    _assert_phoenix_first_passage(
        LocalVolPhoenixMCEngine(params=_params(), local_vol_surface=surface),
        PATH_VOL,
    )


def test_slv_phoenix_bridge_carries_the_leverage():
    from quantark.asset.equity.engine.mc import HestonSLVPhoenixMCEngine

    leverage = LeverageSurface(
        time_grid=np.linspace(0.0, 1.0, 4),
        strike_grid=np.array(list(S0 * np.exp(np.linspace(-1.2, 1.2, 11)))),
        leverage_grid=np.full((4, 11), 2.0),
    )
    _assert_phoenix_first_passage(
        HestonSLVPhoenixMCEngine(
            model_params=_degenerate_heston(PATH_VOL / 2.0),
            params=_params(),
            leverage_surface=leverage,
        ),
        PATH_VOL,
    )


def test_substep_refinement_folds_the_variance_onto_the_contractual_grid():
    """``substeps_per_interval`` refines the SDE but records nodes on the
    contractual grid, so the bridge spans a whole contractual interval and
    needs the variance accumulated across ALL of its substeps.  Variance is
    additive, so with a constant local vol the folded total must come back to
    sigma**2 * dt however finely the interval was stepped."""
    surface = LocalVolSurface(
        strike_grid=np.array([1.0, 1.0e6]),
        time_grid=np.array([0.0, 10.0]),
        lv_grid=np.full((2, 2), PATH_VOL),
    )
    product, env = _knock_in_only_snowball(), _env()
    recorded = {}
    for substeps in (1, 4):
        engine = LocalVolSnowballMCEngine(
            params=MCParams(num_paths=256, time_steps=52, seed=3),
            local_vol_surface=surface,
        )
        engine.substeps_per_interval = substeps
        engine.calculate_event_stats(product, env)
        recorded[substeps] = engine._step_log_variance

    _, dt_array, _, _ = LocalVolSnowballMCEngine(
        params=MCParams(num_paths=1, seed=1)
    )._build_time_grid(product, env, T)

    for substeps, h2 in recorded.items():
        assert h2.shape[1] == dt_array.size, substeps
        np.testing.assert_allclose(
            h2,
            np.broadcast_to(PATH_VOL ** 2 * dt_array, h2.shape),
            rtol=1e-12,
            atol=0,
        )
