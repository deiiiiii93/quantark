"""Phase 4: opt-in BGK continuous-KI mode for the Snowball PDE solver [§11.6].

The exact discrete-KI path stays the default; ``KnockInMonitoringMode.BGK_APPROXIMATION``
replaces a dense daily KI schedule with continuous monitoring at a
Broadie-Glasserman-Kou (1997) shifted barrier. It engages ONLY for
discretely-monitored KI (European / continuous / no-KI are inert with a log),
shifts the barrier AWAY from spot (standard down-KI down, reverse up-KI up),
registers the shifted barrier as a spatial critical point, and leaves the
KO/coupon probabilities unchanged (they are KI-independent).
"""

import logging
import math
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import KnockInMonitoringMode, ObservationType
from quantark.util.exceptions import ValidationError

BUS_DAYS = 252
BETA = 0.5825971579390107  # -zeta(1/2)/sqrt(2*pi)


def _env(vol=0.25):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1),
        bus_days_in_year=BUS_DAYS,
    )


def _daily_ki_dates():
    return [i / BUS_DAYS for i in range(1, BUS_DAYS + 1)]


def _daily_ki_snowball(ki_barrier=75.0):
    """Standard snowball, dense daily discrete KI, monthly KO."""
    cfg = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[i / 12 for i in range(1, 13)],
        ki_barrier=ki_barrier,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=_daily_ki_dates(),
        ki_continuous=False,
    )
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=cfg,
        contract_multiplier=10_000.0,
        maturity=1.0,
    )


def _reverse_daily_ki_snowball(ki_barrier=125.0):
    """Reverse snowball (UP-KI), dense daily discrete KI, monthly (DOWN) KO."""
    cfg = BarrierConfig(
        ko_barrier=97.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[i / 12 for i in range(1, 13)],
        ki_barrier=ki_barrier,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=_daily_ki_dates(),
        ki_continuous=False,
    )
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=cfg,
        contract_multiplier=10_000.0,
        maturity=1.0,
        is_reverse=True,
    )


def _continuous_ki_snowball(ki_barrier=75.0):
    cfg = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[i / 12 for i in range(1, 13)],
        ki_barrier=ki_barrier,
        ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True,
    )
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=cfg,
        contract_multiplier=10_000.0,
        maturity=1.0,
    )


def _no_ki_snowball():
    cfg = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[i / 12 for i in range(1, 13)],
    )
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=cfg,
        contract_multiplier=10_000.0,
        maturity=1.0,
    )


# --------------------------------------------------------------------------
# Task 4.1 — PDEParams.ki_monitoring_mode
# --------------------------------------------------------------------------


def test_ki_monitoring_mode_defaults_to_exact_discrete():
    assert PDEParams().ki_monitoring_mode is KnockInMonitoringMode.EXACT_DISCRETE


def test_ki_monitoring_mode_accepts_string_and_rejects_unknown():
    params = PDEParams(ki_monitoring_mode="BGK_APPROXIMATION")
    assert params.ki_monitoring_mode is KnockInMonitoringMode.BGK_APPROXIMATION
    with pytest.raises(ValidationError, match="ki_monitoring_mode"):
        PDEParams(ki_monitoring_mode="continuous_magic")


# --------------------------------------------------------------------------
# Task 4.2(a) — direction-aware BGK shift AWAY from spot
# --------------------------------------------------------------------------


def test_bgk_shift_standard_down_ki_shifts_down():
    env = _env(vol=0.25)
    product = _daily_ki_snowball(ki_barrier=75.0)
    engine = SnowballPDESolver(
        PDEParams(grid_size=100, ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION)
    )
    sigma = env.get_vol(product.strike, product.get_maturity(env))
    shifted = engine._bgk_shifted_ki_barrier(product, env, sigma)
    expected = 75.0 * math.exp(-BETA * sigma * math.sqrt(1.0 / BUS_DAYS))
    assert shifted < 75.0
    assert shifted == pytest.approx(expected, rel=1e-12)


def test_bgk_shift_reverse_up_ki_shifts_up():
    env = _env(vol=0.25)
    product = _reverse_daily_ki_snowball(ki_barrier=125.0)
    engine = SnowballPDESolver(
        PDEParams(grid_size=100, ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION)
    )
    sigma = env.get_vol(product.strike, product.get_maturity(env))
    shifted = engine._bgk_shifted_ki_barrier(product, env, sigma)
    expected = 125.0 * math.exp(+BETA * sigma * math.sqrt(1.0 / BUS_DAYS))
    assert shifted > 125.0
    assert shifted == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------
# Task 4.2(b) — shifted barrier is a spatial critical point
# --------------------------------------------------------------------------


def test_bgk_shifted_barrier_is_a_critical_point():
    env = _env(vol=0.25)
    product = _daily_ki_snowball(ki_barrier=75.0)
    engine = SnowballPDESolver(
        PDEParams(grid_size=100, ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION)
    )
    sigma = env.get_vol(product.strike, product.get_maturity(env))
    shifted = engine._bgk_shifted_ki_barrier(product, env, sigma)
    points = engine.get_critical_points(product, env)
    assert any(abs(p - shifted) < 1e-9 for p in points)

    # In EXACT_DISCRETE the shifted barrier is NOT injected.
    exact_engine = SnowballPDESolver(PDEParams(grid_size=100))
    exact_points = exact_engine.get_critical_points(product, env)
    assert not any(abs(p - shifted) < 1e-9 for p in exact_points)


# --------------------------------------------------------------------------
# Task 4.2(c) — engages only for daily-discrete KI; inert + log otherwise
# --------------------------------------------------------------------------


def test_bgk_inert_for_continuous_ki_price_unchanged_and_logs(caplog):
    env = _env(vol=0.25)
    product = _continuous_ki_snowball()
    exact = SnowballPDESolver(PDEParams(grid_size=150)).price(product, env)
    bgk_engine = SnowballPDESolver(
        PDEParams(grid_size=150, ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION)
    )
    with caplog.at_level(logging.WARNING):
        bgk = bgk_engine.price(product, env)
    assert bgk == pytest.approx(exact, abs=1e-9)
    assert any("inert" in rec.message.lower() for rec in caplog.records)


def test_bgk_inert_for_no_ki_price_unchanged(caplog):
    env = _env(vol=0.25)
    product = _no_ki_snowball()
    exact = SnowballPDESolver(PDEParams(grid_size=150)).price(product, env)
    bgk_engine = SnowballPDESolver(
        PDEParams(grid_size=150, ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION)
    )
    with caplog.at_level(logging.WARNING):
        bgk = bgk_engine.price(product, env)
    assert bgk == pytest.approx(exact, abs=1e-9)
    assert any("inert" in rec.message.lower() for rec in caplog.records)


# --------------------------------------------------------------------------
# Task 4.2(d) — BGK price near exact discrete for a far-below-spot KI
# --------------------------------------------------------------------------


def test_bgk_price_within_band_of_exact_discrete():
    env = _env(vol=0.25)
    product = _daily_ki_snowball(ki_barrier=75.0)
    exact = SnowballPDESolver(PDEParams(grid_size=200)).price(product, env)
    bgk = SnowballPDESolver(
        PDEParams(grid_size=200, ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION)
    ).price(product, env)
    rel = abs(bgk - exact) / max(abs(exact), 1.0)
    assert rel <= 0.015, f"BGK {bgk} vs exact {exact} rel={rel}"


# --------------------------------------------------------------------------
# Task 4.2(e) — KO/coupon probabilities are KI-independent across modes
# --------------------------------------------------------------------------


def test_ko_probabilities_identical_between_modes():
    env = _env(vol=0.25)
    product = _daily_ki_snowball(ki_barrier=75.0)
    exact_stats = SnowballPDESolver(PDEParams(grid_size=200)).calculate_event_stats(
        product, env
    )
    bgk_stats = SnowballPDESolver(
        PDEParams(grid_size=200, ki_monitoring_mode=KnockInMonitoringMode.BGK_APPROXIMATION)
    ).calculate_event_stats(product, env)
    assert np.allclose(
        exact_stats.ko_probability, bgk_stats.ko_probability, atol=5e-3, rtol=0.0
    )
    assert np.allclose(
        exact_stats.survival_probability,
        bgk_stats.survival_probability,
        atol=5e-3,
        rtol=0.0,
    )
