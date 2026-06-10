import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import create_standard_phoenix
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.cashleg import (
    AccrualLeg,
    BaseAmount,
    BaseAmountMode,
    KOBehavior,
    LegDirection,
    LegSchedule,
    PaymentConvention,
    SurvivalBasis,
    value_leg,
)
from quantark.cashleg.event_distribution import EventType
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.22),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=ContinuousDividendYield(div_yield=0.03),
        valuation_date=datetime(2026, 1, 1),
    )


def _snowball(num_observations=4):
    return create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        num_observations=num_observations,
        ko_barrier=103.0,
        ko_rate=0.10,
        ki_barrier=75.0,
        include_principal=False,
    )


def test_snowball_mc_price_with_events_adapts_event_stats():
    engine = SnowballMCEngine(params=MCParams(num_paths=3000, time_steps=80, seed=7))
    result = engine.price_with_events(_snowball(), _env())

    dist = result.event_distribution
    assert dist is not None
    assert EventType.KO in dist.probabilities
    assert dist.probabilities[EventType.KO].shape == (4,)
    assert len(dist.survival_probability) == 5
    assert np.all(np.diff(dist.survival_probability) <= 1e-9)
    total = float(np.sum(dist.probabilities[EventType.KO]))
    total += float(dist.probabilities[EventType.MATURITY_NO_KO])
    total += float(dist.probabilities[EventType.MATURITY_WITH_KI])
    assert total == pytest.approx(1.0, abs=1e-6)


def test_phoenix_mc_price_with_events_exposes_coupon_probabilities():
    phoenix = create_standard_phoenix(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        coupon_barrier=85.0,
        coupon_rate=0.01,
        num_observations=6,
        memory_coupon=True,
        coupon_pay_type=CouponPayType.INSTANT,
    )
    engine = PhoenixMCEngine(params=MCParams(num_paths=3000, time_steps=80, seed=11))
    result = engine.price_with_events(phoenix, _env())
    dist = result.event_distribution

    assert dist is not None
    assert EventType.KO in dist.probabilities
    assert EventType.COUPON in dist.probabilities
    assert dist.probabilities[EventType.COUPON].shape == dist.event_times.shape


def test_snowball_quad_price_with_events_uses_existing_stats_adapter():
    engine = SnowballQuadEngine(params=QuadParams(grid_points=201))
    result = engine.price_with_events(_snowball(num_observations=3), _env())
    assert result.event_distribution is not None
    assert result.event_distribution.probabilities[EventType.KO].shape == (3,)


def test_snowball_pde_price_with_events_uses_existing_stats_adapter():
    engine = SnowballPDESolver(params=PDEParams(grid_size=60, time_steps=40))
    result = engine.price_with_events(_snowball(num_observations=3), _env())
    assert result.event_distribution is not None
    assert result.event_distribution.probabilities[EventType.KO].shape == (3,)


def test_accrual_leg_truncated_by_snowball_ko_is_smaller_than_full_schedule():
    engine = SnowballMCEngine(params=MCParams(num_paths=5000, time_steps=80, seed=5))
    result = engine.price_with_events(_snowball(), _env())
    schedule = LegSchedule(
        period_starts=np.array([0.0, 0.25, 0.5, 0.75]),
        period_ends=np.array([0.25, 0.5, 0.75, 1.0]),
        payment_times=np.array([0.25, 0.5, 0.75, 1.0]),
    )
    base = BaseAmount(value=1_000_000.0, mode=BaseAmountMode.ABSOLUTE)
    common = dict(
        rate=0.04,
        base=base,
        schedule=schedule,
        day_count=DayCountConvention.ACT_365,
        payment_convention=PaymentConvention.AT_PERIOD_END,
        survival_basis=SurvivalBasis.ENTER_PERIOD,
        direction=LegDirection.BUYER_RECEIVES,
    )
    leg_full = AccrualLeg(**common, ko_behavior=KOBehavior.PAY_FULL_SCHEDULE)
    leg_trunc = AccrualLeg(**common, ko_behavior=KOBehavior.TRUNCATE_AT_KO)

    pv_full = value_leg(leg_full, result.event_distribution, _env(), 0.0)
    pv_trunc = value_leg(leg_trunc, result.event_distribution, _env(), 0.0)
    assert 0.0 < pv_trunc < pv_full
