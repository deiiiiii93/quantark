"""Single-pass price_with_events: NPV parity + short-circuits + stream wiring.

Plan Task 2.4 [§11.2, §11.3]: price_with_events returns the exact price() NPV
(including the expired / immediate-KO / already-KO short-circuits) plus an event
distribution, from one value sweep reused for the residual.
"""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde import PhoenixPDESolver, SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.cashleg.event_distribution import EventType
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType


def _env(spot=100.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1),
    )


def _snowball(ki_continuous=True):
    cfg = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[i / 12 for i in range(1, 13)],
        ki_barrier=75.0,
        ki_observation_type=(
            ObservationType.CONTINUOUS if ki_continuous else ObservationType.DISCRETE
        ),
        ki_continuous=ki_continuous,
    )
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=cfg,
        contract_multiplier=10_000.0,
        maturity=1.0,
    )


def _phoenix():
    ko_dates = [i / 4 for i in range(1, 5)]
    bc = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.0,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=ko_dates,
        ki_barrier=None,
    )
    cc = CouponBarrierConfig(
        coupon_barrier=85.0,
        coupon_rate=0.02,
        coupon_pay_type=CouponPayType.INSTANT,
        day_count_convention=DayCountConvention.ACT_365,
        memory_coupon=True,
    )
    pf = PayoffConfig(rebate_rate=0.0, include_principal=True)
    return PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=bc,
        coupon_config=cc,
        payoff_config=pf,
        contract_multiplier=100.0,
        maturity=1.0,
    )


def test_npv_matches_price_normal():
    solver = SnowballPDESolver(PDEParams())
    env = _env()
    product = _snowball()
    assert abs(solver.price_with_events(product, env).npv - solver.price(product, env)) < 1e-8


def test_npv_matches_price_immediate_ko():
    # spot already above the KO barrier => immediate-KO short-circuit [§11.3]
    solver = SnowballPDESolver(PDEParams())
    env = _env(spot=130.0)
    product = _snowball()
    res = solver.price_with_events(product, env)
    assert abs(res.npv - solver.price(product, env)) < 1e-8
    assert res.event_distribution is not None


def test_phoenix_npv_matches_price():
    solver = PhoenixPDESolver(PDEParams())
    env = _env()
    product = _phoenix()
    assert abs(solver.price_with_events(product, env).npv - solver.price(product, env)) < 1e-8


def test_lean_streams_preserve_npv_and_ko():
    # A KO-only stream request prunes KI columns but must not move NPV, and the
    # KO probabilities in the distribution stay identical to the full run.
    solver = SnowballPDESolver(PDEParams())
    env = _env()
    product = _snowball()
    full = solver.price_with_events(product, env, streams=None)
    lean = solver.price_with_events(product, env, streams=frozenset({EventType.KO}))
    assert abs(full.npv - lean.npv) < 1e-10
    ko_full = np.asarray(full.event_distribution.probabilities[EventType.KO])
    ko_lean = np.asarray(lean.event_distribution.probabilities[EventType.KO])
    assert np.array_equal(ko_full, ko_lean)


def test_emit_distribution_false_is_trivial():
    solver = SnowballPDESolver(PDEParams())
    env = _env()
    product = _snowball()
    res = solver.price_with_events(product, env, emit_distribution=False)
    assert abs(res.npv - solver.price(product, env)) < 1e-8
    assert res.event_distribution is not None
