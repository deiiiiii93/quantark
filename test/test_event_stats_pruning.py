"""Event-stats stream pruning + npv passthrough (Phase 2a, plan Task 2.4 slice).

Golden invariant: pruning the KI indicator columns and/or Phoenix coupon columns
(when no leg requests them) leaves ko_probability / survival / pv bit-identical —
the KI regime jump still runs; only auxiliary columns are dropped [§11.1]. And a
caller-supplied npv reproduces the internal self.price() result exactly.
"""

from datetime import datetime

import numpy as np

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


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1),
    )


def _snowball():
    cfg = BarrierConfig(
        ko_barrier=103.0,
        ko_rate=0.15,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=[i / 12 for i in range(1, 13)],
        ki_barrier=75.0,
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


# Golden values captured from the pre-2a code (commit before stream pruning).
SNOWBALL_PV = 988334.2318758955
SNOWBALL_KI = 0.1103589664
SNOWBALL_KI_EVER = 0.1478791194


def test_snowball_full_stream_matches_golden():
    stats = SnowballPDESolver(PDEParams(grid_size=200)).calculate_event_stats(
        _snowball(), _env()
    )
    assert abs(float(stats.pv) - SNOWBALL_PV) < 1e-6
    assert abs(float(stats.ki_probability) - SNOWBALL_KI) < 1e-9
    assert abs(float(stats.ki_ever_probability) - SNOWBALL_KI_EVER) < 1e-9


def test_snowball_ki_pruning_preserves_ko_and_pv():
    solver = SnowballPDESolver(PDEParams(grid_size=200))
    env = _env()
    product = _snowball()
    full = solver._compute_event_stats(product, env, streams=None)
    lean = solver._compute_event_stats(
        product, env, streams=frozenset({EventType.KO})
    )
    # KO/survival/pv bit-identical whether or not KI columns are computed.
    assert np.array_equal(
        np.asarray(full.ko_probability), np.asarray(lean.ko_probability)
    )
    assert np.array_equal(
        np.asarray(full.survival_probability),
        np.asarray(lean.survival_probability),
    )
    assert float(full.pv) == float(lean.pv)
    # KI pruned => reported as 0 (no leg needs it).
    assert float(lean.ki_probability) == 0.0
    assert float(lean.ki_ever_probability) == 0.0
    # ...but the full path still reports the real KI mass.
    assert float(full.ki_probability) > 0.0


def test_npv_passthrough_skips_internal_price():
    solver = SnowballPDESolver(PDEParams(grid_size=200))
    env = _env()
    product = _snowball()
    internal = solver._compute_event_stats(product, env)
    sentinel = 123456.0
    passed = solver._compute_event_stats(product, env, npv=sentinel)
    assert float(passed.pv) == sentinel
    # maturity residual shifts by exactly the pv delta; KO cashflows unchanged.
    assert np.array_equal(
        np.asarray(internal.expected_discounted_ko_cashflow),
        np.asarray(passed.expected_discounted_ko_cashflow),
    )
    delta = sentinel - float(internal.pv)
    assert abs(
        float(passed.expected_discounted_maturity_cashflow)
        - (float(internal.expected_discounted_maturity_cashflow) + delta)
    ) < 1e-6


def test_phoenix_coupon_pruning_preserves_ko_and_pv():
    solver = PhoenixPDESolver(PDEParams(grid_size=150))
    env = _env()
    product = _phoenix()
    full = solver._compute_event_stats(product, env, streams=None)
    lean = solver._compute_event_stats(
        product, env, streams=frozenset({EventType.KO})
    )
    assert np.array_equal(
        np.asarray(full.ko_probability), np.asarray(lean.ko_probability)
    )
    assert float(full.pv) == float(lean.pv)
    # Coupon columns pruned => coupon_probability empty; full path has them.
    assert np.asarray(full.coupon_probability).size > 0
    assert np.asarray(lean.coupon_probability).size == 0
