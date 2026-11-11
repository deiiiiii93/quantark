"""Phoenix MC coupon cash is the cash the price pays, on the price's own paths.

Two defects of the reference engine (unified event statistics spec, DESIGN §3):

* under EXPIRY pay every coupon was discounted at the terminal payment, even on paths that
  knock out early and pay the rolled-up coupons at the knock-out settlement;
* under memory a knock-out releases the arrears whatever the current coupon barrier says,
  and the statistics dropped them when the barrier was missed.

The check is independent of the engine's coupon bookkeeping: it captures the paths the
engine simulated, prices the maturity redemption directly on them, and requires the coupon
cash to close ``pv - KO cash - maturity cash`` to round-off. Non-memory rows are also checked
one by one against a per-path recomputation.
"""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import PhoenixOption
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.numerical import Tolerance

OBS = [0.25, 0.5, 0.75, 1.0]
KO_BARRIER = 103.0
# The second observation's coupon barrier sits ABOVE the knock-out barrier, so a path can
# knock out there while missing the coupon: the memory release rule then pays arrears only.
COUPON_BARRIERS = [85.0, 108.0, 85.0, 85.0]


def _env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.25),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2026, 1, 1),
    )


def _phoenix(*, memory: bool, pay_type: CouponPayType) -> PhoenixOption:
    return PhoenixOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=KO_BARRIER,
            ko_rate=0.0,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=list(OBS),
            ki_barrier=70.0,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=list(COUPON_BARRIERS),
            coupon_rate=0.03,
            coupon_pay_type=pay_type,
            day_count_convention=DayCountConvention.ACT_365,
            memory_coupon=memory,
        ),
        payoff_config=PayoffConfig(include_principal=True),
        maturity=1.0,
        settlement_convention=SettlementConvention(
            lag=0.1, lag_unit=SettlementLagUnit.YEAR_FRACTION
        ),
    )


def _run(product, env):
    """calculate_event_stats, plus the paths and per-path outcomes it priced."""
    engine = PhoenixMCEngine(params=MCParams(seed=7, num_paths=20_000, use_qmc=False))
    captured = {}
    original = engine._compute_payoffs

    def spy(product_, env_, paths, all_times, ko_indices, *args, **kwargs):
        result = original(product_, env_, paths, all_times, ko_indices, *args, **kwargs)
        captured.update(paths=paths, ko_indices=ko_indices, result=result)
        return result

    engine._compute_payoffs = spy
    stats = engine.calculate_event_stats(product, env)
    return engine, stats, captured


def _maturity_cash(engine, product, env, paths, outcome):
    """Discounted maturity redemption (no coupons) of every surviving path, averaged."""
    terminal = engine._payment_timings.terminal
    cash = np.zeros(len(paths), dtype=float)
    for i in np.flatnonzero(outcome["is_v0"]):
        cash[i] = product.get_maturity_payoff_v0(float(paths[i, -1]), pricing_env=env)
    for i in np.flatnonzero(outcome["is_v1"]):
        cash[i] = product.get_maturity_payoff_v1(float(paths[i, -1]), pricing_env=env)
    return float(np.mean(cash * engine._df(np.full(len(paths), terminal.payment_time))))


@pytest.mark.parametrize("memory", [False, True], ids=["plain", "memory"])
@pytest.mark.parametrize(
    "pay_type", [CouponPayType.INSTANT, CouponPayType.EXPIRY], ids=["instant", "expiry"]
)
def test_coupon_cash_closes_the_price_on_its_own_paths(memory, pay_type):
    env = _env()
    product = _phoenix(memory=memory, pay_type=pay_type)
    engine, stats, captured = _run(product, env)
    outcome = captured["result"][2]

    maturity = _maturity_cash(engine, product, env, captured["paths"], outcome)
    coupon_cash_implied = stats.pv - float(np.sum(stats.observations.discounted_cash)) - maturity

    assert float(np.sum(stats.coupons.discounted_cash)) == pytest.approx(
        coupon_cash_implied, abs=1.0e-12 * max(1.0, abs(stats.pv))
    )


@pytest.mark.parametrize(
    "pay_type", [CouponPayType.INSTANT, CouponPayType.EXPIRY], ids=["instant", "expiry"]
)
def test_plain_coupon_rows_discount_at_the_actual_payment(pay_type):
    env = _env()
    product = _phoenix(memory=False, pay_type=pay_type)
    engine, stats, captured = _run(product, env)
    paths, ko_indices = captured["paths"], captured["ko_indices"]
    payoffs, settlement_times, outcome = captured["result"][:3]
    is_ko, first_ko_idx = outcome["is_ko"], outcome["first_ko_idx"]
    timings = engine._payment_timings
    periods = product.get_coupon_period_year_fractions(env)

    for k in range(len(OBS)):
        alive_before = (~is_ko) | (first_ko_idx >= k)
        hit = alive_before & (paths[:, ko_indices[k]] >= COUPON_BARRIERS[k])
        coupon = product.get_coupon_payoff(k, year_fraction=periods[k])
        if pay_type == CouponPayType.INSTANT:
            # paid at the observation's own settlement (a knock-out here settles there too)
            df = np.full(len(paths), float(timings.observation_payment_dfs[k]))
            ko_here = is_ko & (first_ko_idx == k)
            df[ko_here] = engine._df(settlement_times[ko_here])
        else:
            # rolled up and paid when the note ends: the path's own settlement
            df = engine._df(settlement_times)
        expected = float(np.mean(np.where(hit, coupon, 0.0) * df))
        assert stats.coupons.discounted_cash[k] == pytest.approx(
            expected, abs=Tolerance.ZERO
        ), f"row {k}"


def test_memory_knock_out_on_a_missed_coupon_releases_the_arrears():
    """The scenario the second defect hides in must actually occur in this test."""
    env = _env()
    product = _phoenix(memory=True, pay_type=CouponPayType.INSTANT)
    _, _, captured = _run(product, env)
    paths, ko_indices = captured["paths"], captured["ko_indices"]
    outcome = captured["result"][2]
    ko_on_missed = (
        outcome["is_ko"]
        & (outcome["first_ko_idx"] == 1)
        & (paths[:, ko_indices[1]] < COUPON_BARRIERS[1])
    )
    assert ko_on_missed.sum() > 100
