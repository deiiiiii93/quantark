"""A dividend created from ``None`` by a bump follows the environment's declaration (0.16.0, trading-clock run D1).

On a ``BUSINESS_DAYS`` environment whose ``div_yield`` is ``None`` and declared ``CALENDAR_YEAR``, a dividend bump
used to assign a native ``ContinuousDividendYield(shift)``, read per TRADING year. The shift is a calendar-year quote
like every other cash curve the environment holds, so the bumped curve is wrapped on the environment's map.
"""
from copy import deepcopy
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.report.term_structure import BucketedDividendYield
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.asset.equity.riskmeasures.greek_conventions_report import _reprice_div
from quantark.asset.equity.riskmeasures.greeks.bump_envs import build_div_bumped_env
from quantark.param import SpotQuote
from quantark.param.div import TradingClockDividendYield
from quantark.param.div.dividend_yield import ContinuousDividendYield
from quantark.param.rrf import TradingClockRateCurve
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.vol import FlatVolSurface
from quantark.priceenv import CurveQuotingBasis, PricingEnvironment
from quantark.priceenv.clock_validation import place_created_cash_curve
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2025, 2029))
VAL = datetime(2026, 2, 9)
HORIZON = datetime(2027, 6, 9)
CAL_Q, TRD_Q = CurveQuotingBasis.CALENDAR_YEAR, CurveQuotingBasis.TRADING_YEAR
CALL = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, exercise_date=datetime(2026, 12, 15))
ENG = BlackScholesEngine()
SHIFT = 0.01


def env(div=None, quoting=CAL_Q, convention=DayCountConvention.BUSINESS_DAYS):
    kw = dict(valuation_date=VAL, spot_quote=SpotQuote(100.0), vol_surface=FlatVolSurface(0.2),
              rate_curve=FlatRateCurve(0.02), div_yield=div, day_count_convention=convention,
              bus_days_in_year=244, calendar=CAL)
    if convention == DayCountConvention.BUSINESS_DAYS:
        kw.update(rate_curve_quoting=quoting, div_yield_quoting=quoting, trading_clock_horizon=HORIZON)
    return PricingEnvironment(**kw)


def fresh_price(curve):
    return ENG.price(CALL, env(div=curve))


def test_div_rho_bump_of_none_is_calendar_quoted():
    bumped = build_div_bumped_env(env(), CALL, 0.0, SHIFT, direction=1.0)
    assert isinstance(bumped.div_yield, TradingClockDividendYield)
    assert ENG.price(CALL, bumped) == fresh_price(ContinuousDividendYield(SHIFT))


def test_greek_conventions_report_div_bump_of_none_is_calendar_quoted():
    assert _reprice_div(CALL, env(), ENG, SHIFT) == fresh_price(ContinuousDividendYield(SHIFT))


def test_carry_rhoq_bucket_bump_of_none_is_calendar_quoted(monkeypatch):
    from quantark.asset.equity.riskmeasures.bucketed_greeks import BucketedGreekCoordinate, BucketedGreeksRequest

    seen = []
    original = BlackScholesEngine.price

    def spy(self, product, pricing_env, **kw):
        seen.append(pricing_env.div_yield)
        return original(self, product, pricing_env, **kw)

    monkeypatch.setattr(BlackScholesEngine, "price", spy)
    request = BucketedGreeksRequest(coordinates=(BucketedGreekCoordinate.CARRY_RHOQ,), carry_bump=SHIFT)
    GreeksCalculator().calculate_bucketed_greeks(CALL, env(), ENG, request)
    bumped = [d for d in seen if d is not None]
    assert bumped and all(isinstance(d, TradingClockDividendYield) for d in bumped)
    assert all(isinstance(d.inner, BucketedDividendYield) for d in bumped)


@pytest.mark.parametrize("div", [None, ContinuousDividendYield(0.0), ContinuousDividendYield(0.03)],
                         ids=["none", "zero", "nonzero"])
def test_carry_rhoq_buckets_sum_to_the_parallel_rho_q(div):
    """Checkpoint finding: bucket windows on the env axis read through the wrapper's calendar time missed every
    bucket (an all-zero table). The windows are now mapped onto the calendar axis: the buckets tile the product's
    life and their one-sided sum equals the parallel one-sided rho-q, and no bucket table is all zero."""
    from quantark.asset.equity.riskmeasures.bucketed_greeks import (
        BucketedGreekCoordinate,
        BucketedGreekDifferenceMode,
        BucketedGreeksRequest,
    )

    base = env(div=div)
    request = BucketedGreeksRequest(coordinates=(BucketedGreekCoordinate.CARRY_RHOQ,), carry_bump=SHIFT,
                                    difference_mode_overrides={BucketedGreekCoordinate.CARRY_RHOQ:
                                                               BucketedGreekDifferenceMode.ONE_SIDED_UP})
    res = GreeksCalculator().calculate_bucketed_greeks(CALL, base, ENG, request)
    points = res.by_coordinate(BucketedGreekCoordinate.CARRY_RHOQ)
    assert points and all(p.status == "ok" for p in points)
    bucket_pnl = sum(p.up_price - p.base_price for p in points)
    parallel = build_div_bumped_env(base, CALL, 0.0, SHIFT, direction=1.0)
    parallel_pnl = ENG.price(CALL, parallel) - ENG.price(CALL, base)
    assert parallel_pnl < 0.0
    # Black-Scholes reads q only at expiry: exactly one bucket moves, by the parallel amount
    assert bucket_pnl == pytest.approx(parallel_pnl, rel=1e-9)


@pytest.mark.parametrize("div", [None, ContinuousDividendYield(0.03)], ids=["none", "nonzero"])
def test_carry_rhoq_custom_buckets_tile_and_survive_an_edge_past_the_horizon(div):
    """A bucket edge past the trading-clock horizon used to raise in to_calendar (re-review finding); the bumped
    windows tile the calendar axis: at every readable time exactly one bucket adds the bump."""
    from quantark.asset.equity.report.term_structure import TenorBucket
    from quantark.asset.equity.riskmeasures.bucketed_coordinates.carry_rhoq import _bucket_bump

    base = env(div=div)
    edges = [0.0, 0.25, 0.5, 5.0]
    curves = [_bucket_bump(base, a, b, SHIFT) for a, b in zip(edges[:-1], edges[1:])]
    q0 = base.div_yield.inner.get_yield if base.div_yield is not None else (lambda t: 0.0)
    for u in (0.01, 0.2, 0.25, 0.3, 0.5, 0.7, 1.2):
        c_time = curves[0].to_inner_time(u)          # the calendar time the wrapper reads its inner at
        moved = [c.inner.get_yield(c_time) - q0(c_time) for c in curves]
        assert sum(abs(m) > 0.0 for m in moved) == 1
        assert sum(moved) == pytest.approx(SHIFT, abs=1e-15)
    from quantark.asset.equity.riskmeasures.bucketed_greeks import BucketedGreekCoordinate, BucketedGreeksRequest

    request = BucketedGreeksRequest(coordinates=(BucketedGreekCoordinate.CARRY_RHOQ,), carry_bump=SHIFT,
                                    tenor_buckets=tuple(TenorBucket(label=f"b{i}", start=a, end=b)
                                                        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))))
    res = GreeksCalculator().calculate_bucketed_greeks(CALL, base, ENG, request)
    assert all(p.status == "ok" for p in res.by_coordinate(BucketedGreekCoordinate.CARRY_RHOQ))


@pytest.mark.parametrize("quoting, convention", [(TRD_Q, DayCountConvention.BUSINESS_DAYS),
                                                 (None, DayCountConvention.ACT_365)])
def test_other_environments_assign_as_before(quoting, convention):
    base = env(quoting=quoting, convention=convention)
    bumped = build_div_bumped_env(base, CALL, 0.0, SHIFT, direction=1.0)
    assert type(bumped.div_yield) is ContinuousDividendYield and bumped.div_yield.div_yield == SHIFT


def test_existing_dividend_bump_is_unchanged():
    base = env(div=ContinuousDividendYield(0.03))
    assert isinstance(base.div_yield, TradingClockDividendYield)
    bumped = build_div_bumped_env(base, CALL, 0.0, SHIFT, direction=1.0)
    assert isinstance(bumped.div_yield, TradingClockDividendYield)
    assert bumped.div_yield.time_map.anchor_date == base.div_yield.time_map.anchor_date
    assert bumped.div_yield.inner.div_yield == 0.03 + SHIFT


def test_helper_refuses_a_curve_built_over_a_wrapper():
    base = env()
    composite = BucketedDividendYield(base=TradingClockDividendYield(ContinuousDividendYield(0.0),
                                                                     base.rate_curve.time_map),
                                      bucket_start=0.0, bucket_end=1.0, bump=SHIFT)
    with pytest.raises(ValidationError, match="wrapper"):
        place_created_cash_curve(deepcopy(base), "div_yield", composite)


def test_helper_leaves_none_none():
    base = env()
    place_created_cash_curve(base, "div_yield", None)
    assert base.div_yield is None
    assert isinstance(base.rate_curve, TradingClockRateCurve)
