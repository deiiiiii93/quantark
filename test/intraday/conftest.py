"""Shared intraday fixtures: SSE calendar (fixed +08:00), dated snowball, digital."""
from __future__ import annotations

import copy
from datetime import datetime, time, timedelta, timezone

import pytest

from quantark.intraday.session import TradingSession, TradingSessionCalendar
from quantark.asset.equity.product.option.digital_option import CashOrNothingDigitalOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.execution.cache.fingerprint import Uncanonicalizable, canonical_tree
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum.option_enums import ObservationType, OptionType

SHANGHAI = timezone(timedelta(hours=8))  # fixed offset: no tz database needed


@pytest.fixture(scope="session")
def sse_calendar():
    return create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))


@pytest.fixture(scope="session")
def sse_sessions(sse_calendar):
    return TradingSessionCalendar(name="SSE", tz=SHANGHAI, calendar=sse_calendar,
                                  sessions=(TradingSession(time(9, 30), time(11, 30)),
                                            TradingSession(time(13, 0), time(15, 0))))


def flat_env(valuation, spot=100.0, vol=0.20, r=0.03, q=0.01, spot_ts=None):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(r),
        valuation_date=valuation,
        spot_quote=SpotQuote(spot, timestamp=spot_ts if spot_ts is not None else valuation),
        vol_surface=FlatVolSurface(vol),
        div_yield=ContinuousDividendYield(q),
    )


def _monthly_trading_dates(cal, t0, months):
    out = []
    for k in range(1, months + 1):
        month = (t0.month - 1 + k) % 12 + 1
        year = t0.year + (t0.month - 1 + k) // 12
        d = datetime(year, month, min(t0.day, 28))
        while not cal.is_business_day(d):
            d += timedelta(days=1)
        out.append(d)
    return out


def dated_snowball(cal, t0, months=12, ko=103.0, ki=75.0, ko_rate=0.12):
    """Monthly KO dates from t0 with daily-close discrete KI on the same dates."""
    dates = _monthly_trading_dates(cal, t0, months)
    ko_schedule = ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=ko) for d in dates])
    ki_schedule = ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=ki) for d in dates])
    return SnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0,
        initial_date=t0, exercise_date=dates[-1],
        barrier_config=BarrierConfig(
            ko_barrier=ko, ko_rate=ko_rate, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ko_schedule,
            ki_barrier=ki, ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ki_schedule),
        payoff_config=PayoffConfig(rebate_rate=ko_rate, include_principal=False),
    )


def digital(expiry, strike=100.0, payout=1.0, option_type=OptionType.CALL):
    return CashOrNothingDigitalOption(strike=strike, option_type=option_type, payout=payout, exercise_date=expiry)


def snapshot(obj):
    try:
        return canonical_tree(obj)
    except Uncanonicalizable:
        return repr(copy.deepcopy(obj))


def _assert_unchanged(before, obj):
    assert snapshot(obj) == before, "intraday valuation mutated a caller-owned input"
