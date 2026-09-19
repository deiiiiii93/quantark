"""Shared intraday fixtures: SSE calendar (fixed +08:00), dated snowball, digital."""
from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping
from datetime import datetime, time, timedelta, timezone
from enum import Enum

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


def dated_snowball(cal, t0, months=12, ko=103.0, ki=75.0, ko_rate=0.12, rebate_rate=None):
    """Monthly KO dates from t0 with daily-close discrete KI on the same dates.

    ``rebate_rate`` defaults to the knock-out coupon. Pass another rate where a test must tell a knock-out from a
    survival: with equal rates the two pay the same cash, and an engine that misses the knock-out looks right."""
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
        payoff_config=PayoffConfig(rebate_rate=ko_rate if rebate_rate is None else rebate_rate, include_principal=False),
    )


def dated_phoenix(cal, t0, months=12, ko=103.0, ki=75.0, coupon_barrier=80.0, coupon_rate=0.12, memory=True):
    """Monthly KO/coupon dates from t0 with discrete KI on the same dates; coupons accrue ACT/365 per period."""
    from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
    dates = _monthly_trading_dates(cal, t0, months)
    ko_schedule = ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=ko) for d in dates])
    ki_schedule = ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=ki) for d in dates])
    return PhoenixOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0,
        initial_date=t0, exercise_date=dates[-1],
        barrier_config=BarrierConfig(
            ko_barrier=ko, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ko_schedule,
            ki_barrier=ki, ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ki_schedule),
        coupon_config=CouponBarrierConfig(coupon_barrier=coupon_barrier, coupon_rate=coupon_rate, memory_coupon=memory),
        payoff_config=PayoffConfig(include_principal=True),
    )


def dated_ko_reset(cal, t0, pre_months=6, post_months=12, pre_ko=103.0, post_ko=95.0, ki=75.0, pre_rate=0.15, post_rate=0.03):
    """Monthly pre-KI KO dates for ``pre_months``, post-KI KO dates for ``post_months`` (absolute), monthly discrete KI."""
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
    pre_dates = _monthly_trading_dates(cal, t0, pre_months)
    post_dates = _monthly_trading_dates(cal, t0, post_months)
    record = lambda d, b: ObservationRecord(observation_date=d, barrier=b)   # noqa: E731
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, initial_date=t0, exercise_date=post_dates[-1],
        barrier_config=BarrierConfig(
            ko_barrier=pre_ko, ko_rate=pre_rate, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ObservationSchedule(records=[record(d, pre_ko) for d in pre_dates]),
            ki_barrier=ki, ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ObservationSchedule(records=[record(d, ki) for d in post_dates])),
        post_barrier_config=BarrierConfig(
            ko_barrier=post_ko, ko_rate=post_rate, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ObservationSchedule(records=[record(d, post_ko) for d in post_dates])),
        payoff_config=PayoffConfig(rebate_rate=pre_rate, include_principal=False),
    )


def digital(expiry, strike=100.0, payout=1.0, option_type=OptionType.CALL):
    return CashOrNothingDigitalOption(strike=strike, option_type=option_type, payout=payout, exercise_date=expiry)


def _structure(obj, seen):
    """Structural value of ``obj``: leaves via canonical_tree, containers and plain objects walked field by field."""
    is_composite = isinstance(obj, (list, tuple, set, frozenset, Mapping)) or (
        not isinstance(obj, (type, Enum)) and (dataclasses.is_dataclass(obj) or hasattr(obj, "__dict__")))
    if not is_composite:
        try:
            return canonical_tree(obj)
        except Uncanonicalizable:
            return ("repr", repr(obj))
    if id(obj) in seen:
        return ("cycle", type(obj).__qualname__)
    seen = seen | {id(obj)}
    if isinstance(obj, (list, tuple)):
        return ("seq", tuple(_structure(x, seen) for x in obj))
    if isinstance(obj, (set, frozenset)):
        return ("set", tuple(sorted((_structure(x, seen) for x in obj), key=repr)))
    if isinstance(obj, Mapping):
        return ("map", tuple(sorted(((repr(k), _structure(v, seen)) for k, v in obj.items()), key=lambda kv: kv[0])))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = [f.name for f in dataclasses.fields(obj)]
        extra = sorted(set(vars(obj)) - set(fields)) if hasattr(obj, "__dict__") else []
        return ("dc", type(obj).__qualname__, tuple((n, _structure(getattr(obj, n), seen)) for n in fields + extra))
    if hasattr(obj, "__dict__"):
        return ("obj", type(obj).__qualname__, tuple((k, _structure(v, seen)) for k, v in sorted(vars(obj).items())))
    return ("repr", repr(obj))


def snapshot(obj):
    """Deep structural snapshot (no memory addresses), including attributes set outside dataclass fields."""
    return _structure(copy.deepcopy(obj), frozenset())


def _assert_unchanged(before, obj):
    assert snapshot(obj) == before, "intraday valuation mutated a caller-owned input"
