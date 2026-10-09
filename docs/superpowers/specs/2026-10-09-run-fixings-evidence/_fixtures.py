"""Shared fixtures for the run-fixings probes (spec 2026-10-09-run-fixings-design.md).

One path, one contract shape, every run path. The path holds spot inside the [90, 110] band until the first fixing
(run day FIX_DAY), then ramps OUT of the band, so a fixing re-decided at a later day's spot flips from in-range to
out-of-range (range accrual) or takes a higher price (Asian).
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

from quantark.asset.equity.product.option import (
    RangeAccrualConfig,
    RangeAccrualObservationRecord,
    RangeAccrualOption,
)
from quantark.asset.equity.product.option.asian_option import AsianObservationRecord, AsianOption
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import AsianStrikeType, AveragingType, OptionType
from quantark.util.fixings import MissingFixingWarning
from quantark.util.marketdata.models import MarketDataSet, TimeSeriesData

START = datetime(2024, 1, 1)          # a Monday
DAYS = 25                             # calendar days in the dynamic scenario / replay paths
FIX_DAY = 4                           # the first fixing is on run day 4 (calendar day START + 4)
FIX2_DAY = 18                         # a second fixing inside the run
VOL, RATE, DIV = 0.25, 0.05, 0.02


def spot_on(day: int) -> float:
    """105 up to the first fixing (inside [90, 110]), then +1 a day: 115 by day 14, out of the band."""
    return 105.0 if day <= FIX_DAY else 105.0 + (day - FIX_DAY)


def fixing_dates() -> List[datetime]:
    return [START + timedelta(days=FIX_DAY), START + timedelta(days=FIX2_DAY),
            START + timedelta(days=120), START + timedelta(days=240), START + timedelta(days=365)]


def ra_dated(outcomes=None) -> RangeAccrualOption:
    outcomes = outcomes or {}
    return RangeAccrualOption(
        initial_price=100.0,
        range_config=RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0, accrual_rate=0.10),
        exercise_date=START + timedelta(days=365),
        initial_date=START,
        observation_records=[
            RangeAccrualObservationRecord(observation_date=d, observed_in_range=outcomes.get(i))
            for i, d in enumerate(fixing_dates())
        ],
    )


def ra_numeric(outcomes=None) -> RangeAccrualOption:
    """The numeric twin at START (times = calendar days / 365 from START; env CALENDAR_DAYS)."""
    outcomes = outcomes or {}
    return RangeAccrualOption(
        initial_price=100.0,
        range_config=RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0, accrual_rate=0.10),
        maturity=1.0,
        observation_records=[
            RangeAccrualObservationRecord(observation_time=(d - START).days / 365.0,
                                          observed_in_range=outcomes.get(i))
            for i, d in enumerate(fixing_dates())
        ],
    )


def asian_dated(prices=None) -> AsianOption:
    prices = prices or {}
    return AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        asian_strike_type=AsianStrikeType.FIXED,
        averaging_type=AveragingType.ARITHMETIC,
        exercise_date=START + timedelta(days=365),
        observation_records=[
            AsianObservationRecord(observation_date=d, observed_price=prices.get(i))
            for i, d in enumerate(fixing_dates())
        ],
    )


def asian_numeric(prices=None) -> AsianOption:
    prices = prices or {}
    return AsianOption(
        strike=100.0,
        option_type=OptionType.CALL,
        asian_strike_type=AsianStrikeType.FIXED,
        averaging_type=AveragingType.ARITHMETIC,
        maturity=1.0,
        observation_records=[
            AsianObservationRecord(observation_time=(d - START).days / 365.0, observed_price=prices.get(i))
            for i, d in enumerate(fixing_dates())
        ],
    )


def eq_env(day: int, date: Optional[datetime] = None, spot: Optional[float] = None) -> PricingEnvironment:
    return PricingEnvironment(
        valuation_date=date if date is not None else START + timedelta(days=day),
        spot_quote=SpotQuote(spot=spot if spot is not None else spot_on(day)),
        vol_surface=FlatVolSurface(volatility=VOL),
        rate_curve=FlatRateCurve(rate=RATE),
        div_yield=ContinuousDividendYield(div_yield=DIV),
    )


def recorded_by_hand(kind: str, day: int, day_of_date):
    """The contract with every fixing dated on/before run day ``day`` recorded at ITS OWN day's spot."""
    passed = {i: day_of_date(d) for i, d in enumerate(fixing_dates()) if day_of_date(d) <= day}
    if kind == "ra":
        prod = ra_dated()
        return ra_dated({i: prod.is_in_range(spot_on(k), i, prod.get_observation_records()[i])
                         for i, k in passed.items()})
    return asian_dated({i: spot_on(k) for i, k in passed.items()})


class DailyAdapter:
    """Market data adapter over calendar days START .. START + n - 1 at ``spot_on``."""

    def __init__(self, n: int, freq: str = "D"):
        self.dates = pd.date_range(START, periods=n, freq=freq)
        self.days = [(d.to_pydatetime() - START).days for d in self.dates]

    def get_market_data_set(self, asset_name, start_date, end_date, currency, frequency):
        idx = self.dates
        frame = lambda col, vals: TimeSeriesData(pd.DataFrame({col: vals}, index=idx), asset_name, col)  # noqa: E731
        return MarketDataSet(
            spot_data=frame("spot", [spot_on(k) for k in self.days]),
            vol_data=frame("volatility", [VOL] * len(idx)),
            rate_data=frame("rate", [RATE] * len(idx)),
            div_yield_data=frame("div_yield", [DIV] * len(idx)),
        )


def count_missing(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, sum(issubclass(w.category, MissingFixingWarning) for w in caught)
