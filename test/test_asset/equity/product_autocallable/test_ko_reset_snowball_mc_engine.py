"""
Unit tests for KO-reset snowball MC support.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantark.asset.equity.engine.event_stats import KOResetEventStats
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import create_ko_reset_snowball
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import PostKOScheduleMode
from quantark.util.exceptions import ValidationError


def create_pricing_env(
    spot: float = 100.0,
    vol: float = 0.20,
    rate: float = 0.05,
    div_yield: float = 0.02,
) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
        valuation_date=datetime(2024, 1, 1),
    )


def test_ko_reset_pricing_absolute():
    pricing_env = create_pricing_env()
    product = create_ko_reset_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity_pre=1.0,
        maturity_post=2.0,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
        ki_continuous=True,
    )
    engine = SnowballMCEngine(params=MCParams(num_paths=2000, time_steps=252))
    price = engine.price(product, pricing_env)
    assert np.isfinite(price)


def test_ko_reset_rebased_requires_discrete_ki():
    with pytest.raises(ValidationError):
        create_ko_reset_snowball(
            initial_price=100.0,
            strike=100.0,
            maturity_pre=1.0,
            maturity_post=1.0,
            post_ko_mode=PostKOScheduleMode.REBASED,
            ki_continuous=True,
        )


@pytest.mark.parametrize(
    "post_ko_mode",
    [
        PostKOScheduleMode.ABSOLUTE,
        PostKOScheduleMode.REBASED,
    ],
)
def test_ko_reset_event_stats_shape(post_ko_mode):
    pricing_env = create_pricing_env()
    product = create_ko_reset_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity_pre=1.0,
        maturity_post=2.0,
        post_ko_mode=post_ko_mode,
        ki_continuous=False,
    )
    engine = SnowballMCEngine(params=MCParams(num_paths=1000, time_steps=252))
    stats = engine.calculate_event_stats(product, pricing_env)
    assert isinstance(stats, KOResetEventStats)
    assert len(stats.pre_ko_times) == len(stats.pre_ko_probability)
    assert len(stats.post_ko_times) == len(stats.post_ko_probability)
    assert len(stats.determination_times) == len(stats.payment_times)
    assert np.all(stats.payment_times >= stats.determination_times)
    assert stats.pv == pytest.approx(
        float(stats.expected_discounted_cashflows.sum()),
        abs=1.0e-8,
    )


def _dated_ko_reset_on_business_day_clock():
    """A dated KO-reset contract whose record times all come from the environment's own clock."""
    from quantark.asset.equity.product.option import (
        KnockOutResetSnowballOption,
        ObservationRecord,
        ObservationSchedule,
    )
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
    from quantark.util.enum import ObservationType

    calendar = create_calendar(CalendarType.CHINA, year_range=(2025, 2028))
    valuation_date = datetime(2026, 1, 21)   # a business day
    assert calendar.is_business_day(valuation_date)
    pre_dates = [datetime(2026, 4, 21), datetime(2026, 7, 21)]
    post_dates = pre_dates + [datetime(2026, 10, 21), datetime(2027, 1, 21)]
    ki_dates = []
    day = valuation_date + timedelta(days=1)
    while day <= pre_dates[-1]:
        if calendar.is_business_day(day):
            ki_dates.append(day)
        day += timedelta(days=1)

    def schedule(dates, barrier, rate=None):
        return ObservationSchedule(records=[
            ObservationRecord(observation_date=d, barrier=barrier, return_rate=rate) for d in dates
        ])

    product = KnockOutResetSnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=schedule(pre_dates, 103.0, 0.15),
            ki_barrier=80.0, ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=schedule(ki_dates, 80.0), ki_continuous=False,
        ),
        post_barrier_config=BarrierConfig(
            ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=schedule(post_dates, 95.0, 0.03),
        ),
        maturity=1.0,
        initial_date=valuation_date,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
    )
    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=ContinuousDividendYield(div_yield=0.04),
        valuation_date=valuation_date,
        bus_days_in_year=244,
        day_count_convention=DayCountConvention.BUSINESS_DAYS,
        calendar=calendar,
    )
    return product, pricing_env, calendar


def test_business_day_grid_runs_on_the_environment_clock():
    """use_business_day_grid places each business day at the environment's time for it: the valuation date
    is t = 0, the next business day 1/bus_days_in_year. The grid used to count the valuation date as day 1,
    one business day ahead of the clock that times the records, so its daily nodes overran maturity."""
    from quantark.util.calendar.day_counter import calculate_year_fraction

    product, pricing_env, calendar = _dated_ko_reset_on_business_day_clock()
    engine = SnowballMCEngine(
        params=MCParams(num_paths=500, time_steps=244, use_business_day_grid=True, seed=7)
    )
    T = product.get_max_maturity_time(pricing_env)
    grid = engine._build_time_grid_ko_reset(product, pricing_env, T)

    last_date = datetime(2027, 1, 21)
    env_times = []
    day = pricing_env.valuation_date
    while day < last_date:
        day += timedelta(days=1)
        if calendar.is_business_day(day):
            env_times.append(calculate_year_fraction(
                pricing_env.valuation_date, day, pricing_env.day_count_convention,
                pricing_env.bus_days_in_year, calendar=calendar,
            ))
    np.testing.assert_array_equal(grid["all_times"], np.array(env_times))
    assert float(np.sum(grid["dt_array"])) == pytest.approx(T, abs=1e-12)
    assert np.isfinite(engine.price(product, pricing_env))
