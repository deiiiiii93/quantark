"""Date forms for the equity times that had only a float form (clock rule 2026-10-26, Decision 6b).

A product is all dates or all floats. A dated Asian books its legacy schedule as ``observation_dates`` (resolved
from the valuation date like its records' dates); a dated accumulator books its realized fixings as
``(date, spot)``. Either prices bitwise like the numeric booking with the same times at the same valuation date.
"""
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.analytical import AccumulatorAnalyticalEngine, AsianOptionAnalyticalEngine
from quantark.asset.equity.engine.mc import AsianOptionMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import AccumulatorOption, AsianOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import AsianStrikeType, AveragingType, OptionType
from quantark.util.enum.engine_enums import AsianAnalyticalMethod, MonteCarloMethod
from quantark.util.exceptions import ValidationError

VAL = datetime(2024, 1, 1)
DAYS = (30, 60, 90, 120)
DATES = [VAL + timedelta(days=d) for d in DAYS]


def _env() -> PricingEnvironment:
    return PricingEnvironment(spot_quote=SpotQuote(spot=100.0), rate_curve=FlatRateCurve(rate=0.03),
                              vol_surface=FlatVolSurface(volatility=0.25),
                              div_yield=ContinuousDividendYield(div_yield=0.01), valuation_date=VAL)


# --- Asian: legacy observation_dates -------------------------------------------------------------------------------

def _asian(**kw) -> AsianOption:
    return AsianOption(strike=100.0, option_type=OptionType.CALL, asian_strike_type=AsianStrikeType.FIXED,
                       averaging_type=AveragingType.ARITHMETIC, **kw)


def test_asian_observation_dates_resolve_like_its_records():
    env = _env()
    dated = _asian(exercise_date=DATES[-1], observation_dates=DATES)
    _, _, future_times, _, total = dated.resolve_observations(env)
    assert future_times == [d / 365.0 for d in DAYS] and total == len(DAYS)


@pytest.mark.parametrize("engine", [lambda: AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN),
                                    lambda: AsianOptionMCEngine(params=MCParams(num_paths=4_000, seed=11),
                                                                method=MonteCarloMethod.PSEUDO)],
                         ids=["analytical", "mc"])
def test_a_dated_asian_prices_like_its_numeric_twin(engine):
    env = _env()
    dated = _asian(exercise_date=DATES[-1], observation_dates=DATES)
    _, _, times, _, _ = dated.resolve_observations(env)
    numeric = _asian(maturity=dated.get_maturity(env), observation_times=times)
    assert engine().price(dated, env) == engine().price(numeric, env)


def test_asian_dates_win_over_times():
    both = _asian(exercise_date=DATES[-1], observation_dates=DATES, observation_times=[0.1, 0.2])
    assert both.resolve_observations(_env())[2] == [d / 365.0 for d in DAYS]


def test_a_past_asian_observation_date_is_refused_as_a_negative_time_is():
    dated = _asian(exercise_date=DATES[-1], observation_dates=[VAL - timedelta(days=5)] + DATES)
    with pytest.raises(ValidationError, match="strictly-past"):
        dated.resolve_observations(_env())


# --- accumulator: dated realized fixings ---------------------------------------------------------------------------

PAST = [(-20, 101.0), (-10, 98.0)]


def _accumulator(**kw) -> AccumulatorOption:
    return AccumulatorOption(strike=95.0, knock_out_barrier=110.0, initial_price=100.0, daily_share_accumulation=1.0,
                             **kw)


def _dated_accumulator() -> AccumulatorOption:
    schedule = ObservationSchedule(records=[ObservationRecord(observation_date=d) for d in DATES])
    return _accumulator(exercise_date=DATES[-1], observation_schedule=schedule,
                        past_observations=[(VAL + timedelta(days=d), s) for d, s in PAST])


def test_a_dated_accumulator_books_its_fixings_by_date_and_prices_like_its_numeric_twin():
    env = _env()
    dated = _dated_accumulator()
    schedule = dated.observation_schedule
    assert schedule is not None
    times = [record.resolve_time(env) for record in schedule.records]
    numeric = _accumulator(maturity=dated.get_maturity(env), observation_dates=times,
                           past_observations=[(d / 365.0, s) for d, s in PAST])
    assert dated.get_realized_accrual() == numeric.get_realized_accrual() != 0.0
    assert AccumulatorAnalyticalEngine().price(dated, env) == AccumulatorAnalyticalEngine().price(numeric, env)


def test_accumulator_fixings_mixing_dates_and_times_are_refused():
    with pytest.raises(ValidationError, match="past_observations"):
        _accumulator(maturity=1.0, observation_dates=[0.5, 1.0],
                     past_observations=[(-0.1, 100.0), (VAL - timedelta(days=5), 99.0)])
