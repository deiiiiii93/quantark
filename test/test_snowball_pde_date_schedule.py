"""A date-based snowball schedule must reach the PDE grid (bug found 2026-09-04).

``grid_request`` declares the time grid, and knock-out dates have to land on grid nodes
exactly. It read ``observation_time`` straight off the config, which is None for a schedule
stated in dates, so a date-based contract produced a grid with no knock-out nodes. The
observation map then resolved the same schedule against the pricing environment and asked
the grid for a node that had never been created, raising a bare KeyError from the time axis.

MC and QUAD were unaffected: every one of their schedule reads already passed the pricing
environment. Only the PDE declares geometry before resolving.
"""
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.observation_schedule import (
    ObservationRecord, ObservationSchedule,
)
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.enum import ObservationType
from quantark.util.exceptions import ValidationError

T0 = datetime(2026, 9, 2)
MATURITY_DATE = datetime(2027, 9, 2)
CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
KO_DATES = [T0 + timedelta(days=91 * i) for i in range(1, 5)]      # 4 quarterly observations
KI_DATES = [T0 + timedelta(days=30 * i) for i in range(1, 12)]     # 11 monthly observations
KO_BARRIERS = [105.0, 104.0, 103.0, 102.0]
KO_RATES = [0.12, 0.12, 0.10, 0.10]
INITIAL, STRIKE, KI = 100.0, 100.0, 75.0


def _env(day_count=None):
    kw = dict(spot_quote=SpotQuote(spot=98.0), vol_surface=FlatVolSurface(volatility=0.22),
              rate_curve=FlatRateCurve(rate=0.02),
              div_yield=ContinuousDividendYield(div_yield=0.01),
              valuation_date=T0, calendar=CAL, bus_days_in_year=244)
    if day_count is not None:
        kw["day_count_convention"] = day_count
    return PricingEnvironment(**kw)


def _yf(d):
    return (d - T0).days / 365.0


def _snowball(dated: bool):
    """The same contract, stated in dates or in float year fractions."""
    if dated:
        ko = ObservationSchedule(records=[
            ObservationRecord(observation_date=d, barrier=b, return_rate=r)
            for d, b, r in zip(KO_DATES, KO_BARRIERS, KO_RATES)])
        ki = ObservationSchedule(records=[
            ObservationRecord(observation_date=d, barrier=KI) for d in KI_DATES])
        extra = dict(ko_observation_schedule=ko, ki_observation_schedule=ki)
        dates = dict(exercise_date=MATURITY_DATE, maturity_date=MATURITY_DATE, maturity=None)
    else:
        extra = dict(ko_observation_dates=[_yf(d) for d in KO_DATES],
                     ki_observation_dates=[_yf(d) for d in KI_DATES])
        dates = dict(maturity=_yf(MATURITY_DATE))
    return SnowballOption(
        initial_price=INITIAL, strike=STRIKE,
        barrier_config=BarrierConfig(
            ko_barrier=list(KO_BARRIERS), ko_rate=list(KO_RATES),
            ko_observation_type=ObservationType.DISCRETE,
            ki_barrier=KI, ki_observation_type=ObservationType.DISCRETE,
            ki_continuous=False, **extra),
        payoff_config=PayoffConfig(rebate_rate=0.10, include_principal=False),
        # tenor is stated on both spellings so the rebate accrual is identical; a dated
        # contract would otherwise derive it from initial_date and a float one from maturity
        tenor=1.0,
        contract_multiplier=1.0, is_reverse=False, **dates)


def test_a_date_based_schedule_prices_on_the_pde():
    """It used to raise KeyError from the time axis before pricing at all."""
    price = SnowballPDESolver(PDEParams(accuracy="fast")).price(_snowball(True), _env())
    assert price == pytest.approx(price)          # finite
    assert abs(price) > 0.0


def test_dates_and_float_times_are_the_same_contract():
    """Same schedule, two spellings: the PDE must not care which one it is given."""
    eng = SnowballPDESolver(PDEParams(accuracy="fast"))
    env = _env()
    assert eng.price(_snowball(True), env) == pytest.approx(
        eng.price(_snowball(False), env), rel=1e-12)


def test_the_grid_is_told_about_every_resolved_knock_out_time():
    """The defect in one line: geometry and the event map must agree on the times."""
    eng = SnowballPDESolver(PDEParams(accuracy="fast"))
    env = _env()
    product = _snowball(True)
    tau = eng._prepare_for_request(product, env)
    align = eng._ko_coupon_align_times(product, tau)
    resolved = [r.observation_time for r in product.resolve_ko_observations(env)]
    interior = [t for t in resolved if 0.0 < t < tau]
    assert align == pytest.approx(interior, rel=1e-12)
    assert len(align) == len(KO_DATES)            # all four fall strictly inside the horizon
    assert align                                  # the bug was this coming back empty
    # and the knock-in monitor dates reach the grid too
    assert len(eng._ki_monitor_times(product, tau)) == len(KI_DATES)


def test_a_date_schedule_without_a_prepared_environment_fails_loudly():
    """Dates cannot be resolved without an environment; say so, do not build a wrong grid."""
    eng = SnowballPDESolver(PDEParams(accuracy="fast"))
    product = _snowball(True)
    eng._grid_request_env = None
    with pytest.raises(ValidationError, match="pricing environment"):
        eng._ko_coupon_align_times(product, 1.0)


def test_the_clock_comes_from_the_environment_for_a_dated_contract():
    """A dated contract on a business-day count is a shorter contract in vol time."""
    eng = SnowballPDESolver(PDEParams(accuracy="fast"))
    product = _snowball(True)
    cal_tau = product.get_maturity(_env())
    bus_tau = product.get_maturity(_env(DayCountConvention.BUSINESS_DAYS))
    assert cal_tau == pytest.approx(365 / 365.0, rel=1e-12)
    assert bus_tau < cal_tau                      # ~244 trading days over 244
    assert eng.price(product, _env(DayCountConvention.BUSINESS_DAYS)) != pytest.approx(
        eng.price(product, _env()), rel=1e-6)
