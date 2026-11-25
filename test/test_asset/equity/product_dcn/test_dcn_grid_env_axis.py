"""The DCN grid on the environment's axis (DCN time-axis spec D6/D13/D16-D19; AC2c, AC3).

Every DCN time is ``env_time(env, date, anchor=schedule.valuation_date)``; KI dates on one env instant share one
grid column (``column_dates``); column 0 is monitored when it holds a KI date; a convention whose times run
backwards is refused.
"""
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context, dcn_row_accruals
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.priceenv.date_axis import env_time, env_times
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.calendar.business_calendar import BusinessDayConvention as BDC
from quantark.util.exceptions import ValidationError

from dcn_fixtures import DCN_A, FLAT, SSE, make_dcn

US = create_calendar(CalendarType.US, year_range=(2020, 2030))
#: a one-year DCN through July 2025 (SSE schedule; Fri 2025-07-04 is an SSE trading day and a US holiday)
DCN_25 = dict(DCN_A, initial_date=datetime(2025, 1, 3), valuation_date=datetime(2025, 1, 3),
              maturity_date=datetime(2026, 1, 5), tenor_months=12, settlement_date=datetime(2026, 1, 7))


def env_on(d, dc=DayCountConvention.CALENDAR_DAYS, calendar=None, adjust=BDC.PRECEDING, days=252):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=6000.0), vol_surface=FlatVolSurface(volatility=FLAT["sigma"]),
        rate_curve=FlatRateCurve(rate=FLAT["r"]), div_yield=ContinuousDividendYield(div_yield=FLAT["q"]),
        valuation_date=d, day_count_convention=dc, calendar=calendar, bus_days_in_year=days,
        holiday_time_adjustment=adjust)


ENVS = {
    "act_360": lambda d: env_on(d, DayCountConvention.ACT_360),
    "business_days_sse": lambda d: env_on(d, DayCountConvention.BUSINESS_DAYS, SSE, days=244),
}


@pytest.mark.parametrize("name", sorted(ENVS))
def test_times_are_env_axis(name):
    """AC3: grid, payment and maturity times are env_time of the schedule / C4 dates from the schedule's anchor."""
    v = datetime(2024, 5, 15)
    product = make_dcn(DCN_A, valuation_date=v)
    env = ENVS[name](v)
    ctx = build_dcn_grid_context(product, env)
    s = product.schedule
    np.testing.assert_array_equal(ctx.times, env_times(env, s.daily_ki_dates, anchor=v))
    assert all(len(c) == 1 for c in ctx.column_dates)
    pay = product.payment_dates()
    np.testing.assert_array_equal(ctx.coupon_pay_times, env_times(env, pay.coupon_payment_dates, anchor=v))
    np.testing.assert_array_equal(ctx.ko_pay_times, env_times(env, pay.ko_payment_dates, anchor=v))
    assert ctx.loss_pay_time == env_time(env, pay.loss_payment_date, anchor=v)
    assert product.get_maturity(env) == env_time(env, s.monthly[-1].observation_date, anchor=v)
    assert abs(product.get_maturity(env) - product.get_maturity()) > 1e-3   # really another axis


def test_anchor_is_the_schedule_valuation_date():
    """D13: an env dated elsewhere (date-rolled theta, theta_shift) does not move the DCN's times."""
    v = datetime(2024, 5, 15)
    product = make_dcn(DCN_A, valuation_date=v)
    a = build_dcn_grid_context(product, env_on(v, DayCountConvention.ACT_360))
    b = build_dcn_grid_context(product, env_on(datetime(2024, 5, 14), DayCountConvention.ACT_360))
    np.testing.assert_array_equal(a.times, b.times)
    assert a.times[0] == 0.0


def _ki_col(ctx, d):
    return next(i for i, dates in enumerate(ctx.column_dates) if d in dates)


def test_us_holiday_preceding_merges_with_the_day_before():
    """AC3: on a US BUSINESS_DAYS env the SSE KI date Fri 2025-07-04 shares Thu 07-03's instant."""
    v = datetime(2025, 6, 3)
    product = make_dcn(DCN_25, valuation_date=v)
    ctx = build_dcn_grid_context(product, env_on(v, DayCountConvention.BUSINESS_DAYS, US))
    j3, j4 = _ki_col(ctx, datetime(2025, 7, 3)), _ki_col(ctx, datetime(2025, 7, 4))
    assert j3 == j4 and ctx.column_dates[j3] == (datetime(2025, 7, 3), datetime(2025, 7, 4))
    assert np.all(np.diff(ctx.times) > 0.0)
    assert ctx.times.size == len(product.schedule.daily_ki_dates) - sum(len(c) - 1 for c in ctx.column_dates)


def test_us_holiday_following_merges_with_the_day_after():
    v = datetime(2025, 6, 3)
    product = make_dcn(DCN_25, valuation_date=v)
    ctx = build_dcn_grid_context(product, env_on(v, DayCountConvention.BUSINESS_DAYS, US, BDC.FOLLOWING))
    assert _ki_col(ctx, datetime(2025, 7, 4)) == _ki_col(ctx, datetime(2025, 7, 7))
    assert _ki_col(ctx, datetime(2025, 7, 3)) != _ki_col(ctx, datetime(2025, 7, 4))


def test_future_date_at_zero_joins_column_zero():
    """D7/D17: valued Thu 07-03, the Fri 07-04 KI date is at env time 0 -> column 0, monitored."""
    v = datetime(2025, 7, 3)
    ctx = build_dcn_grid_context(make_dcn(DCN_25, valuation_date=v),
                                 env_on(v, DayCountConvention.BUSINESS_DAYS, US))
    assert ctx.column_dates[0] == (datetime(2025, 7, 3), datetime(2025, 7, 4)) and ctx.valuation_monitored
    assert ctx.times[0] == 0.0 and ctx.times[1] > 0.0


def test_unmonitored_valuation_with_a_future_date_at_zero_is_monitored():
    """D17: valued Sat 2025-01-18 (SSE closed); the next SSE date Mon 01-20 is a US holiday -> at 0, column 0."""
    v = datetime(2025, 1, 18)
    ctx = build_dcn_grid_context(make_dcn(DCN_25, valuation_date=v),
                                 env_on(v, DayCountConvention.BUSINESS_DAYS, US))
    assert ctx.valuation_monitored and ctx.column_dates[0] == (datetime(2025, 1, 20),)
    assert ctx.times[0] == 0.0


def test_unmonitored_valuation_prepends_t0():
    v = datetime(2024, 6, 8)   # Saturday
    ctx = build_dcn_grid_context(make_dcn(DCN_A, valuation_date=v), env_on(v))
    assert not ctx.valuation_monitored and ctx.column_dates[0] == () and ctx.times[0] == 0.0
    assert ctx.column_dates[1] == (datetime(2024, 6, 11),)   # Mon 06-10 is the SSE Dragon Boat holiday


def test_thirty_360_merges_and_prices_the_grid():
    """D18: 30/360 maps the 30th and 31st to one time -- merged, not refused."""
    v = datetime(2024, 5, 15)
    product, env = make_dcn(DCN_A, valuation_date=v), env_on(v, DayCountConvention.THIRTY_360_US)
    ctx = build_dcn_grid_context(product, env)
    assert np.all(np.diff(ctx.times) > 0.0)
    assert any(len(c) > 1 for c in ctx.column_dates)
    from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
    from quantark.asset.equity.param import PDEParams

    assert np.isfinite(DCNPDEEngine(PDEParams(accuracy="fast")).price(product, env))


def test_backwards_convention_is_refused():
    """AC2c / D18: ACT_365L runs backwards across 2024-02-29 from a 2023 anchor."""
    v = datetime(2023, 1, 3)
    with pytest.raises(ValidationError, match="ACT_365L|act_365l"):
        build_dcn_grid_context(make_dcn(DCN_A, valuation_date=v), env_on(v, DayCountConvention.ACT_365L))


def test_row_accruals_are_env_free_and_match_the_grid():
    """D16: the tracker's accruals need no env."""
    product = make_dcn(DCN_A)
    acc_c, acc_k = dcn_row_accruals(product)
    ctx = build_dcn_grid_context(product, env_on(product.schedule.valuation_date))
    np.testing.assert_array_equal(acc_c, ctx.coupon_accruals)
    np.testing.assert_array_equal(acc_k, ctx.ko_accruals)
