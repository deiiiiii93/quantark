"""Construction fills every record's period and the terminals (spec §4.2–4.7)."""
from dataclasses import replace
from datetime import timedelta

import pytest

from quantark.asset.equity.product.option.accrual import AccrualKind as K
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, PayoffConfig
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit as U
from quantark.util.calendar import BusinessDayConvention, CalendarType, create_calendar
from quantark.util.calendar.day_counter import DayCountConvention as DC
from quantark.util.enum import AccrualEndType, CouponPayType, PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from .conftest import T0, dates, ko_reset, periods, phoenix, rebuilt, snowball, times

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))


def test_numeric_records_end_at_the_quoted_fraction():
    p = snowball()
    assert [(a.kind, a.origin, a.start, a.end) for a in periods(p.barrier_config.ko_observation_schedule)] == [
        (K.TIME, 0.0, 0.0, 1 / 12), (K.TIME, 0.0, 1 / 12, 2 / 12), (K.TIME, 0.0, 2 / 12, 3 / 12)]


def test_aged_legacy_contract_banks_its_offset():
    p = snowball(ko=times(0.2), accrual=AccrualConfig(accrued_offset=0.3, first_period_start=0.25))
    a = p.barrier_config.ko_observation_schedule.records[0].accrual
    assert (a.start, a.end) == (0.25, 0.3 + 0.2)


def test_unknown_first_period_start_stays_unknown():
    p = snowball(ko=times(0.2), accrual=AccrualConfig(accrued_offset=0.3, first_period_start=None))
    assert p.barrier_config.ko_observation_schedule.records[0].accrual.start is None


def test_dated_records_chain_from_initial_date():
    p = snowball(ko=dates(30, 60), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=60))
    a0, a1 = periods(p.barrier_config.ko_observation_schedule)
    assert (a0.kind, a0.origin, a0.start, a0.end) == (K.DATE, T0, T0, T0 + timedelta(days=30))
    assert a1.start == a0.end and a1.basis.day_count is DC.ACT_365


def test_dated_annualized_without_initial_date_is_refused():
    with pytest.raises(ValidationError, match="initial_date"):
        snowball(ko=dates(30, 60), maturity=None, exercise_date=T0 + timedelta(days=60))


def test_dated_non_annualized_without_initial_date_is_none():
    p = snowball(ko=dates(30, 60), maturity=None, exercise_date=T0 + timedelta(days=60),
                 accrual=AccrualConfig(is_annualized=False))
    assert {a.kind for a in periods(p.barrier_config.ko_observation_schedule)} == {K.NONE}


def test_accrual_factors_set_both_legs_on_schedule_one():
    p = phoenix(accrual=AccrualConfig(accrual_factors=[0.1, 0.2, 0.3]))
    assert [(a.cumulative_fraction, a.period_fraction) for a in periods(p.barrier_config.ko_observation_schedule)] == [
        (0.1, 0.1), (0.2, 0.2), (0.3, 0.3)]


def test_fixed_coupon_fraction_sets_the_coupon_leg_only():
    p = phoenix(coupon=CouponBarrierConfig(coupon_barrier=80.0, coupon_rate=0.12, fixed_coupon_year_fraction=1 / 12))
    a = p.barrier_config.ko_observation_schedule.records[2].accrual
    assert (a.cumulative_fraction, a.period_fraction, a.end) == (None, 1 / 12, 3 / 12)


def test_schedule_specific_factors():
    ok = ko_reset(accrual=AccrualConfig(accrual_factors=[0.3, 0.6]))
    assert [a.cumulative_fraction for a in periods(ok.post_barrier_config.ko_observation_schedule)] == [None] * 3
    with pytest.raises(ValidationError, match="post_accrual_factors"):
        ko_reset(accrual=AccrualConfig(post_accrual_factors=[0.1, 0.2]))
    with pytest.raises(ValidationError, match="accrual_factors"):
        ko_reset(accrual=AccrualConfig(accrual_factors=[0.1, 0.2, 0.3]))


def test_rebased_schedule_two_is_relative_offsets():
    p = ko_reset(M.REBASED, accrual=AccrualConfig(accrued_offset=0.3))
    a = periods(p.post_barrier_config.ko_observation_schedule)
    assert [(x.relative, x.start, x.end) for x in a] == [(True, 0.0, 0.25), (True, 0.25, 0.5), (True, 0.5, 0.75)]


def test_rebased_post_factor_is_the_offsets_accrual():
    p = ko_reset(M.REBASED, accrual=AccrualConfig(post_accrual_factors=[0.2, 0.4, 0.6]))
    a = periods(p.post_barrier_config.ko_observation_schedule)
    assert [(x.relative, x.end, x.cumulative_fraction) for x in a] == [(True, 0.2, None), (True, 0.4, None),
                                                                       (True, 0.6, None)]


def test_knock_in_records_accrue_to_their_fixing_even_under_settlement():
    conv = SettlementConvention(lag=0.01, lag_unit=U.YEAR_FRACTION)
    p = ko_reset(M.REBASED, accrual=AccrualConfig(accrual_end=AccrualEndType.SETTLEMENT), settlement_convention=conv)
    assert [a.end for a in periods(p.barrier_config.ki_observation_schedule)] == [0.25, 0.5]
    assert periods(p.barrier_config.ko_observation_schedule)[0].end == 0.25 + 0.01


def test_rebased_settlement_endpoint_is_the_explicit_offset_payment():
    post = times(0.25, 0.5, level=100.0)
    post.records[1].settlement_time = 0.52
    p = ko_reset(M.REBASED, post=post, accrual=AccrualConfig(accrual_end=AccrualEndType.SETTLEMENT))
    assert periods(p.post_barrier_config.ko_observation_schedule)[1].end == 0.52


@pytest.mark.parametrize("conv, sched, match", [
    (SettlementConvention(lag=2, lag_unit=U.BUSINESS_DAYS), "dates", "calendar"),
    (SettlementConvention(lag=0.01, lag_unit=U.YEAR_FRACTION), "dates", "YEAR_FRACTION"),
    (SettlementConvention(lag=2, lag_unit=U.BUSINESS_DAYS, calendar=CAL), "times", "BUSINESS_DAYS"),
])
def test_unresolvable_settlement_endpoints_are_refused(conv, sched, match):
    ko = dates(30, 60) if sched == "dates" else times(0.25)
    kw = dict(initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=60)) if sched == "dates" else {}
    with pytest.raises(ValidationError, match=match):
        snowball(ko=ko, accrual=AccrualConfig(accrual_end=AccrualEndType.SETTLEMENT), settlement_convention=conv, **kw)


def test_business_day_settlement_endpoint_uses_the_conventions_calendar():
    conv = SettlementConvention(lag=2, lag_unit=U.BUSINESS_DAYS, calendar=CAL,
                                business_day_convention=BusinessDayConvention.FOLLOWING)
    p = snowball(ko=dates(30, 60), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=60),
                 accrual=AccrualConfig(accrual_end=AccrualEndType.SETTLEMENT), settlement_convention=conv)
    fixing = T0 + timedelta(days=30)
    assert p.barrier_config.ko_observation_schedule.records[0].accrual.end == CAL.adjust_date(
        CAL.add_business_days(fixing, 2), BusinessDayConvention.FOLLOWING)


def test_expiry_coupons_under_settlement_are_refused():
    with pytest.raises(ValidationError, match="EXPIRY"):
        snowball(accrual=AccrualConfig(accrual_end=AccrualEndType.SETTLEMENT, coupon_pay_type=CouponPayType.EXPIRY))


def test_business_days_annualized_needs_its_terms():
    kw = dict(ko=dates(30, 60), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=60),
              annualization_day_count=DC.BUSINESS_DAYS)
    with pytest.raises(ValidationError, match="BUSINESS_DAYS accrual needs its calendar and denominator"):
        snowball(**kw)
    p = snowball(accrual=AccrualConfig(accrual_calendar=CAL, accrual_bus_days_in_year=244.0), **kw)
    assert p.barrier_config.ko_observation_schedule.records[0].accrual.basis.calendar is CAL


def test_shared_schedule_is_detached():
    shared = times(1 / 12)
    a = snowball(ko=shared, maturity=1 / 12, accrual=AccrualConfig(accrual_factors=[1 / 12]))
    b = snowball(ko=shared, maturity=1 / 12, accrual=AccrualConfig(accrual_factors=[0.1]))
    assert shared.records[0].accrual is None
    assert a.barrier_config.ko_observation_schedule.records[0].accrual.cumulative_fraction == 1 / 12
    assert b.barrier_config.ko_observation_schedule.records[0].accrual.cumulative_fraction == 0.1


def test_a_record_carrying_a_period_keeps_it():
    p = snowball()
    q = rebuilt(p, accrual_config=AccrualConfig(accrued_offset=0.5))
    assert periods(q.barrier_config.ko_observation_schedule) == periods(p.barrier_config.ko_observation_schedule)


def test_a_copy_with_other_contract_terms_derives_its_periods_again():
    """User direction: a copy whose accrual terms changed drops the derived periods and derives them anew."""
    p = snowball(ko=dates(30), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=30))
    q = rebuilt(p, annualization_day_count=DC.ACT_360)
    assert q.barrier_config.ko_observation_schedule.records[0].accrual.basis.day_count is DC.ACT_360
    r = rebuilt(p, initial_date=T0 - timedelta(days=1))
    assert r.barrier_config.ko_observation_schedule.records[0].accrual.origin == T0 - timedelta(days=1)
    p.initial_date = T0 - timedelta(days=1)
    p.validate()
    assert p.barrier_config.ko_observation_schedule.records[0].accrual.origin == T0 - timedelta(days=1)


def test_a_copy_with_other_factors_derives_its_periods_again():
    p = snowball(ko=times(0.25), maturity=0.25, accrual=AccrualConfig(accrual_factors=[0.4]))
    q = rebuilt(p, accrual_config=AccrualConfig(accrual_factors=[0.5]))
    assert q.barrier_config.ko_observation_schedule.records[0].accrual.cumulative_fraction == 0.5


def test_explicit_periods_are_always_kept():
    from quantark.asset.equity.product.option.accrual import AccrualKind, AccrualPeriod
    ko = times(0.25)
    ko.records[0].accrual = AccrualPeriod(AccrualKind.TIME, origin=0.0, end=0.3)
    p = snowball(ko=ko, maturity=0.25, initial_date=T0)
    assert p.barrier_config.ko_observation_schedule.records[0].accrual.end == 0.3


def test_terminals():
    s = snowball(tenor=1.0)
    assert s._terminal_accrual["contract"].cumulative_fraction == 1.0
    n = snowball(accrual=AccrualConfig(accrued_offset=0.1))
    assert (n._terminal_accrual["contract"].kind, n._terminal_accrual["contract"].end) == (K.TIME, 0.1 + 0.25)
    d = snowball(ko=dates(30, 60), initial_date=T0, maturity=None, exercise_date=T0 + timedelta(days=60))
    assert d._terminal_accrual["contract"].end == T0 + timedelta(days=60)
    r = ko_reset(M.REBASED)
    assert (r._terminal_accrual["pre"].end, r._terminal_accrual["post"].relative, r._terminal_accrual["post"].end) == (
        0.5, True, 0.75)
    a = ko_reset()
    assert a._terminal_accrual["post"].end == 0.75


def test_rebased_terminal_ignores_the_payment_offset():
    post = times(0.25, 0.5, level=100.0)
    post.records[1].settlement_time = 0.52
    p = ko_reset(M.REBASED, post=post, accrual=AccrualConfig(accrual_end=AccrualEndType.SETTLEMENT))
    assert p._terminal_accrual["post"].end == 0.5


def test_annualized_rebate_without_dated_origin_is_refused():
    with pytest.raises(ValidationError, match="initial_date"):
        snowball(ko=dates(30), maturity=None, exercise_date=T0 + timedelta(days=30),
                 accrual=AccrualConfig(is_annualized=False, is_annualized_rebate=True),
                 payoff_config=PayoffConfig(rebate_rate=0.05))
