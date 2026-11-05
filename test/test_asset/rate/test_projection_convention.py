"""Floating-rate projection convention (spec 2026-11-05, T4 / AC18).

Every floating projection in the rate and FRN engines is the simple forward on
the projection curve over the window the index covers, computed by
``simple_forward_rate``. The telescoping identity (b) fails by
≈ N·f²τ/2·annuity under the old continuously-compounded shortcut.
"""
from __future__ import annotations

import math
from datetime import datetime

import pytest

from quantark.asset.bond.engine.discount.frn_engine import FRNDiscountEngine
from quantark.asset.bond.product.couponbond.frn import FloatingRateBond
from quantark.asset.rate import (
    CapFloor,
    CapFloorEngine,
    CapFloorType,
    FixedLeg,
    FloatingLeg,
    FRAEngine,
    InterestRateSwap,
    IRSDiscountEngine,
)
from quantark.asset.rate.product.fra import create_fra
from quantark.asset.rate.product.irs import NotionalSchedule, create_compounding_irs
from quantark.param import CurveInterpolation, FlatRateCurve, ZeroCurve, simple_forward_rate
from quantark.param.index import SHIBOR_3M, SOFR
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import BusinessDayConvention, CalendarType, DayCountConvention, create_calendar
from quantark.util.enum import PaymentFrequency, ResetConvention
from quantark.util.exceptions import MarketDataError

VD = datetime(2026, 1, 15)  # a Thursday, away from month-end weekends
R = 0.05
FLAT = FlatRateCurve(R)
PILLARS = [(0.25, 0.030), (0.5, 0.031), (1.0, 0.033), (2.0, 0.0345), (5.0, 0.036), (10.0, 0.035)]
TERM = ZeroCurve(PILLARS, interpolation=CurveInterpolation.LOG_CUBIC_DF)
NONE_CAL = create_calendar(CalendarType.NONE, year_range=(2020, 2035))


def _t(d: datetime, vd: datetime = VD) -> float:
    return (d - vd).days / 365.0


def _env(curve, vd: datetime = VD) -> PricingEnvironment:
    return PricingEnvironment(rate_curve=curve, valuation_date=vd)


def _floating_leg(start: datetime, end: datetime, **kw) -> FloatingLeg:
    base = dict(
        start_date=start,
        end_date=end,
        notional_schedule=NotionalSchedule.constant(1_000_000.0),
        index=SHIBOR_3M,  # fixing lag 0
        spread=0.0,
        payment_frequency=PaymentFrequency.QUARTERLY,
        day_count_convention=DayCountConvention.ACT_365,
        calendar=NONE_CAL,
        business_day_convention=BusinessDayConvention.UNADJUSTED,  # payment == accrual end
    )
    base.update(kw)
    return FloatingLeg(**base)


# --------------------------------------------------------------------------- (a) closed forms


def test_fra_forward_is_simple_forward_over_accrual():
    fra = create_fra(
        trade_date=VD, settlement_date=datetime(2026, 4, 15), tenor_months=3, notional=1e6,
        fixed_rate=0.05, index=SHIBOR_3M, day_count_convention=DayCountConvention.ACT_360,
        calendar=NONE_CAL,
    )
    t1, t2 = _t(fra.accrual_start), _t(fra.accrual_end)
    dcf = fra.day_count_fraction()
    fwd = FRAEngine(_env(FLAT)).forward_rate(fra)
    assert fwd == pytest.approx((math.exp(R * (t2 - t1)) - 1.0) / dcf, rel=1e-14)
    assert fwd == pytest.approx(simple_forward_rate(FLAT, t1, t2, dcf), rel=1e-15)
    fwd_term = FRAEngine(_env(TERM)).forward_rate(fra)
    assert fwd_term == pytest.approx(simple_forward_rate(TERM, t1, t2, dcf), rel=1e-15)
    # par rate == forward, NPV 0 at par
    assert FRAEngine(_env(TERM)).par_rate(fra) == pytest.approx(fwd_term, rel=1e-15)


def test_caplet_forwards_are_simple_forwards():
    cap = CapFloor(
        notional=1e6, strike=0.05, cap_floor_type=CapFloorType.CAP,
        start_date=datetime(2026, 4, 15), end_date=datetime(2028, 4, 15), index=SHIBOR_3M,
        payment_frequency=PaymentFrequency.QUARTERLY, day_count_convention=DayCountConvention.ACT_360,
        calendar=NONE_CAL,
    )
    for curve in (FLAT, TERM):
        results = CapFloorEngine(_env(curve), vol=0.20).full_analysis(cap)
        caplets = cap.get_future_caplets(VD)
        assert len(results.caplet_details) == len(caplets) > 0
        for detail, caplet in zip(results.caplet_details, caplets):
            t1, t2 = _t(caplet.accrual_start), _t(caplet.accrual_end)
            expected = simple_forward_rate(curve, t1, t2, caplet.day_count_fraction)
            assert detail.forward_rate == pytest.approx(expected, rel=1e-15)
            if curve is FLAT:
                assert detail.forward_rate == pytest.approx(
                    (math.exp(R * (t2 - t1)) - 1.0) / caplet.day_count_fraction, rel=1e-14
                )


# --------------------------------------------------------------------------- (b) telescoping identity


@pytest.mark.parametrize("curve", [FLAT, TERM])
def test_floating_leg_pv_telescopes_in_single_curve(curve):
    end = datetime(2031, 1, 15)
    leg = _floating_leg(VD, end)
    leg.update_forward_rates(curve, VD)
    pv = sum(cf.amount * curve.get_discount_factor(_t(cf.payment_date)) for cf in leg.get_cashflows(VD))
    notional = 1_000_000.0
    expected = notional * (curve.get_discount_factor(0.0) - curve.get_discount_factor(_t(end)))
    assert pv == pytest.approx(expected, abs=1e-12 * notional)
    # the same through the engine
    fixed = FixedLeg(
        start_date=VD, end_date=end, notional_schedule=NotionalSchedule.constant(notional),
        fixed_rate=0.03, payment_frequency=PaymentFrequency.QUARTERLY,
        day_count_convention=DayCountConvention.ACT_365, calendar=NONE_CAL,
        business_day_convention=BusinessDayConvention.UNADJUSTED,
    )
    swap = InterestRateSwap(fixed_leg=fixed, floating_leg=leg, effective_date=VD)
    engine = IRSDiscountEngine(_env(curve))
    assert engine.floating_leg_pv(swap.floating_leg) == pytest.approx(expected, abs=1e-12 * notional)
    # payer NPV = float PV - fixed PV, consistent with the leg identity
    fixed_pv = engine.fixed_leg_pv(swap.fixed_leg)
    assert engine.npv(swap) == pytest.approx(expected - fixed_pv, abs=1e-12 * notional)
    # under the old shortcut the gap was O(N f^2 tau/2 * annuity) ~ hundreds of currency units
    cc_gap = sum(
        notional * (curve.get_forward_rate(_t(cf.accrual_start_date), _t(cf.accrual_end_date)) - cf.forward_rate)
        * cf.day_count_fraction * curve.get_discount_factor(_t(cf.payment_date))
        for cf in leg.get_floating_cashflows(VD)
    )
    assert abs(cc_gap) > 50.0


@pytest.mark.parametrize("curve", [FLAT, TERM])
def test_frn_prices_at_par_on_issue_date(curve):
    frn = FloatingRateBond(
        issue_date=VD, maturity_date=datetime(2029, 1, 15), denominator=100.0, index=SHIBOR_3M, spread=0.0,
        payment_frequency=PaymentFrequency.QUARTERLY, day_count_convention=DayCountConvention.ACT_365,
        calendar=NONE_CAL, business_day_convention=BusinessDayConvention.UNADJUSTED,
    )
    price = FRNDiscountEngine(_env(curve)).price(frn)
    assert price == pytest.approx(100.0, abs=1e-10)


def test_dual_curve_forwards_come_from_projection_curve():
    leg = _floating_leg(VD, datetime(2028, 1, 15))
    swap = InterestRateSwap(
        fixed_leg=FixedLeg(
            start_date=VD, end_date=datetime(2028, 1, 15),
            notional_schedule=NotionalSchedule.constant(1_000_000.0), fixed_rate=0.03,
            payment_frequency=PaymentFrequency.QUARTERLY, day_count_convention=DayCountConvention.ACT_365,
            calendar=NONE_CAL, business_day_convention=BusinessDayConvention.UNADJUSTED,
        ),
        floating_leg=leg, effective_date=VD,
    )
    engine = IRSDiscountEngine(_env(FLAT), projection_curve=TERM)
    engine.npv(swap)
    for cf in leg.get_floating_cashflows(VD):
        expected = simple_forward_rate(TERM, _t(cf.accrual_start_date), _t(cf.accrual_end_date), cf.day_count_fraction)
        assert cf.forward_rate == pytest.approx(expected, rel=1e-15)


# --------------------------------------------------------------------------- (c) running periods and missing fixings


def test_running_period_projects_pro_rata_spot_and_paid_flows_are_skipped():
    start = datetime(2025, 10, 15)
    leg = _floating_leg(start, datetime(2030, 10, 15))
    vd = datetime(2026, 2, 16)  # inside the second period [2026-01-15, 2026-04-15]; first coupon paid 2026-01-15
    leg.update_forward_rates(TERM, vd)
    flows = leg.get_floating_cashflows()
    first, second = flows[0], flows[1]
    assert first.payment_date <= vd and first.forward_rate is None  # paid: left alone
    t_s, t_e = _t(second.accrual_start_date, vd), _t(second.accrual_end_date, vd)
    assert t_s < 0.0 < t_e
    df_e = TERM.get_discount_factor(t_e)
    expected = (1.0 / df_e - 1.0) / (second.day_count_fraction * t_e / (t_e - t_s))
    assert second.forward_rate == pytest.approx(expected, rel=1e-15)
    assert second.forward_rate == pytest.approx(
        simple_forward_rate(TERM, t_s, t_e, second.day_count_fraction), rel=1e-15
    )
    # later periods: the plain forward over their accrual window
    third = flows[2]
    assert third.forward_rate == pytest.approx(
        simple_forward_rate(TERM, _t(third.accrual_start_date, vd), _t(third.accrual_end_date, vd),
                            third.day_count_fraction), rel=1e-15
    )


def test_elapsed_unpaid_coupon_without_fixing_raises():
    start = datetime(2025, 10, 15)
    leg = _floating_leg(start, datetime(2030, 10, 15), settlement_days=5)  # paid 5 business days after accrual end
    vd = datetime(2026, 1, 19)  # Monday: first accrual ended 2026-01-15, paid 2026-01-22
    first = leg.get_floating_cashflows()[0]
    assert first.accrual_end_date <= vd < first.payment_date
    with pytest.raises(MarketDataError) as excinfo:
        leg.update_forward_rates(TERM, vd)
    assert "SHIBOR_3M" in str(excinfo.value) and "2025-10-15" in str(excinfo.value)
    # with the fixing stored the flow uses it and nothing raises
    leg.add_fixing(first.fixing_date, 0.0312)
    leg.update_forward_rates(TERM, vd)
    fixed_flow = leg.get_floating_cashflows()[0]
    assert fixed_flow.is_projected is False and fixed_flow.effective_rate == pytest.approx(0.0312)


def test_frn_elapsed_unpaid_coupon_without_fixing_raises():
    frn = FloatingRateBond(
        issue_date=datetime(2025, 10, 15), maturity_date=datetime(2028, 10, 15), denominator=100.0,
        index=SHIBOR_3M, spread=0.0, payment_frequency=PaymentFrequency.QUARTERLY,
        day_count_convention=DayCountConvention.ACT_365, calendar=NONE_CAL,
        business_day_convention=BusinessDayConvention.UNADJUSTED, settlement_days=5,
    )
    vd = datetime(2026, 1, 19)
    with pytest.raises(MarketDataError):
        FRNDiscountEngine(_env(TERM, vd)).price(frn)
    first = frn.get_floating_cashflows(vd)[0]
    assert first.accrual_end_date <= vd < first.payment_date
    frn.add_fixing(first.fixing_date, 0.0312)
    assert FRNDiscountEngine(_env(TERM, vd)).price(frn) > 0.0


def test_caplet_elapsed_unpaid_without_fixing_raises():
    # accrual dates fall on the 19th; 2025-10-19 is a Sunday, so FOLLOWING rolls that
    # caplet's payment to Monday 2025-10-20 while its accrual ends on the 19th
    cap = CapFloor(
        notional=1e6, strike=0.03, cap_floor_type=CapFloorType.CAP,
        start_date=datetime(2025, 7, 19), end_date=datetime(2027, 7, 19), index=SHIBOR_3M,
        payment_frequency=PaymentFrequency.QUARTERLY, day_count_convention=DayCountConvention.ACT_360,
        calendar=NONE_CAL, business_day_convention=BusinessDayConvention.FOLLOWING,
    )
    elapsed = [c for c in cap.get_caplets() if c.accrual_end == datetime(2025, 10, 19)][0]
    assert elapsed.payment_date == datetime(2025, 10, 20)
    vd = datetime(2025, 10, 19)
    assert elapsed.accrual_end <= vd < elapsed.payment_date and elapsed.is_projected
    with pytest.raises(MarketDataError):
        CapFloorEngine(_env(TERM, vd), vol=0.20).full_analysis(cap)


def test_overnight_compounded_in_arrears_running_period():
    swap = create_compounding_irs(
        effective_date=datetime(2025, 10, 15), maturity_date=datetime(2030, 10, 15), denominator=1e6,
        fixed_rate=0.03, index=SOFR, payment_frequency=PaymentFrequency.QUARTERLY, lookback_days=2,
    )
    vd = datetime(2026, 2, 16)
    swap.update_forward_rates(TERM, vd)
    flows = swap.floating_leg.get_floating_cashflows()
    running = [cf for cf in flows if cf.accrual_start_date < vd < cf.accrual_end_date][0]
    assert running.fixing_date > vd  # in arrears: fixing at the end of the period
    t_s, t_e = _t(running.accrual_start_date, vd), _t(running.accrual_end_date, vd)
    assert running.forward_rate == pytest.approx(
        simple_forward_rate(TERM, t_s, t_e, running.day_count_fraction), rel=1e-15
    )
    future = [cf for cf in flows if cf.accrual_start_date > vd][0]
    assert future.forward_rate == pytest.approx(
        simple_forward_rate(TERM, _t(future.accrual_start_date, vd), _t(future.accrual_end_date, vd),
                            future.day_count_fraction), rel=1e-15
    )


def test_in_arrears_term_index_uses_index_period():
    leg = _floating_leg(VD, datetime(2028, 1, 15), reset_convention=ResetConvention.IN_ARREARS)
    leg.update_forward_rates(TERM, VD)
    for cf in leg.get_floating_cashflows(VD):
        assert cf.fixing_date == cf.accrual_end_date
        t_fix = _t(cf.fixing_date)
        expected = simple_forward_rate(TERM, t_fix, t_fix + SHIBOR_3M.tenor_years)
        assert cf.forward_rate == pytest.approx(expected, rel=1e-15)
