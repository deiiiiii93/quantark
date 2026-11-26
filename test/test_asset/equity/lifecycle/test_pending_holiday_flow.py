"""A pending flow paid on an environment holiday right after the run date (0.16.0, DCN/dated run D7).

Under ``BUSINESS_DAYS`` business-day counting, a payment on the US holiday 2025-07-04 after a 2025-07-03 run date is
zero trading days away: the resolver used to refuse it ("payment must be strictly after valuation"), so the run
ledger's ``_revalue_ledger`` raised for the whole book. The flow is still owed (its date is after the run date); the
clock places it at the valuation instant, so it is timed at 0 (DF(0) = 1), as 0.14.0 times DCN flows.
"""
from datetime import datetime

import pytest

from quantark.asset.equity.lifecycle import ValuationPoint
from quantark.asset.equity.lifecycle.cashflows import LifecycleCashflowLedger, RealizedCashflow
from quantark.asset.equity.lifecycle.events import LifecycleEventType
from quantark.asset.equity.settlement import SettlementResolver
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import CurveQuotingBasis, PricingEnvironment
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.exceptions import ValidationError

US = create_calendar(CalendarType.US, year_range=(2024, 2027))
RUN, HOLIDAY, NEXT = datetime(2025, 7, 3), datetime(2025, 7, 4), datetime(2025, 7, 7)
TQ = CurveQuotingBasis.TRADING_YEAR


def env(convention=DayCountConvention.BUSINESS_DAYS):
    kw = dict(spot_quote=SpotQuote(100.0), vol_surface=FlatVolSurface(0.2), rate_curve=FlatRateCurve(0.03),
              div_yield=ContinuousDividendYield(0.0), valuation_date=RUN, day_count_convention=convention, calendar=US)
    if convention == DayCountConvention.BUSINESS_DAYS:
        kw.update(rate_curve_quoting=TQ, div_yield_quoting=TQ)
    return PricingEnvironment(**kw)


def flow(payment_date, event=LifecycleEventType.COUPON):
    return RealizedCashflow(cashflow_id="c1", event_type=event, amount=1000.0, determination_date=RUN,
                            payment_date=payment_date)


@pytest.mark.parametrize("event", [LifecycleEventType.COUPON, LifecycleEventType.KNOCK_OUT])
def test_holiday_payment_after_the_run_date_is_timed_at_the_valuation_instant(event):
    timing = SettlementResolver.resolve_pending(flow(HOLIDAY, event), env(), valuation_point=ValuationPoint(date=RUN))
    assert timing.payment_time == 0.0 and timing.payment_df == timing.determination_df
    ledger = LifecycleCashflowLedger([flow(HOLIDAY, event)])
    assert ledger.pending_pv(ValuationPoint(date=RUN), env()) == 1000.0


def test_business_day_payment_is_unchanged():
    e = env()
    timing = SettlementResolver.resolve_pending(flow(NEXT), e, valuation_point=ValuationPoint(date=RUN))
    assert timing.payment_time > 0.0
    assert timing.payment_df == e.get_discount_factor(timing.payment_time)


def test_calendar_day_environment_is_unchanged():
    e = env(DayCountConvention.ACT_365)
    timing = SettlementResolver.resolve_pending(flow(HOLIDAY), e, valuation_point=ValuationPoint(date=RUN))
    assert timing.payment_time == pytest.approx(1 / 365.0, abs=1e-15)


def test_payment_on_the_run_date_is_not_pending():
    """A flow paid ON the run date is paid, not pending: the resolver still refuses it."""
    with pytest.raises(ValidationError, match="strictly after"):
        SettlementResolver.resolve_pending(flow(RUN), env(), valuation_point=ValuationPoint(date=RUN))
