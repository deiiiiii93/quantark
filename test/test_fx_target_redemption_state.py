"""TargetRedemptionState (spec 2026-10-09-fx-tarn-lifecycle-design.md section 3)."""
import math
from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from quantark.asset.fx.product.option.target_redemption_state import (
    TARGET_HIT_RTOL,
    PendingPayment,
    ResolvedTargetRedemption,
    TargetRedemptionEvent,
    TargetRedemptionEventType,
    TargetRedemptionState,
    TerminalBookingWarning,
    is_target_hit,
    pending_present_value,
    target_hit_tol,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment
from quantark.util.exceptions import ValidationError

D1, D2, D3 = datetime(2026, 7, 15), datetime(2026, 8, 15), datetime(2026, 9, 15)


def test_hit_rtol_keeps_the_engines_value():
    assert TARGET_HIT_RTOL == 1e-12


def test_hit_tolerance_is_the_engine_rule():
    assert target_hit_tol(math.inf) == 0.0
    assert target_hit_tol(3.0) == abs(3.0) * 1e-12
    assert is_target_hit(3.0 - 1e-13, 3.0)
    assert not is_target_hit(3.0 - 1e-11, 3.0)
    assert not is_target_hit(1e300, math.inf)


def test_inception_state():
    s = TargetRedemptionState(0.0)
    assert s.accumulated == 0.0 and s.pending == () and s.as_of is None


def test_pending_list_is_stored_as_a_tuple():
    s = TargetRedemptionState(1.0, pending=[PendingPayment(1.0, pay_time=0.1)])
    assert s.pending == (PendingPayment(1.0, pay_time=0.1),)


def test_state_is_immutable():
    s = TargetRedemptionState(0.0)
    with pytest.raises(FrozenInstanceError):
        s.accumulated = 1.0


@pytest.mark.parametrize("acc", [-1e-9, math.nan, math.inf])
def test_rejects_bad_accumulated(acc):
    with pytest.raises(ValidationError, match="accumulated"):
        TargetRedemptionState(acc)


@pytest.mark.parametrize("kw", [{}, {"pay_time": 0.1, "pay_date": D1}])
def test_pending_needs_exactly_one_of_time_or_date(kw):
    with pytest.raises(ValidationError, match="exactly one"):
        PendingPayment(1.0, **kw)


@pytest.mark.parametrize("t", [0.0, -0.1, 1e-12, math.nan, math.inf])
def test_numeric_pay_time_must_be_after_valuation(t):
    with pytest.raises(ValidationError, match="pay_time"):
        PendingPayment(1.0, pay_time=t)


@pytest.mark.parametrize("amount", [math.nan, math.inf])
def test_pending_amount_must_be_finite(amount):
    with pytest.raises(ValidationError, match="finite"):
        PendingPayment(amount, pay_time=0.1)


def test_rejects_mixed_pending_forms():
    with pytest.raises(ValidationError, match="mixes"):
        TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_time=0.1), PendingPayment(1.0, pay_date=D3)),
                              as_of=D1)


def test_rejects_unsorted_pending():
    with pytest.raises(ValidationError, match="sorted"):
        TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_time=0.2), PendingPayment(1.0, pay_time=0.1)))


def test_equal_pay_times_are_sorted():
    TargetRedemptionState(1.0, pending=(PendingPayment(0.5, pay_time=0.1), PendingPayment(100.0, pay_time=0.1)))


def test_dated_pending_needs_as_of():
    with pytest.raises(ValidationError, match="as_of"):
        TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_date=D2),))


@pytest.mark.parametrize("pay", [D1, datetime(2026, 7, 1)])
def test_dated_pending_must_be_after_as_of(pay):
    with pytest.raises(ValidationError, match="after as_of"):
        TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_date=pay),), as_of=D1)


def test_terminal_booking_warning_is_a_user_warning():
    assert issubclass(TerminalBookingWarning, UserWarning)


def test_event_defaults():
    e = TargetRedemptionEvent(TargetRedemptionEventType.PAYMENT, 1.0)
    assert (e.fixing, e.time, e.redeemed, e.label) == (None, 0.0, False, "")
    assert TargetRedemptionEventType.FIXING.value == "fixing"


def test_resolved_view_is_frozen():
    v = ResolvedTargetRedemption((0.5,), (0.5,), None, 0.0, (), False, False)
    with pytest.raises(FrozenInstanceError):
        v.accumulated = 1.0


def test_pending_present_value_sums_in_order():
    e = FxPricingEnvironment(valuation_date=D1, spot_quote=SpotQuote(spot=1.2),
                             domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
                             vol_surface=FlatVolSurface(volatility=0.1))
    expected = 0.0 + 0.5 * e.get_domestic_df(0.1) + 100.0 * e.get_domestic_df(0.1)
    assert pending_present_value(((0.1, 0.5), (0.1, 100.0)), e) == expected
    assert pending_present_value((), e) == 0.0
