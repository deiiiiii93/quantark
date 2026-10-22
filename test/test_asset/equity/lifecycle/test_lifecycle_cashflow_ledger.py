"""Immutable lifecycle cashflow-ledger contract tests."""

from copy import deepcopy
from datetime import datetime
from math import exp

import pytest

from quantark.asset.equity.lifecycle import (
    AutocallableLifecycleState,
    LifecycleCashflowLedger,
    LifecycleEventType,
    RealizedCashflow,
    ValuationPoint,
)
from quantark.param import FlatRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError


def _cashflow(**overrides):
    terms = {
        "cashflow_id": "trade-1:coupon:3",
        "event_type": LifecycleEventType.COUPON,
        "amount": 12.5,
        "determination_date": datetime(2026, 8, 3),
        "payment_date": datetime(2026, 8, 5),
    }
    terms.update(overrides)
    return RealizedCashflow(**terms)


def test_register_is_idempotent_for_an_identical_payload():
    cashflow = _cashflow()
    ledger = LifecycleCashflowLedger()

    assert ledger.register(cashflow)
    assert not ledger.register(cashflow)
    assert ledger.cashflows == (cashflow,)


@pytest.mark.parametrize(
    "override",
    [
        {"amount": 13.0},
        {"payment_date": datetime(2026, 8, 6)},
        {
            "determination_date": None,
            "payment_date": None,
            "determination_time": 0.25,
            "payment_time": 0.51,
        },
    ],
)
def test_register_rejects_conflicting_payload_for_existing_id(override):
    ledger = LifecycleCashflowLedger()
    assert ledger.register(_cashflow())

    with pytest.raises(ValidationError, match="conflicting cashflow"):
        ledger.register(_cashflow(**override))


def test_pending_and_paid_partition_by_date():
    cashflow = _cashflow()
    ledger = LifecycleCashflowLedger([cashflow])

    assert ledger.pending(
        ValuationPoint(date=datetime(2026, 8, 4))
    ) == (cashflow,)
    assert ledger.paid(ValuationPoint(date=datetime(2026, 8, 4))) == ()
    assert ledger.pending(ValuationPoint(date=datetime(2026, 8, 5))) == ()
    assert ledger.paid(
        ValuationPoint(date=datetime(2026, 8, 5))
    ) == (cashflow,)


def test_pending_and_paid_partition_by_numeric_time():
    cashflow = _cashflow(
        determination_date=None,
        payment_date=None,
        determination_time=0.25,
        payment_time=0.50,
    )
    ledger = LifecycleCashflowLedger([cashflow])

    assert ledger.pending(ValuationPoint(time=0.40)) == (cashflow,)
    assert ledger.paid(ValuationPoint(time=0.40)) == ()
    assert ledger.pending(ValuationPoint(time=0.50)) == ()
    assert ledger.paid(ValuationPoint(time=0.50)) == (cashflow,)


def test_pending_pv_discounts_from_current_valuation_to_payment():
    amount = 1_000.0
    cashflow = _cashflow(amount=amount)
    ledger = LifecycleCashflowLedger([cashflow])
    env = PricingEnvironment(
        rate_curve=FlatRateCurve(0.03),
        valuation_date=datetime(2026, 8, 4),
    )

    actual = ledger.pending_pv(
        ValuationPoint(date=env.valuation_date),
        env,
    )

    expected = amount * exp(-0.03 / 365.0)
    original_determination_discounting = amount * exp(-0.03 * 2.0 / 365.0)
    assert actual == pytest.approx(expected)
    assert actual != pytest.approx(original_determination_discounting)


def test_realized_cashflows_compatibility_property_includes_paid_cash_only():
    paid = _cashflow(
        cashflow_id="paid",
        amount=7.0,
        payment_date=datetime(2026, 8, 3),
    )
    pending = _cashflow(
        cashflow_id="pending",
        amount=11.0,
        payment_date=datetime(2026, 8, 5),
    )
    state = AutocallableLifecycleState(
        valuation_point=ValuationPoint(date=datetime(2026, 8, 4)),
        ledger=LifecycleCashflowLedger([pending, paid]),
    )

    assert state.realized_cashflows == pytest.approx(7.0)


def test_state_deepcopy_preserves_independent_ledger_contents():
    state = AutocallableLifecycleState(
        valuation_point=ValuationPoint(date=datetime(2026, 8, 4)),
        ledger=LifecycleCashflowLedger([_cashflow()]),
    )

    copied = deepcopy(state)
    copied.ledger.register(
        _cashflow(
            cashflow_id="trade-1:coupon:4",
            determination_date=datetime(2026, 9, 3),
            payment_date=datetime(2026, 9, 7),
        )
    )

    assert copied.ledger.cashflows[:1] == state.ledger.cashflows
    assert len(state.ledger.cashflows) == 1
    assert len(copied.ledger.cashflows) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"date": datetime(2026, 8, 4), "time": 0.25},
        {"time": float("nan")},
    ],
)
def test_valuation_point_requires_one_finite_representation(kwargs):
    with pytest.raises(ValidationError):
        ValuationPoint(**kwargs)


def test_cashflow_rejects_payment_before_determination():
    with pytest.raises(ValidationError, match="before determination"):
        _cashflow(payment_date=datetime(2026, 8, 2))
