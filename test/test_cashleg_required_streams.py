"""CashLeg.required_event_types stream declarations [§11.1].

Each leg declares which EventDistribution streams it reads so the engine can
prune indicator columns: KO-only legs must not force the expensive KI split,
while a MATURITY_WITH_KI terminal (or an AT_KI trigger) does force KI columns.
"""

import numpy as np
import pytest

from quantark.cashleg.accrual_leg import AccrualLeg, KOBehavior
from quantark.cashleg.autocallable_leg import (
    AccrualBasis,
    AutocallableCashLeg,
    AutocallableLegType,
)
from quantark.cashleg.base import LegDirection
from quantark.cashleg.base_amount import BaseAmount, BaseAmountMode
from quantark.cashleg.deterministic_leg import DeterministicLeg
from quantark.cashleg.event_distribution import EventType
from quantark.cashleg.fixed_payoff_leg import FixedPayoffLeg, PaymentTrigger
from quantark.cashleg.leg_schedule import LegSchedule


@pytest.fixture
def make_ko_autocallable_leg():
    def _make(**over):
        kw = dict(
            direction=LegDirection.BUYER_RECEIVES,
            leg_type=AutocallableLegType.REBATE,
            notional=100.0,
            rate=0.05,
            accrual_basis=AccrualBasis.KO_MATURITY,
            terminal_events=frozenset({EventType.MATURITY_NO_KO}),
        )
        kw.update(over)
        return AutocallableCashLeg(**kw)

    return _make


@pytest.fixture
def make_with_ki_autocallable_leg():
    def _make(terminal_events=frozenset({EventType.MATURITY_WITH_KI}), **over):
        kw = dict(
            direction=LegDirection.BUYER_RECEIVES,
            leg_type=AutocallableLegType.MINIMUM_RETURN,
            notional=100.0,
            rate=0.05,
            accrual_basis=AccrualBasis.KO_MATURITY,
            terminal_events=frozenset(terminal_events),
        )
        kw.update(over)
        return AutocallableCashLeg(**kw)

    return _make


@pytest.fixture
def make_fixed_payoff_leg():
    def _make(trigger="AT_MATURITY_WITH_KI", **over):
        kw = dict(
            direction=LegDirection.BUYER_RECEIVES,
            amount=10.0,
            trigger=PaymentTrigger[trigger],
        )
        kw.update(over)
        return FixedPayoffLeg(**kw)

    return _make


def test_autocallable_ko_leg_needs_ko_not_ki(make_ko_autocallable_leg):
    req = make_ko_autocallable_leg().required_event_types()
    assert EventType.KO in req
    assert EventType.KI not in req and EventType.MATURITY_WITH_KI not in req


def test_autocallable_ki_terminal_leg_needs_ki_split(make_with_ki_autocallable_leg):
    req = make_with_ki_autocallable_leg(
        terminal_events={EventType.MATURITY_WITH_KI}
    ).required_event_types()
    assert EventType.MATURITY_WITH_KI in req


def test_autocallable_coupon_basis_needs_coupon(make_with_ki_autocallable_leg):
    req = make_with_ki_autocallable_leg(
        accrual_basis=AccrualBasis.COUPON
    ).required_event_types()
    assert EventType.COUPON in req
    assert EventType.KO in req  # terminal buckets always need KO mass


def test_fixed_payoff_with_ki_needs_ki(make_fixed_payoff_leg):
    req = make_fixed_payoff_leg(trigger="AT_MATURITY_WITH_KI").required_event_types()
    assert EventType.MATURITY_WITH_KI in req


def test_fixed_payoff_at_ki_needs_ki(make_fixed_payoff_leg):
    req = make_fixed_payoff_leg(trigger="AT_KI").required_event_types()
    assert req == frozenset({EventType.KI})


def test_fixed_payoff_at_ko_needs_only_ko(make_fixed_payoff_leg):
    req = make_fixed_payoff_leg(trigger="AT_KO").required_event_types()
    assert req == frozenset({EventType.KO})


def test_accrual_leg_truncated_needs_ko():
    leg = AccrualLeg(
        direction=LegDirection.BUYER_RECEIVES,
        rate=0.03,
        base=BaseAmount(value=1000.0, mode=BaseAmountMode.ABSOLUTE),
        schedule=LegSchedule(
            period_starts=np.array([0.0, 0.5]),
            period_ends=np.array([0.5, 1.0]),
            payment_times=np.array([0.5, 1.0]),
        ),
        ko_behavior=KOBehavior.TRUNCATE_AT_KO,
    )
    assert leg.required_event_types() == frozenset({EventType.KO})


def test_accrual_leg_full_schedule_needs_nothing():
    leg = AccrualLeg(
        direction=LegDirection.BUYER_RECEIVES,
        rate=0.03,
        base=BaseAmount(value=1000.0, mode=BaseAmountMode.ABSOLUTE),
        schedule=LegSchedule(
            period_starts=np.array([0.0, 0.5]),
            period_ends=np.array([0.5, 1.0]),
            payment_times=np.array([0.5, 1.0]),
        ),
        ko_behavior=KOBehavior.PAY_FULL_SCHEDULE,
    )
    assert leg.required_event_types() == frozenset()


def test_deterministic_leg_needs_nothing():
    leg = DeterministicLeg(
        direction=LegDirection.BUYER_PAYS, amount=50.0, payment_time=0.0
    )
    assert leg.required_event_types() == frozenset()
