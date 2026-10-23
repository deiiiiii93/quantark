"""Leg roll rule shared by theta and multi-day booking (ship spec 2026-10-23 portfolio-trade-value D10; AC8).

``time_shift(dt, *, keep_landing=False)`` moves a leg's own times by ``-dt``: a flow whose payment falls strictly
before the new origin drops; one landing on it (``is_zero``) stays at 0 only with ``keep_landing``; ``None`` when
nothing remains. Amounts never change. ``AccrualLeg(AT_MATURITY)`` pays at the maturity settlement, so its periods
never drop on their schedule times; ``FixedPayoffLeg`` has no own times.
"""
import copy
import math
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine import BlackScholesEngine
from quantark.asset.equity.product import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.cashleg import (
    AccrualLeg,
    BaseAmount,
    BaseAmountMode,
    DeterministicLeg,
    EventDistribution,
    KOBehavior,
    LegDirection,
    LegSchedule,
)
from quantark.cashleg.accrual_leg import PaymentConvention, SurvivalBasis
from quantark.cashleg.fixed_payoff_leg import FixedPayoffLeg, PaymentTrigger
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio import EquityPosition
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType

VAL = datetime(2026, 3, 16)
RATE = 0.03


def _env():
    return PricingEnvironment(spot_quote=SpotQuote(spot=100.0, asset_name="X"), vol_surface=FlatVolSurface(0.2),
                              rate_curve=FlatRateCurve(RATE), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=VAL)


def _accrual(convention=PaymentConvention.AT_PERIOD_END, ko=KOBehavior.TRUNCATE_AT_KO,
             basis=SurvivalBasis.ENTER_PERIOD):
    starts = np.array([0.0, 0.1, 0.35, 0.6])
    ends = np.array([0.1, 0.35, 0.6, 0.85])
    return AccrualLeg(direction=LegDirection.BUYER_RECEIVES, rate=0.04,
                      base=BaseAmount(value=1_000_000.0, mode=BaseAmountMode.ABSOLUTE),
                      schedule=LegSchedule(period_starts=starts, period_ends=ends, payment_times=ends.copy()),
                      payment_convention=convention, ko_behavior=ko, survival_basis=basis)


# --- DeterministicLeg ------------------------------------------------------------------------------------------

def test_deterministic_moves_drops_and_lands():
    leg = DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=250.0, payment_time=0.25)
    moved = leg.time_shift(0.1)
    assert moved.payment_time == 0.25 - 0.1
    assert moved.amount == leg.amount and moved.leg_id == leg.leg_id and moved.direction is leg.direction
    assert leg.time_shift(0.3) is None                       # strictly past
    assert leg.time_shift(0.25) is None                      # landing, dropped without keep_landing
    landed = leg.time_shift(0.25 + 1e-12, keep_landing=True)  # landing within Tolerance.ZERO, kept at 0
    assert landed.payment_time == 0.0
    assert leg.time_shift(0.0) is leg and leg.time_shift(-0.1) is leg


# --- AccrualLeg ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ko", list(KOBehavior))
def test_accrual_period_end_drops_paid_periods_and_keeps_amounts(ko):
    leg = _accrual(ko=ko)
    fractions = leg.accrual_fractions()
    moved = leg.time_shift(0.4)
    # periods paid at 0.1 and 0.35 drop; 0.6 and 0.85 survive with their exact original fractions
    np.testing.assert_array_equal(moved.schedule.payment_times, np.array([0.6, 0.85]) - 0.4)
    np.testing.assert_array_equal(moved.schedule.period_starts, np.array([0.35, 0.6]) - 0.4)
    assert list(moved.accrual_fractions()) == list(fractions[2:])
    assert moved.leg_id == leg.leg_id and moved.ko_behavior is ko
    # a period whose start passed 0 (0.35 - 0.4 < 0) is still representable and reads as survived
    assert moved.schedule.period_starts[0] < 0.0
    dist = EventDistribution.trivial(maturity=0.85 - 0.4)
    pv = moved.value(dist, _env(), 0.0)
    want = 0.04 * 1_000_000.0 * sum(f * math.exp(-RATE * (t - 0.4)) for f, t in zip(fractions[2:], [0.6, 0.85]))
    assert pv == pytest.approx(want, rel=1e-14)


def test_accrual_landing_and_exhaustion():
    leg = _accrual()
    assert len(leg.time_shift(0.35).schedule.payment_times) == 2           # 0.35 lands: dropped
    kept = leg.time_shift(0.35, keep_landing=True)
    np.testing.assert_array_equal(kept.schedule.payment_times, [0.0, 0.6 - 0.35, 0.85 - 0.35])
    assert leg.time_shift(0.9) is None


def test_repeated_shifts_keep_fractions_exact():
    leg = _accrual()
    fractions = leg.accrual_fractions()
    moved = leg.time_shift(0.05).time_shift(0.07).time_shift(0.11)
    assert list(moved.accrual_fractions()) == list(fractions[1:])


@pytest.mark.parametrize("ko", list(KOBehavior))
def test_accrual_at_maturity_never_drops_on_schedule_times(ko):
    leg = _accrual(convention=PaymentConvention.AT_MATURITY, ko=ko)
    fractions = leg.accrual_fractions()
    moved = leg.time_shift(0.7)
    assert list(moved.accrual_fractions()) == list(fractions)              # all four periods still owed
    assert np.all(moved.schedule.payment_times >= 0.0)
    # on a live note (past survival 1) the value is every period at the maturity settlement
    dist = EventDistribution.trivial(maturity=1.0 - 0.7)
    want = 0.04 * 1_000_000.0 * float(np.sum(fractions)) * math.exp(-RATE * 0.3)
    assert moved.value(dist, _env(), 0.0) == pytest.approx(want, rel=1e-14)


def test_dated_accrual_keeps_its_dates_with_the_kept_periods():
    starts = [datetime(2026, 3, 16), datetime(2026, 6, 16), datetime(2026, 9, 16)]
    ends = [datetime(2026, 6, 16), datetime(2026, 9, 16), datetime(2026, 12, 16)]
    t = lambda d: (d - VAL).days / 365.0  # noqa: E731
    leg = AccrualLeg(direction=LegDirection.BUYER_RECEIVES, rate=0.04,
                     base=BaseAmount(value=1_000_000.0, mode=BaseAmountMode.ABSOLUTE),
                     schedule=LegSchedule(period_starts=np.array([t(d) for d in starts]),
                                          period_ends=np.array([t(d) for d in ends]),
                                          payment_times=np.array([t(d) for d in ends]),
                                          period_start_dates=starts, period_end_dates=ends, payment_dates=ends))
    moved = leg.time_shift(0.3)
    assert moved.schedule.period_start_dates == starts[1:]
    assert moved.schedule.payment_dates == ends[1:]
    assert list(moved.accrual_fractions()) == list(leg.accrual_fractions()[1:])


# --- FixedPayoffLeg --------------------------------------------------------------------------------------------

def test_fixed_payoff_has_no_own_times():
    leg = FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=10.0, trigger=PaymentTrigger.AT_KO)
    assert leg.time_shift(0.3) is leg
    assert leg.time_shift(0.3, keep_landing=True) is leg


# --- theta picks the rolls up ----------------------------------------------------------------------------------

def _position(legs):
    vanilla = EuropeanVanillaOption(strike=100.0, maturity=0.5, option_type=OptionType.CALL)
    return EquityPosition(product=vanilla, quantity=10, entry_price=0.0, underlying="X",
                          engine=BlackScholesEngine(), entry_timestamp=VAL, cash_legs=legs)


def _theta_bump(gc, env):
    bc = gc._bump_config
    _date, time_bump, _mode = gc._advance_theta_bump(env, bc.time_bump_days, getattr(bc, "time_bump_mode", "auto"))
    return time_bump


def test_trade_theta_includes_the_deterministic_leg_carry():
    gc = GreeksCalculator()
    env = _env()
    leg = DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=250.0, payment_time=0.25)
    risk = _position([leg]).get_trade_risk(copy.deepcopy(env), gc, greeks=["theta"])
    dt = _theta_bump(gc, env)
    carry = -250.0 * (math.exp(-RATE * (0.25 - dt)) - math.exp(-RATE * 0.25))
    assert risk.total["theta"] - risk.product["theta"] == pytest.approx(carry, rel=1e-10)
    assert carry != 0.0


def test_trade_theta_loses_a_payment_inside_the_bump():
    """User ruling 2026-10-07: engine-determined cash paid within the bump is gone at t1 (theta ≈ −payment)."""
    gc = GreeksCalculator()
    env = _env()
    dt = _theta_bump(gc, env)
    leg = DeterministicLeg(direction=LegDirection.BUYER_RECEIVES, amount=250.0, payment_time=dt / 2)
    risk = _position([leg]).get_trade_risk(copy.deepcopy(env), gc, greeks=["theta"])
    assert risk.total["theta"] - risk.product["theta"] == pytest.approx(-250.0 * math.exp(-RATE * dt / 2), rel=1e-12)
