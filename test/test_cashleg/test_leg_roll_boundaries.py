"""Leg-roll boundary cases found at CP2 (ship spec 2026-10-23 portfolio-trade-value D10, D12; AC8, AC9).

- A survival reference that lands on the new origin reads the post-observation survival of an event kept at 0
  (theta keeps the landing observation); only a strictly past reference reads 1.
- An ``AutocallableCashLeg`` whose observations have all passed keeps its terminal branch while the terminal
  settlement is ahead.
"""
from datetime import datetime

import math
import numpy as np
import pytest

from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.asset.equity.riskmeasures.greeks.theta_shift import shift_cash_legs, shift_for_theta
from quantark.cashleg import (
    AccrualLeg,
    BaseAmount,
    BaseAmountMode,
    EventDistribution,
    EventType,
    KOBehavior,
    LegDirection,
    LegSchedule,
)
from quantark.cashleg.accrual_leg import SurvivalBasis
from quantark.cashleg.autocallable_leg import AccrualBasis, AutocallableCashLeg, AutocallableLegType
from quantark.cashleg.booking import LegCashBook
from quantark.execution.errors import CapabilityError
from quantark.util.exceptions import ValidationError
from test_cashleg._autocallable_helpers import make_engine, make_env, make_snowball

D = 1 / 365.0


def test_survival_at_zero_reads_an_event_kept_at_zero():
    knocked = EventDistribution(event_times=np.array([0.0, 0.5]), event_dates=None,
                                probabilities={EventType.KO: np.array([1.0, 0.0]), EventType.MATURITY_NO_KO: 0.0},
                                survival_probability=np.array([1.0, 0.0, 0.0]))
    assert knocked.survival_at(0.0) == 0.0
    assert knocked.survival_at(1e-12) == 0.0
    assert knocked.survival_at(-1e-3) == 1.0
    assert EventDistribution.trivial(maturity=1.0).survival_at(0.0) == 1.0
    no_event_at_zero = EventDistribution(event_times=np.array([0.25, 0.5]), event_dates=None,
                                         probabilities={EventType.KO: np.array([0.3, 0.1]), EventType.MATURITY_NO_KO: 0.6},
                                         survival_probability=np.array([1.0, 0.7, 0.6]))
    assert no_event_at_zero.survival_at(0.0) == 1.0


def test_theta_roll_of_an_accrual_starting_on_a_landing_ko_observation():
    """CP2 Critical: spot above the KO level, KO observation one day ahead, an ENTER_PERIOD period starting on
    it. The note knocks out there, so the period never accrues at t0 or at t1."""
    env = make_env(spot=110.0)
    product = make_snowball(ko_dates=(D, 0.5, 1.0))
    engine = make_engine("quad")
    leg = AccrualLeg(direction=LegDirection.BUYER_RECEIVES, rate=0.04,
                     base=BaseAmount(value=1_000_000.0, mode=BaseAmountMode.ABSOLUTE),
                     schedule=LegSchedule(period_starts=np.array([D, 0.5]), period_ends=np.array([0.5, 1.0]),
                                          payment_times=np.array([0.5, 1.0])),
                     ko_behavior=KOBehavior.TRUNCATE_AT_KO, survival_basis=SurvivalBasis.ENTER_PERIOD)
    coupon = 0.04 * 1e6 * (0.5 - D)
    v0 = leg.value(engine.price_with_events(product, env, emit_distribution=True).event_distribution, env, 1.0)
    gc = GreeksCalculator()
    bc = gc._bump_config
    bumped_date, dt, _ = gc._advance_theta_bump(env, bc.time_bump_days, getattr(bc, "time_bump_mode", "auto"))
    shift = shift_for_theta(product, env, dt, bumped_date)
    t1 = engine.price_with_events(shift.product, shift.env, emit_distribution=True).event_distribution
    (moved,) = shift_cash_legs([leg], dt)
    assert moved.schedule.period_starts[0] == 0.0          # the reference landed on t1
    v1 = moved.value(t1, shift.env, 1.0)
    assert abs(v0) < 1e-6 * coupon
    assert abs(v1) < 1e-6 * coupon


def _rebate(obs, terminal):
    return AutocallableCashLeg(direction=LegDirection.BUYER_RECEIVES, leg_type=AutocallableLegType.REBATE,
                               notional=10_000.0, rate=0.05, observation_schedule=obs, accrual_factors=obs,
                               settlement_schedule=tuple(t + 2 * D for t in obs), terminal_accrual_factor=1.0,
                               terminal_settlement_time=terminal, accrual_basis=AccrualBasis.KO_MATURITY,
                               terminal_events=frozenset({EventType.MATURITY_NO_KO}))


def test_autocallable_leg_keeps_its_terminal_branch_after_the_last_observation():
    leg = _rebate((0.25, 0.5), 1.0)
    after = leg.time_shift(0.6)
    assert after is not None
    assert tuple(after.observation_schedule) == () and after.terminal_settlement_time == pytest.approx(0.4)
    env = make_env()
    assert after.value(EventDistribution.trivial(maturity=0.4), env, 1.0) == pytest.approx(
        10_000.0 * 0.05 * 1.0 * math.exp(-0.03 * 0.4), rel=1e-12)
    assert leg.time_shift(1.1) is None                                  # terminal settlement passed
    assert leg.time_shift(1.0) is None                                  # landing, no keep_landing
    assert leg.time_shift(1.0, keep_landing=True).terminal_settlement_time == 0.0
    terminal_only = _rebate((), 1.0)
    assert terminal_only.time_shift(1.1) is None


def test_book_refuses_duplicate_leg_ids_and_pay_before_survival_on_a_truncating_leg():
    leg = _rebate((0.25, 0.5), 1.0)
    with pytest.raises(ValidationError, match="leg_id"):
        LegCashBook([leg, leg], 1.0, datetime(2026, 3, 16))
    in_advance = AccrualLeg(direction=LegDirection.BUYER_RECEIVES, rate=0.04,
                            base=BaseAmount(value=1.0, mode=BaseAmountMode.ABSOLUTE),
                            schedule=LegSchedule(period_starts=np.array([0.0]), period_ends=np.array([0.5]),
                                                 payment_times=np.array([0.1])),
                            ko_behavior=KOBehavior.TRUNCATE_AT_KO, survival_basis=SurvivalBasis.COMPLETE_PERIOD)
    with pytest.raises(CapabilityError, match="before"):
        LegCashBook([in_advance], 1.0, datetime(2026, 3, 16))
