"""Trade-mode lifecycle booking of cash legs (ship spec 2026-10-23 portfolio-trade-value D13–D17; AC10, AC14).

AC10 as a same-environment identity on every run day ``d`` up to the knock-out:

    trade value before the close (the tracker's pre-close copy, legs rolled to ``τ_d`` keeping flows due today)
      + cash already paid
    == portfolio value after ``process_day`` + pending receivables + paid cash

so every leg flow that leaves the position's value that day is booked, once, at the value it left with.
"""
import copy
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.lifecycle import PortfolioLifecycleManager
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.cashleg import (
    AccrualLeg,
    BaseAmount,
    BaseAmountMode,
    DeterministicLeg,
    EventType,
    KOBehavior,
    LegDirection,
    LegSchedule,
)
from quantark.cashleg.accrual_leg import PaymentConvention, SurvivalBasis
from quantark.cashleg.autocallable_leg import AccrualBasis, AutocallableCashLeg, AutocallableLegType
from quantark.cashleg.booking import run_time
from quantark.cashleg.fixed_payoff_leg import FixedPayoffLeg, PaymentTrigger
from quantark.execution.errors import CapabilityError
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio import EquityPosition, Portfolio
from quantark.priceenv import PricingEnvironment

START = datetime(2026, 3, 16)
D = 1 / 365.0
OBS = [30, 60, 90]
KO_DAY = 60
PATH = [0, 5, 10, 15, 29, 30, 31, 45, 59, KO_DAY]


def _snowball():
    return SnowballOption(initial_price=100.0, strike=100.0, maturity=90 * D, contract_multiplier=1_000.0,
                          barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15,
                                                       ko_observation_dates=[d * D for d in OBS],
                                                       ki_barrier=75.0, ki_continuous=True))


def _phoenix():
    """A Phoenix note on the same schedule: a coupon at every observation (barrier 80) and the same
    KO_MATURITY cash legs as a Snowball (user ruling 2026-10-23: a Phoenix's AutocallableCashLeg is never
    coupon-basis)."""
    from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix

    phoenix = create_standard_phoenix(initial_price=100.0, strike=100.0, maturity=OBS[-1] * D, ko_barrier=103.0,
                                      ki_barrier=75.0, coupon_barrier=80.0, coupon_rate=0.01,
                                      num_observations=len(OBS))
    assert phoenix.barrier_config.ko_observation_schedule.times == pytest.approx([d * D for d in OBS], abs=1e-15)
    return phoenix


def _ko_reset():
    """A knock-out-reset note on the same schedule: a knock-in moves it to a second schedule at 98 (rule:
    ``KnockOutResetSnowballOption``); not knocked in here, so it knocks out on the first schedule's day 60."""
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption

    dates = [d * D for d in OBS]
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=90 * D, contract_multiplier=1_000.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_dates=dates,
                                     ki_barrier=75.0, ki_continuous=True),
        post_barrier_config=BarrierConfig(ko_barrier=98.0, ko_rate=0.03, ko_observation_dates=dates))


def _engine(kind):
    from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
    from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine

    engines = {"phoenix": PhoenixQuadEngine, "ko_reset": KOResetSnowballQuadEngine}
    return engines.get(kind, SnowballQuadEngine)(params=QuadParams())


def _accrual(ko, basis, convention=PaymentConvention.AT_PERIOD_END):
    starts = np.array([0] + OBS[:-1]) * D
    ends = np.array(OBS) * D
    return AccrualLeg(direction=LegDirection.BUYER_RECEIVES, rate=0.04,
                      base=BaseAmount(value=100_000.0, mode=BaseAmountMode.ABSOLUTE),
                      schedule=LegSchedule(period_starts=starts, period_ends=ends, payment_times=ends.copy()),
                      payment_convention=convention, ko_behavior=ko, survival_basis=basis)


def _legs(grid=OBS):
    """``grid`` is the note's knock-out event grid in days (a knock-out-reset holds both schedules' records)."""
    obs = tuple(d * D for d in grid)
    return [
        DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=250.0, payment_time=10 * D),
        _accrual(KOBehavior.TRUNCATE_AT_KO, SurvivalBasis.ENTER_PERIOD),
        _accrual(KOBehavior.PAY_FULL_SCHEDULE, SurvivalBasis.COMPLETE_PERIOD),
        _accrual(KOBehavior.TRUNCATE_AT_KO, SurvivalBasis.ENTER_PERIOD, PaymentConvention.AT_MATURITY),
        FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=1_000.0, trigger=PaymentTrigger.AT_KO),
        AutocallableCashLeg(direction=LegDirection.BUYER_RECEIVES, leg_type=AutocallableLegType.REBATE,
                            notional=10_000.0, rate=0.05, observation_schedule=obs, accrual_factors=obs,
                            settlement_schedule=obs, terminal_accrual_factor=90 * D,
                            terminal_settlement_time=90 * D, accrual_basis=AccrualBasis.KO_MATURITY,
                            terminal_events=frozenset({EventType.MATURITY_NO_KO})),
    ]


GRID = {"ko_reset": sorted(OBS * 2)}                                 # each date on both schedules


def _env(day, spot):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot, asset_name="X"), vol_surface=FlatVolSurface(0.2),
                              rate_curve=FlatRateCurve(0.03), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=START + timedelta(days=day))


def _book(legs=None, trade_value=True, kind="snowball"):
    pf = Portfolio(portfolio_name="legs", pricing_environments={"X": _env(0, 100.0)}, creation_date=START,
                   trade_value=trade_value)
    product = {"phoenix": _phoenix, "ko_reset": _ko_reset}.get(kind, _snowball)()
    pf.positions["note"] = EquityPosition(product=product, quantity=1.0, entry_price=0.0, underlying="X",
                                          engine=_engine(kind), entry_timestamp=START,
                                          position_id="note", cash_legs=_legs(GRID.get(kind, OBS)) if legs is None else legs,
                                          entry_trade_value=0.0)
    return pf


def _manager(pf):
    manager = PortfolioLifecycleManager(base_date=START, trade_value=pf.trade_value)
    manager.register_positions(pf)
    manager.register_leg_books(pf)
    return manager


def _pre_close_value(manager, pf, original_legs, day, env):
    """Trade value before ``day``'s close: the pre-close copy, legs rolled keeping flows due today."""
    position = pf.positions["note"]
    tau = run_time(START, START + timedelta(days=day))
    pre = copy.copy(position)
    pre.product = manager._autocallable["note"].product_for_pricing(env.valuation_date, env, initial=True)
    pre.cash_legs = [l for l in (leg.time_shift(tau, keep_landing=True) for leg in original_legs) if l is not None]
    return pre.get_trade_value(env)


@pytest.mark.parametrize("kind", ["snowball", "phoenix", "ko_reset"])
def test_value_plus_booked_cash_is_conserved_through_payments_and_the_knock_out(kind):
    pf = _book(kind=kind)
    original = list(pf.positions["note"].cash_legs)
    manager = _manager(pf)
    paid_before = 0.0
    for i, day in enumerate(PATH):
        env = _env(day, 110.0 if day == KO_DAY else 100.0)
        pf.pricing_environments["X"] = env
        before = _pre_close_value(manager, pf, original, day, env) + paid_before
        manager.process_day(pf, i, env.valuation_date)
        after = pf.get_portfolio_value() + manager.pending_receivable_pv + manager.paid_cash
        assert after == pytest.approx(before, rel=1e-10), f"day {day}"
        paid_before = manager.paid_cash
    assert "note" not in pf.positions                                  # knocked out on day 60
    assert manager.leg_receivable_pv > 0.0                             # PAY_FULL + AT_MATURITY flows still owed


@pytest.mark.parametrize("kind", ["snowball", "phoenix"])
def test_receivables_pay_at_nominal_and_total_leg_cash_is_the_realized_flows(kind):
    pf = _book(kind=kind)
    manager = _manager(pf)
    for i, day in enumerate(PATH + [70, 90, 95]):
        env = _env(day, 110.0 if day == KO_DAY else 100.0)
        pf.pricing_environments["X"] = env
        manager.process_day(pf, i, env.valuation_date)
    c = lambda k: 0.04 * 100_000.0 * 30 * D  # noqa: E731  every period is 30 days
    realized = (-250.0                                    # fee, day 10
                + c(0) + c(1)                             # TRUNCATE/ENTER: periods starting before the KO
                + c(0) + c(1) + c(2)                      # PAY_FULL: every period
                + c(0) + c(1)                             # AT_MATURITY TRUNCATE/ENTER, paid at maturity
                + 1_000.0                                 # FixedPayoff on KO
                + 10_000.0 * 0.05 * KO_DAY * D)           # rebate at the KO observation
    assert manager.leg_paid_cash == pytest.approx(realized, rel=1e-13)
    assert manager.leg_receivable_pv == 0.0


def _dated_snowball():
    """``_snowball`` on the date clock (clock rule 2026-10-26: only a dated contract has an initial date): struck on
    START, knock-out records on START + 30 / 60 / 90 days, expiring on the last."""
    from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule

    records = [ObservationRecord(observation_date=START + timedelta(days=d), barrier=103.0) for d in OBS]
    return SnowballOption(initial_price=100.0, strike=100.0, initial_date=START,
                          exercise_date=START + timedelta(days=OBS[-1]), contract_multiplier=1_000.0,
                          barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15,
                                                       ko_observation_schedule=ObservationSchedule(records=records),
                                                       ki_barrier=75.0, ki_continuous=True))


def test_mid_life_run_anchors_legs_on_the_contract_initial_date():
    """User ruling 2026-10-23: legs count from the contract's initial date (the tracker's clock); a run started 40
    days later sees each flow's time to payment from its own start, and a flow paid before it is history. The note
    is dated (only a dated contract has an initial date; clock rule 2026-10-26)."""
    run_start = START + timedelta(days=40)
    product = _dated_snowball()
    early = DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=100.0, payment_time=10 * D)   # history
    late = DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=250.0, payment_time=50 * D)
    legs = [early, late] + [l for l in _legs() if not isinstance(l, DeterministicLeg)]
    pf = _book(legs=legs)
    pf.positions["note"].product = product
    pf.pricing_environments["X"] = _env(40, 100.0)
    manager = PortfolioLifecycleManager(base_date=run_start, trade_value=True)
    manager.register_positions(pf)
    manager.register_leg_books(pf)
    book = manager._leg_books["note"]
    # pre-run history: the early fee and the first period (paid on day 30) of each period-end accrual
    accruals = [l for l in legs if isinstance(l, AccrualLeg) and l.payment_convention is PaymentConvention.AT_PERIOD_END]
    assert [key for key, _ in book.history] == [(early.leg_id, 0)] + [(l.leg_id, 0) for l in accruals]
    # the run's first valuation sees the late fee 10 days out, the early one gone
    fees = [l for l in pf.positions["note"].cash_legs if isinstance(l, DeterministicLeg)]
    assert [f.payment_time for f in fees] == [50 * D - 40 * D]

    paid_before = 0.0
    for i, day in enumerate([40, 45, 50, 59, KO_DAY]):
        env = _env(day, 110.0 if day == KO_DAY else 100.0)
        pf.pricing_environments["X"] = env
        before = _pre_close_value(manager, pf, legs, day, env) + paid_before
        manager.process_day(pf, i, env.valuation_date)
        after = pf.get_portfolio_value() + manager.pending_receivable_pv + manager.paid_cash
        assert after == pytest.approx(before, rel=1e-10), f"day {day}"
        paid_before = manager.paid_cash
    assert "note" not in pf.positions
    paid = dict(book.paid_today)
    assert (early.leg_id, 0) not in paid
    # leg cash booked in the run: the late fee, accruals of periods starting before the KO, the KO payoff, rebate
    c = 0.04 * 100_000.0 * 30 * D
    assert manager.leg_paid_cash == pytest.approx(-250.0 + c + c + 1_000.0 + 10_000.0 * 0.05 * KO_DAY * D, rel=1e-13)


def test_live_position_carries_the_rolled_legs():
    pf = _book()
    manager = _manager(pf)
    env = _env(15, 100.0)
    pf.pricing_environments["X"] = env
    manager.process_day(pf, 0, env.valuation_date)
    legs = pf.positions["note"].cash_legs
    assert not any(isinstance(l, DeterministicLeg) for l in legs)      # the day-10 fee is paid
    assert manager.leg_paid_cash == -250.0


def test_product_mode_books_no_leg_cash():
    pf = _book(trade_value=False)
    manager = _manager(pf)
    env = _env(15, 100.0)
    pf.pricing_environments["X"] = env
    manager.process_day(pf, 0, env.valuation_date)
    assert manager.leg_paid_cash == 0.0
    assert len(pf.positions["note"].cash_legs) == len(_legs())         # untouched


def test_untracked_position_refuses_termination_paying_legs():
    pf = _book()
    manager = PortfolioLifecycleManager(base_date=START, trade_value=True)   # no trackers registered
    with pytest.raises(CapabilityError):
        manager.register_leg_books(pf)
