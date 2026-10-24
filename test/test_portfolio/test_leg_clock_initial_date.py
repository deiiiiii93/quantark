"""Cash-leg times count from the trade's initial date in every consumer (user decision 2026-10-24).

A position's cash-leg times are measured from its contract's clock origin: the product's ``initial_date`` when it
has one (the lifecycle trackers' origin, ``lifecycle_clock_origin``), else the valuation date. Point-in-time
valuation (portfolio, static stress, VaR, SIMM) rolls the legs to the valuation date by whole days / 365 exactly as
the multi-day runs do (``LegCashBook``): a flow paid strictly before the valuation date is history, a flow due that
day is kept at 0. Positions whose legs a run has already rolled carry ``LegClock.VALUATION`` and are not rolled again.
So an aged trade's day 0 is one number across the static tools, the backtest and dynamic scenarios.
"""
import copy
from datetime import timedelta

import numpy as np
import pytest

from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.backtest import BacktestEngine
from quantark.cashleg import (
    AccrualLeg,
    BaseAmount,
    BaseAmountMode,
    DeterministicLeg,
    EventType,
    KOBehavior,
    LegClock,
    LegDirection,
    LegSchedule,
)
from quantark.cashleg.accrual_leg import SurvivalBasis
from quantark.cashleg.autocallable_leg import AccrualBasis, AutocallableCashLeg, AutocallableLegType
from quantark.cashleg.booking import LegCashBook
from quantark.cashleg.fixed_payoff_leg import FixedPayoffLeg, PaymentTrigger
from quantark.dynamicscenario import DynamicScenarioConfig, DynamicScenarioEngine
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio import EquityPosition, Portfolio
from quantark.priceenv import PricingEnvironment
from quantark.stresstest import ScenarioBuilder, StressTestConfig, StressTestEngine
from quantark.util.enum.option_enums import ObservationType
from quantark.util.numerical import is_close
from test_backtest import test_trade_value_runs as runs

D = 1 / 365.0
START = runs.START
AGE = 28                                        # the trade was struck 28 days before the valuation / run start
INITIAL = START - timedelta(days=AGE)
OBS = (56, 84, 112)                              # KO observations, days from INITIAL (Mondays)


def _dated_snowball(initial_date=INITIAL):
    dates = [INITIAL + timedelta(days=d) for d in OBS]
    ko = ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=103.0) for d in dates])
    return SnowballOption(initial_price=100.0, strike=100.0, contract_multiplier=1_000.0, initial_date=initial_date,
                          exercise_date=dates[-1],
                          barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15,
                                                       ko_observation_type=ObservationType.DISCRETE,
                                                       ko_observation_schedule=ko, ki_barrier=75.0,
                                                       ki_continuous=True))


def _legs():
    """Leg times in years from INITIAL: a fee already paid, a fee due on the run start, a fee ahead, and the
    accrual / KO payoff / rebate legs of the note's remaining periods."""
    obs = tuple(d * D for d in OBS)
    starts = np.array([AGE, OBS[0], OBS[1]]) * D
    ends = np.array(OBS) * D
    return [
        DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=100.0, payment_time=14 * D, leg_id="paid"),
        DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=200.0, payment_time=AGE * D, leg_id="today"),
        DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=250.0, payment_time=35 * D, leg_id="fee"),
        AccrualLeg(direction=LegDirection.BUYER_RECEIVES, rate=0.04,
                   base=BaseAmount(value=100_000.0, mode=BaseAmountMode.ABSOLUTE),
                   schedule=LegSchedule(period_starts=starts, period_ends=ends, payment_times=ends.copy()),
                   ko_behavior=KOBehavior.TRUNCATE_AT_KO, survival_basis=SurvivalBasis.ENTER_PERIOD,
                   leg_id="accrual"),
        FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=1_000.0, trigger=PaymentTrigger.AT_KO,
                       leg_id="ko"),
        AutocallableCashLeg(direction=LegDirection.BUYER_RECEIVES, leg_type=AutocallableLegType.REBATE,
                            notional=10_000.0, rate=0.05, observation_schedule=obs, accrual_factors=obs,
                            settlement_schedule=obs, terminal_accrual_factor=OBS[-1] * D,
                            terminal_settlement_time=OBS[-1] * D, accrual_basis=AccrualBasis.KO_MATURITY,
                            terminal_events=frozenset({EventType.MATURITY_NO_KO}), leg_id="rebate"),
    ]


def _env(valuation_date=START, underlying=runs.UNDERLYING, spot=100.0):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot, asset_name=underlying),
                              vol_surface=FlatVolSurface(runs.VOL), rate_curve=FlatRateCurve(runs.RATE),
                              div_yield=ContinuousDividendYield(runs.DIV), valuation_date=valuation_date)


def _position(legs=None, product=None, underlying=runs.UNDERLYING, **kwargs):
    return EquityPosition(product=product or _dated_snowball(), quantity=1.0, entry_price=0.0,
                          underlying=underlying, engine=SnowballQuadEngine(params=QuadParams()),
                          entry_timestamp=START, cash_legs=_legs() if legs is None else legs,
                          entry_trade_value=-500.0, **kwargs)


def _rolled_by_hand(legs, tau):
    rolled = (leg.time_shift(tau, keep_landing=True) for leg in legs)
    return [leg for leg in rolled if leg is not None]


def _portfolio(underlying=runs.UNDERLYING):
    pf = Portfolio(portfolio_name="aged", pricing_environments={underlying: _env(underlying=underlying)},
                   creation_date=START, trade_value=True)
    pf.positions["note"] = _position(underlying=underlying)
    return pf


# --- the position's clock ----------------------------------------------------------------------------------------

def test_a_position_reads_its_leg_times_from_the_trade_initial_date_by_default():
    position = _position()
    assert position.cash_legs_clock is LegClock.CONTRACT
    by_hand = _position(legs=_rolled_by_hand(_legs(), AGE * D), cash_legs_clock=LegClock.VALUATION)
    env = _env()
    assert position.get_trade_value(env) == by_hand.get_trade_value(env)


def test_rolled_legs_drop_history_and_keep_a_flow_due_today_at_zero():
    aged = _position().on_valuation_clock(_env())
    assert aged.cash_legs_clock is LegClock.VALUATION
    ids = [leg.leg_id for leg in aged.cash_legs]
    assert "paid" not in ids
    today = next(leg for leg in aged.cash_legs if leg.leg_id == "today")
    assert today.payment_time == 0.0
    fee = next(leg for leg in aged.cash_legs if leg.leg_id == "fee")
    assert fee.payment_time == pytest.approx(7 * D, abs=1e-15)


def test_the_roll_is_the_multi_day_books_run_start_roll():
    book = LegCashBook(_legs(), 100_000.0, INITIAL, run_start=START)
    aged = _position().on_valuation_clock(_env())
    assert [repr(leg) for leg in aged.cash_legs] == [repr(leg) for leg in book.legs_at_run_start()]


def test_no_initial_date_valuation_frame_clock_or_no_age_leave_the_position_as_is():
    no_origin = _position(product=runs._snowball())
    assert no_origin.on_valuation_clock(_env()) is no_origin
    rolled = _position(cash_legs_clock=LegClock.VALUATION)
    assert rolled.on_valuation_clock(_env()) is rolled
    fresh = _position()
    assert fresh.on_valuation_clock(_env(valuation_date=INITIAL)) is fresh


def test_valuation_before_the_initial_date_does_not_shift_the_legs():
    early = _position()
    assert early.on_valuation_clock(_env(valuation_date=INITIAL - timedelta(days=5))) is early


def test_trade_risk_and_breakdown_follow_the_same_clock():
    env = _env()
    gc = GreeksCalculator()
    by_hand = _position(legs=_rolled_by_hand(_legs(), AGE * D), cash_legs_clock=LegClock.VALUATION)
    assert _position().get_trade_risk(env, gc).total == by_hand.get_trade_risk(env, gc).total
    assert _position().get_trade_value_breakdown(env).total == by_hand.get_trade_value_breakdown(env).total


def test_cash_legs_clock_must_be_a_leg_clock():
    from quantark.util.exceptions import ValidationError

    with pytest.raises(ValidationError):
        _position(cash_legs_clock="valuation")


def test_the_users_example_amount_counts_from_the_initial_date_discount_from_the_valuation_date():
    """User example 2026-10-24: initial 2026-10-01, valuation 2026-10-10, maturity 2027-10-01; a 5 % annualized
    cash leg pays (maturity - initial) x 5 % of its base, discounted over (maturity - valuation)."""
    from datetime import datetime

    initial, valuation, maturity = datetime(2026, 10, 1), datetime(2026, 10, 10), datetime(2027, 10, 1)
    T = (maturity - initial).days * D
    ko_dates = [datetime(2027, 1, 4), datetime(2027, 4, 1), maturity]
    ko = ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=103.0) for d in ko_dates])
    note = SnowballOption(initial_price=100.0, strike=100.0, contract_multiplier=1_000.0, initial_date=initial,
                          exercise_date=maturity,
                          barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15,
                                                       ko_observation_type=ObservationType.DISCRETE,
                                                       ko_observation_schedule=ko, ki_barrier=75.0,
                                                       ki_continuous=True))
    leg = AccrualLeg(direction=LegDirection.BUYER_RECEIVES, rate=0.05,
                     base=BaseAmount(value=1_000_000.0, mode=BaseAmountMode.ABSOLUTE),
                     schedule=LegSchedule(period_starts=np.array([0.0]), period_ends=np.array([T]),
                                          payment_times=np.array([T])),
                     ko_behavior=KOBehavior.PAY_FULL_SCHEDULE)
    position = _position(legs=[leg], product=note)
    env = _env(valuation_date=valuation)
    legs_pv = position.get_trade_value(env) - position.get_market_value(env)
    expected = 1_000_000.0 * 0.05 * T * env.get_discount_factor((maturity - valuation).days * D)
    assert legs_pv == pytest.approx(expected, rel=1e-12)


# --- one day-0 number across the families --------------------------------------------------------------------------

def test_an_aged_trades_day_zero_is_one_number_across_static_stress_dynamic_and_backtest():
    value = copy.deepcopy(_portfolio()).get_portfolio_value()

    stress = StressTestEngine(StressTestConfig(calculate_greeks=False)).run_static_scenarios(
        _portfolio(), [ScenarioBuilder().name("spot -5%").spot_stress(-0.05).build()])
    assert stress.baseline_value == value

    # the run books the fee due on its first day as cash, the static value holds it at t = 0
    backtest = BacktestEngine(runs._config(positions=[_position()])).run()
    assert is_close(backtest.states_df["portfolio_value"].iloc[0], value, rel_tol=1e-12)

    dynamic = DynamicScenarioEngine(DynamicScenarioConfig(calculate_greeks=False, handle_lifecycle_events=True)).run(
        _portfolio("IDX"), runs._rally())
    assert is_close(dynamic.baseline_value, copy.deepcopy(_portfolio("IDX")).get_portfolio_value(), rel_tol=1e-12)
