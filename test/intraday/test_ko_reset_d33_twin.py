"""Spec D34 / AC37: the intraday twin of a dated KO-reset note in the D33 class carries the first-schedule payment.

A KO-reset note that ends with its first schedule (never knocked in; under EXPIRY pay also a schedule-1 knock-out,
D30) is paid on the product's explicit ``settlement_date`` when one is booked (D33). The intraday twin is numeric and
cannot carry a date, so it carries that payment as ``first_schedule_settlement_time``, resolved from the dated product
once; every engine reads it through the D33 helper (``settlement_support.first_schedule_settlement``).

The reference is the dated product priced on the same engine at the session's valuation instant, on a clock that is
the twin's own: valuation at the 15:00 close, so every fixing (the close) and every payment (deemed at the close) is a
whole number of days away and the twin's seconds-exact ACT/365 fractions equal the dated CALENDAR_DAYS ones; the same
market objects (the trading-clock vol surface included). The two prices then agree to round-off -- bitwise, in
practice -- and a note outside the D33 class carries nothing new (its prices are unchanged).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, time, timedelta

import pytest

from intraday.conftest import SHANGHAI, dated_ko_reset, flat_env
from quantark.asset.equity.engine.mc import SnowballMCEngine
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import KOResetSnowballQuadEngineV2
from quantark.asset.equity.engine.settlement_support import first_schedule_settlement
from quantark.asset.equity.lifecycle import AutocallableLifecycleState, ValuationPoint
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.intraday.context import resolve_context
from quantark.intraday.engines import route_for
from quantark.intraday.events import _schedule_env
from quantark.intraday.profile import VarianceProfile
from quantark.intraday.request import IntradayValuationRequest
from quantark.intraday.timestamp import calendar_year_fraction
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType
from quantark.util.exceptions import ValidationError

T0 = datetime(2026, 3, 16)
DESK = VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))
FRESH, AGED = datetime(2026, 3, 17), datetime(2026, 6, 17)
PAYS = (CouponPayType.INSTANT, CouponPayType.EXPIRY)
SETTLED = 30              # settlement_date = exercise_date + 30 days: the D33 class (schedule 1 ends after 6 months)


def _note(cal, pay, settle_days, *, pre_months=6, spot_ko=103.0):
    note = dated_ko_reset(cal, T0, pre_months=pre_months, post_months=12, pre_ko=spot_ko, rebate_rate=0.10)
    note.accrual_config = replace(note.accrual_config, coupon_pay_type=pay)
    note.settlement_date = None if settle_days is None else note.exercise_date + timedelta(days=settle_days)
    note.validate()
    return note


def _state(day, knocked_in=False):
    return AutocallableLifecycleState(knocked_in=knocked_in, valuation_point=ValuationPoint(date=day))


def _context(sessions, note, day, knocked_in=False):
    ts = datetime(day.year, day.month, day.day, 15, 0, tzinfo=SHANGHAI)       # the close: whole days to every event
    request = IntradayValuationRequest(product=deepcopy(note), pricing_env=flat_env(ts), session_calendar=sessions,
                                       variance_profile=DESK, lifecycle_state=_state(day, knocked_in))
    return resolve_context(request)


def _dated_env(ctx, day):
    """The dated product's environment on the twin's clock: the same market objects, valued at ``day``."""
    env = ctx.pricing_env
    return PricingEnvironment(rate_curve=env.rate_curve, valuation_date=day, spot_quote=env.spot_quote,
                              vol_surface=env.vol_surface, div_yield=env.div_yield, basis_yield=env.basis_yield,
                              day_count_convention=DayCountConvention.CALENDAR_DAYS,
                              bus_days_in_year=env.bus_days_in_year, calendar=env.calendar)


ENGINES = {
    "quad_v2": KOResetSnowballQuadEngineV2,
    "quad_v1": KOResetSnowballQuadEngine,
    "pde_1d": lambda: KOResetSnowballPDESolver(PDEParams()),
    "mc": lambda: SnowballMCEngine(params=MCParams(num_paths=4096, seed=11)),
}
STATEFUL = {"quad_v2"}                # takes the lifecycle state; the others read the knock-in flag off the product


def _price(name, product, env, state):
    engine = ENGINES[name]()
    if name in STATEFUL:
        return float(engine.price(product, env, lifecycle_state=state))
    return float(engine.price(product, env))


def _dated_price(name, note, ctx, day, knocked_in=False):
    dated = deepcopy(note)
    if knocked_in:
        setattr(dated, "_otc_lifecycle_knocked_in", True)
    return _price(name, dated, _dated_env(ctx, day), _state(day, knocked_in))


def _twin_price(name, ctx):
    return _price(name, ctx.numerical.product, ctx.pricing_env, ctx.numerical.lifecycle_state)


def _assert_round_off(got, want):
    assert got == pytest.approx(want, rel=1e-13, abs=1e-13)


# --- the D33 class: the intraday price follows the dated product (AC37) --------------------------------------------

@pytest.mark.parametrize("engine", sorted(ENGINES))
@pytest.mark.parametrize("pay", PAYS, ids=lambda p: p.name)
@pytest.mark.parametrize("day", (FRESH, AGED), ids=("fresh", "aged"))
def test_the_twin_of_a_settled_note_prices_as_the_dated_note(sse_calendar, sse_sessions, engine, pay, day):
    note = _note(sse_calendar, pay, SETTLED)
    ctx = _context(sse_sessions, note, day)
    _assert_round_off(_twin_price(engine, ctx), _dated_price(engine, note, ctx, day))


@pytest.mark.parametrize("pay", PAYS, ids=lambda p: p.name)
@pytest.mark.parametrize("day", (FRESH, AGED), ids=("fresh", "aged"))
def test_the_intraday_route_price_of_a_settled_note_is_the_dated_price_on_quad_v2(sse_calendar, sse_sessions, pay, day):
    note = _note(sse_calendar, pay, SETTLED)
    ctx = _context(sse_sessions, note, day)
    engine = KOResetSnowballQuadEngineV2()
    _assert_round_off(route_for(ctx, engine).price(ctx, engine).contingent_pv, _dated_price("quad_v2", note, ctx, day))


@pytest.mark.parametrize("pay", PAYS, ids=lambda p: p.name)
def test_the_twin_carries_the_settlement_date_as_its_first_schedule_payment(sse_calendar, sse_sessions, pay):
    note = _note(sse_calendar, pay, SETTLED)
    ctx = _context(sse_sessions, note, AGED)
    twin = ctx.numerical.product
    paid = sse_sessions.payment_at(note.settlement_date.date())
    assert twin.first_schedule_settlement_time == calendar_year_fraction(ctx.valuation_timestamp, paid)
    assert ctx.timeline.first_schedule_payment == paid
    settlement = first_schedule_settlement(twin, ctx.pricing_env)
    assert settlement.settlement_date_governs
    assert settlement.payment_time == twin.first_schedule_settlement_time
    # the never-knocked-in note and the knocked-in one are paid on the same settlement date
    assert settlement.payment_time == twin.maturity + twin.settlement_convention.lag


# --- outside the D33 class: nothing is carried, prices agree as before -----------------------------------------------

@pytest.mark.parametrize("engine", sorted(ENGINES))
@pytest.mark.parametrize("pay", PAYS, ids=lambda p: p.name)
@pytest.mark.parametrize("day", (FRESH, AGED), ids=("fresh", "aged"))
def test_a_note_without_a_settlement_date_carries_nothing(sse_calendar, sse_sessions, engine, pay, day):
    note = _note(sse_calendar, pay, None)
    ctx = _context(sse_sessions, note, day)
    assert ctx.numerical.product.first_schedule_settlement_time is None
    assert ctx.timeline.first_schedule_payment is None
    _assert_round_off(_twin_price(engine, ctx), _dated_price(engine, note, ctx, day))


@pytest.mark.parametrize("pay", PAYS, ids=lambda p: p.name)
def test_a_one_maturity_note_carries_nothing(sse_calendar, sse_sessions, pay):
    """Schedule 1 runs to the product maturity: the product terminal already pays on the settlement date (D33)."""
    note = _note(sse_calendar, pay, SETTLED, pre_months=12)
    ctx = _context(sse_sessions, note, AGED)
    assert ctx.numerical.product.first_schedule_settlement_time is None
    _assert_round_off(_twin_price("quad_v2", ctx), _dated_price("quad_v2", note, ctx, AGED))


@pytest.mark.parametrize("engine", ("quad_v2", "pde_1d", "mc"))
@pytest.mark.parametrize("pay", PAYS, ids=lambda p: p.name)
def test_a_knocked_in_settled_note_prices_as_the_dated_note(sse_calendar, sse_sessions, engine, pay):
    """Knocked in, the note is paid with the product terminal: the carried time is never read."""
    note = _note(sse_calendar, pay, SETTLED)
    ctx = _context(sse_sessions, note, AGED, knocked_in=True)
    _assert_round_off(_twin_price(engine, ctx), _dated_price(engine, note, ctx, AGED, knocked_in=True))


# --- the numeric field --------------------------------------------------------------------------------------------

def _numeric_note(pre_end=0.5, convention=None, **kw):
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
    from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
    from quantark.util.enum.option_enums import ObservationType

    pre = [k / 12.0 for k in range(1, int(round(pre_end * 12)) + 1)]
    post = [k / 12.0 for k in range(1, 13)]
    if convention is not None:
        kw["settlement_convention"] = convention
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=103.0) for t in pre]),
            ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE,
            ki_observation_schedule=ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=75.0) for t in post])),
        post_barrier_config=BarrierConfig(
            ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_schedule=ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=95.0) for t in post])),
        payoff_config=PayoffConfig(rebate_rate=0.10, include_principal=False),
        **kw,
    )


def test_a_numeric_note_states_its_first_schedule_payment_explicitly():
    note = _numeric_note(first_schedule_settlement_time=1.1)
    env = flat_env(datetime(2026, 1, 5))
    settlement = first_schedule_settlement(note, env)
    assert settlement.settlement_date_governs and settlement.payment_time == 1.1
    assert settlement.delay_df == pytest.approx(env.get_discount_factor(1.1) / env.get_discount_factor(0.5), rel=1e-15)
    assert note.cache_key() != _numeric_note().cache_key()
    assert "first_schedule_settlement_time" not in _numeric_note().cache_key()      # unchanged keys without it
    without = first_schedule_settlement(_numeric_note(), env)
    assert not without.settlement_date_governs and without.payment_time == 0.5


def test_the_first_schedule_payment_is_a_numeric_time_on_the_numeric_clock_only(sse_calendar):
    with pytest.raises(ValidationError, match="first_schedule_settlement_time"):
        note = _note(sse_calendar, CouponPayType.INSTANT, None)
        note.first_schedule_settlement_time = 0.6
        note.validate()
    with pytest.raises(ValidationError, match="first_schedule_settlement_time"):
        _numeric_note(first_schedule_settlement_time=float("nan"))


def test_aging_a_numeric_note_moves_its_first_schedule_payment():
    note = _numeric_note(first_schedule_settlement_time=1.1)
    note.time_shift(0.25, None, flat_env(datetime(2026, 1, 5)))
    assert note.first_schedule_settlement_time == pytest.approx(0.85, abs=1e-15)


# --- CP9 I1 / M3: the argument means a payment of a note that ends with schedule 1, after schedule 1 --------------

def _lag(years):
    from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit

    return SettlementConvention(lag=years, lag_unit=SettlementLagUnit.YEAR_FRACTION)


def test_a_one_maturity_note_refuses_a_first_schedule_payment_other_than_its_terminal_payment():
    """Schedule 1 runs to the product maturity: the note's one payment is the product terminal's, so a different
    first-schedule time means nothing (MC read it, the grid did not: CP9 I1)."""
    with pytest.raises(ValidationError, match="first_schedule_settlement_time"):
        _numeric_note(pre_end=1.0, first_schedule_settlement_time=3.0)
    with pytest.raises(ValidationError, match="first_schedule_settlement_time"):
        _numeric_note(pre_end=1.0, convention=_lag(0.05), first_schedule_settlement_time=1.0)
    # equal to the terminal payment (round-off is equality): accepted, and it changes nothing
    assert _numeric_note(pre_end=1.0, convention=_lag(0.05), first_schedule_settlement_time=1.05 + 1e-15) is not None
    assert _numeric_note(pre_end=1.0, first_schedule_settlement_time=1.0) is not None


def test_a_first_schedule_payment_before_schedule_one_ends_is_refused():
    with pytest.raises(ValidationError, match="first_schedule_settlement_time"):
        _numeric_note(pre_end=0.5, first_schedule_settlement_time=0.25)
    assert _numeric_note(pre_end=0.5, first_schedule_settlement_time=0.5).first_schedule_settlement_time == 0.5


def test_equality_reads_the_first_schedule_payment_only_when_set():
    assert _numeric_note() == _numeric_note()
    assert _numeric_note(first_schedule_settlement_time=1.1) == _numeric_note(first_schedule_settlement_time=1.1)
    assert _numeric_note(first_schedule_settlement_time=1.1) != _numeric_note(first_schedule_settlement_time=1.2)
    assert _numeric_note(first_schedule_settlement_time=1.1) != _numeric_note()
    assert _numeric_note() != _numeric_note(pre_end=0.25)            # the other fields still count
