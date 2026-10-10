"""FX TARN / TARF lifecycle: construction, ageing, pricing, dated twins (spec 2026-10-09-fx-tarn-lifecycle-design.md,
section 10)."""
import copy
import dataclasses
import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import pytest

from golden_compare import GOLDEN_REL_TOL
from fx_target_redemption_fixtures import (
    FIX,
    FIX_DATES,
    INCEPTION,
    LAG2,
    PAIR,
    TARGET_CAL,
    VAL,
    _numeric_twin,
    dated_note,
    dated_tarf,
    engine_for,
    env,
    following,
    mc,
    note,
    tarf,
)
from quantark.asset.fx.engine.mc import FxTargetRedemptionNoteMCEngine, FxTarnForwardMCEngine, FxTarnMCResult
from quantark.asset.fx.product.option import FxTargetRedemptionForward, FxTargetRedemptionNote
from quantark.asset.fx.product.option.target_redemption_state import (
    PendingPayment,
    TargetRedemptionEvent,
    TargetRedemptionEventType,
    TargetRedemptionState,
    TerminalBookingWarning,
)
from quantark.param import SpotQuote
from quantark.util.calendar import DayCountConvention, calculate_year_fraction
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import MissingFixingWarning, freeze_missing_fixings
from quantark.util.numerical import is_close

FIXING, PAYMENT = TargetRedemptionEventType.FIXING, TargetRedemptionEventType.PAYMENT


# --- R1 (findings section 10): the period start is a contract term -------------------------------------------------

def test_r1_rebuild_without_a_period_source_is_refused():
    remaining = [t - 0.125 for t in FIX[1:]]
    with pytest.raises(ValidationError, match="accrual_start"):
        FxTargetRedemptionNote(fixing_times=remaining, coupon_rate=0.12, notional=100.0, strike=1.10,
                               currency_pair=PAIR, state=INCEPTION)


def test_r1_accrual_start_restores_the_full_first_coupon():
    remaining = [t - 0.125 for t in FIX[1:]]
    rebuilt = note(fix=remaining, accrual_start=FIX[0] - 0.125)
    assert is_close(rebuilt.accrual_factors[0], 1 / 12, rel_tol=1e-12)
    assert is_close(rebuilt.full_coupon(0), 1.0, rel_tol=1e-12)


def test_accrual_start_zero_is_todays_default_bitwise():
    explicit = note(fix=[0.5, 1.0, 1.5])
    assert explicit.accrual_factors == [0.5 - 0.0, 1.0 - 0.5, 1.5 - 1.0]


def test_inception_horizon_is_unchanged():
    n = note(pay_times=[t + 0.02 for t in FIX])
    assert n.maturity == FIX[-1] and n.delivery == FIX[-1] + 0.02
    assert n.lifecycle_events == () and n.is_numeric and not n.is_terminal and not n.is_redeemed


def test_hit_tolerance_lives_on_the_product():
    assert note(target=3.0).hit_tol == 3.0 * 1e-12
    assert note().hit_tol == 0.0


# --- R2 (numeric TARN refusals) ------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw, match", [
    (dict(accrual_factors=[1 / 12] * 12, accrual_start=0.0), "not both"),
    (dict(state=None), "state is required"),
    (dict(state=TargetRedemptionState(3.0 + 1e-9), target=3.0), "exceeds the target"),
    (dict(accrual_start=FIX[0]), "before the first listed fixing"),
    (dict(accrual_start=0.01, state=TargetRedemptionState(1.0), target=3.0), "contiguous"),
    (dict(state=TargetRedemptionState(1.0, pending=(PendingPayment(-1.0, pay_time=0.1),)), target=3.0),
     "positive"),
    (dict(state=TargetRedemptionState(3.0), target=3.0), "redeemed"),
    (dict(coupon_rate=None), "coupon_rate"),
])
def test_r2_numeric_tarn_refusals(kw, match):
    with pytest.raises(ValidationError, match=match):
        note(**kw)


def test_r2_numeric_state_refuses_as_of_and_pay_dates():
    with pytest.raises(ValidationError, match="as_of"):
        note(state=TargetRedemptionState(0.0, as_of=datetime(2026, 6, 1)))
    with pytest.raises(ValidationError, match="dated pending cash"):
        note(state=TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_date=datetime(2026, 7, 1)),)),
             target=3.0)


def test_r2_missing_state_is_a_validation_error_not_a_type_error():
    with pytest.raises(ValidationError, match="state is required"):
        FxTargetRedemptionNote(fixing_times=FIX, coupon_rate=0.12, strike=1.10, accrual_start=0.0)


def test_past_fixing_times_are_accepted():
    n = note(fix=[-0.01] + FIX[1:], accrual_start=-0.05)
    assert n.fixing_times[0] == -0.01


# --- WARN1 (numeric construction part; values in Task 6) -----------------------------------------------------------

def test_warn1_empty_schedule_booking_warns_once():
    with pytest.warns(TerminalBookingWarning) as rec:
        n = note(fix=[], accrual_factors=[], target=3.0,
                 state=TargetRedemptionState(1.0, pending=(PendingPayment(100.0, pay_time=0.05),)))
    assert len([w for w in rec if issubclass(w.category, TerminalBookingWarning)]) == 1
    assert n.is_terminal and not n.is_redeemed
    assert n.maturity == 0.05 and n.delivery == 0.05


def test_warn1_redeemed_booking_warns_once():
    with pytest.warns(TerminalBookingWarning):
        n = note(fix=[], accrual_factors=[], target=1.5, state=TargetRedemptionState(1.5))
    assert n.is_redeemed and n.is_terminal and n.maturity == 0.0


def test_live_booking_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", TerminalBookingWarning)
        note()


def test_deepcopy_does_not_warn_and_keeps_the_state():
    with pytest.warns(TerminalBookingWarning):
        n = note(fix=[], accrual_factors=[], target=1.5, state=TargetRedemptionState(1.5))
    with warnings.catch_warnings():
        warnings.simplefilter("error", TerminalBookingWarning)
        c = copy.deepcopy(n)
    assert c.state == n.state and c.lifecycle_events == ()


@pytest.mark.parametrize("kw, match", [
    (dict(state=None), "state is required"),
    (dict(state=TargetRedemptionState(0.2 + 1e-9), target=0.2), "exceeds the target"),
    (dict(state=TargetRedemptionState(0.2), target=0.2), "redeemed"),
])
def test_r2_numeric_tarf_refusals(kw, match):
    with pytest.raises(ValidationError, match=match):
        tarf(**kw)


def test_tarf_pending_may_be_negative():
    f = tarf(gearing=2.0, target=0.2, state=TargetRedemptionState(0.0, pending=(PendingPayment(-0.1, pay_time=0.01),)))
    assert f.state.pending[0].amount == -0.1


def test_tarf_terminal_booking_warns():
    with pytest.warns(TerminalBookingWarning):
        f = tarf(fix=[], target=0.2, state=TargetRedemptionState(0.05))
    assert f.is_terminal and not f.is_redeemed and f.maturity == 0.0


def test_tarf_inception_horizon_is_unchanged():
    f = tarf(pay_times=[t + 0.02 for t in FIX])
    assert f.maturity == FIX[-1] and f.delivery == FIX[-1] + 0.02 and f.hit_tol == 0.0


# --- numeric ageing hooks (spec section 6, Q3) ---------------------------------------------------------------

LAG = [t + 0.02 for t in FIX]


def _kinds(events):
    return [(e.event_type, e.amount, e.fixing, e.time, e.redeemed) for e in events]


def test_time_shift_keeps_a_passed_fixing_unbanked():
    n = note(target=3.0, pay_times=LAG)
    a = copy.deepcopy(n)
    a.time_shift(0.1)
    assert a.fixing_times == [t - 0.1 for t in FIX] and a.pay_times == [t - 0.1 for t in LAG]
    assert a.state == n.state and a.lifecycle_events == ()
    assert a.maturity == FIX[-1] - 0.1


def test_record_banks_the_passed_fixing_at_the_held_spot():
    n = note(target=3.0, pay_times=LAG)
    a = copy.deepcopy(n)
    a.time_shift(0.1)
    a.record_passed_fixings(env(spot=1.20))
    c = n.full_coupon(0)
    assert a.state == TargetRedemptionState(c, pending=(PendingPayment(c, pay_time=LAG[0] - 0.1),))
    assert a.fixing_times == [t - 0.1 for t in FIX[1:]] and a.accrual_factors == n.accrual_factors[1:]
    assert _kinds(a.lifecycle_events) == [(FIXING, c, 1.20, FIX[0] - 0.1, False)]
    assert a.lifecycle_events[0].label.startswith("#0 t=")


def test_pending_cash_leaves_when_a_later_shift_reaches_its_pay_time():
    n = note(target=3.0, pay_times=LAG)
    a = copy.deepcopy(n)
    a.time_shift(0.1)
    a.record_passed_fixings(env())
    a.time_shift(0.01)
    c = n.full_coupon(0)
    assert a.state.pending == ()
    assert _kinds(a.lifecycle_events[-1:]) == [(PAYMENT, c, None, (LAG[0] - 0.1) - 0.01, False)]


def test_redemption_clips_the_coupon_and_settles_the_principal():
    n = note(target=1.5, pay_times=LAG)
    a = copy.deepcopy(n)
    a.time_shift(0.175)
    a.record_passed_fixings(env(spot=1.20))
    c0 = n.full_coupon(0)
    c1 = min(n.full_coupon(1), 1.5 - c0)
    t_pay = LAG[1] - 0.175
    assert a.is_redeemed and a.is_terminal
    assert a.fixing_times == [] and a.pay_times == [] and a.accrual_factors == []
    assert a.state == TargetRedemptionState(c0 + c1, pending=(PendingPayment(c1, pay_time=t_pay),
                                                              PendingPayment(100.0, pay_time=t_pay)))
    assert _kinds(a.lifecycle_events) == [
        (FIXING, c0, 1.20, FIX[0] - 0.175, False),
        (PAYMENT, c0, None, LAG[0] - 0.175, False),
        (FIXING, c1, 1.20, FIX[1] - 0.175, True),
    ]
    assert a.lifecycle_events[1].label == a.lifecycle_events[0].label
    assert a.maturity == t_pay


def test_quiet_fixing_banks_zero_and_no_cash():
    a = note(target=3.0, pay_times=LAG)
    a.time_shift(0.1)
    a.record_passed_fixings(env(spot=1.05))
    assert a.state == TargetRedemptionState(0.0)
    assert _kinds(a.lifecycle_events) == [(FIXING, 0.0, 1.05, FIX[0] - 0.1, False)]


@pytest.mark.parametrize("principal", [True, False])
def test_last_fixing_without_a_hit_settles_the_principal(principal):
    a = note(fix=[0.05], pay_times=[0.07], include_principal=principal)
    a.time_shift(0.06)
    a.record_passed_fixings(env(spot=1.05))
    assert a.is_terminal and not a.is_redeemed
    assert a.state.pending == ((PendingPayment(100.0, pay_time=0.07 - 0.06),) if principal else ())
    assert _kinds(a.lifecycle_events) == [(FIXING, 0.0, 1.05, 0.05 - 0.06, True)]


def test_tarf_unfavourable_fixing_is_negative_pending_cash():
    a = tarf(gearing=2.0, target=0.2, pay_times=LAG)
    a.time_shift(0.1)
    a.record_passed_fixings(env(spot=1.10))
    diff = 1.0 * (1.10 - 1.15)
    cf = 0.0 - 2.0 * (1.0 * max(-diff, 0.0))
    assert cf < 0.0
    assert a.state == TargetRedemptionState(0.0, pending=(PendingPayment(cf, pay_time=LAG[0] - 0.1),))


def test_tarf_redeems_on_target():
    a = tarf(target=0.075, pay_times=LAG)
    a.time_shift(0.175)
    a.record_passed_fixings(env(spot=1.20))
    fav = 1.0 * max(1.0 * (1.20 - 1.15), 0.0)
    second = min(fav, 0.075 - fav)
    assert a.is_redeemed and a.fixing_times == []
    assert a.state.accumulated == fav + second
    assert a.state.pending == (PendingPayment(second, pay_time=LAG[1] - 0.175),)  # gearing 1, favourable fixing


def test_record_before_the_first_fixing_changes_nothing():
    a = note(target=3.0)
    before = copy.deepcopy(a)
    a.record_passed_fixings(env())
    assert a.fixing_times == before.fixing_times and a.state == before.state and a.lifecycle_events == ()


def test_lifecycle_events_survive_deepcopy():
    a = note(target=3.0)
    a.time_shift(0.1)
    a.record_passed_fixings(env())
    assert copy.deepcopy(a).lifecycle_events == a.lifecycle_events


def test_banking_uses_quoted_spot_not_effective_spot():
    e = env(spot=1.20, spot_days=2)
    assert e.effective_spot() != e.spot
    a = note(target=3.0)
    a.time_shift(0.1)
    a.record_passed_fixings(e)
    assert a.lifecycle_events[0].fixing == e.spot


PSEUDO, QUASI = MonteCarloMethod.PSEUDO, MonteCarloMethod.QUASI


def _age(product, tau, spot):
    aged = copy.deepcopy(product)
    aged.time_shift(tau)
    aged.record_passed_fixings(env(spot=spot))
    return aged


def _rebuild_note(aged, source, *, accrual_start=None):
    kw = dict(fixing_times=list(aged.fixing_times), pay_times=list(aged.pay_times), coupon_rate=source.coupon_rate,
              notional=source.notional, target=source.target, strike=source.strike,
              include_principal=source.include_principal, currency_pair=PAIR, state=aged.state)
    if accrual_start is None:
        kw["accrual_factors"] = list(aged.accrual_factors)  # C0's surviving factors
    else:
        kw["accrual_start"] = accrual_start
    return FxTargetRedemptionNote(**kw)


def _rebuild_tarf(aged, source):
    return FxTargetRedemptionForward(strike=source.strike, fixing_times=list(aged.fixing_times),
                                     pay_times=list(aged.pay_times), notional=source.notional, target=source.target,
                                     gearing=source.gearing, currency_pair=PAIR, state=aged.state)


def _price(product, method=PSEUDO, e=None, n=20_000):
    return engine_for(product, n=n, method=method).price(product, e or env())


# --- P1-P5: numeric-vs-aged parity (A1.1) --------------------------------------------------------------------------

@pytest.mark.parametrize("method", [PSEUDO, QUASI])
@pytest.mark.parametrize("spot_h", [1.20, 1.05])
@pytest.mark.parametrize("tau", [0.125, 91 / 365])
def test_p1a_tarn_aged_equals_rebuilt_from_surviving_factors(tau, spot_h, method):
    c0 = note(target=3.0)
    aged = _age(c0, tau, spot_h)
    assert _price(aged, method) == _price(_rebuild_note(aged, c0), method)


@pytest.mark.parametrize("tau, n_banked, bitwise", [(0.125, 1, True), (91 / 365, 2, False)])
def test_p1b_tarn_rebuilt_from_accrual_start(tau, n_banked, bitwise):
    c0 = note(target=3.0)
    aged = _age(c0, tau, 1.20)
    assert len(aged.fixing_times) == 12 - n_banked
    p_aged = _price(aged)
    p_rebuilt = _price(_rebuild_note(aged, c0, accrual_start=FIX[n_banked - 1] - tau))  # Ruling R4
    assert p_aged == p_rebuilt if bitwise else is_close(p_aged, p_rebuilt, rel_tol=1e-12)


@pytest.mark.parametrize("tau", [0.1, 0.125, 91 / 365])  # Ruling R10: tau = 0.1 leaves the coupon pending
def test_p2_pay_lag_pending_is_stated_and_prices_bitwise(tau):
    c0 = note(target=3.0, pay_times=LAG)
    aged = _age(c0, tau, 1.20)
    if tau == 0.1:
        c = c0.full_coupon(0)
        stated = TargetRedemptionState(c, pending=(PendingPayment(c, pay_time=LAG[0] - 0.1),))
        assert aged.state == stated
    rebuilt = _rebuild_note(aged, c0)
    assert _price(aged) == _price(rebuilt)


def test_p3_across_redemption_aged_and_rebuilt_terminal():
    c0 = note(target=1.5, pay_times=LAG)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TerminalBookingWarning)  # WARN2: an aged copy that redeems does not warn
        aged = _age(c0, 0.175, 1.20)
        e = env()
        p_aged = _price(aged, e=e)
    c1 = min(c0.full_coupon(1), 1.5 - c0.full_coupon(0))
    t_pay = LAG[1] - 0.175
    assert p_aged == 0.0 + c1 * e.get_domestic_df(t_pay) + 100.0 * e.get_domestic_df(t_pay)
    with pytest.warns(TerminalBookingWarning):  # WARN1
        rebuilt = note(fix=[], accrual_factors=[], target=1.5, state=aged.state)
    assert _price(rebuilt, e=e) == p_aged


@pytest.mark.parametrize("method", [PSEUDO, QUASI])
@pytest.mark.parametrize("spot_h", [1.20, 1.10])
@pytest.mark.parametrize("tau", [0.1, 0.125, 91 / 365])
def test_p4_tarf_aged_equals_rebuilt(tau, spot_h, method):
    c0 = tarf(target=0.20, gearing=2.0, pay_times=LAG)
    aged = _age(c0, tau, spot_h)
    if tau == 0.1 and spot_h == 1.10:
        assert aged.state.pending and aged.state.pending[0].amount < 0.0
    assert _price(aged, method) == _price(_rebuild_tarf(aged, c0), method)


def test_p4_tarf_across_redemption():
    c0 = tarf(target=0.075, pay_times=LAG)
    aged = _age(c0, 0.175, 1.20)
    e = env()
    fav = 1.0 * max(1.0 * (1.20 - 1.15), 0.0)
    second = min(fav, 0.075 - fav)
    assert aged.is_terminal and _price(aged, e=e) == 0.0 + second * e.get_domestic_df(LAG[1] - 0.175)
    with pytest.warns(TerminalBookingWarning):
        rebuilt = tarf(fix=[], target=0.075, state=aged.state)
    assert _price(rebuilt, e=e) == _price(aged, e=e)


def test_p5_daily_ageing_equals_one_rebuild():
    c0 = note(target=3.0)
    f0 = tarf(target=0.20)
    aged_n, aged_f = copy.deepcopy(c0), copy.deepcopy(f0)
    e = env()
    for _ in range(91):
        for a in (aged_n, aged_f):
            a.time_shift(1 / 365)
            a.record_passed_fixings(e)
    tau = 91 / 365
    rebuilt_n = note(fix=[t - tau for t in FIX[2:]], accrual_factors=c0.accrual_factors[2:], target=3.0,
                     state=TargetRedemptionState(c0.full_coupon(0) + c0.full_coupon(1)))
    fav = 1.0 * max(1.0 * (1.20 - 1.15), 0.0)
    rebuilt_f = tarf(fix=[t - tau for t in FIX[2:]], target=0.20, state=TargetRedemptionState(fav + fav))
    assert aged_n.state.accumulated == rebuilt_n.state.accumulated
    assert aged_f.state.accumulated == rebuilt_f.state.accumulated
    assert is_close(_price(aged_n), _price(rebuilt_n), rel_tol=1e-12)
    assert is_close(_price(aged_f), _price(rebuilt_f), rel_tol=1e-12)


# --- WARN1 values (numeric) and terminal engine result -------------------------------------------------------------

def test_warn1_terminal_values_and_no_repeat_warning():
    with pytest.warns(TerminalBookingWarning):
        empty = note(fix=[], accrual_factors=[], target=3.0, state=TargetRedemptionState(1.0))
        paid_out = tarf(fix=[], target=0.2, state=TargetRedemptionState(0.2))
        owed = note(fix=[], accrual_factors=[], target=1.5,
                    state=TargetRedemptionState(1.5, pending=(PendingPayment(100.5, pay_time=0.05),)))
    e = env()
    with warnings.catch_warnings():
        warnings.simplefilter("error", TerminalBookingWarning)
        assert _price(empty, e=e) == 0.0
        assert _price(paid_out, e=e) == 0.0
        assert _price(owed, e=e) == 0.0 + 100.5 * e.get_domestic_df(0.05)
        for product in (empty, paid_out, owed):
            engine_for(product).calculate_greeks(product, e)


def test_terminal_engine_result_fields():
    with pytest.warns(TerminalBookingWarning):
        redeemed = note(fix=[], accrual_factors=[], target=1.5, state=TargetRedemptionState(1.5))
    eng = FxTargetRedemptionNoteMCEngine(mc())
    assert eng.price(redeemed, env()) == 0.0
    assert eng.get_last_result() == FxTarnMCResult(0.0, 0.0, 0, 0.0, 1.0)


# --- WARN3 (numeric): an unbanked passed fixing is filled once, on a copy ------------------------------------------

@pytest.mark.parametrize("make", [lambda: note(target=3.0), lambda: tarf(target=0.20)])
def test_warn3_numeric_passed_fixing_is_filled_with_one_warning(make):
    bare = make()
    bare.time_shift(0.125)
    snapshot = copy.deepcopy(bare)
    e = env(spot=1.20)
    with pytest.warns(MissingFixingWarning) as rec:
        p = _price(bare, e=e)
    missing = [w for w in rec if issubclass(w.category, MissingFixingWarning)]
    assert len(missing) == 1 and "#0" in str(missing[0].message)
    recorded = copy.deepcopy(bare)
    recorded.record_passed_fixings(e)
    assert p == _price(recorded, e=e)
    assert (bare.fixing_times, bare.pay_times, bare.state, bare.lifecycle_events) == (
        snapshot.fixing_times, snapshot.pay_times, snapshot.state, snapshot.lifecycle_events)


# --- T1, T2: theta across a fixing / past the last fixing ----------------------------------------------------------

@pytest.mark.parametrize("first", [0.5 / 365, 1 / 365])
@pytest.mark.parametrize("make", [lambda f: note(fix=f, target=3.0), lambda f: tarf(fix=f, target=0.05)])
def test_t1_theta_steps_over_a_fixing(first, make):
    product = make([first] + FIX[1:])
    greeks = engine_for(product).calculate_greeks(product, env())
    assert math.isfinite(greeks["theta"])


def test_t1_two_days_away_is_mains_theta_to_the_bit():
    fix = [2 / 365] + FIX[1:]
    n, f = note(fix=fix, target=3.0), tarf(fix=fix, target=0.05)
    gn = engine_for(n).calculate_greeks(n, env())
    gf = engine_for(f).calculate_greeks(f, env())
    # main dcea77b6 (plan Task 1 Step 4): inception prices and theta unchanged to the bit
    assert (gn["price"], gn["theta"]) == (101.63518163030555, 0.01686139016959487)
    assert (gf["price"], gf["theta"]) == (0.01925287164907336, 0.003076911159477322)


def test_t2_bump_past_the_last_fixing_is_the_pending_carry():
    opt = note(fix=[0.5 / 365], pay_times=[30 / 365], target=3.0)
    e = env()
    g = engine_for(opt).calculate_greeks(opt, e)
    c = opt.full_coupon(0)  # spot 1.20 >= 1.10: the coupon pays
    t = 30 / 365 - 1 / 365
    future = 0.0 + c * e.get_domestic_df(t) + 100.0 * e.get_domestic_df(t)
    assert g["theta"] == (future - g["price"]) / 1.0


def test_t2_terminal_theta_is_pure_carry():
    with pytest.warns(TerminalBookingWarning):
        owed = note(fix=[], accrual_factors=[], target=1.5,
                    state=TargetRedemptionState(1.5, pending=(PendingPayment(100.5, pay_time=30 / 365),)))
    e = env()
    g = engine_for(owed).calculate_greeks(owed, e)
    base = 0.0 + 100.5 * e.get_domestic_df(30 / 365)
    future = 0.0 + 100.5 * e.get_domestic_df(30 / 365 - 1 / 365)
    assert g["price"] == base and g["theta"] == (future - base) / 1.0


BUS = dict(dc=DayCountConvention.BUSINESS_DAYS, calendar=TARGET_CAL)
BUS_DATES = [following(d) for d in FIX_DATES]


def _dated(kind, dates=FIX_DATES, **kw):
    if kind == "tarf":
        return dated_tarf(fixing_dates=dates, target=kw.pop("target", 0.20), **kw)
    return dated_note(fixing_dates=dates, target=kw.pop("target", 3.0), include_principal=(kind == "tarn"), **kw)


def _at(e, days, spot=None):
    changes = {"valuation_date": e.valuation_date + timedelta(days=days)}
    if spot is not None:
        changes["spot_quote"] = SpotQuote(spot=spot)
    return dataclasses.replace(e, **changes)


# --- D1: twins bitwise at booking ----------------------------------------------------------------------------------

@pytest.mark.parametrize("calendar", ["calendar", "business"])
@pytest.mark.parametrize("method", [PSEUDO, QUASI])
@pytest.mark.parametrize("kind", ["tarn", "tarn_no_principal", "tarf"])
def test_d1_twin_prices_bitwise_at_booking(kind, method, calendar):
    e = env() if calendar == "calendar" else env(**BUS)
    dated = _dated(kind, dates=FIX_DATES if calendar == "calendar" else BUS_DATES)
    twin = _numeric_twin(dated, e)
    assert _price(dated, method, e) == _price(twin, method, e)


# --- D3: a dated booking rebuilt mid-life equals the aged contract -------------------------------------------------

@pytest.mark.parametrize("days", [61, 100])  # 61: on the 2nd fixing, its coupon pending; 100: redeemed and paid
def test_d3_dated_rebuild_mid_life_equals_aged_and_so_does_its_twin(days):
    e0 = env(spot=1.20)
    aged = dated_note(target=3.0, pay_dates=LAG2)
    twin = _numeric_twin(aged, e0)
    for k in range(1, days + 1):
        ek = _at(e0, k)
        aged.record_passed_fixings(ek)
        twin.time_shift(1 / 365)
        twin.record_passed_fixings(ek)
    mid = _at(e0, days)
    assert aged.is_terminal == (days == 100) and bool(aged.state.pending) == (days == 61)
    with warnings.catch_warnings(record=True) as rec:  # WARN1: only the terminal rebuild is a terminal booking
        warnings.simplefilter("always", TerminalBookingWarning)
        rebuilt = dated_note(target=3.0, pay_dates=LAG2, state=aged.state)
        rebuilt_twin = _numeric_twin(rebuilt, mid)
    assert len([w for w in rec if issubclass(w.category, TerminalBookingWarning)]) == (2 if days == 100 else 0)
    assert rebuilt.state == aged.state and aged.state.as_of == mid.valuation_date
    assert is_close(_price(rebuilt, e=mid), _price(aged, e=mid), rel_tol=1e-12)
    assert rebuilt_twin.state.accumulated == twin.state.accumulated
    assert [p.amount for p in rebuilt_twin.state.pending] == [p.amount for p in twin.state.pending]
    assert is_close(_price(rebuilt_twin, e=mid), _price(twin, e=mid), rel_tol=1e-12)


# --- D4 (CALENDAR_DAYS): theta twins agree, including a fixing inside the bump -------------------------------------

@pytest.mark.parametrize("inside_bump", [False, True])
@pytest.mark.parametrize("kind", ["tarn", "tarf"])
def test_d4_theta_twins_agree_calendar_days(kind, inside_bump):
    dates = ([VAL + timedelta(days=1)] + FIX_DATES[1:]) if inside_bump else FIX_DATES
    kw = {"initial_date": VAL - timedelta(days=29)} if (inside_bump and kind == "tarn") else {}
    dated = _dated(kind, dates=dates, **kw)
    e = env()
    twin = _numeric_twin(dated, e)
    gd = engine_for(dated).calculate_greeks(dated, e)
    gn = engine_for(twin).calculate_greeks(twin, e)
    assert gd["price"] == gn["price"]
    assert abs(gd["theta"] - gn["theta"]) <= 1e-12 * abs(gd["price"])  # Ruling R5



@pytest.mark.parametrize("kind", ["tarn", "tarf"])
@pytest.mark.parametrize("val", [datetime(2026, 6, 12), datetime(2026, 6, 15)])  # Friday, Monday
def test_d4_theta_twins_agree_business_days(kind, val):
    e = env(val=val, **BUS)
    dated = _dated(kind, dates=BUS_DATES, **({"initial_date": val} if kind != "tarf" else {}))
    twin = _numeric_twin(dated, e)
    gd = engine_for(dated).calculate_greeks(dated, e)
    gn = engine_for(twin).calculate_greeks(twin, e)
    assert abs(gd["theta"] - gn["theta"]) <= 1e-12 * abs(gd["price"])  # Ruling R5

# --- D5: dated accrual factors -------------------------------------------------------------------------------------

def test_d5_initial_date_gives_act_365_factors():
    view = dated_note().resolve(env())
    starts = [VAL] + FIX_DATES[:-1]
    assert view.accrual_factors == tuple(
        calculate_year_fraction(s, d, DayCountConvention.ACT_365, 252) for s, d in zip(starts, FIX_DATES))


def test_d5_business_days_contract_count_reads_the_environment_calendar():
    dated = dated_note(fixing_dates=BUS_DATES, day_count_convention=DayCountConvention.BUSINESS_DAYS)
    view = dated.resolve(env(calendar=TARGET_CAL))
    starts = [VAL] + BUS_DATES[:-1]
    assert view.accrual_factors == tuple(
        calculate_year_fraction(s, d, DayCountConvention.BUSINESS_DAYS, 252, calendar=TARGET_CAL)
        for s, d in zip(starts, BUS_DATES))
    with pytest.raises(ValidationError, match="calendar"):
        dated.resolve(env())  # no calendar on the environment


def test_d5_both_period_sources_refused():
    with pytest.raises(ValidationError, match="not both"):
        dated_note(initial_date=VAL, accrual_factors=[1 / 12] * 12)


def test_d5_neither_period_source_refused():
    with pytest.raises(ValidationError, match="initial_date"):
        FxTargetRedemptionNote(fixing_dates=FIX_DATES, coupon_rate=0.12, strike=1.10, state=INCEPTION)


# --- R2 (dated and mixed forms) ------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    dict(fixing_times=FIX),                                   # both schedules
    dict(pay_times=FIX),                                      # numeric pay with dates
    dict(accrual_start=0.0),                                  # numeric start with dates
    dict(accrual_factors=[1 / 12] * 12, day_count_convention=DayCountConvention.ACT_365, initial_date=None),
    dict(initial_date=FIX_DATES[0]),                          # initial_date not before the first fixing
    dict(fixing_dates=[]),
    dict(fixing_dates=list(reversed(FIX_DATES))),
    dict(fixing_dates=FIX_DATES[:1] * 2),
    dict(pay_dates=FIX_DATES[:-1]),
    dict(pay_dates=[d - timedelta(days=1) for d in FIX_DATES]),
    dict(state=TargetRedemptionState(1.0)),                   # accumulated > 0 with as_of None
    dict(state=TargetRedemptionState(1.0, as_of=VAL)),        # as_of before the first fixing
])
def test_r2_dated_refusals(kw):
    with pytest.raises(ValidationError):
        dated_note(**kw)


@pytest.mark.parametrize("kw", [
    dict(pay_dates=FIX_DATES), dict(initial_date=VAL), dict(day_count_convention=DayCountConvention.ACT_365)])
def test_r2_dated_terms_refused_on_the_numeric_form(kw):
    with pytest.raises(ValidationError, match="dated form"):
        note(**kw)


def test_r2_neither_schedule_refused():
    with pytest.raises(ValidationError, match="schedule"):
        FxTargetRedemptionForward(strike=1.15, state=INCEPTION)


def test_r2_dated_pending_must_not_precede_as_of():
    with pytest.raises(ValidationError, match="after as_of"):
        dated_note(state=TargetRedemptionState(1.0, pending=(PendingPayment(1.0, pay_date=FIX_DATES[0]),),
                                               as_of=FIX_DATES[1]))


def test_r2_as_of_after_valuation_is_refused_at_pricing():
    booked = dated_note(state=TargetRedemptionState(0.0, as_of=FIX_DATES[1]))
    with pytest.raises(ValidationError, match="as_of"):
        _price(booked, e=env())


THURSDAY, FRIDAY, SATURDAY, MONDAY = (datetime(2026, 8, 13), datetime(2026, 8, 14), datetime(2026, 8, 15),
                                      datetime(2026, 8, 17))


def _business_twin_on_friday(dated):
    """The numeric twin booked on the Thursday and aged to the Friday by the business-day step (both forms are then
    valued at the Friday with the same history)."""
    twin = _numeric_twin(dated, env(val=THURSDAY, **BUS))
    twin.time_shift(calculate_year_fraction(THURSDAY, FRIDAY, DayCountConvention.BUSINESS_DAYS, 252,
                                            calendar=TARGET_CAL))
    return twin


def _kinds_only(events):
    return [(e.event_type, e.amount, e.fixing, e.redeemed) for e in events]


def test_bookings_that_state_only_as_of_keep_their_meaning():
    """OQ-3 ruling: banked is tracked per fixing; a booking stating only as_of banks the fixings dated on or before
    it."""
    d = dated_note(target=3.0, state=TargetRedemptionState(1.0, as_of=FIX_DATES[0] + timedelta(days=1)))
    assert d._banked_count() == 1 and not d._is_unbanked(0) and d._is_unbanked(1)
    assert dated_note(target=3.0)._banked_count() == 0
    with pytest.warns(TerminalBookingWarning):
        done = dated_note(target=3.0, state=TargetRedemptionState(1.0, as_of=FIX_DATES[-1]))
    assert done._banked_count() == 12 and done.is_terminal


def test_record_counts_banked_fixings_and_as_of_is_the_valuation_date():  # CALENDAR_DAYS: R24 is the date itself
    d = dated_note(target=3.0, pay_dates=LAG2)
    e = env(val=FIX_DATES[1] + timedelta(days=1))
    d.record_passed_fixings(e)
    assert (d.state.fixings_banked, d.state.as_of) == (2, e.valuation_date)


@pytest.mark.parametrize("kw, match", [
    (dict(state=TargetRedemptionState(1.0, as_of=FIX_DATES[0], fixings_banked=13)), "fixings_banked"),
    (dict(state=TargetRedemptionState(1.0, as_of=FIX_DATES[0], fixings_banked=0)), "as_of"),  # cash, nothing banked
])
def test_r2_dated_banked_count_refusals(kw, match):
    with pytest.raises(ValidationError, match=match):
        dated_note(target=3.0, **kw)


def test_r2_numeric_state_refuses_a_banked_count():
    with pytest.raises(ValidationError, match="fixings_banked"):
        note(state=TargetRedemptionState(0.0, fixings_banked=0))


def test_future_fixing_at_zero_business_time_is_live_like_its_twin():  # OQ-2 (b) (was refused, Ruling R7)
    d = dated_note(fixing_dates=[SATURDAY, datetime(2026, 9, 15)], initial_date=datetime(2026, 7, 15), target=3.0)
    e_fri = env(val=FRIDAY, **BUS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)          # Task 9 makes a t = 0 fill silent
        view = d.resolve(e_fri)                                         # banked on a copy at the quoted spot
        banked = copy.deepcopy(d)
        banked.record_passed_fixings(e_fri)
        assert view == banked.resolve(e_fri)
        c = banked.lifecycle_events[0].amount
        assert view.pending == ((0.0, c),)                              # its cash is due "today"
        assert (banked.state.fixings_banked, banked.state.as_of) == (1, FRIDAY)   # never past the valuation date
        assert is_close(_price(_business_twin_on_friday(d), e=e_fri), _price(d, e=e_fri), rel_tol=1e-12)


def test_pending_pay_date_at_zero_business_time_is_due_today():  # OQ-2 (b) (was refused, Ruling R7)
    c = 1.0
    owed = dated_note(fixing_dates=[THURSDAY, datetime(2026, 9, 15)], pay_dates=[SATURDAY, datetime(2026, 9, 17)],
                      initial_date=datetime(2026, 7, 15), target=3.0,
                      state=TargetRedemptionState(c, pending=(PendingPayment(c, pay_date=SATURDAY),), as_of=THURSDAY))
    assert owed.resolve(env(val=FRIDAY, **BUS)).pending == ((0.0, c),)
    owed.record_passed_fixings(env(val=FRIDAY, **BUS))
    assert owed.state.pending == (PendingPayment(c, pay_date=SATURDAY),)            # not paid on the Friday
    owed.record_passed_fixings(env(val=MONDAY, **BUS))
    assert owed.state.pending == () and owed.lifecycle_events[-1].event_type is PAYMENT


def test_oq3_friday_and_saturday_fixings_price_on_friday_and_bank_once():
    """OQ-3 ruling: zero-lag fixings on a Friday and the Saturday after it, under BUSINESS_DAYS. On the Friday both are
    passed by time (the Saturday one live like the Friday one, at Friday's spot): the dated contract prices, equal to
    its numeric twin, and banks them once; as_of stays the Friday, so the Friday coupon due that day does not precede
    it. On Monday nothing is banked again (fixing identity) and both coupons are paid, as in the twin. The twin is
    booked on the Friday from the dated view, so FIXING events are compared from the Friday on (Task 8c adds the
    Thursday valuation, both fixings at one business time, and twin event parity for shared fixing dates)."""
    dated = dated_note(fixing_dates=[FRIDAY, SATURDAY, datetime(2026, 9, 15)], initial_date=datetime(2026, 7, 15),
                       target=50.0)
    e_fri, e_mon = env(val=FRIDAY, **BUS), env(val=MONDAY, **BUS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        twin = _numeric_twin(dated, e_fri)
        assert is_close(_price(twin, e=e_fri), _price(dated, e=e_fri), rel_tol=1e-12)
        dated.record_passed_fixings(e_fri)
    assert [e.event_type for e in dated.lifecycle_events] == [FIXING, FIXING]
    assert (dated.state.fixings_banked, dated.state.as_of) == (2, FRIDAY)
    assert [p.pay_date for p in dated.state.pending] == [FRIDAY, SATURDAY]            # both due today, in the value
    assert dated.state.accumulated == twin.state.accumulated
    assert [p.amount for p in dated.state.pending] == [p.amount for p in twin.state.pending]
    twin.time_shift(calculate_year_fraction(FRIDAY, MONDAY, DayCountConvention.BUSINESS_DAYS, 252,
                                            calendar=TARGET_CAL))
    twin.record_passed_fixings(e_mon)
    dated.record_passed_fixings(e_mon)
    assert [e.event_type for e in dated.lifecycle_events] == [FIXING, FIXING, PAYMENT, PAYMENT]   # banked once
    assert _kinds_only(dated.lifecycle_events[2:]) == _kinds_only(twin.lifecycle_events)          # same payments
    assert (dated.state.fixings_banked, dated.state.as_of) == (2, MONDAY)


def test_as_of_on_a_weekend_valuation_is_the_friday():
    """Ruling R24: valued on the Saturday under BUSINESS_DAYS (a weekend row), a zero-lag Friday fixing is still live
    (t = 0, W6 Ruling 4) and its Friday coupon is due at that instant; banking writes as_of = the Friday, the first date
    of the valuation's zero-time stretch (never after the valuation date), so the kept check "a pending pay_date must
    not precede as_of" holds and the contract prices like its numeric twin."""
    d = dated_note(fixing_dates=[FRIDAY, datetime(2026, 9, 15)], initial_date=datetime(2026, 7, 15), target=50.0)
    e_sat = env(val=SATURDAY, **BUS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        banked = copy.deepcopy(d)
        banked.record_passed_fixings(e_sat)
        assert (banked.state.as_of, banked.state.fixings_banked) == (FRIDAY, 1)
        assert banked.state.pending[0].pay_date == FRIDAY
        twin = _numeric_twin(d, env(val=THURSDAY, **BUS))
        twin.time_shift(calculate_year_fraction(THURSDAY, SATURDAY, DayCountConvention.BUSINESS_DAYS, 252,
                                                calendar=TARGET_CAL))
        assert is_close(_price(twin, e=e_sat), _price(d, e=e_sat), rel_tol=1e-12)


SUNDAY = datetime(2026, 8, 16)


@pytest.mark.parametrize("form", ["dated", "numeric"])
def test_saturday_cash_valued_on_the_sunday_after_a_friday_record(form):
    """Task 8a's interim corner, closed by Ruling R24: as_of Friday, cash dated Saturday, valued on the Sunday under
    BUSINESS_DAYS. The Saturday is at the Sunday's instant, so the cash is due today: resolve / price hold it at t = 0
    and record_passed_fixings keeps it pending (as_of = the Friday, not the Sunday, so the kept check "a pending
    pay_date must not precede as_of" holds). Dated == numeric: the numeric twin booked on the Friday and aged to the
    Sunday by the business-day step holds the same cash at t = 0 and the same price."""
    c = 1.0
    dated = dated_note(fixing_dates=[THURSDAY, datetime(2026, 9, 15)], pay_dates=[SATURDAY, datetime(2026, 9, 17)],
                       initial_date=datetime(2026, 7, 15), target=3.0,
                       state=TargetRedemptionState(c, pending=(PendingPayment(c, pay_date=SATURDAY),), as_of=FRIDAY))
    e_sun = env(val=SUNDAY, **BUS)
    twin = _numeric_twin(dated, env(val=FRIDAY, **BUS))
    twin.time_shift(calculate_year_fraction(FRIDAY, SUNDAY, DayCountConvention.BUSINESS_DAYS, 252,
                                            calendar=TARGET_CAL))
    product, other = (dated, twin) if form == "dated" else (twin, dated)
    assert product.resolve(e_sun).pending == ((0.0, c),)
    price = _price(product, e=e_sun)
    assert is_close(price, _price(other, e=e_sun), rel_tol=1e-12)          # dated == numeric, each way round
    product.record_passed_fixings(e_sun)
    assert [p.amount for p in product.state.pending] == [c] and product.lifecycle_events == ()
    if form == "dated":
        assert (product.state.as_of, product.state.fixings_banked) == (FRIDAY, 1)
    assert is_close(_price(product, e=e_sun), price, rel_tol=1e-12)


# --- dated record semantics, maturity ------------------------------------------------------------------------------

def test_dated_record_banks_through_the_valuation_date_and_pays_cash_once_it_is_past():
    d = dated_note(target=3.0, pay_dates=LAG2)
    d.record_passed_fixings(env(val=FIX_DATES[0]))
    coupon = 100.0 * 0.12 * calculate_year_fraction(VAL, FIX_DATES[0], DayCountConvention.ACT_365, 252)
    assert d.state == TargetRedemptionState(coupon, pending=(PendingPayment(coupon, pay_date=LAG2[0]),),
                                            as_of=FIX_DATES[0], fixings_banked=1)
    d.record_passed_fixings(env(val=LAG2[0]))                    # its pay date: due today, still in the value
    assert d.state.pending == (PendingPayment(coupon, pay_date=LAG2[0]),) and d.state.as_of == LAG2[0]
    assert [e.event_type for e in d.lifecycle_events] == [FIXING]
    d.record_passed_fixings(env(val=LAG2[0] + timedelta(days=1)))
    assert d.state.pending == () and d.lifecycle_events[-1].event_type is PAYMENT
    assert d.state.as_of == LAG2[0] + timedelta(days=1)


def test_dated_time_shift_is_a_noop():
    d = dated_note()
    d.time_shift(0.5)
    assert d.state == INCEPTION and d.lifecycle_events == ()


def test_dated_maturity_and_delivery_after_the_end_are_zero():
    with pytest.warns(TerminalBookingWarning):
        done = dated_note(target=3.0, pay_dates=LAG2, state=TargetRedemptionState(1.0, as_of=FIX_DATES[-1]))
    late = env(val=LAG2[-1] + timedelta(days=5))
    assert done.get_maturity(late) == 0.0 and done.get_delivery(late) == 0.0
    assert done.get_maturity(env()) == calculate_year_fraction(VAL, FIX_DATES[-1],
                                                                DayCountConvention.CALENDAR_DAYS, 252)


# --- WARN1 / WARN3 (dated) -----------------------------------------------------------------------------------------

def test_warn1_dated_terminal_bookings():
    e = env(val=FIX_DATES[-1] + timedelta(days=1))
    with pytest.warns(TerminalBookingWarning):
        all_banked = dated_note(target=3.0, pay_dates=LAG2, state=TargetRedemptionState(
            1.0, pending=(PendingPayment(100.0, pay_date=LAG2[-1]),), as_of=FIX_DATES[-1]))
    with pytest.warns(TerminalBookingWarning):
        redeemed = dated_note(target=1.5, pay_dates=LAG2, state=TargetRedemptionState(1.5, as_of=FIX_DATES[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", TerminalBookingWarning)
        t = calculate_year_fraction(e.valuation_date, LAG2[-1], DayCountConvention.CALENDAR_DAYS, 252)
        assert _price(all_banked, e=e) == 0.0 + 100.0 * e.get_domestic_df(t)
        assert _price(redeemed, e=e) == 0.0
        engine_for(all_banked).calculate_greeks(all_banked, e)


def test_warn3_dated_passed_fixing_is_filled_with_one_warning():
    d = dated_note(target=3.0)
    snapshot = copy.deepcopy(d)
    e = env(val=FIX_DATES[0] + timedelta(days=5), spot=1.20)
    with pytest.warns(MissingFixingWarning) as rec:
        p = _price(d, e=e)
    missing = [w for w in rec if issubclass(w.category, MissingFixingWarning)]
    assert len(missing) == 1 and FIX_DATES[0].date().isoformat() in str(missing[0].message)
    recorded = copy.deepcopy(d)
    recorded.record_passed_fixings(e)
    assert p == _price(recorded, e=e)
    assert d.state == snapshot.state and d.lifecycle_events == snapshot.lifecycle_events


def test_dated_tarn_without_factors_points_coupon_queries_at_resolve():
    d = dated_note()
    for query in (lambda: d.full_coupon(0), lambda: d.get_payoff(1.20)):
        with pytest.raises(ValidationError, match=r"resolve\(fx_env\)"):
            query()


# --- W5 protocol: a filled past fixing stays at the base spot under bumps and shocks (user decision 2026-10-09) -----
# The fill of an unbanked passed TARN / TARF fixing is its fixing RATE, the env's quoted spot (Q3). It is decided
# once at the base environment (freeze_missing_fixings, at the top of BaseFxEngine.calculate_greeks) and every bumped
# or shocked repricing banks the fixing at that rate: the Greeks equal those of the contract with the fixing banked
# at the base spot, and the warning fires once per call, never per bump.

FILL_VAL = FIX_DATES[0] + timedelta(days=2)  # one dated fixing passed and unbanked


def _numeric_unbanked(kind, tau=0.125, **kw):
    product = note(target=3.0, **kw) if kind == "tarn" else tarf(target=0.20, **kw)
    product.time_shift(tau)  # FIX[0] (and FIX[1] for tau > 1/6) now passed, unbanked
    return product


def _dated_unbanked(kind, **kw):
    return dated_note(target=3.0, **kw) if kind == "tarn" else dated_tarf(target=0.20, **kw)


def _unbanked(form, kind, **kw):
    return _numeric_unbanked(kind, **kw) if form == "numeric" else _dated_unbanked(kind, **kw)


def _fill_env(form, spot=1.20):
    return env(spot=spot) if form == "numeric" else env(spot=spot, val=FILL_VAL)


def _banked_at(product, e):
    banked = copy.deepcopy(product)
    banked.record_passed_fixings(e)
    return banked


def _missing(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [w for w in caught if issubclass(w.category, MissingFixingWarning)]


def _greeks(product, e):
    return engine_for(product, n=4_000).calculate_greeks(product, e)


FILL_FORMS = pytest.mark.parametrize("form", ["numeric", "dated"])
FILL_KINDS = pytest.mark.parametrize("kind", ["tarn", "tarf"])


@FILL_FORMS
@FILL_KINDS
def test_fill_greeks_equal_the_contract_banked_at_the_base_spot(form, kind):
    product, e = _unbanked(form, kind), _fill_env(form)
    greeks, caught = _missing(_greeks, product, e)
    assert len(caught) == 1  # (b) once for the base valuation, never once per bump
    explicit, explicit_caught = _missing(_greeks, _banked_at(product, e), e)
    assert explicit_caught == []
    assert greeks == explicit  # (a) bitwise: the fixing is banked at the base rate under every bump


@FILL_FORMS
@pytest.mark.parametrize("spot", [1.0990, 1.0999, 1.1001, 1.1010])
def test_fill_near_the_strike_has_no_delta_jump(form, spot):
    # (c) one 0.1% spot bump crosses the 1.10 coupon strike: a fill that followed the bumped spot would flip the coupon
    product, e = _unbanked(form, "tarn"), _fill_env(form, spot)
    greeks, caught = _missing(_greeks, product, e)
    assert len(caught) == 1
    assert greeks == _greeks(_banked_at(product, e), e)


@FILL_FORMS
@FILL_KINDS
def test_freeze_records_the_base_rate_on_a_copy(form, kind):
    product, e = _unbanked(form, kind), _fill_env(form)
    snapshot = copy.deepcopy(product)
    assert product.missing_fixing_fills(e) == {0: 1.20}
    frozen, caught = _missing(freeze_missing_fixings, product, e)
    assert len(caught) == 1 and "#0" in str(caught[0].message)
    assert frozen is not product
    assert (product.state, product.lifecycle_events, product.missing_fixing_fills(e)) == (
        snapshot.state, snapshot.lifecycle_events, {0: 1.20})
    assert frozen.missing_fixing_fills(e) == {}
    again, caught = _missing(freeze_missing_fixings, frozen, _fill_env(form, 1.05))
    assert again is frozen and caught == []
    shocked = _fill_env(form, 1.05)  # history does not move under a shocked spot
    assert _price(frozen, e=shocked) == _price(_banked_at(product, e), e=shocked)
    assert frozen.with_fixings_recorded({0: 1.05}) is frozen  # a recorded rate is never overwritten


@FILL_FORMS
@FILL_KINDS
def test_nothing_unbanked_returns_the_product_itself(form, kind):
    product = _banked_at(_unbanked(form, kind), _fill_env(form))
    e = _fill_env(form)
    assert product.missing_fixing_fills(e) == {}
    frozen, caught = _missing(product.with_missing_fixings_filled, e)
    assert frozen is product and caught == []
    assert product.with_fixings_recorded({}) is product


@FILL_KINDS
def test_numeric_fills_two_passed_fixings_in_order(kind):
    product, e = _numeric_unbanked(kind, tau=0.2), env(spot=1.20)
    assert product.missing_fixing_fills(e) == {0: 1.20, 1: 1.20}
    frozen = product.with_missing_fixings_filled(e, warn=False)
    shocked = env(spot=1.05)
    assert _price(frozen, e=shocked) == _price(_banked_at(product, e), e=shocked)


def test_fill_that_redeems_lists_only_the_fixings_banked():
    product, e = _numeric_unbanked("tarn", tau=0.2, notional=100.0), env(spot=1.20)
    product.target = 0.5  # the first coupon (1.0) hits the target: fixing #1 is dead
    assert product.missing_fixing_fills(e) == {0: 1.20}
    frozen = product.with_missing_fixings_filled(e, warn=False)
    assert frozen.resolve(_fill_env("numeric", 1.05)).terminal


@FILL_FORMS
def test_carried_fills_bank_at_the_first_valuations_rate(form):
    # multi-day callers decide at the first valuation and carry {index: rate} to later contracts
    product, e0 = _unbanked(form, "tarn"), _fill_env(form)
    fills = product.missing_fixing_fills(e0)
    later = product.with_fixings_recorded(fills)
    e1 = _fill_env(form, 1.05)
    view, caught = _missing(later.resolve, e1)
    assert caught == []
    assert view == _banked_at(product, e0).resolve(e1)


@FILL_FORMS
def test_resolve_keeps_the_recorded_rate(form):
    frozen = _unbanked(form, "tarn").with_missing_fixings_filled(_fill_env(form), warn=False)
    snapshot = copy.deepcopy(frozen)
    frozen.resolve(_fill_env(form, 1.05))
    assert frozen.missing_fixing_fills(_fill_env(form)) == {}
    assert (frozen.state, frozen.lifecycle_events) == (snapshot.state, snapshot.lifecycle_events)
    assert _price(frozen, e=_fill_env(form, 1.05)) == _price(snapshot, e=_fill_env(form, 1.05))


@FILL_FORMS
def test_a_rate_recorded_for_a_future_fixing_is_refused(form):
    product = (note(target=3.0) if form == "numeric" else dated_note(target=3.0)).with_fixings_recorded({0: 1.2})
    with pytest.raises(ValidationError, match="future"):
        product.resolve(env())


@pytest.mark.parametrize("rate", [0.0, -1.2, float("nan"), float("inf")])
def test_a_recorded_rate_must_be_positive_and_finite(rate):
    with pytest.raises(ValidationError, match="rate"):
        _numeric_unbanked("tarn").with_fixings_recorded({0: rate})


# --- OQ-1 (b), user decision 2026-10-10: cash due today stays in today's value --------------------------------------

def test_numeric_cash_due_today_stays_pending_until_its_pay_time_is_past():
    n = note(fix=[10 / 365] + FIX[1:], pay_times=[12 / 365] + [t + 2 / 365 for t in FIX[1:]], target=3.0)
    c = n.full_coupon(0)
    n.time_shift(11 / 365)
    n.record_passed_fixings(env())
    assert [p.amount for p in n.state.pending] == [c]
    n.time_shift(1 / 365)                                        # its pay day: snapped to exactly 0.0, still pending
    assert n.state.pending == (PendingPayment(c, pay_time=0.0),)
    assert [e.event_type for e in n.lifecycle_events] == [FIXING]
    assert n.pending_pv(env()) == c
    n.time_shift(1 / 365)                                        # strictly past: paid
    assert n.state.pending == () and [e.event_type for e in n.lifecycle_events] == [FIXING, PAYMENT]


def test_a_zero_lag_fixing_banked_today_leaves_its_cash_in_the_value():
    n = note(fix=[0.0] + FIX[1:], accrual_start=-30 / 365, target=3.0)
    c = n.full_coupon(0)
    n.record_passed_fixings(env())
    assert [e.event_type for e in n.lifecycle_events] == [FIXING]
    assert n.state.pending == (PendingPayment(c, pay_time=0.0),)


@pytest.mark.parametrize("kind", ["tarn", "tarf"])
def test_value_on_a_pay_date_holds_the_cash_due_that_day(kind):
    d = dated_note(target=3.0) if kind == "tarn" else dated_tarf(target=0.20)    # pay dates = fixing dates
    e0 = env(val=FIX_DATES[0])                                                    # live fixing, its cash due today
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)                     # Task 9 makes a t = 0 fill silent
        view = d.resolve(e0)
        banked = copy.deepcopy(d)
        banked.record_passed_fixings(e0)
        (cash,) = [x.amount for x in banked.lifecycle_events if x.event_type is FIXING]
        assert view.pending == ((0.0, cash),) and banked.state.pending[0].pay_date == FIX_DATES[0]
        without = copy.deepcopy(banked)
        without.state = TargetRedemptionState(banked.state.accumulated, as_of=banked.state.as_of)
        assert is_close(_price(d, e=e0), _price(without, e=e0) + cash, rel_tol=1e-12)
        twin = _numeric_twin(d, e0)
        assert twin.state.pending == (PendingPayment(cash, pay_time=0.0),)
        assert is_close(_price(twin, e=e0), _price(d, e=e0), rel_tol=1e-12)


# Moved values (sign-off): measured on 51186894 (before Task 8a), MC 20_000 paths, seed 7, flat EURUSD 1.20; (price,
# theta) the day before the pay date and on it. After Task 8a: the day-before price is unchanged; the day-before theta
# gains the cash (the bump lands on the pay date, where the cash is still in the value); the pay-date price gains the
# cash; the pay-date theta loses it (the cash leaves the value the next day).
_PAY_DAY_OLD = {
    "tarn_zero_lag": ((102.09343055572586, -0.9778012736891526), (101.1156292820367, 0.015259847019819972)),
    "tarf_zero_lag": ((0.09637222423452786, -0.05217110639974772), (0.04420111783478014, 0.0008503526677482928)),
    "tarn_lag2_pending": ((102.08935210063005, -0.9702789813483861), (101.11907311928167, 0.01592892165280091)),
}


def _pay_day_case(name):
    if name == "tarn_zero_lag":
        return dated_note(target=3.0), FIX_DATES[0]
    if name == "tarf_zero_lag":
        return dated_tarf(target=0.20), FIX_DATES[0]
    lagged = dated_note(target=3.0, pay_dates=LAG2)
    lagged.record_passed_fixings(env(val=FIX_DATES[0]))
    return lagged, LAG2[0]


@pytest.mark.parametrize("name", sorted(_PAY_DAY_OLD))
def test_moved_values_around_a_pay_date(name):
    product, pay = _pay_day_case(name)
    probe = copy.deepcopy(product)
    probe.record_passed_fixings(env(val=pay))
    cash = probe.state.pending[0].amount                         # due on the pay date, still pending
    (p_before, th_before), (p_on, th_on) = _PAY_DAY_OLD[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        g_before = engine_for(product).calculate_greeks(product, env(val=pay - timedelta(days=1)))
        g_on = engine_for(product).calculate_greeks(product, env(val=pay))
    # pinned numbers were measured on ARM64; CI runs x86_64 (memory: golden cross-arch), hence 1e-9, not bitwise
    assert is_close(g_before["price"], p_before, rel_tol=1e-9)
    assert is_close(g_before["theta"], th_before + cash, rel_tol=0.0, abs_tol=1e-9)
    assert is_close(g_on["price"], p_on + cash, rel_tol=1e-9)
    assert is_close(g_on["theta"], th_on - cash, rel_tol=0.0, abs_tol=1e-9)


# --- numeric == dated, aged independently day by day (no _numeric_twin): a fixing day, its pay day, the days after --

_AGE_START = FIX_DATES[0] - timedelta(days=2)                       # Monday 2026-07-13
_AGE_DAYS = [FIX_DATES[0] + timedelta(days=k) for k in range(-1, 6)]  # Tue .. Mon 07-20: fixing Wed, pay Fri (LAG2)


def _independent_numeric(kind, e0, start=_AGE_START, fix_dates=FIX_DATES, pay_dates=LAG2, initial=VAL):
    """The dated contract's numeric twin written from its terms (times in the environment's day count from the
    booking date ``start``, TARN factors in ACT/365 from ``initial``), not from the dated resolve view."""
    yf = lambda a, b: calculate_year_fraction(a, b, e0.day_count_convention, e0.bus_days_in_year,  # noqa: E731
                                              calendar=e0.calendar)
    fix = [yf(start, d) for d in fix_dates]
    pays = [yf(start, d) for d in pay_dates]
    if kind == "tarf":
        return tarf(fix=fix, pay_times=pays, target=0.20)
    starts = [initial] + fix_dates[:-1]
    factors = [calculate_year_fraction(s, d, DayCountConvention.ACT_365, 252) for s, d in zip(starts, fix_dates)]
    return note(fix=fix, pay_times=pays, accrual_factors=factors, target=3.0)


def _event_parts(events):
    return [(e.event_type, e.amount, e.fixing, e.redeemed) for e in events]


@pytest.mark.parametrize("calendar", ["calendar", "business"])
@pytest.mark.parametrize("kind", ["tarn", "tarf"])
def test_independent_daily_twin_across_a_fixing_and_its_pay_day(kind, calendar):
    """Task 8a's pay-day twin builds its numeric contract from the dated resolve view; this one does not. Both forms
    are booked at inception on the Monday before the first fixing and aged one calendar day at a time through the
    fixing (Wednesday), its pay day (Friday, LAG2) and the days after (under BUSINESS_DAYS the Saturday and Sunday are
    at Friday's instant: the cash is still due, paid on the Monday). Events agree exactly (type, amount, rate,
    redeemed; times to ULPs) and so do prices."""
    kw = {} if calendar == "calendar" else BUS
    e0 = env(val=_AGE_START, **kw)
    dated = dated_note(target=3.0, pay_dates=LAG2) if kind == "tarn" else dated_tarf(target=0.20, pay_dates=LAG2)
    numeric = _independent_numeric(kind, e0)
    prev = _AGE_START
    for day in _AGE_DAYS:
        ek = env(val=day, spot=1.21, **kw)
        numeric.time_shift(calculate_year_fraction(prev, day, ek.day_count_convention, ek.bus_days_in_year,
                                                   calendar=ek.calendar))
        numeric.record_passed_fixings(ek)
        dated.record_passed_fixings(ek)
        prev = day
        assert _event_parts(numeric.lifecycle_events) == _event_parts(dated.lifecycle_events), day
        assert all(is_close(a.time, b.time, rel_tol=0.0, abs_tol=1e-12)
                   for a, b in zip(numeric.lifecycle_events, dated.lifecycle_events)), day
        assert numeric.state.accumulated == dated.state.accumulated
        assert [p.amount for p in numeric.state.pending] == [p.amount for p in dated.state.pending], day
        assert dated.state.as_of <= day
        assert is_close(_price(numeric, e=ek, n=5_000), _price(dated, e=ek, n=5_000), rel_tol=1e-12), day
    assert [e.event_type for e in dated.lifecycle_events] == [FIXING, PAYMENT]


_SAT_FIX, _SAT_PAY = [SATURDAY, datetime(2026, 9, 15)], [MONDAY, datetime(2026, 9, 17)]
_SAT_INITIAL = datetime(2026, 7, 15)


@pytest.mark.parametrize("kind", ["tarn", "tarf"])
def test_independent_daily_twin_across_a_saturday_fixing(kind):
    """OQ-2 (b) without _numeric_twin: under TARGET BUSINESS_DAYS a fixing on Saturday 08-15 (paid Monday 08-17) is
    at Friday's instant. Both forms are booked on the Thursday and aged one calendar day at a time to the Tuesday: the
    fixing banks on the Friday (at Friday's spot, as_of the Friday through the weekend), its cash is due on the Monday
    and paid on the Tuesday. Events and their times agree exactly every day, and so do prices."""
    e0 = env(val=THURSDAY, **BUS)
    if kind == "tarn":
        dated = dated_note(fixing_dates=_SAT_FIX, pay_dates=_SAT_PAY, initial_date=_SAT_INITIAL, target=3.0)
    else:
        dated = dated_tarf(fixing_dates=_SAT_FIX, pay_dates=_SAT_PAY, target=0.20)
    numeric = _independent_numeric(kind, e0, start=THURSDAY, fix_dates=_SAT_FIX, pay_dates=_SAT_PAY,
                                   initial=_SAT_INITIAL)
    prev, as_of = THURSDAY, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        for day in [THURSDAY + timedelta(days=k) for k in range(6)]:          # Thu .. Tue
            ek = env(val=day, spot=1.21, **BUS)
            if day > prev:                                                     # the Thursday is the booking
                numeric.time_shift(calculate_year_fraction(prev, day, DayCountConvention.BUSINESS_DAYS, 252,
                                                           calendar=TARGET_CAL))
            numeric.record_passed_fixings(ek)
            dated.record_passed_fixings(ek)
            prev = day
            assert _event_parts(numeric.lifecycle_events) == _event_parts(dated.lifecycle_events), day
            assert [e.time for e in numeric.lifecycle_events] == [e.time for e in dated.lifecycle_events], day
            assert numeric.state.accumulated == dated.state.accumulated
            assert [p.amount for p in numeric.state.pending] == [p.amount for p in dated.state.pending], day
            assert is_close(_price(numeric, e=ek, n=5_000), _price(dated, e=ek, n=5_000), rel_tol=1e-12), day
            as_of.append(dated.state.as_of)
    assert as_of == [THURSDAY, FRIDAY, FRIDAY, FRIDAY, MONDAY, MONDAY + timedelta(days=1)]
    assert [e.event_type for e in dated.lifecycle_events] == [FIXING, PAYMENT]
    assert dated.lifecycle_events[0].fixing == 1.21 and dated.state.fixings_banked == 1


# --- fix round 1: an explicit banked count may not claim a fixing ahead of valuation (no check dropped, OQ-3) -------

@pytest.mark.parametrize("call", ["price", "resolve", "record"])
@pytest.mark.parametrize("case", ["tarn_count_5_on_first_fixing", "tarf_count_3_no_as_of"])
def test_banked_count_ahead_of_valuation_is_refused(case, call):
    if case == "tarn_count_5_on_first_fixing":
        product = dated_note(target=3.0, state=TargetRedemptionState(1.0, as_of=FIX_DATES[0], fixings_banked=5))
        e = env(val=FIX_DATES[0])
    else:
        product = dated_tarf(target=0.20, state=TargetRedemptionState(0.0, fixings_banked=3))
        e = env()
    run = {"price": lambda: _price(product, e=e), "resolve": lambda: product.resolve(e),
           "record": lambda: product.record_passed_fixings(e)}[call]
    with pytest.raises(ValidationError, match="fixings_banked"):
        run()


def test_banked_count_equal_to_the_passed_fixings_is_accepted():
    booked = dated_note(target=3.0, state=TargetRedemptionState(1.0, as_of=FIX_DATES[1], fixings_banked=2))
    e = env(val=FIX_DATES[1])                                     # fixing 1 at t = 0: passed
    assert booked.resolve(e).fixing_times[0] > 0.0 and len(booked.resolve(e).fixing_times) == 10
    _price(booked, e=e)
    booked.record_passed_fixings(e)
    assert booked.state.fixings_banked == 2 and booked.lifecycle_events == ()


def test_a_saturday_fixing_banked_on_the_friday_stays_valid_under_business_days_only():
    """Decided by TIME (R16): the Saturday fixing banked on the Friday is at zero business time, so the booked count
    is valid under BUSINESS_DAYS; the same state priced under CALENDAR_DAYS on that Friday claims a fixing a day ahead
    and is refused (accepted side effect, controller ruling 2026-10-10)."""
    d = dated_note(fixing_dates=[FRIDAY, SATURDAY, datetime(2026, 9, 15)], initial_date=datetime(2026, 7, 15),
                   target=50.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingFixingWarning)
        d.record_passed_fixings(env(val=FRIDAY, **BUS))
    assert (d.state.fixings_banked, d.state.as_of) == (2, FRIDAY)
    rebooked = dated_note(fixing_dates=[FRIDAY, SATURDAY, datetime(2026, 9, 15)], initial_date=datetime(2026, 7, 15),
                          target=50.0, state=d.state)
    _price(rebooked, e=env(val=FRIDAY, **BUS))
    with pytest.raises(ValidationError, match="fixings_banked"):
        _price(rebooked, e=env(val=FRIDAY))


# --- OQ-4 (b), user decision 2026-10-10: periods may share a fixing date / time ------------------------------------

SHARED_FACTORS = [1 / 12, 1 / 24] + [1 / 12] * 11


def _shared_dated(kind, target, **kw):
    """Thirteen periods: the first two share FIX_DATES[0]; the second pays two days later."""
    dates = [FIX_DATES[0]] + FIX_DATES
    pays = [FIX_DATES[0], LAG2[0]] + FIX_DATES[1:]
    if kind == "tarf":
        return dated_tarf(fixing_dates=dates, pay_dates=pays, target=target, **kw)
    return dated_note(fixing_dates=dates, pay_dates=pays, accrual_factors=SHARED_FACTORS, target=target, **kw)


def test_numeric_form_accepts_equal_fixing_times():
    n = note(fix=[FIX[0], FIX[0]] + FIX[1:], pay_times=[FIX[0], FIX[0] + 2 / 365] + FIX[1:],
             accrual_factors=SHARED_FACTORS, target=50.0)
    assert n.fixing_times[:2] == [FIX[0], FIX[0]]
    with pytest.raises(ValidationError, match="ascending"):
        note(fix=[FIX[1], FIX[0]] + FIX[2:], target=50.0)              # decreasing is still refused


def test_the_default_derivation_cannot_share_a_fixing_date():
    """User ruling 2026-10-10 on R26 (b): a zero-length DERIVED period is refused; a stated accrual is not."""
    with pytest.raises(ValidationError, match="zero length"):
        dated_note(fixing_dates=[FIX_DATES[0]] + FIX_DATES, target=50.0)  # initial_date=VAL: a zero-length period
    assert _shared_dated("tarn", target=50.0).fixing_dates[:2] == [FIX_DATES[0]] * 2  # explicit factors: accepted


def test_engines_simulate_a_shared_time_once():
    e = env()
    eng = engine_for(note(target=50.0), n=1_000)
    t0, t1 = FIX[0], FIX[1]
    shared = eng._simulate(e, 1.20, 0.10, np.array([t0, t0, t1]))
    distinct = eng._simulate(e, 1.20, 0.10, np.array([t0, t1]))
    assert np.array_equal(shared[:, 0], shared[:, 1]) and np.array_equal(shared[:, [0, 2]], distinct)
    tarf_eng = engine_for(tarf(target=0.2), n=1_000)
    assert np.array_equal(tarf_eng._simulate(e, 1.20, 0.10, np.array([t0, t0, t1]))[:, [0, 2]],
                          tarf_eng._simulate(e, 1.20, 0.10, np.array([t0, t1])))


@pytest.mark.parametrize("kind", ["tarn", "tarf"])
def test_two_periods_on_one_fixing_date_price_like_the_twin_and_each_is_paid(kind):
    dated = _shared_dated(kind, target=50.0)
    twin = _numeric_twin(dated, env())
    assert twin.fixing_times[0] == twin.fixing_times[1]
    assert is_close(_price(twin), _price(dated), rel_tol=1e-12)
    e0, e3 = env(val=FIX_DATES[0]), env(val=FIX_DATES[0] + timedelta(days=3))
    twin.time_shift(calculate_year_fraction(VAL, FIX_DATES[0], DayCountConvention.CALENDAR_DAYS, 252))
    for product in (dated, twin):
        product.record_passed_fixings(e0)                              # the shared date: both periods bank together
    fixings = [x for x in dated.lifecycle_events if x.event_type is FIXING]
    assert len(fixings) == 2 and _kinds_only(dated.lifecycle_events) == _kinds_only(twin.lifecycle_events)
    assert fixings[0].fixing == fixings[1].fixing == 1.20                # one spot for both
    assert dated.state.fixings_banked == 2
    assert [p.pay_date for p in dated.state.pending] == [FIX_DATES[0], LAG2[0]]      # each its own pay date
    assert [p.amount for p in dated.state.pending] == [x.amount for x in fixings]    # each its own cash
    twin.time_shift(3 / 365)
    for product, e in ((dated, e3), (twin, e3)):
        product.record_passed_fixings(e)
    assert _kinds_only(dated.lifecycle_events) == _kinds_only(twin.lifecycle_events)
    assert [x.event_type for x in dated.lifecycle_events] == [FIXING, FIXING, PAYMENT, PAYMENT]   # banked once
    assert dated.state.fixings_banked == 2 and dated.state.pending == ()


def test_a_target_hit_by_the_first_of_two_same_date_periods_kills_the_second():
    """TARN spec section 6: "On a hit ... every later fixing is dead" (periods in schedule order)."""
    dated = _shared_dated("tarn", target=0.9)                           # the first coupon (1.0) is clipped to 0.9
    twin = _numeric_twin(dated, env())
    dated.record_passed_fixings(env(val=FIX_DATES[0]))
    (fixing,) = [x for x in dated.lifecycle_events if x.event_type is FIXING]  # the second period never banks
    assert fixing.amount == 0.9 and fixing.redeemed and dated.is_redeemed and dated.is_terminal
    assert [(p.amount, p.pay_date) for p in dated.state.pending] == [(0.9, FIX_DATES[0]), (100.0, FIX_DATES[0])]
    e = env(vol=1e-7)                                                   # the engine walk applies the same rule
    t = calculate_year_fraction(VAL, FIX_DATES[0], DayCountConvention.CALENDAR_DAYS, 252)
    expected = (0.9 + 100.0) * e.get_domestic_df(t)                     # nothing from the second period
    assert is_close(_price(twin, e=e), expected, rel_tol=1e-6)
    assert is_close(_price(_shared_dated("tarn", target=0.9), e=e), expected, rel_tol=1e-6)


def test_friday_and_saturday_fixings_price_on_the_thursday_like_the_twin():
    """OQ-3 case valued before the shared instant (was refused: OQ-4 status quo): both fixings at one business time,
    one simulated spot; dated == numeric."""
    dated = dated_note(fixing_dates=[FRIDAY, SATURDAY, datetime(2026, 9, 15)], initial_date=datetime(2026, 7, 15),
                       target=50.0)
    e_thu = env(val=THURSDAY, **BUS)
    twin = _numeric_twin(dated, e_thu)
    assert twin.fixing_times[0] == twin.fixing_times[1]
    assert is_close(_price(twin, e=e_thu), _price(dated, e=e_thu), rel_tol=1e-12)


# Prices at e2e07886 (the commit before this task; n=4_000, seed 7), frozen on the ARM64 dev machine: compared at
# GOLDEN_REL_TOL because a frozen literal crosses architectures (test/golden_compare.py). The same-machine invariant
# against the e2e07886 `_simulate` body below stays exact.
_E2E07886_PRICES = {
    ("numeric_tarn", "PSEUDO"): "0x1.969d22447738ap+6",
    ("numeric_tarn", "QUASI"): "0x1.96954aa30923dp+6",
    ("numeric_tarf", "PSEUDO"): "0x1.269000ddbd7f0p-4",
    ("numeric_tarf", "QUASI"): "0x1.190e35665df27p-4",
    ("dated_tarn", "PSEUDO"): "0x1.9669b703a8eecp+6",
    ("dated_tarn", "QUASI"): "0x1.965e327b383c3p+6",
    ("dated_tarf", "PSEUDO"): "0x1.25aac3449aeb8p-4",
    ("dated_tarf", "QUASI"): "0x1.183baf0447154p-4",
    ("dated_tarn_business_days", "PSEUDO"): "0x1.964ab26d97d67p+6",
    ("dated_tarn_business_days", "QUASI"): "0x1.963cb125622e9p+6",
}
_DISTINCT_CASES = {
    "numeric_tarn": lambda: (note(target=3.0), env()),
    "numeric_tarf": lambda: (tarf(target=0.2), env()),
    "dated_tarn": lambda: (dated_note(target=3.0, pay_dates=LAG2), env()),
    "dated_tarf": lambda: (dated_tarf(target=0.2, pay_dates=LAG2), env()),
    "dated_tarn_business_days": lambda: (dated_note(target=3.0), env(**BUS)),
}


def _e2e07886_simulate(self, fx_env, spot, sigma, fixings):
    """Both engines' `_simulate` body at e2e07886."""
    gen = self._make_generator(fx_env, spot, sigma, fixings)
    paths, _ = gen.generate_paths()
    return paths[:, 1:]


@pytest.mark.parametrize("method", [PSEUDO, QUASI])
@pytest.mark.parametrize("name", sorted(_DISTINCT_CASES))
def test_distinct_fixing_times_take_the_old_engine_path_bitwise(name, method, monkeypatch):
    """A schedule of distinct times is simulated exactly as at e2e07886: no reindexing (Task 8c)."""
    product, e = _DISTINCT_CASES[name]()
    eng = engine_for(product, n=4_000, method=method)
    times = np.asarray(product.resolve(e).fixing_times, dtype=float)
    live_spots = eng._simulate(e, 1.20, 0.10, times)
    live = _price(product, method=method, e=e, n=4_000)
    monkeypatch.setattr(type(eng), "_simulate", _e2e07886_simulate)
    assert np.array_equal(eng._simulate(e, 1.20, 0.10, times), live_spots)    # same machine: byte-identical
    assert _price(product, method=method, e=e, n=4_000) == live
    assert math.isclose(live, float.fromhex(_E2E07886_PRICES[name, method.name]), rel_tol=GOLDEN_REL_TOL)
