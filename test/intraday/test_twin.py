from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.settlement_support import pending_receivable_pv
from quantark.intraday.events import EventKind, EventPhase, _schedule_env, resolve_timeline
from quantark.intraday.fixings import Fixing
from quantark.intraday.provisional import LifecycleReconstruction, reconstruct_lifecycle
from quantark.intraday.timestamp import SECONDS_PER_YEAR, calendar_year_fraction
from quantark.intraday.twin import build_numerical_contract
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError
from intraday.conftest import SHANGHAI, _assert_unchanged, dated_phoenix, dated_snowball, digital, flat_env, snapshot

T0 = datetime(2026, 3, 16)


def _kos(tl):
    return [e for e in tl.events if e.kind is EventKind.KO]


def _base_timeline(sse_calendar, sse_sessions, prod=None):
    prod = prod if prod is not None else dated_snowball(sse_calendar, T0)
    return resolve_timeline(prod, sse_sessions, flat_env(datetime(2026, 9, 15, tzinfo=SHANGHAI)))


def _twin(sse_sessions, prod, ts, phase=EventPhase.AFTER, fixings=(), spot=100.0, checkpoint=None):
    env = flat_env(ts, spot=spot)
    tl = resolve_timeline(prod, sse_sessions, env)
    rec = reconstruct_lifecycle(prod, tl, checkpoint, list(fixings), valuation_timestamp=ts, phase=phase, spot=spot,
                                spot_timestamp=ts, schedule_env=_schedule_env(prod, env), session_calendar=sse_sessions)
    return tl, rec, build_numerical_contract(prod, tl, rec, valuation_timestamp=ts, phase=phase, session_calendar=sse_sessions)


def test_twin_times_are_seconds_exact_and_cash_is_contractual(sse_calendar, sse_sessions):
    kos0 = _kos(_base_timeline(sse_calendar, sse_sessions))
    ts = kos0[5].timestamp - timedelta(seconds=1)                    # the sixth fixing is one second away
    fixings = [Fixing(k.timestamp, 100.0) for k in kos0[:5]]
    prod = dated_snowball(sse_calendar, T0)
    before = snapshot(prod)
    tl, rec, num = _twin(sse_sessions, prod, ts, fixings=fixings)
    _assert_unchanged(before, prod)
    twin = num.product
    assert twin is not prod and twin.exercise_date is None and twin.initial_date is None
    sched = twin.barrier_config.ko_observation_schedule.records
    assert len(sched) == 7                                          # 12 - 5 confirmed
    assert sched[0].observation_time == 1.0 / SECONDS_PER_YEAR
    assert twin.maturity == calendar_year_fraction(ts, tl.terminal().timestamp)
    assert twin.tenor == tl.contractual_tenor
    naive_env = PricingEnvironment(rate_curve=FlatRateCurve(0.03), valuation_date=ts, spot_quote=SpotQuote(100.0),
                                   vol_surface=FlatVolSurface(0.2), div_yield=ContinuousDividendYield(0.01))
    resolved = twin.resolve_ko_observations(naive_env)
    remaining_kos = [e for e in num.remaining_events if e.kind is EventKind.KO]
    assert len(resolved) == len(remaining_kos) == 7
    for r, e in zip(resolved, remaining_kos):
        assert r.payoff == pytest.approx(e.cash, rel=1e-12) and r.observation_time == num.event_taus[e.event_id]
        assert r.settlement_time == pytest.approx(calendar_year_fraction(ts, e.payment_timestamp), abs=1e-15)
    assert twin.get_contract_tenor(naive_env) == tl.contractual_tenor
    assert not num.knocked_in and not num.terminated and num.paid_cash == 0.0 and num.pending_cashflows == ()


def test_after_phase_drops_the_fixing_at_valuation_and_before_keeps_it_at_tau_zero(sse_calendar, sse_sessions):
    tl0 = _base_timeline(sse_calendar, sse_sessions)
    fix = _kos(tl0)[5].timestamp
    fixings = [Fixing(k.timestamp, 100.0) for k in _kos(tl0)[:5]]
    _, _, before = _twin(sse_sessions, dated_snowball(sse_calendar, T0), fix, EventPhase.BEFORE, fixings)
    _, _, after = _twin(sse_sessions, dated_snowball(sse_calendar, T0), fix, EventPhase.AFTER, fixings + [Fixing(fix, 100.0)])
    assert before.product.barrier_config.ko_observation_schedule.records[0].observation_time == 0.0
    assert after.product.barrier_config.ko_observation_schedule.records[0].observation_time > 0.0
    assert len(before.remaining_events) == len(after.remaining_events) + len(tl0.at(fix))    # KI[5] and KO[5]
    assert min(after.event_taus.values()) > 0.0


def test_assumed_ko_gives_a_terminated_contract_with_paid_cash(sse_calendar, sse_sessions):
    tl0 = _base_timeline(sse_calendar, sse_sessions)
    fix = _kos(tl0)[5]
    ts = fix.timestamp + timedelta(seconds=30)
    fixings = [Fixing(k.timestamp, 100.0) for k in _kos(tl0)[:5]]
    tl, rec, num = _twin(sse_sessions, dated_snowball(sse_calendar, T0), ts, EventPhase.AFTER, fixings, spot=104.0)
    assert rec.provisional and num.terminated and not num.lifecycle_state.alive and num.product is None
    (flow,) = num.lifecycle_state.ledger.cashflows
    assert flow.amount == pytest.approx(fix.cash, rel=1e-12)
    assert flow.payment_time == pytest.approx(-30.0 / SECONDS_PER_YEAR, abs=1e-15)      # paid at 15:00, 30 s ago
    assert num.pending_cashflows == () and num.paid_cash == pytest.approx(fix.cash, rel=1e-12)


def test_ledger_is_time_based_and_engine_readable(sse_calendar, sse_sessions):
    prod0 = dated_snowball(sse_calendar, T0)
    for r in prod0.barrier_config.ko_observation_schedule.records:
        r.settlement_date = r.observation_date + timedelta(days=2)          # T+2 payment
    tl0 = resolve_timeline(prod0, sse_sessions, flat_env(datetime(2026, 9, 15, tzinfo=SHANGHAI)))
    fix = _kos(tl0)[5].timestamp
    ts = fix + timedelta(seconds=30)
    env = flat_env(ts, spot=104.0)
    fixings = [Fixing(k.timestamp, 100.0) for k in _kos(tl0)[:5]]            # alive until the assumed sixth fixing
    rec = reconstruct_lifecycle(prod0, tl0, None, fixings, valuation_timestamp=ts, phase=EventPhase.AFTER, spot=104.0,
                                spot_timestamp=ts, schedule_env=_schedule_env(prod0, env), session_calendar=sse_sessions)
    num = build_numerical_contract(prod0, tl0, rec, valuation_timestamp=ts, phase=EventPhase.AFTER, session_calendar=sse_sessions)
    st = num.lifecycle_state
    assert st.valuation_point.time == 0.0
    assert all(c.payment_time is not None and c.payment_date is None for c in st.ledger.cashflows)
    (cid, amount, pay_tau), = num.pending_cashflows
    assert pay_tau == pytest.approx(calendar_year_fraction(ts, sse_sessions.payment_at((fix + timedelta(days=2)).date())), abs=1e-15)
    pv = pending_receivable_pv(st, env)
    assert pv == pytest.approx(amount * env.get_discount_factor(pay_tau), rel=1e-12)


def test_phoenix_twin_reproduces_contractual_coupons(sse_calendar, sse_sessions):
    prod = dated_phoenix(sse_calendar, T0)
    tl0 = _base_timeline(sse_calendar, sse_sessions, prod=prod)
    ts = _kos(tl0)[0].timestamp - timedelta(hours=3)
    tl, rec, num = _twin(sse_sessions, prod, ts, EventPhase.BEFORE)
    twin = num.product
    assert twin.accrual_config.accrual_factors is not None
    from quantark.intraday.events import phoenix_coupon_fractions
    times = [r.observation_time for r in twin.barrier_config.ko_observation_schedule.records]
    fractions = phoenix_coupon_fractions(twin, times)
    coupons = [e for e in num.remaining_events if e.kind is EventKind.COUPON]
    assert [twin.get_coupon_payoff(i, year_fraction=f) for i, f in enumerate(fractions)] == pytest.approx([c.cash for c in coupons], rel=1e-12)


def test_a_per_period_coupon_rate_accrues_no_year_fraction(sse_calendar, sse_sessions):
    """is_annualized_coupon=False: the rate IS the period amount, in the intraday inventory too."""
    from dataclasses import replace

    from quantark.intraday.events import phoenix_coupon_fractions

    prod = dated_phoenix(sse_calendar, T0)
    prod.accrual_config = replace(prod.accrual_config, is_annualized_coupon=False)
    tl0 = _base_timeline(sse_calendar, sse_sessions, prod=prod)
    ts = _kos(tl0)[0].timestamp - timedelta(hours=3)
    _, _, num = _twin(sse_sessions, prod, ts, EventPhase.BEFORE)
    twin = num.product
    times = [r.observation_time for r in twin.barrier_config.ko_observation_schedule.records]
    assert phoenix_coupon_fractions(twin, times) == [1.0] * len(times)
    unit = twin.initial_price * twin.contract_multiplier
    coupons = [e for e in num.remaining_events if e.kind is EventKind.COUPON]
    assert [c.cash for c in coupons] == pytest.approx([unit * twin.coupon_config.coupon_rate] * len(coupons), rel=1e-12)


def test_digital_twin(sse_calendar, sse_sessions):
    ts = datetime(2026, 9, 15, 14, 59, 59, tzinfo=SHANGHAI)
    prod = digital(expiry=datetime(2026, 9, 15))
    tl = resolve_timeline(prod, sse_sessions, flat_env(ts))
    rec = LifecycleReconstruction(state=None, applied_event_ids=(), confirmed_event_ids=(), assumptions=(),
                                  continuous_assumption=None, checkpoint_fingerprint=None)
    num = build_numerical_contract(prod, tl, rec, valuation_timestamp=ts, phase=EventPhase.BEFORE, session_calendar=sse_sessions)
    assert num.product.maturity == 1.0 / SECONDS_PER_YEAR and num.product.exercise_date is None and num.lifecycle_state is None
    after_expiry = ts + timedelta(seconds=5)
    with pytest.raises(ValidationError, match="terminal"):
        build_numerical_contract(prod, tl, rec, valuation_timestamp=after_expiry, phase=EventPhase.AFTER, session_calendar=sse_sessions)
