from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.lifecycle import TerminalLifecycleState, AutocallableLifecycleState, ValuationPoint
from quantark.execution.errors import CapabilityError
from quantark.intraday.events import EventKind, EventPhase, _schedule_env, resolve_timeline
from quantark.intraday.fixings import Fixing
from quantark.intraday.provisional import lifecycle_state_fingerprint, reconstruct_lifecycle
from quantark.util.enum.option_enums import ObservationType
from quantark.util.exceptions import ValidationError
from intraday.conftest import SHANGHAI, _assert_unchanged, dated_phoenix, dated_snowball, digital, flat_env, snapshot

T0 = datetime(2026, 3, 16)


def _setup(sse_calendar, sse_sessions, months=12, prod=None):
    prod = prod if prod is not None else dated_snowball(sse_calendar, T0, months=months)
    env = flat_env(datetime(2026, 9, 15, 15, 0, 30, tzinfo=SHANGHAI))
    tl = resolve_timeline(prod, sse_sessions, env)
    return prod, env, tl, _schedule_env(prod, env)


def _ko(tl, i):
    return [e for e in tl.events if e.kind is EventKind.KO][i]


def _run(prod, tl, cp, fixings, ts, phase, spot, spot_ts, senv, sessions):
    return reconstruct_lifecycle(prod, tl, cp, fixings, valuation_timestamp=ts, phase=phase, spot=spot,
                                 spot_timestamp=spot_ts, schedule_env=senv, session_calendar=sessions)


def test_missing_fixing_becomes_a_flagged_assumption_using_the_spot(sse_calendar, sse_sessions):
    prod, env, tl, senv = _setup(sse_calendar, sse_sessions)
    fix5 = _ko(tl, 5).timestamp                                   # Sept fixing at 15:00
    ts = fix5 + timedelta(seconds=30)
    fixings = [Fixing(_ko(tl, i).timestamp, 100.0) for i in range(5)]   # first five confirmed below KO
    before = snapshot(prod)
    rec = _run(prod, tl, None, fixings, ts, EventPhase.AFTER, 104.0, ts, senv, sse_sessions)
    _assert_unchanged(before, prod)
    assert rec.provisional and len(rec.assumptions) == 1
    a = rec.assumptions[0]
    assert a.scheduled_at == fix5 and a.assumed_value == 104.0 and a.spot_timestamp == ts
    assert set(a.event_ids) == {e.event_id for e in tl.at(fix5)}
    assert rec.state.knocked_out and not rec.state.alive
    assert len(rec.confirmed_event_ids) == 10                     # 5 KO + 5 KI instants confirmed


def test_actual_fixing_replaces_the_assumption_and_the_ko_disappears(sse_calendar, sse_sessions):
    prod, env, tl, senv = _setup(sse_calendar, sse_sessions)
    fix5 = _ko(tl, 5).timestamp
    ts = fix5 + timedelta(seconds=30)
    fixings = [Fixing(_ko(tl, i).timestamp, 100.0) for i in range(5)] + [Fixing(fix5, 102.9)]
    rec = _run(prod, tl, None, fixings, ts, EventPhase.AFTER, 104.0, ts, senv, sse_sessions)
    assert not rec.provisional and rec.assumptions == ()
    assert rec.state.alive and not rec.state.knocked_out


def test_before_phase_at_the_fixing_leaves_it_undetermined(sse_calendar, sse_sessions):
    prod, env, tl, senv = _setup(sse_calendar, sse_sessions)
    fix5 = _ko(tl, 5).timestamp
    fixings = [Fixing(_ko(tl, i).timestamp, 100.0) for i in range(5)]
    rec = _run(prod, tl, None, fixings, fix5, EventPhase.BEFORE, 104.0, fix5, senv, sse_sessions)
    assert not rec.provisional and rec.state.alive
    with pytest.raises(ValidationError, match="not (yet )?determined|not due"):
        _run(prod, tl, None, fixings + [Fixing(fix5, 104.0)], fix5, EventPhase.BEFORE, 104.0, fix5, senv, sse_sessions)


def test_several_missed_fixings_are_several_assumptions_in_order(sse_calendar, sse_sessions):
    prod, env, tl, senv = _setup(sse_calendar, sse_sessions)
    ts = _ko(tl, 5).timestamp + timedelta(seconds=30)
    rec = _run(prod, tl, None, [], ts, EventPhase.AFTER, 70.0, ts, senv, sse_sessions)
    assert len(rec.assumptions) == 6 and [a.scheduled_at for a in rec.assumptions] == [_ko(tl, i).timestamp for i in range(6)]
    assert rec.state.knocked_in and rec.state.alive           # spot 70 < KI 75 assumed at every missed fixing


def test_checkpoint_is_authoritative_and_fixings_before_it_are_rejected(sse_calendar, sse_sessions):
    prod, env, tl, senv = _setup(sse_calendar, sse_sessions)
    cp_day = _ko(tl, 3).timestamp.replace(tzinfo=None).replace(hour=0, minute=0)
    cp = AutocallableLifecycleState(knocked_in=True, valuation_point=ValuationPoint(date=cp_day))
    cp_before = snapshot(cp)
    ts = _ko(tl, 5).timestamp + timedelta(seconds=30)
    rec = _run(prod, tl, cp, [Fixing(_ko(tl, 4).timestamp, 100.0), Fixing(_ko(tl, 5).timestamp, 100.0)],
               ts, EventPhase.AFTER, 100.0, ts, senv, sse_sessions)
    _assert_unchanged(cp_before, cp)                              # never mutated
    assert rec.state.knocked_in and rec.state is not cp and not rec.provisional
    with pytest.raises(ValidationError, match="checkpoint"):
        _run(prod, tl, cp, [Fixing(_ko(tl, 2).timestamp, 100.0)], ts, EventPhase.AFTER, 100.0, ts, senv, sse_sessions)


def test_events_covered_by_a_checkpoint_are_never_replayed_at_a_later_spot(sse_calendar, sse_sessions):
    # A hand-built checkpoint has no observed indices; the tracker would otherwise re-observe
    # fixings 0-3 at fixing 4's value and knock out at index 0 with index 0's cash.
    prod, env, tl, senv = _setup(sse_calendar, sse_sessions)
    cp_day = _ko(tl, 3).timestamp.replace(tzinfo=None).replace(hour=0, minute=0)
    cp = AutocallableLifecycleState(valuation_point=ValuationPoint(date=cp_day))
    ts = _ko(tl, 4).timestamp + timedelta(seconds=30)
    rec = _run(prod, tl, cp, [Fixing(_ko(tl, 4).timestamp, 104.0)], ts, EventPhase.AFTER, 104.0, ts, senv, sse_sessions)
    assert rec.state.knocked_out
    assert [cf.amount for cf in rec.state.ledger.cashflows] == [pytest.approx(_ko(tl, 4).cash)]
    assert rec.state.ledger.cashflows[0].cashflow_id == "knock-out:4"


def test_unknown_instant_and_missing_spot_timestamp_are_errors(sse_calendar, sse_sessions):
    prod, env, tl, senv = _setup(sse_calendar, sse_sessions)
    ts = _ko(tl, 5).timestamp + timedelta(seconds=30)
    with pytest.raises(ValidationError, match="matches no contract event"):
        _run(prod, tl, None, [Fixing(ts - timedelta(hours=1), 100.0)], ts, EventPhase.AFTER, 100.0, ts, senv, sse_sessions)
    with pytest.raises(ValidationError, match="spot timestamp"):
        _run(prod, tl, None, [], ts, EventPhase.AFTER, 100.0, None, senv, sse_sessions)


def test_terminal_fixing_settles_maturity(sse_calendar, sse_sessions):
    prod, env, tl, senv = _setup(sse_calendar, sse_sessions)
    ts = tl.terminal().timestamp + timedelta(seconds=30)
    fixings = [Fixing(_ko(tl, i).timestamp, 100.0) for i in range(12)]   # never KO, never KI
    rec = _run(prod, tl, None, fixings, ts, EventPhase.AFTER, 100.0, ts, senv, sse_sessions)
    assert rec.state.matured and not rec.state.alive and not rec.provisional
    assert len(rec.state.ledger.cashflows) >= 1


def test_terminal_only_product_is_alive_with_an_empty_ledger_before_its_fixing(sse_calendar, sse_sessions):
    prod = digital(datetime(2026, 12, 15))
    env = flat_env(datetime(2026, 9, 15, 14, 0, tzinfo=SHANGHAI))
    tl = resolve_timeline(prod, sse_sessions, env)
    ts = env.valuation_date
    rec = _run(prod, tl, None, [], ts, EventPhase.BEFORE, 100.0, ts, _schedule_env(prod, env), sse_sessions)
    assert isinstance(rec.state, TerminalLifecycleState) and rec.state.alive and not rec.state.expired
    assert not rec.provisional and rec.applied_event_ids == () and rec.state.ledger.cashflows == ()
    with pytest.raises(ValidationError, match="TerminalLifecycleState"):
        _run(prod, tl, AutocallableLifecycleState(), [], ts, EventPhase.BEFORE, 100.0, ts, _schedule_env(prod, env), sse_sessions)


def test_terminal_fixing_becomes_a_fixed_receivable(sse_calendar, sse_sessions):
    """Review 2026-09-16 finding 11: after its fixing a digital still owns a pending claim."""
    expiry = datetime(2026, 9, 15)
    prod = digital(expiry)
    prod.settlement_date = expiry + timedelta(days=5)
    ts = expiry.replace(hour=15, minute=1, tzinfo=SHANGHAI)
    env = flat_env(ts)
    tl = resolve_timeline(prod, sse_sessions, env)
    fixing = Fixing(expiry.replace(hour=15, tzinfo=SHANGHAI), 101.0)
    rec = _run(prod, tl, None, [fixing], ts, EventPhase.AFTER, 100.0, ts, _schedule_env(prod, env), sse_sessions)
    assert rec.state.expired and not rec.state.alive and not rec.provisional
    (flow,) = rec.state.ledger.cashflows
    assert flow.amount == 1.0 and flow.payment_date.date() == (expiry + timedelta(days=5)).date()
    # no supplied fixing: the latest spot is assumed and the receivable is provisional
    assumed = _run(prod, tl, None, [], ts, EventPhase.AFTER, 100.0, ts, _schedule_env(prod, env), sse_sessions)
    assert assumed.provisional and assumed.state.ledger.cashflows[0].amount == 0.0   # 100 is not > strike 100


def test_phoenix_coupon_replay_books_the_contractual_amount(sse_calendar, sse_sessions):
    """The tracker books what the contract event says, which is what the engines pay for that period."""
    prod, env, tl, senv = _setup(sse_calendar, sse_sessions, prod=dated_phoenix(sse_calendar, T0))
    ts = _ko(tl, 0).timestamp + timedelta(seconds=30)
    rec = _run(prod, tl, None, [Fixing(_ko(tl, 0).timestamp, 100.0)], ts, EventPhase.AFTER, 100.0, ts, senv, sse_sessions)
    coupon = next(e for e in tl.events if e.kind is EventKind.COUPON and e.index == 0)
    assert rec.state.alive and not rec.state.knocked_out            # spot 100: coupon barrier 80, KO barrier 103
    assert [cf.cashflow_id for cf in rec.state.ledger.cashflows] == ["coupon:0"]
    assert rec.state.ledger.cashflows[0].amount == pytest.approx(coupon.cash, rel=1e-12)
    unit = prod.initial_price * prod.contract_multiplier
    assert coupon.cash < unit * prod.coupon_config.coupon_rate      # an annualized rate accrues over one period
    assert rec.state.coupon_memory_count == 0 and not rec.state.missed_coupon_indices


def test_a_memorized_phoenix_coupon_is_released_with_the_period_that_triggers(sse_calendar, sse_sessions):
    prod, env, tl, senv = _setup(sse_calendar, sse_sessions, prod=dated_phoenix(sse_calendar, T0))
    ts = _ko(tl, 1).timestamp + timedelta(seconds=30)
    fixings = [Fixing(_ko(tl, 0).timestamp, 70.0), Fixing(_ko(tl, 1).timestamp, 100.0)]
    rec = _run(prod, tl, None, fixings, ts, EventPhase.AFTER, 100.0, ts, senv, sse_sessions)
    paid = rec.state.ledger.cashflows[-1]
    c0 = next(e for e in tl.events if e.kind is EventKind.COUPON and e.index == 0)
    c1 = next(e for e in tl.events if e.kind is EventKind.COUPON and e.index == 1)
    assert rec.state.knocked_in                                     # spot 70 is below the KI barrier
    assert paid.amount == pytest.approx(c0.cash + c1.cash, rel=1e-12), "the memorized period is released with it"


def test_coupon_memory_outstanding_at_the_valuation_instant_fails_closed(sse_calendar, sse_sessions):
    """Memory reaches the twin as a COUNT: without equal periods it cannot say what the arrears are worth."""
    from quantark.intraday import VarianceProfile
    from quantark.intraday.context import resolve_context
    from quantark.intraday.request import IntradayValuationRequest

    prod = dated_phoenix(sse_calendar, T0)
    _, env, tl, _ = _setup(sse_calendar, sse_sessions, prod=prod)
    ts = _ko(tl, 0).timestamp + timedelta(seconds=30)
    request = IntradayValuationRequest(
        product=prod, pricing_env=flat_env(ts, spot=70.0), session_calendar=sse_sessions,
        variance_profile=VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,)),
        fixings=(Fixing(_ko(tl, 0).timestamp, 70.0),))
    with pytest.raises(CapabilityError, match="fixed_coupon_year_fraction"):
        resolve_context(request)


def test_continuous_ki_history_is_a_disclosed_assumption(sse_calendar, sse_sessions):
    prod = dated_snowball(sse_calendar, T0)
    prod.barrier_config = replace(prod.barrier_config, ki_observation_type=ObservationType.CONTINUOUS, ki_observation_schedule=None)
    _, env, tl, senv = _setup(sse_calendar, sse_sessions, prod=prod)
    ts = _ko(tl, 0).timestamp - timedelta(hours=2)
    safe = _run(prod, tl, None, [], ts, EventPhase.BEFORE, 90.0, ts, senv, sse_sessions)
    a = safe.continuous_assumption
    assert safe.provisional and a.uncovered_from == tl.initial_timestamp and a.uncovered_to == ts and a.assumed_hit_at is None
    assert not safe.state.knocked_in
    hit = _run(prod, tl, None, [], ts, EventPhase.BEFORE, 75.0, ts, senv, sse_sessions)      # inclusive: 75 <= 75
    assert hit.continuous_assumption.assumed_hit_at == ts and hit.state.knocked_in and hit.state.alive


def test_checkpoint_fingerprint_distinguishes_states():
    a = AutocallableLifecycleState(valuation_point=ValuationPoint(date=datetime(2026, 9, 1)))
    b = AutocallableLifecycleState(knocked_in=True, valuation_point=ValuationPoint(date=datetime(2026, 9, 1)))
    assert lifecycle_state_fingerprint(a) != lifecycle_state_fingerprint(b)
    assert lifecycle_state_fingerprint(a) == lifecycle_state_fingerprint(AutocallableLifecycleState(
        valuation_point=ValuationPoint(date=datetime(2026, 9, 1))))
