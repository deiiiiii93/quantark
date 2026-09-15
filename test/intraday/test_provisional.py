from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.lifecycle import AutocallableLifecycleState, ValuationPoint
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


def test_product_without_lifecycle_reconstructs_to_nothing(sse_calendar, sse_sessions):
    prod = digital(datetime(2026, 12, 15))
    env = flat_env(datetime(2026, 9, 15, 14, 0, tzinfo=SHANGHAI))
    tl = resolve_timeline(prod, sse_sessions, env)
    ts = env.valuation_date
    rec = _run(prod, tl, None, [], ts, EventPhase.BEFORE, 100.0, ts, _schedule_env(prod, env), sse_sessions)
    assert rec.state is None and not rec.provisional and rec.applied_event_ids == ()
    with pytest.raises(ValidationError, match="lifecycle"):
        _run(prod, tl, AutocallableLifecycleState(), [], ts, EventPhase.BEFORE, 100.0, ts, _schedule_env(prod, env), sse_sessions)


def test_phoenix_coupon_replay_fails_closed(sse_calendar, sse_sessions):
    prod, env, tl, senv = _setup(sse_calendar, sse_sessions, prod=dated_phoenix(sse_calendar, T0))
    ts = _ko(tl, 0).timestamp + timedelta(seconds=30)
    with pytest.raises(CapabilityError, match="coupon"):
        _run(prod, tl, None, [Fixing(_ko(tl, 0).timestamp, 100.0)], ts, EventPhase.AFTER, 100.0, ts, senv, sse_sessions)
    before_first = _ko(tl, 0).timestamp - timedelta(hours=1)          # nothing due yet: prices fine
    rec = _run(prod, tl, None, [], before_first, EventPhase.BEFORE, 100.0, before_first, senv, sse_sessions)
    assert rec.state.alive and rec.applied_event_ids == ()


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
