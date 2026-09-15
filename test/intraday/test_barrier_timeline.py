from datetime import datetime

import pytest

from quantark.asset.equity.lifecycle import BarrierLifecycleState, ValuationPoint
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
from quantark.execution.errors import CapabilityError
from quantark.intraday.context import resolve_context
from quantark.intraday.events import EventKind, resolve_timeline
from quantark.intraday.fixings import Fixing
from quantark.intraday.profile import VarianceProfile
from quantark.intraday.request import IntradayValuationRequest
from quantark.util.enum.option_enums import BarrierDirection, BarrierType, ObservationType, OptionType, TouchType
from intraday.conftest import SHANGHAI, _assert_unchanged, flat_env, snapshot

EXPIRY = datetime(2026, 12, 15)
TS = datetime(2026, 9, 15, 14, 0, tzinfo=SHANGHAI)


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def _uo(observation_type=ObservationType.CONTINUOUS, schedule=None, rebate=1.0, pay_at_hit=True, barrier_type=BarrierType.UP_OUT):
    return BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=110.0, barrier_type=barrier_type,
                         exercise_date=EXPIRY, initial_date=datetime(2026, 6, 15), rebate=rebate, pay_at_hit=pay_at_hit,
                         observation_type=observation_type, observation_schedule=schedule)


def _ctx(sessions, profile, product, spot=105.0, ts=TS, **kw):
    return resolve_context(IntradayValuationRequest(product=product, pricing_env=flat_env(ts, spot=spot),
                                                    session_calendar=sessions, variance_profile=profile, **kw))


def _schedule(dates, payoff=1.0):
    return ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=110.0, payoff=payoff) for d in dates])


def test_continuous_barrier_timeline_has_a_terminal_and_a_continuous_barrier(sse_sessions):
    tl = resolve_timeline(_uo(), sse_sessions, flat_env(datetime(2026, 9, 15, tzinfo=SHANGHAI)))
    assert [e.kind for e in tl.events] == [EventKind.TERMINAL]
    cb = tl.continuous_barrier
    assert cb.level == 110.0 and cb.is_up and cb.is_knock_out and cb.pays_at_hit and cb.hit_cash == 1.0
    assert tl.terminal().timestamp == datetime(2026, 12, 15, 15, 0, tzinfo=SHANGHAI)


def test_discrete_barrier_records_become_ko_events_paying_at_hit(sse_sessions):
    dates = [datetime(2026, 10, 15), datetime(2026, 11, 16), datetime(2026, 12, 15)]
    tl = resolve_timeline(_uo(ObservationType.DISCRETE, _schedule(dates, payoff=1.5)), sse_sessions,
                          flat_env(datetime(2026, 9, 15, tzinfo=SHANGHAI)))
    kos = [e for e in tl.events if e.kind is EventKind.KO]
    assert len(kos) == 3 and all(e.payment_timestamp == e.timestamp for e in kos) and kos[0].cash == 1.5
    assert tl.continuous_barrier is None
    at_hit_false = resolve_timeline(_uo(ObservationType.DISCRETE, _schedule(dates), pay_at_hit=False), sse_sessions,
                                    flat_env(datetime(2026, 9, 15, tzinfo=SHANGHAI)))
    assert all(e.payment_timestamp == at_hit_false.terminal().payment_timestamp
               for e in at_hit_false.events if e.kind is EventKind.KO)


def test_one_touch_and_no_touch_hit_cash(sse_sessions):
    def touch(kind):
        return OneTouchOption(barrier=110.0, barrier_direction=BarrierDirection.UP, exercise_date=EXPIRY, rebate=2.0,
                              touch_type=kind, observation_type=ObservationType.CONTINUOUS)
    env = flat_env(datetime(2026, 9, 15, tzinfo=SHANGHAI))
    assert resolve_timeline(touch(TouchType.ONE_TOUCH), sse_sessions, env).continuous_barrier.hit_cash == 2.0
    no_touch = resolve_timeline(touch(TouchType.NO_TOUCH), sse_sessions, env).continuous_barrier
    assert no_touch.hit_cash == 0.0 and no_touch.is_knock_out


def test_continuous_history_assumption_is_recorded_and_hit_uses_latest_spot(sse_sessions, desk):
    prod = _uo()
    before = snapshot(prod)
    ctx = _ctx(sse_sessions, desk, prod, spot=105.0)
    _assert_unchanged(before, prod)
    a = ctx.reconstruction.continuous_assumption
    assert ctx.provisional and a is not None and a.assumed_hit_at is None and a.uncovered_to == TS
    assert a.uncovered_from == datetime(2026, 6, 15, 15, 0, tzinfo=SHANGHAI)
    hit = _ctx(sse_sessions, desk, _uo(), spot=110.0)                      # inclusive: 110 >= 110
    assert hit.reconstruction.continuous_assumption.assumed_hit_at == TS and hit.numerical.terminated
    assert hit.numerical.paid_cash == pytest.approx(1.0)                 # rebate paid at the assumed hit
    later = _ctx(sse_sessions, desk, _uo(pay_at_hit=False), spot=110.0)
    assert later.numerical.paid_cash == 0.0 and later.numerical.pending_cashflows[0][1] == pytest.approx(1.0)


def test_authoritative_state_supersedes_the_scenario(sse_sessions, desk):
    cp = BarrierLifecycleState(alive=True, valuation_point=ValuationPoint(date=datetime(2026, 9, 15)))
    ctx = _ctx(sse_sessions, desk, _uo(), spot=105.0, lifecycle_state=cp)
    assert not ctx.provisional and ctx.reconstruction.continuous_assumption is None


def test_knocked_in_history_yields_a_vanilla_twin(sse_sessions, desk):
    ui = _uo(barrier_type=BarrierType.UP_IN, rebate=0.0)
    cp = BarrierLifecycleState(alive=True, knocked_in=True, hit_date=datetime(2026, 9, 1),
                               valuation_point=ValuationPoint(date=datetime(2026, 9, 15)))
    ctx = _ctx(sse_sessions, desk, ui, spot=100.0, lifecycle_state=cp)
    twin = ctx.numerical.product
    assert isinstance(twin, EuropeanVanillaOption) and twin.maturity == ctx.numerical.maturity_tau and twin.exercise_date is None


def test_alive_discrete_twin_is_float_time_with_contractual_rebates(sse_sessions, desk):
    dates = [datetime(2026, 10, 15), datetime(2026, 11, 16), datetime(2026, 12, 15)]
    ctx = _ctx(sse_sessions, desk, _uo(ObservationType.DISCRETE, _schedule(dates, payoff=1.5)), spot=105.0)
    twin = ctx.numerical.product
    recs = twin.observation_schedule.records
    assert twin.exercise_date is None and twin.initial_date is None and len(recs) == 3
    kos = [e for e in ctx.numerical.remaining_events if e.kind is EventKind.KO]
    assert [r.observation_time for r in recs] == [ctx.numerical.event_taus[e.event_id] for e in kos]
    assert [r.payoff for r in recs] == [1.5, 1.5, 1.5] and not ctx.provisional


def test_discrete_barrier_replay_knocks_out_with_contractual_cash(sse_sessions, desk):
    dates = [datetime(2026, 7, 15), datetime(2026, 8, 17), datetime(2026, 12, 15)]
    prod = _uo(ObservationType.DISCRETE, _schedule(dates, payoff=1.0))
    probe = resolve_timeline(prod, sse_sessions, flat_env(TS))
    kos = [e for e in probe.events if e.kind is EventKind.KO]
    confirmed = _ctx(sse_sessions, desk, prod, spot=105.0, fixings=(Fixing(kos[0].timestamp, 100.0), Fixing(kos[1].timestamp, 111.0)))
    assert confirmed.numerical.terminated and not confirmed.provisional and confirmed.numerical.paid_cash == pytest.approx(1.0)
    assumed = _ctx(sse_sessions, desk, prod, spot=105.0)
    assert assumed.provisional and len(assumed.reconstruction.assumptions) == 2 and not assumed.numerical.terminated


def test_multiplier_mismatch_between_tracker_and_contract_fails_closed(sse_sessions, desk):
    dates = [datetime(2026, 7, 15), datetime(2026, 8, 17), datetime(2026, 12, 15)]
    prod = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=110.0, barrier_type=BarrierType.UP_OUT,
                         exercise_date=EXPIRY, initial_date=datetime(2026, 6, 15), rebate=1.0, pay_at_hit=True,
                         observation_type=ObservationType.DISCRETE, observation_schedule=ObservationSchedule(
                             records=[ObservationRecord(observation_date=d, barrier=110.0) for d in dates]),
                         contract_multiplier=100.0)
    kos = [e for e in resolve_timeline(prod, sse_sessions, flat_env(TS)).events if e.kind is EventKind.KO]
    with pytest.raises(CapabilityError, match="cash"):
        _ctx(sse_sessions, desk, prod, spot=105.0, fixings=(Fixing(kos[0].timestamp, 111.0),))
