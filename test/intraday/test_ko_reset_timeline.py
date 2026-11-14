from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.quad.v2 import KOResetSnowballQuadEngineV2
from quantark.asset.equity.lifecycle import AutocallableLifecycleState, ValuationPoint
from quantark.execution.errors import CapabilityError
from quantark.intraday import value_intraday
from quantark.intraday.context import resolve_context
from quantark.intraday.engines import route_for
from quantark.intraday.events import EventKind, resolve_timeline
from quantark.intraday.profile import VarianceProfile
from quantark.intraday.request import IntradayValuationRequest
from quantark.util.enum.option_enums import PostKOScheduleMode
from intraday.conftest import SHANGHAI, _assert_unchanged, dated_ko_reset, dated_snowball, flat_env, snapshot

T0 = datetime(2026, 3, 16)
TS = datetime(2026, 9, 15, 14, 0, tzinfo=SHANGHAI)
ENGINE = KOResetSnowballQuadEngineV2()


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def _checkpoint(knocked_in):
    return AutocallableLifecycleState(knocked_in=knocked_in, valuation_point=ValuationPoint(date=datetime(2026, 9, 14)))


def _req(sessions, profile, prod, ts=TS, spot=100.0, **kw):
    return IntradayValuationRequest(product=prod, pricing_env=flat_env(ts, spot=spot), session_calendar=sessions,
                                    variance_profile=profile, **kw)


def test_pre_and_post_ko_events_carry_regime_and_contractual_cash(sse_calendar, sse_sessions):
    prod = dated_ko_reset(sse_calendar, T0, pre_months=6, post_months=12)
    tl = resolve_timeline(prod, sse_sessions, flat_env(TS))
    pre = [e for e in tl.events if e.kind is EventKind.KO and e.regime == "pre"]
    post = [e for e in tl.events if e.kind is EventKind.KO and e.regime == "post"]
    assert len(pre) == 6 and len(post) == 12
    d0 = prod.barrier_config.ko_observation_schedule.records[0].observation_date
    assert pre[0].cash == pytest.approx(100.0 * 0.15 * (d0 - T0).days / 365.0)
    assert post[0].cash == pytest.approx(100.0 * 0.03 * (d0 - T0).days / 365.0)
    assert pre[0].event_id != post[0].event_id and pre[0].timestamp == post[0].timestamp


def test_twin_carries_both_schedules_and_prices_through_quad_v2(sse_calendar, sse_sessions, desk):
    prod = dated_ko_reset(sse_calendar, T0)
    before = snapshot(prod)
    for knocked_in in (False, True):
        ctx = resolve_context(_req(sse_sessions, desk, prod, lifecycle_state=_checkpoint(knocked_in)))
        twin = ctx.numerical.product
        assert twin.exercise_date is None and twin.initial_date is None
        assert getattr(twin, "_otc_lifecycle_knocked_in") is knocked_in
        assert len(twin.barrier_config.ko_observation_schedule.records) == 1       # Sept pre-KI fixing left of 6
        assert len(twin.post_barrier_config.ko_observation_schedule.records) == 7  # Sept..Mar post-KI fixings
        out = route_for(ctx, ENGINE).price(ctx, ENGINE)
        direct = KOResetSnowballQuadEngineV2().price(twin, ctx.pricing_env, lifecycle_state=ctx.numerical.lifecycle_state)
        assert out.contingent_pv == pytest.approx(direct, abs=1e-12) and abs(out.numerical["reconciliation_error"]) < 1e-10
    _assert_unchanged(before, prod)
    alive = value_intraday(ENGINE, _req(sse_sessions, desk, prod, lifecycle_state=_checkpoint(False)))
    knocked = value_intraday(ENGINE, _req(sse_sessions, desk, prod, lifecycle_state=_checkpoint(True)))
    assert alive.price != knocked.price and not alive.provisional


def test_rebased_post_schedule_is_a_capability_error(sse_calendar, sse_sessions, desk):
    prod = dated_ko_reset(sse_calendar, T0)
    prod.post_ko_mode = PostKOScheduleMode.REBASED
    with pytest.raises(CapabilityError, match="REBASED"):
        resolve_context(_req(sse_sessions, desk, prod, lifecycle_state=_checkpoint(False)))


def test_uncovered_fixings_fail_closed(sse_calendar, sse_sessions, desk):
    prod = dated_ko_reset(sse_calendar, T0)
    with pytest.raises(CapabilityError, match="KO-reset"):
        resolve_context(_req(sse_sessions, desk, prod))
    one_day_late = AutocallableLifecycleState(valuation_point=ValuationPoint(date=datetime(2026, 8, 16)))
    with pytest.raises(CapabilityError, match="2026-08-17"):
        resolve_context(_req(sse_sessions, desk, prod, lifecycle_state=one_day_late))


def test_fixings_at_different_times_of_day_keep_the_contractual_cash(sse_calendar, sse_sessions, desk):
    """The twin used to bank one accrued offset for every record, so a fixing at another time of day was refused.
    Each twin record now carries its source record's accrual period (accrual time spec §6): the twin builds, and its
    own cash verification (equal to the contract's to 1e-12) passes."""
    prod = dated_ko_reset(sse_calendar, T0)
    rec = prod.post_barrier_config.ko_observation_schedule.records[8]
    rec.observation_timestamp = datetime(rec.observation_date.year, rec.observation_date.month, rec.observation_date.day,
                                         10, 0, tzinfo=SHANGHAI)
    assert resolve_context(_req(sse_sessions, desk, prod, lifecycle_state=_checkpoint(True))) is not None


# --- the contract's two rules (KnockOutResetSnowballOption), past the first schedule ---------------------------------
LATE = datetime(2026, 10, 15, 14, 0, tzinfo=SHANGHAI)      # the six-month first schedule ended on 16 September


def _late_checkpoint(knocked_in):
    return AutocallableLifecycleState(knocked_in=knocked_in, valuation_point=ValuationPoint(date=datetime(2026, 10, 14)))


def test_a_knocked_in_contract_past_its_first_schedule_is_a_knocked_in_snowball_on_the_second(sse_calendar, sse_sessions, desk):
    """Rule 1: the knock-in replaced the first schedule by the second and stays to the end, so the contract is still
    alive in October and worth what a knocked-in snowball with the second schedule's dates, level and rate is worth.
    The twin used to fail closed here (its emptied first schedule cannot reproduce the pre-KI tenor, which nothing
    reads any more)."""
    from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
    from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
    from quantark.asset.equity.param import PDEParams

    ctx = resolve_context(_req(sse_sessions, desk, dated_ko_reset(sse_calendar, T0), ts=LATE, spot=90.0,
                               lifecycle_state=_late_checkpoint(True)))
    twin = ctx.numerical.product
    assert not twin.barrier_config.ko_observation_schedule.records                  # the first schedule is behind us
    assert len(twin.post_barrier_config.ko_observation_schedule.records) == 6      # Oct..Mar of the second
    value = route_for(ctx, ENGINE).price(ctx, ENGINE).contingent_pv

    plain = dated_snowball(sse_calendar, T0, months=12, ko=95.0, ki=75.0, ko_rate=0.03)
    plain_ctx = resolve_context(_req(sse_sessions, desk, plain, ts=LATE, spot=90.0, lifecycle_state=_late_checkpoint(True)))
    plain_engine = SnowballQuadEngineV2()
    assert value == pytest.approx(route_for(plain_ctx, plain_engine).price(plain_ctx, plain_engine).contingent_pv, abs=1e-9)

    pde = KOResetSnowballPDESolver(PDEParams())
    assert route_for(ctx, pde).price(ctx, pde).contingent_pv == pytest.approx(value, abs=2e-2)


def test_a_contract_called_alive_past_its_first_schedule_without_a_knock_in_is_refused(sse_calendar, sse_sessions, desk):
    """Rule 2: it matured when the first schedule ended. The checkpoint is wrong, and pricing it would extend the first
    schedule to the final maturity."""
    from quantark.util.exceptions import ValidationError

    with pytest.raises(ValidationError, match="matured"):
        resolve_context(_req(sse_sessions, desk, dated_ko_reset(sse_calendar, T0), ts=LATE, lifecycle_state=_late_checkpoint(False)))


# --- EXPIRY pay: a knock-out is booked when the note it ends is paid (spec D30, D33) ---------------------------------

def _expiry(prod, settlement_date=None):
    from dataclasses import replace

    from quantark.util.enum import CouponPayType

    prod.accrual_config = replace(prod.accrual_config, coupon_pay_type=CouponPayType.EXPIRY)
    prod.settlement_date = settlement_date
    prod.validate()
    return prod


def _ko_events(timeline, regime):
    return [e for e in timeline.events if e.kind is EventKind.KO and e.regime == regime]


def test_an_expiry_schedule_one_knock_out_is_booked_when_schedule_one_is_paid(sse_calendar, sse_sessions):
    """A schedule-1 knock-out happens only on a note never knocked in, which ends at schedule 1's maturity (D30): under
    EXPIRY pay it is booked on that note's payment -- here (no settlement date, no lag) the payment schedule 1's last
    record makes under INSTANT pay -- not with the product terminal. A schedule-2 knock-out keeps the terminal's."""
    timeline = resolve_timeline(_expiry(dated_ko_reset(sse_calendar, T0, pre_months=6, post_months=12)), sse_sessions,
                                flat_env(TS))
    instant = resolve_timeline(dated_ko_reset(sse_calendar, T0, pre_months=6, post_months=12), sse_sessions, flat_env(TS))
    schedule_one_paid = max(_ko_events(instant, "pre"), key=lambda e: e.timestamp).payment_timestamp
    terminal = timeline.terminal().payment_timestamp
    assert schedule_one_paid < terminal
    for event in _ko_events(timeline, "pre"):
        assert event.payment_timestamp == max(schedule_one_paid, event.timestamp)
    for event in _ko_events(timeline, "post"):
        assert event.payment_timestamp == max(terminal, event.timestamp)


def test_an_expiry_schedule_one_knock_out_is_booked_on_the_settlement_date(sse_calendar, sse_sessions):
    """With an explicit settlement date the note that ends with schedule 1 is paid on it (D33), as the product terminal
    is: every knock-out, pre or post, is booked on the settlement date's payment instant."""
    prod = dated_ko_reset(sse_calendar, T0, pre_months=6, post_months=12)
    settlement = prod.exercise_date + timedelta(days=7)
    timeline = resolve_timeline(_expiry(prod, settlement), sse_sessions, flat_env(TS))
    paid = sse_sessions.payment_at(settlement.date())
    assert timeline.terminal().payment_timestamp == paid
    for event in _ko_events(timeline, "pre") + _ko_events(timeline, "post"):
        assert event.payment_timestamp == paid
