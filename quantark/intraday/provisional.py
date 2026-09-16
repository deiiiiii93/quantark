"""Confirmed + provisional lifecycle reconstruction.

Starts from the authoritative checkpoint every time, merges supplied fixings
with latest-spot assumptions for the fixings that are due but missing, and
replays them chronologically through the existing lifecycle trackers. The
checkpoint and product are never mutated; assumed outcomes live only in the
returned reconstruction (design §Confirmed and provisional state).

The daily trackers observe by DATE at one spot. Guards keep that honest on an
intraday timeline: events the checkpoint already covers are marked observed
before replay (so they are never re-observed at a later spot), every replay
step must process exactly the events of its instant, and two due instants on
one local date fail closed. A continuously monitored barrier discloses the
interval whose touch history is unknown (design decision 8).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional, Sequence, Tuple

import pandas as pd

from quantark.asset.equity.lifecycle import (
    AutocallableLifecycleState,
    AutocallableLifecycleTracker,
    BarrierLifecycleState,
    BarrierLifecycleTracker,
    LifecycleEventType,
    RealizedCashflow,
    TerminalLifecycleState,
    ValuationPoint,
)
from quantark.execution.cache.fingerprint import fingerprint
from quantark.execution.errors import CapabilityError
from quantark.intraday.events import ContractTimeline, EventKind, EventPhase
from quantark.intraday.fixings import AssumedFixing, ContinuousHistoryAssumption, Fixing
from quantark.intraday.timestamp import require_aware, to_utc
from quantark.util.exceptions import ValidationError

#: Metadata keys carrying the exact instants of a cashflow the intraday layer created itself.
DETERMINATION_TIMESTAMP = "intraday_determination_timestamp"
PAYMENT_TIMESTAMP = "intraday_payment_timestamp"
ASSUMED = "intraday_assumed"


@dataclass(frozen=True)
class LifecycleReconstruction:
    state: object                                    # lifecycle state (date-based ledger, as the tracker writes it) or None
    applied_event_ids: Tuple[str, ...]               # events determined by this call, chronological
    confirmed_event_ids: Tuple[str, ...]             # subset determined from supplied fixings
    assumptions: Tuple[AssumedFixing, ...]
    continuous_assumption: Optional[ContinuousHistoryAssumption]
    checkpoint_fingerprint: Optional[str]

    @property
    def provisional(self) -> bool:
        return bool(self.assumptions) or self.continuous_assumption is not None


def _iso(value) -> Optional[str]:
    return None if value is None else value.isoformat()


def lifecycle_state_fingerprint(state) -> str:
    """Value fingerprint of a lifecycle state (its ledger and index sets have no generic canonicalizer)."""
    point = state.valuation_point
    flows = tuple(
        (cf.cashflow_id, cf.event_type.value, float(cf.amount), _iso(cf.determination_date), _iso(cf.payment_date),
         cf.determination_time, cf.payment_time, tuple(sorted((str(k), str(v)) for k, v in cf.metadata.items())))
        for cf in state.ledger.cashflows
    )
    scalars = tuple((name, _iso(value) if isinstance(value, datetime) else value)
                    for name, value in sorted(vars(state).items())
                    if name not in ("ledger", "valuation_point") and not isinstance(value, set))
    sets = tuple((name, tuple(sorted(value))) for name, value in sorted(vars(state).items()) if isinstance(value, set))
    return fingerprint((type(state).__name__, scalars, sets,
                        None if point is None else (_iso(point.date), point.time), flows))


#: Declared reading of a checkpoint that carries no time of day.
DATE_ONLY_CHECKPOINT_CONVENTION = "after that day's close"


def checkpoint_instant(checkpoint, cal, valuation_timestamp: datetime, phase: EventPhase, timeline=None):
    """(instant the checkpoint covers up to and including, convention) — ``(None, "")`` without one.

    ``ValuationPoint.date`` is a naive LOCAL datetime. The daily trackers write it
    with no time of day after observing that day's fixings, so midnight is the
    legacy date-only checkpoint: the state AFTER THAT DAY'S CLOSE (on a day
    without a session, after the last close before it). That instant is fixed by
    the declaration. It is never moved to fit the request: a state that already
    contains the close's outcomes still contains them when its timestamp is
    clamped, so a date-only checkpoint whose close is after the valuation instant
    is rejected like any other report of the future (review 2026-09-16 R1).

    Any other wall-clock time is an EXPLICIT intraday checkpoint and covers exactly
    the events at or before that instant: widening it to its whole date would
    suppress fixings it never saw.

    Coverage is inclusive, so a checkpoint AT the valuation instant has decided the
    events there; under ``before`` those events are still open, and the two
    statements contradict each other.
    """
    point = getattr(checkpoint, "valuation_point", None) if checkpoint is not None else None
    if point is None:
        return None, ""
    if point.date is None:
        raise ValidationError("intraday reconstruction needs a date-based checkpoint valuation_point "
                              "(time-based checkpoints cannot be placed on the session calendar)")
    local = point.date
    if local.tzinfo is not None:
        raise ValidationError(f"checkpoint valuation_point {local.isoformat()} is timezone-aware; the daily lifecycle "
                              f"trackers write LOCAL naive datetimes on {cal.name}")
    day = local.date()
    if local.time() == time(0, 0):
        close_day = day if cal.is_trading_day(day) else cal.previous_trading_day(day)
        instant, convention = cal.close_at(close_day), DATE_ONLY_CHECKPOINT_CONVENTION
    else:
        instant, convention = cal.localize(day, local.time()), "at the stated local instant"
    if to_utc(instant) > to_utc(valuation_timestamp):
        raise ValidationError(f"checkpoint covers {instant.isoformat()} ({convention} of {local.isoformat()}), which is "
                              f"after the valuation instant {valuation_timestamp.isoformat()}: a checkpoint cannot "
                              "report the future")
    if (timeline is not None and phase is EventPhase.BEFORE and to_utc(instant) == to_utc(valuation_timestamp)
            and timeline.at(valuation_timestamp)):
        raise ValidationError(f"checkpoint covers {instant.isoformat()} inclusively, deciding "
                              f"{timeline.at(valuation_timestamp)[0].event_id}, but the valuation is BEFORE that event: "
                              "a checkpoint cannot report the future")
    return instant, convention


def _earliest_instant(stamp: datetime, cal) -> datetime:
    """The earliest instant a naive LOCAL lifecycle stamp can denote: a date-only stamp is its whole day."""
    return cal.localize(stamp.date(), stamp.time())


def _require_covered_contents(checkpoint, timeline, checkpoint_at: datetime, convention: str, cal) -> None:
    """Reject a checkpoint carrying an outcome determined after the instant it claims to cover.

    Its instant is a claim about its CONTENTS: every realized cashflow, every
    observation it marks as seen and every hit/expiry date must lie at or before
    it. Each is placed as precisely as it can be — a cashflow by the exact
    determination instant the intraday layer recorded or by the contract event its
    id names, an observation index by its event, a bare date only by the start of
    its day — so nothing is rejected on a guess.
    """
    from quantark.intraday.twin import event_for_cashflow

    late = []
    for cf in checkpoint.ledger.cashflows:
        stamp = cf.metadata.get(DETERMINATION_TIMESTAMP)
        event = None if stamp else event_for_cashflow(cf, timeline, cal.tz)
        if stamp:
            instant = datetime.fromisoformat(stamp)
        elif event is not None:
            instant = event.timestamp
        elif cf.determination_date is not None:
            instant = _earliest_instant(cf.determination_date, cal)
        else:
            continue
        if to_utc(instant) > to_utc(checkpoint_at):
            late.append(f"cashflow {cf.cashflow_id} determined {instant.isoformat()}")
    if isinstance(checkpoint, AutocallableLifecycleState):
        for kind, seen in ((EventKind.KO, checkpoint.observed_ko_indices), (EventKind.KI, checkpoint.observed_ki_indices),
                           (EventKind.COUPON, checkpoint.observed_coupon_indices)):
            late.extend(f"observed {e.event_id}" for e in timeline.events
                        if e.kind is kind and e.index in seen and to_utc(e.timestamp) > to_utc(checkpoint_at))
    for name in ("ki_date", "ko_date", "maturity_date", "hit_date", "expiry_date"):
        stamp = getattr(checkpoint, name, None)
        if stamp is not None and to_utc(_earliest_instant(stamp, cal)) > to_utc(checkpoint_at):
            late.append(f"{name} {stamp.isoformat()}")
    if late:
        raise ValidationError(f"checkpoint covers {checkpoint_at.isoformat()} ({convention}) but carries outcomes "
                              f"determined after it: {'; '.join(late)}. A checkpoint cannot report the future")


def _covered(checkpoint_at: Optional[datetime], instant: datetime) -> bool:
    """Whether the authoritative checkpoint already determined an event at ``instant`` (inclusive)."""
    return checkpoint_at is not None and to_utc(instant) <= to_utc(checkpoint_at)


class _ReplayPlan:
    """Validated fixings and the chronological due-and-uncovered instants of one reconstruction."""

    def __init__(self, timeline, checkpoint, fixings, ts, phase, spot, spot_timestamp, cal):
        self.ts, self.phase, self.cal, self.tz = ts, phase, cal, cal.tz
        self.spot, self._spot_timestamp = float(spot), spot_timestamp
        if checkpoint is not None and not checkpoint.alive and fixings:
            raise ValidationError("checkpoint is already terminal; fixings after termination are contradictory")
        self.checkpoint_at, self.checkpoint_convention = checkpoint_instant(checkpoint, cal, ts, phase, timeline)
        if self.checkpoint_at is not None:
            _require_covered_contents(checkpoint, timeline, self.checkpoint_at, self.checkpoint_convention, cal)
        self.instants = {}
        for e in timeline.events:
            self.instants.setdefault(to_utc(e.timestamp), []).append(e)
        self.due_ids = {e.event_id for e in timeline.due(ts, phase)}
        self.supplied = {}
        for f in fixings:
            key = to_utc(f.timestamp)
            if key not in self.instants:
                raise ValidationError(f"fixing at {f.timestamp.isoformat()} matches no contract event")
            if not any(e.event_id in self.due_ids for e in self.instants[key]):
                raise ValidationError(f"fixing at {f.timestamp.isoformat()} is for an event not yet determined at "
                                      f"{ts.isoformat()} ({phase.value})")
            if _covered(self.checkpoint_at, f.timestamp):
                raise ValidationError(f"fixing at {f.timestamp.isoformat()} is already covered by the checkpoint "
                                      f"({self.checkpoint_convention}: {self.checkpoint_at.isoformat()})")
            if key in self.supplied and self.supplied[key].value != f.value:
                raise ValidationError(f"two different fixings supplied for {f.timestamp.isoformat()}")
            self.supplied[key] = f
        self.replay = [key for key in sorted(self.instants)
                       if any(e.event_id in self.due_ids for e in self.instants[key])
                       and not _covered(self.checkpoint_at, self.instants[key][0].timestamp)]
        days = [self.instants[key][0].timestamp.astimezone(self.tz).date() for key in self.replay]
        if len(set(days)) != len(days):
            raise ValidationError("two due fixing instants share one local date; the daily lifecycle tracker "
                                  "cannot tell them apart")
        self.applied, self.confirmed, self.assumptions = [], [], []

    def group(self, key):
        return [e for e in self.instants[key] if e.event_id in self.due_ids]

    def spot_timestamp(self) -> datetime:
        if self._spot_timestamp is None:
            raise ValidationError("a provisional scenario needs the spot timestamp (SpotQuote.timestamp) "
                                  "to record its assumption")
        require_aware(self._spot_timestamp, "spot timestamp")
        if to_utc(self._spot_timestamp) > to_utc(self.ts):
            raise ValidationError("spot timestamp is after the valuation timestamp")
        return self._spot_timestamp

    def value(self, key, group) -> float:
        if key in self.supplied:
            self.confirmed.extend(e.event_id for e in group)
            return self.supplied[key].value
        self.assumptions.append(AssumedFixing(tuple(e.event_id for e in group), group[0].timestamp, self.spot,
                                              self.spot_timestamp()))
        return self.spot

    def local_day(self, group) -> pd.Timestamp:
        return pd.Timestamp(group[0].timestamp.astimezone(self.tz).date())

    def uncovered_start(self, timeline) -> Optional[datetime]:
        """Start of the interval whose continuous touch history nobody reported."""
        return self.checkpoint_at if self.checkpoint_at is not None else timeline.initial_timestamp

    def result(self, state, checkpoint, continuous) -> "LifecycleReconstruction":
        return LifecycleReconstruction(
            state=state, applied_event_ids=tuple(self.applied), confirmed_event_ids=tuple(self.confirmed),
            assumptions=tuple(self.assumptions), continuous_assumption=continuous,
            checkpoint_fingerprint=lifecycle_state_fingerprint(checkpoint) if checkpoint is not None else None)


def _reconstruct_terminal(product, timeline, checkpoint, plan: "_ReplayPlan") -> LifecycleReconstruction:
    """Terminal-only contract (digital, European vanilla): its single fixing decides everything.

    Before the fixing nothing is realized and the whole claim is contingent.
    At or after it the payoff is a FIXED receivable settling on the contract's
    own payment instant, so the remaining value is that receivable -- exactly
    the pending/paid ledger every other family already carries.
    """
    state = deepcopy(checkpoint) if checkpoint is not None else TerminalLifecycleState()
    for key in plan.replay:
        if not state.alive:
            break
        group = plan.group(key)
        kinds = {e.kind for e in group}
        if kinds != {EventKind.TERMINAL}:
            raise ValidationError(f"a terminal-only contract has no {sorted(k.value for k in kinds)} observation at "
                                  f"{group[0].timestamp.isoformat()}")
        event = group[0]
        value = plan.value(key, group)
        local = event.timestamp.astimezone(plan.tz).replace(tzinfo=None)
        state.ledger.register(RealizedCashflow(
            cashflow_id=f"expiry:{local.date().isoformat()}", event_type=LifecycleEventType.EXPIRY,
            amount=float(product.get_payoff(value)), determination_date=local,
            payment_date=event.payment_timestamp.astimezone(plan.tz).replace(tzinfo=None),
            metadata={DETERMINATION_TIMESTAMP: event.timestamp.isoformat(),
                      PAYMENT_TIMESTAMP: event.payment_timestamp.isoformat()}))
        state.mark_expired(local)
        state.valuation_point = ValuationPoint(date=plan.ts.astimezone(plan.tz).replace(tzinfo=None))
        plan.applied.extend(e.event_id for e in group)
    return plan.result(state, checkpoint, None)


def reconstruct_lifecycle(product, timeline: ContractTimeline, checkpoint, fixings: Sequence[Fixing], *,
                          valuation_timestamp: datetime, phase: EventPhase, spot: float,
                          spot_timestamp: Optional[datetime], schedule_env, session_calendar) -> LifecycleReconstruction:
    """Authoritative checkpoint + supplied fixings + assumed missing fixings, replayed chronologically."""
    from quantark.asset.equity.product.option.barrier_option import BarrierOption
    from quantark.asset.equity.product.option.one_touch_option import OneTouchOption

    require_aware(valuation_timestamp, "valuation_timestamp")
    if type(product) in (BarrierOption, OneTouchOption):
        if checkpoint is not None and not isinstance(checkpoint, BarrierLifecycleState):
            raise ValidationError(f"barrier products take a BarrierLifecycleState checkpoint, got {type(checkpoint).__name__}")
        plan = _ReplayPlan(timeline, checkpoint, fixings, valuation_timestamp, phase, spot, spot_timestamp, session_calendar)
        return _reconstruct_barrier(product, timeline, checkpoint, plan, schedule_env)
    if not hasattr(product, "barrier_config"):
        if checkpoint is not None and not isinstance(checkpoint, TerminalLifecycleState):
            raise ValidationError(f"a terminal-only contract takes a TerminalLifecycleState checkpoint, "
                                  f"got {type(checkpoint).__name__}")
        plan = _ReplayPlan(timeline, checkpoint, fixings, valuation_timestamp, phase, spot, spot_timestamp, session_calendar)
        return _reconstruct_terminal(product, timeline, checkpoint, plan)
    if checkpoint is not None and not isinstance(checkpoint, AutocallableLifecycleState):
        raise ValidationError(f"intraday reconstruction supports AutocallableLifecycleState checkpoints, "
                              f"got {type(checkpoint).__name__}")
    plan = _ReplayPlan(timeline, checkpoint, fixings, valuation_timestamp, phase, spot, spot_timestamp, session_calendar)
    return _reconstruct_autocallable(product, timeline, checkpoint, plan, schedule_env)


# ---------------------------------------------------------------------------
# autocallables
def _reconstruct_autocallable(product, timeline, checkpoint, plan: _ReplayPlan, schedule_env) -> LifecycleReconstruction:
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption

    if type(product) is KnockOutResetSnowballOption and plan.replay:
        raise CapabilityError(
            "KO-reset snowball replay is not in the intraday inventory: the daily lifecycle tracker observes only the "
            "pre-KI schedule and settles at the post-KI maturity. Supply a checkpoint that covers every fixing due "
            f"before the valuation instant (first uncovered: {plan.instants[plan.replay[0]][0].timestamp.isoformat()}).")
    state = deepcopy(checkpoint) if checkpoint is not None else AutocallableLifecycleState()
    # The checkpoint is authoritative for everything on or before its day.
    for e in timeline.events:
        if _covered(plan.checkpoint_at, e.timestamp):
            {EventKind.KO: state.observed_ko_indices, EventKind.KI: state.observed_ki_indices,
             EventKind.COUPON: state.observed_coupon_indices}.get(e.kind, set()).add(e.index)
    tracker = AutocallableLifecycleTracker(product=product, quantity=1.0, lifecycle=state,
                                           start_date=getattr(product, "initial_date", None))
    for key in plan.replay:
        if not state.alive:
            break
        group = plan.group(key)
        value = plan.value(key, group)
        local_day = plan.local_day(group)
        lifecycle_product = tracker.product_for_lifecycle()
        before = (set(state.observed_ko_indices), set(state.observed_ki_indices), set(state.observed_coupon_indices))
        ko_disabled = bool(state.knocked_in and product.barrier_config.disable_ko_after_ki)
        tracker.observe(local_day, lifecycle_product, schedule_env, value)
        _check_replay_step(group, state, before, ko_disabled)
        if any(e.kind is EventKind.TERMINAL for e in group) and state.alive:
            tracker.settle_maturity_if_due(local_day, lifecycle_product, schedule_env, value)
            if state.alive:
                raise ValidationError(f"terminal event at {group[0].timestamp.isoformat()} did not settle the contract")
        plan.applied.extend(e.event_id for e in group)

    continuous = None
    if timeline.continuous_ki_barrier is not None and state.alive and not state.knocked_in:
        start = plan.uncovered_start(timeline)
        if start is None:
            raise ValidationError("continuous KI history needs a checkpoint or the contract's initial instant")
        if to_utc(start) < to_utc(plan.ts):
            plan.spot_timestamp()
            hit = AutocallableLifecycleTracker._barrier_hit(plan.spot, timeline.continuous_ki_barrier,
                                                           bool(product.is_reverse), is_ko=False)
            if hit:
                state.mark_ki(plan.ts.astimezone(plan.tz).replace(tzinfo=None))
            continuous = ContinuousHistoryAssumption(uncovered_from=start, uncovered_to=plan.ts,
                                                     assumed_hit_at=plan.ts if hit else None)
    return plan.result(state, checkpoint, continuous)


def _check_replay_step(group, state, before, ko_disabled) -> None:
    """The tracker must have observed exactly this instant's events (a KO ends the step early)."""
    expected = {kind: {e.index for e in group if e.kind is kind} for kind in (EventKind.KO, EventKind.KI, EventKind.COUPON)}
    new = {EventKind.KO: state.observed_ko_indices - before[0], EventKind.KI: state.observed_ki_indices - before[1],
           EventKind.COUPON: state.observed_coupon_indices - before[2]}
    stray = {k.value: sorted(new[k] - expected[k]) for k in new if new[k] - expected[k]}
    if stray:
        raise ValidationError(f"the daily lifecycle tracker observed events outside instant "
                              f"{group[0].timestamp.isoformat()}: {stray}")
    if state.knocked_out:
        return
    missing = {k.value: sorted(expected[k] - new[k]) for k in new
               if expected[k] - new[k] and not (k is EventKind.KO and ko_disabled)}
    if missing:
        raise ValidationError(f"the daily lifecycle tracker could not place instant {group[0].timestamp.isoformat()} "
                              f"on its replay clock: unobserved {missing}")


# ---------------------------------------------------------------------------
# single-barrier products (barrier options, one-touch / no-touch)
def _reconstruct_barrier(product, timeline, checkpoint, plan: _ReplayPlan, schedule_env) -> LifecycleReconstruction:
    from quantark.intraday.events import barrier_terms

    state = deepcopy(checkpoint) if checkpoint is not None else BarrierLifecycleState()
    discrete = any(e.kind in (EventKind.KO, EventKind.KI) for e in timeline.events)
    tracker = BarrierLifecycleTracker(product=product, quantity=1.0, start_date=schedule_env.valuation_date)
    tracker.state = state
    for key in plan.replay:
        if not state.alive:
            break
        group = plan.group(key)
        kinds = {e.kind for e in group}
        if discrete and kinds == {EventKind.TERMINAL}:
            raise CapabilityError("discrete barrier whose schedule does not observe at expiry: the daily barrier "
                                  "tracker tests the barrier on every observation, including expiry; not in the "
                                  "intraday inventory")
        if EventKind.KI in kinds and state.knocked_in and kinds <= {EventKind.KI}:
            plan.value(key, group)             # already knocked in: the fixing changes nothing
            plan.applied.extend(e.event_id for e in group)
            continue
        value = plan.value(key, group)
        tracker.observe(plan.local_day(group), schedule_env, value)
        if EventKind.TERMINAL in kinds and state.alive:
            raise ValidationError(f"terminal event at {group[0].timestamp.isoformat()} did not settle the contract")
        plan.applied.extend(e.event_id for e in group)

    continuous = None
    terms = timeline.continuous_barrier
    still_open = state.alive and not (terms is not None and not terms.is_knock_out and state.knocked_in)
    if terms is not None and still_open and to_utc(plan.ts) < to_utc(timeline.terminal().timestamp):
        start = plan.uncovered_start(timeline)
        if start is None or to_utc(start) < to_utc(plan.ts):
            plan.spot_timestamp()
            hit = bool(product.is_barrier_hit(plan.spot))
            if hit:
                _assume_hit(state, barrier_terms(product), timeline, plan)
            continuous = ContinuousHistoryAssumption(uncovered_from=start, uncovered_to=plan.ts,
                                                     assumed_hit_at=plan.ts if hit else None)
    return plan.result(state, checkpoint, continuous)


def _assume_hit(state: BarrierLifecycleState, terms, timeline, plan: _ReplayPlan) -> None:
    """Apply a hit assumed at the valuation instant: knock-in switches, knock-out terminates with its cash."""
    local = plan.ts.astimezone(plan.tz).replace(tzinfo=None)
    state.hit_date = local
    if not terms.is_knock_out:
        state.knocked_in = True
        return
    pay = plan.ts if terms.pays_at_hit else max(timeline.terminal().payment_timestamp, plan.ts, key=to_utc)
    state.ledger.register(RealizedCashflow(
        cashflow_id=f"ko:assumed:{plan.ts.isoformat()}", event_type=LifecycleEventType.KNOCK_OUT,
        amount=float(terms.hit_cash or 0.0), determination_date=local,
        payment_date=pay.astimezone(plan.tz).replace(tzinfo=None),
        metadata={DETERMINATION_TIMESTAMP: plan.ts.isoformat(), PAYMENT_TIMESTAMP: pay.isoformat(), ASSUMED: True}))
    state.knocked_out = True
    state.alive = False
