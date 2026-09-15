"""Confirmed + provisional lifecycle reconstruction.

Starts from the authoritative checkpoint every time, merges supplied fixings
with latest-spot assumptions for the fixings that are due but missing, and
replays them chronologically through the existing lifecycle trackers. The
checkpoint and product are never mutated; assumed outcomes live only in the
returned reconstruction (design §Confirmed and provisional state).

The daily tracker observes every not-yet-observed record whose DATE has come,
at one spot. Two guards keep that honest on an intraday timeline: events the
checkpoint already covers are marked observed before replay (so they are never
re-observed at a later spot), and every replay step must process exactly the
events of its instant (two fixings on one local date fail closed).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence, Tuple

import pandas as pd

from quantark.asset.equity.lifecycle import AutocallableLifecycleState, AutocallableLifecycleTracker
from quantark.execution.cache.fingerprint import fingerprint
from quantark.execution.errors import CapabilityError
from quantark.intraday.events import ContractTimeline, EventKind, EventPhase
from quantark.intraday.fixings import AssumedFixing, ContinuousHistoryAssumption, Fixing
from quantark.intraday.timestamp import require_aware, to_utc
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class LifecycleReconstruction:
    state: object                                    # AutocallableLifecycleState (date-based ledger, as the tracker writes it) or None
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


def lifecycle_state_fingerprint(state: AutocallableLifecycleState) -> str:
    """Value fingerprint of a lifecycle state (its ledger and index sets have no generic canonicalizer)."""
    point = state.valuation_point
    flows = tuple(
        (cf.cashflow_id, cf.event_type.value, float(cf.amount), _iso(cf.determination_date), _iso(cf.payment_date),
         cf.determination_time, cf.payment_time)
        for cf in state.ledger.cashflows
    )
    return fingerprint((
        "AutocallableLifecycleState", state.alive, state.knocked_in, state.knocked_out, state.matured,
        _iso(state.ki_date), _iso(state.ko_date), _iso(state.maturity_date), int(state.coupon_memory_count),
        None if point is None else (_iso(point.date), point.time), flows,
        tuple(sorted(state.observed_ko_indices)), tuple(sorted(state.observed_ki_indices)),
        tuple(sorted(state.observed_coupon_indices)), float(state.pending_settlement_cashflow),
        _iso(state.settlement_date), state.settled,
    ))


def _checkpoint_day(checkpoint):
    point = getattr(checkpoint, "valuation_point", None) if checkpoint is not None else None
    if point is None:
        return None
    if point.date is None:
        raise ValidationError("intraday reconstruction needs a date-based checkpoint valuation_point "
                              "(time-based checkpoints cannot be placed on the session calendar)")
    return point.date.date()


def _covered(checkpoint_day, instant: datetime, tz) -> bool:
    return checkpoint_day is not None and instant.astimezone(tz).date() <= checkpoint_day


def _empty(checkpoint, fixings) -> LifecycleReconstruction:
    if checkpoint is not None or fixings:
        raise ValidationError("this product has no lifecycle: a checkpoint or fixings cannot be applied")
    return LifecycleReconstruction(None, (), (), (), None, None)


def reconstruct_lifecycle(product, timeline: ContractTimeline, checkpoint, fixings: Sequence[Fixing], *,
                          valuation_timestamp: datetime, phase: EventPhase, spot: float,
                          spot_timestamp: Optional[datetime], schedule_env, session_calendar) -> LifecycleReconstruction:
    """Authoritative checkpoint + supplied fixings + assumed missing fixings, replayed chronologically."""
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption

    require_aware(valuation_timestamp, "valuation_timestamp")
    if not hasattr(product, "barrier_config"):
        return _empty(checkpoint, fixings)
    tz = session_calendar.tz
    if checkpoint is not None and not isinstance(checkpoint, AutocallableLifecycleState):
        raise ValidationError(f"intraday reconstruction supports AutocallableLifecycleState checkpoints, "
                              f"got {type(checkpoint).__name__}")
    if checkpoint is not None and not checkpoint.alive and fixings:
        raise ValidationError("checkpoint is already terminal; fixings after termination are contradictory")
    checkpoint_day = _checkpoint_day(checkpoint)
    state = deepcopy(checkpoint) if checkpoint is not None else AutocallableLifecycleState()

    def need_spot_timestamp():
        if spot_timestamp is None:
            raise ValidationError("a provisional scenario needs the spot timestamp (SpotQuote.timestamp) "
                                  "to record its assumption")
        require_aware(spot_timestamp, "spot timestamp")
        if to_utc(spot_timestamp) > to_utc(valuation_timestamp):
            raise ValidationError("spot timestamp is after the valuation timestamp")
        return spot_timestamp

    instants = {}
    for e in timeline.events:
        instants.setdefault(to_utc(e.timestamp), []).append(e)
    due_ids = {e.event_id for e in timeline.due(valuation_timestamp, phase)}

    supplied = {}
    for f in fixings:
        key = to_utc(f.timestamp)
        if key not in instants:
            raise ValidationError(f"fixing at {f.timestamp.isoformat()} matches no contract event")
        if not any(e.event_id in due_ids for e in instants[key]):
            raise ValidationError(f"fixing at {f.timestamp.isoformat()} is for an event not yet determined at "
                                  f"{valuation_timestamp.isoformat()} ({phase.value})")
        if _covered(checkpoint_day, f.timestamp, tz):
            raise ValidationError(f"fixing at {f.timestamp.isoformat()} is already covered by the checkpoint "
                                  f"(valuation_point {checkpoint_day})")
        if key in supplied and supplied[key].value != f.value:
            raise ValidationError(f"two different fixings supplied for {f.timestamp.isoformat()}")
        supplied[key] = f

    # The checkpoint is authoritative for everything on or before its day.
    for e in timeline.events:
        if _covered(checkpoint_day, e.timestamp, tz):
            {EventKind.KO: state.observed_ko_indices, EventKind.KI: state.observed_ki_indices,
             EventKind.COUPON: state.observed_coupon_indices}.get(e.kind, set()).add(e.index)

    tracker = AutocallableLifecycleTracker(product=product, quantity=1.0, lifecycle=state,
                                           start_date=getattr(product, "initial_date", None))
    applied, confirmed, assumptions = [], [], []
    replay = [key for key in sorted(instants)
              if any(e.event_id in due_ids for e in instants[key])
              and not _covered(checkpoint_day, instants[key][0].timestamp, tz)]
    days = [instants[key][0].timestamp.astimezone(tz).date() for key in replay]
    if len(set(days)) != len(days):
        raise ValidationError("two due fixing instants share one local date; the daily lifecycle tracker "
                              "cannot tell them apart")
    for key in replay:
        if not state.alive:
            break
        group = [e for e in instants[key] if e.event_id in due_ids]
        if type(product) is PhoenixOption and any(e.kind is EventKind.COUPON for e in group):
            raise CapabilityError(
                "Phoenix coupon replay is not in the intraday inventory: the daily lifecycle tracker books "
                "get_coupon_payoff(idx) while the Phoenix engines pay per-period year fractions, so a realized "
                "or memorized coupon has no single contractual amount yet. Value before the first due coupon, "
                "or supply a checkpoint that covers it.")
        if key in supplied:
            value = supplied[key].value
            confirmed.extend(e.event_id for e in group)
        else:
            value = float(spot)
            assumptions.append(AssumedFixing(tuple(e.event_id for e in group), group[0].timestamp, value,
                                             need_spot_timestamp()))
        local_day = pd.Timestamp(group[0].timestamp.astimezone(tz).date())
        lifecycle_product = tracker.product_for_lifecycle()
        before = (set(state.observed_ko_indices), set(state.observed_ki_indices), set(state.observed_coupon_indices))
        ko_disabled = bool(state.knocked_in and product.barrier_config.disable_ko_after_ki)
        tracker.observe(local_day, lifecycle_product, schedule_env, value)
        _check_replay_step(group, state, before, ko_disabled)
        if any(e.kind is EventKind.TERMINAL for e in group) and state.alive:
            tracker.settle_maturity_if_due(local_day, lifecycle_product, schedule_env, value)
            if state.alive:
                raise ValidationError(f"terminal event at {group[0].timestamp.isoformat()} did not settle the contract")
        applied.extend(e.event_id for e in group)

    continuous = None
    if timeline.continuous_ki_barrier is not None and state.alive and not state.knocked_in:
        continuous = _continuous_ki_assumption(timeline, state, checkpoint_day, session_calendar,
                                               valuation_timestamp, float(spot), bool(product.is_reverse),
                                               need_spot_timestamp)

    return LifecycleReconstruction(
        state=state, applied_event_ids=tuple(applied), confirmed_event_ids=tuple(confirmed),
        assumptions=tuple(assumptions), continuous_assumption=continuous,
        checkpoint_fingerprint=lifecycle_state_fingerprint(checkpoint) if checkpoint is not None else None)


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


def _continuous_ki_assumption(timeline, state, checkpoint_day, cal, ts, spot, is_reverse, need_spot_timestamp):
    """Design decision 8 for a continuously monitored KI: disclose the unobserved interval, test the latest spot."""
    if checkpoint_day is not None:
        start = cal.close_at(checkpoint_day) if cal.is_trading_day(checkpoint_day) else cal.payment_at(checkpoint_day)
    else:
        start = timeline.initial_timestamp
    if start is None:
        raise ValidationError("continuous KI history needs a checkpoint or the contract's initial instant")
    if to_utc(start) >= to_utc(ts):
        return None
    need_spot_timestamp()
    hit = AutocallableLifecycleTracker._barrier_hit(spot, timeline.continuous_ki_barrier, is_reverse, is_ko=False)
    if hit:
        state.mark_ki(ts.astimezone(cal.tz).replace(tzinfo=None))
    return ContinuousHistoryAssumption(uncovered_from=start, uncovered_to=ts, assumed_hit_at=ts if hit else None)
