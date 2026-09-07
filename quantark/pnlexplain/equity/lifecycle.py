"""LifecycleTransition: the alive-at-t1 contract and the event row (spec §8)."""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from quantark.asset.equity.lifecycle.events import LifecycleEvent, LifecycleEventType
from quantark.pnlexplain.base import ExplainMethod, ExplainRow, Factor, RowKind
from quantark.pnlexplain.equity.fingerprints import check_contract_roll, engines_equivalent, lifecycle_fingerprint
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot, is_terminal
from quantark.util.calendar import DayCountConvention
from quantark.util.exceptions import NumericalError, ValidationError


def _validate_event(event: Any) -> None:
    """The event protocol the event row reports: a LifecycleEvent with a typed
    kind, a real date, finite amounts and a boolean termination flag."""
    if not isinstance(event, LifecycleEvent):
        raise ValidationError(
            f"lifecycle events must be LifecycleEvent instances, got {type(event).__name__}"
        )
    if not isinstance(event.event_type, LifecycleEventType):
        raise ValidationError(f"lifecycle event_type must be a LifecycleEventType, got {event.event_type!r}")
    try:
        stamp = pd.Timestamp(event.date)
    except (TypeError, ValueError):
        raise ValidationError(f"lifecycle event date is not a date: {event.date!r}") from None
    if pd.isna(stamp):
        raise ValidationError(f"lifecycle event {event.event_type.value} has no date")
    for name in ("payoff", "cashflow"):
        raw = getattr(event, name)
        try:
            amount = float(raw)
        except (TypeError, ValueError):
            raise ValidationError(
                f"lifecycle event {event.event_type.value} {name} must be a number, got {raw!r}"
            ) from None
        if not math.isfinite(amount):
            raise NumericalError(f"non-finite {name} on lifecycle event {event.event_type.value}: {raw!r}")
    if not isinstance(event.terminates_position, bool):
        raise ValidationError(
            f"lifecycle event terminates_position must be a bool, got {event.terminates_position!r}"
        )


@dataclass(frozen=True)
class LifecycleTransition:
    """The alive-at-t1 contract, the state pair and the events of one step.

    ``contract_roll_days`` declares by how many calendar days the alive
    contract was rolled from ``product_t0``. None (default) means "by the
    step's calendar days" (spec §5.3). ``0`` declares, explicitly, that the
    holder repriced the SAME float-maturity contract without rolling it
    (the equity BacktestEngine does this for untracked positions); the time
    row then carries only the valuation-date effect and no contract theta.
    """
    product_alive_t1: Any
    engine_alive_t1: Any
    state_before: Any
    state_after: Any
    events: Tuple[Any, ...] = ()
    contract_roll_days: Optional[int] = None

    def __post_init__(self) -> None:
        events = tuple(self.events)
        for ev in events:
            _validate_event(ev)
        object.__setattr__(self, "events", events)
        days = self.contract_roll_days
        if days is not None:
            if isinstance(days, bool) or not isinstance(days, numbers.Integral):
                raise ValidationError(
                    f"contract_roll_days must be None or an integer day count, got {days!r}"
                )
            if int(days) < 0:
                raise ValidationError("contract_roll_days must be None or a non-negative day count")
            object.__setattr__(self, "contract_roll_days", int(days))

    @property
    def changed(self) -> bool:
        return lifecycle_fingerprint(self.state_before) != lifecycle_fingerprint(self.state_after)


def _event_date(event: Any) -> datetime:
    return pd.Timestamp(event.date).normalize().to_pydatetime()


def float_maturity_decrement(pricing_env, start: datetime, days: int) -> Optional[float]:
    """How much a float maturity ages over ``days``, when the calendar year is not the clock.

    A float maturity carries no clock of its own, so the default rule is
    ``days/365``. On a BUSINESS_DAYS environment that rule does not describe the
    contract: it ages by the trading days elapsed over the environment's trading
    year, and one trading day is 1/244 of a year, not 1/365. Returns None to keep
    the calendar default.

    This never loosens the check. It replaces one exact expectation with another,
    so a contract rolled on the wrong clock still fails, it just fails against the
    right number.
    """
    if getattr(pricing_env, "day_count_convention", None) is not DayCountConvention.BUSINESS_DAYS:
        return None
    calendar = getattr(pricing_env, "calendar", None)
    if calendar is None:
        raise ValidationError(
            "a BUSINESS_DAYS environment needs a calendar to say how much a float-maturity "
            "contract ages over the step"
        )
    if days <= 0:
        return 0.0
    end = start + timedelta(days=int(days))
    trading_days = len(calendar.get_working_days(start, end, side="right"))
    return trading_days / float(pricing_env.bus_days_in_year)


def resolve_transition(
    snap0: ValuationSnapshot,
    snap1: ValuationSnapshot,
    transition: Optional[LifecycleTransition],
    *,
    calendar_days: int,
) -> LifecycleTransition:
    fp0 = lifecycle_fingerprint(snap0.lifecycle_state)
    fp1 = lifecycle_fingerprint(snap1.lifecycle_state)
    if transition is None:
        if fp0 != fp1:
            raise ValidationError(
                "lifecycle state changed between the snapshots: supply a LifecycleTransition "
                "with the alive-at-t1 product (guessing it would mislabel the event row)"
            )
        if not is_terminal(snap0.lifecycle_state):    # a terminal position has no contract to roll (spec §8)
            check_contract_roll(
                snap0.product, snap1.product, calendar_days,
                expected_decrement=float_maturity_decrement(
                    snap0.pricing_env, snap0.date, calendar_days))
        return LifecycleTransition(
            product_alive_t1=snap1.product, engine_alive_t1=snap1.engine,
            state_before=snap0.lifecycle_state, state_after=snap1.lifecycle_state, events=(),
        )
    if lifecycle_fingerprint(transition.state_before) != fp0:
        raise ValidationError("transition.state_before does not match snapshot_t0.lifecycle_state")
    if lifecycle_fingerprint(transition.state_after) != fp1:
        raise ValidationError("transition.state_after does not match snapshot_t1.lifecycle_state")
    roll_days = calendar_days if transition.contract_roll_days is None else transition.contract_roll_days
    if roll_days > calendar_days:
        raise ValidationError(
            f"contract_roll_days {roll_days} exceeds the step's {calendar_days} calendar days"
        )
    terminal_t0 = is_terminal(snap0.lifecycle_state)
    if not terminal_t0:
        check_contract_roll(
            snap0.product, transition.product_alive_t1, roll_days,
            expected_decrement=float_maturity_decrement(
                snap0.pricing_env, snap0.date, roll_days))
    if fp0 == fp1:
        if transition.events:
            raise ValidationError("transition carries events but the lifecycle state is unchanged")
        if not engines_equivalent(snap1.engine, transition.engine_alive_t1):
            raise ValidationError(
                "engine substitution without a lifecycle event is a MODEL change: "
                "set engine_alive_t1 to snapshot_t1.engine"
            )
        if not terminal_t0:
            check_contract_roll(transition.product_alive_t1, snap1.product, 0)
        return transition
    if not transition.events:
        raise ValidationError("lifecycle state changed but the transition carries no events")
    last = None
    for ev in transition.events:
        d = _event_date(ev)
        if not (snap0.date < d <= snap1.date):
            raise ValidationError(f"lifecycle event dated {d} is outside ({snap0.date}, {snap1.date}]")
        if last is not None and d < last:
            raise ValidationError("lifecycle events must be chronological")
        last = d
    # Termination must agree between the events and the state endpoints: a
    # terminal position never comes back, an event that terminates it is the
    # single last event and leaves a terminal state, and a newly terminal
    # state needs the event that terminated it.
    terminal_t1 = is_terminal(snap1.lifecycle_state)
    if terminal_t0 and not terminal_t1:
        raise ValidationError("a terminal position cannot become alive again")
    terminating = [i for i, ev in enumerate(transition.events) if ev.terminates_position]
    if terminating:
        if not terminal_t1:
            raise ValidationError(
                "an event terminates the position but snapshot_t1.lifecycle_state is not terminal"
            )
        if len(terminating) != 1 or terminating[0] != len(transition.events) - 1:
            raise ValidationError("the terminating lifecycle event must be the single last event")
    elif terminal_t1 and not terminal_t0:
        raise ValidationError(
            "snapshot_t1.lifecycle_state is terminal but no lifecycle event terminates the position"
        )
    return transition


def event_summary(transition: LifecycleTransition) -> Tuple[Dict[str, Any], ...]:
    """Events as reported in the event row's metadata (validated on construction)."""
    return tuple(
        {
            "event_type": ev.event_type.value, "date": _event_date(ev).isoformat(),
            "payoff": float(ev.payoff), "cashflow": float(ev.cashflow),
            "terminates_position": ev.terminates_position,
        }
        for ev in transition.events
    )


def event_row(cache: Any, transition: LifecycleTransition, level: str = "instrument") -> ExplainRow:
    pnl = cache.value_t1().total - cache.all_market().total
    return ExplainRow(
        factor=Factor.LIFECYCLE_EVENT, term=Factor.LIFECYCLE_EVENT.value,
        method=ExplainMethod.SHARED, kind=RowKind.COMPONENT, level=level, pnl=pnl,
        metadata={"changed": transition.changed, "events": event_summary(transition)},
    )
