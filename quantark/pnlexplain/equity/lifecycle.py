"""LifecycleTransition: the alive-at-t1 contract and the event row (spec §8)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from quantark.pnlexplain.base import ExplainMethod, ExplainRow, Factor, RowKind
from quantark.pnlexplain.equity.fingerprints import check_contract_roll, lifecycle_fingerprint
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot, is_terminal
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class LifecycleTransition:
    product_alive_t1: Any
    engine_alive_t1: Any
    state_before: Any
    state_after: Any
    events: Tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))

    @property
    def changed(self) -> bool:
        return lifecycle_fingerprint(self.state_before) != lifecycle_fingerprint(self.state_after)


def _event_date(event: Any) -> datetime:
    return pd.Timestamp(event.date).normalize().to_pydatetime()


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
            check_contract_roll(snap0.product, snap1.product, calendar_days)
        return LifecycleTransition(
            product_alive_t1=snap1.product, engine_alive_t1=snap1.engine,
            state_before=snap0.lifecycle_state, state_after=snap1.lifecycle_state, events=(),
        )
    if lifecycle_fingerprint(transition.state_before) != fp0:
        raise ValidationError("transition.state_before does not match snapshot_t0.lifecycle_state")
    if lifecycle_fingerprint(transition.state_after) != fp1:
        raise ValidationError("transition.state_after does not match snapshot_t1.lifecycle_state")
    terminal_t0 = is_terminal(snap0.lifecycle_state)
    if not terminal_t0:
        check_contract_roll(snap0.product, transition.product_alive_t1, calendar_days)
    if fp0 == fp1:
        if transition.events:
            raise ValidationError("transition carries events but the lifecycle state is unchanged")
        if snap1.engine is not transition.engine_alive_t1:
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
    return transition


def event_summary(transition: LifecycleTransition) -> Tuple[Dict[str, Any], ...]:
    return tuple(
        {
            "event_type": ev.event_type.value, "date": _event_date(ev).isoformat(),
            "payoff": float(getattr(ev, "payoff", 0.0)), "cashflow": float(getattr(ev, "cashflow", 0.0)),
            "terminates_position": bool(getattr(ev, "terminates_position", False)),
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
