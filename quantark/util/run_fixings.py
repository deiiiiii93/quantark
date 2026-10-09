"""Run fixings: a fixing that passes during a multi-day run is recorded once, at its own run day's spot.

Spec: docs/superpowers/specs/2026-10-09-run-fixings-design.md (W6). Asset-neutral and duck-typed like
``PortfolioLifecycleManager`` (never imports ``quantark.portfolio``): it acts on any object with ``positions``
(each with ``product``, ``quantity``, ``underlying``) and ``pricing_environments``.

A product takes part when ``has_run_fixings`` is true and it has the protocol methods ``missing_fixing_fills``,
``with_fixings_recorded`` (W5b), ``with_time_elapsed`` and ``open_fixings``. Rules (spec §3.3): R1 the run's quoted
spot; R2 t = 0 is live, t < 0 is history; R3 when the clock moves from path instant P to P', every record strictly
past at P' without an outcome is recorded at the spot of the FIRST instant it was live (t = 0) — P's spot on a
calendar-day clock, where a record is live at one instant only (R3 amendment / Ruling 4: a Friday fixing on a
``BUSINESS_DAYS`` clock with weekend rows is recorded at Friday's spot); R4 a record no instant fell on (never live)
takes the previous instant's spot, ``gap=True``; R5 the first valuation is W5b's freeze; R6 a numeric contract
ages by the environment's year fraction; R7 an undated run never advances; R8 time-of-day instants are refused; R9
the clock never goes back; R10 the run's working positions get copies, the booked product is never mutated.
"""
from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Collection, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from quantark.util.calendar import calculate_year_fraction
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import freeze_missing_fixings
from quantark.util.numerical import is_zero

RUN_FIXINGS_PROTOCOL = ("missing_fixing_fills", "with_fixings_recorded", "with_time_elapsed", "open_fixings")


@dataclass(frozen=True)
class RecordedFixing:
    """One fixing the run's clock recorded (R3/R4)."""

    position_id: str
    index: int            # record index (records are never dropped by these products)
    label: str            # W5b ``fixing_label`` at the step's instant
    outcome: Any          # bool (range accrual) or price / rate (Asian, TARN, TARF)
    spot: float           # the spot it was fixed at: its first live instant's (Ruling 4); a gap: the previous one's
    gap: bool             # True when it was not live (t = 0) at that instant (R4)


@dataclass(frozen=True)
class RunFixingDay:
    """What one ``register`` / ``advance`` did."""

    date: Optional[datetime]
    fixings: Tuple[RecordedFixing, ...]
    product_events: Tuple[Tuple[str, float, Any], ...] = ()   # (position_id, quantity, event): W4b (spec §6)
    pending_pv: float = 0.0                                   # Σ quantity × pending_pv(env): W4b (spec §6)


def participates(product: Any) -> bool:
    """True for a product that takes part in run fixings (every other position is left exactly as it is)."""
    return bool(getattr(product, "has_run_fixings", False)) and all(
        callable(getattr(product, name, None)) for name in RUN_FIXINGS_PROTOCOL
    )


def require_date_instant(instant: Any) -> None:
    """R8 / D6: a run instant holding a participating product must be a date (midnight)."""
    stamp = pd.Timestamp(instant)
    if stamp != stamp.normalize():
        # TODO(run-fixings): intraday fixing instants — a fixing-time term on the records (spec §10 follow-up)
        raise ValidationError(
            f"run instant {stamp} carries a time of day: a dated fixing has no fixing time, so a run holding a "
            "range accrual / Asian / FX range accrual must step on dates (spec 2026-10-09 run fixings R8)"
        )


def run_step_dt(prev_date: Any, env: Any) -> float:
    """R6: the environment's year fraction from ``prev_date`` to ``env.valuation_date`` (0.0 on the same date)."""
    new = env.valuation_date
    if pd.Timestamp(new) < pd.Timestamp(prev_date):
        raise ValidationError(f"the run clock cannot go back: from {prev_date} to {new} (R9)")
    if pd.Timestamp(new) == pd.Timestamp(prev_date):
        return 0.0
    return calculate_year_fraction(prev_date, new, env.day_count_convention, env.bus_days_in_year,
                                   calendar=getattr(env, "calendar", None))


def at_spot(env: Any, spot: float) -> Any:
    """A copy of ``env`` quoting ``spot`` (everything else shared)."""
    return dataclasses.replace(env, spot_quote=dataclasses.replace(env.spot_quote, spot=float(spot)))


def step_fixings(product: Any, env: Any, *, prev_date: Any, prev_spot: float,
                 run_fills: Mapping[int, Any],
                 live_spots: Optional[Mapping[int, float]] = None) -> Tuple[Any, Dict[int, Any]]:
    """R3/R4 for one product, stateless (the recorder and ``ProductReplay`` share it).

    ``product`` is already aged to ``env``'s date. Returns ``(recorded, new_fills)``: ``recorded`` carries
    ``run_fills`` (decisions of earlier steps) and ``new_fills``, the records strictly past at ``env`` and still open.
    A record in ``live_spots`` was live at an earlier instant and is decided at the spot of the first instant it was
    live (Ruling 4: the last path day on or before its date); any other (a gap) at ``prev_spot``. ``run_fills`` is
    not mutated.
    """
    require_date_instant(prev_date)
    require_date_instant(env.valuation_date)
    if pd.Timestamp(env.valuation_date) < pd.Timestamp(prev_date):
        raise ValidationError(f"the run clock cannot go back: from {prev_date} to {env.valuation_date} (R9)")
    if run_fills:
        product = product.with_fixings_recorded(dict(run_fills))
    new_fills = dict(product.missing_fixing_fills(at_spot(env, prev_spot)))
    by_spot: Dict[float, List[int]] = {}
    for index in new_fills:
        if live_spots and index in live_spots:
            by_spot.setdefault(float(live_spots[index]), []).append(index)
    for spot, indices in by_spot.items():          # grouped by the stored value: one decision per distinct spot
        at_live = product.missing_fixing_fills(at_spot(env, spot))
        for index in indices:
            new_fills[index] = at_live[index]
    return (product.with_fixings_recorded(new_fills) if new_fills else product), new_fills


def update_live(open_fixings: Mapping[int, Tuple[float, str]], date: Any, spot: float,
                before: Optional[Mapping[int, Tuple[Any, float]]] = None,
                exclude: Collection[int] = ()) -> Dict[int, Tuple[Any, float]]:
    """``{index: (date, spot)}`` of the instant each live (t = 0) record first went live (Ruling 4)."""
    before = before or {}
    return {i: before.get(i, (date, float(spot)))
            for i, (t, _label) in open_fixings.items() if is_zero(t) and i not in exclude}


class RunFixingRecorder:
    """Registers a run's participating positions at its first valuation and records their fixings as the clock moves.

    Call ``register(portfolio)`` once, after the first valuation's market is applied (it replaces W5b's
    ``freeze_portfolio_fixings`` in the run engines), and ``advance(portfolio)`` after every later path instant's
    market and date are applied, before anything is priced.

    Precondition: between ``register`` and ``advance`` a run must not replace a participant's product except with a
    copy that keeps its records (outcomes included); a reset to the booked product would re-record its fixings.
    """

    def __init__(self) -> None:
        self._last: Dict[str, Tuple[Any, float]] = {}      # underlying -> (date, quoted spot) of the last instant
        self._live: Dict[str, Dict[int, Tuple[Any, float]]] = {}   # position id -> index -> (date, spot) first live
        self._days: List[RunFixingDay] = []

    @property
    def days(self) -> Tuple[RunFixingDay, ...]:
        return tuple(self._days)

    @property
    def recorded(self) -> Tuple[RecordedFixing, ...]:
        return tuple(f for day in self._days for f in day.fixings)

    @staticmethod
    def _participants(portfolio) -> List[Tuple[str, Any]]:
        positions = getattr(portfolio, "positions", None) or {}
        return [(pid, pos) for pid, pos in list(positions.items()) if participates(getattr(pos, "product", None))]

    def _register_position(self, portfolio, pid: str, position: Any, env: Any) -> bool:
        """R5 for one position; False when it matured before the run and left the working portfolio (spec §6)."""
        matured_before = getattr(position.product, "matured_before", None)
        if callable(matured_before) and matured_before(env):
            from quantark.asset.equity.lifecycle.autocallable import MaturedBeforeRunWarning  # lazy: util -> asset
            warnings.warn(MaturedBeforeRunWarning(
                f"position {pid} ({type(position.product).__name__}) matured before the run start "
                f"{env.valuation_date}: it is not settled in the run, books no cash and is valued at zero"),
                stacklevel=3)
            portfolio.remove_position(pid)
            return False
        frozen = freeze_missing_fixings(position.product, env)
        if frozen is not position.product:
            position.product = frozen
        self._live[pid] = update_live(frozen.open_fixings(env), env.valuation_date, env.spot)
        return True

    def _day(self, portfolio, participants, fixings, events=()) -> RunFixingDay:
        dates = [portfolio.pricing_environments[pos.underlying].valuation_date for _, pos in participants]
        pending = 0.0
        for pid, pos in participants:
            hook = getattr(pos.product, "pending_pv", None)
            if pid in portfolio.positions and callable(hook):
                pending += float(pos.quantity) * float(hook(portfolio.pricing_environments[pos.underlying]))
        day = RunFixingDay(date=max(dates) if dates else None, fixings=tuple(fixings),
                           product_events=tuple(events), pending_pv=pending)
        self._days.append(day)
        return day

    def register(self, portfolio) -> RunFixingDay:
        """The run's first valuation (R5): fill strictly past open records once at its spot; remember the instant."""
        participants = self._participants(portfolio)
        envs = portfolio.pricing_environments
        for pid, position in participants:
            env = envs[position.underlying]
            if self._register_position(portfolio, pid, position, env):
                self._last.setdefault(position.underlying, (env.valuation_date, float(env.spot)))
        return self._day(portfolio, self._participants(portfolio), ())

    def advance(self, portfolio) -> RunFixingDay:
        """One path instant later (R3/R4/R6/R8/R9); a position first seen here is registered here (Ruling 8)."""
        participants = self._participants(portfolio)
        envs = portfolio.pricing_environments
        fixings: List[RecordedFixing] = []
        events: List[Tuple[str, float, Any]] = []
        for underlying in sorted({pos.underlying for _, pos in participants}):
            env = envs[underlying]
            if underlying in self._last:
                prev_date = self._last[underlying][0]
                require_date_instant(prev_date)
                require_date_instant(env.valuation_date)
                if pd.Timestamp(env.valuation_date) < pd.Timestamp(prev_date):
                    raise ValidationError(
                        f"the run clock cannot go back: {underlying} from {prev_date} to {env.valuation_date} (R9)")
        for pid, position in participants:
            env = envs[position.underlying]
            last = self._last.get(position.underlying)
            if pid not in self._live or last is None:
                self._register_position(portfolio, pid, position, env)
                continue
            prev_date, prev_spot = last
            live = self._live[pid]
            if pd.Timestamp(env.valuation_date) == pd.Timestamp(prev_date):
                # same instant: only its spot is refreshed (Ruling 9), including records that went live on it
                day = pd.Timestamp(env.valuation_date)
                self._live[pid] = {i: ((d, float(env.spot)) if pd.Timestamp(d) == day else (d, s))
                                   for i, (d, s) in live.items()}
                continue
            aged = position.product.with_time_elapsed(run_step_dt(prev_date, env))
            labels = aged.open_fixings(env)
            recorded, new = step_fixings(aged, env, prev_date=prev_date, prev_spot=prev_spot, run_fills={},
                                         live_spots={i: s for i, (_d, s) in live.items()})
            for index in sorted(new):
                fixings.append(RecordedFixing(position_id=pid, index=index, label=labels[index][1],
                                              outcome=new[index],
                                              spot=float(live[index][1]) if index in live else float(prev_spot),
                                              gap=index not in live))
            self._live[pid] = update_live(labels, env.valuation_date, env.spot, before=live, exclude=new)
            prev_events = getattr(position.product, "lifecycle_events", None)
            new_events = getattr(recorded, "lifecycle_events", None)
            if prev_events is not None and new_events is not None:
                events.extend((pid, float(position.quantity), ev) for ev in list(new_events)[len(prev_events):])
            if recorded is not position.product:
                position.product = recorded
        for _, position in participants:
            env = envs[position.underlying]
            self._last[position.underlying] = (env.valuation_date, float(env.spot))
        live_ids = {pid for pid, _ in participants}
        for pid in [p for p in self._live if p not in live_ids]:
            del self._live[pid]
        return self._day(portfolio, [(p, x) for p, x in participants if p in portfolio.positions],
                         fixings, events)
