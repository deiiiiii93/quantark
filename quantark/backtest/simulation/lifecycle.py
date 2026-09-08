"""The snowball lifecycle over a batch of paths (spec 4.2, 8.4).

Every rule here is ``AutocallableLifecycleTracker``'s, applied to arrays
instead of one product at a time, and the days each observation fires on
come from the tracker itself (``resolve_calendar_schedule``).  Where the
tracker returns early -- a knock-out ends the day's observation before any
knock-in is considered -- this keeps a mask to the same effect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from quantark.asset.equity.lifecycle.autocallable import CalendarSchedule
from quantark.util.exceptions import ValidationError

PayoffFn = Callable[[int, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class LifecycleRecord:
    """One event, for the run's event log."""

    product: int
    path: int
    day: int
    event: str            # "knock_in" | "knock_out" | "maturity" | "settlement"
    index: int            # observation index, -1 where there is none
    spot: float
    barrier: float
    cashflow: float


def _barrier_hit(spot: np.ndarray, barrier: float, is_reverse: bool, *, is_ko: bool) -> np.ndarray:
    """``AutocallableLifecycleTracker._barrier_hit``, vectorised (inclusive both ways)."""
    if not np.isfinite(barrier):
        return np.zeros(spot.shape, dtype=bool)
    if is_ko:
        return spot <= barrier if is_reverse else spot >= barrier
    return spot >= barrier if is_reverse else spot <= barrier


class VectorLifecycle:
    """Per-product, per-path lifecycle state advanced one day at a time."""

    def __init__(
        self, schedules: Sequence[CalendarSchedule], quantities: np.ndarray, n_paths: int
    ) -> None:
        if len(schedules) != len(quantities):
            raise ValidationError("one schedule per product is required")
        if n_paths < 1:
            raise ValidationError("n_paths must be positive")
        self.schedules = list(schedules)
        self.quantities = np.asarray(quantities, dtype=float)
        self.n_products = len(self.schedules)
        self.n_paths = int(n_paths)
        shape = (self.n_products, self.n_paths)
        self.alive = np.ones(shape, dtype=bool)
        self.knocked_in = np.zeros(shape, dtype=bool)
        self.knocked_out = np.zeros(shape, dtype=bool)
        self.matured = np.zeros(shape, dtype=bool)
        self.settled = np.zeros(shape, dtype=bool)
        self.pending = np.zeros(shape)
        self.realized = np.zeros(shape)
        self.terminal_day = np.full(shape, -1, dtype=np.int64)
        self.settlement_day = np.full(shape, -1, dtype=np.int64)
        self.ko_index = np.full(shape, -1, dtype=np.int64)
        self.observed_ko = [
            np.zeros((self.n_paths, s.ko_due_day.size), dtype=bool) for s in self.schedules
        ]
        self.observed_ki = [
            np.zeros((self.n_paths, s.ki_due_day.size), dtype=bool) for s in self.schedules
        ]

    def step(self, day_index: int, spot: np.ndarray, payoff_fn: PayoffFn) -> List[LifecycleRecord]:
        """Advance one day; returns the events that fired.

        ``payoff_fn(product_index, path_indices, knocked_in)`` supplies the
        maturity payoff per unit product for the paths that mature today.
        """
        spot = np.asarray(spot, dtype=float)
        if spot.shape != (self.n_paths,):
            raise ValidationError(f"spot must have shape {(self.n_paths,)}, got {spot.shape}")
        records: List[LifecycleRecord] = []
        for p, schedule in enumerate(self.schedules):
            # ``today_ko`` is per product: one product knocking out must not
            # silence another product's knock-in on the same path.
            ko_records, today_ko = self._knock_out(p, schedule, day_index, spot)
            records += ko_records
            records += self._knock_in(p, schedule, day_index, spot, today_ko)
            records += self._maturity(p, schedule, day_index, spot, payoff_fn)
            records += self._settle(p, day_index)
        return records

    # -- the day's three blocks, in the tracker's order -----------------

    def _knock_out(self, p, schedule, d, spot) -> Tuple[List[LifecycleRecord], np.ndarray]:
        records: List[LifecycleRecord] = []
        today_ko = np.zeros(self.n_paths, dtype=bool)
        # A knocked-in path skips the KO block entirely under this flag, so
        # its indices are not marked observed either.
        eligible_block = self.alive[p] & ~(
            self.knocked_in[p] if schedule.disable_ko_after_ki else np.zeros(self.n_paths, bool)
        )
        for idx in np.flatnonzero(schedule.ko_due_day == d):
            idx = int(idx)
            due = eligible_block & self.alive[p] & ~self.observed_ko[p][:, idx]
            self.observed_ko[p][due, idx] = True     # observed whether or not it hits
            hit = due & _barrier_hit(spot, float(schedule.ko_barrier[idx]),
                                     schedule.is_reverse, is_ko=True)
            if not hit.any():
                continue
            cashflow = float(self.quantities[p]) * float(schedule.ko_payoff[idx])
            self.knocked_out[p][hit] = True
            self.alive[p][hit] = False
            self.ko_index[p][hit] = idx
            self.terminal_day[p][hit] = d
            settle_day = int(schedule.ko_settlement_day[idx])
            self.settlement_day[p][hit] = settle_day
            self.pending[p][hit] = cashflow
            today_ko |= hit
            for i in np.flatnonzero(hit):
                records.append(LifecycleRecord(p, int(i), d, "knock_out", idx, float(spot[i]),
                                               float(schedule.ko_barrier[idx]), cashflow))
        return records, today_ko

    def _knock_in(self, p, schedule, d, spot, today_ko: np.ndarray) -> List[LifecycleRecord]:
        records: List[LifecycleRecord] = []
        # ``observe`` returns as soon as a knock-out fires, so a path that
        # knocked out today never reaches its knock-in test.
        candidates = self.alive[p] & ~self.knocked_in[p] & ~today_ko
        if schedule.ki_continuous:
            barrier = float(schedule.ki_continuous_barrier)
            hit = candidates & _barrier_hit(spot, barrier, schedule.is_reverse, is_ko=False)
            self.knocked_in[p][hit] = True
            for i in np.flatnonzero(hit):
                records.append(LifecycleRecord(p, int(i), d, "knock_in", -1, float(spot[i]),
                                               barrier, 0.0))
            return records
        for idx in np.flatnonzero(schedule.ki_due_day == d):
            idx = int(idx)
            due = self.alive[p] & ~today_ko & ~self.observed_ki[p][:, idx]
            self.observed_ki[p][due, idx] = True
            barrier = float(schedule.ki_barrier[idx])
            hit = due & ~self.knocked_in[p] & _barrier_hit(spot, barrier, schedule.is_reverse, is_ko=False)
            self.knocked_in[p][hit] = True
            for i in np.flatnonzero(hit):
                records.append(LifecycleRecord(p, int(i), d, "knock_in", idx, float(spot[i]),
                                               barrier, 0.0))
        return records

    def _maturity(self, p, schedule, d, spot, payoff_fn) -> List[LifecycleRecord]:
        if schedule.terminal_due_day != d:
            return []
        due = np.flatnonzero(self.alive[p])
        if due.size == 0:
            return []
        payoffs = np.asarray(payoff_fn(p, due, self.knocked_in[p][due]), dtype=float)
        cashflows = float(self.quantities[p]) * payoffs
        self.matured[p][due] = True
        self.alive[p][due] = False
        self.terminal_day[p][due] = d
        self.settlement_day[p][due] = int(schedule.terminal_settlement_day)
        self.pending[p][due] = cashflows
        return [
            LifecycleRecord(p, int(i), d, "maturity", -1, float(spot[i]), float("nan"), float(c))
            for i, c in zip(due, cashflows)
        ]

    def _settle(self, p, d) -> List[LifecycleRecord]:
        # A terminal cashflow counts as paid from the first day on or after
        # its payment day (``ProductReplay.settle_pending_if_due``); before
        # that it sits in ``pending``.
        landing = (
            ~self.settled[p] & (self.settlement_day[p] >= 0) & (self.settlement_day[p] <= d)
            & (self.terminal_day[p] >= 0)
        )
        if not landing.any():
            return []
        amounts = self.pending[p][landing]
        self.realized[p][landing] += amounts
        self.pending[p][landing] = 0.0
        self.settled[p][landing] = True
        return [
            LifecycleRecord(p, int(i), d, "settlement", -1, float("nan"), float("nan"), float(a))
            for i, a in zip(np.flatnonzero(landing), amounts)
        ]

    # -- aggregation ---------------------------------------------------

    def book_flags(self) -> Dict[str, np.ndarray]:
        """The replay's book-level reduction over products, per path."""
        return {
            "alive": self.alive.any(axis=0),
            "knocked_in": self.knocked_in.any(axis=0),
            "knocked_out": self.knocked_out.all(axis=0),
            "matured": self.matured.all(axis=0),
            "settled": self.settled.all(axis=0),
        }
