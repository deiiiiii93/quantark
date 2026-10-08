"""The KO-reset snowball lifecycle over a batch of paths (spec 2026-10-08 KO-reset lifecycle, section 6).

Every rule is ``KOResetLifecycleTracker``'s applied to arrays, on the days its ``resolve_calendar_schedule``
names: a schedule-1 knock-out ends the path's day (R3); a knock-in is tested only while schedule 1 is live (R2)
and switches the path to schedule 2 from that close, inclusive (R1), consuming schedule-2 records that fell due
on an earlier close; a path not knocked in at schedule 1's end matures there on V0; a knocked-in path matures on
V1 at its schedule 2's end -- its own end under REBASED, placed by its knock-in day.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from quantark.asset.equity.lifecycle.autocallable import CalendarSchedule
from quantark.asset.equity.lifecycle.ko_reset import KOResetCalendarSchedule, SchedulePlacement

from .lifecycle import LifecycleRecord, VectorLifecycle, _barrier_hit

__all__ = ["KOResetVectorLifecycle"]


class KOResetVectorLifecycle(VectorLifecycle):
    """``VectorLifecycle`` with the KO-reset rules for every product whose schedule is a ``KOResetCalendarSchedule``.

    ``payoff_fn(product, path_indices, knocked_in, tenors)`` for those products: V0 (``knocked_in`` False) on
    schedule 1's tenor, V1 on the path's knocked-in contract tenor.
    """

    def __init__(self, schedules: Sequence[CalendarSchedule], quantities: np.ndarray, n_paths: int) -> None:
        super().__init__(schedules, quantities, n_paths)
        self.knock_in_day = np.full((self.n_products, self.n_paths), -1, dtype=np.int64)
        self.observed_post: Dict[int, np.ndarray] = {}
        for p, s in enumerate(self.schedules):
            if isinstance(s, KOResetCalendarSchedule):
                count = (s.absolute.ko_due_day.size if not s.rebased
                         else self._post_count(s))
                self.observed_post[p] = np.zeros((self.n_paths, count), dtype=bool)

    @staticmethod
    def _post_count(schedule: KOResetCalendarSchedule) -> int:
        days = [int(d) for d in schedule.ki_due_day if d >= 0]
        return schedule.place(days[0]).ko_due_day.size if days else 0

    def _step_product(self, p, schedule, d, spot, payoff_fn) -> List[LifecycleRecord]:
        if not isinstance(schedule, KOResetCalendarSchedule):
            return super()._step_product(p, schedule, d, spot, payoff_fn)
        records: List[LifecycleRecord] = []
        # schedule 1 (R3: a knock-out ends the path's day before any knock-in)
        ko_records, today_ko = self._knock_out(p, schedule, d, spot,
                                               eligible=self.alive[p] & ~self.knocked_in[p])
        records += ko_records
        # the knock-in, only while schedule 1 is live (R2)
        blocked = today_ko.copy()
        if 0 <= schedule.pre_due_day < d:
            blocked[:] = True
        before = self.knocked_in[p].copy()
        records += self._knock_in(p, schedule, d, spot, blocked)
        fresh = self.knocked_in[p] & ~before
        if fresh.any():
            self.knock_in_day[p][fresh] = d
            self._ensure_post(p, schedule, d)
            due = schedule.placement(d).ko_due_day
            consumed = (due >= 0) & (due < d)            # fell due on an earlier close: never tested
            if consumed.any():
                rows = np.flatnonzero(fresh)
                self.observed_post[p][np.ix_(rows, np.flatnonzero(consumed))] = True
        # schedule 2, the knock-in close included (R1)
        if not schedule.disable_ko_after_ki:
            records += self._knock_out_post(p, schedule, d, spot)
        records += self._pre_maturity(p, schedule, d, spot, payoff_fn)
        records += self._final_maturity(p, schedule, d, spot, payoff_fn)
        records += self._settle(p, schedule, d)
        return records

    def _ensure_post(self, p, schedule, d) -> None:
        count = schedule.placement(d).ko_due_day.size
        if self.observed_post[p].shape[1] != count:
            raise ValueError(f"schedule 2 of product {p} has {count} records on day {d}, "
                             f"{self.observed_post[p].shape[1]} elsewhere")

    def _groups(self, p, schedule, mask):
        """(paths, placement) per knock-in day among ``mask`` (one group for ABSOLUTE)."""
        if not mask.any():
            return []
        if not schedule.rebased:
            return [(mask, schedule.absolute)]
        return [(mask & (self.knock_in_day[p] == k), schedule.place(int(k)))
                for k in np.unique(self.knock_in_day[p][mask])]

    def _knock_out_post(self, p, schedule, d, spot) -> List[LifecycleRecord]:
        records: List[LifecycleRecord] = []
        for group, placement in self._groups(p, schedule, self.alive[p] & self.knocked_in[p]):
            for idx in np.flatnonzero(placement.ko_due_day == d):
                idx = int(idx)
                due = group & self.alive[p] & ~self.observed_post[p][:, idx]
                self.observed_post[p][due, idx] = True
                barrier = float(placement.ko_barrier[idx])
                hit = due & _barrier_hit(spot, barrier, schedule.is_reverse, is_ko=True)
                if not hit.any():
                    continue
                cashflow = float(self.quantities[p]) * float(placement.ko_payoff[idx])
                self._terminate(p, schedule, d, hit, cashflow,
                                settle_day=int(placement.ko_settlement_day[idx]),
                                delay=float(placement.ko_settlement_delay[idx]), knocked_out=True)
                self.ko_index[p][hit] = idx
                records += [LifecycleRecord(p, int(i), d, "knock_out", idx, float(spot[i]), barrier, cashflow)
                            for i in np.flatnonzero(hit)]
        return records

    def _book_maturities(self, p, schedule, d, spot, payoff_fn, paths, knocked_in, tenor, settle_day,
                         delay) -> List[LifecycleRecord]:
        if paths.size == 0:
            return []
        payoffs = np.asarray(payoff_fn(p, paths, np.full(paths.size, knocked_in), np.full(paths.size, tenor)),
                             dtype=float)
        cashflows = float(self.quantities[p]) * payoffs
        for i, c in zip(paths, cashflows):
            one = np.zeros(self.n_paths, dtype=bool)
            one[int(i)] = True
            self._terminate(p, schedule, d, one, float(c), settle_day=int(settle_day), delay=float(delay),
                            knocked_out=False)
        return [LifecycleRecord(p, int(i), d, "maturity", -1, float(spot[i]), float("nan"), float(c))
                for i, c in zip(paths, cashflows)]

    def _pre_maturity(self, p, schedule, d, spot, payoff_fn) -> List[LifecycleRecord]:
        """R2: not knocked in at schedule 1's end, matured there on V0."""
        if schedule.pre_due_day != d:
            return []
        paths = np.flatnonzero(self.alive[p] & ~self.knocked_in[p])
        return self._book_maturities(p, schedule, d, spot, payoff_fn, paths, False, schedule.pre_tenor,
                                     schedule.pre_settlement_day, schedule.pre_settlement_delay)

    def _final_maturity(self, p, schedule, d, spot, payoff_fn) -> List[LifecycleRecord]:
        """Knocked in: V1 at the knocked-in contract's end (each knock-in day's own under REBASED)."""
        records: List[LifecycleRecord] = []
        for group, placement in self._groups(p, schedule, self.alive[p] & self.knocked_in[p]):
            if placement.final_due_day != d:
                continue
            records += self._book_maturities(p, schedule, d, spot, payoff_fn, np.flatnonzero(group & self.alive[p]),
                                             True, placement.final_tenor, placement.final_settlement_day,
                                             placement.final_settlement_delay)
        return records
