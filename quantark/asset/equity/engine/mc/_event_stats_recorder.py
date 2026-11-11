"""Event statistics recorded on ``price()``'s own paths (unified event statistics spec AC2, D14, D23).

The autocallable MC engines produce their event statistics by running their own ``price()`` under a
recording context. Every payoff-kernel call that prices a batch of paths deposits ONE batch record: plain
per-batch sums over its paths (path count, per-row knock-out counts and cash, coupon counts and cash,
first knock-in counts, maturity-branch counts and cash, discounted and undiscounted). The statistics are
the path-count-weighted aggregate of those records, so they describe exactly the paths, seeds, bridge
seeds, batches and estimator of the price, on every route (serial, Dask batches, adaptive RQMC, the
conditional-moment estimators), and the value ``price()`` returned is the statistics' ``pv``.

Recording never touches the price arithmetic: the kernels compute their records from arrays they have
already finished, and only while a recorder is active.

Batches priced in another process (a Dask process or distributed scheduler) cannot deposit into the
caller's recorder: a batch function opens :meth:`EventStatsRecordingMixin._stats_batch_capture`, which
collects the records its kernel deposits in that thread, and returns them inside its own result under
:data:`STATS_RECORDS_KEY`; the caller deposits them in batch order. The recorder therefore holds only
plain picklable data.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "NON_ADDITIVE_KEYS",
    "STATS_RECORDS_KEY",
    "EventStatsRecorder",
    "EventStatsRecordingMixin",
    "aggregate_records",
    "grouped_sums",
    "row_sums",
]

#: the key a batch function's result carries its captured records under (Dask batch route)
STATS_RECORDS_KEY = "_event_stats_records"
#: record fields that describe the batch's layout, the same in every batch (kept from the first record)
NON_ADDITIVE_KEYS = ("ki_row_times", "ki_breached_at_valuation")


class EventStatsRecorder:
    """Batch records deposited by the payoff kernels during one recorded ``price()``.

    ``records`` keeps the deposit order (batch order on every route). ``expired`` is set when ``price()``
    took its zero-time-to-maturity shortcut, which prices no paths (the statistics are then ``None``,
    the documented expired rule). Per-thread capture stacks and suppression depths are plain dicts keyed
    by thread id, so the object pickles into a worker process unchanged.
    """

    def __init__(self) -> None:
        self.records: List[dict] = []
        self.expired = False
        self._captures: Dict[int, List[List[dict]]] = {}
        self._suppressed: Dict[int, int] = {}

    def active(self) -> bool:
        """Whether a kernel running in this thread should deposit."""
        return not self._suppressed.get(threading.get_ident(), 0)

    def deposit(self, record: dict) -> None:
        """Keep ``record``: in this thread's open capture, else in ``records``."""
        stack = self._captures.get(threading.get_ident())
        if stack:
            stack[-1].append(record)
        else:
            self.records.append(record)

    @contextmanager
    def capture(self):
        """Collect the records deposited in this thread inside the block into the yielded list."""
        tid = threading.get_ident()
        captured: List[dict] = []
        self._captures.setdefault(tid, []).append(captured)
        try:
            yield captured
        finally:
            stack = self._captures.get(tid)
            if stack:
                stack.pop()
                if not stack:
                    self._captures.pop(tid, None)

    @contextmanager
    def suppressed(self):
        """No deposits from this thread inside the block (a kernel call that prices no batch of the price:
        a control variate's proxy paths, the RQMC result assembly's extra statistics batch)."""
        tid = threading.get_ident()
        self._suppressed[tid] = self._suppressed.get(tid, 0) + 1
        try:
            yield
        finally:
            depth = self._suppressed.get(tid, 1) - 1
            if depth:
                self._suppressed[tid] = depth
            else:
                self._suppressed.pop(tid, None)


class EventStatsRecordingMixin:
    """The recording seam shared by the autocallable MC engines."""

    #: the recorder of a running ``calculate_event_stats``; None on every other call
    _stats_recorder: Optional[EventStatsRecorder] = None
    #: the batch records of the last recorded price, in batch order (landing 2's batch-mean errors)
    _last_stats_records: tuple = ()

    @contextmanager
    def _recording_event_stats(self):
        """Run ``price()`` inside the block with a fresh recorder; never left behind."""
        recorder = EventStatsRecorder()
        self._stats_recorder = recorder
        try:
            yield recorder
        finally:
            self._stats_recorder = None
            self._last_stats_records = tuple(recorder.records)

    def _stats_recording(self) -> bool:
        """Whether a kernel called now (in this thread) must deposit its batch record."""
        recorder = self._stats_recorder
        return recorder is not None and recorder.active()

    def _record_stats_batch(self, record: dict) -> None:
        recorder = self._stats_recorder
        if recorder is not None and recorder.active():
            recorder.deposit(record)

    @contextmanager
    def _without_stats_recording(self):
        recorder = self._stats_recorder
        if recorder is None:
            yield
            return
        with recorder.suppressed():
            yield

    @contextmanager
    def _stats_batch_capture(self):
        """A batch function's capture: the yielded list (None when not recording) receives the records its
        kernel deposits; the batch returns it under ``STATS_RECORDS_KEY``."""
        recorder = self._stats_recorder
        if recorder is None:
            yield None
            return
        with recorder.capture() as captured:
            yield captured

    def _deposit_captured_stats(self, batch_results: Iterable[Mapping]) -> None:
        """Deposit the records each batch result carries, in batch order (the Dask reducer's callback)."""
        recorder = self._stats_recorder
        if recorder is None:
            return
        for result in batch_results:
            for record in result.get(STATS_RECORDS_KEY, ()):
                recorder.records.append(record)

    def _note_stats_expired(self) -> None:
        """``price()`` decided the claim on the spot at zero time to maturity (no batch is priced)."""
        recorder = self._stats_recorder
        if recorder is not None:
            recorder.expired = True


def row_sums(rows: np.ndarray, n_rows: int, mask: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-row sums over the paths in ``mask``: counts, or ``weights`` summed (``rows`` the path's row)."""
    selected = np.asarray(rows)[mask]
    if n_rows == 0:
        return np.zeros(0, dtype=float)
    if weights is None:
        return np.bincount(selected, minlength=n_rows).astype(float)[:n_rows]
    return np.bincount(selected, weights=np.asarray(weights, dtype=float)[mask], minlength=n_rows)[:n_rows]


def grouped_sums(
    tag: str,
    mask: np.ndarray,
    determination: np.ndarray,
    payment: np.ndarray,
    discounted: np.ndarray,
    cash: np.ndarray,
) -> Dict[tuple, List[float]]:
    """``{(tag, determination, payment): [discounted sum, cash sum, path count]}`` over the paths in ``mask``
    (a ledger of path-dependent cash, grouped by its determination and payment times)."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return {}
    pairs = np.column_stack((np.asarray(determination, dtype=float)[idx], np.asarray(payment, dtype=float)[idx]))
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    disc = np.bincount(inverse, weights=np.asarray(discounted, dtype=float)[idx], minlength=len(unique))
    paid = np.bincount(inverse, weights=np.asarray(cash, dtype=float)[idx], minlength=len(unique))
    count = np.bincount(inverse, minlength=len(unique)).astype(float)
    return {
        (tag, float(det), float(pay)): [float(d), float(c), float(k)]
        for (det, pay), d, c, k in zip(unique, disc, paid, count)
    }


def aggregate_records(records: Sequence[Mapping]) -> dict:
    """Sum the additive fields of batch records (scalars and aligned arrays); keep the first record's
    non-additive fields (row times, flags, kind). A field missing from any record is left out."""
    if not records:
        return {}
    first = records[0]
    totals: dict = {}
    for key, value in first.items():
        if key in ("kind", "ledger") or key in NON_ADDITIVE_KEYS:
            continue
        if not all(key in record and record[key] is not None for record in records):
            continue
        if isinstance(value, np.ndarray):
            total = np.zeros_like(np.asarray(value, dtype=float))
            for record in records:
                total = total + np.asarray(record[key], dtype=float)
        else:
            total = 0.0
            for record in records:
                total += float(record[key])
        totals[key] = total
    totals["n"] = int(sum(int(record["n"]) for record in records))
    totals["kind"] = tuple(record.get("kind") for record in records)
    for key in NON_ADDITIVE_KEYS:
        if key in first:
            totals[key] = first[key]
    if any("ledger" in record for record in records):
        ledger: Dict[tuple, List[float]] = {}
        for record in records:
            for key, sums in record.get("ledger", {}).items():
                slot = ledger.setdefault(key, [0.0] * len(sums))
                for i, value in enumerate(sums):
                    slot[i] += float(value)
        totals["ledger"] = ledger
    return totals
