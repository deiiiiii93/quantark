"""Shared assembly of the MC event statistics from recorded batches (unified event statistics spec AC13,
AC21, D18).

The autocallable MC engines record one batch of outcome sums per priced batch of ``price()`` (see
``_event_stats_recorder``). Every number of an :class:`EventStatistics` that is a path average is linear in
those sums, so one function maps a batch's sums (or their aggregate) to the statistics' fields; the batch-mean
standard errors apply the same map to each batch.

Pieces shared by the MC producers:

- :func:`knock_in_day_axis`: the trading days of a continuous knock-in window (spec AC13, D18);
- :func:`step_overlap`: each MC step's first-crossing mass spread over the trading days the step overlaps, in
  proportion to the overlap (spec D18): the cumulative crossing mass, linear between the step ends, read at
  each day's end;
- :func:`knock_in_rows` / :class:`KnockInLayout`: the per-date knock-in table on the product's KI schedule;
- :func:`batch_standard_errors` / :func:`ratio_standard_error`: standard errors from independent batch means
  (spec AC21), the ratio estimator by the delta method.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from quantark.util.calendar.day_counter import calculate_year_fraction
from quantark.util.numerical import Tolerance

__all__ = [
    "DEPENDENT_BATCHES",
    "FEW_KNOCK_IN_PATHS",
    "NO_KNOCK_IN",
    "SINGLE_BATCH",
    "STEP_OVERLAP",
    "KnockInLayout",
    "batch_standard_errors",
    "knock_in_day_axis",
    "knock_in_rows",
    "ratio_standard_error",
    "standard_errors_or_reason",
    "step_overlap",
]

#: ``unavailable`` reasons of the MC producers (spec Contracts, AC7, AC21)
SINGLE_BATCH = "undefined: single batch"
DEPENDENT_BATCHES = "undefined: deterministic QMC batches are not independent"
FEW_KNOCK_IN_PATHS = "undefined: fewer than two knock-in paths"
NO_KNOCK_IN = "undefined: P(knock in) = 0"
#: the continuous knock-in rows' interpolation scheme (spec D18)
STEP_OVERLAP = "step_overlap"

Field = Union[float, np.ndarray]


# --- the continuous knock-in table (spec AC13, D18) ------------------------------------------------------


def knock_in_day_axis(product, pricing_env, horizon: float) -> Tuple[np.ndarray, Optional[tuple]]:
    """End time (and date) of each trading day of the continuous knock-in window ``(0, horizon]``.

    A dated product (``ProductClock.DATE``) priced in an environment with a business calendar: one row per
    business day after the valuation date, at its time on the environment's axis, the last row the day the
    window ends (its time is ``horizon``). Otherwise (a numeric-clock contract, or no calendar to tell trading
    days): steps of ``1 / bus_days_in_year`` on the product's time axis, the last row at ``horizon``, no dates.
    An empty axis when the window ends at the valuation.
    """
    from quantark.asset.clock import ProductClock, product_clock

    horizon = float(horizon)
    calendar = getattr(pricing_env, "calendar", None)
    dated = product_clock(product) is ProductClock.DATE and calendar is not None
    if horizon <= Tolerance.ZERO:
        return np.zeros(0, dtype=float), (() if dated else None)
    if dated:
        valuation: datetime = pricing_env.valuation_date
        times: List[float] = []
        dates: List[datetime] = []
        day = valuation
        while True:
            day = day + timedelta(days=1)
            time = calculate_year_fraction(
                valuation, day, pricing_env.day_count_convention, pricing_env.bus_days_in_year,
                calendar=calendar,
            )
            if time >= horizon - Tolerance.ZERO:
                times.append(horizon)
                dates.append(day)
                break
            if calendar.is_business_day(day):
                times.append(float(time))
                dates.append(day)
        return np.asarray(times, dtype=float), tuple(dates)
    days_per_year = float(pricing_env.bus_days_in_year)
    count = int(math.ceil(horizon * days_per_year - Tolerance.ZERO))
    times = [k / days_per_year for k in range(1, count + 1) if k / days_per_year < horizon - Tolerance.ZERO]
    times.append(horizon)
    return np.asarray(times, dtype=float), None


def step_overlap(step_ends: np.ndarray, step_mass: np.ndarray, day_ends: np.ndarray) -> np.ndarray:
    """Mass per day: each step's mass spread over the days it overlaps in proportion to the overlap (D18).

    Step ``k`` spans ``(step_ends[k-1], step_ends[k]]`` (the first from 0); day ``d`` spans
    ``(day_ends[d-1], day_ends[d]]``. Spreading a step's mass uniformly over its time is the cumulative mass,
    linear between step ends, so day ``d`` holds ``C(day_ends[d]) - C(day_ends[d-1])``; mass of a step past
    the last day end is kept on the last day.
    """
    step_ends = np.asarray(step_ends, dtype=float)
    day_ends = np.asarray(day_ends, dtype=float)
    if day_ends.size == 0:
        return np.zeros(0, dtype=float)
    knots = np.concatenate([[0.0], step_ends])
    cumulative = np.concatenate([[0.0], np.cumsum(np.asarray(step_mass, dtype=float))])
    at_day_end = np.interp(day_ends, knots, cumulative)
    at_day_end[-1] = cumulative[-1]
    return np.diff(np.concatenate([[0.0], at_day_end]))


def _not_a_grid_time(boundaries: np.ndarray, grid: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(grid, boundaries), 0, grid.size - 1)
    nearest = np.minimum(np.abs(grid[idx] - boundaries), np.abs(grid[np.maximum(idx - 1, 0)] - boundaries))
    return nearest > Tolerance.ZERO


@dataclass(frozen=True)
class KnockInLayout:
    """The per-date knock-in table's rows for one (product, valuation), fixed before any path is read.

    ``kind``: ``"none"`` (no knock-in barrier), ``"carried"`` (knocked in before valuation), ``"discrete"``
    (rows = the product's KI observations) or ``"continuous"`` (rows = trading days, mass by step overlap).
    ``step_ends``: the MC steps of a continuous window (the recorded per-step counts align to them).
    """

    kind: str
    row_times: np.ndarray
    row_dates: Optional[tuple]
    step_ends: Optional[np.ndarray] = None

    @property
    def interpolated(self) -> np.ndarray:
        """Continuous rows whose day boundary is not an MC grid time (labelled ``step_overlap``)."""
        if self.kind != "continuous" or self.row_times.size == 0:
            return np.zeros(self.row_times.size, dtype=bool)
        grid = np.concatenate([[0.0], np.asarray(self.step_ends, dtype=float)])
        starts = np.concatenate([[0.0], self.row_times[:-1]])
        return _not_a_grid_time(self.row_times, grid) | _not_a_grid_time(starts, grid)


def knock_in_rows(
    layout: KnockInLayout,
    *,
    n: float,
    first_counts: np.ndarray,
    knocked_in_now: bool,
    knocked_out_now: bool,
    knock_in_count: float,
    clean_ko_times: np.ndarray,
    clean_ko_counts: np.ndarray,
) -> Tuple[np.ndarray, Optional[tuple], np.ndarray, np.ndarray, np.ndarray]:
    """``(times, dates, probability, survival, interpolated)`` of the per-date knock-in table (DESIGN §4.5).

    Args:
        layout: the table's rows.
        n: the path weight of the sums.
        first_counts: first knock-in before any knock-out per KI row (discrete) or per MC step (continuous).
        knocked_in_now: the knock-in is decided at valuation (carried, or a level breached now): one row at 0.
        knocked_out_now: the note is knocked out at valuation without a carried knock-in: the table is empty.
        knock_in_count: paths knocked in before any knock-out (the one row's mass when ``knocked_in_now``).
        clean_ko_times, clean_ko_counts: knock-outs with no knock-in before them, by time (survival).
    """
    empty = np.zeros(0, dtype=float)
    if layout.kind == "none" or (knocked_out_now and layout.kind != "carried"):
        return empty, None, empty, empty, np.zeros(0, dtype=bool)
    if layout.kind == "carried" or knocked_in_now:
        probability = np.array([knock_in_count / n], dtype=float)
        return (np.zeros(1), None, probability, 1.0 - probability, np.zeros(1, dtype=bool))
    times = layout.row_times
    if layout.kind == "continuous":
        probability = step_overlap(layout.step_ends, first_counts, times) / n
        interpolated = layout.interpolated
    else:
        probability = np.asarray(first_counts, dtype=float) / n
        interpolated = np.zeros(times.size, dtype=bool)
    clean_ko_times = np.asarray(clean_ko_times, dtype=float)
    order = np.argsort(clean_ko_times, kind="stable")
    cumulative_ko = np.concatenate([[0.0], np.cumsum(np.asarray(clean_ko_counts, dtype=float)[order])])
    # a knock-out at a row's own time is out by then (it wins the tie against a discrete knock-in)
    ko_by_row = cumulative_ko[np.searchsorted(clean_ko_times[order], times + Tolerance.ZERO, side="right")] / n
    survival = 1.0 - np.cumsum(probability) - ko_by_row
    return times, layout.row_dates, probability, survival, interpolated


# --- batch-mean standard errors (spec AC21) -----------------------------------------------------------------


def batch_standard_errors(
    batch_fields: Sequence[Mapping[str, Field]],
    weights: Sequence[float],
    estimate: Mapping[str, Field],
) -> Dict[str, Field]:
    """Standard error of each field from independent batch means.

    ``estimate`` is the path-weighted mean of ``batch_fields`` (weights = the batches' path counts). The ratio
    estimator over batches of unequal size: ``SE^2 = B / (B - 1) * sum_b (w_b / W)^2 (x_b - x)^2``, the
    textbook ``std(x_b) / sqrt(B)`` when the batches are equal. Needs ``B >= 2``.
    """
    batches = len(batch_fields)
    if batches < 2:
        raise ValueError("batch-mean standard errors need at least two batches")
    w = np.asarray(weights, dtype=float)
    share = w / w.sum()
    errors: Dict[str, Field] = {}
    for key, value in estimate.items():
        deviations = [np.asarray(fields[key], dtype=float) - np.asarray(value, dtype=float)
                      for fields in batch_fields]
        variance = batches / (batches - 1) * sum(s * s * d * d for s, d in zip(share, deviations))
        se = np.sqrt(variance)
        errors[key] = float(se) if np.ndim(value) == 0 else np.asarray(se, dtype=float)
    return errors


def ratio_standard_error(numerators: Sequence[float], denominators: Sequence[float]) -> float:
    """Delta-method standard error of ``sum(numerators) / sum(denominators)`` over independent batches:
    ``SE^2 = B / (B - 1) * sum_b (N_b - R D_b)^2 / (sum_b D_b)^2``."""
    num = np.asarray(numerators, dtype=float)
    den = np.asarray(denominators, dtype=float)
    batches = num.size
    if batches < 2:
        raise ValueError("batch-mean standard errors need at least two batches")
    total = float(den.sum())
    ratio = float(num.sum()) / total
    variance = batches / (batches - 1) * float(np.sum((num - ratio * den) ** 2)) / (total * total)
    return math.sqrt(variance)


def standard_errors_or_reason(
    records: Sequence[Mapping],
    *,
    independent: bool,
    fields_of: Callable[[Mapping], Mapping[str, Field]],
    estimate: Mapping[str, Field],
) -> Tuple[Optional[Dict[str, Field]], Optional[str]]:
    """``(standard_errors, None)`` from the batch records, or ``(None, reason)`` when a batch-mean error is
    undefined: one batch, or batches that are not independent draws (deterministic QMC)."""
    if len(records) < 2:
        return None, SINGLE_BATCH
    if not independent:
        return None, DEPENDENT_BATCHES
    batch_fields = [fields_of(record) for record in records]
    return batch_standard_errors(batch_fields, [float(record["n"]) for record in records], estimate), None
