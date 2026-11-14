"""The trading-day rows of a continuously monitored knock-in (unified event statistics spec AC13, D18).

A continuous knock-in's per-date table holds one row per trading day of the monitoring window, each the
first-crossing probability within that day. An engine whose steps are not trading days spreads each step's mass over
the days the step overlaps, in proportion to the overlap: the cumulative mass is linear between the step ends and read
at each day's end, and a row whose day boundary is not a step end is ``interpolated`` (scheme ``"step_overlap"``).

Shared by the MC producers (``mc/_mc_event_statistics.py``) and the PDE 1D forward adjoint pass
(``pde/core/statistics/adjoint.py``); engine-neutral, so a PDE module never imports the MC package.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np

from quantark.util.calendar.day_counter import calculate_year_fraction
from quantark.util.numerical import Tolerance

__all__ = ["STEP_OVERLAP", "at_valuation_dates", "knock_in_day_axis", "not_a_grid_time", "step_overlap"]

#: the continuous knock-in rows' interpolation scheme (spec D18)
STEP_OVERLAP = "step_overlap"


def at_valuation_dates(product, pricing_env) -> Optional[tuple]:
    """``(valuation_date,)`` on a dated note, else ``None``: the dates (and payment dates) of a knock-in row decided at
    valuation -- a discrete KI observed then with the spot through it, a carried knock-in, a continuous barrier
    already breached. One rule on every engine (CP8: a unified table does not differ by engine on dates)."""
    from quantark.asset.clock import ProductClock, product_clock

    valuation = getattr(pricing_env, "valuation_date", None)
    if valuation is not None and product_clock(product) is ProductClock.DATE:
        return (valuation,)
    return None


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


def not_a_grid_time(boundaries: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Per boundary: True when no grid time lies within ``Tolerance.ZERO`` of it."""
    idx = np.clip(np.searchsorted(grid, boundaries), 0, grid.size - 1)
    nearest = np.minimum(np.abs(grid[idx] - boundaries), np.abs(grid[np.maximum(idx - 1, 0)] - boundaries))
    return nearest > Tolerance.ZERO
