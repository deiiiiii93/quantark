"""Trading-day rows of a continuous knock-in on the QUAD recursions (unified event statistics spec AC13).

QUAD V1 and QUAD V2 both read their forward adjoint pass's continuous crossing mass per recursion interval and split
it over the trading days the interval overlaps by the interval's own Brownian first-passage law (scheme
``"interval_first_passage"``). This module holds the parts the two share:

* :func:`crossed_by` -- the drifted Brownian first-passage law of an interval, time-changed by its variance;
* :func:`split_interval_mass` -- each interval's crossing mass over its days (the engine supplies the law per
  interval), the mass decided at an interval's end on the day it ends;
* :func:`survival_after_rows` -- the knock-in table's survival column.

Engine-specific parts stay with each engine: how the variance fraction of a day boundary is measured (V1: linear in
time on a step with constant coefficients; V2: the environment's cumulative variance) and which law an interval has.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from quantark.util.exceptions import PricingError
from quantark.util.numerical import Tolerance

__all__ = ["crossed_by", "split_interval_mass", "survival_after_rows"]


def survival_after_rows(times, probability, ko_times, ko_probability) -> np.ndarray:
    """P(not yet knocked in and not knocked out after each row); a knock-out at a row's own time is out by then."""
    order = np.argsort(ko_times, kind="stable")
    cumulative_ko = np.concatenate([[0.0], np.cumsum(ko_probability[order])])
    ko_by_row = cumulative_ko[np.searchsorted(ko_times[order], times + Tolerance.ZERO, side="right")]
    return 1.0 - np.cumsum(probability) - ko_by_row


def crossed_by(law, fractions) -> np.ndarray:
    """``Σ w P_x(hit by a fraction of the interval's variance)`` per fraction: the drifted Brownian first-passage law
    of the reflection kernel (``continuous_readout``), time-changed by the interval's variance.

    ``law`` is ``((x, weights), mean, variance, barrier, reverse)``: log-spot nodes and their weights at the interval
    start, the interval's log drift and variance, the log barrier and the barrier side (``reverse``: hit from below).
    """
    from scipy.special import ndtr

    (x, weights), mean, variance, barrier, reverse = law
    fractions = np.atleast_1d(np.asarray(fractions, dtype=float))
    keep = weights != 0.0
    x, weights = x[keep], weights[keep]
    distance = (x - barrier) if reverse else (barrier - x)      # < 0 on the safe side
    safe = distance < 0.0
    through = float(np.sum(weights[~safe]))                     # already through the barrier: crossed at once
    d, w = distance[safe][:, None], weights[safe][:, None]
    out = np.zeros(fractions.size)
    positive = fractions > 0.0
    if np.any(positive):
        f = fractions[positive][None, :]
        m, v = mean * f, variance * f
        drift = -m if reverse else m
        sd = np.sqrt(v)
        with np.errstate(over="ignore", invalid="ignore"):
            hit = ndtr((d - drift) / sd) + np.exp(2 * drift * d / v) * ndtr((d + drift) / sd)
        out[positive] = np.sum(w * hit, axis=0) + through
    return out


def split_interval_mass(days: np.ndarray, t: np.ndarray, interval_p: np.ndarray, event_p: np.ndarray,
                        crossed_at: Callable[[int, np.ndarray], Optional[np.ndarray]], *, engine: str) -> np.ndarray:
    """Mass per trading day ``days`` (day ends) from the per-interval crossing mass ``interval_p`` and the mass
    ``event_p`` decided at each interval's end ``t`` (intervals ``(t[i-1], t[i]]``, the first from 0).

    Each interval's cumulative crossing mass is 0 up to its start and all of it from its end; at a day end strictly
    inside the interval it is the crossing times ``crossed_at(i, inside)[:-1] / crossed_at(i, inside)[-1]`` (the law's
    crossed fraction at each inner day end, the last entry the whole interval). Mass past the window's last day is kept
    on that day; decided mass goes to the day the interval ends in.

    Raises:
        PricingError: an interval with crossing mass whose law crosses nothing (``engine`` names the engine).
    """
    starts_t = np.concatenate([[0.0], t[:-1]])
    day_mass = np.zeros(days.size)
    for i in range(t.size):
        crossing, decided = float(interval_p[i]), float(event_p[i])
        if days.size == 0 or (crossing == 0.0 and decided == 0.0):
            continue
        s0, s1 = float(starts_t[i]), float(t[i])
        at_day_end = np.where(days >= s1 - Tolerance.ZERO, crossing, 0.0)
        at_day_end[-1] = crossing
        inside = np.flatnonzero((days > s0 + Tolerance.ZERO) & (days < s1 - Tolerance.ZERO))
        inside = inside[inside < days.size - 1]
        if crossing != 0.0 and inside.size:
            crossed = crossed_at(i, inside)
            if crossed[-1] == 0.0:
                raise PricingError(f"a {engine} continuous knock-in interval has crossing mass but no crossing law")
            at_day_end[inside] = crossing * crossed[:-1] / crossed[-1]
        share = np.diff(np.concatenate([[0.0], at_day_end]))
        if decided != 0.0:
            share[min(int(np.searchsorted(days, s1 - Tolerance.ZERO, side="left")), days.size - 1)] += decided
        day_mass += share
    return day_mass
