"""Hand-built ``EventDistribution`` on one observation grid (unified event statistics T6).

Most leg tests describe a parent whose array streams all share one observation grid and whose survival
vector has a leading 1 at the valuation; ``on_grid`` builds that distribution on the per-stream axes.
"""
from __future__ import annotations

import numpy as np

from quantark.cashleg.event_distribution import EventDistribution
from quantark.util.numerical import Tolerance, almost_equal


def on_grid(*, times, probabilities, survival_probability, **kwargs) -> EventDistribution:
    """Every array stream on ``times``; survival given on ``times`` with a leading valuation value (1).

    With an observation at 0 the leading value is dropped: the survival at 0 is that observation's.
    """
    times = np.asarray(times, dtype=float)
    survival = np.asarray(survival_probability, dtype=float)
    starts_at_zero = times.size > 0 and almost_equal(float(times[0]), 0.0, tol=Tolerance.PROBABILITY)
    if survival.size == times.size + 1:
        survival_times = times if starts_at_zero else np.r_[0.0, times]
        survival = survival[1:] if starts_at_zero else survival
    elif survival.size == times.size and starts_at_zero:
        survival_times = times
    else:
        raise ValueError(f"{survival.size} survival values for {times.size} observation times")
    stream_times = {event: times for event, p in probabilities.items() if isinstance(p, np.ndarray)}
    return EventDistribution(probabilities=probabilities, stream_times=stream_times, survival_times=survival_times,
                             survival_probability=survival, **kwargs)
