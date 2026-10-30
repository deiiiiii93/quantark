"""Density of the continuous knock-in monitoring grid shared by the snowball and phoenix MC engines.

A continuously monitored knock-in barrier is checked on an equally spaced node set, one node per business
day by default, with a Brownian-bridge crossing probability between neighbouring nodes. The bridge treats
each interval as Brownian motion with the variance the path accumulated over it; under a spot-dependent or
stochastic volatility that law is wrong inside the interval, and the error does not shrink with SDE
substeps because the bridge interval stays one day long. ``ki_bridge_nodes_per_day`` (opt-in) multiplies
the node density so the bridge intervals themselves shrink. Evidence:
docs/research/mc_ki_bridge_refinement/README.md.
"""

from __future__ import annotations

import numpy as np

from quantark.util.exceptions import ValidationError


def validate_ki_bridge_nodes_per_day(value) -> int:
    """Return ``value`` as an int >= 1, or raise ValidationError (bool and non-integers are refused)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValidationError(
            f"ki_bridge_nodes_per_day must be a positive integer, got {value!r}"
        )
    return int(value)


def continuous_ki_step_count(bus_days_in_year: float, horizon: float, nodes_per_day: int) -> int:
    """Number of continuous-KI monitoring intervals over ``horizon`` years.

    One per business day plus one at ``nodes_per_day == 1`` -- the historical expression, kept verbatim so
    the default grid is unchanged bit for bit -- and ``nodes_per_day`` times denser otherwise.
    """
    if nodes_per_day == 1:
        return int(bus_days_in_year * horizon) + 1
    return int(bus_days_in_year * nodes_per_day * horizon) + 1
