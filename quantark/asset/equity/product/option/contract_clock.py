"""The contract clock: one rule for "the same instant" (spec 2026-11-23-contract-clock-design.md).

Layer 1 -- ``schedule_times`` -- is the one generator every factory uses for evenly spaced numeric schedules. It
computes ``(i + 1) * maturity / n``, which rounds once: whenever ``(i + 1) * maturity`` is exact (every maturity with a
short binary expansion), the result is the correctly rounded value of the exact rational, so two schedules that name
the same calendar instant hold the same float. The old ``(i + 1) / n * maturity`` rounded twice and left 105 such
instants across common tenors one ulp apart (11/12 * 1.0 vs 11/18 * 1.5).
"""
from __future__ import annotations

from math import isfinite
from typing import List

from quantark.util.exceptions import ValidationError

__all__ = ["schedule_times"]


def schedule_times(maturity: float, num_observations: int) -> List[float]:
    """Evenly spaced observation times ``(i + 1) * maturity / num_observations``, the last exactly ``maturity``.

    Args:
        maturity: The schedule's horizon in years (positive, finite).
        num_observations: How many observations (at least one).

    Returns:
        ``num_observations`` increasing times in years.

    Raises:
        ValidationError: On a non-positive or non-finite maturity, or fewer than one observation.
    """
    maturity = float(maturity)
    if not isfinite(maturity) or maturity <= 0.0:
        raise ValidationError(f"schedule_times: maturity must be positive and finite, got {maturity}")
    if isinstance(num_observations, bool) or int(num_observations) != num_observations or num_observations < 1:
        raise ValidationError(f"schedule_times: num_observations must be a positive integer, got {num_observations!r}")
    n = int(num_observations)
    times = [(i + 1) * maturity / n for i in range(n)]
    times[-1] = maturity
    return times
