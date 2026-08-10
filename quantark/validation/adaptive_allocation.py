"""Estimate-blind batch allocation and precision-based stopping.

Used by the certification banking loop: a pilot measures per-cell batch SD and
cost, :func:`neyman_allocation` freezes where further batches go, and
:func:`precision_stop` halts on achieved *precision* (never on the estimate),
so the final fixed-confidence verdict needs no sequential-testing correction.

The blindness is structural rather than conventional: :class:`CellPrecision`
has no field an estimate could travel through, so nothing on the stopping path
can read one (spec gate S-G1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from scipy.stats import t as student_t

from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class CellPrecision:
    """Precision state of one certification cell. Deliberately estimate-free."""

    name: str
    n_batches: int
    batch_sd: float
    seconds_per_batch: float

    def __post_init__(self):
        if self.n_batches < 2:
            raise ValidationError(f"{self.name}: need at least 2 batches for an SD")
        if self.batch_sd < 0.0 or not math.isfinite(self.batch_sd):
            raise ValidationError(f"{self.name}: batch_sd must be finite and >= 0")
        if self.seconds_per_batch <= 0.0:
            raise ValidationError(f"{self.name}: seconds_per_batch must be > 0")


@dataclass(frozen=True)
class StopDecision:
    """Why banking stopped, and the precision it had reached when it did."""

    stop: bool
    trigger: Optional[str]  # "target-reached" | "budget-cap" | None
    projected_halfwidth: float

    def as_dict(self) -> dict:
        return {
            "stop": bool(self.stop),
            "trigger": self.trigger,
            "projected_halfwidth": float(self.projected_halfwidth),
        }


def projected_aggregate_halfwidth(
    cells: Sequence[CellPrecision], confidence: float = 0.975
) -> float:
    """t x SE of the equal-weight mean of cell means (schema-11 convention)."""
    if not cells:
        raise ValidationError("need at least one cell")
    k = len(cells)
    variance = sum((c.batch_sd**2) / c.n_batches for c in cells) / (k * k)
    degrees_of_freedom = sum(c.n_batches - 1 for c in cells)
    return float(student_t.ppf(confidence, degrees_of_freedom)) * math.sqrt(variance)


def neyman_allocation(
    cells: Sequence[CellPrecision],
    budget_seconds: float,
    min_batches: int = 16,
) -> dict:
    """Cost-weighted Neyman allocation: n_j proportional to sd_j / sqrt(cost_j).

    Returns the total batch count per cell (pilot batches included), floored at
    ``min_batches`` and fitted inside ``budget_seconds``.
    """
    if not cells:
        raise ValidationError("need at least one cell")
    if budget_seconds <= 0.0:
        raise ValidationError("budget_seconds must be > 0")

    # Minimizing sum(sd_j^2 / n_j) subject to sum(n_j * cost_j) = budget gives
    # n_j = alpha * sd_j / sqrt(cost_j), with alpha fixed by the budget:
    # alpha = budget / sum_i(sd_i * sqrt(cost_i)).
    shares = {c.name: c.batch_sd / math.sqrt(c.seconds_per_batch) for c in cells}
    budget_normalizer = sum(c.batch_sd * math.sqrt(c.seconds_per_batch) for c in cells)
    if budget_normalizer <= 0.0:
        # Every cell reports zero SD: more batches reduce nothing. Keep the floor.
        return {c.name: max(min_batches, c.n_batches) for c in cells}

    alpha = budget_seconds / budget_normalizer
    allocation = {}
    for c in cells:
        allocation[c.name] = max(min_batches, int(alpha * shares[c.name]))

    spent = sum(allocation[c.name] * c.seconds_per_batch for c in cells)
    if spent > budget_seconds:
        scale = budget_seconds / spent
        for c in cells:
            allocation[c.name] = max(min_batches, int(allocation[c.name] * scale))
    return allocation


def precision_stop(
    cells: Sequence[CellPrecision],
    target_halfwidth: float,
    elapsed_seconds: float,
    budget_seconds: float,
    confidence: float = 0.975,
) -> StopDecision:
    """Stop on achieved precision or an exhausted budget; never on the estimate."""
    if target_halfwidth <= 0.0:
        raise ValidationError("target_halfwidth must be > 0")
    halfwidth = projected_aggregate_halfwidth(cells, confidence=confidence)
    if halfwidth <= target_halfwidth:
        return StopDecision(
            stop=True, trigger="target-reached", projected_halfwidth=halfwidth
        )
    if elapsed_seconds >= budget_seconds:
        return StopDecision(
            stop=True, trigger="budget-cap", projected_halfwidth=halfwidth
        )
    return StopDecision(stop=False, trigger=None, projected_halfwidth=halfwidth)
