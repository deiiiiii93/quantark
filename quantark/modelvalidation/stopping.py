"""Gate-driven sequential stopping for the stochastic reference arm.

Sampling stops when the benchmark is sharp enough for the gate that will judge
it -- not at a fixed batch count. The budget is a fraction of the *cell bound*,
so the reference is only as precise as the decision requires, and no more.

Two rules keep this sound:

* every quantity must meet the budget (the noisiest one sets the pace), and
* the decision depends only on the batch count and the current standard errors,
  never on how close the candidate happens to look -- stopping when the answer
  looks good is how a sequential procedure turns into a biased one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.study import EconomicScale, GateBounds, SamplingPolicy


class StopReason(str, Enum):
    """Why sampling stopped, or why it did not."""

    #: Below ``min_batches``; the standard error is not yet meaningful.
    BELOW_MIN_BATCHES = "below_min_batches"
    #: Still sampling: the budget is reachable but not yet met.
    CONTINUE = "continue"
    #: Every quantity reached the standard-error budget.
    SE_BUDGET_MET = "se_budget_met"
    #: Hit the hard cap without meeting the budget; verdicts will be UNRESOLVED.
    MAX_BATCHES = "max_batches"


@dataclass(frozen=True)
class StopDecision:
    """Whether to stop sampling, and why."""

    stop: bool
    reason: StopReason
    batches: int


def should_stop(
    std_errors_raw: Mapping[str, float],
    batches: int,
    scale: EconomicScale,
    bounds: GateBounds,
    policy: SamplingPolicy,
    budgets_c: Optional[Mapping[str, float]] = None,
) -> StopDecision:
    """Decide whether the reference arm has sampled enough.

    Args:
        std_errors_raw: Current standard error per quantity, engine units.
        batches: Batches completed so far.
        scale: Raw-to-economic converter.
        bounds: Study bounds (supplies the standard-error budget).
        policy: Sampling policy (supplies min/max batch limits).
        budgets_c: Schema 2: each targeted quantity's own cell budget in
            economic units. A quantity absent from it (an untargeted proxy)
            never drives sampling. ``None`` keeps schema 1's single bound.

    Raises:
        ValidationError: empty ``std_errors_raw`` or a negative batch count.
    """
    if not std_errors_raw:
        raise ValidationError("should_stop requires at least one quantity")
    if batches < 0:
        raise ValidationError(f"batches must be non-negative, got {batches}")

    if batches < policy.min_batches:
        return StopDecision(
            stop=False, reason=StopReason.BELOW_MIN_BATCHES, batches=batches
        )

    budget_met = True
    for quantity, raw_se in std_errors_raw.items():
        if budgets_c is not None and quantity not in budgets_c:
            continue                                  # schema 2: an untargeted quantity never drives sampling
        # Schema 2 gate values are fractions of the cell's own budget, so the SE budget is that budget scaled
        # exactly as the gate scales it; with budgets_c None this is schema 1's arithmetic (x * 1.0 == x).
        budget_c = bounds.se_budget_fraction * bounds.cell * (1.0 if budgets_c is None else budgets_c[quantity])
        if not math.isfinite(raw_se) or abs(scale.to_economic(quantity, raw_se)) > budget_c:
            budget_met = False
            break

    if budget_met:
        return StopDecision(stop=True, reason=StopReason.SE_BUDGET_MET, batches=batches)

    if batches >= policy.max_batches:
        return StopDecision(stop=True, reason=StopReason.MAX_BATCHES, batches=batches)

    return StopDecision(stop=False, reason=StopReason.CONTINUE, batches=batches)
