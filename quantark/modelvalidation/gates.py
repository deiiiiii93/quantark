"""Gate arithmetic: does the candidate agree with the benchmark?

Pure data in, pure data out -- no pricing, no I/O. Every comparison happens in
the study's economic unit, so a bound reads as "how wrong could a hedger be"
rather than as an abstract tolerance.

Two gates:

* the **cell gate** compares one candidate quantity against the benchmark for
  one case, and
* the **aggregate gate** checks the mean *signed* error across the cells of one
  quantity, catching a systematic tilt that per-cell bounds tolerate one cell at
  a time.

Both require the benchmark to be sharp enough to discriminate before their
verdict counts: a pass earned against a noisy reference is not evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.study import EconomicScale, GateBounds


@dataclass(frozen=True)
class CellGateResult:
    """Outcome of one cell gate, in economic units ("c").

    Attributes:
        signed_err_c: ``candidate - reference``; the sign matters because the
            aggregate gate sums it.
        se_c: Benchmark standard error.
        interval_c: ``|signed_err_c| + interval_k * se_c`` -- the conservative
            edge of the disagreement interval.
        se_budget_met: Benchmark sharp enough for this bound.
        interval_within_bound: Interval edge inside the cell bound.
        envelope_c: Candidate's own discretization envelope, when ladders were
            supplied; ``None`` otherwise.
        envelope_within_bound: Envelope inside its share of the cell bound.
        passed: All three checks.
    """

    signed_err_c: float
    se_c: float
    interval_c: float
    se_budget_met: bool
    interval_within_bound: bool
    envelope_c: Optional[float]
    envelope_within_bound: bool
    passed: bool


@dataclass(frozen=True)
class AggregateGateResult:
    """Outcome of one aggregate (mean signed bias) gate, in economic units."""

    mean_signed_bias_c: float
    se_of_mean_c: float
    within_bound: bool
    se_adequate: bool
    passed: bool
    cells: int


def _require_finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value}")
    return value


def evaluate_cell_gate(
    candidate_raw: float,
    reference_raw: float,
    reference_se_raw: float,
    quantity: str,
    scale: EconomicScale,
    bounds: GateBounds,
    envelope_raw: Optional[float] = None,
) -> CellGateResult:
    """Compare one candidate quantity against the benchmark for one case.

    All raw inputs are converted through ``scale``; the conversion is linear per
    quantity, which is what lets the error and the standard error share it.

    Args:
        candidate_raw: Deterministic engine value, engine units.
        reference_raw: Benchmark estimate, engine units.
        reference_se_raw: Benchmark standard error, engine units.
        quantity: One of the study quantities.
        scale: Raw-to-economic converter.
        bounds: Study bounds.
        envelope_raw: Candidate's own discretization envelope from its
            refinement ladders, engine units; ``None`` when no ladder was run.

    Raises:
        ValidationError: non-finite input, or a negative standard error.
    """
    _require_finite(candidate_raw, "candidate_raw")
    _require_finite(reference_raw, "reference_raw")
    _require_finite(reference_se_raw, "reference_se_raw")
    if reference_se_raw < 0.0:
        raise ValidationError(
            f"reference_se_raw must be non-negative, got {reference_se_raw}"
        )

    signed_err_c = scale.to_economic(quantity, candidate_raw - reference_raw)
    se_c = scale.to_economic(quantity, reference_se_raw)
    # to_economic is linear but may carry a sign for exotic scales; an SE is a
    # magnitude either way.
    se_c = abs(se_c)
    interval_c = abs(signed_err_c) + bounds.interval_k * se_c

    se_budget_met = se_c <= bounds.se_budget_fraction * bounds.cell
    interval_within_bound = interval_c <= bounds.cell

    if envelope_raw is None:
        envelope_c: Optional[float] = None
        envelope_within_bound = True
    else:
        _require_finite(envelope_raw, "envelope_raw")
        envelope_c = abs(scale.to_economic(quantity, envelope_raw))
        envelope_within_bound = envelope_c <= bounds.envelope_fraction * bounds.cell

    return CellGateResult(
        signed_err_c=signed_err_c,
        se_c=se_c,
        interval_c=interval_c,
        se_budget_met=se_budget_met,
        interval_within_bound=interval_within_bound,
        envelope_c=envelope_c,
        envelope_within_bound=envelope_within_bound,
        passed=se_budget_met and interval_within_bound and envelope_within_bound,
    )


def evaluate_aggregate_gate(
    signed_errs_c: Sequence[float],
    ses_c: Sequence[float],
    bounds: GateBounds,
) -> AggregateGateResult:
    """Check the mean signed error across the cells of one quantity.

    Per-cell bounds are blind to a small error repeated with the same sign in
    every cell; this gate is what catches it.

    Args:
        signed_errs_c: Per-cell signed errors, economic units.
        ses_c: Per-cell benchmark standard errors, economic units.
        bounds: Study bounds.

    Raises:
        ValidationError: empty input, or mismatched sequence lengths.
    """
    if len(signed_errs_c) == 0:
        raise ValidationError("evaluate_aggregate_gate requires at least one cell")
    if len(signed_errs_c) != len(ses_c):
        raise ValidationError(
            f"signed_errs_c ({len(signed_errs_c)}) and ses_c ({len(ses_c)}) must have "
            "the same length"
        )

    count = len(signed_errs_c)
    for value in signed_errs_c:
        _require_finite(value, "signed_errs_c entry")
    for value in ses_c:
        _require_finite(value, "ses_c entry")

    mean_signed_bias_c = sum(signed_errs_c) / count
    # Cells are independent runs, so their errors add in quadrature.
    se_of_mean_c = math.sqrt(sum(se * se for se in ses_c)) / count

    within_bound = (
        abs(mean_signed_bias_c) + bounds.interval_k * se_of_mean_c
        <= bounds.mean_signed_bias
    )
    se_adequate = se_of_mean_c <= bounds.se_budget_fraction * bounds.mean_signed_bias

    return AggregateGateResult(
        mean_signed_bias_c=mean_signed_bias_c,
        se_of_mean_c=se_of_mean_c,
        within_bound=within_bound,
        se_adequate=se_adequate,
        passed=within_bound and se_adequate,
        cells=count,
    )
