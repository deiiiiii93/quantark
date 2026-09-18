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
from dataclasses import asdict, dataclass
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
        lower_error_c: Schema 2: the conservative lower edge
            ``max(0, |signed_err_c| - interval_k * se_c)``.
        lower_exceeds_bound: Schema 2: the whole interval lies beyond the bound.
        bound_c: Schema 2: the per-cell budget every value above was divided by;
            ``None`` for schema 1.
        radius_c: Schema 2, deterministic reference: its declared error radius.
            ``se_c`` is then ``None`` -- a radius is the reference's whole declared
            allowance (analytical or a calibrated estimate), never a zero
            standard error -- the interval edges are ``|err| +/- radius_c``
            without ``interval_k``, and ``se_budget_met`` reads "the radius is
            within ``radius_budget_fraction`` of the bound".
    """

    signed_err_c: float
    se_c: Optional[float]
    interval_c: float
    se_budget_met: bool
    interval_within_bound: bool
    envelope_c: Optional[float]
    envelope_within_bound: bool
    passed: bool
    #: Schema 2: the conservative lower edge ``max(0, |err| - k*SE)``, whether it exceeds the
    #: cell bound, and the per-cell budget every value above was divided by (None for schema 1).
    lower_error_c: float = 0.0
    lower_exceeds_bound: bool = False
    bound_c: Optional[float] = None
    radius_c: Optional[float] = None


@dataclass(frozen=True)
class AggregateGateResult:
    """Outcome of one aggregate (mean signed bias) gate, in economic units.

    Under a deterministic reference ``se_of_mean_c`` is ``None`` and ``radius_of_mean_c`` is the
    MEAN of the cell radii: discretization errors may share a sign across cells, so they add
    linearly, never in quadrature.
    """

    mean_signed_bias_c: float
    se_of_mean_c: Optional[float]
    within_bound: bool
    se_adequate: bool
    passed: bool
    cells: int
    #: Schema 2: ``max(0, |mean| - k*SE)`` and whether it exceeds the aggregate bound.
    lower_c: float = 0.0
    lower_exceeds_bound: bool = False
    radius_of_mean_c: Optional[float] = None


def _require_finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value}")
    return value


def evaluate_cell_gate(
    candidate_raw: float,
    reference_raw: float,
    reference_se_raw: Optional[float],
    quantity: str,
    scale: EconomicScale,
    bounds: GateBounds,
    envelope_raw: Optional[float] = None,
    bound_c: Optional[float] = None,
    reference_radius_raw: Optional[float] = None,
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
        bound_c: Schema 2: this cell's own budget in economic units. Every
            economic value is divided by it, so the study's shared bounds read
            as fractions of the budget. ``None`` keeps schema 1's arithmetic.
        reference_radius_raw: Schema 2, deterministic reference: its declared
            error radius, engine units. Exactly one of this and
            ``reference_se_raw`` is given.

    Raises:
        ValidationError: non-finite input, a negative standard error or radius,
            both or neither uncertainty, or a non-positive budget.
    """
    _require_finite(candidate_raw, "candidate_raw")
    _require_finite(reference_raw, "reference_raw")
    if (reference_se_raw is None) == (reference_radius_raw is None):
        raise ValidationError("a cell gate takes exactly one of reference_se_raw and reference_radius_raw")
    deterministic = reference_radius_raw is not None
    if deterministic and bounds.radius_budget_fraction is None:
        raise ValidationError("a deterministic reference needs bounds.radius_budget_fraction")
    uncertainty_raw = reference_radius_raw if deterministic else reference_se_raw
    name = "reference_radius_raw" if deterministic else "reference_se_raw"
    _require_finite(uncertainty_raw, name)
    if uncertainty_raw < 0.0:
        raise ValidationError(f"{name} must be non-negative, got {uncertainty_raw}")

    signed_err_c = scale.to_economic(quantity, candidate_raw - reference_raw)
    # to_economic is linear but may carry a sign for exotic scales; an SE or a
    # radius is a magnitude either way.
    se_c = abs(scale.to_economic(quantity, uncertainty_raw))

    envelope_c: Optional[float] = None
    if envelope_raw is not None:
        _require_finite(envelope_raw, "envelope_raw")
        envelope_c = abs(scale.to_economic(quantity, envelope_raw))
    if bound_c is not None:
        _require_finite(bound_c, "bound_c")
        if bound_c <= 0.0:
            raise ValidationError(f"bound_c must be positive, got {bound_c}")
        signed_err_c /= bound_c
        se_c /= bound_c
        envelope_c = None if envelope_c is None else envelope_c / bound_c
    # a standard error is widened by interval_k; a radius is already the whole declared allowance
    half_width_c = se_c if deterministic else bounds.interval_k * se_c
    sharp_fraction = bounds.radius_budget_fraction if deterministic else bounds.se_budget_fraction
    interval_c = abs(signed_err_c) + half_width_c
    lower_error_c = max(0.0, abs(signed_err_c) - half_width_c)
    se_budget_met = se_c <= sharp_fraction * bounds.cell
    interval_within_bound = interval_c <= bounds.cell
    envelope_within_bound = envelope_c is None or envelope_c <= bounds.envelope_fraction * bounds.cell

    return CellGateResult(
        signed_err_c=signed_err_c,
        se_c=None if deterministic else se_c,
        radius_c=se_c if deterministic else None,
        interval_c=interval_c,
        se_budget_met=se_budget_met,
        interval_within_bound=interval_within_bound,
        envelope_c=envelope_c,
        envelope_within_bound=envelope_within_bound,
        passed=se_budget_met and interval_within_bound and envelope_within_bound,
        lower_error_c=lower_error_c,
        lower_exceeds_bound=lower_error_c > bounds.cell,
        bound_c=bound_c,
    )


def evaluate_aggregate_gate(
    signed_errs_c: Sequence[float],
    ses_c: Optional[Sequence[float]],
    bounds: GateBounds,
    radii_c: Optional[Sequence[float]] = None,
) -> AggregateGateResult:
    """Check the mean signed error across the cells of one quantity.

    Per-cell bounds are blind to a small error repeated with the same sign in
    every cell; this gate is what catches it.

    Args:
        signed_errs_c: Per-cell signed errors, economic units.
        ses_c: Per-cell benchmark standard errors, economic units; ``None``
            under a deterministic reference.
        bounds: Study bounds.
        radii_c: Per-cell deterministic radii, economic units; exactly one of
            this and ``ses_c`` is given.

    Raises:
        ValidationError: empty input, mismatched sequence lengths, or both or
            neither uncertainty.
    """
    if len(signed_errs_c) == 0:
        raise ValidationError("evaluate_aggregate_gate requires at least one cell")
    if (ses_c is None) == (radii_c is None):
        raise ValidationError("an aggregate gate takes exactly one of ses_c and radii_c")
    deterministic = radii_c is not None
    if deterministic and bounds.radius_budget_fraction is None:
        raise ValidationError("a deterministic reference needs bounds.radius_budget_fraction")
    spreads = radii_c if deterministic else ses_c
    label = "radii_c" if deterministic else "ses_c"
    if len(signed_errs_c) != len(spreads):
        raise ValidationError(
            f"signed_errs_c ({len(signed_errs_c)}) and {label} ({len(spreads)}) must have "
            "the same length"
        )

    count = len(signed_errs_c)
    for value in signed_errs_c:
        _require_finite(value, "signed_errs_c entry")
    for value in spreads:
        _require_finite(value, f"{label} entry")

    mean_signed_bias_c = sum(signed_errs_c) / count
    if deterministic:
        # Discretization errors may share a sign in every cell: the radius of the mean is the mean radius.
        spread_of_mean_c = sum(spreads) / count
        half_width_c = spread_of_mean_c
        sharp_fraction = bounds.radius_budget_fraction
    else:
        # Cells are independent runs, so their errors add in quadrature.
        spread_of_mean_c = math.sqrt(sum(se * se for se in spreads)) / count
        half_width_c = bounds.interval_k * spread_of_mean_c
        sharp_fraction = bounds.se_budget_fraction

    within_bound = abs(mean_signed_bias_c) + half_width_c <= bounds.mean_signed_bias
    se_adequate = spread_of_mean_c <= sharp_fraction * bounds.mean_signed_bias
    lower_c = max(0.0, abs(mean_signed_bias_c) - half_width_c)

    return AggregateGateResult(
        mean_signed_bias_c=mean_signed_bias_c,
        se_of_mean_c=None if deterministic else spread_of_mean_c,
        radius_of_mean_c=spread_of_mean_c if deterministic else None,
        within_bound=within_bound,
        se_adequate=se_adequate,
        passed=within_bound and se_adequate,
        cells=count,
        lower_c=lower_c,
        lower_exceeds_bound=lower_c > bounds.mean_signed_bias,
    )


_CELL_WIRE_V1 = ("signed_err_c", "se_c", "interval_c", "se_budget_met", "interval_within_bound", "envelope_c",
                 "envelope_within_bound", "passed")
_AGGREGATE_WIRE_V1 = ("mean_signed_bias_c", "se_of_mean_c", "within_bound", "se_adequate", "passed", "cells")


def cell_gate_wire(gate: CellGateResult, schema: int) -> dict:
    """The serialized gate. Schema 1 keeps exactly the keys it has always had, in their order."""
    data = asdict(gate)
    return {key: data[key] for key in _CELL_WIRE_V1} if schema == 1 else data


def aggregate_gate_wire(gate: AggregateGateResult, schema: int) -> dict:
    """The serialized aggregate gate. Schema 1 keeps exactly its original keys, in their order."""
    data = asdict(gate)
    return {key: data[key] for key in _AGGREGATE_WIRE_V1} if schema == 1 else data
