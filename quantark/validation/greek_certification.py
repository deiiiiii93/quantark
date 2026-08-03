"""Uncertainty-aware certification for deterministic PDE Greeks.

The reference confidence interval and deterministic PDE refinement envelope
are both part of the verdict.  A comparison that overlaps the economic bound
is inconclusive; it is never silently promoted to pass or fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np
from scipy.stats import t as student_t


class EquivalenceStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class EquivalenceResult:
    status: EquivalenceStatus
    estimate_difference: float
    lower: float
    upper: float
    economic_bound: float
    confidence: float
    reference_standard_error: float
    reference_degrees_of_freedom: Optional[int]
    reference_half_width: float
    pde_discretization_envelope: float
    reference_bias_envelope: float
    total_uncertainty: float
    label: str = ""
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "status": self.status.value,
            "estimate_difference": self.estimate_difference,
            "interval": [self.lower, self.upper],
            "economic_bound": self.economic_bound,
            "confidence": self.confidence,
            "reference_standard_error": self.reference_standard_error,
            "reference_degrees_of_freedom": self.reference_degrees_of_freedom,
            "reference_half_width": self.reference_half_width,
            "pde_discretization_envelope": self.pde_discretization_envelope,
            "reference_bias_envelope": self.reference_bias_envelope,
            "total_uncertainty": self.total_uncertainty,
            "label": self.label,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EconomicGreekScale:
    """Convert normalized-model Greeks to hedge-instrument contracts.

    ``model_spot`` is the coordinate used by the pricing problem and controls
    the absolute spot move used to interpret gamma.  ``hedge_inception_spot``
    is the *actual* index level that fixed the position multiplier at trade
    inception and therefore controls the futures-contract delta quantum.  The
    two must not be conflated when a homogeneous model is normalized around
    100 for numerical certification.
    """

    model_spot: float
    hedge_inception_spot: float
    study_notional: float
    hedge_multiplier: float
    gamma_relative_move: float = 0.01

    def __post_init__(self):
        values = {
            "model_spot": self.model_spot,
            "hedge_inception_spot": self.hedge_inception_spot,
            "study_notional": self.study_notional,
            "hedge_multiplier": self.hedge_multiplier,
            "gamma_relative_move": self.gamma_relative_move,
        }
        for name, value in values.items():
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def delta_quantum_per_contract(self) -> float:
        return (
            float(self.hedge_multiplier)
            * float(self.hedge_inception_spot)
            / float(self.study_notional)
        )

    def delta_contracts(self, per_unit_delta_difference: float) -> float:
        return float(per_unit_delta_difference) / self.delta_quantum_per_contract

    def gamma_hedge_contract_change(self, per_unit_gamma_difference: float) -> float:
        delta_change = (
            float(per_unit_gamma_difference)
            * float(self.model_spot)
            * float(self.gamma_relative_move)
        )
        return delta_change / self.delta_quantum_per_contract

    def as_dict(self) -> dict:
        return {
            "model_spot": float(self.model_spot),
            "hedge_inception_spot": float(self.hedge_inception_spot),
            "study_notional": float(self.study_notional),
            "hedge_multiplier": float(self.hedge_multiplier),
            "gamma_relative_move": float(self.gamma_relative_move),
            "delta_quantum_per_contract": self.delta_quantum_per_contract,
        }


def _inconclusive(
    *,
    estimate_difference: float,
    economic_bound: float,
    confidence: float,
    reference_standard_error: float,
    reference_degrees_of_freedom: Optional[int],
    pde_discretization_envelope: float,
    reference_bias_envelope: float,
    label: str,
    reason: str,
) -> EquivalenceResult:
    return EquivalenceResult(
        status=EquivalenceStatus.INCONCLUSIVE,
        estimate_difference=float(estimate_difference),
        lower=float("nan"),
        upper=float("nan"),
        economic_bound=float(economic_bound),
        confidence=float(confidence),
        reference_standard_error=float(reference_standard_error),
        reference_degrees_of_freedom=reference_degrees_of_freedom,
        reference_half_width=float("nan"),
        pde_discretization_envelope=float(pde_discretization_envelope),
        reference_bias_envelope=float(reference_bias_envelope),
        total_uncertainty=float("nan"),
        label=label,
        reason=reason,
    )


def certify_equivalence(
    estimate_difference: float,
    reference_standard_error: float,
    economic_bound: float,
    *,
    reference_degrees_of_freedom: Optional[int] = None,
    pde_discretization_envelope: float = 0.0,
    reference_bias_envelope: float = 0.0,
    confidence: float = 0.95,
    label: str = "",
) -> EquivalenceResult:
    """Return PASS/FAIL/INCONCLUSIVE for a symmetric equivalence bound.

    PASS requires the complete comparison interval to lie inside
    ``[-economic_bound, economic_bound]``.  FAIL requires it to lie wholly
    outside on one side.  Any overlap with a boundary is INCONCLUSIVE.
    """
    difference = float(estimate_difference)
    standard_error = float(reference_standard_error)
    bound = float(economic_bound)
    pde_envelope = float(pde_discretization_envelope)
    bias_envelope = float(reference_bias_envelope)
    dof = (
        None
        if reference_degrees_of_freedom is None
        else int(reference_degrees_of_freedom)
    )
    finite = all(
        np.isfinite(value)
        for value in (
            difference,
            standard_error,
            bound,
            pde_envelope,
            bias_envelope,
            confidence,
        )
    )
    if not finite:
        return _inconclusive(
            estimate_difference=difference,
            economic_bound=bound,
            confidence=confidence,
            reference_standard_error=standard_error,
            reference_degrees_of_freedom=dof,
            pde_discretization_envelope=pde_envelope,
            reference_bias_envelope=bias_envelope,
            label=label,
            reason="non-finite certification input",
        )
    if bound <= 0.0:
        raise ValueError("economic_bound must be positive")
    if standard_error < 0.0 or pde_envelope < 0.0 or bias_envelope < 0.0:
        raise ValueError("uncertainty inputs must be non-negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if standard_error > 0.0 and (dof is None or dof < 1):
        return _inconclusive(
            estimate_difference=difference,
            economic_bound=bound,
            confidence=confidence,
            reference_standard_error=standard_error,
            reference_degrees_of_freedom=dof,
            pde_discretization_envelope=pde_envelope,
            reference_bias_envelope=bias_envelope,
            label=label,
            reason="positive reference standard error requires degrees of freedom",
        )

    critical = (
        0.0
        if standard_error == 0.0
        else float(student_t.ppf(0.5 + 0.5 * confidence, dof))
    )
    reference_half_width = critical * standard_error
    total_uncertainty = reference_half_width + pde_envelope + bias_envelope
    lower = difference - total_uncertainty
    upper = difference + total_uncertainty
    if lower >= -bound and upper <= bound:
        status = EquivalenceStatus.PASS
        reason = "comparison interval is wholly inside the economic bound"
    elif lower > bound or upper < -bound:
        status = EquivalenceStatus.FAIL
        reason = "comparison interval is wholly outside the economic bound"
    else:
        status = EquivalenceStatus.INCONCLUSIVE
        reason = "comparison interval overlaps the economic bound"
    return EquivalenceResult(
        status=status,
        estimate_difference=difference,
        lower=float(lower),
        upper=float(upper),
        economic_bound=bound,
        confidence=float(confidence),
        reference_standard_error=standard_error,
        reference_degrees_of_freedom=dof,
        reference_half_width=float(reference_half_width),
        pde_discretization_envelope=pde_envelope,
        reference_bias_envelope=bias_envelope,
        total_uncertainty=float(total_uncertainty),
        label=label,
        reason=reason,
    )


def certify_signed_bias_from_batches(
    difference_batches: Sequence[float],
    economic_bound: float,
    *,
    pde_discretization_envelope: float = 0.0,
    reference_bias_envelope: float = 0.0,
    confidence: float = 0.95,
    label: str = "signed_bias",
) -> EquivalenceResult:
    """Certify mean signed bias from already paired outer-batch estimates.

    Callers should first average all date/case differences *within each RQMC
    scramble*.  The standard error across those batch aggregates then retains
    cross-cell covariance induced by reusing the same scrambles.
    """
    values = np.asarray(difference_batches, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("difference_batches must contain at least two values")
    if not np.all(np.isfinite(values)):
        return certify_equivalence(
            float("nan"),
            float("nan"),
            economic_bound,
            pde_discretization_envelope=pde_discretization_envelope,
            reference_bias_envelope=reference_bias_envelope,
            confidence=confidence,
            label=label,
        )
    return certify_equivalence(
        float(np.mean(values)),
        float(np.std(values, ddof=1) / np.sqrt(values.size)),
        economic_bound,
        reference_degrees_of_freedom=int(values.size - 1),
        pde_discretization_envelope=pde_discretization_envelope,
        reference_bias_envelope=reference_bias_envelope,
        confidence=confidence,
        label=label,
    )
