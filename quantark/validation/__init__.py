"""Reusable numerical-model certification primitives."""

from .adaptive_allocation import (
    CellPrecision,
    StopDecision,
    neyman_allocation,
    precision_stop,
    projected_aggregate_halfwidth,
)
from .greek_certification import (
    EconomicGreekScale,
    EquivalenceResult,
    EquivalenceStatus,
    IndependentCohortMean,
    certify_equivalence,
    certify_signed_bias_from_batches,
    certify_signed_bias_from_independent_cohorts,
    summarize_independent_cohort_means,
)
from .sequential_admission import (
    SequentialAdmissionPolicy,
    SequentialAdmissionStatus,
    SequentialDecision,
    confidence_sequence_half_width,
    scan_admission_stream,
    sequential_admission,
    tune_rho_squared,
)

__all__ = [
    "CellPrecision",
    "StopDecision",
    "neyman_allocation",
    "precision_stop",
    "projected_aggregate_halfwidth",
    "SequentialAdmissionPolicy",
    "SequentialAdmissionStatus",
    "SequentialDecision",
    "confidence_sequence_half_width",
    "scan_admission_stream",
    "sequential_admission",
    "tune_rho_squared",
    "EconomicGreekScale",
    "EquivalenceResult",
    "EquivalenceStatus",
    "IndependentCohortMean",
    "certify_equivalence",
    "certify_signed_bias_from_batches",
    "certify_signed_bias_from_independent_cohorts",
    "summarize_independent_cohort_means",
]
