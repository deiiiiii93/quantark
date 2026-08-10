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

__all__ = [
    "CellPrecision",
    "StopDecision",
    "neyman_allocation",
    "precision_stop",
    "projected_aggregate_halfwidth",
    "EconomicGreekScale",
    "EquivalenceResult",
    "EquivalenceStatus",
    "IndependentCohortMean",
    "certify_equivalence",
    "certify_signed_bias_from_batches",
    "certify_signed_bias_from_independent_cohorts",
    "summarize_independent_cohort_means",
]
