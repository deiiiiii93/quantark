"""Anytime-valid sequential admission for Greek certification cells.

The frozen certification gate spends a declared number of RQMC batches per
cell and judges once, comparing

    |pde - mc_mean| + t_{c,B-1} * SE + pde_envelope + bias_envelope  <=  bound

That single look is what licenses the Student-t quantile.  Watching the same
quantity as batches arrive and stopping at the first favourable look does not
work: the interval is only valid at one pre-chosen ``B``, so repeated peeking
inflates the error rate without bound.

This module supplies the sequential analogue.  The Student-t half-width is
replaced by the asymptotic confidence sequence of Waudby-Smith, Arbour,
Sinha, Kennedy and Ramdas (*Ann. Statist.* 2024),

    w_t = sigma_t * sqrt( 2 (t rho^2 + 1) / (t^2 rho^2)
                          * ln( sqrt(t rho^2 + 1) / alpha ) )

which covers the mean simultaneously at every ``t``, so stopping at a
data-dependent time is legitimate.  Width is the price: ``w_t`` is strictly
wider than the fixed-``B`` half-width at the same ``t`` for any usable alpha,
which gives the property that makes adoption defensible -- **a sequential
admission at ``t`` implies the frozen gate would also have passed had it been
judged at ``t``.**  Early stopping trades wall-clock for width, never evidence
for optimism.

Two components move with the batches, not one.  Both the Greek's own interval
and the substep bias envelope are estimated from the same stream, so the rule
tracks them jointly and splits alpha between them.  A rule that froze the bias
envelope at its final value would mis-attribute why a cell needs batches: on
the near-KI Heston cell it is the bias interval, not the Greek interval, that
the large allocation was buying.

Every parameter -- family alpha, the number of tests it is spread over, the
floors, the cap, and the ``rho`` that fixes the sequence's shape -- is declared
in advance and hashed, in the same spirit as the frozen adaptive allocation in
``adaptive_allocation``: a policy chosen after seeing the data is not a policy.

Adoption note: the decision floor must not fall below what a downstream
aggregate gate consumes.  The aggregate reads a *common scramble prefix* across
cells, so a cell that stopped earlier than that prefix cannot contribute a
cohort row at all.  ``aggregate_floor_batches`` carries that requirement, and
it is the binding constraint on realisable savings whenever it exceeds the
per-cell minimum.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from quantark.util.numerical import is_zero, safe_log, safe_sqrt

__all__ = [
    "SequentialAdmissionPolicy",
    "SequentialAdmissionStatus",
    "SequentialDecision",
    "confidence_sequence_half_width",
    "scan_admission_stream",
    "sequential_admission",
    "tune_rho_squared",
]

# Smallest batch count at which a sample standard deviation exists at all.
_MIN_DECIDABLE_BATCHES = 2
# Grid over which rho is tuned.  Fixed here so tuning is reproducible from the
# declared plan alone and cannot vary with the data or the platform.
_RHO_SQUARED_GRID_DECADES = (-4.0, 1.0)
_RHO_SQUARED_GRID_POINTS = 200


class SequentialAdmissionStatus(str, Enum):
    """Outcome of one look at a growing batch stream."""

    ADMIT = "ADMIT"
    REJECT = "REJECT"
    CONTINUE = "CONTINUE"
    EXHAUSTED = "EXHAUSTED"


def confidence_sequence_half_width(
    batches: int,
    *,
    standard_deviation: float,
    rho_squared: float,
    alpha: float,
) -> float:
    """Anytime-valid half-width for the mean of ``batches`` batch estimates.

    Returns ``inf`` below two batches, where no sample dispersion exists, and
    exactly zero for a degenerate stream so a deterministic cell is not charged
    width it does not need.
    """
    count = int(batches)
    sigma = float(standard_deviation)
    rho2 = float(rho_squared)
    if sigma < 0.0:
        raise ValueError("standard_deviation must be non-negative")
    if rho2 <= 0.0:
        raise ValueError("rho_squared must be positive")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if is_zero(sigma):
        return 0.0
    if count < _MIN_DECIDABLE_BATCHES:
        return float("inf")

    growth = count * rho2 + 1.0
    scale = (2.0 * growth) / (count * count * rho2)
    logarithm = safe_log(safe_sqrt(growth) / float(alpha))
    return sigma * safe_sqrt(max(scale * logarithm, 0.0))


def tune_rho_squared(planned_batches: int, alpha: float) -> float:
    """Choose ``rho^2`` to minimise the width at the declared planning horizon.

    A confidence sequence trades width across time: any ``rho`` is valid, but
    each is tightest near one horizon.  Tuning at a *declared* ``planned_batches``
    keeps the choice free of the data, which is what preserves the guarantee.
    """
    horizon = int(planned_batches)
    if horizon < _MIN_DECIDABLE_BATCHES:
        raise ValueError("planned_batches must be at least two")
    grid = np.logspace(*_RHO_SQUARED_GRID_DECADES, _RHO_SQUARED_GRID_POINTS)
    widths = [
        confidence_sequence_half_width(
            horizon, standard_deviation=1.0, rho_squared=float(rho2), alpha=alpha
        )
        for rho2 in grid
    ]
    return float(grid[int(np.argmin(widths))])


@dataclass(frozen=True)
class SequentialAdmissionPolicy:
    """Every sequential-stopping parameter, declared before the run and hashed.

    Parameters
    ----------
    family_alpha:
        Family-wise error budget across all tests.
    tests:
        Number of ``(cell, greek)`` tests the budget is spread over.  Declared
        up front from the regime matrix, never counted from what happened to
        run: a data-dependent ``K`` is not a Bonferroni correction.
    min_batches:
        Per-cell minimum before any decision is taken.
    aggregate_floor_batches:
        Common-scramble prefix a downstream aggregate gate consumes.  Zero when
        no aggregate reads this cell.
    planned_batches:
        Horizon at which ``rho`` is tuned.
    max_batches:
        Cap.  Reaching it yields ``EXHAUSTED``, never a lenient admission.
    """

    family_alpha: float
    tests: int
    min_batches: int
    aggregate_floor_batches: int
    planned_batches: int
    max_batches: int
    components_per_test: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < float(self.family_alpha) < 1.0:
            raise ValueError("family_alpha must lie strictly between 0 and 1")
        if int(self.tests) < 1:
            raise ValueError("tests must be at least one")
        if int(self.components_per_test) < 1:
            raise ValueError("components_per_test must be at least one")
        if int(self.min_batches) < _MIN_DECIDABLE_BATCHES:
            raise ValueError("min_batches must be at least two")
        if int(self.aggregate_floor_batches) < 0:
            raise ValueError("aggregate_floor_batches must be non-negative")
        if int(self.max_batches) < self.first_decidable_batch:
            raise ValueError("max_batches must not fall below the decision floor")
        if not (
            self.first_decidable_batch
            <= int(self.planned_batches)
            <= int(self.max_batches)
        ):
            raise ValueError(
                "planned_batches must lie between the decision floor and the cap"
            )

    @property
    def first_decidable_batch(self) -> int:
        """Earliest batch at which a decision may be taken."""
        return max(
            int(self.min_batches),
            int(self.aggregate_floor_batches),
            _MIN_DECIDABLE_BATCHES,
        )

    @property
    def alpha_per_component(self) -> float:
        """Bonferroni share for one component of one test."""
        return float(self.family_alpha) / (
            int(self.tests) * int(self.components_per_test)
        )

    @property
    def rho_squared(self) -> float:
        """Shape parameter, tuned at the declared horizon."""
        return tune_rho_squared(int(self.planned_batches), self.alpha_per_component)

    def declaration(self) -> dict:
        """Canonical record of the declared policy, for the run evidence."""
        return {
            "family_alpha": float(self.family_alpha),
            "tests": int(self.tests),
            "components_per_test": int(self.components_per_test),
            "alpha_per_component": self.alpha_per_component,
            "min_batches": int(self.min_batches),
            "aggregate_floor_batches": int(self.aggregate_floor_batches),
            "first_decidable_batch": self.first_decidable_batch,
            "planned_batches": int(self.planned_batches),
            "max_batches": int(self.max_batches),
            "rho_squared": self.rho_squared,
            "confidence_sequence": "waudby-smith-2024-asymptotic",
        }

    def sha256(self) -> str:
        """Digest of the declaration, so the policy can be frozen before the run."""
        canonical = json.dumps(self.declaration(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def with_(self, **overrides) -> "SequentialAdmissionPolicy":
        """Return a validated copy, for per-cell floors and caps."""
        return replace(self, **overrides)


@dataclass(frozen=True)
class SequentialDecision:
    """One look at the stream, and why it decided what it did."""

    status: SequentialAdmissionStatus
    batches_used: int
    reference_gap: float
    greek_half_width: float
    pde_discretization_envelope: float
    bias_envelope: float
    total_uncertainty: float
    economic_bound: float
    reason: str

    def batches_saved_against(self, fixed_allocation: int) -> int:
        """Batches not spent versus a fixed allocation (never negative)."""
        return max(int(fixed_allocation) - int(self.batches_used), 0)

    def as_dict(self) -> dict:
        return {
            "status": self.status.value,
            "batches_used": int(self.batches_used),
            "reference_gap": float(self.reference_gap),
            "greek_half_width": float(self.greek_half_width),
            "pde_discretization_envelope": float(self.pde_discretization_envelope),
            "bias_envelope": float(self.bias_envelope),
            "total_uncertainty": float(self.total_uncertainty),
            "economic_bound": float(self.economic_bound),
            "reason": self.reason,
        }


def sequential_admission(
    *,
    policy: SequentialAdmissionPolicy,
    batches_used: int,
    reference_gap: float,
    greek_batch_standard_deviation: float,
    pde_discretization_envelope: float,
    economic_bound: float,
    substep_bias_mean: float = 0.0,
    substep_batch_standard_deviation: Optional[float] = None,
    frozen_bias_envelope: float = 0.0,
) -> SequentialDecision:
    """Judge one look at a cell's growing batch stream.

    ``substep_batch_standard_deviation`` supplies the second component: when it
    is given, the bias envelope is ``|mean| + w_t`` and shrinks with the
    batches, mirroring production, where the substep interval is estimated from
    the same batches.  When it is ``None`` the caller's
    ``frozen_bias_envelope`` is used unchanged.
    """
    bound = float(economic_bound)
    if bound <= 0.0:
        raise ValueError("economic_bound must be positive")
    gap = abs(float(reference_gap))
    pde_envelope = float(pde_discretization_envelope)
    if pde_envelope < 0.0:
        raise ValueError("pde_discretization_envelope must be non-negative")
    count = int(batches_used)
    alpha = policy.alpha_per_component
    rho2 = policy.rho_squared

    greek_width = confidence_sequence_half_width(
        count,
        standard_deviation=greek_batch_standard_deviation,
        rho_squared=rho2,
        alpha=alpha,
    )
    if substep_batch_standard_deviation is None:
        bias_envelope = float(frozen_bias_envelope)
    else:
        bias_envelope = abs(float(substep_bias_mean)) + confidence_sequence_half_width(
            count,
            standard_deviation=substep_batch_standard_deviation,
            rho_squared=rho2,
            alpha=alpha,
        )
    if bias_envelope < 0.0:
        raise ValueError("bias envelope must be non-negative")

    total = greek_width + pde_envelope + bias_envelope

    def decision(status: SequentialAdmissionStatus, reason: str) -> SequentialDecision:
        return SequentialDecision(
            status=status,
            batches_used=count,
            reference_gap=gap,
            greek_half_width=greek_width,
            pde_discretization_envelope=pde_envelope,
            bias_envelope=bias_envelope,
            total_uncertainty=total,
            economic_bound=bound,
            reason=reason,
        )

    if count < policy.first_decidable_batch:
        return decision(
            SequentialAdmissionStatus.CONTINUE,
            f"below the declared decision floor of {policy.first_decidable_batch}"
            " batches",
        )

    if gap + total <= bound:
        return decision(
            SequentialAdmissionStatus.ADMIT,
            "anytime-valid comparison interval lies inside the economic bound",
        )
    # A rejection must be provable from the gap alone.  The bias envelope only
    # widens the interval, so spending it here would let an unresolved bias
    # masquerade as a failed cell.
    if gap - greek_width - pde_envelope > bound:
        return decision(
            SequentialAdmissionStatus.REJECT,
            "anytime-valid interval lies wholly outside the economic bound",
        )
    if count >= int(policy.max_batches):
        return decision(
            SequentialAdmissionStatus.EXHAUSTED,
            "reached the declared cap without deciding",
        )
    return decision(
        SequentialAdmissionStatus.CONTINUE,
        "interval still overlaps the economic bound",
    )


def scan_admission_stream(
    *,
    policy: SequentialAdmissionPolicy,
    pde_value: float,
    greek_series: Sequence[float],
    substep_series: Optional[Sequence[float]],
    pde_discretization_envelope: float,
    economic_bound: float,
    frozen_bias_envelope: float = 0.0,
) -> SequentialDecision:
    """Walk a batch stream and return the first decision the policy reaches.

    The stream must reach the decision floor; a shorter one cannot be judged
    under this policy at all, and silently reporting ``EXHAUSTED`` would
    disguise a misconfiguration as an inconclusive cell.
    """
    greek = np.asarray(greek_series, dtype=float)
    if greek.ndim != 1:
        raise ValueError("greek_series must be one-dimensional")
    available = int(greek.size)
    floor = policy.first_decidable_batch
    if available < floor:
        raise ValueError(
            f"stream has {available} batches but the policy cannot decide before "
            f"{floor}"
        )
    substep = None
    if substep_series is not None:
        substep = np.asarray(substep_series, dtype=float)
        if substep.size < available:
            raise ValueError("substep_series must be at least as long as greek_series")

    last: Optional[SequentialDecision] = None
    limit = min(available, int(policy.max_batches))
    for count in range(floor, limit + 1):
        prefix = greek[:count]
        gap = float(pde_value) - float(np.mean(prefix))
        substep_mean = 0.0
        substep_sd: Optional[float] = None
        if substep is not None:
            window = substep[:count]
            substep_mean = float(np.mean(window))
            substep_sd = float(np.std(window, ddof=1))
        last = sequential_admission(
            policy=policy,
            batches_used=count,
            reference_gap=gap,
            greek_batch_standard_deviation=float(np.std(prefix, ddof=1)),
            pde_discretization_envelope=pde_discretization_envelope,
            economic_bound=economic_bound,
            substep_bias_mean=substep_mean,
            substep_batch_standard_deviation=substep_sd,
            frozen_bias_envelope=frozen_bias_envelope,
        )
        if last.status in (
            SequentialAdmissionStatus.ADMIT,
            SequentialAdmissionStatus.REJECT,
            SequentialAdmissionStatus.EXHAUSTED,
        ):
            return last

    assert last is not None  # the loop runs at least once: available >= floor
    return replace(
        last,
        status=SequentialAdmissionStatus.EXHAUSTED,
        reason="stream ended without deciding",
    )
