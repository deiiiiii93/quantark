"""Core configuration types for engine-release certification studies.

A *study* is the complete definition of one certification: which market/product
scenarios to test (``CaseSpec``), which quantities to certify, the pass/fail
bounds, how much sampling the stochastic reference arm may spend, and the two
pluggable arms (reference builder + candidate evaluators).

The framework pipeline consumes exactly one ``CertificationStudy``; both study
front doors (YAML over the builder registry, and hand-written Python studies)
produce this type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Tuple, runtime_checkable

from quantark.util.exceptions import ValidationError

#: The quantities a study may certify. PV plus the two spot Greeks.
QUANTITIES: Tuple[str, ...] = ("pv", "delta", "gamma")

#: Only schema 1 exists. This module is versioned independently of any prior
#: certification work; a bump requires a migration path for banked evidence.
SCHEMA: int = 1


@dataclass(frozen=True)
class CaseSpec:
    """One market/product scenario within a study.

    ``environment_params`` and ``product_params`` are per-case *overrides*
    applied on top of the study-level environment/product specs by the
    reference and candidate builders. Both are treated as immutable: nothing
    in the framework mutates them.
    """

    name: str
    environment_params: Mapping[str, Any] = field(default_factory=dict)
    product_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("CaseSpec.name must be a non-empty string")


@dataclass(frozen=True)
class GateBounds:
    """Pass/fail bounds for a study, expressed in economic units.

    Attributes:
        cell: Per-cell error bound. A cell passes when
            ``|error| + interval_k * SE <= cell``.
        mean_signed_bias: Aggregate bound on the mean signed error across the
            cells of one quantity -- catches a systematic tilt that per-cell
            bounds tolerate individually.
        se_budget_fraction: The reference arm must reach
            ``SE <= se_budget_fraction * cell`` before its verdict counts.
            Without this a cell could "pass" purely because the benchmark was
            too noisy to discriminate.
        interval_k: Interval half-width multiplier on the reference SE.
        envelope_fraction: The candidate's own discretization envelope must sit
            within ``envelope_fraction * cell``.
    """

    cell: float
    mean_signed_bias: float
    se_budget_fraction: float = 0.25
    interval_k: float = 2.0
    envelope_fraction: float = 0.5

    def __post_init__(self) -> None:
        for name in ("cell", "mean_signed_bias", "se_budget_fraction", "envelope_fraction"):
            value = getattr(self, name)
            if not (value > 0.0):
                raise ValidationError(f"GateBounds.{name} must be positive, got {value}")
        if self.interval_k < 0.0:
            raise ValidationError(
                f"GateBounds.interval_k must be non-negative, got {self.interval_k}"
            )


@dataclass(frozen=True)
class SamplingPolicy:
    """How much sampling the stochastic reference arm may spend, and how.

    Attributes:
        paths_per_batch: Paths in one batch. One batch is one independent
            randomization of the RQMC point set.
        min_batches: Batches before stopping may fire. At least 2 -- the
            batch-mean standard error needs ``ddof=1``.
        max_batches: Hard cap. Reaching it without meeting the SE budget caps
            the achievable verdict rather than failing the cell.
        seed: Base seed; batch ``i`` uses ``seed + i``.
        bump: Relative spot bump for the paired (common-random-numbers) Greek
            estimates.
    """

    paths_per_batch: int
    min_batches: int
    max_batches: int
    seed: int
    bump: float = 0.01

    def __post_init__(self) -> None:
        if self.paths_per_batch <= 0:
            raise ValidationError(
                f"SamplingPolicy.paths_per_batch must be positive, got {self.paths_per_batch}"
            )
        if self.min_batches < 2:
            raise ValidationError(
                "SamplingPolicy.min_batches must be at least 2 (batch-mean SE needs "
                f"ddof=1), got {self.min_batches}"
            )
        if self.max_batches < self.min_batches:
            raise ValidationError(
                f"SamplingPolicy.max_batches ({self.max_batches}) must be >= "
                f"min_batches ({self.min_batches})"
            )
        if not (self.bump > 0.0):
            raise ValidationError(
                f"SamplingPolicy.bump must be positive, got {self.bump}"
            )


@runtime_checkable
class EconomicScale(Protocol):
    """Converts a raw quantity error into the study's economic unit."""

    def to_economic(self, quantity: str, raw: float) -> float:
        """Map ``raw`` (engine units) for ``quantity`` into economic units."""
        ...


@dataclass(frozen=True)
class HedgeContractScale:
    """Economic scale in *hedge contracts* -- the unit a desk actually trades.

    One contract carries ``delta_quantum`` of delta, so a delta error of one
    quantum is an error of one contract in the hedge. PV and gamma are mapped
    onto the same axis through a 1% spot move: gamma is the delta drift a 1%
    move produces, and a PV error is equated to the P&L that whole contracts
    generate over that same 1% move. This makes every gate bound readable as
    "how many contracts of hedge could a trader be wrong by".
    """

    hedge_multiplier: float
    hedge_inception_spot: float
    notional: float

    #: Spot move used to put PV and gamma on the delta-contract axis.
    reference_move: float = 0.01

    def __post_init__(self) -> None:
        for name in ("hedge_multiplier", "hedge_inception_spot", "notional", "reference_move"):
            value = getattr(self, name)
            if not (value > 0.0):
                raise ValidationError(
                    f"HedgeContractScale.{name} must be positive, got {value}"
                )

    @property
    def delta_quantum(self) -> float:
        """Delta carried by one hedge contract, per unit of study notional."""
        return self.hedge_multiplier * self.hedge_inception_spot / self.notional

    @property
    def _move(self) -> float:
        """Absolute spot move corresponding to ``reference_move``."""
        return self.reference_move * self.hedge_inception_spot

    def to_economic(self, quantity: str, raw: float) -> float:
        """Convert ``raw`` into hedge contracts. Linear in ``raw`` for every
        quantity, which is what lets gates convert errors and standard errors
        with the same map."""
        quantum = self.delta_quantum
        if quantity == "delta":
            return raw / quantum
        if quantity == "gamma":
            return raw * self._move / quantum
        if quantity == "pv":
            return raw / (quantum * self._move)
        raise ValidationError(
            f"HedgeContractScale cannot convert unknown quantity {quantity!r}; "
            f"expected one of {QUANTITIES}"
        )


@dataclass(frozen=True)
class CertificationStudy:
    """The complete, typed definition of one certification run.

    Attributes:
        name: Study identifier; also the output subdirectory name.
        schema: Study schema version (must be 1).
        cases: Scenarios to certify; names must be unique.
        quantities: Subset of :data:`QUANTITIES` to certify.
        bounds: Pass/fail bounds in economic units.
        scale: Raw-to-economic converter.
        reference: The stochastic benchmark arm (``ReferenceBuilder``).
        candidates: Deterministic engines under certification
            (``CandidateEvaluator``); one decision is issued per candidate.
        sampling: Sampling budget and stopping limits for the reference arm.
        source_text: Verbatim YAML text when the study was loaded from a file.
            Embedded in evidence so a certificate ships its own definition;
            anchors require it.
    """

    name: str
    schema: int
    cases: Tuple[CaseSpec, ...]
    quantities: Tuple[str, ...]
    bounds: GateBounds
    scale: EconomicScale
    reference: Any
    candidates: Tuple[Any, ...]
    sampling: SamplingPolicy
    source_text: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("CertificationStudy.name must be a non-empty string")
        if self.schema != SCHEMA:
            raise ValidationError(
                f"CertificationStudy.schema must be {SCHEMA}, got {self.schema}"
            )
        if not self.cases:
            raise ValidationError("CertificationStudy.cases must not be empty")
        if not self.candidates:
            raise ValidationError("CertificationStudy.candidates must not be empty")
        if not self.quantities:
            raise ValidationError("CertificationStudy.quantities must not be empty")

        unknown = [q for q in self.quantities if q not in QUANTITIES]
        if unknown:
            raise ValidationError(
                f"CertificationStudy.quantities contains unknown entries {unknown}; "
                f"expected a subset of {QUANTITIES}"
            )

        names = [case.name for case in self.cases]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValidationError(
                f"CertificationStudy.cases has duplicate case names: {duplicates}"
            )
