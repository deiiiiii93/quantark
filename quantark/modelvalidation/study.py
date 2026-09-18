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

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol, Tuple, runtime_checkable

import numpy as np

from quantark.util.exceptions import ValidationError

#: The quantities a schema-1 study may certify. PV plus the two spot Greeks.
QUANTITIES: Tuple[str, ...] = ("pv", "delta", "gamma")

#: Default study schema. Schema 2 adds catalogue quantities, per-quantity budgets,
#: case context and semantic assertions; schema 1 keeps its exact behaviour.
SCHEMA: int = 1
SUPPORTED_SCHEMAS: Tuple[int, ...] = (1, 2)

#: How a raw engine number maps onto the notional-normalized axis (schema 2).
SCALE_KINDS: Tuple[str, ...] = ("pv", "delta", "gamma", "theta_per_hour", "point_move", "desk_move")
#: Statuses a case may assert for a quantity it cannot define numerically.
EXPECTED_STATUSES: Tuple[str, ...] = ("undefined", "failed")
#: How batch seeds are drawn: schema 1's ``seed + i``, or schema 2's per-case substreams.
SEED_SCHEMES: Tuple[str, ...] = ("sequential", "substream")


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
    #: Schema 2: overrides of the study-level context (valuation instant, phase, profile, history).
    context_params: Mapping[str, Any] = field(default_factory=dict)
    #: Schema 2: quantity -> the status the candidate must report where no number exists.
    expected: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("CaseSpec.name must be a non-empty string")
        bad = {q: s for q, s in self.expected.items() if s not in EXPECTED_STATUSES}
        if bad:
            raise ValidationError(f"CaseSpec.expected statuses must be one of {EXPECTED_STATUSES}, got {bad}")


@dataclass(frozen=True)
class QuantityDefinition:
    """One schema-2 estimand: a stable id, its convention, its scaling rule and unit."""

    id: str
    convention: str      # "pv" | "point" | "desk"
    scale: str           # one of SCALE_KINDS
    unit: str

    def __post_init__(self) -> None:
        if not self.id:
            raise ValidationError("QuantityDefinition.id must be a non-empty string")
        if self.convention not in ("pv", "point", "desk"):
            raise ValidationError(f"QuantityDefinition.convention must be pv, point or desk, got {self.convention!r}")
        if self.scale not in SCALE_KINDS:
            raise ValidationError(f"QuantityDefinition.scale must be one of {SCALE_KINDS}, got {self.scale!r}")


def _catalogue() -> Dict[str, QuantityDefinition]:
    defs = [QuantityDefinition("pv", "pv", "pv", "price per contract")]
    for convention in ("point", "desk"):
        move = "point_move" if convention == "point" else "desk_move"
        move_unit = "per unit {}" if convention == "point" else "PnL per desk {} move"
        defs += [
            QuantityDefinition(f"{convention}_delta", convention, "delta", "per unit spot"),
            QuantityDefinition(f"{convention}_gamma", convention, "gamma", "per unit spot^2"),
            QuantityDefinition(f"{convention}_theta", convention, "theta_per_hour", "PnL per hour"),
            QuantityDefinition(f"{convention}_vega", convention, move, move_unit.format("vol")),
            QuantityDefinition(f"{convention}_rho", convention, move, move_unit.format("rate")),
            QuantityDefinition(f"{convention}_dividend_rho", convention, move, move_unit.format("dividend yield")),
        ]
    return {d.id: d for d in defs}


#: Every quantity a schema-2 study may name. Shared by all products; a study picks a subset.
QUANTITY_CATALOGUE: Dict[str, QuantityDefinition] = _catalogue()


@dataclass(frozen=True)
class QuantityBounds:
    """Per-quantity budget: ``max(abs_floor, rel * |reference|)`` in normalized units."""

    abs_floor: float
    rel: float = 0.0

    def __post_init__(self) -> None:
        if not (self.abs_floor > 0.0):
            raise ValidationError(f"QuantityBounds.abs_floor must be positive, got {self.abs_floor}")
        if self.rel < 0.0:
            raise ValidationError(f"QuantityBounds.rel must be non-negative, got {self.rel}")

    def budget(self, reference_normalized: float) -> float:
        return max(self.abs_floor, self.rel * abs(reference_normalized))


@dataclass(frozen=True)
class NormalizedScale:
    """The spec's normalized axis: PV/N, delta*S/N, gamma*S^2/N, theta-per-hour/N, moves/N."""

    notional: float
    spot_scale: float
    point_move: float = 0.01

    def __post_init__(self) -> None:
        for name in ("notional", "spot_scale", "point_move"):
            if not (getattr(self, name) > 0.0):
                raise ValidationError(f"NormalizedScale.{name} must be positive, got {getattr(self, name)}")

    def to_economic(self, quantity: str, raw: float) -> float:
        try:
            kind = QUANTITY_CATALOGUE[quantity].scale
        except KeyError:
            raise ValidationError(
                f"NormalizedScale cannot convert {quantity!r}: not in the schema-2 quantity catalogue "
                f"{sorted(QUANTITY_CATALOGUE)}"
            ) from None
        n, s = self.notional, self.spot_scale
        if kind in ("pv", "desk_move", "theta_per_hour"):
            return raw / n
        if kind == "delta":
            return raw * s / n
        if kind == "gamma":
            return raw * s * s / n
        return raw * self.point_move / n


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
        seed: Base seed; see :func:`batch_seed` for how batch seeds derive from it.
        bump: Relative spot bump for the paired (common-random-numbers) Greek
            estimates.
        seed_scheme: ``sequential`` (schema 1: batch ``i`` uses ``seed + i``) or
            ``substream`` (schema 2: independent per-case substreams).
    """

    paths_per_batch: int
    min_batches: int
    max_batches: int
    seed: int
    bump: float = 0.01
    seed_scheme: str = "sequential"

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
        if self.seed_scheme not in SEED_SCHEMES:
            raise ValidationError(
                f"SamplingPolicy.seed_scheme must be one of {SEED_SCHEMES}, got {self.seed_scheme!r}"
            )


def batch_seed(policy: SamplingPolicy, case_name: str, batch_index: int) -> int:
    """The seed of batch ``batch_index`` of ``case_name``.

    ``sequential`` is schema 1's rule, ``seed + index`` for every case, kept so banked evidence
    replays. ``substream`` draws each (study seed, case, batch) its own statistically independent
    stream through ``numpy.random.SeedSequence``: cases never share a scramble, the bump arms of one
    case always do (they are priced at this one seed), and a pilot under another study seed shares
    nothing with production.
    """
    if policy.seed_scheme == "sequential":
        return policy.seed + batch_index
    case_key = int.from_bytes(hashlib.sha256(case_name.encode("utf-8")).digest()[:4], "big")
    sequence = np.random.SeedSequence(entropy=policy.seed, spawn_key=(case_key, batch_index))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


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
        schema: Study schema version, one of :data:`SUPPORTED_SCHEMAS`.
        cases: Scenarios to certify; names must be unique.
        quantities: Schema 1: a subset of :data:`QUANTITIES`. Schema 2: a subset
            of :data:`QUANTITY_CATALOGUE`.
        bounds: Pass/fail bounds in economic units (schema 2: fractions of each
            cell's own budget).
        scale: Raw-to-economic converter.
        reference: The stochastic benchmark arm (``ReferenceBuilder``).
        candidates: Deterministic engines under certification
            (``CandidateEvaluator``); one decision is issued per candidate.
        sampling: Sampling budget and stopping limits for the reference arm.
        source_text: Verbatim YAML text when the study was loaded from a file.
            Embedded in evidence so a certificate ships its own definition;
            anchors require it.
        quantity_bounds: Schema 2 only: one budget per certified quantity.
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
    quantity_bounds: Mapping[str, QuantityBounds] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("CertificationStudy.name must be a non-empty string")
        if self.schema not in SUPPORTED_SCHEMAS:
            raise ValidationError(
                f"CertificationStudy.schema must be one of {SUPPORTED_SCHEMAS}, got {self.schema}"
            )
        if not self.cases:
            raise ValidationError("CertificationStudy.cases must not be empty")
        if not self.candidates:
            raise ValidationError("CertificationStudy.candidates must not be empty")
        if not self.quantities:
            raise ValidationError("CertificationStudy.quantities must not be empty")

        if self.schema == 1:
            unknown = [q for q in self.quantities if q not in QUANTITIES]
            if unknown:
                raise ValidationError(
                    f"CertificationStudy.quantities contains unknown entries {unknown}; "
                    f"expected a subset of {QUANTITIES}"
                )
            if self.quantity_bounds:
                raise ValidationError("quantity_bounds is a schema-2 field; a schema 1 study budgets with bounds.cell")
            for case in self.cases:
                if case.expected or case.context_params:
                    raise ValidationError(
                        f"case {case.name!r} declares context or expected statuses, which are schema-2 fields; "
                        "this is a schema 1 study"
                    )
            if self.sampling.seed_scheme != "sequential":
                raise ValidationError(
                    "a schema 1 study samples with seed_scheme 'sequential'; 'substream' is a schema-2 scheme"
                )
        else:
            unknown = [q for q in self.quantities if q not in QUANTITY_CATALOGUE]
            if unknown:
                raise ValidationError(
                    f"CertificationStudy.quantities {unknown} are not in the schema-2 quantity catalogue "
                    f"{sorted(QUANTITY_CATALOGUE)}"
                )
            missing = [q for q in self.quantities if q not in self.quantity_bounds]
            extra = sorted(set(self.quantity_bounds) - set(self.quantities))
            if missing or extra:
                raise ValidationError(
                    f"quantity_bounds must name exactly the study quantities; missing {missing}, extra {extra}"
                )
            for case in self.cases:
                stray = sorted(set(case.expected) - set(self.quantities))
                if stray:
                    raise ValidationError(
                        f"case {case.name!r}: expected statuses name quantities the study does not certify: {stray}"
                    )
            if self.sampling.seed_scheme != "substream":
                raise ValidationError(
                    "a schema-2 study samples on independent per-case substreams: set seed_scheme 'substream'"
                )

        names = [case.name for case in self.cases]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValidationError(
                f"CertificationStudy.cases has duplicate case names: {duplicates}"
            )
