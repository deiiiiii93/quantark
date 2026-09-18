"""The reference arm: a stochastic benchmark in banked batches, or a deterministic one with a declared radius.

The benchmark is sampled in independent batches until the gate-driven stopping
policy says it is sharp enough. Two properties make the result usable as
evidence rather than as a number someone once saw:

**Durability per batch.** Every completed batch is checkpointed immediately, so
an interrupt costs at most one batch -- not the hours already spent.

**Reproducible stopping.** A resumed bank is only trusted when the recorded stop
decision can be *replayed* from the banked batches under the current policy. A
bank longer than the policy allows, or one claiming a budget it never met, is
rejected outright. Otherwise a stale bank could silently supply a stopping point
the current configuration would never have chosen -- which is exactly how a
sequential estimator acquires selection bias.

**Two kinds, never confused.** A deterministic reference (schema 2) states a
value, an error *radius* and the refinement ladder that justifies it. It has no
batches, seeds or standard error, and it is never written as identical batches
with zero standard error: the record is typed, and a consumer that forgets the
radius meets ``None`` where it expected a standard error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.evidence import CheckpointStore
from quantark.modelvalidation.stopping import StopReason, should_stop
from quantark.modelvalidation.study import (
    CaseSpec,
    EconomicScale,
    GateBounds,
    SamplingPolicy,
    batch_seed,
    reference_kind,
)

CHECKPOINT_KIND = "reference"


@dataclass(frozen=True)
class BatchResult:
    """One independent batch of the benchmark, all quantities at once.

    Quantities travel together because they come from one set of paired paths:
    splitting them would either re-simulate or break the pairing that makes the
    Greek estimates precise.
    """

    index: int
    seed: int
    values: Mapping[str, float]


@dataclass(frozen=True)
class ReferenceEstimate:
    """The benchmark estimate: replicate means with standard errors, or a deterministic solve with radii.

    ``kind`` says which. A deterministic estimate has empty ``std_errors`` and no batches; its
    uncertainty is ``radii`` (``math.inf``: the ladder could not bound the quantity), ``evidence`` is
    the ladder, and ``undefined`` names the quantities with no value at this case and why.
    """

    values: Mapping[str, float]
    std_errors: Mapping[str, float]
    batches: int
    seeds: Tuple[int, ...]
    stopped_reason: str
    kind: str = "stochastic"
    radii: Mapping[str, float] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    undefined: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DeterministicResult:
    """One deterministic solve of a case.

    Attributes:
        values: Every quantity that has a value at this case.
        radii: The declared error radius of each value, in the value's units. ``math.inf``
            says the refinement ladder could not bound it; the cell is then unresolved.
        evidence: The ladder behind the radii (levels, per-level values, observed orders),
            JSON-ready. It is banked, reported and never re-derived.
        undefined: quantity -> why it has no value here (a derivative on an event instant).
            A case must declare the matching expected status; a numeric cell without a
            reference value is an error.
    """

    values: Mapping[str, float]
    radii: Mapping[str, float]
    evidence: Mapping[str, Any]
    undefined: Mapping[str, str] = field(default_factory=dict)


class DeterministicReferenceBuilder(Protocol):
    """A deterministic benchmark for one schema-2 study (``reference_kind == "deterministic"``)."""

    reference_kind: str

    def identity(self, case: CaseSpec) -> Mapping[str, Any]:
        """Everything that would change a solved value: inputs, ladder, radius rule, implementation."""
        ...

    def bind(self, policy: SamplingPolicy, quick: bool = False) -> "DeterministicReferenceBuilder":
        """The builder under the effective policy (it supplies the desk bump); ``quick`` selects a wiring ladder."""
        ...

    def error_model(self) -> Mapping[str, Any]:
        """The declared method, its order, and the rule that turns a ladder into a radius (part of the contract)."""
        ...

    def solve(self, case: CaseSpec) -> DeterministicResult:
        ...


class ReferenceBuilder(Protocol):
    """A stochastic benchmark for one study."""

    def identity(self, case: CaseSpec) -> Mapping[str, Any]:
        """Everything that would change the meaning of a banked batch.

        Must include the builder name and params, the case, the quantities, the
        sampling policy, and the bump width -- and must NOT include the
        candidates, so one bank serves every candidate engine.
        """
        ...

    def run_batch(self, case: CaseSpec, batch_index: int) -> BatchResult:
        """Run batch ``batch_index``; seed must be ``batch_seed(policy, case.name, batch_index)``."""
        ...


def reference_targets(builder: Any, quantities: Sequence[str]) -> Dict[str, Optional[dict]]:
    """Which quantities the reference estimates as targets.

    A builder may declare ``targets()``: quantity -> estimator description, or ``None`` for a
    quantity it records only as a proxy with no declared estimator (the pipeline gates
    nothing against such a value). Undeclared quantities are replicate means, which is what
    every schema-1 reference has always been.
    """
    declare = getattr(builder, "targets", None)
    declared = dict(declare()) if callable(declare) else {}
    out: Dict[str, Optional[dict]] = {}
    for quantity in quantities:
        if quantity in declared:
            out[quantity] = None if declared[quantity] is None else dict(declared[quantity])
        else:
            out[quantity] = {"estimator": "replicate_mean"}
    return out


def bound_reference(builder: Any, policy: SamplingPolicy, *, schema: int = 1, quick: bool = False) -> Any:
    """The builder that actually samples under ``policy``.

    Real builders capture their sampling at construction, so the framework's quick policy never
    reached the engine: the payload recorded one path count and the run used another. A schema-2
    builder declares ``bind(policy)`` and the pipeline binds the effective policy before sampling.
    Schema-1 builders are returned unchanged, with the behaviour their banked evidence describes.
    """
    bind = getattr(builder, "bind", None)
    if callable(bind):
        # a deterministic builder has no sampling to shrink: quick selects its wiring ladder instead
        return bind(policy, quick=quick) if reference_kind(builder) == "deterministic" else bind(policy)
    if schema == 2:
        raise ValidationError(
            f"schema-2 reference {type(builder).__name__} must declare bind(policy) so the path count it runs "
            "is the one the evidence records")
    return builder


def _estimate(
    batches: Sequence[BatchResult], quantities: Sequence[str]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Batch means and the standard error of those means."""
    values: Dict[str, float] = {}
    std_errors: Dict[str, float] = {}
    count = len(batches)
    for quantity in quantities:
        samples = np.array([b.values[quantity] for b in batches], dtype=float)
        values[quantity] = float(samples.mean())
        if count < 2:
            # Undefined below two batches; infinite SE can never meet a budget,
            # which is the honest representation of "not yet measurable".
            std_errors[quantity] = math.inf
        else:
            std_errors[quantity] = float(samples.std(ddof=1) / math.sqrt(count))
    return values, std_errors


def _validate_batch(
    batch: BatchResult,
    expected_index: int,
    policy: SamplingPolicy,
    quantities: Sequence[str],
    case_name: str,
) -> None:
    if batch.index != expected_index:
        raise ValidationError(
            f"Reference builder returned batch index {batch.index}, expected "
            f"{expected_index}"
        )
    expected_seed = batch_seed(policy, case_name, expected_index)
    if batch.seed != expected_seed:
        raise ValidationError(
            f"Reference builder used seed {batch.seed} for batch {expected_index} of {case_name!r}, "
            f"expected {expected_seed} ({policy.seed_scheme} scheme)"
        )
    for quantity in quantities:
        if quantity not in batch.values:
            raise ValidationError(
                f"Reference batch {expected_index} is missing quantity {quantity!r}"
            )
        value = batch.values[quantity]
        if not math.isfinite(value):
            raise ValidationError(
                f"Reference batch {expected_index} produced non-finite {quantity}: {value}"
            )


def _serialize(batches: Sequence[BatchResult], stopped_reason: Optional[str]) -> dict:
    return {
        "batches": [
            {"index": b.index, "seed": b.seed, "values": dict(b.values)} for b in batches
        ],
        "stopped_reason": stopped_reason,
    }


def _deserialize(payload: Mapping[str, Any]) -> Tuple[List[BatchResult], Optional[str]]:
    batches = [
        BatchResult(index=int(b["index"]), seed=int(b["seed"]), values=dict(b["values"]))
        for b in payload["batches"]
    ]
    return batches, payload.get("stopped_reason")


def _stop_budgets(
    values: Mapping[str, float],
    quantity_bounds: Optional[Mapping[str, Any]],
    stop_quantities: Optional[Sequence[str]],
    scale: EconomicScale,
) -> Optional[Dict[str, float]]:
    """Schema 2: each targeted quantity's budget at the current estimate; None keeps schema 1's single bound."""
    if quantity_bounds is None:
        return None
    return {
        q: quantity_bounds[q].budget(scale.to_economic(q, values[q]))
        for q in (stop_quantities if stop_quantities is not None else quantity_bounds)
    }


def _validate_banked_bank(
    batches: Sequence[BatchResult],
    stopped_reason: Optional[str],
    quantities: Sequence[str],
    scale: EconomicScale,
    bounds: GateBounds,
    policy: SamplingPolicy,
    case_name: str,
    quantity_bounds: Optional[Mapping[str, Any]] = None,
    stop_quantities: Optional[Sequence[str]] = None,
) -> None:
    """Replay the stopping policy over a banked bank; reject what it cannot explain."""
    if len(batches) > policy.max_batches:
        raise ValidationError(
            f"Banked reference holds {len(batches)} batches but the policy caps at "
            f"{policy.max_batches}; the bank did not come from this configuration"
        )
    for position, batch in enumerate(batches):
        _validate_batch(batch, position, policy, quantities, case_name)

    if stopped_reason is None:
        # An interrupted bank: no decision was recorded, so there is nothing to
        # contradict. Sampling continues from here.
        return

    values, std_errors = _estimate(batches, quantities)
    replayed = should_stop(
        std_errors_raw=std_errors,
        batches=len(batches),
        scale=scale,
        bounds=bounds,
        policy=policy,
        budgets_c=_stop_budgets(values, quantity_bounds, stop_quantities, scale),
    )
    if not replayed.stop or replayed.reason.value != stopped_reason:
        raise ValidationError(
            f"Banked reference claims it stopped for {stopped_reason!r} after "
            f"{len(batches)} batches, but replaying the policy gives "
            f"{replayed.reason.value!r} (stop={replayed.stop}). Refusing to reuse it."
        )


def _validate_deterministic(result: DeterministicResult, quantities: Sequence[str], case_name: str) -> None:
    for quantity in quantities:
        if quantity in result.undefined:
            if quantity in result.values:
                raise ValidationError(f"deterministic reference for {case_name!r} gives {quantity} a value and calls it undefined")
            continue
        if quantity not in result.values or quantity not in result.radii:
            raise ValidationError(f"deterministic reference for {case_name!r} is missing a value or a radius for {quantity!r}")
        value, radius = result.values[quantity], result.radii[quantity]
        if not math.isfinite(value):
            raise ValidationError(f"deterministic reference for {case_name!r} produced non-finite {quantity}: {value}")
        if math.isnan(radius) or radius < 0.0:
            raise ValidationError(f"deterministic reference for {case_name!r} has an invalid radius for {quantity}: {radius}")


def _deterministic_estimate(result: DeterministicResult) -> ReferenceEstimate:
    return ReferenceEstimate(values=dict(result.values), std_errors={}, batches=0, seeds=(), stopped_reason="deterministic",
                             kind="deterministic", radii=dict(result.radii), evidence=dict(result.evidence),
                             undefined=dict(result.undefined))


def run_deterministic_reference(
    builder: DeterministicReferenceBuilder,
    case: CaseSpec,
    quantities: Sequence[str],
    store: Optional[CheckpointStore] = None,
    resume: bool = False,
) -> ReferenceEstimate:
    """Solve ``case`` once, or reuse the banked solve whose identity still matches."""
    identity = builder.identity(case)
    if resume and store is not None:
        banked = store.load(CHECKPOINT_KIND, case.name, identity)
        if banked is not None:
            if banked.get("kind") != "deterministic":
                raise ValidationError(f"banked reference for {case.name!r} is not a deterministic solve")
            result = DeterministicResult(values=dict(banked["values"]), radii=dict(banked["radii"]),
                                         evidence=dict(banked["evidence"]), undefined=dict(banked["undefined"]))
            _validate_deterministic(result, quantities, case.name)
            return _deterministic_estimate(result)
    result = builder.solve(case)
    _validate_deterministic(result, quantities, case.name)
    if store is not None:
        store.save(CHECKPOINT_KIND, case.name, identity,
                   {"kind": "deterministic", "values": dict(result.values), "radii": dict(result.radii),
                    "evidence": dict(result.evidence), "undefined": dict(result.undefined)})
    return _deterministic_estimate(result)


def run_reference(
    builder: ReferenceBuilder,
    case: CaseSpec,
    quantities: Sequence[str],
    scale: EconomicScale,
    bounds: GateBounds,
    policy: SamplingPolicy,
    store: Optional[CheckpointStore] = None,
    resume: bool = False,
    quantity_bounds: Optional[Mapping[str, Any]] = None,
    stop_quantities: Optional[Sequence[str]] = None,
    checkpoint_kind: str = CHECKPOINT_KIND,
) -> ReferenceEstimate:
    """Sample the benchmark for ``case`` until the stopping policy says stop.

    A deterministic builder is solved once instead (:func:`run_deterministic_reference`).
    ``checkpoint_kind`` keeps a qualifying arm's bank apart from the reference's.

    Args:
        builder: The stochastic benchmark.
        case: The scenario being certified.
        quantities: Quantities to estimate (all produced by each batch).
        scale: Raw-to-economic converter.
        bounds: Study bounds (supplies the standard-error budget).
        policy: Sampling budget and limits.
        store: Checkpoint store; ``None`` disables banking.
        resume: Reuse a banked bank when its identity and stop decision hold up.
        quantity_bounds: Schema 2: per-quantity budgets; the stop rule measures
            each quantity's SE against its own budget. ``None`` for schema 1.
        stop_quantities: Schema 2: the targeted quantities that drive stopping
            (an untargeted proxy never does).

    Raises:
        ValidationError: a malformed batch, or a banked bank the current policy
            cannot explain.
    """
    if reference_kind(builder) == "deterministic":
        return run_deterministic_reference(builder, case, quantities, store, resume)
    identity = builder.identity(case)
    batches: List[BatchResult] = []
    stopped_reason: Optional[str] = None

    if resume and store is not None:
        banked = store.load(checkpoint_kind, case.name, identity)
        if banked is not None:
            batches, stopped_reason = _deserialize(banked)
            _validate_banked_bank(
                batches, stopped_reason, quantities, scale, bounds, policy, case.name,
                quantity_bounds=quantity_bounds, stop_quantities=stop_quantities,
            )
            if stopped_reason is not None:
                values, std_errors = _estimate(batches, quantities)
                return ReferenceEstimate(
                    values=values,
                    std_errors=std_errors,
                    batches=len(batches),
                    seeds=tuple(b.seed for b in batches),
                    stopped_reason=stopped_reason,
                )

    while True:
        batch = builder.run_batch(case, len(batches))
        _validate_batch(batch, len(batches), policy, quantities, case.name)
        batches.append(batch)

        values, std_errors = _estimate(batches, quantities)
        decision = should_stop(
            std_errors_raw=std_errors,
            batches=len(batches),
            scale=scale,
            bounds=bounds,
            policy=policy,
            budgets_c=_stop_budgets(values, quantity_bounds, stop_quantities, scale),
        )
        stopped_reason = decision.reason.value if decision.stop else None

        if store is not None:
            store.save(
                checkpoint_kind, case.name, identity, _serialize(batches, stopped_reason)
            )

        if decision.stop:
            break

    values, std_errors = _estimate(batches, quantities)
    return ReferenceEstimate(
        values=values,
        std_errors=std_errors,
        batches=len(batches),
        seeds=tuple(b.seed for b in batches),
        stopped_reason=stopped_reason or StopReason.MAX_BATCHES.value,
    )
