"""The stochastic reference arm: banked batches, sound resume.

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
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    """The benchmark estimate assembled from banked batches."""

    values: Mapping[str, float]
    std_errors: Mapping[str, float]
    batches: int
    seeds: Tuple[int, ...]
    stopped_reason: str


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
        """Run batch ``batch_index``; seed must be ``policy.seed + batch_index``."""
        ...


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
) -> None:
    if batch.index != expected_index:
        raise ValidationError(
            f"Reference builder returned batch index {batch.index}, expected "
            f"{expected_index}"
        )
    expected_seed = policy.seed + expected_index
    if batch.seed != expected_seed:
        raise ValidationError(
            f"Reference builder used seed {batch.seed} for batch {expected_index}, "
            f"expected {expected_seed} (policy.seed + index)"
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


def _validate_banked_bank(
    batches: Sequence[BatchResult],
    stopped_reason: Optional[str],
    quantities: Sequence[str],
    scale: EconomicScale,
    bounds: GateBounds,
    policy: SamplingPolicy,
) -> None:
    """Replay the stopping policy over a banked bank; reject what it cannot explain."""
    if len(batches) > policy.max_batches:
        raise ValidationError(
            f"Banked reference holds {len(batches)} batches but the policy caps at "
            f"{policy.max_batches}; the bank did not come from this configuration"
        )
    for position, batch in enumerate(batches):
        _validate_batch(batch, position, policy, quantities)

    if stopped_reason is None:
        # An interrupted bank: no decision was recorded, so there is nothing to
        # contradict. Sampling continues from here.
        return

    _, std_errors = _estimate(batches, quantities)
    replayed = should_stop(
        std_errors_raw=std_errors,
        batches=len(batches),
        scale=scale,
        bounds=bounds,
        policy=policy,
    )
    if not replayed.stop or replayed.reason.value != stopped_reason:
        raise ValidationError(
            f"Banked reference claims it stopped for {stopped_reason!r} after "
            f"{len(batches)} batches, but replaying the policy gives "
            f"{replayed.reason.value!r} (stop={replayed.stop}). Refusing to reuse it."
        )


def run_reference(
    builder: ReferenceBuilder,
    case: CaseSpec,
    quantities: Sequence[str],
    scale: EconomicScale,
    bounds: GateBounds,
    policy: SamplingPolicy,
    store: Optional[CheckpointStore] = None,
    resume: bool = False,
) -> ReferenceEstimate:
    """Sample the benchmark for ``case`` until the stopping policy says stop.

    Args:
        builder: The stochastic benchmark.
        case: The scenario being certified.
        quantities: Quantities to estimate (all produced by each batch).
        scale: Raw-to-economic converter.
        bounds: Study bounds (supplies the standard-error budget).
        policy: Sampling budget and limits.
        store: Checkpoint store; ``None`` disables banking.
        resume: Reuse a banked bank when its identity and stop decision hold up.

    Raises:
        ValidationError: a malformed batch, or a banked bank the current policy
            cannot explain.
    """
    identity = builder.identity(case)
    batches: List[BatchResult] = []
    stopped_reason: Optional[str] = None

    if resume and store is not None:
        banked = store.load(CHECKPOINT_KIND, case.name, identity)
        if banked is not None:
            batches, stopped_reason = _deserialize(banked)
            _validate_banked_bank(
                batches, stopped_reason, quantities, scale, bounds, policy
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
        _validate_batch(batch, len(batches), policy, quantities)
        batches.append(batch)

        _, std_errors = _estimate(batches, quantities)
        decision = should_stop(
            std_errors_raw=std_errors,
            batches=len(batches),
            scale=scale,
            bounds=bounds,
            policy=policy,
        )
        stopped_reason = decision.reason.value if decision.stop else None

        if store is not None:
            store.save(
                CHECKPOINT_KIND, case.name, identity, _serialize(batches, stopped_reason)
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
