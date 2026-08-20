"""The deterministic candidate arm: values plus a self-assessed error bound.

Agreeing with the benchmark is not by itself evidence that an engine is
converged -- a coarse grid can land on the right answer by luck. So a candidate
reports two things: its value at the target configuration, and a *refinement
ladder* showing how that value moves when each axis is coarsened.

The gap between the target rung and the next-coarser rung estimates that axis's
remaining discretization error. Summing the per-axis gaps (rather than taking
the largest) is deliberately conservative: axes can err in the same direction,
and an envelope that under-reports is worse than one that over-reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.study import CaseSpec

#: Ladder levels, coarse to fine. ``target`` is the configuration being certified.
LADDER_LEVELS: Tuple[str, ...] = ("coarse", "medium", "target")

CHECKPOINT_KIND = "candidate"


@dataclass(frozen=True)
class LadderRung:
    """One rung of a refinement ladder: one axis at one level of resolution."""

    axis: str
    level: str
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.axis:
            raise ValidationError("LadderRung.axis must be a non-empty string")
        if self.level not in LADDER_LEVELS:
            raise ValidationError(
                f"LadderRung.level must be one of {LADDER_LEVELS}, got {self.level!r}"
            )


@dataclass(frozen=True)
class CandidateResult:
    """A candidate's target-configuration values and its refinement ladders."""

    values: Mapping[str, float]
    ladders: Tuple[LadderRung, ...] = ()

    def __post_init__(self) -> None:
        if not self.values:
            raise ValidationError("CandidateResult.values must not be empty")


class CandidateEvaluator(Protocol):
    """A deterministic engine under certification."""

    def name(self) -> str:
        """Stable identifier; one decision is issued per name."""
        ...

    def params(self) -> Mapping[str, Any]:
        """Configuration recorded in evidence and in the identity hash."""
        ...

    def evaluate(self, case: CaseSpec) -> CandidateResult:
        """Value every study quantity for ``case`` at the target configuration."""
        ...


def envelope_from_ladders(
    ladders: Sequence[LadderRung], quantity: str
) -> Optional[float]:
    """Estimate the candidate's own discretization error for ``quantity``.

    Per axis, the increment is ``|target - medium|``; the envelope is their sum.

    Returns:
        The envelope, or ``None`` when no axis has both a target and a medium
        rung carrying this quantity -- the honest answer when no ladder ran.

    Raises:
        ValidationError: a rung carries a non-finite value.
    """
    by_axis: Dict[str, Dict[str, float]] = {}
    for rung in ladders:
        if quantity not in rung.values:
            continue
        value = rung.values[quantity]
        if not math.isfinite(value):
            raise ValidationError(
                f"Ladder rung {rung.axis}/{rung.level} has non-finite {quantity}: {value}"
            )
        by_axis.setdefault(rung.axis, {})[rung.level] = value

    increments = [
        abs(levels["target"] - levels["medium"])
        for levels in by_axis.values()
        if "target" in levels and "medium" in levels
    ]
    if not increments:
        return None
    return float(sum(increments))


def candidate_identity(
    evaluator: CandidateEvaluator, case: CaseSpec
) -> Mapping[str, Any]:
    """Everything that would change a candidate evaluation, for checkpoint reuse."""
    return {
        "candidate": evaluator.name(),
        "params": dict(evaluator.params()),
        "case": {
            "name": case.name,
            "environment_params": dict(case.environment_params),
            "product_params": dict(case.product_params),
        },
    }


def serialize_candidate_result(result: CandidateResult) -> dict:
    """Represent a candidate result for checkpointing."""
    return {
        "values": dict(result.values),
        "ladders": [
            {"axis": r.axis, "level": r.level, "values": dict(r.values)}
            for r in result.ladders
        ],
    }


def deserialize_candidate_result(payload: Mapping[str, Any]) -> CandidateResult:
    """Rebuild a candidate result from its checkpoint."""
    return CandidateResult(
        values=dict(payload["values"]),
        ladders=tuple(
            LadderRung(axis=r["axis"], level=r["level"], values=dict(r["values"]))
            for r in payload.get("ladders", [])
        ),
    )
