"""The deterministic candidate arm: values plus a self-assessed error bound.

Agreeing with the benchmark is not by itself evidence that an engine is
converged -- a coarse grid can land on the right answer by luck. So a candidate
reports two things: its value at the target configuration, and a *refinement
ladder* showing how that value moves when each axis is coarsened.

The gap between the target rung and the next-coarser rung estimates that axis's
remaining discretization error. Summing the per-axis gaps (rather than taking
the largest) is deliberately conservative: axes can err in the same direction,
and an envelope that under-reports is worse than one that over-reports.

Schema 2 replaces the two-rung ladder with typed convergence axes of at least
three levels (:class:`ConvergenceAxis`), per-quantity statuses for outputs that
have no number (``undefined`` / ``failed``), and an identity over the resolved
inputs and the implementation. Schema-1 results keep their exact shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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


#: What a candidate may report for a quantity: a number, no derivative at this point, or a
#: computation that produced nothing usable.
CANDIDATE_STATUSES: Tuple[str, ...] = ("ok", "undefined", "failed")
CONVERGENCE_KINDS: Tuple[str, ...] = ("refinement", "placement")
#: Schema 2: fewer levels than this on a declared axis is missing evidence, not a pass.
MIN_CONVERGENCE_LEVELS = 3


@dataclass(frozen=True)
class ConvergenceLevel:
    """One level of a schema-2 convergence axis, with the settings it actually ran at."""

    label: str
    resolution: float                       # increases with refinement; a placement level uses its shift
    values: Mapping[str, float]
    settings: Mapping[str, Any] = field(default_factory=dict)
    is_target: bool = False


@dataclass(frozen=True)
class ConvergenceAxis:
    """A refinement axis (coarser to finer around the shipped target) or a placement axis (grid shifts)."""

    name: str
    kind: str
    levels: Tuple[ConvergenceLevel, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("ConvergenceAxis.name must be a non-empty string")
        if self.kind not in CONVERGENCE_KINDS:
            raise ValidationError(f"ConvergenceAxis.kind must be one of {CONVERGENCE_KINDS}, got {self.kind!r}")
        if sum(1 for level in self.levels if level.is_target) != 1:
            raise ValidationError(f"ConvergenceAxis {self.name!r} needs exactly one target level")


@dataclass(frozen=True)
class ConvergenceEvidence:
    """What a candidate's axes establish for one quantity."""

    envelope: Optional[float]               # None when the evidence is incomplete
    complete: bool
    missing: Tuple[str, ...]                # axes with fewer than MIN_CONVERGENCE_LEVELS usable levels
    observed_orders: Mapping[str, Optional[float]]
    non_monotone: Tuple[str, ...]           # refinement axes whose successive differences grow


@dataclass(frozen=True)
class CandidateResult:
    """A candidate's target values, its schema-1 ladders, and its schema-2 statuses and convergence axes."""

    values: Mapping[str, float]
    ladders: Tuple[LadderRung, ...] = ()
    statuses: Mapping[str, str] = field(default_factory=dict)
    reasons: Mapping[str, str] = field(default_factory=dict)
    convergence: Tuple[ConvergenceAxis, ...] = ()
    #: A closed form with no discretisation to refine: complete evidence with a zero envelope.
    exact: bool = False

    def __post_init__(self) -> None:
        if not self.values:
            raise ValidationError("CandidateResult.values must not be empty")
        for quantity, status in self.statuses.items():
            if status not in CANDIDATE_STATUSES:
                raise ValidationError(
                    f"CandidateResult.statuses[{quantity!r}] must be one of {CANDIDATE_STATUSES}, got {status!r}")
            if status != "ok" and quantity in self.values:
                raise ValidationError(f"CandidateResult reports {quantity} as {status} but also carries a value for it")

    def status(self, quantity: str) -> str:
        """``ok`` for a reported value, the declared status otherwise, ``failed`` when neither exists."""
        if quantity in self.statuses:
            return self.statuses[quantity]
        return "ok" if quantity in self.values else "failed"


def convergence_evidence(result: CandidateResult, quantity: str) -> ConvergenceEvidence:
    """The refinement envelope of ``quantity`` and whether the evidence behind it is complete.

    Per refinement axis the increment is the gap between the target and its next FINER level
    (the better estimate of what is left), falling back to the next coarser one; per placement
    axis it is the largest gap to any shifted level. The envelope is their sum, as in schema 1.
    An axis with fewer than three levels carrying the quantity is missing evidence.
    """
    if result.exact:
        return ConvergenceEvidence(0.0, True, (), {}, ())
    if not result.convergence:
        return ConvergenceEvidence(None, False, ("<no axis declared>",), {}, ())
    increments, missing, orders, non_monotone = [], [], {}, []
    for axis in result.convergence:
        levels = sorted((lv for lv in axis.levels if quantity in lv.values and math.isfinite(lv.values[quantity])),
                        key=lambda lv: lv.resolution)
        target = next((i for i, lv in enumerate(levels) if lv.is_target), None)
        if target is None or len(levels) < MIN_CONVERGENCE_LEVELS:
            missing.append(axis.name)
            continue
        v = [lv.values[quantity] for lv in levels]
        if axis.kind == "placement":
            increments.append(max(abs(x - v[target]) for i, x in enumerate(v) if i != target))
            continue
        neighbour = target + 1 if target + 1 < len(v) else target - 1
        increments.append(abs(v[target] - v[neighbour]))
        a = max(0, min(target, len(v) - 3))                       # the three levels starting at (or nearest) the target
        d1, d2 = abs(v[a + 1] - v[a]), abs(v[a + 2] - v[a + 1])
        ratio = levels[a + 1].resolution / levels[a].resolution
        orders[axis.name] = math.log(d1 / d2) / math.log(ratio) if d1 > 0.0 and d2 > 0.0 and ratio > 1.0 else None
        if d2 > d1:
            non_monotone.append(axis.name)
    if missing:
        return ConvergenceEvidence(None, False, tuple(missing), orders, tuple(non_monotone))
    return ConvergenceEvidence(float(sum(increments)), True, (), orders, tuple(non_monotone))


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
    evaluator: CandidateEvaluator, case: CaseSpec, *, schema: int = 1
) -> Mapping[str, Any]:
    """Everything that would change a candidate evaluation, for checkpoint reuse.

    Schema 1 keeps its original shape exactly. Schema 2 adds what that shape cannot see: the case's
    context and semantic expectations, the candidate's resolved study-level and case inputs, and a
    digest of its implementation. A schema-2 candidate that cannot declare those is refused, because a
    checkpoint whose identity ignores them can be reused against a different market or clock.
    """
    identity = {
        "candidate": evaluator.name(),
        "params": dict(evaluator.params()),
        "case": {
            "name": case.name,
            "environment_params": dict(case.environment_params),
            "product_params": dict(case.product_params),
        },
    }
    if schema == 1:
        return identity
    resolved, fingerprint = getattr(evaluator, "resolved_inputs", None), getattr(evaluator, "fingerprint", None)
    if not callable(resolved) or not callable(fingerprint):
        raise ValidationError(
            f"schema-2 candidate {evaluator.name()!r} must declare resolved_inputs(case) and fingerprint(): its "
            "study-level inputs and its implementation enter the checkpoint identity")
    identity["case"]["context_params"] = dict(case.context_params)
    identity["case"]["expected"] = dict(case.expected)
    identity["inputs"] = dict(resolved(case))
    identity["implementation"] = str(fingerprint())
    return identity


def _serialize_axis(axis: ConvergenceAxis) -> dict:
    return {
        "name": axis.name,
        "kind": axis.kind,
        "levels": [
            {"label": lv.label, "resolution": lv.resolution, "values": dict(lv.values),
             "settings": dict(lv.settings), "is_target": lv.is_target}
            for lv in axis.levels
        ],
    }


def _deserialize_axis(payload: Mapping[str, Any]) -> ConvergenceAxis:
    return ConvergenceAxis(
        name=payload["name"],
        kind=payload["kind"],
        levels=tuple(
            ConvergenceLevel(label=lv["label"], resolution=float(lv["resolution"]), values=dict(lv["values"]),
                             settings=dict(lv.get("settings", {})), is_target=bool(lv.get("is_target", False)))
            for lv in payload["levels"]
        ),
    )


def serialize_candidate_result(result: CandidateResult) -> dict:
    """Represent a candidate result for checkpointing.

    The schema-2 fields are written only when set, so a schema-1 result keeps
    exactly its original two-key shape.
    """
    payload: dict = {
        "values": dict(result.values),
        "ladders": [
            {"axis": r.axis, "level": r.level, "values": dict(r.values)}
            for r in result.ladders
        ],
    }
    if result.statuses:
        payload["statuses"] = dict(result.statuses)
    if result.reasons:
        payload["reasons"] = dict(result.reasons)
    if result.exact:
        payload["exact"] = True
    if result.convergence:
        payload["convergence"] = [_serialize_axis(axis) for axis in result.convergence]
    return payload


def deserialize_candidate_result(payload: Mapping[str, Any]) -> CandidateResult:
    """Rebuild a candidate result from its checkpoint."""
    return CandidateResult(
        values=dict(payload["values"]),
        ladders=tuple(
            LadderRung(axis=r["axis"], level=r["level"], values=dict(r["values"]))
            for r in payload.get("ladders", [])
        ),
        statuses=dict(payload.get("statuses", {})),
        reasons=dict(payload.get("reasons", {})),
        convergence=tuple(_deserialize_axis(axis) for axis in payload.get("convergence", [])),
        exact=bool(payload.get("exact", False)),
    )
