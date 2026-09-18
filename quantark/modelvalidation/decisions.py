"""Cell verdicts and per-candidate decisions.

The lattice separates three genuinely different outcomes: the engine disagreed
(``REJECTED``), the engine agreed (``ADMITTED``), and the evidence cannot say
(``INCONCLUSIVE``). That third state is the point -- a certification framework
that collapses "we could not tell" into a pass or a fail is worse than useless,
because both collapses are lies about what was measured.

A ``FAIL`` cell is only reachable when the benchmark met its standard-error
budget, so a failure is always a *confident* failure; noise produces
``UNRESOLVED`` instead.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Sequence

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.gates import AggregateGateResult, CellGateResult


class Verdict(str, Enum):
    """Outcome for one cell (case x quantity x candidate)."""

    PASS = "PASS"
    FAIL = "FAIL"
    #: The engine (or the benchmark) raised; nothing was measured.
    ERROR = "ERROR"
    #: The benchmark never got sharp enough to discriminate at this bound.
    UNRESOLVED = "UNRESOLVED"


class Decision(str, Enum):
    """Outcome for one candidate engine across the whole study."""

    ADMITTED = "ADMITTED"
    REJECTED = "REJECTED"
    INCONCLUSIVE = "INCONCLUSIVE"


def decide_cell(gate: Optional[CellGateResult], error: bool, *, schema: int = 1) -> Verdict:
    """Turn one cell gate into a verdict.

    Args:
        gate: The cell gate result, or ``None`` when the cell errored.
        error: True when the engine or benchmark raised for this cell.
        schema: Study schema. Schema 1 fails any interval beyond the bound;
            schema 2 fails only when the whole interval lies beyond it and
            leaves a straddle UNRESOLVED.

    Raises:
        ValidationError: no gate and no error -- the caller lost a result --
            or an unknown schema.
    """
    if error:
        return Verdict.ERROR
    if gate is None:
        raise ValidationError(
            "decide_cell requires a gate result when the cell did not error"
        )
    if schema not in (1, 2):
        raise ValidationError(f"decide_cell knows schemas 1 and 2, got {schema}")
    if not gate.se_budget_met:
        return Verdict.UNRESOLVED
    if schema == 1:
        return Verdict.PASS if gate.passed else Verdict.FAIL
    # Schema 2: uncertainty consumes the budget. A declared convergence requirement that is
    # demonstrably violated is a failure; a disagreement interval that straddles the budget
    # is not evidence either way.
    if not gate.envelope_within_bound:
        return Verdict.FAIL
    if gate.interval_within_bound:
        return Verdict.PASS
    if gate.lower_exceeds_bound:
        return Verdict.FAIL
    return Verdict.UNRESOLVED


def decide_candidate(
    cell_verdicts: Sequence[Verdict],
    aggregates: Sequence[AggregateGateResult],
    *,
    schema: int = 1,
) -> Decision:
    """Combine one candidate's cell verdicts and aggregate gates.

    ``REJECTED`` requires *confident* evidence of disagreement: a FAIL cell
    (whose benchmark met budget by construction), or an aggregate tilt measured
    with adequate standard error -- under schema 2, one whose whole uncertainty
    interval lies beyond the bound. Everything else that is not a clean sweep is
    ``INCONCLUSIVE``.
    """
    if schema not in (1, 2):
        raise ValidationError(f"decide_candidate knows schemas 1 and 2, got {schema}")
    verdicts = list(cell_verdicts)

    confident_cell_failure = any(v is Verdict.FAIL for v in verdicts)
    if schema == 1:
        confident_aggregate_failure = any(
            agg.se_adequate and not agg.within_bound for agg in aggregates
        )
    else:
        # the whole uncertainty interval of the mean signed error lies beyond the bound
        confident_aggregate_failure = any(
            agg.se_adequate and agg.lower_exceeds_bound for agg in aggregates
        )
    if confident_cell_failure or confident_aggregate_failure:
        return Decision.REJECTED

    if not verdicts:
        return Decision.INCONCLUSIVE

    all_cells_pass = all(v is Verdict.PASS for v in verdicts)
    all_aggregates_pass = all(agg.passed for agg in aggregates)
    if all_cells_pass and all_aggregates_pass:
        return Decision.ADMITTED

    return Decision.INCONCLUSIVE
