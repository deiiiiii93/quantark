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


def decide_cell(gate: Optional[CellGateResult], error: bool) -> Verdict:
    """Turn one cell gate into a verdict.

    Args:
        gate: The cell gate result, or ``None`` when the cell errored.
        error: True when the engine or benchmark raised for this cell.

    Raises:
        ValidationError: no gate and no error -- the caller lost a result.
    """
    if error:
        return Verdict.ERROR
    if gate is None:
        raise ValidationError(
            "decide_cell requires a gate result when the cell did not error"
        )
    if not gate.se_budget_met:
        return Verdict.UNRESOLVED
    return Verdict.PASS if gate.passed else Verdict.FAIL


def decide_candidate(
    cell_verdicts: Sequence[Verdict],
    aggregates: Sequence[AggregateGateResult],
) -> Decision:
    """Combine one candidate's cell verdicts and aggregate gates.

    ``REJECTED`` requires *confident* evidence of disagreement: a FAIL cell
    (whose benchmark met budget by construction), or an aggregate tilt measured
    with adequate standard error. Everything else that is not a clean sweep is
    ``INCONCLUSIVE``.
    """
    verdicts = list(cell_verdicts)

    confident_cell_failure = any(v is Verdict.FAIL for v in verdicts)
    confident_aggregate_failure = any(
        agg.se_adequate and not agg.within_bound for agg in aggregates
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
