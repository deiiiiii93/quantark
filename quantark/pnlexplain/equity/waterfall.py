"""Sequential full-revaluation waterfall (spec §6)."""
from __future__ import annotations

from typing import Sequence, Tuple

from quantark.pnlexplain.base import ExplainMethod, ExplainRow, Factor, RowKind
from quantark.pnlexplain.equity.scenario import ScenarioCache


def _row(cache: ScenarioCache, factor: Factor, pnl: float, level: str, step) -> ExplainRow:
    """Build one waterfall row; never prices."""
    metadata = {
        "changed": factor in cache.effective,
        "applicable": factor in cache.moves.coordinate.applicable,
    }
    if factor is Factor.MODEL:
        metadata["engine"] = type(cache.bump_engine_alive).__name__
    return ExplainRow(
        factor=factor, term=factor.value, method=ExplainMethod.WATERFALL, kind=RowKind.COMPONENT,
        level=level, pnl=pnl, moves=cache.moves.display(factor), step=step, metadata=metadata,
    )


def sequential_rows(cache: ScenarioCache, order: Sequence[Factor], level: str = "instrument"
                    ) -> Tuple[ExplainRow, ...]:
    applied = set()
    previous = cache.value_for(()).total
    rows = []
    for step, factor in enumerate(order, start=1):
        applied.add(factor)
        if factor in cache.effective:
            current = cache.value_for(applied).total
            pnl, previous = current - previous, current
        else:
            pnl = 0.0
        rows.append(_row(cache, factor, pnl, level, step))
    return tuple(rows)
