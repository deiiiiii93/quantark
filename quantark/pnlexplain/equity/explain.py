"""Two-snapshot explain orchestrator (spec §5.6)."""
from __future__ import annotations

from typing import List, Optional

from quantark.pnlexplain.base import MARKET_FACTORS, ExplainMethod, ExplainRow, PnLExplainResult, make_total_row
from quantark.pnlexplain.config import PnLExplainConfig
from quantark.pnlexplain.equity.coordinate import resolve_coordinate
from quantark.pnlexplain.equity.factor_diff import build_factor_moves, validate_pair
from quantark.pnlexplain.equity.lifecycle import LifecycleTransition, event_row, resolve_transition
from quantark.pnlexplain.equity.scenario import ScenarioCache
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot
from quantark.pnlexplain.equity.waterfall import sequential_rows, shapley_rows
from quantark.util.exceptions import NumericalError
from quantark.util.numerical import is_close

LEVEL = "instrument"


def explain(
    snapshot_t0: ValuationSnapshot,
    snapshot_t1: ValuationSnapshot,
    *,
    config: Optional[PnLExplainConfig] = None,
    transition: Optional[LifecycleTransition] = None,
) -> PnLExplainResult:
    config = config if config is not None else PnLExplainConfig()
    validate_pair(snapshot_t0, snapshot_t1)
    days = (snapshot_t1.date - snapshot_t0.date).days
    transition = resolve_transition(snapshot_t0, snapshot_t1, transition, calendar_days=days)
    coordinate = resolve_coordinate(
        snapshot_t0.product, float(snapshot_t0.pricing_env.spot),
        transition.product_alive_t1, snapshot_t1.pricing_env,
    )
    moves = build_factor_moves(
        snapshot_t0, snapshot_t1, coordinate,
        engine_alive_t1=transition.engine_alive_t1, lifecycle_changed=transition.changed,
    )
    cache = ScenarioCache(snapshot_t0, snapshot_t1, transition, moves, config)
    pv_t0 = cache.value_for(())
    pv_alive_t1 = cache.all_market()
    pv_t1 = cache.value_t1()
    total_pnl = pv_t1.total - pv_t0.total

    # Explicit gates for modes built by later tasks (each is removed by the task that
    # implements the mode; a gate is never a fallback, it refuses).
    if ExplainMethod.TAYLOR in config.methods:
        raise NotImplementedError("the Taylor explainer lands in Task 8")
    rows: List[ExplainRow] = []
    if ExplainMethod.WATERFALL in config.methods:
        if config.interaction == "sequential":
            rows.extend(sequential_rows(cache, config.waterfall_order, LEVEL))
        else:
            rows.extend(shapley_rows(cache, LEVEL))
    rows.append(event_row(cache, transition, LEVEL))
    unexplained: Optional[float] = None
    rows.append(make_total_row(LEVEL, total_pnl))

    metadata = {
        "time_pure": cache.time_pure(),
        "effective_factors": tuple(f.value for f in MARKET_FACTORS if f in cache.effective),
        "coordinate": (coordinate.reference_strike, coordinate.tenor_t1),
        "transition_changed": transition.changed,
        "interaction": config.interaction,
    }
    result = PnLExplainResult(
        date_t0=snapshot_t0.date, date_t1=snapshot_t1.date, pv_t0=pv_t0,
        pv_alive_t1=pv_alive_t1, pv_t1=pv_t1, total_pnl=total_pnl, moves=moves,
        rows=tuple(rows), unexplained=unexplained, metadata=metadata,
    )
    if ExplainMethod.WATERFALL in config.methods:
        gap = result.reconcile(ExplainMethod.WATERFALL)
        if not is_close(gap, 0.0, rel_tol=0.0, abs_tol=1e-10 * max(1.0, abs(total_pnl))):
            raise NumericalError(f"waterfall does not reconcile: gap {gap}")
    return result
