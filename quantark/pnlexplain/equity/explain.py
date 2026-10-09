"""Two-snapshot explain orchestrator (spec §5.6)."""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

from quantark.pnlexplain.base import (
    MARKET_FACTORS, ExplainMethod, ExplainRow, PnLExplainResult, RowKind, make_total_row,
)
from quantark.pnlexplain.config import PnLExplainConfig
from quantark.pnlexplain.equity.coordinate import resolve_coordinate
from quantark.pnlexplain.equity.factor_diff import build_factor_moves, validate_pair
from quantark.pnlexplain.equity.fingerprints import engines_equivalent
from quantark.pnlexplain.equity.lifecycle import (
    LifecycleTransition, event_row, fixing_refill_row, resolve_transition,
)
from quantark.pnlexplain.equity.scenario import ScenarioCache
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot
from quantark.pnlexplain.equity.taylor import taylor_rows
from quantark.pnlexplain.equity.waterfall import sequential_rows, shapley_rows
from quantark.util.exceptions import NumericalError
from quantark.util.fixings import freeze_missing_fixings
from quantark.util.numerical import is_close

LEVEL = "instrument"
RECONCILE_TOL = 1e-10


def reconcile_scale(result: PnLExplainResult, method: ExplainMethod) -> float:
    """The magnitude the waterfall's rounding error is proportional to.

    The rows are differences of scenario values and the endpoints are sums of
    value parts, so floating-point error scales with the endpoint PVs and the
    absolute component sizes, not with the (possibly offsetting) net PnL.
    """
    parts = [1.0, abs(result.total_pnl), abs(result.pv_t0.total), abs(result.pv_alive_t1.total),
             abs(result.pv_t1.total)]
    parts.append(sum(abs(r.pnl) for r in result.rows_for(method, kind=RowKind.COMPONENT)))
    return max(parts)


def _carry_fills(product: Any, fills: Dict[int, Any]) -> Any:
    """``product`` with the t0 ``fills`` recorded on the same records. The roll check, run first, holds a range
    accrual's or an Asian's records identical across the pair (they are not a roll-suffix schedule), so a record
    index names the same fixing at t0 and t1."""
    record = getattr(product, "with_fixings_recorded", None)
    return product if record is None else record(fills)


def _freeze_pair_fixings(
    snapshot_t0: ValuationSnapshot,
    snapshot_t1: ValuationSnapshot,
    transition: LifecycleTransition,
) -> Tuple[ValuationSnapshot, LifecycleTransition, Optional[ValuationSnapshot]]:
    """The t0 snapshot and the alive-at-t1 contract with t0's missing past fixings recorded at t0's spot.

    A fixing already missing at t0 is history (user decision 2026-10-09): every shocked or intermediate state of
    the explain - the waterfall states, the Taylor greeks, the alive-at-t1 contract - prices t0's decision.
    ``snapshot_t1`` itself is NOT changed: ``pv_t1`` is the t1 snapshot's own value, so consecutive daily explains
    chain. Returns ``(snapshot_t0, transition, snapshot_t1_carried)``; the last is the t1 snapshot with t0's
    decision carried (None when nothing was missing), against which ``FIXING_REFILL`` measures a re-decided fixing.

    TODO(lifecycle, W6): a fixing that passes between t0 and t1 is decided at the spot of each revaluation that sees
    it as past; it should be recorded once at its own day's real spot.
    """
    product_t0 = snapshot_t0.product
    fills_of = getattr(product_t0, "missing_fixing_fills", None)
    fills = fills_of(snapshot_t0.pricing_env) if fills_of is not None else {}
    if not fills:
        return snapshot_t0, transition, None
    snapshot_t0 = dataclasses.replace(
        snapshot_t0, product=freeze_missing_fixings(product_t0, snapshot_t0.pricing_env)
    )
    transition = dataclasses.replace(
        transition, product_alive_t1=_carry_fills(transition.product_alive_t1, fills)
    )
    snapshot_t1_carried = dataclasses.replace(
        snapshot_t1, product=_carry_fills(snapshot_t1.product, fills)
    )
    return snapshot_t0, transition, snapshot_t1_carried


def explain(
    snapshot_t0: ValuationSnapshot,
    snapshot_t1: ValuationSnapshot,
    *,
    config: Optional[PnLExplainConfig] = None,
    transition: Optional[LifecycleTransition] = None,
) -> PnLExplainResult:
    config = config if config is not None else PnLExplainConfig()
    # The pair and its roll are checked on the contracts as supplied, before any missing fixing is recorded
    validate_pair(snapshot_t0, snapshot_t1)
    days = (snapshot_t1.date - snapshot_t0.date).days
    transition = resolve_transition(snapshot_t0, snapshot_t1, transition, calendar_days=days)
    snapshot_t0, transition, snapshot_t1_carried = _freeze_pair_fixings(snapshot_t0, snapshot_t1, transition)
    coordinate = resolve_coordinate(
        snapshot_t0.product, float(snapshot_t0.pricing_env.spot),
        transition.product_alive_t1, snapshot_t1.pricing_env,
    )
    moves = build_factor_moves(
        snapshot_t0, snapshot_t1, coordinate,
        engine_alive_t1=transition.engine_alive_t1, lifecycle_changed=transition.changed,
    )
    cache = ScenarioCache(snapshot_t0, snapshot_t1, transition, moves, config,
                          snap1_carried=snapshot_t1_carried)
    pv_t0 = cache.value_for(())
    pv_alive_t1 = cache.all_market()
    pv_t1 = cache.value_t1()
    total_pnl = pv_t1.total - pv_t0.total

    taylor_meta: Dict[str, Any] = {"route": None, "vega_scale": None, "n_steps": None, "clock": None,
                                   "gap_scale": None}
    rows: List[ExplainRow] = []
    if ExplainMethod.WATERFALL in config.methods:
        if config.interaction == "sequential":
            rows.extend(sequential_rows(cache, config.waterfall_order, LEVEL))
        else:
            rows.extend(shapley_rows(cache, LEVEL))
    rows.append(event_row(cache, transition, LEVEL))
    if snapshot_t1_carried is not None:
        rows.append(fixing_refill_row(cache, LEVEL))
    # The residual is assigned exactly once: None unless the Taylor method is requested.
    unexplained: Optional[float] = None
    if ExplainMethod.TAYLOR in config.methods:
        trows, unexplained, taylor_meta = taylor_rows(cache, config, LEVEL)
        rows.extend(trows)
    rows.append(make_total_row(LEVEL, total_pnl))

    metadata = {
        "time_pure": cache.time_pure(),
        "effective_factors": tuple(f.value for f in MARKET_FACTORS if f in cache.effective),
        "coordinate": (coordinate.reference_strike, coordinate.tenor_t1),
        "transition_changed": transition.changed,
        "model_equivalent": engines_equivalent(snapshot_t0.engine, transition.engine_alive_t1),
        "contract_roll_days": days if transition.contract_roll_days is None else int(transition.contract_roll_days),
        "interaction": config.interaction,
        **taylor_meta,
    }
    result = PnLExplainResult(
        date_t0=snapshot_t0.date, date_t1=snapshot_t1.date, pv_t0=pv_t0,
        pv_alive_t1=pv_alive_t1, pv_t1=pv_t1, total_pnl=total_pnl, moves=moves,
        rows=tuple(rows), unexplained=unexplained, metadata=metadata,
    )
    if ExplainMethod.WATERFALL in config.methods:
        gap = result.reconcile(ExplainMethod.WATERFALL)
        tol = RECONCILE_TOL * reconcile_scale(result, ExplainMethod.WATERFALL)
        if not is_close(gap, 0.0, rel_tol=0.0, abs_tol=tol):
            raise NumericalError(f"waterfall does not reconcile: gap {gap} exceeds {tol}")
    return result
