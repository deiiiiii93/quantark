"""Hedge-quality measures of one path, in bp of notional (spec 9).

These are the q term-structure study's ``hedge_measures``, moved here so the
historical replay and the simulated ensemble report ONE implementation; the
study's function delegates to this one.  The arithmetic is the definition
and is kept verbatim.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

#: The keys ``path_measures`` returns, in order.
MEASURE_COLUMNS: Tuple[str, ...] = (
    "days", "terminal_pnl_bp", "terminal_pnl_gross_bp", "daily_pnl_std_bp", "variance_reduction_r2",
    "max_drawdown_bp", "turnover", "rebalance_turnover", "roll_turnover", "cost_bp", "roll_days",
    "roll_day_mtm_jump_bp", "other_day_mtm_jump_bp", "roll_day_q_jump", "delta_churn",
)


def bp_of(value: float, notional: float) -> float:
    """``value`` as basis points of ``notional``."""
    return float(value) / float(notional) * 1e4


def max_drawdown(series: Sequence[float]) -> float:
    """Largest peak-to-trough fall of a cumulative P&L series (0 for an empty one)."""
    peak = -math.inf
    worst = 0.0
    for v in series:
        peak = max(peak, float(v))
        worst = max(worst, peak - float(v))
    return worst


def path_measures(states: pd.DataFrame, trades: pd.DataFrame, *, notional: float) -> Dict[str, Any]:
    """Hedge-quality measures of one replay or simulated path, in bp of notional.

    - ``terminal_pnl_bp``: final total P&L (product + hedge - costs);
      ``terminal_pnl_gross_bp`` adds the costs back.
    - ``daily_pnl_std_bp``: std of the DAILY hedged P&L increments -- the
      hedge error a desk lives with day to day.
    - ``variance_reduction_r2``: 1 - var(hedged daily P&L)/var(unhedged
      product daily P&L); how much of the product's daily variance the
      futures hedge removes.
    - ``max_drawdown_bp``: peak-to-trough of cumulative hedged P&L.
    - ``turnover``: sum of |trade notional| / notional, split into
      rebalance vs roll legs; ``cost_bp``: cumulative transaction cost.
    - ``roll_day_mtm_jump_bp`` vs ``other_day_mtm_jump_bp``: mean
      |delta product MTM| on days the active hedge contract changed vs all
      other days.  A pricing model tied to the active contract re-marks the
      book on every roll for no economic reason; a term model does not.
    - ``roll_day_q_jump``: mean |delta pricing_q| on roll days.
    - ``delta_churn``: mean |contracts traded| per day.

    ``states`` needs ``total_pnl``, ``product_pnl``, ``product_mtm``,
    ``pricing_q``, ``active_contract`` and ``transaction_costs``; ``trades``
    needs ``trade_type``, ``notional`` and ``quantity``.  A path shorter
    than three days reports NaN for the variance measures.
    """
    n = int(len(states))
    total = states["total_pnl"].to_numpy(dtype=float)
    product = states["product_pnl"].to_numpy(dtype=float)
    d_total = np.diff(total)
    d_product = np.diff(product)
    nan = float("nan")
    if n >= 3:
        std_total = float(np.std(d_total, ddof=1))
        var_total = float(np.var(d_total, ddof=1))
        var_product = float(np.var(d_product, ddof=1))
        r2 = 1.0 - var_total / var_product if var_product > 0.0 else nan
    else:
        std_total, r2 = nan, nan

    if len(trades):
        trade_notional = trades["notional"].to_numpy(dtype=float)
        is_roll = trades["trade_type"].astype(str).str.startswith("roll").to_numpy()
        turnover = float(trade_notional.sum()) / notional
        roll_turnover = float(trade_notional[is_roll].sum()) / notional
        rebalance_turnover = float(trade_notional[~is_roll].sum()) / notional
        contracts_traded = float(np.abs(trades["quantity"].to_numpy(dtype=float)[~is_roll]).sum())
    else:
        turnover = roll_turnover = rebalance_turnover = contracts_traded = 0.0

    active = states["active_contract"].astype(str).to_numpy()
    roll_mask = np.zeros(n, dtype=bool)
    if n >= 2:
        roll_mask[1:] = active[1:] != active[:-1]
    d_mtm = np.abs(np.diff(states["product_mtm"].to_numpy(dtype=float)))
    d_q = np.abs(np.diff(states["pricing_q"].to_numpy(dtype=float)))
    roll_idx = roll_mask[1:]
    roll_days = int(roll_idx.sum())
    roll_jump = float(d_mtm[roll_idx].mean()) if roll_days else nan
    other_jump = float(d_mtm[~roll_idx].mean()) if (~roll_idx).sum() else nan
    roll_q_jump = float(d_q[roll_idx].mean()) if roll_days else nan

    costs = float(states["transaction_costs"].iloc[-1]) if n else 0.0
    return {
        "days": n,
        "terminal_pnl_bp": bp_of(total[-1], notional) if n else 0.0,
        "terminal_pnl_gross_bp": bp_of(total[-1] + costs, notional) if n else 0.0,
        "daily_pnl_std_bp": bp_of(std_total, notional) if math.isfinite(std_total) else nan,
        "variance_reduction_r2": r2,
        "max_drawdown_bp": bp_of(max_drawdown(total), notional) if n else 0.0,
        "turnover": turnover,
        "rebalance_turnover": rebalance_turnover,
        "roll_turnover": roll_turnover,
        "cost_bp": bp_of(costs, notional),
        "roll_days": roll_days,
        "roll_day_mtm_jump_bp": bp_of(roll_jump, notional) if math.isfinite(roll_jump) else nan,
        "other_day_mtm_jump_bp": bp_of(other_jump, notional) if math.isfinite(other_jump) else nan,
        "roll_day_q_jump": roll_q_jump,
        "delta_churn": contracts_traded / n if n else 0.0,
    }
