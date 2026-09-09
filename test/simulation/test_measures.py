from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.simulation.measures import MEASURE_COLUMNS, bp_of, max_drawdown, path_measures

NOTIONAL = 10_000.0


def _states(total, product, mtm, q, active, costs) -> pd.DataFrame:
    return pd.DataFrame({
        "total_pnl": total, "product_pnl": product, "product_mtm": mtm, "pricing_q": q,
        "active_contract": active, "transaction_costs": costs,
    })


def _hand_built():
    states = _states(
        total=[0.0, 100.0, 50.0, 200.0], product=[0.0, 300.0, -100.0, 400.0],
        mtm=[1000.0, 1100.0, 1000.0, 1300.0], q=[0.01, 0.01, 0.02, 0.02],
        active=["IM2401", "IM2401", "IM2402", "IM2402"], costs=[0.0, 1.0, 2.0, 3.0],
    )
    trades = pd.DataFrame({
        "trade_type": ["hedge_rebalance", "roll_close", "roll_open", "hedge_rebalance"],
        "notional": [1000.0, 500.0, 500.0, 2000.0], "quantity": [2.0, -1.0, 1.0, -3.0],
    })
    return states, trades


def test_every_measure_on_a_hand_built_path():
    states, trades = _hand_built()
    m = path_measures(states, trades, notional=NOTIONAL)
    assert tuple(m) == MEASURE_COLUMNS
    d_total = np.diff([0.0, 100.0, 50.0, 200.0])          # 100, -50, 150
    d_product = np.diff([0.0, 300.0, -100.0, 400.0])      # 300, -400, 500
    assert m["days"] == 4
    assert m["terminal_pnl_bp"] == 200.0                  # 200 / 10 000 * 1e4
    assert m["terminal_pnl_gross_bp"] == 203.0            # costs added back
    assert m["daily_pnl_std_bp"] == pytest.approx(np.std(d_total, ddof=1))
    assert m["variance_reduction_r2"] == pytest.approx(1.0 - np.var(d_total, ddof=1) / np.var(d_product, ddof=1))
    assert m["max_drawdown_bp"] == 50.0                   # peak 100 -> trough 50
    assert (m["turnover"], m["rebalance_turnover"], m["roll_turnover"]) == (0.4, 0.3, 0.1)
    assert m["cost_bp"] == pytest.approx(3.0)                # 3 / 10 000 * 1e4 rounds one ulp short
    assert m["roll_days"] == 1                            # the contract changes once, on day 2
    assert m["roll_day_mtm_jump_bp"] == 100.0             # |1000 - 1100| on the roll day
    assert m["other_day_mtm_jump_bp"] == 200.0            # mean(|1100-1000|, |1300-1000|)
    assert m["roll_day_q_jump"] == pytest.approx(0.01)
    assert m["delta_churn"] == 1.25                       # (2 + 3) rebalance contracts / 4 days


def test_short_paths_and_no_trades_report_nan_and_zero_not_errors():
    states = _states([0.0, 5.0], [0.0, 7.0], [10.0, 12.0], [0.0, 0.0], ["A", "A"], [0.0, 0.0])
    m = path_measures(states, pd.DataFrame(columns=["trade_type", "notional", "quantity"]), notional=NOTIONAL)
    assert np.isnan(m["daily_pnl_std_bp"]) and np.isnan(m["variance_reduction_r2"])
    assert m["turnover"] == 0.0 and m["delta_churn"] == 0.0 and m["roll_days"] == 0
    assert np.isnan(m["roll_day_mtm_jump_bp"])


def test_helpers():
    assert bp_of(25.0, NOTIONAL) == 25.0
    assert max_drawdown([0.0, 3.0, 1.0, 4.0, -2.0]) == 6.0
    assert max_drawdown([]) == 0.0


def test_the_q_study_delegates_to_the_library():
    path = Path(__file__).resolve().parents[2] / "example" / "snowball_q_term_structure" / "_common.py"
    spec = importlib.util.spec_from_file_location("q_term_structure_common_for_measures", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    states, trades = _hand_built()
    assert module.hedge_measures(states, trades, notional=NOTIONAL) == path_measures(states, trades, notional=NOTIONAL)
    assert module.max_drawdown is max_drawdown
