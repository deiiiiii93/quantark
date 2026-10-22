"""The q-study delegates its path measures to quantark.backtest.simulation (split from test/simulation/test_measures.py)."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

from quantark.backtest.simulation.measures import max_drawdown, path_measures

NOTIONAL = 10_000.0


def _hand_built():
    states = pd.DataFrame({
        "total_pnl": [0.0, 100.0, 50.0, 200.0], "product_pnl": [0.0, 300.0, -100.0, 400.0],
        "product_mtm": [1000.0, 1100.0, 1000.0, 1300.0], "pricing_q": [0.01, 0.01, 0.02, 0.02],
        "active_contract": ["IM2401", "IM2401", "IM2402", "IM2402"], "transaction_costs": [0.0, 1.0, 2.0, 3.0],
    })
    trades = pd.DataFrame({
        "trade_type": ["hedge_rebalance", "roll_close", "roll_open", "hedge_rebalance"],
        "notional": [1000.0, 500.0, 500.0, 2000.0], "quantity": [2.0, -1.0, 1.0, -3.0],
    })
    return states, trades


def test_the_q_study_delegates_to_the_library():
    path = Path(__file__).resolve().parents[4] / "example" / "case_studies" / "snowball_q_term_structure" / "_common.py"
    spec = importlib.util.spec_from_file_location("q_term_structure_common_for_measures", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    states, trades = _hand_built()
    assert module.hedge_measures(states, trades, notional=NOTIONAL) == path_measures(states, trades, notional=NOTIONAL)
    assert module.max_drawdown is max_drawdown
