"""credit backtest result container."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class CreditBacktestResults:
    """Per-step history and summary metrics for an credit backtest."""

    rows: List[Dict[str, Any]] = field(default_factory=list)
    initial_value: float = 0.0
    final_value: float = 0.0
    num_hedges: int = 0
    total_transaction_costs: float = 0.0
    total_hedge_pnl: float = 0.0
    config_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_portfolio_pnl(self) -> float:
        return self.final_value - self.initial_value

    @property
    def total_net_pnl(self) -> float:
        return self.total_portfolio_pnl + self.total_hedge_pnl - self.total_transaction_costs

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).set_index("timestamp")

    def get_hedge_effectiveness(self) -> Dict[str, float]:
        """Std of net daily P&L vs unhedged (portfolio-only) daily P&L."""
        df = self.to_dataframe()
        net_vol = df["net_pnl"].diff().dropna().std()
        gross_vol = df["portfolio_pnl"].diff().dropna().std()
        reduction = 1.0 - (net_vol / gross_vol) if gross_vol else 0.0
        return {
            "net_pnl_vol": float(net_vol),
            "portfolio_pnl_vol": float(gross_vol),
            "vol_reduction_pct": float(reduction * 100.0),
        }

    def get_summary(self) -> str:
        eff = self.get_hedge_effectiveness()
        return (
            f"Credit Backtest: {self.config_summary.get('strategy', '?')}\n"
            f"  Steps:               {len(self.rows)}\n"
            f"  Hedges executed:     {self.num_hedges}\n"
            f"  Portfolio P&L:       ${self.total_portfolio_pnl:,.2f}\n"
            f"  Hedge P&L:           ${self.total_hedge_pnl:,.2f}\n"
            f"  Transaction costs:   ${self.total_transaction_costs:,.2f}\n"
            f"  Net P&L:             ${self.total_net_pnl:,.2f}\n"
            f"  P&L vol reduction:   {eff['vol_reduction_pct']:.1f}%"
        )

    def __repr__(self) -> str:
        return (
            f"CreditBacktestResults(steps={len(self.rows)}, hedges={self.num_hedges}, "
            f"net_pnl=${self.total_net_pnl:,.2f})"
        )
