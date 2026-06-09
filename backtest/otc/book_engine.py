"""Per-underlying multi-product net-delta hedging backtest (autocallable lifecycle aware)."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import numpy as np
import pandas as pd

from util.exceptions import ValidationError
from .config import AutocallableEngineConfig, FuturesRollPolicy, SurfaceGridConfig
from .market import AutocallableMarketDataSet
from .state import AutocallableDeltaHedgeStrategy, AutocallableLifecycleState, FuturesHedgePosition
from backtest.transaction_costs import TransactionCostModel, ZeroCostModel
from .engine_factory import create_pricing_engine, create_surface_engine, create_event_stats_engine
from ._replay import ProductReplay


@dataclass
class BookProduct:
    product: Any
    quantity: float
    position_id: int
    has_lifecycle: bool
    initial_price: Optional[float] = None

    def __post_init__(self):
        if self.product is None:
            raise ValidationError("BookProduct.product is required")
        if self.quantity == 0:
            raise ValidationError("BookProduct.quantity must be non-zero")


@dataclass
class HedgeSpec:
    kind: str = "futures"
    multiplier: float = 1.0
    roll_policy: Optional[FuturesRollPolicy] = None

    def __post_init__(self):
        if self.kind not in ("futures", "spot"):
            raise ValidationError(f"HedgeSpec.kind must be futures|spot, got {self.kind}")
        if self.kind == "futures" and self.roll_policy is None:
            self.roll_policy = FuturesRollPolicy()


@dataclass
class BookAutocallableBacktestConfig:
    products: list[BookProduct]
    market_data: AutocallableMarketDataSet
    hedge: HedgeSpec = field(default_factory=HedgeSpec)
    engine_config: AutocallableEngineConfig = field(default_factory=AutocallableEngineConfig)
    strategy: Any = None
    transaction_cost_model: TransactionCostModel = field(default_factory=ZeroCostModel)
    underlying: str = "equity_index"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    fixed_dividend_yield: Optional[float] = None
    delta_bump_size: Optional[float] = None
    gamma_bump_size: Optional[float] = None
    surface_config: SurfaceGridConfig = field(default_factory=SurfaceGridConfig)
    calculate_surfaces: bool = False
    calculate_event_probabilities: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.products:
            raise ValidationError("BookAutocallableBacktestConfig.products must be non-empty")
        if self.market_data is None:
            raise ValidationError("market_data is required")
        if self.strategy is None:
            self.strategy = AutocallableDeltaHedgeStrategy()


class BookBacktestResults:
    def __init__(self, *, config, states, greeks, trades, actions,
                 daily_event_summary, event_probabilities, products_meta):
        self.config = config
        self._states = states
        self._greeks = greeks
        self._trades = trades
        self._actions = actions
        self._daily_event_summary = daily_event_summary
        self._event_probabilities = event_probabilities
        self._products_meta = products_meta

    @staticmethod
    def _frame(rows, index=None):
        df = pd.DataFrame(rows)
        if index and not df.empty:
            df = df.set_index(index)
        return df

    def states_df(self): return self._frame(self._states)
    def greeks_df(self): return self._frame(self._greeks)
    def trades_df(self): return self._frame(self._trades)
    def actions_df(self): return self._frame(self._actions)
    def daily_event_summary_df(self): return self._frame(self._daily_event_summary)
    def event_probability_df(self): return self._frame(self._event_probabilities)

    def get_summary(self):
        states = self.states_df()
        if states.empty:
            return {"num_days": 0, "num_trades": len(self._trades), "total_pnl": 0.0,
                    "num_products": len(self._products_meta), "num_lifecycle_events": len(self._actions)}
        return {
            "num_days": int(len(states)),
            "start_date": str(states["date"].iloc[0]),
            "end_date": str(states["date"].iloc[-1]),
            "initial_portfolio_value": float(states["portfolio_value"].iloc[0]),
            "final_portfolio_value": float(states["portfolio_value"].iloc[-1]),
            "total_pnl": float(states["total_pnl"].iloc[-1]),
            "product_pnl": float(states["product_pnl"].iloc[-1]),
            "hedge_pnl": float(states["hedge_pnl"].iloc[-1]),
            "transaction_costs": float(states["transaction_costs"].iloc[-1]),
            "num_trades": int(len(self._trades)),
            "num_products": len(self._products_meta),
            "num_lifecycle_events": len(self._actions),
        }
