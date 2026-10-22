"""
FX Backtest Demo
================

Delta-hedges a long EUR/USD option over a simulated 60-day spot path, comparing
an unhedged book against a delta-neutral strategy and reporting P&L volatility
reduction and transaction-cost drag.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.backtest import FXBacktestConfig, FXBacktestEngine
from quantark.backtest.strategy import FXDeltaNeutralStrategy
from quantark.backtest.transaction_costs import ProportionalCostModel, ZeroCostModel
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.fx import FXPortfolio
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import OptionType


def build_book() -> FXPortfolio:
    env = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12), spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10))
    pf = FXPortfolio(portfolio_name="FX Hedge Book", pricing_environments={"EURUSD": env})
    pf.add_position(
        product=FxVanillaOption(currency_pair=CurrencyPair("EUR", "USD"), strike=1.20,
                                option_type=OptionType.CALL, maturity=1.0,
                                notional_foreign=5_000_000.0),
        quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=GarmanKohlhagenEngine())
    return pf


def market_path(days=60, seed=20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-06-15", periods=days, freq="B")
    spot = 1.20 * np.exp(np.cumsum(rng.normal(0.0, 0.009, days)))
    return pd.DataFrame({
        "EURUSD_spot": spot,
        "EURUSD_vol": np.full(days, 0.10),
        "EURUSD_dom_rate": np.full(days, 0.05),
        "EURUSD_for_rate": np.full(days, 0.03),
    }, index=idx)


def main() -> None:
    path = market_path()

    unhedged = FXBacktestEngine(FXBacktestConfig(
        portfolio=build_book(), market_path=path,
        strategy=FXDeltaNeutralStrategy(delta_threshold=1e15),  # never hedges
        transaction_cost_model=ZeroCostModel())).run()

    hedged = FXBacktestEngine(FXBacktestConfig(
        portfolio=build_book(), market_path=path,
        strategy=FXDeltaNeutralStrategy(delta_threshold=50_000.0,
                                        rebalance_frequency="on_threshold"),
        transaction_cost_model=ProportionalCostModel(commission_rate=0.00005))).run()

    print("=== Unhedged ===")
    print(unhedged.get_summary())
    print("\n=== Delta-neutral hedged ===")
    print(hedged.get_summary())

    print("\nDaily P&L volatility:")
    print(f"  Unhedged: ${unhedged.to_dataframe()['net_pnl'].diff().std():,.2f}")
    print(f"  Hedged:   ${hedged.to_dataframe()['net_pnl'].diff().std():,.2f}")


if __name__ == "__main__":
    main()
