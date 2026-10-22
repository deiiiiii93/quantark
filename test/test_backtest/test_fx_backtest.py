"""
Tests for FX backtesting (delta-neutral hedging of an FX book).

Steps an FX portfolio through a market path, repricing each day, and hedges
spot delta with FX spot, tracking hedge P&L and transaction costs.
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.backtest.transaction_costs import ProportionalCostModel, ZeroCostModel
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.fx import FXPortfolio
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import OptionType


def _portfolio():
    env = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12), spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10))
    pf = FXPortfolio(portfolio_name="fx", pricing_environments={"EURUSD": env})
    pf.add_position(
        product=FxVanillaOption(currency_pair=CurrencyPair("EUR", "USD"), strike=1.20,
                                option_type=OptionType.CALL, maturity=1.0,
                                notional_foreign=1_000_000.0),
        quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=GarmanKohlhagenEngine())
    return pf


def _path(days=40, seed=3, sigma=0.01):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-06-15", periods=days, freq="B")
    spot = 1.20 * np.exp(np.cumsum(rng.normal(0, sigma, days)))
    return pd.DataFrame({
        "EURUSD_spot": spot,
        "EURUSD_vol": np.full(days, 0.10),
        "EURUSD_dom_rate": np.full(days, 0.05),
        "EURUSD_for_rate": np.full(days, 0.03),
    }, index=idx)


def test_delta_neutral_strategy_keeps_post_hedge_delta_small():
    from quantark.backtest import FXBacktestConfig, FXBacktestEngine
    from quantark.backtest.strategy import FXDeltaNeutralStrategy

    strat = FXDeltaNeutralStrategy(delta_threshold=10_000.0,
                                   rebalance_frequency="on_threshold")
    config = FXBacktestConfig(portfolio=_portfolio(), market_path=_path(),
                              strategy=strat, transaction_cost_model=ZeroCostModel())
    results = FXBacktestEngine(config).run()
    df = results.to_dataframe()
    # After hedging, residual delta is bounded by the threshold each rebalance day.
    hedged_days = df[df["hedged"]]
    assert (hedged_days["delta_post"].abs() <= 10_000.0 + 1e-6).all()
    assert results.num_hedges > 0


def test_hedging_reduces_pnl_volatility():
    from quantark.backtest import FXBacktestConfig, FXBacktestEngine
    from quantark.backtest.strategy import FXDeltaNeutralStrategy

    path = _path()
    unhedged = FXBacktestEngine(FXBacktestConfig(
        portfolio=_portfolio(), market_path=path,
        strategy=FXDeltaNeutralStrategy(delta_threshold=1e12),  # never hedges
        transaction_cost_model=ZeroCostModel())).run()
    hedged = FXBacktestEngine(FXBacktestConfig(
        portfolio=_portfolio(), market_path=path,
        strategy=FXDeltaNeutralStrategy(delta_threshold=5_000.0,
                                        rebalance_frequency="on_threshold"),
        transaction_cost_model=ZeroCostModel())).run()

    u = unhedged.to_dataframe()["net_pnl"].diff().dropna().std()
    h = hedged.to_dataframe()["net_pnl"].diff().dropna().std()
    assert h < u
    assert unhedged.num_hedges == 0


def test_transaction_costs_accumulate_when_hedging():
    from quantark.backtest import FXBacktestConfig, FXBacktestEngine
    from quantark.backtest.strategy import FXDeltaNeutralStrategy

    config = FXBacktestConfig(
        portfolio=_portfolio(), market_path=_path(),
        strategy=FXDeltaNeutralStrategy(delta_threshold=5_000.0,
                                        rebalance_frequency="on_threshold"),
        transaction_cost_model=ProportionalCostModel(commission_rate=0.0001))
    results = FXBacktestEngine(config).run()
    assert results.total_transaction_costs > 0
    assert len(results.to_dataframe()) == len(_path())


def test_results_dataframe_shape_and_columns():
    from quantark.backtest import FXBacktestConfig, FXBacktestEngine
    from quantark.backtest.strategy import FXDeltaNeutralStrategy

    results = FXBacktestEngine(FXBacktestConfig(
        portfolio=_portfolio(), market_path=_path(days=25),
        strategy=FXDeltaNeutralStrategy(delta_threshold=8_000.0,
                                        rebalance_frequency="on_threshold"))).run()
    df = results.to_dataframe()
    assert len(df) == 25
    for col in ["net_pnl", "hedge_pnl", "delta_pre", "delta_post", "transaction_costs"]:
        assert col in df.columns
