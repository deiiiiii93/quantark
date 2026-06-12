"""
Multi-Greek hedging strategy demo.

Backtests a short-call book under four hedging strategies and compares
the resulting P&L volatility and trading activity:

1. Delta-neutral (spot only)            - baseline
2. Delta + gamma control                - 3M ATM option + spot
3. Delta + gamma + vega control         - 3M & 1Y ATM options + spot
4. Whalley-Wilmott no-trade band        - cost-aware delta hedging
"""

from datetime import datetime

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.backtest import BacktestConfig, BacktestEngine, ProportionalCostModel
from quantark.backtest.strategy import (
    DeltaGammaNeutralStrategy,
    DeltaGammaVegaNeutralStrategy,
    DeltaNeutralStrategy,
    WhalleyWilmottStrategy,
)
from quantark.portfolio import Position
from quantark.util.enum import OptionType
from quantark.util.marketdata.adapter.mock_adapter import MockMarketDataAdapter

UNDERLYING = "DEMO"
START = datetime(2024, 1, 1)
END = datetime(2024, 6, 30)


def make_config(strategy):
    """Short 500 one-year ATM calls, hedged per the given strategy."""
    option = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    short_calls = Position(
        product=option,
        quantity=-500.0,
        entry_price=10.0,
        underlying=UNDERLYING,
        engine=BlackScholesEngine(),
        entry_timestamp=START,
    )
    return BacktestConfig(
        strategy=strategy,
        start_date=START,
        end_date=END,
        underlying=UNDERLYING,
        initial_positions=[short_calls],
        market_data_adapter=MockMarketDataAdapter(seed=7),
        transaction_cost_model=ProportionalCostModel(commission_rate=0.0005),
        logging_level="WARNING",
    )


def run_strategy(label, strategy):
    engine = BacktestEngine(make_config(strategy))
    results = engine.run()
    pnl = results.get_pnl_series()
    greeks = engine.portfolio.get_portfolio_greeks(engine.greeks_calculator)
    print(f"\n=== {label} ===")
    print(f"  Final net P&L:      {results.get_total_pnl():>12,.2f}")
    print(f"  Daily P&L stdev:    {pnl.diff().std():>12,.2f}")
    print(f"  Hedge rebalances:   {engine._num_hedges_executed:>12d}")
    print(
        "  Residual greeks:    "
        f"delta={greeks['delta']:.2f}, gamma={greeks['gamma']:.3f}, "
        f"vega={greeks['vega']:.2f}"
    )


def main():
    run_strategy(
        "1) Delta-neutral (spot only)",
        DeltaNeutralStrategy(
            delta_threshold=10.0, rebalance_frequency="continuous"
        ),
    )
    run_strategy(
        "2) Delta + gamma control",
        DeltaGammaNeutralStrategy(
            delta_threshold=10.0,
            gamma_threshold=1.0,
            rebalance_frequency="continuous",
        ),
    )
    run_strategy(
        "3) Delta + gamma + vega control",
        DeltaGammaVegaNeutralStrategy(
            delta_threshold=10.0,
            gamma_threshold=1.0,
            vega_threshold=1.0,
            rebalance_frequency="continuous",
        ),
    )
    run_strategy(
        "4) Whalley-Wilmott no-trade band",
        WhalleyWilmottStrategy(risk_aversion=0.1, cost_rate=0.0005),
    )


if __name__ == "__main__":
    main()
