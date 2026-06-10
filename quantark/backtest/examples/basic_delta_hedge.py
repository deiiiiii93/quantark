"""
Basic delta-neutral hedging backtest example.

This script demonstrates the simplest use case:
- Create an option position
- Set up a delta-neutral hedging strategy
- Run backtest with mock market data
- Generate results and visualizations
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from quantark.backtest import (
    BacktestEngine,
    BacktestConfig,
    ZeroCostModel,
    StaticVisualizer,
    InteractiveDashboard,
    ReportGenerator,
)
from quantark.backtest.strategy import DeltaNeutralStrategy
from quantark.portfolio import Position
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.util.enum import OptionType
from quantark.util.marketdata import MockMarketDataAdapter


def main():
    """Run basic delta hedge backtest."""
    print("=" * 80)
    print("BASIC DELTA-NEUTRAL HEDGING BACKTEST".center(80))
    print("=" * 80)
    print()

    # =========================================================================
    # STEP 1: Set up dates and underlying
    # =========================================================================
    print("STEP 1: Configuration")
    print("-" * 80)

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)  # 6 months
    underlying = "AAPL"

    print(f"Underlying: {underlying}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print()

    # =========================================================================
    # STEP 2: Create initial option position
    # =========================================================================
    print("STEP 2: Create Initial Position")
    print("-" * 80)

    # Long 100 ATM call options with 1 year maturity
    option = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )

    engine = BlackScholesEngine()

    # Create position
    initial_position = Position(
        product=option,
        quantity=100,
        entry_price=10.0,  # Approximate entry price
        underlying=underlying,
        engine=engine,
        entry_timestamp=start_date,
    )

    print(f"Created position: LONG 100 x {underlying} Call (K=100, T=1y)")
    print(f"Entry price: ${initial_position.entry_price:.2f}")
    print()

    # =========================================================================
    # STEP 3: Configure strategy
    # =========================================================================
    print("STEP 3: Configure Delta-Neutral Strategy")
    print("-" * 80)

    strategy = DeltaNeutralStrategy(
        name="BasicDeltaNeutral",
        delta_threshold=50.0,  # Hedge when |delta| > 50
        rebalance_frequency="daily",  # Check daily
        hedge_instrument="spot",  # Use spot for hedging
        hedge_ratio=1.0,  # Full hedge
        target_delta=0.0,  # Target delta = 0
    )

    print(f"Strategy: {strategy.name}")
    print(f"Delta threshold: {strategy.delta_threshold}")
    print(f"Rebalance frequency: {strategy.rebalance_frequency}")
    print(f"Hedge instrument: {strategy.hedge_instrument}")
    print()

    # =========================================================================
    # STEP 4: Set up market data and backtest configuration
    # =========================================================================
    print("STEP 4: Configure Backtest")
    print("-" * 80)

    # Use mock data adapter
    market_data_adapter = MockMarketDataAdapter(seed=42)

    # Use zero transaction costs for simplicity
    cost_model = ZeroCostModel()

    # Create backtest configuration
    config = BacktestConfig(
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        underlying=underlying,
        initial_positions=[initial_position],
        market_data_adapter=market_data_adapter,
        transaction_cost_model=cost_model,
        frequency="D",  # Daily data
        logging_level="INFO",
        calculate_greeks=True,
        greeks_method="analytical",
    )

    print(f"Market data: Mock (synthetic)")
    print(f"Transaction costs: Zero (frictionless)")
    print(f"Data frequency: Daily")
    print()

    # =========================================================================
    # STEP 5: Run backtest
    # =========================================================================
    print("=" * 80)
    print("RUNNING BACKTEST")
    print("=" * 80)
    print()

    engine = BacktestEngine(config)
    results = engine.run()

    print()
    print("=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)
    print()

    # =========================================================================
    # STEP 6: Display results
    # =========================================================================
    print("STEP 6: Results Summary")
    print("-" * 80)

    summary = results.get_summary()
    print(f"Initial Value:       ${summary['initial_value']:,.2f}")
    print(f"Final Value:         ${summary['final_value']:,.2f}")
    print(f"Total P&L:           ${summary['total_pnl']:,.2f}")
    print(f"Total Return:        {summary['total_return_pct']:.2f}%")
    print(f"Number of Hedges:    {summary['num_hedges']}")
    print(f"Transaction Costs:   ${summary['total_transaction_costs']:,.2f}")
    print()

    # Performance metrics
    metrics = results.metrics
    print("Performance Metrics:")
    print(f"  Sharpe Ratio:      {metrics.sharpe_ratio():.3f}")
    print(f"  Max Drawdown:      {metrics.max_drawdown():.2%}")
    print(f"  Win Rate:          {metrics.win_rate():.2%}")
    print(f"  Delta Track Error: {metrics.delta_tracking_error():.2f}")
    print()

    # =========================================================================
    # STEP 7: Generate visualizations
    # =========================================================================
    print("STEP 7: Generate Visualizations")
    print("-" * 80)

    # Static plots
    print("Creating static plots...")
    visualizer = StaticVisualizer(results, save_dir="plots/basic_example")
    static_plots = visualizer.generate_all_plots(save=True)
    print(f"  Saved {len(static_plots)} static plots to plots/basic_example/")

    # Interactive dashboard
    print("Creating interactive dashboard...")
    dashboard = InteractiveDashboard(
        results, save_dir="plots/basic_example/interactive"
    )
    dashboard.generate_all_interactive_plots(save=True)
    print(f"  Saved interactive plots to plots/basic_example/interactive/")

    print()

    # =========================================================================
    # STEP 8: Generate report
    # =========================================================================
    print("STEP 8: Generate Report")
    print("-" * 80)

    report_gen = ReportGenerator(results, output_dir="reports/basic_example")

    # HTML report
    html_path = report_gen.generate_html_report()
    print(f"HTML report: {html_path}")

    # Text report
    text_path = report_gen.generate_text_report()
    print(f"Text report: {text_path}")

    print()

    # =========================================================================
    # COMPLETION
    # =========================================================================
    print("=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("  - Set up a delta-neutral hedging strategy")
    print("  - Ran backtest with mock market data")
    print("  - Generated comprehensive results and visualizations")
    print("  - Created HTML and text reports")
    print()
    print(
        f"Final P&L: ${summary['total_pnl']:,.2f} ({summary['total_return_pct']:.2f}%)"
    )
    print(f"Number of hedges executed: {summary['num_hedges']}")
    print()

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
