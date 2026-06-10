"""
Advanced backtest example with transaction costs and multiple options.

This script demonstrates advanced features:
- Multiple option positions
- Transaction cost modeling
- Custom strategy parameters
- Comprehensive analysis and reporting
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quantark.backtest import (
    BacktestEngine,
    BacktestConfig,
    CompleteCostModel,
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
    """Run advanced backtest with multiple positions and transaction costs."""
    print("=" * 80)
    print("ADVANCED DELTA-NEUTRAL HEDGING BACKTEST".center(80))
    print("=" * 80)
    print()

    # =========================================================================
    # STEP 1: Configuration
    # =========================================================================
    print("STEP 1: Advanced Configuration")
    print("-" * 80)

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)  # 1 year
    underlying = "SPY"

    print(f"Underlying: {underlying}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print()

    # =========================================================================
    # STEP 2: Create option portfolio
    # =========================================================================
    print("STEP 2: Create Option Portfolio")
    print("-" * 80)

    engine = BlackScholesEngine()
    initial_positions = []

    # Long call spread
    long_call = Position(
        product=EuropeanVanillaOption(
            strike=400.0, option_type=OptionType.CALL, maturity=1.0
        ),
        quantity=50,
        entry_price=20.0,
        underlying=underlying,
        engine=engine,
        entry_timestamp=start_date,
    )
    initial_positions.append(long_call)
    print(f"Position 1: LONG 50 x Call (K=400, T=1y) @ $20.00")

    short_call = Position(
        product=EuropeanVanillaOption(
            strike=420.0, option_type=OptionType.CALL, maturity=1.0
        ),
        quantity=-50,
        entry_price=12.0,
        underlying=underlying,
        engine=engine,
        entry_timestamp=start_date,
    )
    initial_positions.append(short_call)
    print(f"Position 2: SHORT 50 x Call (K=420, T=1y) @ $12.00")

    # Long put for protection
    long_put = Position(
        product=EuropeanVanillaOption(
            strike=380.0, option_type=OptionType.PUT, maturity=1.0
        ),
        quantity=30,
        entry_price=15.0,
        underlying=underlying,
        engine=engine,
        entry_timestamp=start_date,
    )
    initial_positions.append(long_put)
    print(f"Position 3: LONG 30 x Put (K=380, T=1y) @ $15.00")

    print(f"\nTotal initial positions: {len(initial_positions)}")
    print()

    # =========================================================================
    # STEP 3: Configure advanced strategy
    # =========================================================================
    print("STEP 3: Configure Advanced Strategy")
    print("-" * 80)

    strategy = DeltaNeutralStrategy(
        name="AdvancedDeltaNeutral",
        delta_threshold=100.0,  # Hedge when |delta| > 100
        rebalance_frequency="daily",
        hedge_instrument="spot",
        hedge_ratio=0.95,  # 95% hedge (not fully neutral)
        target_delta=0.0,
        min_time_between_hedges=timedelta(hours=12),  # Minimum 12 hours between hedges
    )

    print(f"Strategy: {strategy.name}")
    print(f"Delta threshold: {strategy.delta_threshold}")
    print(f"Hedge ratio: {strategy.hedge_ratio} (allows some delta exposure)")
    print(f"Min time between hedges: {strategy.min_time_between_hedges}")
    print()

    # =========================================================================
    # STEP 4: Configure transaction costs
    # =========================================================================
    print("STEP 4: Configure Transaction Costs")
    print("-" * 80)

    cost_model = CompleteCostModel(
        fixed_commission=2.0,  # $2 per trade
        proportional_rate=0.0005,  # 5 bps
        slippage_coefficient=0.0001,  # 1 bp per unit
        slippage_type="linear",
        spread_bps=5.0,  # 5 bps spread
    )

    print(f"Fixed commission: ${cost_model.fixed_commission} per trade")
    print(
        f"Proportional rate: {cost_model.proportional_rate:.4f} ({cost_model.proportional_rate * 10000:.1f} bps)"
    )
    print(
        f"Slippage: {cost_model.slippage_coefficient:.6f} ({cost_model.slippage_type})"
    )
    print(f"Bid-ask spread: {cost_model.spread_bps} bps")
    print()

    # =========================================================================
    # STEP 5: Configure market data
    # =========================================================================
    print("STEP 5: Configure Market Data")
    print("-" * 80)

    # Custom market data configuration for SPY-like behavior
    market_data_adapter = MockMarketDataAdapter(seed=123)
    market_data_adapter.set_asset_config(
        underlying,
        {
            "initial_spot": 400.0,
            "initial_vol": 0.18,  # 18% vol
            "initial_rate": 0.045,  # 4.5% rate
            "initial_div_yield": 0.015,  # 1.5% div yield
            "drift": 0.10,  # 10% annual drift
            "vol_of_vol": 0.3,
            "jump_intensity": 1.0,
        },
    )

    print(f"Initial spot: $400.00")
    print(f"Initial volatility: 18%")
    print(f"Initial rate: 4.5%")
    print(f"Drift: 10% annually")
    print()

    # =========================================================================
    # STEP 6: Create backtest configuration
    # =========================================================================
    print("STEP 6: Create Backtest Configuration")
    print("-" * 80)

    config = BacktestConfig(
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        underlying=underlying,
        initial_positions=initial_positions,
        market_data_adapter=market_data_adapter,
        transaction_cost_model=cost_model,
        frequency="D",
        logging_level="INFO",
        results_path="results/advanced_example",
        save_snapshots=True,
        calculate_greeks=True,
        greeks_method="analytical",
        metadata={
            "description": "Advanced delta-neutral backtest",
            "portfolio_type": "call_spread_with_put_protection",
        },
    )

    print(f"Configuration complete")
    print(f"Results will be saved to: {config.results_path}")
    print()

    # =========================================================================
    # STEP 7: Run backtest
    # =========================================================================
    print("=" * 80)
    print("RUNNING ADVANCED BACKTEST")
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
    # STEP 8: Comprehensive analysis
    # =========================================================================
    print("STEP 8: Comprehensive Analysis")
    print("-" * 80)

    summary = results.get_summary()
    metrics = results.metrics.calculate_all_metrics()

    print("\nExecutive Summary:")
    print(f"  Initial Value:       ${summary['initial_value']:,.2f}")
    print(f"  Final Value:         ${summary['final_value']:,.2f}")
    print(f"  Total P&L:           ${summary['total_pnl']:,.2f}")
    print(f"  Total Return:        {summary['total_return_pct']:.2f}%")
    print(f"  Number of Hedges:    {summary['num_hedges']}")
    print(f"  Transaction Costs:   ${summary['total_transaction_costs']:,.2f}")
    print(
        f"  Cost as % of P&L:    {(summary['total_transaction_costs'] / abs(summary['total_pnl']) * 100) if summary['total_pnl'] != 0 else 0:.2f}%"
    )

    print("\nP&L Metrics:")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Win Rate:            {metrics['win_rate']:.2%}")
    print(f"  Profit Factor:       {metrics['profit_factor']:.2f}")
    print(f"  Volatility:          {metrics['volatility']:.2%}")

    print("\nHedging Performance:")
    print(f"  Hedge Frequency:     {metrics['hedge_frequency']:.3f} per day")
    print(f"  Avg Hedge Cost:      ${metrics['avg_hedge_cost']:.2f}")
    print(f"  Delta Track Error:   {metrics['delta_tracking_error']:.2f}")
    print(f"  Avg Abs Delta:       {metrics['avg_abs_delta']:.2f}")

    print("\nRisk Metrics:")
    print(f"  VaR (95%):           {metrics['var_95']:.2%}")
    print(f"  CVaR (95%):          {metrics['cvar_95']:.2%}")
    print(f"  Skewness:            {metrics['skewness']:.3f}")
    print(f"  Kurtosis:            {metrics['kurtosis']:.3f}")
    print()

    # =========================================================================
    # STEP 9: Generate comprehensive visualizations
    # =========================================================================
    print("STEP 9: Generate Visualizations")
    print("-" * 80)

    # Static plots
    print("Creating static visualizations...")
    visualizer = StaticVisualizer(results, save_dir="plots/advanced_example")
    visualizer.generate_all_plots(save=True)
    print("  ✓ Static plots saved")

    # Interactive dashboard
    print("Creating interactive dashboard...")
    dashboard = InteractiveDashboard(
        results, save_dir="plots/advanced_example/interactive"
    )
    dashboard.generate_all_interactive_plots(save=True)
    print("  ✓ Interactive dashboard saved")
    print()

    # =========================================================================
    # STEP 10: Generate reports
    # =========================================================================
    print("STEP 10: Generate Reports")
    print("-" * 80)

    report_gen = ReportGenerator(results, output_dir="reports/advanced_example")

    html_path = report_gen.generate_html_report()
    text_path = report_gen.generate_text_report()

    print(f"  HTML report: {html_path}")
    print(f"  Text report: {text_path}")
    print()

    # Export results
    print("Exporting results...")
    results.export_to_excel("reports/advanced_example/results.xlsx")
    results.export_to_parquet("reports/advanced_example/results.parquet")
    print("  ✓ Results exported to Excel and Parquet")
    print()

    # =========================================================================
    # COMPLETION
    # =========================================================================
    print("=" * 80)
    print("ADVANCED EXAMPLE COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  Strategy:            {strategy.name}")
    print(f"  Underlying:          {underlying}")
    print(f"  Period:              {(end_date - start_date).days} days")
    print(f"  Initial Positions:   {len(initial_positions)}")
    print(f"  Hedges Executed:     {summary['num_hedges']}")
    print(
        f"  Final P&L:           ${summary['total_pnl']:,.2f} ({summary['total_return_pct']:.2f}%)"
    )
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Transaction Costs:   ${summary['total_transaction_costs']:,.2f}")
    print()
    print("All reports and visualizations have been generated!")
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
