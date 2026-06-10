"""
Demonstration of Dynamic Scenario Analysis with multi-day paths and hedging.

This script demonstrates:
1. Creating day paths using PathBuilder and PathLibrary
2. Running dynamic scenarios on a portfolio
3. Running scenarios with hedging strategies
4. Generating reports and exports
5. Analyzing day-by-day evolution of positions, P&L, and Greeks
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta


from quantark.portfolio import Portfolio
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType

# Dynamic scenario imports
from quantark.dynamicscenario import (
    DayPath, DayStep, ParameterChange,
    PathBuilder, PathLibrary,
    DynamicScenarioConfig, DynamicScenarioEngine,
    DynamicScenarioResults
)
from quantark.dynamicscenario.results.result_exporter import DynamicResultExporter
from quantark.dynamicscenario.report.dynamic_report import DynamicReportGenerator
from quantark.dynamicscenario.report.visualizer import DynamicScenarioVisualizer
from quantark.stresstest.stress.stress_types import StressType

# Hedging strategy from backtest module
from quantark.backtest.strategy.delta_neutral_strategy import DeltaNeutralStrategy
from quantark.backtest.transaction_costs import ProportionalCostModel


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def create_sample_portfolio() -> Portfolio:
    """Create a sample portfolio with options for demonstration."""
    
    # Set up market environment
    underlying = "AAPL"
    valuation_date = datetime(2024, 6, 1)
    
    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=150.0, asset_name=underlying),
        vol_surface=FlatVolSurface(volatility=0.25),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=valuation_date,
    )
    
    # Create portfolio
    portfolio = Portfolio(
        portfolio_name="Options_Portfolio",
        pricing_environments={underlying: pricing_env},
        creation_date=valuation_date,
    )
    
    # Add positions
    engine = BlackScholesEngine()
    
    # Long ATM Call
    call_atm = EuropeanVanillaOption(
        strike=150.0,
        option_type=OptionType.CALL,
        maturity=0.5,  # 6 months
    )
    portfolio.add_position(
        product=call_atm,
        quantity=100,  # 100 contracts (10,000 shares notional)
        entry_price=8.50,
        underlying=underlying,
        engine=engine,
    )
    
    # Long OTM Put (for protection)
    put_otm = EuropeanVanillaOption(
        strike=140.0,
        option_type=OptionType.PUT,
        maturity=0.5,
    )
    portfolio.add_position(
        product=put_otm,
        quantity=50,
        entry_price=3.20,
        underlying=underlying,
        engine=engine,
    )
    
    return portfolio


def demo_path_creation():
    """Demonstrate various ways to create day paths."""
    print_section("PATH CREATION EXAMPLES")
    
    # Example 1: Using PathBuilder
    print("\n1. PathBuilder - Custom 5-day rally")
    print("-" * 40)
    
    path1 = (PathBuilder(num_days=5, name="Custom Rally")
        .description("5 days of gradual price increase with vol decay")
        .spot_trend(daily_change=0.02)  # +2% per day
        .vol_trend(daily_change=-0.05)  # -5% vol per day
        .set_day_label(0, "Rally Start")
        .set_day_label(4, "Rally Peak")
        .build())
    
    print(path1.get_summary())
    
    # Example 2: Using PathLibrary
    print("\n2. PathLibrary - Predefined scenarios")
    print("-" * 40)
    
    # Consecutive decline
    path2 = PathLibrary.consecutive_decline(days=5, daily_pct=-0.03)
    print(f"\nConsecutive Decline: {path2.name}")
    print(f"  Days: {path2.num_days}")
    print(f"  Description: {path2.description}")
    
    # V-shaped recovery
    path3 = PathLibrary.v_shaped_recovery(down_days=3, up_days=5, magnitude=0.15)
    print(f"\nV-Shaped Recovery: {path3.name}")
    print(f"  Days: {path3.num_days}")
    print(f"  Description: {path3.description}")
    
    # Historical scenario
    path4 = PathLibrary.historical_black_monday()
    print(f"\nHistorical Black Monday: {path4.name}")
    print(f"  Days: {path4.num_days}")
    print(f"  Description: {path4.description}")
    
    # Example 3: Custom path with exact values
    print("\n3. PathLibrary - Custom path with exact values")
    print("-" * 40)
    
    path5 = PathLibrary.custom_path(
        spot_values=[150.0, 155.0, 152.0, 158.0, 160.0],
        vol_values=[0.25, 0.28, 0.26, 0.24, 0.23],
        name="Custom Exact Path",
        description="Explicit spot and vol values",
    )
    print(path5.get_summary())
    
    return path1


def demo_basic_scenario(portfolio: Portfolio):
    """Demonstrate basic dynamic scenario without hedging."""
    print_section("BASIC DYNAMIC SCENARIO (NO HEDGING)")
    
    # Create a 5-day rally scenario
    path = PathLibrary.consecutive_rally(
        days=5,
        daily_pct=0.02,
        vol_change_pct=-0.03,
        start_date=datetime(2024, 6, 1)
    )
    
    print(f"\nPath: {path.name}")
    print(f"Description: {path.description}")
    print(f"Days: {path.num_days}")
    
    # Create engine and run
    config = DynamicScenarioConfig(
        calculate_greeks=True,
        greeks_method='analytical',
    )
    engine = DynamicScenarioEngine(config)
    
    print("\nRunning scenario...")
    results = engine.run(portfolio, path)
    
    # Print summary
    print("\n" + results.get_summary())
    
    return results


def demo_scenario_with_hedging(portfolio: Portfolio):
    """Demonstrate dynamic scenario with delta hedging."""
    print_section("DYNAMIC SCENARIO WITH DELTA HEDGING")
    
    # Create a volatile V-shaped scenario
    path = PathLibrary.v_shaped_recovery(
        down_days=3,
        up_days=4,
        magnitude=0.10,
        vol_spike=0.50,
        start_date=datetime(2024, 6, 1)
    )
    
    print(f"\nPath: {path.name}")
    print(f"Description: {path.description}")
    print(f"Days: {path.num_days}")
    
    # Create hedging strategy
    hedge_strategy = DeltaNeutralStrategy(
        name="DeltaHedge_100",
        delta_threshold=100.0,  # Hedge when delta exceeds 100
        rebalance_frequency='daily',
        hedge_instrument='spot',
        hedge_ratio=1.0,
    )
    print(f"\nHedge Strategy: {hedge_strategy.name}")
    print(f"  Delta Threshold: {hedge_strategy.delta_threshold}")
    print(f"  Frequency: {hedge_strategy.rebalance_frequency}")
    
    # Transaction cost model
    cost_model = ProportionalCostModel(commission_rate=0.001)  # 10 bps
    print(f"  Transaction Costs: 10 bps")
    
    # Create engine and run
    config = DynamicScenarioConfig(
        calculate_greeks=True,
        greeks_method='analytical',
    )
    engine = DynamicScenarioEngine(config)
    
    print("\nRunning scenario with hedging...")
    results = engine.run(
        portfolio=portfolio,
        day_path=path,
        hedge_strategy=hedge_strategy,
        transaction_cost_model=cost_model,
    )
    
    # Print summary
    print("\n" + results.get_summary())
    
    return results


def demo_crash_scenario(portfolio: Portfolio):
    """Demonstrate a market crash scenario."""
    print_section("MARKET CRASH SCENARIO")
    
    # Create gradual crash scenario
    path = PathLibrary.gradual_crash(
        days=10,
        total_decline=-0.25,  # 25% decline
        vol_spike=0.80,       # 80% vol increase
        rate_change=-0.01,    # 100 bps rate cut
        start_date=datetime(2024, 6, 1)
    )
    
    print(f"\nPath: {path.name}")
    print(f"Description: {path.description}")
    print(f"Days: {path.num_days}")
    
    # Create engine and run
    config = DynamicScenarioConfig(calculate_greeks=True)
    engine = DynamicScenarioEngine(config)
    
    print("\nRunning crash scenario...")
    results = engine.run(portfolio, path)
    
    # Print summary
    print("\n" + results.get_summary())
    
    # Show worst and best days
    worst = results.get_worst_day()
    best = results.get_best_day()
    
    if worst:
        print(f"\nWorst Day Details:")
        print(worst.get_summary())
    
    if best:
        print(f"\nBest Day Details:")
        print(best.get_summary())
    
    return results


def demo_compare_hedged_vs_unhedged(portfolio: Portfolio):
    """Compare hedged vs unhedged performance."""
    print_section("HEDGED VS UNHEDGED COMPARISON")
    
    # Create a volatile scenario
    path = PathLibrary.volatility_spike_decay(
        spike_pct=0.80,
        decay_days=7,
        spot_shock=-0.08,
        start_date=datetime(2024, 6, 1)
    )
    
    print(f"\nPath: {path.name}")
    print(f"Description: {path.description}")
    
    config = DynamicScenarioConfig(calculate_greeks=True)
    engine = DynamicScenarioEngine(config)
    
    # Run without hedging
    print("\n1. Running WITHOUT hedging...")
    results_unhedged = engine.run(portfolio, path)
    
    # Run with hedging
    print("\n2. Running WITH hedging...")
    hedge_strategy = DeltaNeutralStrategy(
        delta_threshold=50.0,
        rebalance_frequency='daily',
    )
    cost_model = ProportionalCostModel(commission_rate=0.001)
    
    results_hedged = engine.run(
        portfolio=portfolio,
        day_path=path,
        hedge_strategy=hedge_strategy,
        transaction_cost_model=cost_model,
    )
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON: HEDGED VS UNHEDGED")
    print("=" * 60)
    print(f"{'Metric':<30} {'Unhedged':>15} {'Hedged':>15}")
    print("-" * 60)
    print(f"{'Initial Value':<30} ${results_unhedged.baseline_value:>14,.2f} ${results_hedged.baseline_value:>14,.2f}")
    print(f"{'Final Value':<30} ${results_unhedged.final_value:>14,.2f} ${results_hedged.final_value:>14,.2f}")
    print(f"{'Total P&L':<30} ${results_unhedged.total_pnl:>14,.2f} ${results_hedged.total_pnl:>14,.2f}")
    print(f"{'Total P&L %':<30} {results_unhedged.total_pnl_pct:>14.2f}% {results_hedged.total_pnl_pct:>14.2f}%")
    print(f"{'Transaction Costs':<30} ${results_unhedged.total_transaction_costs:>14,.2f} ${results_hedged.total_transaction_costs:>14,.2f}")
    print(f"{'Net P&L':<30} ${results_unhedged.net_pnl:>14,.2f} ${results_hedged.net_pnl:>14,.2f}")
    print(f"{'Hedge Trades':<30} {results_unhedged.total_hedges:>15} {results_hedged.total_hedges:>15}")
    
    dd_unhedged = results_unhedged.get_max_drawdown()
    dd_hedged = results_hedged.get_max_drawdown()
    print(f"{'Max Drawdown':<30} ${dd_unhedged[0]:>14,.2f} ${dd_hedged[0]:>14,.2f}")
    print(f"{'Max Drawdown %':<30} {dd_unhedged[1]:>14.2f}% {dd_hedged[1]:>14.2f}%")
    
    return results_hedged, results_unhedged


def demo_export_and_report(results: DynamicScenarioResults):
    """Demonstrate export, report, and visualization generation."""
    print_section("EXPORT, VISUALIZATION AND REPORT GENERATION")
    
    output_dir = Path("./dynamic_results/demo_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export to CSV and JSON
    print("\n1. Exporting results...")
    exporter = DynamicResultExporter(results)
    files = exporter.export_all(
        output_dir=output_dir,
        formats=['csv', 'json'],
        prefix="demo_scenario"
    )
    
    print(f"\nExported {len(files)} files to {output_dir}")
    
    # Generate plots and dashboard
    print("\n2. Generating plots and dashboard...")
    visualizer = DynamicScenarioVisualizer()
    plot_dir = output_dir / "plots"
    plot_files = visualizer.create_all_plots(
        results=results,
        output_dir=plot_dir,
        prefix="demo"
    )
    print(f"\nGenerated {len(plot_files)} plot files in {plot_dir}")
    
    # Generate HTML report
    print("\n3. Generating HTML report...")
    report_gen = DynamicReportGenerator()
    report_path = output_dir / "demo_report.html"
    report_gen.generate_report(
        results=results,
        output_path=report_path,
        title="Dynamic Scenario Analysis Demo Report",
        include_charts=True,
    )
    
    print(f"\nGenerated report: {report_path}")
    
    # Show DataFrame examples
    print("\n4. Data Analysis Examples")
    print("-" * 40)
    
    # P&L evolution
    pnl_df = results.get_pnl_evolution()
    print("\nP&L Evolution (first 5 rows):")
    print(pnl_df.head().to_string(index=False))
    
    # Greeks evolution
    greeks_df = results.get_greeks_evolution()
    print("\nGreeks Evolution (first 5 rows):")
    print(greeks_df[['day_index', 'delta', 'gamma', 'vega']].head().to_string(index=False))
    
    return output_dir


def main():
    """Run all demonstrations."""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  QUANTARK - Dynamic Scenario Analysis Demonstration".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    
    try:
        # Demo 1: Path creation
        path = demo_path_creation()
        
        # Create sample portfolio
        print_section("CREATING SAMPLE PORTFOLIO")
        portfolio = create_sample_portfolio()
        print(f"\nPortfolio: {portfolio.portfolio_name}")
        print(f"Positions: {len(portfolio.positions)}")
        print(f"Initial Value: ${portfolio.get_portfolio_value():,.2f}")
        
        # Demo 2: Basic scenario
        results_basic = demo_basic_scenario(portfolio)
        
        # Demo 3: Scenario with hedging
        results_hedged = demo_scenario_with_hedging(portfolio)
        
        # Demo 4: Crash scenario
        results_crash = demo_crash_scenario(portfolio)
        
        # Demo 5: Compare hedged vs unhedged
        results_comparison = demo_compare_hedged_vs_unhedged(portfolio)
        
        # Demo 6: Export and report
        output_dir = demo_export_and_report(results_hedged)
        
        print_section("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("\nSummary:")
        print(f"  - Demonstrated path creation (PathBuilder, PathLibrary)")
        print(f"  - Ran basic scenario without hedging")
        print(f"  - Ran scenario with delta hedging strategy")
        print(f"  - Simulated market crash scenario")
        print(f"  - Compared hedged vs unhedged performance")
        print(f"  - Generated exports, plots, and HTML report in {output_dir}")
        print("\nGenerated visualizations include:")
        print("  - Portfolio value evolution plot")
        print("  - P&L evolution (daily and cumulative)")
        print("  - Greeks evolution (delta, gamma, vega, theta)")
        print("  - Market parameters evolution (spot, vol, rate)")
        print("  - Drawdown chart")
        print("  - Summary dashboard (static PNG)")
        print("  - Interactive dashboard (HTML with Plotly)")
        print("\nThe dynamic scenario module enables:")
        print("  - Multi-day market evolution simulation")
        print("  - Integration with hedging strategies")
        print("  - Day-by-day tracking of positions, P&L, Greeks")
        print("  - Rich visualization and reporting capabilities")
        
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

