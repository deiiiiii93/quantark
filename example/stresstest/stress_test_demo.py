"""
Comprehensive demonstration of the Stress Test module.

This example shows how to:
1. Create a portfolio with options
2. Define custom scenarios and use predefined ones
3. Run stress tests
4. Export results to multiple formats
5. Generate reports and visualizations
6. Save/load scenarios from files
"""

from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import sys


# Core imports
from quantark.portfolio import Portfolio
from quantark.priceenv import PricingEnvironment
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.asset.equity.product import EuropeanVanillaOption
from quantark.asset.equity.engine import BlackScholesEngine
from quantark.util.enum import OptionType

# Stress test imports
from quantark.stresstest import (
    StressTestEngine,
    StressTestConfig,
    ScenarioBuilder,
    StressType,
    StressLevel,
)
from quantark.stresstest.scenario import Scenario, Stress
from quantark.stresstest.scenario.scenario_library import ScenarioLibrary
from quantark.stresstest.scenario.scenario_storage import ScenarioStorage
from quantark.stresstest.results.result_exporter import ResultExporter
from quantark.stresstest.report import ReportGenerator, StressTestVisualizer


def create_sample_portfolio() -> Portfolio:
    """
    Create a sample portfolio with options on AAPL and GOOGL.

    Returns:
        Portfolio with multiple positions
    """
    print("=" * 60)
    print("Creating Sample Portfolio")
    print("=" * 60)

    valuation_date = datetime(2024, 1, 15)
    maturity_time = 0.5  # 6 months in years

    # Create pricing environments for two underlyings
    pricing_envs = {}

    # AAPL environment
    aapl_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=150.0, asset_name="AAPL"),
        vol_surface=FlatVolSurface(volatility=0.25),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=valuation_date,
    )
    pricing_envs["AAPL"] = aapl_env

    # GOOGL environment
    googl_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=2800.0, asset_name="GOOGL"),
        vol_surface=FlatVolSurface(volatility=0.30),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.005),
        valuation_date=valuation_date,
    )
    pricing_envs["GOOGL"] = googl_env

    # Create portfolio
    portfolio = Portfolio(
        portfolio_name="Tech Options Portfolio",
        pricing_environments=pricing_envs,
        creation_date=valuation_date,
    )

    # Create engines
    bs_engine = BlackScholesEngine()

    # Add AAPL positions
    # Long ATM call
    aapl_call = EuropeanVanillaOption(
        strike=150.0, maturity=maturity_time, option_type=OptionType.CALL
    )
    portfolio.add_position(
        product=aapl_call,
        quantity=100,
        entry_price=bs_engine.price(aapl_call, aapl_env),
        underlying="AAPL",
        engine=bs_engine,
        entry_timestamp=valuation_date,
    )

    # Short OTM put (protective)
    aapl_put = EuropeanVanillaOption(
        strike=140.0, maturity=maturity_time, option_type=OptionType.PUT
    )
    portfolio.add_position(
        product=aapl_put,
        quantity=-50,
        entry_price=bs_engine.price(aapl_put, aapl_env),
        underlying="AAPL",
        engine=bs_engine,
        entry_timestamp=valuation_date,
    )

    # Add GOOGL positions
    # Long OTM call
    googl_call = EuropeanVanillaOption(
        strike=2900.0, maturity=maturity_time, option_type=OptionType.CALL
    )
    portfolio.add_position(
        product=googl_call,
        quantity=50,
        entry_price=bs_engine.price(googl_call, googl_env),
        underlying="GOOGL",
        engine=bs_engine,
        entry_timestamp=valuation_date,
    )

    print(f"\nPortfolio created with {len(portfolio)} positions")
    print(f"Baseline portfolio value: ${portfolio.get_portfolio_value():,.2f}")
    print(f"Baseline P&L: ${portfolio.get_portfolio_pnl():,.2f}")

    return portfolio


def define_custom_scenarios():
    """
    Define custom stress test scenarios using the builder API.

    Returns:
        List of custom scenarios
    """
    print("\n" + "=" * 60)
    print("Defining Custom Scenarios")
    print("=" * 60)

    scenarios = []

    # Scenario 1: Tech sell-off with vol spike
    tech_selloff = (
        ScenarioBuilder()
        .name("Tech Sector Sell-Off")
        .description("15% drop in tech stocks with volatility spike")
        .spot_stress(-0.15)
        .vol_stress(0.40)
        .build()
    )
    scenarios.append(tech_selloff)

    # Scenario 2: Rate hike impact
    rate_hike_scenario = (
        ScenarioBuilder()
        .name("Fed Rate Hike")
        .description("200bps rate increase with moderate equity pressure")
        .rate_stress(0.02, stress_type=StressType.ABSOLUTE)
        .spot_stress(-0.05)
        .vol_stress(0.15)
        .build()
    )
    scenarios.append(rate_hike_scenario)

    # Scenario 3: AAPL-specific stress
    aapl_specific = (
        ScenarioBuilder()
        .name("AAPL Specific Risk")
        .description("AAPL drops 20% while GOOGL unaffected")
        .spot_stress(-0.20, underlying="AAPL")
        .vol_stress(0.50, underlying="AAPL")
        .build()
    )
    scenarios.append(aapl_specific)

    # Scenario 4: Calm recovery
    calm_recovery = (
        ScenarioBuilder()
        .name("Calm Recovery")
        .description("Modest gains with volatility compression")
        .spot_stress(0.10)
        .vol_stress(-0.25)
        .build()
    )
    scenarios.append(calm_recovery)

    print(f"Created {len(scenarios)} custom scenarios")
    return scenarios


def demonstrate_scenario_storage(scenarios: list):
    """
    Demonstrate saving and loading scenarios from files.

    Args:
        scenarios: List of scenarios to save
    """
    print("\n" + "=" * 60)
    print("Demonstrating Scenario Storage")
    print("=" * 60)

    output_dir = Path("./stress_scenarios")
    output_dir.mkdir(exist_ok=True)

    # Save to YAML
    yaml_path = output_dir / "custom_scenarios.yaml"
    ScenarioStorage.save_scenarios(scenarios, yaml_path)
    print(f"Saved scenarios to: {yaml_path}")

    # Save to JSON
    json_path = output_dir / "custom_scenarios.json"
    ScenarioStorage.save_scenarios(scenarios, json_path)
    print(f"Saved scenarios to: {json_path}")

    # Load back from YAML
    loaded_scenarios = ScenarioStorage.load_scenarios(yaml_path)
    print(f"Loaded {len(loaded_scenarios)} scenarios from YAML")

    return loaded_scenarios


def run_stress_test(portfolio: Portfolio, scenarios: list):
    """
    Run stress test with all scenarios.

    Args:
        portfolio: Portfolio to stress test
        scenarios: List of scenarios to test

    Returns:
        StressTestResults
    """
    print("\n" + "=" * 60)
    print("Running Stress Test")
    print("=" * 60)

    # Configure stress test
    config = StressTestConfig(
        calculate_greeks=True,
        greeks_method="analytical",
        export_formats=["parquet", "csv", "json"],
        output_dir="./stress_results",
        save_detailed_results=True,
    )

    print(f"Configuration: {config}")

    # Create engine
    engine = StressTestEngine(config)

    # Run stress test
    print("\nExecuting stress test...")
    results = engine.run_static_scenarios(portfolio, scenarios)

    print("\n" + results.get_summary())

    return results


def export_results(results):
    """
    Export results to multiple formats.

    Args:
        results: StressTestResults to export
    """
    print("\n" + "=" * 60)
    print("Exporting Results")
    print("=" * 60)

    output_dir = Path("./stress_results")

    # Export to all formats
    ResultExporter.export(
        results,
        output_dir,
        formats=["parquet", "csv", "json"],
        base_name="tech_portfolio_stress",
    )

    # Export risk metrics separately
    risk_metrics_path = output_dir / "risk_metrics.csv"
    ResultExporter.export_risk_metrics(results, risk_metrics_path)


def generate_visualizations(results):
    """
    Generate all visualizations.

    Args:
        results: StressTestResults to visualize
    """
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    output_dir = Path("./stress_results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizer
    visualizer = StressTestVisualizer()

    # Generate all plots
    visualizer.create_all_plots(results, output_dir, prefix="tech_portfolio")


def generate_html_report(results):
    """
    Generate HTML report.

    Args:
        results: StressTestResults to report
    """
    print("\n" + "=" * 60)
    print("Generating HTML Report")
    print("=" * 60)

    output_dir = Path("./stress_results/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create report generator
    report_gen = ReportGenerator()

    # Generate report
    report_path = output_dir / "stress_test_report.html"
    report_gen.generate_report(
        results, report_path, title="Tech Options Portfolio - Stress Test Report"
    )


def main():
    """Main demonstration function."""
    print("\n" + "=" * 80)
    print("STRESS TEST MODULE DEMONSTRATION")
    print("=" * 80)

    # Step 1: Create portfolio
    portfolio = create_sample_portfolio()

    # Step 2: Define custom scenarios
    custom_scenarios = define_custom_scenarios()

    # Step 3: Get predefined scenarios
    print("\n" + "=" * 60)
    print("Loading Predefined Scenarios")
    print("=" * 60)
    predefined = ScenarioLibrary.get_all_predefined()
    historical = ScenarioLibrary.get_historical_scenarios()
    print(f"Loaded {len(predefined)} predefined scenarios")
    print(f"Loaded {len(historical)} historical scenarios")

    # Combine scenarios
    all_scenarios = custom_scenarios + predefined[:3] + historical[:1]  # Select subset
    print(f"\nTotal scenarios to test: {len(all_scenarios)}")

    # Step 4: Demonstrate scenario storage
    demonstrate_scenario_storage(custom_scenarios)

    # Step 5: Run stress test
    results = run_stress_test(portfolio, all_scenarios)

    # Step 6: Export results
    export_results(results)

    # Step 7: Generate visualizations
    generate_visualizations(results)

    # Step 8: Generate HTML report
    generate_html_report(results)

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nOutput files created:")
    print("  - ./stress_scenarios/        - Saved scenario definitions")
    print("  - ./stress_results/          - Exported results (parquet, CSV, JSON)")
    print("  - ./stress_results/plots/    - Visualization plots")
    print("  - ./stress_results/reports/  - HTML report")
    print(
        "\nOpen ./stress_results/reports/stress_test_report.html to view the full report."
    )
    print(
        "Open ./stress_results/plots/tech_portfolio_interactive_dashboard.html for interactive visualizations."
    )


if __name__ == "__main__":
    main()
