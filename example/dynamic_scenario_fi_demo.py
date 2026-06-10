#!/usr/bin/env python3
"""
FI Dynamic Scenario Analysis Demo

This example demonstrates how to use the FI dynamic scenario analysis module
to simulate a Fixed Income portfolio through multi-day rate scenarios.

Features demonstrated:
1. Creating an FI portfolio with Treasury bonds
2. Using FIPathLibrary for rate scenario patterns
3. Running dynamic simulations with FIDynamicScenarioEngine
4. Analyzing DV01/duration evolution
5. Optional DV01 hedging with bond futures
6. Visualization and reporting
"""

from datetime import datetime, timedelta
import os

import sys
from pathlib import Path


# Import FI portfolio and products
from quantark.portfolio.fi import FIPortfolio
from quantark.asset.bond.product import FixedBond, create_simple_fixed_bond
from quantark.asset.bond.engine import BondDiscountEngine
from quantark.priceenv import PricingEnvironment
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.util.enum import PaymentFrequency
from quantark.util.calendar import DayCountConvention

# Import dynamic scenario components
from quantark.dynamicscenario import (
    FIDynamicScenarioConfig,
    FIDynamicScenarioEngine,
    FIPathLibrary,
    DynamicScenarioVisualizer,
    DynamicReportGenerator,
)
from quantark.dynamicscenario.results.result_exporter import FIResultExporter


def create_demo_fi_portfolio():
    """
    Create a demo FI portfolio with Treasury bonds.

    Returns:
        Tuple of (FIPortfolio, BondDiscountEngine)
    """
    # Set up valuation date
    valuation_date = datetime.now()

    # Create pricing environment with flat rate curve
    rate_curve = FlatRateCurve(rate=0.045)  # 4.5% rate

    pricing_env = PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=valuation_date,
    )

    # Create portfolio
    portfolio = FIPortfolio(
        portfolio_name="Treasury Bond Portfolio",
        pricing_environments={"UST": pricing_env},
    )

    # Create pricing engine
    engine = BondDiscountEngine(pricing_env)

    # Add some Treasury bonds to portfolio

    # 1. 2-Year Treasury Note (shorter duration)
    maturity_2y = valuation_date + timedelta(days=730)

    bond_2y = create_simple_fixed_bond(
        issue_date=valuation_date,
        maturity_date=maturity_2y,
        denominator=1000000,  # $1M face
        coupon_rate=0.04,  # 4% coupon
        payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        day_count_convention=DayCountConvention.ACT_ACT_ISDA,
    )

    portfolio.add_position(
        product=bond_2y,
        quantity=100,  # 100 bonds
        entry_price=99.5,
        underlying="UST",
        engine=engine,
    )

    # 2. 5-Year Treasury Note (medium duration)
    maturity_5y = valuation_date + timedelta(days=1825)

    bond_5y = create_simple_fixed_bond(
        issue_date=valuation_date,
        maturity_date=maturity_5y,
        denominator=1000000,
        coupon_rate=0.0425,  # 4.25% coupon
        payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        day_count_convention=DayCountConvention.ACT_ACT_ISDA,
    )

    portfolio.add_position(
        product=bond_5y,
        quantity=50,
        entry_price=98.0,
        underlying="UST",
        engine=engine,
    )

    # 3. 10-Year Treasury Bond (longer duration)
    maturity_10y = valuation_date + timedelta(days=3650)

    bond_10y = create_simple_fixed_bond(
        issue_date=valuation_date,
        maturity_date=maturity_10y,
        denominator=1000000,
        coupon_rate=0.0450,  # 4.5% coupon
        payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        day_count_convention=DayCountConvention.ACT_ACT_ISDA,
    )

    portfolio.add_position(
        product=bond_10y,
        quantity=25,
        entry_price=97.0,
        underlying="UST",
        engine=engine,
    )

    print(f"\nCreated FI Portfolio: {portfolio.portfolio_name}")
    print(f"  Positions: {len(portfolio.positions)}")
    print(f"  Total Value: ${portfolio.get_portfolio_value():,.2f}")
    print(f"  Total DV01: ${portfolio.get_portfolio_dv01():,.2f}")
    print(f"  Portfolio Duration: {portfolio.get_portfolio_duration():.2f} years")

    return portfolio, engine


def run_parallel_shift_scenario(portfolio: FIPortfolio, output_dir: str):
    """
    Run a parallel rate shift scenario.

    Simulates a 50bps rate increase over 5 days.
    """
    print("\n" + "=" * 60)
    print("SCENARIO 1: Parallel Rate Shift (+50bps)")
    print("=" * 60)

    # Create FI dynamic scenario config
    config = FIDynamicScenarioConfig(
        calculate_dv01=True,
        calculate_convexity=True,
        calculate_duration=True,
        hedge_enabled=False,  # No hedging for this scenario
        output_dir=output_dir,
    )

    # Create engine
    engine = FIDynamicScenarioEngine(config)

    # Create rate path using FIPathLibrary
    path = FIPathLibrary.parallel_shift(
        days=5,
        total_bps=50,
        start_date=datetime.now(),
    )

    print(f"\nPath: {path.name}")
    print(f"Description: {path.description}")
    print(path.get_summary())

    # Run simulation
    results = engine.run(portfolio, path)

    # Print results summary
    print("\n" + results.get_summary())

    return results


def run_rate_hike_cycle_with_hedging(portfolio: FIPortfolio, output_dir: str):
    """
    Run a rate hike cycle scenario with DV01 hedging.

    Simulates a Fed tightening cycle with bond futures hedging.
    """
    print("\n" + "=" * 60)
    print("SCENARIO 2: Rate Hike Cycle with DV01 Hedging")
    print("=" * 60)

    # Create config with hedging enabled
    config = FIDynamicScenarioConfig(
        calculate_dv01=True,
        calculate_convexity=True,
        calculate_duration=True,
        hedge_enabled=True,
        hedge_dv01_threshold=50000,  # Hedge when DV01 exceeds $50k
        futures_dv01_per_contract=80.0,  # ~$80 DV01 per Treasury futures contract
        output_dir=output_dir,
    )

    # Create engine
    engine = FIDynamicScenarioEngine(config)

    # Create rate hike cycle path
    path = FIPathLibrary.rate_hike_cycle(
        days=10,
        total_bps=100,  # 100bps total increase
        start_date=datetime.now(),
    )

    print(f"\nPath: {path.name}")
    print(f"Description: {path.description}")

    # Run simulation
    results = engine.run(portfolio, path)

    # Print results
    print("\n" + results.get_summary())

    return results


def run_curve_steepener_scenario(portfolio: FIPortfolio, output_dir: str):
    """
    Run a curve steepener scenario.

    Short rates down, long rates up - impacts duration differently.
    """
    print("\n" + "=" * 60)
    print("SCENARIO 3: Curve Steepener")
    print("=" * 60)

    config = FIDynamicScenarioConfig(
        calculate_dv01=True,
        calculate_duration=True,
        hedge_enabled=False,
        output_dir=output_dir,
    )

    engine = FIDynamicScenarioEngine(config)

    # Create steepener path
    path = FIPathLibrary.steepener(
        days=5,
        short_bps=-25,
        long_bps=25,
        start_date=datetime.now(),
    )

    print(f"\nPath: {path.name}")
    print(f"Description: {path.description}")

    results = engine.run(portfolio, path)

    print("\n" + results.get_summary())

    return results


def generate_reports(results_list, output_dir: str):
    """
    Generate visualizations and reports for all scenarios.
    """
    print("\n" + "=" * 60)
    print("GENERATING REPORTS")
    print("=" * 60)

    viz = DynamicScenarioVisualizer()
    reporter = DynamicReportGenerator()

    for i, results in enumerate(results_list, 1):
        scenario_dir = os.path.join(output_dir, f"scenario_{i}")
        os.makedirs(scenario_dir, exist_ok=True)

        # Generate FI-specific plots
        print(f"\nGenerating plots for: {results.path_name}")
        viz.create_fi_all_plots(results, scenario_dir)

        # Generate HTML report
        report_path = os.path.join(scenario_dir, "report.html")
        reporter.generate_fi_report(results, report_path)

        # Export data
        exporter = FIResultExporter(results)
        exporter.export_all(scenario_dir, formats=["csv", "json"])


def main():
    """Main demo entry point."""
    print("=" * 60)
    print("FI DYNAMIC SCENARIO ANALYSIS DEMO")
    print("=" * 60)

    # Set up output directory
    output_dir = "./dynamic_results/fi_demo"
    os.makedirs(output_dir, exist_ok=True)

    # Create demo portfolio
    portfolio, engine = create_demo_fi_portfolio()

    # Run scenarios
    results_list = []

    # Scenario 1: Parallel shift
    results1 = run_parallel_shift_scenario(portfolio, output_dir)
    results_list.append(results1)

    # Re-create portfolio for fresh state
    portfolio, _ = create_demo_fi_portfolio()

    # Scenario 2: Rate hike with hedging
    results2 = run_rate_hike_cycle_with_hedging(portfolio, output_dir)
    results_list.append(results2)

    # Re-create portfolio again
    portfolio, _ = create_demo_fi_portfolio()

    # Scenario 3: Curve steepener
    results3 = run_curve_steepener_scenario(portfolio, output_dir)
    results_list.append(results3)

    # Generate reports
    generate_reports(results_list, output_dir)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print("\nScenario Summary:")
    for i, results in enumerate(results_list, 1):
        print(
            f"  {i}. {results.path_name}: P&L ${results.total_pnl:+,.2f} ({results.total_pnl_pct:+.2f}%)"
        )


if __name__ == "__main__":
    main()
