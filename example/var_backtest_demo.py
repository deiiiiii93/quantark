"""
VaR Backtesting and Model Validation demonstration.

This example demonstrates how to validate VaR models using statistical backtesting
techniques including Kupiec POF test and Christoffersen independence test.

Purpose of Backtesting:
- Validate VaR model accuracy
- Check if actual losses exceed VaR at expected frequency
- Test for independence of violations
- Ensure regulatory compliance (Basel III/IV)
- Identify model weaknesses and areas for improvement

Statistical Tests:
1. Kupiec POF (Proportion of Failures) Test:
   - Tests if violation rate matches VaR confidence level
   - H0: Actual violation rate = Expected violation rate
   - Fail to reject H0 → Model is accurate

2. Christoffersen Independence Test:
   - Tests if violations are independent across time
   - H0: Violations are independent (no clustering)
   - Fail to reject H0 → Violations don't cluster

Basel Traffic Light Regime:
- Green (Zone 1): Acceptable model (≤ 5 violations/250 days)
- Yellow (Zone 2): Acceptable with supervisory response
- Red (Zone 3): Unacceptable model (> 12 violations/250 days)

Regulatory Requirements:
- Daily VaR calculations
- Backtest violations over 250-day window
- Report to regulators (Basel Committee)
- Model review and recalibration if needed
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from portfolio.equity.portfolio import EquityPortfolio
from priceenv import PricingEnvironment
from util.enum.option_enums import OptionType

from var import (
    VaRConfig,
    VaRMethod,
    HistoricalVaREngine,
    VaRBacktester,
    EquityRiskFactorConfig,
)


def create_portfolio_for_backtesting():
    """Create a portfolio for backtesting demonstration."""
    valuation_date = datetime(2024, 1, 1)

    spot_quote = SpotQuote(spot=100.0, timestamp=valuation_date)
    vol_surface = FlatVolSurface(volatility=0.25)
    rate_curve = FlatRateCurve(rate=0.05)
    div_yield = ContinuousDividendYield(div_yield=0.02)

    pricing_env = PricingEnvironment(
        spot_quote=spot_quote,
        vol_surface=vol_surface,
        rate_curve=rate_curve,
        div_yield=div_yield,
        valuation_date=valuation_date,
    )

    portfolio = EquityPortfolio(
        portfolio_name="Backtesting Demo Portfolio",
        pricing_environments={"SPY": pricing_env},
    )

    # Diversified options portfolio
    call_95 = EuropeanVanillaOption(
        strike=95.0, exercise_date=datetime(2024, 3, 1), option_type=OptionType.CALL
    )

    call_105 = EuropeanVanillaOption(
        strike=105.0, exercise_date=datetime(2024, 3, 1), option_type=OptionType.CALL
    )

    put_95 = EuropeanVanillaOption(
        strike=95.0, exercise_date=datetime(2024, 6, 1), option_type=OptionType.PUT
    )

    engine = BlackScholesEngine()

    portfolio.add_position(
        product=call_95,
        quantity=50,
        entry_price=8.5,
        underlying="SPY",
        engine=engine,
    )

    portfolio.add_position(
        product=call_105,
        quantity=50,
        entry_price=3.2,
        underlying="SPY",
        engine=engine,
    )

    portfolio.add_position(
        product=put_95,
        quantity=50,
        entry_price=4.8,
        underlying="SPY",
        engine=engine,
    )

    return portfolio


def generate_historical_data_extended(num_days=800):
    """Generate extended historical market data for backtesting."""
    np.random.seed(42)
    dates = pd.date_range(start=datetime(2021, 1, 1), periods=num_days, freq="D")

    # Generate more realistic market data with volatility clustering
    spot_returns = np.random.normal(0.0004, 0.017, num_days)

    # Add volatility clustering (GARCH-like behavior)
    for i in range(1, num_days):
        vol_factor = 1 + 0.15 * abs(spot_returns[i-1]) * 10
        spot_returns[i] *= vol_factor

    # Add some crisis periods
    crisis_periods = [
        (200, 220, -0.25, 0.80),  # Crisis 1
        (500, 520, -0.20, 0.60),  # Crisis 2
    ]

    for start, end, shock, vol_up in crisis_periods:
        spot_returns[start:end] += np.random.normal(shock, 0.03, end-start)
        spot_returns[start:end] *= (1 + vol_up)

    vol_changes = np.random.normal(0.0, 0.012, num_days)
    rate_shifts = np.random.normal(0.0, 0.0006, num_days)

    data = pd.DataFrame(
        {
            "spot_return": spot_returns,
            "vol_change": vol_changes,
            "rate_shift": rate_shifts,
        },
        index=dates,
    )

    return data


def generate_portfolio_pnl_timeseries(portfolio, historical_data, method="Historical"):
    """Generate historical P&L time series for backtesting."""
    print(f"Generating {len(historical_data)} days of portfolio P&L...")
    print("(This may take a moment for full revaluation...)")

    # Use Historical VaR engine to get scenario P&L
    var_config = VaRConfig(
        confidence_level=0.99,
        holding_period=1,
        var_method=VaRMethod.HISTORICAL,
    )

    engine = HistoricalVaREngine(config=var_config)

    # Calculate VaR for each day (rolling window)
    pnl_series = []
    var_series = []

    # Use 252-day rolling window for backtesting
    window_size = 252
    start_date = historical_data.index[window_size]

    for i in range(window_size, len(historical_data)):
        # Get historical data up to current date
        hist_data_window = historical_data.iloc[:i]

        # Calculate VaR
        result = engine.calculate_var(portfolio, hist_data_window)
        var_series.append(result.var)

        # Get portfolio P&L for current scenario
        if result.scenarios is not None and not result.scenarios.empty:
            # Use the latest scenario P&L
            pnl = result.scenarios.iloc[-1]['portfolio_pnl']
        else:
            # Fallback: estimate P&L
            pnl = portfolio.get_portfolio_value() * historical_data.iloc[i]['spot_return']

        pnl_series.append(pnl)

        # Progress indicator
        if (i - window_size) % 100 == 0:
            print(f"  Progress: {i-window_size}/{len(historical_data)-window_size} days...")

    # Create time series
    dates = historical_data.index[window_size:]
    portfolio_pnl = pd.Series(pnl_series, index=dates, name='portfolio_pnl')
    var_values = pd.Series(var_series, index=dates, name='var')

    return portfolio_pnl, var_values


def main():
    """Run VaR backtesting demonstration and display results."""
    print("=" * 80)
    print("VaR Backtesting and Model Validation Demonstration")
    print("=" * 80)
    print()
    print("Purpose: Validate VaR model accuracy using statistical tests")
    print("Tests: Kupiec POF test, Christoffersen independence test")
    print("Framework: Basel regulatory requirements")
    print()

    portfolio = create_portfolio_for_backtesting()
    historical_data = generate_historical_data_extended()

    print("=" * 80)
    print("Portfolio Summary")
    print("=" * 80)
    print(f"Portfolio Name: {portfolio.portfolio_name}")
    print(f"Portfolio value: ${portfolio.get_portfolio_value():,.2f}")
    print(f"Number of positions: {len(portfolio.positions)}")
    print()

    for i, (pos_id, position) in enumerate(portfolio.positions.items(), 1):
        product_name = position.product.__class__.__name__
        strike = position.product.strike
        opt_type = "CALL" if position.product.option_type == OptionType.CALL else "PUT"
        print(f"  {i}. {opt_type:4s} {strike:6.1f} | Qty: {position.quantity:4d}")

    print()
    print("=" * 80)
    print("Generating Historical P&L and VaR Time Series")
    print("=" * 80)

    # Generate portfolio P&L and VaR time series
    portfolio_pnl, var_values = generate_portfolio_pnl_timeseries(portfolio, historical_data)

    print()
    print(f"Generated {len(portfolio_pnl)} days of data")
    print(f"Date range: {portfolio_pnl.index[0].date()} to {portfolio_pnl.index[-1].date()}")
    print()

    print("=" * 80)
    print("Backtesting Configuration")
    print("=" * 80)

    confidence_level = 0.99
    expected_violation_rate = 1 - confidence_level
    total_days = len(portfolio_pnl)
    expected_violations = int(total_days * expected_violation_rate)

    print(f"Confidence Level: {confidence_level:.1%}")
    print(f"Expected Violation Rate: {expected_violation_rate:.2%}")
    print(f"Total Backtesting Days: {total_days}")
    print(f"Expected Violations (if model perfect): {expected_violations:.1f}")
    print()

    print("=" * 80)
    print("Running VaR Backtester")
    print("=" * 80)

    # Run backtester
    backtester = VaRBacktester(confidence_level=confidence_level)
    backtest_result = backtester.run_backtest(portfolio_pnl, var_values)

    print()
    print("=" * 80)
    print("Backtest Results Summary")
    print("=" * 80)
    print()

    # Summary statistics
    print(f"Total Days: {backtest_result.total_days}")
    print(f"Number of Violations: {backtest_result.num_violations}")
    print(f"Violation Rate: {backtest_result.violation_rate:.2%}")
    print(f"Expected Rate: {expected_violation_rate:.2%}")
    print(f"Expected Violations: {expected_violations:.1f}")

    if backtest_result.num_violations > 0:
        print(f"Average Loss on Violation Days: ${backtest_result.avg_loss_on_violations:,.2f}")
        print(f"Maximum Single-Day Loss: ${backtest_result.max_loss:,.2f}")
    else:
        print("No violations observed (model was conservative)")

    print()

    # Basel Traffic Light Classification
    print("=" * 80)
    print("Basel Traffic Light Regime")
    print("=" * 80)
    print()

    violations = backtest_result.num_violations
    zone_classification = backtest_result.zone_classification

    print(f"Violations (out of {total_days} days): {violations}")
    print(f"Basel Zone: {zone_classification}")

    if zone_classification == "Zone 1 (Green)":
        print()
        print("✓ GREEN ZONE - Acceptable Model")
        print("  • Violations ≤ 5")
        print("  • Model is considered accurate")
        print("  • No supervisory action required")
        print("  • Continue current model")
    elif zone_classification == "Zone 2 (Yellow)":
        print()
        print("⚠ YELLOW ZONE - Acceptable with Response")
        print("  • Violations: 6-10")
        print("  • Model needs review")
        print("  • Supervisory response required")
        print("  • Consider model improvements")
    else:
        print()
        print("✗ RED ZONE - Unacceptable Model")
        print("  • Violations ≥ 11")
        print("  • Model is inadequate")
        print("  • Immediate supervisory action")
        print("  • Model must be revised")

    print()

    # Statistical Tests
    print("=" * 80)
    print("Statistical Test Results")
    print("=" * 80)
    print()

    # Kupiec POF Test
    print("1. Kupiec POF (Proportion of Failures) Test")
    print("-" * 50)
    print(f"   H₀: Violation rate = {expected_violation_rate:.2%}")
    print(f"   H₁: Violation rate ≠ {expected_violation_rate:.2%}")
    print()
    print(f"   Test Statistic (LR_POF): {backtest_result.kupiec_stat:.4f}")
    print(f"   p-value: {backtest_result.kupiec_pvalue:.4f}")
    print(f"   Critical Value (99%): {backtest_result.kupiec_critical_value:.4f}")

    if backtest_result.kupiec_pvalue > 0.01:
        print()
        print("   ✓ PASS: Fail to reject H₀")
        print("     Violation rate is statistically consistent with expected rate")
    else:
        print()
        print("   ✗ FAIL: Reject H₀")
        print("     Violation rate significantly different from expected")

    print()

    # Christoffersen Independence Test
    print("2. Christoffersen Independence Test")
    print("-" * 50)
    print(f"   H₀: Violations are independent across time")
    print(f"   H₁: Violations are not independent (clustered)")
    print()
    print(f"   Test Statistic (LR_ind): {backtest_result.christoffersen_stat:.4f}")
    print(f"   p-value: {backtest_result.christoffersen_pvalue:.4f}")

    if backtest_result.christoffersen_pvalue > 0.01:
        print()
        print("   ✓ PASS: Fail to reject H₀")
        print("     Violations are independent (no clustering)")
    else:
        print()
        print("   ✗ FAIL: Reject H₀")
        print("     Violations cluster together (model may miss volatility)")

    print()

    # Combined Test
    print("=" * 80)
    print("Combined Test Results")
    print("=" * 80)
    print()

    print(f"Combined Test Statistic (LR_cc): {backtest_result.combined_stat:.4f}")
    print(f"p-value: {backtest_result.combined_pvalue:.4f}")

    if backtest_result.combined_pvalue > 0.01:
        print()
        print("✓ OVERALL: Model passes statistical validation")
        print("  VaR model is both accurate (correct violation rate)")
        print("  and well-calibrated (violations are independent)")
    else:
        print()
        print("✗ OVERALL: Model fails statistical validation")
        print("  Model needs revision to improve accuracy or calibration")

    print()

    # Violation Analysis
    if violations > 0:
        print("=" * 80)
        print("Violation Analysis")
        print("=" * 80)
        print()

        print("Violations by Month:")
        violation_dates = [d for d in portfolio_pnl.index if portfolio_pnl[d] < -var_values[d]]
        violation_months = pd.Series(violation_dates).dt.to_period('M').value_counts().sort_index()

        for month, count in violation_months.items():
            print(f"  {month}: {count} violations")

        print()

    # Model Validation Recommendations
    print("=" * 80)
    print("Model Validation & Recommendations")
    print("=" * 80)
    print()

    if violations <= 5 and backtest_result.kupiec_pvalue > 0.01:
        print("✓ Model Status: VALIDATED")
        print()
        print("Recommendations:")
        print("  • Continue using current VaR model")
        print("  • Monitor ongoing performance")
        print("  • Re-validate quarterly")
        print("  • Update historical data as available")
        print("  • Consider stress testing")

    elif violations <= 10:
        print("⚠ Model Status: NEEDS REVIEW")
        print()
        print("Recommendations:")
        print("  • Investigate causes of violations")
        print("  • Review VaR methodology")
        print("  • Consider different VaR method")
        print("  • Update calibration parameters")
        print("  • Implement model improvements")
        print("  • Increase monitoring frequency")

    else:
        print("✗ Model Status: REVISION REQUIRED")
        print()
        print("Recommendations:")
        print("  • IMMEDIATE: Review VaR methodology")
        print("  • Consider alternative VaR methods")
        print("  • Increase confidence level (e.g., 99% → 99.5%)")
        print("  • Implement conservative adjustments")
        print("  • Daily model recalibration")
        print("  • Supervisory approval before use")

    print()
    print("=" * 80)
    print("Key Backtesting Principles")
    print("=" * 80)
    print()
    print("1. Frequency:")
    print("   • Daily backtesting minimum")
    print("   • Monthly reporting to management")
    print("   • Quarterly regulatory submissions")
    print()
    print("2. Sample Size:")
    print("   • Minimum 250 days for reliable statistics")
    print("   • 1+ years for robust validation")
    print("   • Include various market regimes")
    print()
    print("3. Model Lifecycle:")
    print("   • Initial validation before deployment")
    print("   • Ongoing validation (monthly/quarterly)")
    print("   • Revalidation after major model changes")
    print("   • Annual comprehensive review")
    print()
    print("4. Beyond Backtesting:")
    print("   • Stress testing (extreme scenarios)")
    print("   • Sensitivity analysis (key parameters)")
    print("   • Benchmarking (vs. peer institutions)")
    print("   • Model risk documentation")
    print()
    print("=" * 80)
    print("Demonstration Complete")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print(f"  • {violations}/{total_days} violations observed ({backtest_result.violation_rate:.1%})")
    print(f"  • Basel Zone: {zone_classification}")
    print("  • Backtesting validates model accuracy")
    print("  • Statistical tests ensure proper calibration")
    print("  • Regular validation is regulatory requirement")
    print()


if __name__ == "__main__":
    main()
