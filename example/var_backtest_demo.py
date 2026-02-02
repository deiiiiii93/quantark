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

    # Add some crisis periods with milder, economically realistic shocks
    # Reduced from original -25%/-20% to -8%/-6% to avoid arbitrage violations
    crisis_periods = [
        (100, 115, -0.08, 0.30),  # Crisis 1: -8% spot, +30% vol
        (200, 215, -0.06, 0.25),  # Crisis 2: -6% spot, +25% vol
    ]

    for start, end, shock, vol_up in crisis_periods:
        spot_returns[start:end] += np.random.normal(shock, 0.02, end-start)
        spot_returns[start:end] *= (1 + vol_up)

    vol_changes = np.random.normal(0.0, 0.012, num_days)
    rate_shifts = np.random.normal(0.0, 0.0006, num_days)

    # Add spot-vol correlation: vol increases when spot decreases (leverage effect)
    # This makes scenarios more economically realistic
    for i in range(num_days):
        if spot_returns[i] < -0.02:  # Down days
            vol_changes[i] += abs(spot_returns[i]) * 0.5  # Vol increases
        elif spot_returns[i] > 0.02:  # Up days
            vol_changes[i] -= spot_returns[i] * 0.2  # Vol decreases

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
        lookback_days=30,  # Reduced to match backtester's minimum (30 days)
        var_method=VaRMethod.HISTORICAL,
        calculate_component_var=False,
        calculate_marginal_var=False,
        calculate_factor_var=False,
    )

    engine = HistoricalVaREngine(config=var_config)

    # Calculate VaR for each day (rolling window)
    pnl_series = []
    var_series = []

    # Use 30-day rolling window for backtesting
    window_size = 30
    start_date = historical_data.index[window_size]

    base_value = portfolio.get_portfolio_value()
    for i in range(window_size, len(historical_data)):
        # Get historical data up to current date
        hist_data_window = historical_data.iloc[:i]

        # Calculate VaR
        result = engine.calculate_var(portfolio, hist_data_window)
        var_series.append(result.var)

        # Get portfolio P&L for current scenario
        scenario = historical_data.iloc[i]
        # Reuse engine scenario revaluation for consistent shocks.
        stressed_value = engine._revalue_portfolio_under_scenario(portfolio, scenario)
        pnl = stressed_value - base_value

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
    portfolio_pnl, var_values = generate_portfolio_pnl_timeseries(
        portfolio, historical_data
    )
    # Keep only dates with realized P&L for backtesting.
    backtest_data = historical_data.join(portfolio_pnl, how="inner")
    backtest_data = backtest_data.rename(columns={"portfolio_pnl": "pnl"})
    backtest_data = backtest_data.dropna(subset=["pnl"])

    print()
    print(f"Generated {len(portfolio_pnl)} days of data")
    print(f"Date range: {portfolio_pnl.index[0].date()} to {portfolio_pnl.index[-1].date()}")
    print()

    print("=" * 80)
    print("Backtesting Configuration")
    print("=" * 80)

    confidence_level = 0.99
    expected_violation_rate = 1 - confidence_level
    print(f"Confidence Level: {confidence_level:.1%}")
    print(f"Expected Violation Rate: {expected_violation_rate:.2%}")
    print(f"Backtest Input Rows: {len(backtest_data)}")
    print()

    print("=" * 80)
    print("Running VaR Backtester")
    print("=" * 80)

    # Create VaR engine for backtesting
    var_config = VaRConfig(
        confidence_level=confidence_level,
        holding_period=1,
        lookback_days=30,  # Match backtester's minimum (30 days)
        var_method=VaRMethod.HISTORICAL,
        calculate_component_var=False,
        calculate_marginal_var=False,
        calculate_factor_var=False,
    )
    var_engine = HistoricalVaREngine(config=var_config)

    # Run backtester with new API
    backtester = VaRBacktester(confidence_level=confidence_level)
    backtest_result = backtester.run_backtest(portfolio, backtest_data, var_engine)

    print()
    print("=" * 80)
    print("Backtest Results Summary")
    print("=" * 80)
    print()

    total_days = backtest_result.num_observations
    expected_violations = total_days * expected_violation_rate

    # Summary statistics
    print(f"Total Days: {total_days}")
    print(f"Number of Violations: {backtest_result.num_exceptions}")
    print(f"Violation Rate: {backtest_result.exception_rate:.2%}")
    print(f"Expected Rate: {expected_violation_rate:.2%}")
    print(f"Expected Violations: {expected_violations:.1f}")

    if backtest_result.num_exceptions > 0:
        losses = [d['actual_pnl'] for d in backtest_result.exception_details]
        print(f"Average Loss on Violation Days: ${sum(losses)/len(losses):,.2f}")
        print(f"Maximum Single-Day Loss: ${min(losses):,.2f}")
    else:
        print("No violations observed (model was conservative)")

    print()

    # Basel Traffic Light Classification
    print("=" * 80)
    print("Basel Traffic Light Regime")
    print("=" * 80)
    print()

    violations = backtest_result.num_exceptions
    zone_classification = backtest_result.basel_zone
    zone_label = {
        "green": "Zone 1 (Green)",
        "yellow": "Zone 2 (Yellow)",
        "red": "Zone 3 (Red)",
    }.get(zone_classification, zone_classification)
    basel_thresholds_apply = confidence_level == 0.99 and total_days >= 250

    print(f"Violations (out of {total_days} days): {violations}")
    print(f"Basel Zone: {zone_label}")

    if zone_classification == "green":
        print()
        print("✓ GREEN ZONE - Acceptable Model")
        if basel_thresholds_apply:
            print("  • Violations ≤ 4 (Basel 250-day rule)")
        else:
            print(f"  • Violations within ~1.5× expected ({expected_violations:.1f})")
        print("  • Model is considered accurate")
        print("  • No supervisory action required")
        print("  • Continue current model")
    elif zone_classification == "yellow":
        print()
        print("⚠ YELLOW ZONE - Acceptable with Response")
        if basel_thresholds_apply:
            print("  • Violations: 5-9 (Basel 250-day rule)")
        else:
            print(f"  • Violations within ~2.5× expected ({expected_violations:.1f})")
        print("  • Model needs review")
        print("  • Supervisory response required")
        print("  • Consider model improvements")
    else:
        print()
        print("✗ RED ZONE - Unacceptable Model")
        if basel_thresholds_apply:
            print("  • Violations ≥ 10 (Basel 250-day rule)")
        else:
            print(f"  • Violations exceed ~2.5× expected ({expected_violations:.1f})")
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
    print(f"   Test Statistic (LR_POF): {backtest_result.kupiec_pof_statistic:.4f}")
    print(f"   p-value: {backtest_result.kupiec_pof_pvalue:.4f}")
    print(f"   Critical Value (99%): {11.34}")  # Chi-square(1, 0.99)

    if backtest_result.kupiec_pof_pvalue > 0.01:
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
    print(f"   Test Statistic (LR_ind): {backtest_result.christoffersen_statistic:.4f}")
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

    # Combined test is not directly provided by VaRBacktestResult
    # For simplicity, we check both tests pass
    combined_pass = backtest_result.kupiec_pof_pass and backtest_result.christoffersen_pass
    print(f"Kupiec POF Test: {'PASS' if backtest_result.kupiec_pof_pass else 'FAIL'}")
    print(f"Christoffersen Test: {'PASS' if backtest_result.christoffersen_pass else 'FAIL'}")

    if combined_pass:
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

    if violations <= 5 and backtest_result.kupiec_pof_pvalue > 0.01:
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
    print(f"  • {violations}/{total_days} violations observed ({backtest_result.exception_rate:.1%})")
    print(f"  • Basel Zone: {zone_classification}")
    print("  • Backtesting validates model accuracy")
    print("  • Statistical tests ensure proper calibration")
    print("  • Regular validation is regulatory requirement")
    print()


if __name__ == "__main__":
    main()
