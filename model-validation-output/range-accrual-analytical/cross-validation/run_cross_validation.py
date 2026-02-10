"""
Cross-validation of Range Accrual Analytical Engine vs Monte Carlo Engine

Tests the analytical pricing engine against the quasi-Monte Carlo engine
across 8 test cases covering various market conditions and product features.
"""

import numpy as np
from datetime import datetime, timedelta

from asset.equity.engine.analytical import RangeAccrualAnalyticalEngine
from asset.equity.engine.mc import RangeAccrualMCEngine
from asset.equity.product.option import RangeAccrualOption, RangeAccrualConfig, RangeAccrualObservationRecord
from asset.equity.param import MCParams
from priceenv import PricingEnvironment
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from util.enum.engine_enums import MonteCarloMethod


def create_price_env(spot: float, vol: float, rate: float, div: float = 0.0):
    """Create a pricing environment with given parameters."""
    val_date = datetime(2024, 1, 1)
    return PricingEnvironment(
        rate_curve=FlatRateCurve(rate=rate),
        valuation_date=val_date,
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        div_yield=ContinuousDividendYield(div_yield=div)
    )


def create_option(
    spot: float,
    lower_barrier: float,
    upper_barrier: float,
    num_obs: int,
    accrual_rate: float,
    contract_multiplier: float,
    is_reverse: bool = False,
    is_rate_annualized: bool = True,
    lower_barriers_list=None,
    upper_barriers_list=None,
    weights=None
):
    """Create a range accrual option with specified parameters."""
    val_date = datetime(2024, 1, 1)
    maturity_date = datetime(2025, 1, 1)
    maturity_years = 1.0

    # Create observation records
    obs_records = []
    time_delta = (maturity_date - val_date) / num_obs

    for i in range(num_obs):
        obs_date = val_date + (i + 1) * time_delta

        # Determine barriers (per-observation if provided)
        lower_per_obs = lower_barriers_list[i] if lower_barriers_list is not None else None
        upper_per_obs = upper_barriers_list[i] if upper_barriers_list is not None else None

        # Determine weight
        weight = weights[i] if weights is not None else 1.0

        obs_records.append(
            RangeAccrualObservationRecord(
                observation_date=obs_date,
                lower_barrier=lower_per_obs,
                upper_barrier=upper_per_obs,
                weight=weight
            )
        )

    # Create config with default barriers
    config = RangeAccrualConfig(
        upper_barrier=upper_barrier,
        lower_barrier=lower_barrier,
        accrual_rate=accrual_rate,
        is_reverse=is_reverse,
        is_rate_annualized=is_rate_annualized
    )

    return RangeAccrualOption(
        initial_price=spot,
        range_config=config,
        maturity=maturity_years,
        observation_records=obs_records,
        contract_multiplier=contract_multiplier
    )


def run_test_case(
    case_name: str,
    option: RangeAccrualOption,
    price_env: PricingEnvironment,
    num_paths: int = 500_000
):
    """Run a single test case comparing analytical and MC pricing."""

    # Analytical pricing
    analytical_engine = RangeAccrualAnalyticalEngine()
    analytical_price = analytical_engine.price(option, price_env)

    # Monte Carlo pricing with QMC
    mc_params = MCParams(
        num_paths=num_paths,
        seed=42,
        use_antithetic=False
    )
    mc_engine = RangeAccrualMCEngine(
        params=mc_params,
        method=MonteCarloMethod.QUASI
    )
    mc_price = mc_engine.price(option, price_env)
    mc_stderr = mc_engine.get_last_std_error()

    # Compute differences
    abs_diff = abs(analytical_price - mc_price)
    rel_diff = abs_diff / mc_price if mc_price != 0 else 0.0
    rel_diff_pct = rel_diff * 100

    return {
        "case_name": case_name,
        "analytical_price": analytical_price,
        "mc_price": mc_price,
        "mc_stderr": mc_stderr,
        "abs_diff": abs_diff,
        "rel_diff_pct": rel_diff_pct
    }


def main():
    """Run all test cases and generate report."""

    print("=" * 100)
    print("Range Accrual Analytical Engine vs Monte Carlo Engine Cross-Validation")
    print("=" * 100)
    print()

    results = []

    # Case 1: Standard
    print("Running Case 1: Standard...")
    option1 = create_option(
        spot=100.0,
        lower_barrier=90.0,
        upper_barrier=110.0,
        num_obs=12,
        accrual_rate=0.05,
        contract_multiplier=10000.0,
        is_rate_annualized=True
    )
    price_env1 = create_price_env(spot=100.0, vol=0.2, rate=0.05)
    results.append(run_test_case("Standard", option1, price_env1))

    # Case 2: Low vol
    print("Running Case 2: Low vol...")
    option2 = create_option(
        spot=100.0,
        lower_barrier=90.0,
        upper_barrier=110.0,
        num_obs=12,
        accrual_rate=0.05,
        contract_multiplier=10000.0,
        is_rate_annualized=True
    )
    price_env2 = create_price_env(spot=100.0, vol=0.1, rate=0.05)
    results.append(run_test_case("Low vol", option2, price_env2))

    # Case 3: High vol
    print("Running Case 3: High vol...")
    option3 = create_option(
        spot=100.0,
        lower_barrier=90.0,
        upper_barrier=110.0,
        num_obs=12,
        accrual_rate=0.05,
        contract_multiplier=10000.0,
        is_rate_annualized=True
    )
    price_env3 = create_price_env(spot=100.0, vol=0.4, rate=0.05)
    results.append(run_test_case("High vol", option3, price_env3))

    # Case 4: Narrow range
    print("Running Case 4: Narrow range...")
    option4 = create_option(
        spot=100.0,
        lower_barrier=95.0,
        upper_barrier=105.0,
        num_obs=12,
        accrual_rate=0.08,
        contract_multiplier=10000.0,
        is_rate_annualized=False
    )
    price_env4 = create_price_env(spot=100.0, vol=0.2, rate=0.05)
    results.append(run_test_case("Narrow range", option4, price_env4))

    # Case 5: Wide range
    print("Running Case 5: Wide range...")
    option5 = create_option(
        spot=100.0,
        lower_barrier=70.0,
        upper_barrier=130.0,
        num_obs=12,
        accrual_rate=0.03,
        contract_multiplier=10000.0,
        is_rate_annualized=False
    )
    price_env5 = create_price_env(spot=100.0, vol=0.2, rate=0.05)
    results.append(run_test_case("Wide range", option5, price_env5))

    # Case 6: Reverse mode
    print("Running Case 6: Reverse mode...")
    option6 = create_option(
        spot=100.0,
        lower_barrier=90.0,
        upper_barrier=110.0,
        num_obs=12,
        accrual_rate=0.05,
        contract_multiplier=10000.0,
        is_reverse=True,
        is_rate_annualized=True
    )
    price_env6 = create_price_env(spot=100.0, vol=0.2, rate=0.05)
    results.append(run_test_case("Reverse mode", option6, price_env6))

    # Case 7: Step-down barriers
    print("Running Case 7: Step-down barriers...")
    option7 = create_option(
        spot=100.0,
        lower_barrier=[85.0, 88.0, 90.0, 92.0],
        upper_barrier=[115.0, 112.0, 110.0, 108.0],
        num_obs=4,
        accrual_rate=0.05,
        contract_multiplier=10000.0,
        is_rate_annualized=True,
        lower_barriers_list=[85.0, 88.0, 90.0, 92.0],
        upper_barriers_list=[115.0, 112.0, 110.0, 108.0]
    )
    price_env7 = create_price_env(spot=100.0, vol=0.2, rate=0.05)
    results.append(run_test_case("Step-down barriers", option7, price_env7))

    # Case 8: Weighted observations
    print("Running Case 8: Weighted observations...")
    option8 = create_option(
        spot=100.0,
        lower_barrier=90.0,
        upper_barrier=110.0,
        num_obs=5,
        accrual_rate=0.05,
        contract_multiplier=10000.0,
        is_rate_annualized=True,
        weights=[1.0, 3.0, 1.0, 3.0, 1.0]
    )
    price_env8 = create_price_env(spot=100.0, vol=0.2, rate=0.05)
    results.append(run_test_case("Weighted obs", option8, price_env8))

    print()
    print("=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print()

    # Print table header
    print(f"{'Case':<20} {'Analytical':>12} {'MC Price':>12} {'MC StdErr':>12} {'Abs Diff':>12} {'Rel Diff %':>12} {'Status':>8}")
    print("-" * 100)

    # Print results
    all_pass = True
    for r in results:
        status = "PASS" if r["rel_diff_pct"] < 1.0 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"{r['case_name']:<20} {r['analytical_price']:>12.2f} {r['mc_price']:>12.2f} "
              f"{r['mc_stderr']:>12.2f} {r['abs_diff']:>12.2f} {r['rel_diff_pct']:>12.4f} {status:>8}")

    print("-" * 100)
    print()

    # Overall status
    if all_pass:
        print("OVERALL STATUS: PASS (All relative differences < 1%)")
    else:
        print("OVERALL STATUS: FAIL (Some relative differences >= 1%)")

    print()

    # Generate markdown report
    report_path = "/Users/fuxinyao/quant-ark/model-validation-output/range-accrual-analytical/cross-validation/mc-comparison-report.md"

    with open(report_path, "w") as f:
        f.write("# Range Accrual Analytical Engine vs Monte Carlo Cross-Validation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Executive Summary\n\n")

        if all_pass:
            f.write("**Status:** PASS ✓\n\n")
            f.write("All test cases show relative differences < 1% between the analytical and Monte Carlo engines.\n\n")
        else:
            f.write("**Status:** FAIL ✗\n\n")
            f.write("Some test cases show relative differences >= 1% between the analytical and Monte Carlo engines.\n\n")

        f.write("## Test Configuration\n\n")
        f.write("- **Monte Carlo Method:** Quasi-Monte Carlo (QMC)\n")
        f.write("- **Number of Paths:** 500,000\n")
        f.write("- **Seed:** 42\n")
        f.write("- **Antithetic Variates:** Disabled\n\n")

        f.write("## Test Cases\n\n")

        for i, r in enumerate(results, 1):
            f.write(f"### Case {i}: {r['case_name']}\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Analytical Price | ${r['analytical_price']:,.2f} |\n")
            f.write(f"| Monte Carlo Price | ${r['mc_price']:,.2f} |\n")
            f.write(f"| MC Standard Error | ${r['mc_stderr']:,.2f} |\n")
            f.write(f"| Absolute Difference | ${r['abs_diff']:,.2f} |\n")
            f.write(f"| Relative Difference | {r['rel_diff_pct']:.4f}% |\n")

            status = "PASS ✓" if r["rel_diff_pct"] < 1.0 else "FAIL ✗"
            f.write(f"| **Status** | **{status}** |\n\n")

        f.write("## Results Summary Table\n\n")
        f.write("| Case | Analytical | MC Price | MC StdErr | Abs Diff | Rel Diff % | Status |\n")
        f.write("|------|------------|----------|-----------|----------|------------|--------|\n")

        for r in results:
            status = "PASS ✓" if r["rel_diff_pct"] < 1.0 else "FAIL ✗"
            f.write(f"| {r['case_name']} | ${r['analytical_price']:,.2f} | ${r['mc_price']:,.2f} | "
                   f"${r['mc_stderr']:,.2f} | ${r['abs_diff']:,.2f} | {r['rel_diff_pct']:.4f}% | {status} |\n")

        f.write("\n## Analysis\n\n")
        f.write("The cross-validation compares the newly implemented Range Accrual Analytical Engine ")
        f.write("against the established Monte Carlo (QMC) engine across 8 diverse test cases:\n\n")
        f.write("1. **Standard**: Baseline case with typical parameters\n")
        f.write("2. **Low vol**: Tests behavior in low volatility regime (σ=0.10)\n")
        f.write("3. **High vol**: Tests behavior in high volatility regime (σ=0.40)\n")
        f.write("4. **Narrow range**: Tighter barriers [95,105] with non-annualized rate\n")
        f.write("5. **Wide range**: Wider barriers [70,130] with non-annualized rate\n")
        f.write("6. **Reverse mode**: Accrues when outside the range\n")
        f.write("7. **Step-down barriers**: Time-varying barriers that tighten over time\n")
        f.write("8. **Weighted observations**: Non-uniform observation weights\n\n")

        f.write("### Key Observations\n\n")

        max_rel_diff = max(r["rel_diff_pct"] for r in results)
        min_rel_diff = min(r["rel_diff_pct"] for r in results)
        avg_rel_diff = sum(r["rel_diff_pct"] for r in results) / len(results)

        f.write(f"- **Maximum Relative Difference:** {max_rel_diff:.4f}%\n")
        f.write(f"- **Minimum Relative Difference:** {min_rel_diff:.4f}%\n")
        f.write(f"- **Average Relative Difference:** {avg_rel_diff:.4f}%\n\n")

        if all_pass:
            f.write("All test cases demonstrate excellent agreement between the analytical and MC methods, ")
            f.write("with relative differences well below the 1% threshold. This validates the correctness ")
            f.write("of the analytical implementation.\n\n")
        else:
            f.write("Some test cases show relative differences exceeding the 1% threshold. ")
            f.write("Further investigation may be needed to understand the source of these discrepancies.\n\n")

        f.write("### Computational Efficiency\n\n")
        f.write("The analytical engine provides instant pricing without statistical noise, ")
        f.write("while the Monte Carlo engine requires 500,000 paths to achieve comparable accuracy. ")
        f.write("This represents a significant computational advantage for the analytical method.\n\n")

        f.write("## Conclusion\n\n")

        if all_pass:
            f.write("The Range Accrual Analytical Engine successfully passes cross-validation against ")
            f.write("the Monte Carlo engine. The implementation is validated for production use.\n")
        else:
            f.write("The Range Accrual Analytical Engine shows some discrepancies with the Monte Carlo ")
            f.write("engine that require further investigation before production deployment.\n")

    print(f"Report written to: {report_path}")
    print()


if __name__ == "__main__":
    main()
