"""
Benchmark Check Script for Barrier Analytical Engine
Benchmark: Barrier Option Monte Carlo Engine
Generated: 2025-12-25
Default Tolerance: 5%
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent.parent))

from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine  # noqa: E402
from quantark.asset.equity.engine.mc import BarrierOptionMCEngine  # noqa: E402
from quantark.asset.equity.param import MCParams  # noqa: E402
from quantark.asset.equity.product.option import (  # noqa: E402
    BarrierOption,
    ObservationSchedule,
)
from quantark.param import (  # noqa: E402
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment  # noqa: E402
from quantark.util.enum import (  # noqa: E402
    BarrierType,
    ObservationFrequency,
    ObservationType,
    OptionType,
)

TOLERANCE = 0.05
MC_PATHS = 50000
MC_STEPS = 252


class BenchmarkResults:
    def __init__(self, tolerance: float = TOLERANCE) -> None:
        self.tolerance = tolerance
        self.results = []

    def add_result(
        self, case_name: str, analytical: float, mc: float, mc_se: float = 0.0
    ) -> None:
        if mc != 0:
            rel_error = abs(analytical - mc) / abs(mc)
        else:
            rel_error = abs(analytical - mc)
        passed = rel_error <= self.tolerance
        self.results.append(
            {
                "case": case_name,
                "analytical": analytical,
                "mc": mc,
                "mc_se": mc_se,
                "rel_error": rel_error,
                "passed": passed,
            }
        )

    def summary(self) -> bool:
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)

        print(f"\n{'='*90}")
        print(f"BENCHMARK CHECK SUMMARY (Tolerance: {self.tolerance*100:.1f}%)")
        print(f"{'='*90}")
        print(
            f"{'Case':<40} {'Analytical':>12} {'MC':>12} "
            f"{'SE':>10} {'Error':>10} {'Status':>8}"
        )
        print(f"{'-'*90}")

        for r in self.results:
            status = "PASS" if r["passed"] else "FAIL"
            print(
                f"{r['case']:<40} {r['analytical']:>12.4f} {r['mc']:>12.4f} "
                f"{r['mc_se']:>10.4f} {r['rel_error']*100:>9.2f}% {status:>8}"
            )

        print(f"{'-'*90}")
        print(
            f"Passed: {passed}/{total} "
            f"({100*passed/total if total > 0 else 0:.1f}%)"
        )

        return passed == total


def create_pricing_env(spot=100.0, rate=0.03, vol=0.25, div=0.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def create_barrier_option(
    strike: float,
    maturity: float,
    barrier: float,
    option_type: OptionType,
    barrier_type: BarrierType,
    observation_type: ObservationType = ObservationType.CONTINUOUS,
    observation_dates: Optional[List[float]] = None,
    observation_schedule: Optional[ObservationSchedule] = None,
    rebate: float = 0.0,
    pay_at_hit: bool = False,
) -> BarrierOption:
    option = BarrierOption(
        strike=strike,
        option_type=option_type,
        barrier=barrier,
        barrier_type=barrier_type,
        maturity=maturity,
        rebate=rebate,
        pay_at_hit=pay_at_hit,
        observation_type=observation_type,
        observation_dates=observation_dates,
        observation_schedule=observation_schedule,
    )
    option.validate()
    return option


def run_benchmark_tests(results: BenchmarkResults) -> None:
    """Run benchmark test cases against Monte Carlo engine."""
    pricing_env = create_pricing_env()
    analytical_engine = BarrierAnalyticalEngine()
    mc_params = MCParams(num_paths=MC_PATHS, time_steps=MC_STEPS, seed=42)
    mc_engine = BarrierOptionMCEngine(
        params=mc_params, use_brownian_bridge=True
    )
    daily_observation_dates = [i / 252 for i in range(1, 253)]
    daily_observation_schedule = ObservationSchedule.from_legacy(
        observation_dates=daily_observation_dates,
        default_barrier=120.0,
        default_payoff=0.0,
        frequency=ObservationFrequency.DAILY,
    )
    daily_down_observation_schedule = ObservationSchedule.from_legacy(
        observation_dates=daily_observation_dates,
        default_barrier=80.0,
        default_payoff=0.0,
        frequency=ObservationFrequency.DAILY,
    )
    monthly_observation_dates = [i / 12 for i in range(1, 13)]
    monthly_observation_schedule = ObservationSchedule.from_legacy(
        observation_dates=monthly_observation_dates,
        default_barrier=120.0,
        default_payoff=0.0,
        frequency=ObservationFrequency.MONTHLY,
    )
    monthly_down_observation_schedule = ObservationSchedule.from_legacy(
        observation_dates=monthly_observation_dates,
        default_barrier=80.0,
        default_payoff=0.0,
        frequency=ObservationFrequency.MONTHLY,
    )

    cases = [
        (
            "Up-and-out Call (cont)",
            create_barrier_option(
                strike=100.0,
                maturity=1.0,
                barrier=120.0,
                option_type=OptionType.CALL,
                barrier_type=BarrierType.UP_OUT,
            ),
        ),
        (
            "Up-and-in Call (cont)",
            create_barrier_option(
                strike=100.0,
                maturity=1.0,
                barrier=120.0,
                option_type=OptionType.CALL,
                barrier_type=BarrierType.UP_IN,
            ),
        ),
        (
            "Down-and-out Put (cont)",
            create_barrier_option(
                strike=100.0,
                maturity=1.0,
                barrier=80.0,
                option_type=OptionType.PUT,
                barrier_type=BarrierType.DOWN_OUT,
            ),
        ),
        (
            "Up-and-out Call (expiry)",
            create_barrier_option(
                strike=100.0,
                maturity=1.0,
                barrier=110.0,
                option_type=OptionType.CALL,
                barrier_type=BarrierType.UP_OUT,
                observation_type=ObservationType.EXPIRY,
            ),
        ),
        (
            "Up-and-out Call (disc, daily)",
            create_barrier_option(
                strike=100.0,
                maturity=1.0,
                barrier=120.0,
                option_type=OptionType.CALL,
                barrier_type=BarrierType.UP_OUT,
                observation_type=ObservationType.DISCRETE,
                observation_schedule=daily_observation_schedule,
            ),
        ),
        (
            "Down-and-out Put (disc, daily)",
            create_barrier_option(
                strike=100.0,
                maturity=1.0,
                barrier=80.0,
                option_type=OptionType.PUT,
                barrier_type=BarrierType.DOWN_OUT,
                observation_type=ObservationType.DISCRETE,
                observation_schedule=daily_down_observation_schedule,
            ),
        ),
        (
            "Up-and-out Call (disc, monthly)",
            create_barrier_option(
                strike=100.0,
                maturity=1.0,
                barrier=120.0,
                option_type=OptionType.CALL,
                barrier_type=BarrierType.UP_OUT,
                observation_type=ObservationType.DISCRETE,
                observation_schedule=monthly_observation_schedule,
            ),
        ),
        (
            "Down-and-out Put (disc, monthly)",
            create_barrier_option(
                strike=100.0,
                maturity=1.0,
                barrier=80.0,
                option_type=OptionType.PUT,
                barrier_type=BarrierType.DOWN_OUT,
                observation_type=ObservationType.DISCRETE,
                observation_schedule=monthly_down_observation_schedule,
            ),
        ),
    ]

    for case_name, option in cases:
        analytical_price = analytical_engine.price(option, pricing_env)
        mc_price = mc_engine.price(option, pricing_env)
        mc_se = mc_engine.get_last_std_error()
        results.add_result(case_name, analytical_price, mc_price, mc_se)


if __name__ == "__main__":
    results = BenchmarkResults(tolerance=TOLERANCE)
    run_benchmark_tests(results)
    success = results.summary()
    sys.exit(0 if success else 1)
