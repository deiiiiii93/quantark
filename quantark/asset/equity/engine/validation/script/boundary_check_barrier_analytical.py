"""
Boundary Check Script for Barrier Analytical Engine
Generated: 2025-12-25
"""
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

from asset.equity.engine.analytical import (  # noqa: E402
    BarrierAnalyticalEngine,
    BlackScholesEngine,
)
from asset.equity.product.option import (  # noqa: E402
    BarrierOption,
    EuropeanVanillaOption,
    ObservationSchedule,
)
from param import (  # noqa: E402
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from priceenv import PricingEnvironment  # noqa: E402
from util.enum import (  # noqa: E402
    BarrierType,
    ObservationFrequency,
    ObservationType,
    OptionType,
)
from util.numerical import Tolerance, is_close  # noqa: E402


class BoundaryCheckResults:
    def __init__(self) -> None:
        self.passed = []
        self.failed = []
        self.warnings = []

    def add_result(self, test_name: str, passed: bool, message: str) -> None:
        if passed:
            self.passed.append((test_name, message))
        else:
            self.failed.append((test_name, message))

    def add_warning(self, test_name: str, message: str) -> None:
        self.warnings.append((test_name, message))

    def summary(self) -> bool:
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*60}")
        print("BOUNDARY CHECK SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {total}")
        print(
            f"Passed: {len(self.passed)} "
            f"({100*len(self.passed)/total if total > 0 else 0:.1f}%)"
        )
        print(
            f"Failed: {len(self.failed)} "
            f"({100*len(self.failed)/total if total > 0 else 0:.1f}%)"
        )
        print(f"Warnings: {len(self.warnings)}")

        if self.failed:
            print("\nFailed Tests:")
            for name, msg in self.failed:
                print(f"  - {name}: {msg}")

        if self.warnings:
            print("\nWarnings:")
            for name, msg in self.warnings:
                print(f"  - {name}: {msg}")

        return len(self.failed) == 0


def create_pricing_env(spot=100.0, rate=0.05, vol=0.20, div=0.0):
    """Helper to create pricing environment."""
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
    participation_rate: float = 1.0,
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
        participation_rate=participation_rate,
    )
    option.validate()
    return option


def test_near_expiry_intrinsic(results: BoundaryCheckResults) -> None:
    """Test: Near expiry -> intrinsic value (barrier far away)."""
    spot = 100.0
    strike = 95.0
    barrier = 50.0
    maturity = 1e-8

    pricing_env = create_pricing_env(spot=spot, rate=0.05, vol=0.2)
    option = create_barrier_option(
        strike=strike,
        maturity=maturity,
        barrier=barrier,
        option_type=OptionType.CALL,
        barrier_type=BarrierType.DOWN_OUT,
    )

    engine = BarrierAnalyticalEngine()
    price = engine.price(option, pricing_env)
    intrinsic = max(spot - strike, 0.0)

    passed = is_close(price, intrinsic, rel_tol=1e-4, abs_tol=1e-2)
    results.add_result(
        "Near expiry (T->0)",
        passed,
        f"Price={price:.6f}, Intrinsic={intrinsic:.6f}",
    )


def test_knock_in_out_parity_continuous(results: BoundaryCheckResults) -> None:
    """Test: KO + KI = Vanilla (continuous, no rebate)."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.03, vol=0.25)
    engine = BarrierAnalyticalEngine()
    bs_engine = BlackScholesEngine()

    strike = 100.0
    barrier = 120.0
    maturity = 1.0

    ko = create_barrier_option(
        strike=strike,
        maturity=maturity,
        barrier=barrier,
        option_type=OptionType.CALL,
        barrier_type=BarrierType.UP_OUT,
    )
    ki = create_barrier_option(
        strike=strike,
        maturity=maturity,
        barrier=barrier,
        option_type=OptionType.CALL,
        barrier_type=BarrierType.UP_IN,
    )
    vanilla = EuropeanVanillaOption(
        strike=strike,
        option_type=OptionType.CALL,
        maturity=maturity,
    )

    ko_price = engine.price(ko, pricing_env)
    ki_price = engine.price(ki, pricing_env)
    vanilla_price = bs_engine.price(vanilla, pricing_env)

    total = ko_price + ki_price
    passed = is_close(
        total,
        vanilla_price,
        rel_tol=1e-3,
        abs_tol=2 * Tolerance.PRECISION,
    )
    results.add_result(
        "KO + KI = Vanilla (continuous)",
        passed,
        f"KO+KI={total:.6f}, Vanilla={vanilla_price:.6f}",
    )


def test_knock_out_le_vanilla(results: BoundaryCheckResults) -> None:
    """Test: KO option value should not exceed vanilla value."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.03, vol=0.25)
    engine = BarrierAnalyticalEngine()
    bs_engine = BlackScholesEngine()

    strike = 100.0
    barrier = 120.0
    maturity = 1.0

    ko = create_barrier_option(
        strike=strike,
        maturity=maturity,
        barrier=barrier,
        option_type=OptionType.CALL,
        barrier_type=BarrierType.UP_OUT,
    )
    vanilla = EuropeanVanillaOption(
        strike=strike,
        option_type=OptionType.CALL,
        maturity=maturity,
    )

    ko_price = engine.price(ko, pricing_env)
    vanilla_price = bs_engine.price(vanilla, pricing_env)

    passed = ko_price <= vanilla_price + Tolerance.PRECISION
    results.add_result(
        "KO <= Vanilla",
        passed,
        f"KO={ko_price:.6f}, Vanilla={vanilla_price:.6f}",
    )


def test_barrier_monotonicity_down_out_call(results: BoundaryCheckResults) -> None:
    """Test: Down-and-out call decreases as barrier moves up."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.04, vol=0.20)
    engine = BarrierAnalyticalEngine()

    strike = 100.0
    maturity = 1.0

    low_barrier = 70.0
    high_barrier = 90.0

    option_low = create_barrier_option(
        strike=strike,
        maturity=maturity,
        barrier=low_barrier,
        option_type=OptionType.CALL,
        barrier_type=BarrierType.DOWN_OUT,
    )
    option_high = create_barrier_option(
        strike=strike,
        maturity=maturity,
        barrier=high_barrier,
        option_type=OptionType.CALL,
        barrier_type=BarrierType.DOWN_OUT,
    )

    price_low = engine.price(option_low, pricing_env)
    price_high = engine.price(option_high, pricing_env)

    passed = price_high <= price_low + Tolerance.PRECISION
    results.add_result(
        "Down-and-out monotonicity (barrier up)",
        passed,
        f"LowH={price_low:.6f}, HighH={price_high:.6f}",
    )


def test_immediate_knock_out_rebate(results: BoundaryCheckResults) -> None:
    """Test: Immediate KO returns rebate (hit vs expiry payment)."""
    pricing_env = create_pricing_env(spot=105.0, rate=0.03, vol=0.2)
    engine = BarrierAnalyticalEngine()

    strike = 100.0
    barrier = 100.0
    maturity = 1.0
    rebate = 5.0

    pay_at_hit = create_barrier_option(
        strike=strike,
        maturity=maturity,
        barrier=barrier,
        option_type=OptionType.CALL,
        barrier_type=BarrierType.UP_OUT,
        rebate=rebate,
        pay_at_hit=True,
    )
    pay_at_expiry = create_barrier_option(
        strike=strike,
        maturity=maturity,
        barrier=barrier,
        option_type=OptionType.CALL,
        barrier_type=BarrierType.UP_OUT,
        rebate=rebate,
        pay_at_hit=False,
    )

    price_hit = engine.price(pay_at_hit, pricing_env)
    price_expiry = engine.price(pay_at_expiry, pricing_env)

    expected_hit = rebate
    expected_expiry = rebate * math.exp(-0.03 * maturity)

    passed_hit = is_close(price_hit, expected_hit, abs_tol=Tolerance.PRECISION)
    passed_expiry = is_close(
        price_expiry,
        expected_expiry,
        rel_tol=1e-6,
        abs_tol=Tolerance.PRECISION,
    )

    results.add_result(
        "Immediate KO rebate (pay at hit)",
        passed_hit,
        f"Price={price_hit:.6f}, Expected={expected_hit:.6f}",
    )
    results.add_result(
        "Immediate KO rebate (pay at expiry)",
        passed_expiry,
        f"Price={price_expiry:.6f}, Expected={expected_expiry:.6f}",
    )


def test_expiry_ko_ki_parity(results: BoundaryCheckResults) -> None:
    """Test: KO + KI = Vanilla for expiry-only monitoring."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.02, vol=0.18)
    engine = BarrierAnalyticalEngine()
    bs_engine = BlackScholesEngine()

    strike = 100.0
    barrier = 110.0
    maturity = 1.0

    ko = create_barrier_option(
        strike=strike,
        maturity=maturity,
        barrier=barrier,
        option_type=OptionType.CALL,
        barrier_type=BarrierType.UP_OUT,
        observation_type=ObservationType.EXPIRY,
    )
    ki = create_barrier_option(
        strike=strike,
        maturity=maturity,
        barrier=barrier,
        option_type=OptionType.CALL,
        barrier_type=BarrierType.UP_IN,
        observation_type=ObservationType.EXPIRY,
    )
    vanilla = EuropeanVanillaOption(
        strike=strike,
        option_type=OptionType.CALL,
        maturity=maturity,
    )

    ko_price = engine.price(ko, pricing_env)
    ki_price = engine.price(ki, pricing_env)
    vanilla_price = bs_engine.price(vanilla, pricing_env)

    total = ko_price + ki_price
    passed = is_close(
        total,
        vanilla_price,
        rel_tol=1e-3,
        abs_tol=2 * Tolerance.PRECISION,
    )
    results.add_result(
        "KO + KI = Vanilla (expiry monitoring)",
        passed,
        f"KO+KI={total:.6f}, Vanilla={vanilla_price:.6f}",
    )


def test_continuous_vs_discrete_knock_out(results: BoundaryCheckResults) -> None:
    """Test: Continuous monitoring knocks out more easily than discrete."""
    pricing_env = create_pricing_env(spot=100.0, rate=0.03, vol=0.25)
    engine = BarrierAnalyticalEngine()

    strike = 100.0
    barrier = 110.0
    maturity = 1.0
    observation_dates = [i / 252 for i in range(1, 253)]
    observation_schedule = ObservationSchedule.from_legacy(
        observation_dates=observation_dates,
        default_barrier=barrier,
        default_payoff=0.0,
        frequency=ObservationFrequency.DAILY,
    )

    continuous = create_barrier_option(
        strike=strike,
        maturity=maturity,
        barrier=barrier,
        option_type=OptionType.CALL,
        barrier_type=BarrierType.UP_OUT,
        observation_type=ObservationType.CONTINUOUS,
        rebate=0.0,
    )
    discrete = create_barrier_option(
        strike=strike,
        maturity=maturity,
        barrier=barrier,
        option_type=OptionType.CALL,
        barrier_type=BarrierType.UP_OUT,
        observation_type=ObservationType.DISCRETE,
        observation_schedule=observation_schedule,
        rebate=0.0,
    )

    price_continuous = engine.price(continuous, pricing_env)
    price_discrete = engine.price(discrete, pricing_env)

    passed = price_continuous <= price_discrete + Tolerance.PRECISION
    results.add_result(
        "Continuous KO <= Discrete KO",
        passed,
        f"Continuous={price_continuous:.6f}, Discrete={price_discrete:.6f}",
    )


if __name__ == "__main__":
    results = BoundaryCheckResults()

    test_near_expiry_intrinsic(results)
    test_knock_in_out_parity_continuous(results)
    test_knock_out_le_vanilla(results)
    test_barrier_monotonicity_down_out_call(results)
    test_immediate_knock_out_rebate(results)
    test_expiry_ko_ki_parity(results)
    test_continuous_vs_discrete_knock_out(results)

    success = results.summary()
    sys.exit(0 if success else 1)
