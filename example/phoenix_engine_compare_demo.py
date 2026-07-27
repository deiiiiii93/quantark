"""
Phoenix Engine Comparison Demo.

This script compares Phoenix pricing engines across several product
configurations and market scenarios.

Usage:
    python example/phoenix_engine_compare_demo.py
    python example/phoenix_engine_compare_demo.py --paths 30000 --quad-grid 601 --pde-grid 240 --method quasi
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path


from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.pde import GridConfig
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import (
    create_reverse_phoenix,
    create_standard_phoenix,
    create_stepdown_phoenix,
)
from quantark.asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod


def parse_mc_method(name: str) -> MonteCarloMethod:
    try:
        return MonteCarloMethod[name.upper()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown MC method '{name}'. Use pseudo, quasi, or randomized_quasi."
        ) from exc


def create_pricing_env(
    spot: float = 100.0,
    vol: float = 0.20,
    rate: float = 0.03,
    div_yield: float = 0.00,
) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
        valuation_date=datetime(2024, 1, 1),
    )


def build_observation_dates(maturity: float, num_observations: int) -> list[float]:
    return [
        (i + 1) / num_observations * maturity for i in range(num_observations)
    ]


def build_observation_schedule(times: list[float], barrier: float) -> ObservationSchedule:
    return ObservationSchedule(
        records=[ObservationRecord(observation_time=t, barrier=barrier) for t in times]
    )


def format_barrier(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        return f"{value[0]:.2f}..{value[-1]:.2f}"
    return f"{value:.2f}"


def describe_ki(phoenix) -> str:
    if not phoenix.has_ki_barrier:
        return "none"
    if (
        phoenix.barrier_config.ki_continuous
        or phoenix.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
    ):
        return "continuous"
    schedule = phoenix.barrier_config.ki_observation_schedule
    if schedule is not None and schedule.records:
        return f"discrete ({len(schedule.records)})"
    if phoenix.barrier_config.ki_observation_dates:
        return f"discrete ({len(phoenix.barrier_config.ki_observation_dates)})"
    return "discrete"


def run_case(
    label: str,
    phoenix,
    pricing_env: PricingEnvironment,
    quad_engine: PhoenixQuadEngine,
    pde_engine: PhoenixPDESolver,
    mc_engine: PhoenixMCEngine,
) -> None:
    maturity = phoenix.get_maturity(pricing_env)
    rate = pricing_env.get_rate(maturity)
    div = pricing_env.get_div_yield(maturity)
    vol = pricing_env.get_vol(phoenix.strike, maturity)

    start = time.perf_counter()
    quad_price = quad_engine.price(phoenix, pricing_env)
    quad_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    pde_price = pde_engine.price(phoenix, pricing_env)
    pde_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    mc_price = mc_engine.price(phoenix, pricing_env)
    mc_elapsed = time.perf_counter() - start
    mc_result = mc_engine.get_last_result()

    diff_quad_mc = quad_price - mc_price
    diff_pde_mc = pde_price - mc_price
    diff_quad_pde = quad_price - pde_price
    rel_quad_mc = diff_quad_mc / mc_price if mc_price != 0.0 else float("nan")
    rel_pde_mc = diff_pde_mc / mc_price if mc_price != 0.0 else float("nan")

    print("\n" + "=" * 76)
    print(label)
    print("=" * 76)
    print(
        f"Spot={pricing_env.spot:.2f} | Vol={vol:.1%} | Rate={rate:.2%} | Div={div:.2%}"
    )
    print(
        "KO Obs: discrete "
        f"({phoenix.num_ko_observations}) | KI Obs: {describe_ki(phoenix)}"
    )
    print(
        "KO Barrier: "
        f"{format_barrier(phoenix.barrier_config.ko_barrier)} | "
        "KI Barrier: "
        f"{format_barrier(phoenix.barrier_config.ki_barrier)} | "
        "Coupon Barrier: "
        f"{format_barrier(phoenix.coupon_config.coupon_barrier)}"
    )
    print(
        "Coupon Pay: "
        f"{phoenix.coupon_config.coupon_pay_type.name} | "
        f"Memory: {phoenix.coupon_config.memory_coupon}"
    )
    print(f"Quad Price: {quad_price:,.2f}")
    print(f"PDE Price:  {pde_price:,.2f}")
    print(f"MC Price:   {mc_price:,.2f}")
    if mc_result is not None:
        print(f"MC StdErr:  {mc_result.std_error:,.4f}")
    print(
        f"Diff Q-M:   {diff_quad_mc:,.2f} ({rel_quad_mc:.2%}) | "
        f"Diff P-M: {diff_pde_mc:,.2f} ({rel_pde_mc:.2%})"
    )
    print(f"Diff Q-P:   {diff_quad_pde:,.2f}")
    print(
        f"Quad Time: {quad_elapsed:.4f}s (grid={quad_engine.params.grid_points}) | "
        f"PDE Time: {pde_elapsed:.4f}s "
        f"(grid={pde_engine.grid_binder.config.points}) | "
        f"MC Time: {mc_elapsed:.4f}s ({mc_engine.params.num_paths:,} paths)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Phoenix engines across several cases."
    )
    parser.add_argument("--paths", type=int, default=100000, help="MC paths")
    parser.add_argument("--quad-grid", type=int, default=1001, help="Quad grid points")
    parser.add_argument(
        "--quad-std-devs",
        type=float,
        default=10.0,
        help="Quad log-domain width in std devs",
    )
    parser.add_argument(
        "--pde-grid",
        type=int,
        default=1000,
        help="PDE spatial points (GridConfig.points)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="randomized_quasi",
        help="MC method: pseudo, quasi, randomized_quasi",
    )
    args = parser.parse_args()

    quad_params = QuadParams(
        grid_points=args.quad_grid,
        num_std_devs=args.quad_std_devs,
    )
    pde_params = PDEParams(
        grid=GridConfig(
            points=args.pde_grid,
            max_points=max(2000, args.pde_grid),
        ),
    )
    mc_params = MCParams(num_paths=args.paths, time_steps=252, seed=42)
    mc_method = parse_mc_method(args.method)

    quad_engine = PhoenixQuadEngine(params=quad_params)
    pde_engine = PhoenixPDESolver(params=pde_params)
    mc_engine = PhoenixMCEngine(
        params=mc_params, method=EngineType.MONTE_CARLO(mc_method)
    )

    base_env = create_pricing_env()
    high_vol_env = create_pricing_env(vol=0.35)
    quarterly_obs = build_observation_dates(1.0, 4)
    monthly_obs = build_observation_dates(1.0, 12)
    ki_monthly = build_observation_schedule(monthly_obs, barrier=75.0)
    ki_quarterly = build_observation_schedule(quarterly_obs, barrier=75.0)
    ki_monthly_reverse = build_observation_schedule(monthly_obs, barrier=125.0)

    cases = [
        (
            "Standard Phoenix (NO memory, monthly KO)",
            create_standard_phoenix(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                ko_barrier=103.0,
                ki_barrier=75.0,
                coupon_barrier=95.0,
                ko_rate=0.12,
                coupon_rate=0.02,
                num_observations=12,
                memory_coupon=False,
                ki_continuous=False,
                ki_observation_type=ObservationType.DISCRETE,
                ki_observation_schedule=ki_monthly,
            ),
            base_env,
        ),
        (
            "Standard Phoenix (MEMORY, monthly KO)",
            create_standard_phoenix(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                ko_barrier=103.0,
                ki_barrier=75.0,
                coupon_barrier=95.0,
                ko_rate=0.12,
                coupon_rate=0.02,
                num_observations=12,
                memory_coupon=True,
                ki_continuous=False,
                ki_observation_type=ObservationType.DISCRETE,
                ki_observation_schedule=ki_monthly,
            ),
            base_env,
        ),
        (
            "Standard Phoenix (MEMORY, quarterly KO)",
            create_standard_phoenix(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                ko_barrier=103.0,
                ki_barrier=75.0,
                coupon_barrier=95.0,
                ko_rate=0.12,
                coupon_rate=0.02,
                num_observations=4,
                memory_coupon=True,
                ki_continuous=False,
                ki_observation_type=ObservationType.DISCRETE,
                ki_observation_schedule=ki_quarterly,
            ),
            base_env,
        ),
        (
            "Standard Phoenix (NO memory, quarterly KO)",
            create_standard_phoenix(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                ko_barrier=103.0,
                ki_barrier=75.0,
                coupon_barrier=95.0,
                ko_rate=0.12,
                coupon_rate=0.02,
                num_observations=4,
                memory_coupon=False,
                ki_continuous=False,
                ki_observation_type=ObservationType.DISCRETE,
                ki_observation_schedule=ki_quarterly,
            ),
            base_env,
        ),
        (
            "Reverse Phoenix (monthly KO)",
            create_reverse_phoenix(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                ko_barrier=97.0,
                ki_barrier=125.0,
                coupon_barrier=115.0,
                ko_rate=0.12,
                coupon_rate=0.02,
                num_observations=12,
                ki_continuous=False,
                ki_observation_type=ObservationType.DISCRETE,
                ki_observation_schedule=ki_monthly_reverse,
            ),
            base_env,
        ),
        (
            "Reverse Phoenix (MEMORY, monthly KO)",
            create_reverse_phoenix(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                ko_barrier=97.0,
                ki_barrier=125.0,
                coupon_barrier=115.0,
                ko_rate=0.12,
                coupon_rate=0.02,
                num_observations=12,
                memory_coupon=True,
                ki_continuous=False,
                ki_observation_type=ObservationType.DISCRETE,
                ki_observation_schedule=ki_monthly_reverse,
            ),
            base_env,
        ),
        (
            "Step-down Phoenix (monthly KO)",
            create_stepdown_phoenix(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                initial_ko_barrier=103.0,
                initial_coupon_barrier=95.0,
                ko_stepdown_rate=0.01,
                coupon_stepdown_rate=0.01,
                ko_rate=0.12,
                ki_barrier=75.0,
                coupon_rate=0.02,
                num_observations=12,
                is_reverse=False,
                ki_continuous=False,
                ki_observation_type=ObservationType.DISCRETE,
                ki_observation_schedule=ki_monthly,
            ),
            base_env,
        ),
        (
            "Step-down Phoenix (MEMORY, monthly KO)",
            create_stepdown_phoenix(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                initial_ko_barrier=103.0,
                initial_coupon_barrier=95.0,
                ko_stepdown_rate=0.01,
                coupon_stepdown_rate=0.01,
                ko_rate=0.12,
                ki_barrier=75.0,
                coupon_rate=0.02,
                num_observations=12,
                memory_coupon=True,
                is_reverse=False,
                ki_continuous=False,
                ki_observation_type=ObservationType.DISCRETE,
                ki_observation_schedule=ki_monthly,
            ),
            base_env,
        ),
        (
            "Standard Phoenix (high vol 35%)",
            create_standard_phoenix(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                ko_barrier=103.0,
                ki_barrier=75.0,
                coupon_barrier=95.0,
                ko_rate=0.12,
                coupon_rate=0.02,
                num_observations=12,
                is_reverse=False,
                ki_continuous=False,
                ki_observation_type=ObservationType.DISCRETE,
                ki_observation_schedule=ki_monthly,
            ),
            high_vol_env,
        ),
        (
            "Standard Phoenix (MEMORY, high vol 35%)",
            create_standard_phoenix(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                ko_barrier=103.0,
                ki_barrier=75.0,
                coupon_barrier=95.0,
                ko_rate=0.12,
                coupon_rate=0.02,
                num_observations=12,
                memory_coupon=True,
                is_reverse=False,
                ki_continuous=False,
                ki_observation_type=ObservationType.DISCRETE,
                ki_observation_schedule=ki_monthly,
            ),
            high_vol_env,
        ),
    ]

    for label, phoenix, env in cases:
        run_case(label, phoenix, env, quad_engine, pde_engine, mc_engine)


if __name__ == "__main__":
    main()
