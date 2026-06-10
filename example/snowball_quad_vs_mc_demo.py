"""
Snowball Quad vs MC sanity-check demo.

This script compares the SnowballQuadEngine against the SnowballMCEngine on
several snowball configurations to provide a quick reasonableness check.

Usage:
    python example/snowball_quad_vs_mc_demo.py
    python example/snowball_quad_vs_mc_demo.py --paths 20000 --grid 801 --method quasi
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path


from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import MCParams, QuadParams
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import MonteCarloMethod, EngineType


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


def run_case(
    label: str,
    snowball,
    pricing_env: PricingEnvironment,
    quad_engine: SnowballQuadEngine,
    mc_engine: SnowballMCEngine,
    ko_freq: str,
    ki_freq: str | None,
) -> None:
    start = time.perf_counter()
    quad_price = quad_engine.price(snowball, pricing_env)
    quad_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    mc_price = mc_engine.price(snowball, pricing_env)
    mc_elapsed = time.perf_counter() - start
    mc_result = mc_engine.get_last_result()

    diff = quad_price - mc_price
    rel = diff / mc_price if mc_price != 0.0 else float("nan")

    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    print(f"Quad Price: {quad_price:,.2f}")
    print(f"MC Price:   {mc_price:,.2f}")
    if mc_result is not None:
        print(f"MC StdErr:  {mc_result.std_error:,.4f}")
        print(f"KO Prob:    {mc_result.ko_probability:.2%}")
        print(f"V0 Prob:    {mc_result.v0_probability:.2%}")
        print(f"V1 Prob:    {mc_result.v1_probability:.2%}")
    ko_count = snowball.num_ko_observations
    if ko_freq:
        print(f"KO Obs:     {ko_freq} ({ko_count})")
    else:
        print(f"KO Obs:     {ko_count}")
    if snowball.has_ki_barrier:
        ki_continuous = (
            snowball.barrier_config.ki_continuous
            or snowball.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        if ki_continuous:
            print("KI Obs:     continuous")
        else:
            ki_count = snowball.num_ki_observations
            if ki_freq:
                print(f"KI Obs:     {ki_freq} ({ki_count})")
            else:
                print(f"KI Obs:     {ki_count}")
    print(f"Quad Time:  {quad_elapsed:.4f}s (grid={quad_engine.params.grid_points})")
    mc_paths = mc_engine.params.num_paths
    mc_rate = mc_paths / mc_elapsed if mc_elapsed > 0.0 else float("inf")
    print(f"MC Time:    {mc_elapsed:.4f}s ({mc_rate:,.0f} paths/s)")
    print(f"Diff:       {diff:,.2f} ({rel:.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity check: SnowballQuadEngine vs SnowballMCEngine."
    )
    parser.add_argument("--paths", type=int, default=20000, help="MC paths")
    parser.add_argument("--grid", type=int, default=801, help="Quad grid points (odd)")
    parser.add_argument(
        "--method",
        type=str,
        default="quasi",
        help="MC method: pseudo, quasi, randomized_quasi",
    )
    args = parser.parse_args()

    quad_params = QuadParams(grid_points=args.grid)
    mc_params = MCParams(num_paths=args.paths, time_steps=252, seed=42)
    mc_method = parse_mc_method(args.method)

    quad_engine = SnowballQuadEngine(params=quad_params)
    mc_engine = SnowballMCEngine(
        params=mc_params, method=EngineType.MONTE_CARLO(mc_method)
    )

    base_env = create_pricing_env()
    quarterly_obs = [i / 4 for i in range(1, 5)]

    cases = [
        (
            "Standard Snowball (continuous KI, monthly KO)",
            create_standard_snowball(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                ko_barrier=103.0,
                ko_rate=0.15,
                ki_barrier=75.0,
                num_observations=12,
                is_reverse=False,
            ),
            base_env,
            "monthly",
            None,
        ),
        (
            "Standard Snowball (discrete KI, quarterly KO)",
            create_standard_snowball(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                ko_barrier=103.0,
                ko_rate=0.15,
                ki_barrier=75.0,
                num_observations=4,
                is_reverse=False,
                ki_continuous=False,
                ki_observation_type=ObservationType.DISCRETE,
                ki_observation_dates=quarterly_obs,
            ),
            base_env,
            "quarterly",
            "quarterly",
        ),
        (
            "Standard Snowball (daily KI, monthly KO)",
            create_standard_snowball(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                ko_barrier=103.0,
                ko_rate=0.15,
                ki_barrier=75.0,
                num_observations=12,
                is_reverse=False,
                ki_continuous=False,
                ki_observation_type=ObservationType.DISCRETE,
                ki_observation_dates=[i / 252 for i in range(1, 253)],
            ),
            base_env,
            "monthly",
            "daily",
        ),
        (
            "Standard Snowball (high vol 35%)",
            create_standard_snowball(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                ko_barrier=103.0,
                ko_rate=0.15,
                ki_barrier=75.0,
                num_observations=12,
                is_reverse=False,
            ),
            create_pricing_env(vol=0.35),
            "monthly",
            None,
        ),
        (
            "Standard Snowball (airbag 50% below 80)",
            create_standard_snowball(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                ko_barrier=103.0,
                ko_rate=0.15,
                ki_barrier=75.0,
                num_observations=12,
                is_reverse=False,
                airbag_barrier=80.0,
                airbag_participation_rate=0.5,
                airbag_strike=90.0,
            ),
            base_env,
            "monthly",
            None,
        ),
        (
            "Standard Snowball (call-rebate V0)",
            create_standard_snowball(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                ko_barrier=103.0,
                ko_rate=0.15,
                ki_barrier=75.0,
                num_observations=12,
                is_reverse=False,
                rebate_rate=0.0,
                include_principal=True,
                call_rebate_enabled=True,
                call_strike=90.0,
                call_participation_rate=0.5,
            ),
            base_env,
            "monthly",
            None,
        ),
        (
            "Standard Snowball (disable_ko_after_ki)",
            create_standard_snowball(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                ko_barrier=103.0,
                ko_rate=0.15,
                ki_barrier=85.0,
                num_observations=4,
                is_reverse=False,
                ki_continuous=False,
                ki_observation_type=ObservationType.DISCRETE,
                ki_observation_dates=quarterly_obs,
                disable_ko_after_ki=True,
            ),
            base_env,
            "quarterly",
            "quarterly",
        ),
        (
            "Standard Snowball (spot near KO)",
            create_standard_snowball(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                ko_barrier=103.0,
                ko_rate=0.15,
                ki_barrier=75.0,
                num_observations=12,
                is_reverse=False,
            ),
            create_pricing_env(spot=102.5),
            "monthly",
            None,
        ),
        (
            "Standard Snowball (spot near KI)",
            create_standard_snowball(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                ko_barrier=103.0,
                ko_rate=0.15,
                ki_barrier=75.0,
                num_observations=12,
                is_reverse=False,
            ),
            create_pricing_env(spot=78.0),
            "monthly",
            None,
        ),
        (
            "Reverse Snowball (continuous KI)",
            create_standard_snowball(
                initial_price=100.0,
                strike=100.0,
                maturity=1.0,
                contract_multiplier=10_000.0,
                ko_barrier=97.0,
                ko_rate=0.15,
                ki_barrier=125.0,
                num_observations=12,
                is_reverse=True,
            ),
            base_env,
            "monthly",
            None,
        ),
        (
            "Standard Snowball (2Y maturity, monthly KO)",
            create_standard_snowball(
                initial_price=100.0,
                strike=100.0,
                maturity=2.0,
                contract_multiplier=10_000.0,
                ko_barrier=103.0,
                ko_rate=0.15,
                ki_barrier=75.0,
                num_observations=24,
                is_reverse=False,
            ),
            base_env,
            "monthly",
            None,
        ),
    ]

    for label, snowball, env, ko_freq, ki_freq in cases:
        run_case(label, snowball, env, quad_engine, mc_engine, ko_freq, ki_freq)


if __name__ == "__main__":
    main()
