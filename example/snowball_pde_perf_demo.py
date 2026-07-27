"""
Snowball PDE performance benchmark.

Usage:
    python example/snowball_pde_perf_demo.py
    python example/snowball_pde_perf_demo.py --compare
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path


from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.engine_enums import MonteCarloMethod, EngineType
from quantark.asset.equity.engine.pde import GridConfig


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


def format_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


def format_time(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Snowball PDE performance.")
    parser.add_argument("--compare", action="store_true", help="Compare vs MC/Quad")
    parser.add_argument("--paths", type=int, default=20000, help="MC paths for compare")
    parser.add_argument("--grid", type=int, default=801, help="Quad grid points")
    args = parser.parse_args()

    env = create_pricing_env()
    snowball = create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        ko_barrier=103.0,
        ko_rate=0.15,
        ki_barrier=75.0,
        num_observations=12,
        is_reverse=False,
    )

    quad_price = mc_price = None
    if args.compare:
        quad_engine = SnowballQuadEngine(params=QuadParams(grid_points=args.grid))
        mc_engine = SnowballMCEngine(
            params=MCParams(num_paths=args.paths, time_steps=252, seed=42),
            method=EngineType.MONTE_CARLO(MonteCarloMethod.QUASI),
        )
        quad_price = quad_engine.price(snowball, env)
        mc_price = mc_engine.price(snowball, env)

    grids = [
        (150, 150),
        (200, 200),
        (300, 300),
        (400, 400),
    ]

    headers = [
        ("Grid", 9, "<"),
        ("Price", 12, ">"),
        ("PDE t", 9, ">"),
        ("Quad", 12, ">"),
        ("MC", 12, ">"),
        ("PDE %", 8, ">"),
    ]
    fmt = "| " + " | ".join(f"{{:{a}{w}}}" for _, w, a in headers) + " |"
    print(fmt.format(*[h[0] for h in headers]))
    print("| " + " | ".join("-" * h[1] for h in headers) + " |")

    profile_rows = []
    for grid_size, time_steps in grids:
        solver = SnowballPDESolver(
            params=PDEParams(grid=GridConfig(points=grid_size)),
            enable_profiling=True,
        )
        start = time.perf_counter()
        price = solver.price(snowball, env)
        elapsed = time.perf_counter() - start

        diff_pct = None
        if mc_price is not None and mc_price != 0.0:
            diff_pct = (price - mc_price) / mc_price * 100.0

        print(
            fmt.format(
                f"{grid_size}x{time_steps}",
                format_money(price),
                f"{elapsed:.4f}s",
                format_money(quad_price),
                format_money(mc_price),
                format_pct(diff_pct),
            )
        )
        profile = solver.get_profile_stats()
        profile_total = (
            profile.get("grid_build", 0.0)
            + profile.get("boundary", 0.0)
            + profile.get("matrix_build", 0.0)
            + profile.get("rhs", 0.0)
            + profile.get("solve", 0.0)
            + profile.get("barrier", 0.0)
        )
        profile_rows.append(
            (
                f"{grid_size}x{time_steps}",
                profile.get("grid_build", 0.0),
                profile.get("boundary", 0.0),
                profile.get("matrix_build", 0.0),
                profile.get("rhs", 0.0),
                profile.get("solve", 0.0),
                profile.get("barrier", 0.0),
                profile_total,
                max(0.0, elapsed - profile_total),
            )
        )

    if profile_rows:
        print("")
        prof_headers = [
            ("Grid", 9, "<"),
            ("Grid Bld", 9, ">"),
            ("Boundary", 9, ">"),
            ("Matrix", 9, ">"),
            ("RHS", 9, ">"),
            ("Solve", 9, ">"),
            ("Barrier", 9, ">"),
            ("Profile", 9, ">"),
            ("Other", 9, ">"),
        ]
        prof_fmt = "| " + " | ".join(f"{{:{a}{w}}}" for _, w, a in prof_headers) + " |"
        print(prof_fmt.format(*[h[0] for h in prof_headers]))
        print("| " + " | ".join("-" * h[1] for h in prof_headers) + " |")
        for row in profile_rows:
            print(
                prof_fmt.format(
                    row[0],
                    format_time(row[1]),
                    format_time(row[2]),
                    format_time(row[3]),
                    format_time(row[4]),
                    format_time(row[5]),
                    format_time(row[6]),
                    format_time(row[7]),
                    format_time(row[8]),
                )
            )

    print("")
    warm_headers = [
        ("Grid", 9, "<"),
        ("Cold t", 9, ">"),
        ("Warm t", 9, ">"),
        ("Speedup", 9, ">"),
    ]
    warm_fmt = "| " + " | ".join(f"{{:{a}{w}}}" for _, w, a in warm_headers) + " |"
    print(warm_fmt.format(*[h[0] for h in warm_headers]))
    print("| " + " | ".join("-" * h[1] for h in warm_headers) + " |")
    for grid_size, time_steps in grids:
        SnowballPDESolver.clear_grid_cache()
        solver = SnowballPDESolver(
            params=PDEParams(grid=GridConfig(points=grid_size))
        )
        start = time.perf_counter()
        _ = solver.price(snowball, env)
        cold = time.perf_counter() - start

        start = time.perf_counter()
        _ = solver.price(snowball, env)
        warm = time.perf_counter() - start

        speedup = cold / warm if warm > 0.0 else 0.0
        print(
            warm_fmt.format(
                f"{grid_size}x{time_steps}",
                format_time(cold),
                format_time(warm),
                f"{speedup:.2f}x",
            )
        )


if __name__ == "__main__":
    main()
