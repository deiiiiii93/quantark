"""
Snowball Quad vs MC vs PDE sanity-check demo.

This script compares SnowballQuadEngine, SnowballMCEngine, and SnowballPDESolver
across several snowball configurations. Results are printed in a table.

Usage:
    python example/snowball_quad_mc_pde_demo.py
    python example/snowball_quad_mc_pde_demo.py --paths 30000 --grid 801 --pde-grid 200 --pde-steps 200
    python example/snowball_quad_mc_pde_demo.py --method randomized_quasi --rqmc-paths-mode total --rqmc-max-batches 4

RQMC examples:
    # Total paths across batches (Sobol-ideal per batch)
    python example/snowball_quad_mc_pde_demo.py --method randomized_quasi --paths 100000 --rqmc-paths-mode total --rqmc-min-batches 4 --rqmc-max-batches 4

    # Per-batch paths (legacy behavior)
    python example/snowball_quad_mc_pde_demo.py --method randomized_quasi --paths 25000 --rqmc-paths-mode per_batch --rqmc-min-batches 4 --rqmc-max-batches 4

    # Absolute target std (price units)
    python example/snowball_quad_mc_pde_demo.py --method randomized_quasi --rqmc-target-std 0.5 --rqmc-target-std-mode absolute
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path


from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import (
    MCParams,
    PDEParams,
    QuadParams,
    make_pde_params,
    make_quad_params,
)
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


def format_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}%"


def run_case(
    label: str,
    snowball,
    pricing_env: PricingEnvironment,
    quad_params: QuadParams,
    pde_params: PDEParams,
    mc_engine: SnowballMCEngine,
    ko_freq: str,
    ki_freq: str | None,
) -> dict:
    quad_price = pde_price = mc_price = None
    quad_time = pde_time = mc_time = None

    quad_engine = SnowballQuadEngine(params=quad_params)
    pde_engine = SnowballPDESolver(params=pde_params)

    start = time.perf_counter()
    quad_price = quad_engine.price(snowball, pricing_env)
    quad_time = time.perf_counter() - start

    start = time.perf_counter()
    pde_price = pde_engine.price(snowball, pricing_env)
    pde_time = time.perf_counter() - start

    start = time.perf_counter()
    mc_price = mc_engine.price(snowball, pricing_env)
    mc_time = time.perf_counter() - start

    quad_diff_pct = None
    pde_diff_pct = None
    if mc_price is not None and mc_price != 0.0:
        quad_diff_pct = (quad_price - mc_price) / mc_price * 100.0
        pde_diff_pct = (pde_price - mc_price) / mc_price * 100.0

    ki_label = "continuous"
    if snowball.has_ki_barrier:
        ki_cont = (
            snowball.barrier_config.ki_continuous
            or snowball.barrier_config.ki_observation_type == ObservationType.CONTINUOUS
        )
        if not ki_cont:
            ki_label = f"{ki_freq} ({snowball.num_ki_observations})" if ki_freq else str(
                snowball.num_ki_observations
            )
    else:
        ki_label = "none"

    return {
        "case": label,
        "ki_mode": "Continuous" if ki_label == "continuous" else "Discrete",
        "ko_obs": f"{ko_freq} ({snowball.num_ko_observations})" if ko_freq else str(snowball.num_ko_observations),
        "ki_obs": ki_label,
        "quad": quad_price,
        "pde": pde_price,
        "mc": mc_price,
        "quad_diff_pct": quad_diff_pct,
        "pde_diff_pct": pde_diff_pct,
        "quad_time": quad_time,
        "pde_time": pde_time,
        "mc_time": mc_time,
    }


def print_table(rows: list[dict]) -> None:
    headers = [
        ("Case", 32, "<"),
        ("KI Mode", 9, "<"),
        ("KO Obs", 14, "<"),
        ("KI Obs", 16, "<"),
        ("Quad", 13, ">"),
        ("PDE", 13, ">"),
        ("MC", 13, ">"),
        ("Quad %", 8, ">"),
        ("PDE %", 8, ">"),
        ("Quad t", 8, ">"),
        ("PDE t", 8, ">"),
        ("MC t", 8, ">"),
    ]

    fmt = "| " + " | ".join(f"{{:{align}{width}}}" for _, width, align in headers) + " |"
    header_line = fmt.format(*[h[0] for h in headers])
    sep_line = "| " + " | ".join("-" * h[1] for h in headers) + " |"

    print(header_line)
    print(sep_line)
    for row in rows:
        quad_t = f"{row['quad_time']:.4f}s" if row["quad_time"] is not None else "n/a"
        pde_t = f"{row['pde_time']:.4f}s" if row["pde_time"] is not None else "n/a"
        mc_t = f"{row['mc_time']:.4f}s" if row["mc_time"] is not None else "n/a"
        print(
            fmt.format(
                row["case"],
                row["ki_mode"],
                row["ko_obs"],
                row["ki_obs"],
                format_money(row["quad"]),
                format_money(row["pde"]),
                format_money(row["mc"]),
                format_pct(row["quad_diff_pct"]),
                format_pct(row["pde_diff_pct"]),
                quad_t,
                pde_t,
                mc_t,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity check: Snowball Quad vs MC vs PDE."
    )
    parser.add_argument("--paths", type=int, default=100000, help="MC paths")
    parser.add_argument("--grid", type=int, default=801, help="Quad grid points (odd)")
    parser.add_argument("--pde-grid", type=int, default=200, help="PDE grid size")
    parser.add_argument("--pde-steps", type=int, default=200, help="PDE time steps")
    parser.add_argument(
        "--method",
        type=str,
        default="randomized_quasi",
        help="MC method: pseudo, quasi, randomized_quasi",
    )
    parser.add_argument(
        "--rqmc-paths-mode",
        type=str,
        default="total",
        choices=["per_batch", "total"],
        help="RQMC paths interpretation: per_batch or total",
    )
    parser.add_argument(
        "--rqmc-min-batches",
        type=int,
        default=4,
        help="Minimum RQMC batches",
    )
    parser.add_argument(
        "--rqmc-max-batches",
        type=int,
        default=4,
        help="Maximum RQMC batches",
    )
    parser.add_argument(
        "--rqmc-target-std",
        type=float,
        default=1e-4,
        help="RQMC target standard error",
    )
    parser.add_argument(
        "--rqmc-target-std-mode",
        type=str,
        default="relative_notional",
        choices=["absolute", "relative_notional", "relative_price"],
        help="RQMC target std scaling mode",
    )
    parser.add_argument(
        "--rqmc-target-std-scale",
        type=float,
        default=None,
        help="Optional scale override for relative target std",
    )
    args = parser.parse_args()

    mc_engine = SnowballMCEngine(
        params=MCParams(
            num_paths=args.paths,
            time_steps=252,
            seed=42,
            rqmc_target_std=args.rqmc_target_std,
            rqmc_target_std_mode=args.rqmc_target_std_mode,
            rqmc_target_std_scale=args.rqmc_target_std_scale,
            rqmc_paths_mode=args.rqmc_paths_mode,
            rqmc_min_batches=args.rqmc_min_batches,
            rqmc_max_batches=args.rqmc_max_batches,
        ),
        method=EngineType.MONTE_CARLO(parse_mc_method(args.method)),
    )

    base_env = create_pricing_env()
    quarterly_obs = [i / 4 for i in range(1, 5)]
    daily_obs = [i / 252 for i in range(1, 253)]

    cases = [
        (
            "Standard (cont KI, monthly KO)",
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
            "Standard (disc KI, quarterly KO)",
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
            "Standard (daily KI, monthly KO)",
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
                ki_observation_dates=daily_obs,
            ),
            base_env,
            "monthly",
            "daily",
        ),
        (
            "Standard (high vol 35%)",
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
            "Standard (airbag 50% below 80)",
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
            "Standard (call-rebate V0)",
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
            "Standard (spot near KO)",
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
            "Standard (spot near KI)",
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
            "Reverse (cont KI, monthly KO)",
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
            "Standard (2Y maturity, monthly KO)",
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

    def resolve_case_params(label: str) -> tuple[QuadParams, PDEParams]:
        quad_params = QuadParams(grid_points=args.grid)
        pde_params = PDEParams(grid_size=args.pde_grid, time_steps=args.pde_steps)

        if label == "Standard (daily KI, monthly KO)":
            pde_params = make_pde_params(
                profile="barrier_sensitive",
                time_steps=max(args.pde_steps, 300),
                event_steps_per_day=6,
            )
        elif label == "Standard (high vol 35%)":
            pde_params = make_pde_params(profile="accurate")
        elif label == "Standard (spot near KO)":
            pde_params = make_pde_params(
                profile="barrier_sensitive",
            )
        elif label == "Reverse (cont KI, monthly KO)":
            quad_params = make_quad_params(
                profile="reverse_sensitive",
                num_std_devs=12.0,
            )
            pde_params = make_pde_params(
                profile="reverse_sensitive",
            )
        elif label == "Standard (2Y maturity, monthly KO)":
            quad_params = make_quad_params(
                profile="accurate",
                grid_points=max(args.grid, 1601),
            )
            pde_params = make_pde_params(
                profile="accurate",
                grid_size=max(args.pde_grid, 800),
                time_steps=max(args.pde_steps, 400) * 2,
            )

        return quad_params, pde_params

    rows = []
    for label, snowball, env, ko_freq, ki_freq in cases:
        quad_params, pde_params = resolve_case_params(label)
        rows.append(
            run_case(
                label,
                snowball,
                env,
                quad_params,
                pde_params,
                mc_engine,
                ko_freq,
                ki_freq,
            )
        )

    print_table(rows)


if __name__ == "__main__":
    main()
