"""
Snowball per-observation event stats demo (QUAD / PDE / MC).

This script demonstrates the engine-level event stats API:
    engine.calculate_event_stats(product, pricing_env)

It prints per-observation KO probabilities, survival, expected discounted KO cashflows,
and a PV decomposition summary.

Usage:
    python example/snowball_event_stats_demo.py
    python example/snowball_event_stats_demo.py --engine quad
    python example/snowball_event_stats_demo.py --engine pde --pde-grid 120 --pde-steps 120
    python example/snowball_event_stats_demo.py --engine mc --paths 50000 --steps 252
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from asset.equity.param import MCParams, PDEParams, QuadParams
from asset.equity.product.option.snowball_helpers import create_standard_snowball
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.enum.engine_enums import EngineType, MonteCarloMethod


def create_pricing_env(
    spot: float = 100.0,
    vol: float = 0.22,
    rate: float = 0.02,
    div_yield: float = 0.03,
) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div_yield),
        valuation_date=datetime(2024, 1, 1),
    )


def parse_engine_list(value: str) -> list[str]:
    items = [v.strip().lower() for v in value.split(",") if v.strip()]
    if not items:
        return ["quad", "pde", "mc"]
    valid = {"quad", "pde", "mc", "all"}
    for it in items:
        if it not in valid:
            raise ValueError(f"Unknown engine '{it}'. Use quad,pde,mc or all.")
    if "all" in items:
        return ["quad", "pde", "mc"]
    return items


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def print_event_stats(engine_name: str, stats) -> None:
    print("\n" + "=" * 88)
    print(f"Event Stats ({engine_name})")
    print("=" * 88)
    if stats is None:
        print("No event stats available.")
        return

    headers = ["idx", "ko_time", "p_ko", "p_survive", "ed_ko_cf"]
    fmt = "| {:>3} | {:>8} | {:>10} | {:>10} | {:>14} |"
    print(fmt.format(*headers))
    print("|" + "-" * (len(fmt.format(*headers)) - 2) + "|")
    for i in range(len(stats.ko_times)):
        print(
            fmt.format(
                i,
                _fmt(float(stats.ko_times[i])),
                _fmt(float(stats.ko_probability[i])),
                _fmt(float(stats.survival_probability[i])),
                _fmt(float(stats.expected_discounted_ko_cashflow[i])),
            )
        )

    pv_ko = float(np.sum(stats.expected_discounted_ko_cashflow))
    pv_mat = float(stats.expected_discounted_maturity_cashflow)
    pv_parts = pv_ko + pv_mat

    print("\nSummary")
    print(f"- PV:                     {_fmt(float(stats.pv))}")
    print(f"- PV(KO cashflows):       {_fmt(pv_ko)}")
    print(f"- PV(maturity cashflow):  {_fmt(pv_mat)}")
    print(f"- PV(decomp total):       {_fmt(pv_parts)}")
    print(f"- KI probability:         {_fmt(float(stats.ki_probability))}")
    print(f"- Recon error:            {_fmt(float(stats.reconciliation_error))}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo: per-observation Snowball event stats for QUAD/PDE/MC."
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="all",
        help="Comma-separated: quad,pde,mc or all (default: all)",
    )
    parser.add_argument("--grid", type=int, default=301, help="Quad grid points (odd)")
    parser.add_argument("--pde-grid", type=int, default=160, help="PDE grid size")
    parser.add_argument("--pde-steps", type=int, default=160, help="PDE time steps")
    parser.add_argument("--paths", type=int, default=20000, help="MC paths")
    parser.add_argument("--steps", type=int, default=252, help="MC time steps")
    parser.add_argument(
        "--mc-method",
        type=str,
        default="quasi",
        help="MC method: pseudo, quasi, randomized_quasi",
    )
    args = parser.parse_args()

    engines = parse_engine_list(args.engine)

    product = create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        ko_barrier=103.0,
        ko_rate=0.15,
        ki_barrier=75.0,
        num_observations=12,
        include_principal=False,
    )
    env = create_pricing_env()

    if "quad" in engines:
        quad = SnowballQuadEngine(params=QuadParams(grid_points=args.grid))
        print_event_stats("QUAD", quad.calculate_event_stats(product, env))

    if "pde" in engines:
        pde = SnowballPDESolver(
            params=PDEParams(grid_size=args.pde_grid, time_steps=args.pde_steps)
        )
        print_event_stats("PDE", pde.calculate_event_stats(product, env))

    if "mc" in engines:
        mc_params = MCParams(num_paths=args.paths, time_steps=args.steps, seed=42)
        mc_method = MonteCarloMethod[args.mc_method.upper()]
        mc = SnowballMCEngine(params=mc_params, method=EngineType.MONTE_CARLO(mc_method))
        print_event_stats("MC", mc.calculate_event_stats(product, env))


if __name__ == "__main__":
    main()

