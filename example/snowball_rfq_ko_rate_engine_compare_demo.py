"""
Smoke-check Snowball RFQ KO-rate consistency across Quad, PDE, and MC engines.

This compares the fair KO rate implied by the same RFQ convention used in
`generate_snowball_rfq_ko_rate_demo.py`:
- ex-principal Snowball PV
- financing leg inferred from a protected 100% KO-rate leg
- affine solve between KO rate 0% and 200%

Usage:
    python example/snowball_rfq_ko_rate_engine_compare_demo.py
    python example/snowball_rfq_ko_rate_engine_compare_demo.py --paths 50000 --quad-grid 1001 --pde-grid 400 --pde-steps 400
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any


from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde import GridConfig, SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from example.snowball_rfq_ko_rate_demo_workflow import (
    BASE_KI_BARRIER,
    BASE_KO_BARRIER,
    solve_fair_ko_rate_with_engine,
)
from quantark.util.enum.engine_enums import EngineType, MonteCarloMethod
from quantark.util.numerical import format_basis_points, format_percentage


@dataclass(frozen=True)
class SmokeCase:
    """Compact RFQ KO-rate scenario definition."""

    label: str
    rate: float
    div_yield: float
    vol: float
    tenor: float
    variant: str = "standard"
    ko_barrier: float = BASE_KO_BARRIER
    ki_barrier: float = BASE_KI_BARRIER


DEFAULT_SMOKE_CASES: tuple[SmokeCase, ...] = (
    SmokeCase("Base 2Y", rate=0.03, div_yield=0.10, vol=0.20, tenor=2.0),
    SmokeCase("Carry 2Y", rate=0.02, div_yield=0.14, vol=0.18, tenor=2.0),
    SmokeCase("Stress 2Y", rate=0.05, div_yield=0.06, vol=0.30, tenor=2.0),
    SmokeCase("Long 3Y", rate=0.04, div_yield=0.09, vol=0.25, tenor=3.0),
)


def build_engines(
    *,
    mc_paths: int = 50_000,
    quad_grid: int = 1001,
    pde_grid: int = 400,
    pde_steps: int = 400,
    seed: int = 42,
) -> dict[str, Any]:
    """Build the requested engine bundle for KO-rate smoke checks."""
    return {
        "quad": SnowballQuadEngine(params=QuadParams(grid_points=quad_grid)),
        "pde": SnowballPDESolver(
            params=PDEParams(grid=GridConfig(points=pde_grid))
        ),
        "mc": SnowballMCEngine(
            params=MCParams(
                num_paths=mc_paths,
                seed=seed,
                rqmc_paths_mode="total",
                rqmc_min_batches=4,
                rqmc_max_batches=4,
                rqmc_target_std=1e9,
            ),
            method=EngineType.MONTE_CARLO(MonteCarloMethod.RANDOMIZED_QUASI),
        ),
    }


def run_case(case: SmokeCase, engines: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Run one KO-rate consistency case across the supplied engines."""
    row: dict[str, dict[str, float]] = {}
    for name, engine in engines.items():
        start = time.perf_counter()
        result = solve_fair_ko_rate_with_engine(
            engine,
            rate=case.rate,
            div_yield=case.div_yield,
            vol=case.vol,
            tenor=case.tenor,
            variant=case.variant,
            ko_barrier=case.ko_barrier,
            ki_barrier=case.ki_barrier,
        )
        elapsed = time.perf_counter() - start
        row[name] = {
            **result,
            "elapsed_seconds": elapsed,
        }
    return row


def format_case(case: SmokeCase) -> str:
    """Render a compact scenario label."""
    return (
        f"{case.label}: T={case.tenor:.1f}, r={format_percentage(case.rate)}, "
        f"q={format_percentage(case.div_yield)}, vol={format_percentage(case.vol)}"
    )


def print_summary(case: SmokeCase, row: dict[str, dict[str, float]]) -> None:
    """Print one formatted comparison block."""
    quad_quote = row["quad"]["quoted_ko_rate"]
    pde_quote = row["pde"]["quoted_ko_rate"]
    mc_quote = row["mc"]["quoted_ko_rate"]
    deterministic_mid = 0.5 * (quad_quote + pde_quote)
    max_spread = max(quad_quote, pde_quote, mc_quote) - min(quad_quote, pde_quote, mc_quote)

    print("\n" + "=" * 88)
    print(format_case(case))
    print("=" * 88)
    print(
        f"{'Engine':<8} {'Quote':>10} {'Protected PV':>14} {'Target PV':>12} "
        f"{'Runtime':>10}"
    )
    print("-" * 88)
    for engine_name in ("quad", "pde", "mc"):
        result = row[engine_name]
        print(
            f"{engine_name.upper():<8} "
            f"{format_percentage(result['quoted_ko_rate']):>10} "
            f"{result['protected_snowball_pv']:>14.6f} "
            f"{result['snowball_target_pv']:>12.6f} "
            f"{result['elapsed_seconds']:>9.3f}s"
        )

    print("-" * 88)
    print(
        f"PDE-Quad: {format_basis_points(pde_quote - quad_quote)} | "
        f"MC-Mid: {format_basis_points(mc_quote - deterministic_mid)} | "
        f"Max Spread: {format_basis_points(max_spread)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-check Snowball RFQ KO-rate consistency across engines."
    )
    parser.add_argument("--paths", type=int, default=50_000, help="MC total paths")
    parser.add_argument(
        "--quad-grid", type=int, default=1001, help="Quad grid points"
    )
    parser.add_argument("--pde-grid", type=int, default=400, help="PDE grid size")
    parser.add_argument("--pde-steps", type=int, default=400, help="PDE time steps")
    args = parser.parse_args()

    engines = build_engines(
        mc_paths=args.paths,
        quad_grid=args.quad_grid,
        pde_grid=args.pde_grid,
        pde_steps=args.pde_steps,
    )

    for case in DEFAULT_SMOKE_CASES:
        print_summary(case, run_case(case, engines))


if __name__ == "__main__":
    main()
