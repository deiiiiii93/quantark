"""
Snowball demo with term-structure volatility and dividend yield.

Usage:
    python example/snowball_term_structure_demo.py
"""

import sys
from datetime import datetime
from pathlib import Path


from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.param import SpotQuote, TermStructureVolSurface
from quantark.param.div import TermStructureDividendYield
from quantark.param.rrf import FlatRateCurve
from quantark.priceenv import PricingEnvironment


def build_term_structure_env() -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=TermStructureVolSurface(
            times=[1.0 / 12.0, 0.25, 0.5, 1.0],
            vols=[0.23, 0.22, 0.215, 0.21],
        ),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=TermStructureDividendYield(
            times=[1.0 / 12.0, 0.25, 0.5, 1.0],
            yields=[0.031, 0.030, 0.029, 0.028],
        ),
        valuation_date=datetime(2024, 1, 1),
    )


def main() -> None:
    product = create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        ko_barrier=103.0,
        ko_rate=0.15,
        ki_barrier=75.0,
        num_observations=12,
        is_reverse=False,
        include_principal=False,
    )
    env = build_term_structure_env()

    engine = SnowballQuadEngine(params=QuadParams(grid_points=401))
    price = engine.price(product, env)
    stats = engine.calculate_event_stats(product, env)

    print("=" * 60)
    print("Snowball with Term-Structure Vol & Dividend")
    print("=" * 60)
    print(f"PV: {price:,.6f}")
    if stats is None:
        print("Event stats unavailable.")
        return

    print(f"KI probability: {stats.ki_probability:.6f}")
    print(f"PV reconciliation error: {stats.reconciliation_error:.6g}")
    print("\nPer-observation KO stats:")
    for t, p_ko, p_surv in zip(stats.ko_times, stats.ko_probability, stats.survival_probability):
        print(f"  t={t:.3f}y  p_ko={p_ko:.6f}  p_survive={p_surv:.6f}")


if __name__ == "__main__":
    main()
