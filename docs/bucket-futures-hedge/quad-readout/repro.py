"""Minimal repro: is the QUAD snowball price rough in spot near a live discrete KI?

Rebuilds the study's product as of the worst audit date (2024-01-19) and
sweeps spot.  Everything is flat and frozen: only spot moves, so any
non-smoothness in V(S) belongs to the engine.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import numpy as np

from quantark.asset.equity.product.option.snowball_helpers import (
    create_standard_snowball,
)
from quantark.util.enum import ObservationType
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.param import FlatVolSurface, SpotQuote
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.priceenv.pricing_environment import PricingEnvironment
from quantark.backtest.replay.market import SignedDividendYield

S0 = 6734.0
NOTIONAL = 50_000_000.0
KO, KI = 1.03 * S0, 0.75 * S0
COUPON = 0.045276
SPOT0, VOL, RATE, Q = 5306.986, 0.1774233515693156, 0.02, 0.1478011792574743
ACT = 365.0
M_REF = 200.0


def build(ttm: float):
    """Remaining schedule: monthly KO anniversaries, KI every trading day."""
    elapsed = 1.0 - ttm
    ko_times = sorted(
        {
            round(t, 12)
            for t in (np.arange(3, 13) / 12.0 - elapsed)
            if t > 1e-9
        }
        | {round(ttm, 12)}
    )
    ki_times = []
    d = 1
    while d / ACT <= ttm + 1e-12:
        if d % 7 not in (6, 0):
            ki_times.append(d / ACT)
        d += 1
    if not ki_times or abs(ki_times[-1] - ttm) > 1e-12:
        ki_times.append(ttm)
    return create_standard_snowball(
        initial_price=S0,
        strike=S0,
        maturity=ttm,
        contract_multiplier=NOTIONAL / S0,
        ko_barrier=KO,
        ko_rate=COUPON,
        ki_barrier=KI,
        num_observations=len(ko_times),
        is_reverse=False,
        ko_observation_dates=list(ko_times),
        ki_continuous=False,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=list(ki_times),
        rebate_rate=COUPON,
        include_principal=False,
    )


def env_at(spot: float, vol: float = VOL, q: float = Q):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(rate=RATE),
        valuation_date=datetime(2024, 1, 19),
        spot_quote=SpotQuote(spot=float(spot), asset_name="CSI1000"),
        vol_surface=FlatVolSurface(volatility=vol),
        div_yield=SignedDividendYield(q),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttm", type=float, default=0.295890)
    ap.add_argument("--n", type=int, default=161)
    ap.add_argument("--half-width-rel", type=float, default=0.012)
    ap.add_argument("--grid", type=int, default=401)
    ap.add_argument("--cells", type=float, default=2.5)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    product = build(args.ttm)
    eng = SnowballQuadEngine(
        QuadParams(grid_points=args.grid, min_diffusion_stddev_cells=args.cells)
    )
    spots = SPOT0 * (
        1.0 + np.linspace(-args.half_width_rel, args.half_width_rel, args.n)
    )
    vals = np.array([eng.price(product, env_at(s)) for s in spots])

    d1 = np.diff(vals) / np.diff(spots) / M_REF  # one-sided delta, ref hands
    d2 = np.diff(d1)
    print(
        f"# ttm={args.ttm} grid_req={args.grid} cells={args.cells} "
        f"n={args.n} halfwidth={args.half_width_rel}"
    )
    print(
        f"# spot [{spots[0]:.2f}, {spots[-1]:.2f}] step {spots[1] - spots[0]:.4f} "
        f"| price [{vals.min():.2f}, {vals.max():.2f}]"
    )
    print(
        f"# one-sided delta hands: min {d1.min():.4f} max {d1.max():.4f} "
        f"ptp {np.ptp(d1):.4f}"
    )
    print(
        f"# roughness: mean|d2| {np.abs(d2).mean():.5f} max|d2| {np.abs(d2).max():.5f} "
        f"at spot {spots[1:-1][int(np.argmax(np.abs(d2)))]:.2f}"
    )
    if args.out:
        np.save(args.out, np.vstack([spots, vals]))
    if not args.quiet:
        for s, v, d in zip(spots[:-1], vals[:-1], d1):
            print(f"{s:.4f} {v:.6f} {d:+.6f}")


if __name__ == "__main__":
    main()
