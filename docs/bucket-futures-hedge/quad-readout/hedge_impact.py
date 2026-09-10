"""Does the staircase reach the hedge?

The replay sizes the hedge from ``calculate_greeks`` -- a 1% bump-and-reprice
delta -- and then rounds to whole futures contracts.  A delta error smaller
than the rounding granularity never reaches a trade.  Measure the 1% delta's
sawtooth across a cell and compare it with one contract.
"""
from __future__ import annotations
import numpy as np

import repro
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams

TTM = 0.295890
M_REF = 200.0
PRICING_BUMP = 0.01  # AutocallableEngineConfig default, what the replay uses


def main():
    product = repro.build(TTM)
    eng = SnowballQuadEngine(
        QuadParams(grid_points=401, min_diffusion_stddev_cells=2.5)
    )
    spots = np.linspace(5300.0, 5341.0, 42)
    fd, ref = [], []
    for s in spots:
        s = float(s)
        row = eng.calculate_spot_greeks_curve(product, repro.env_at(s), [s])[0]
        ref.append(row["delta"] / M_REF)
        up = eng.price(product, repro.env_at(s * (1 + PRICING_BUMP)))
        dn = eng.price(product, repro.env_at(s * (1 - PRICING_BUMP)))
        fd.append((up - dn) / (2 * s * PRICING_BUMP) / M_REF)
    fd, ref = np.asarray(fd), np.asarray(ref)
    err = fd - ref
    # a hand IS a contract here: hedge contracts = delta / multiplier
    print(f"# pricing/hedging bump {PRICING_BUMP:.2%} "
          f"({2 * PRICING_BUMP * 5320 / 19.697:.2f} cells wide)")
    print(f"# product delta {ref.mean():.2f} reference hands "
          f"= {ref.mean():.2f} IM contracts")
    print(f"# delta error vs grid delta: ptp {np.ptp(err):.4f} "
          f"mean|.| {np.abs(err).mean():.4f} max|.| {np.abs(err).max():.4f} contracts")
    print(f"# rounding granularity of the hedge: 1.0000 contract")
    print(f"# worst error as a fraction of one contract: "
          f"{np.abs(err).max():.3f}")
    print(f"# worst error as a fraction of the position: "
          f"{np.abs(err).max() / abs(ref.mean()):.4%}")


if __name__ == "__main__":
    main()
