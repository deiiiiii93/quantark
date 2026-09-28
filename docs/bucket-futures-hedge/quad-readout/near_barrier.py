"""Does the cubic readout overshoot where it matters -- right at the barrier?

The value function has a kink at the KI barrier, and alignment puts a node
exactly ON it.  A 4-point Lagrange stencil straddling that node spans the
kink, which is where a high-order interpolant is supposed to misbehave.  If
the cubic is going to fail anywhere it is here, so measure it here.
"""
from __future__ import annotations
import numpy as np

import repro
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.quad_math import QuadratureMath
from quantark.asset.equity.param import QuadParams
from candidates import cubic_interpolate

TTM = 0.295890
M_REF = 200.0
BUMP = 0.0005

_ORIGINAL = QuadratureMath.interpolate


def run(label, interp, spots):
    if interp is not None:
        QuadratureMath.interpolate = interp
    try:
        product = repro.build(TTM)
        eng = SnowballQuadEngine(
            QuadParams(grid_points=401, min_diffusion_stddev_cells=2.5)
        )
        errs = []
        for s in spots:
            s = float(s)
            row = eng.calculate_spot_greeks_curve(product, repro.env_at(s), [s])[0]
            ref = row["delta"] / M_REF
            up = eng.price(product, repro.env_at(s * (1 + BUMP)))
            dn = eng.price(product, repro.env_at(s * (1 - BUMP)))
            errs.append((up - dn) / (2 * s * BUMP) / M_REF - ref)
        errs = np.asarray(errs)
        print(f"  {label:<34} ptp {np.ptp(errs):8.4f}  mean|.| "
              f"{np.abs(errs).mean():8.4f}  max|.| {np.abs(errs).max():8.4f}")
    finally:
        QuadratureMath.interpolate = _ORIGINAL


def main():
    ki = repro.KI
    print(f"# KI barrier {ki:.2f}; cell ~19.7 index points\n")
    for lo, hi, tag in (
        (ki + 2.0, ki + 42.0, "0.1 to 2.1 cells above the barrier"),
        (ki + 60.0, ki + 100.0, "3 to 5 cells above"),
        (ki + 240.0, ki + 280.0, "12 to 14 cells above (control)"),
    ):
        spots = np.linspace(lo, hi, 42)
        print(f"{tag}  [{lo:.1f}, {hi:.1f}]")
        run("linear readout (current)", None, spots)
        run("cubic readout", cubic_interpolate, spots)
        print()


if __name__ == "__main__":
    main()
