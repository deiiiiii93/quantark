"""Which formula predicts the sawtooth amplitude?

I claimed the sub-cell finite-difference delta error is a sawtooth of
amplitude (cell/2)*Gamma.  That treats the interpolant as linear in S.  It is
linear in x = log S, so the chord equals dV/dx at the cell midpoint in the
LOG coordinate:

    delta error = (x_mid - x) * (S*Gamma + Delta) / 1      [currency/point]

giving amplitude (h/2)*(S*Gamma + Delta), peak-to-peak twice that.  Measure
both against the observed sawtooth at several states.
"""
from __future__ import annotations
import numpy as np

import repro
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.quad_math import QuadratureMath
from quantark.asset.equity.param import QuadParams

TTM = 0.295890
M_REF = 200.0
BUMP = 0.0005


def main():
    import sys

    global BUMP
    if len(sys.argv) > 1:
        BUMP = float(sys.argv[1])
    product = repro.build(TTM)
    eng = SnowballQuadEngine(
        QuadParams(grid_points=401, min_diffusion_stddev_cells=2.5)
    )
    print(f"# bump {BUMP:.4%}")
    print(f"{'centre':>9} {'measured ptp':>13} {'(cell/2)*G':>12} "
          f"{'(h/2)*(S*G+D)':>15} {'ratio':>7} {'bump/cell':>10}")
    for centre in (repro.KI + 260.0, repro.KI + 400.0, repro.KI + 600.0,
                   repro.KI + 900.0):
        env = repro.env_at(centre)
        ki_times = [
            float(r.observation_time) for r in product.resolve_ki_observations(env)
        ]
        n = eng._resolve_grid_points(TTM, repro.VOL, ki_times)
        mu = QuadratureMath(
            grid_x=n, spot=centre, maturity=TTM, vol_max=repro.VOL,
            num_std_devs=eng.params.num_std_devs,
            align_log=eng._select_alignment_log(centre, product),
            integration_rule=eng.params.integration_rule,
        )
        h = float(mu.h)
        row = eng.calculate_spot_greeks_curve(product, env, [centre])[0]
        delta_ccy = row["delta"]
        gamma = row["gamma"]

        # observed sawtooth over one full cell
        span = centre * h
        spots = np.linspace(centre - 0.5 * span, centre + 0.5 * span, 40)
        errs = []
        for s in spots:
            s = float(s)
            ref = eng.calculate_spot_greeks_curve(product, repro.env_at(s), [s])[0]
            up = eng.price(product, repro.env_at(s * (1 + BUMP)))
            dn = eng.price(product, repro.env_at(s * (1 - BUMP)))
            errs.append(
                (up - dn) / (2 * s * BUMP) / M_REF - ref["delta"] / M_REF
            )
        measured = float(np.ptp(errs))
        old = 2.0 * abs(0.5 * span * gamma) / M_REF
        new = 2.0 * abs(0.5 * h * (centre * gamma + delta_ccy)) / M_REF
        print(f"{centre:9.1f} {measured:13.4f} {old:12.4f} {new:15.4f} "
              f"{measured / new:7.3f} {2 * BUMP * centre / span:10.3f}")


if __name__ == "__main__":
    main()
