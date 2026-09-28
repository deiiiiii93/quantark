"""Is V(S) just the linear interpolant of a spot-independent node array?

Grid alignment pins a node onto the closest barrier, so the lattice in
LOG-MONEYNESS moves with spot but the lattice in ABSOLUTE PRICE does not:
node k sits at B * exp(k*h) whatever the spot is.  The value array on those
nodes is therefore the same for every spot in a small sweep, and the engine's
price is np.interp at log-moneyness 0.

Two predictions, both falsifiable:
  1. the absolute price nodes are identical across the sweep;
  2. the kinks in V(S) sit exactly at S = B * exp(k*h).
"""
from __future__ import annotations
import numpy as np

import repro
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.engine.quad.quad_math import QuadratureMath

TTM = 0.295890
M_REF = 200.0


def main():
    product = repro.build(TTM)
    eng = SnowballQuadEngine(
        QuadParams(grid_points=401, min_diffusion_stddev_cells=2.5)
    )
    env0 = repro.env_at(repro.SPOT0)
    ki_times = [float(r.observation_time) for r in product.resolve_ki_observations(env0)]
    n_grid = eng._resolve_grid_points(TTM, repro.VOL, ki_times)

    def lattice(spot):
        align = eng._select_alignment_log(spot, product)
        mu = QuadratureMath(
            grid_x=n_grid, spot=spot, maturity=TTM, vol_max=repro.VOL,
            num_std_devs=eng.params.num_std_devs, align_log=align,
            integration_rule=eng.params.integration_rule,
        )
        return spot * np.exp(mu.grid), float(mu.h), align

    # --- prediction 1: absolute nodes do not move with spot -----------------
    lo, hi = 5319.0, 5320.8
    n_lo, h, _ = lattice(lo)
    n_hi, _, _ = lattice(hi)
    common = min(len(n_lo), len(n_hi))
    inner = slice(common // 2 - 20, common // 2 + 20)
    drift = np.abs(n_lo[inner] - n_hi[inner]).max()
    print(f"1) absolute price nodes over a {hi - lo:.1f}-point spot move:")
    print(f"   max drift of the 40 nodes around the money = {drift:.3e} index points")
    print(f"   cell width h = {h:.8f} log = {repro.SPOT0 * h:.4f} index points")

    # --- prediction 2: kinks sit on the nodes ------------------------------
    spots = np.linspace(lo, hi, 181)
    vals = np.array([eng.price(product, repro.env_at(s)) for s in spots])
    d1 = np.diff(vals) / np.diff(spots) / M_REF
    d2 = np.abs(np.diff(d1) - np.median(np.diff(d1)))
    kink = spots[1:-1][int(np.argmax(d2))]
    near = n_lo[np.argmin(np.abs(n_lo - kink))]
    print()
    print(f"2) largest kink at spot {kink:.4f}; nearest grid node {near:.4f}")
    print(f"   distance {abs(kink - near):.4f} index points "
          f"({abs(kink - near) / (repro.SPOT0 * h):.3f} cells)")
    print(f"   delta step across the kink = {d2.max():.5f} reference hands")

    # --- prediction 3: the engine's price IS the linear interpolant ---------
    node_vals = np.array(
        [eng.price(product, repro.env_at(float(s))) for s in n_lo[inner]]
    )
    probe = np.linspace(n_lo[inner][5], n_lo[inner][-6], 97)
    engine_px = np.array([eng.price(product, repro.env_at(float(s))) for s in probe])
    lin = np.interp(np.log(probe), np.log(n_lo[inner]), node_vals)
    rel = np.abs(engine_px - lin) / np.abs(engine_px)
    print()
    print("3) engine price vs linear interpolation of the node prices, in log S:")
    print(f"   max relative difference over {probe.size} probes = {rel.max():.3e}")
    print(f"   (a pure linear interpolant would give ~1e-16)")


if __name__ == "__main__":
    main()
