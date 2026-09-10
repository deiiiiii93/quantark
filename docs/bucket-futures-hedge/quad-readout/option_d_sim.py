"""Simulate option (d) end to end, runtime patched.

Rule: drop the grid SHIFT entirely (so log-moneyness 0 is the centre node of
an odd grid, exactly), and instead stretch the envelope log_c a hair so the
barrier lands on that same lattice:  (n-1)*|log(B/S)| / (2*log_c) = k.

Two variants of where the residual goes:
  d1  n fixed at the adaptive floor; log_c absorbs it (k flips as spot moves)
  d2  n searched in a window for a near-integer fit; log_c barely moves

Both are measured the same way as the current readout: the sub-cell
finite-difference delta against the engine's grid-gradient delta.
"""
from __future__ import annotations
import math
import numpy as np

import repro
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams

TTM = 0.295890
M_REF = 200.0
BUMP = 0.0005
SPOTS = np.linspace(5300.0, 5341.0, 42)
FLOOR = 683  # what the adaptive rule returns at this state

_ORIG_ALIGN = SnowballQuadEngine._select_alignment_log
_ORIG_GRID = SnowballQuadEngine._resolve_grid_points


def log_c_of(num_std):
    return num_std * repro.VOL * math.sqrt(TTM) + (
        1.0 + 0.5 * repro.VOL * repro.VOL
    ) * TTM


def num_std_for(log_c):
    return (log_c - (1.0 + 0.5 * repro.VOL * repro.VOL) * TTM) / (
        repro.VOL * math.sqrt(TTM)
    )


def choose(spot, variant):
    """(n, num_std_devs) putting both spot and the barrier on nodes."""
    target = abs(math.log(repro.KI / spot))
    base_log_c = log_c_of(10.0)
    if variant == "d1":
        n = FLOOR
        k = max(1, round((n - 1) * target / (2.0 * base_log_c)))
    else:
        best = None
        for cand in range(FLOOR, FLOOR + 400, 2):
            exact = (cand - 1) * target / (2.0 * base_log_c)
            kk = round(exact)
            if kk <= 0:
                continue
            err = abs(exact - kk)
            if best is None or err < best[0]:
                best = (err, cand, kk)
        _, n, k = best
    log_c = (n - 1) * target / (2.0 * k)
    return n, num_std_for(log_c)


def measure(label, variant):
    product = repro.build(TTM)
    errs, ns = [], []
    for s in SPOTS:
        s = float(s)
        n, nsd = choose(s, variant)
        ns.append(n)
        eng = SnowballQuadEngine(
            QuadParams(grid_points=n, min_diffusion_stddev_cells=0.0,
                       num_std_devs=nsd)
        )
        SnowballQuadEngine._select_alignment_log = lambda *a, **k: None
        SnowballQuadEngine._resolve_grid_points = lambda self, *a, **k: n
        try:
            row = eng.calculate_spot_greeks_curve(product, repro.env_at(s), [s])[0]
            ref = row["delta"] / M_REF
            up = eng.price(product, repro.env_at(s * (1 + BUMP)))
            dn = eng.price(product, repro.env_at(s * (1 - BUMP)))
            errs.append((up - dn) / (2 * s * BUMP) / M_REF - ref)
        finally:
            SnowballQuadEngine._select_alignment_log = _ORIG_ALIGN
            SnowballQuadEngine._resolve_grid_points = _ORIG_GRID
    errs = np.asarray(errs)
    print(f"{label:<40} ptp {np.ptp(errs):8.4f}  mean|.| {np.abs(errs).mean():8.4f}"
          f"  n in [{min(ns)}, {max(ns)}]")


def main():
    print("# sub-cell FD delta error vs grid delta, reference hands")
    print(f"# bump {BUMP:.2%}, {SPOTS.size} spots over ~2 cells")
    print("# reference: current linear readout gives ptp 0.2454, mean 0.0544\n")
    measure("d1  n fixed, envelope absorbs residual", "d1")
    measure("d2  n searched, envelope barely moves", "d2")


if __name__ == "__main__":
    main()
