"""C04 (Q3): per-step growth at the KI barrier on the reverse continuous-KI Heston snowball KO 97 / KI 125.

Wraps ``_apply_continuous_ki_correction`` (the V0 step hook's crossing correction) and records, per backward step:
max |V0| on the live side within 10 cells of the barrier (before the correction), the largest |lambda| =
|d[k*] / a[k*]| over the variance columns with its column's variance and k*'s distance in barrier cells, and the
largest |correction|. Growth = the band max over the previous step's. Arms: (288, 48, 192) finite as-is, (384, 48,
192) as-is (diverges), (384, 48, 192) with the half-cell floor measured at the barrier (C02).
"""
import math

import numpy as np

from c02_q3_common import _ORIG_CORR, REV_C, barrier_cell_floor, barrier_geometry, px
from quantark.asset.equity.engine.pde.vol_continuous_ki import Heston2DBarrierCrossingMixin


def instrumented(n, label, every=12):
    g = barrier_geometry(n)
    cell = g["cell"]
    rows = []

    def wrapped(self, U, core, tau, barrier, v1):
        fp = self._ki_fp
        j = self._first_passage_step_index(core, tau)
        if fp is None or j is None:
            return _ORIG_CORR(self, U, core, tau, barrier, v1)
        S = np.asarray(core.S_grid, dtype=float)
        a = math.log(float(barrier)) - np.log(S)                      # reverse: live below the barrier
        band = (a > 0.0) & (a < 10.0 * cell)
        before = float(np.max(np.abs(U[band, :]))) if band.any() else 0.0
        d = v1 - U
        best = (0.0, None, None)
        for col in range(U.shape[1]):
            G, k, dist = fp._geometry(float(fp._dt[j]), float(fp._mu[j, col]), float(fp._sig2[j, col]), S,
                                      float(barrier))
            if G is None:
                continue
            lam = abs(d[k, col] / dist)
            if lam > best[0]:
                best = (lam, float(core.V_grid[col]), dist / cell)
        out = _ORIG_CORR(self, U, core, tau, barrier, v1)
        corr = float(np.max(np.abs(out - U)))
        rows.append((j, before, best, corr))
        return out

    Heston2DBarrierCrossingMixin._apply_continuous_ki_correction = wrapped
    try:
        price = px(n)
    finally:
        Heston2DBarrierCrossingMixin._apply_continuous_ki_correction = _ORIG_CORR
    print(f"\n== {label} n={n}: price {price}; nearest live node {g['cells']:.4f} cells below the barrier", flush=True)
    prev = None
    for i, (j, before, (lam, v, kcells), corr) in enumerate(rows):
        growth = before / prev if prev else float("nan")
        prev = before if before > 0 else prev
        if i < 6 or i % every == 0 or i == len(rows) - 1 or not math.isfinite(before) or before > 1e6:
            kc = f"{kcells:.4f}" if kcells is not None else "-"
            print(f"step j={j:>4} (hook call {i:>3})  band max|V0| {before:.4e}  growth {growth:8.3f}  "
                  f"max|lambda| {lam:.3e} (v={v if v is None else round(v, 5)}, k* at {kc} cells)  "
                  f"max|corr| {corr:.3e}", flush=True)
        if not math.isfinite(before) or before > 1e250:
            break


if __name__ == "__main__":
    instrumented((288, 48, 192), "as-is, finite")
    instrumented((384, 48, 192), "as-is, diverges")
    with barrier_cell_floor():
        instrumented((384, 48, 192), "half-cell floor at the barrier")
