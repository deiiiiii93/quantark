"""E04: is the residual coarse-grid jitter of the cell-average arm (e03: 401 -> 1201 steps -2227, +654, -190, -151,
-90, +312) caused by the pinning? ``_pin_grid_targets`` moves ONE node onto each level, so its two neighbouring
cells become uneven (h_r / h_l from 0.35 to 2.99 in e02) and the ratio jumps around with n_x. The cell-average
projection does not need a node on the level, so this arm drops the pinning (plain sinh grid, the levels fall
mid-cell) and keeps the cell-average events. A nodal unpinned arm is the classic mid-cell control.

    e04_unpinned_ladder.py [arm ...]    (arms: cell_average, nodal; default both)
"""
import sys
import time

from common import RSS_CAP_GB, price, rss_estimate_gb

LADDER = (301, 401, 501, 601, 801, 1001, 1201, 1601, 2401, 3201, 4801)
N_V = 101

if __name__ == "__main__":
    arms = sys.argv[1:] or ["cell_average", "nodal"]
    for arm in arms:
        print(f"arm {arm}, UNPINNED grid, n_v {N_V}, substeps 1", flush=True)
        prev = None
        for n_x in LADDER:
            est = rss_estimate_gb(n_x, N_V)
            if est > RSS_CAP_GB:
                print(f"  n_x {n_x:>5}  SKIPPED (RSS rule {est:.2f} GB > {RSS_CAP_GB})", flush=True)
                continue
            t0 = time.time()
            v = price(n_x, N_V, events=arm, pin=False)
            d = "" if prev is None else f"d {v - prev:+11.4f}"
            print(f"  n_x {n_x:>5}  {v!r:>22}  {d}   ({est:.2f} GB rule, {time.time() - t0:.0f} s)", flush=True)
            prev = v
