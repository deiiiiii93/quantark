"""E03: dense n_x ladder (n_v 101, substeps 1) in three event-treatment arms of the probe (common.py):
nodal (= production, bit-identical, e01), inclusive (ULP-robust nodal masks), cell_average (the 1D DCN route's
dual-cell projection). One solve at a time; each cell's RSS budget (16 B x n_x x n_v x 486 x 1.5) is printed and a
cell above 6 GB is skipped.

    e03_dense_ladder.py [arm ...]       (default: all three)
"""
import sys
import time

from common import RSS_CAP_GB, price, rss_estimate_gb

LADDER = (301, 401, 501, 601, 801, 1001, 1201, 1601, 2401, 3201, 4801)
N_V = 101

if __name__ == "__main__":
    arms = sys.argv[1:] or ["nodal", "inclusive", "cell_average"]
    for arm in arms:
        print(f"arm {arm}, n_v {N_V}, substeps 1", flush=True)
        prev = None
        for n_x in LADDER:
            est = rss_estimate_gb(n_x, N_V)
            if est > RSS_CAP_GB:
                print(f"  n_x {n_x:>5}  SKIPPED (RSS rule {est:.2f} GB > {RSS_CAP_GB})", flush=True)
                continue
            t0 = time.time()
            v = price(n_x, N_V, events=arm)
            d = "" if prev is None else f"d {v - prev:+11.4f}"
            print(f"  n_x {n_x:>5}  {v!r:>22}  {d}   ({est:.2f} GB rule, {time.time() - t0:.0f} s)", flush=True)
            prev = v
