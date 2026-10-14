"""E09 (after the fix): the time axis of the production default at n_x 2401, n_v 41 (substeps 1 / 2 / 4), so the PDE
value can be set against the e06 MC reference at a converged time grid (d04: the time axis moves +205 then -60).
RSS rule at substeps 4: 16 B x 2401 x 41 x 1941 x 1.5 = 4.6 GB (< 6 GB).
"""
import time

from common import RSS_CAP_GB, price, rss_estimate_gb

if __name__ == "__main__":
    prev = None
    for sub in (1, 2, 4):
        est = rss_estimate_gb(2401, 41, sub)
        if est > RSS_CAP_GB:
            print(f"  substeps {sub}  SKIPPED (RSS rule {est:.2f} GB)", flush=True)
            continue
        t0 = time.time()
        v = price(2401, 41, sub)
        print(f"  n_x 2401, n_v 41, substeps {sub}  {v!r:>22}  {'' if prev is None else f'd {v - prev:+10.4f}'}"
              f"  ({est:.2f} GB rule, {time.time() - t0:.0f} s)", flush=True)
        prev = v
