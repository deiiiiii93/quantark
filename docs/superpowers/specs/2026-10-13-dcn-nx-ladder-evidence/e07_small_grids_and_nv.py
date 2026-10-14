"""E07: (a) the small-grid ladder the regression test can afford (n_v 41, n_x 301 / 401 / 501 / 601), production
nodal vs the probe's cell-average + unpinned arm; (b) the variance axis under cell-average + unpinned at n_x 1201
(n_v 41 / 101 / 201), to show the n_v = 41 choice of (a) is harmless (d04: the patched n_v axis is flat).
"""
import time

from common import price

if __name__ == "__main__":
    for label, kw in (("production (nodal, pinned)", dict(events="production")),
                      ("cell_average, unpinned", dict(events="cell_average", pin=False))):
        print(f"(a) {label}, n_v 41", flush=True)
        prev = None
        for n_x in (301, 401, 501, 601):
            t0 = time.time()
            v = price(n_x, 41, **kw)
            print(f"  n_x {n_x:>4}  {v!r:>22}  {'' if prev is None else f'd {v - prev:+10.4f}'}"
                  f"  ({time.time() - t0:.1f} s)", flush=True)
            prev = v
    print("(b) cell_average, unpinned, n_x 1201", flush=True)
    for n_v in (41, 101, 201):
        t0 = time.time()
        print(f"  n_v {n_v:>3}  {price(1201, n_v, events='cell_average', pin=False)!r:>22}  "
              f"({time.time() - t0:.0f} s)", flush=True)
