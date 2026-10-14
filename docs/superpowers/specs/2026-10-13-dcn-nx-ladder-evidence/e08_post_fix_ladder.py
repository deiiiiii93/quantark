"""E08 (after the fix): the production HestonDCNPDESolver ladder at its new default (event_projection None =
CELL_AVERAGE on the unpinned grid), n_v 101, substeps 1, the e03 n_x ladder; plus (a) a bitwise check that the
production default equals the probe's cell_average + unpinned arm (e04) and that event_projection="nodal" equals
the pre-fix production values (e03 nodal arm); (b) the moved values at the solver's default grid (301, 101, 1)
and at the (151, 41) grid of the execution matrix / tests.
"""
import time

from common import RSS_CAP_GB, price, rss_estimate_gb

E04_CA = {301: -92971.8704756959, 601: -93027.99686914583}           # probe, n_v 101
E03_NODAL = {301: -97286.01153979279, 601: -93698.03627772542}       # pre-fix production, n_v 101
LADDER = (301, 401, 501, 601, 801, 1001, 1201, 1601, 2401, 3201, 4801)

if __name__ == "__main__":
    for n_x in (301, 601):
        new = price(n_x, 101)
        old = price(n_x, 101, event_projection="nodal")
        print(f"n_x {n_x}: default {new!r} vs probe CA-unpinned {E04_CA[n_x]!r}: "
              f"{'bit-identical' if new == E04_CA[n_x] else 'DIFFERS'}; nodal {old!r} vs pre-fix {E03_NODAL[n_x]!r}: "
              f"{'bit-identical' if old == E03_NODAL[n_x] else 'DIFFERS'}", flush=True)
    for grid in ((301, 101), (151, 41)):
        new, old = price(*grid), price(*grid, event_projection="nodal")
        print(f"moved value at (n_x, n_v) {grid}: {old:.4f} -> {new:.4f} ({new - old:+.4f})", flush=True)
    print("production default (cell-average, unpinned), n_v 101, substeps 1", flush=True)
    prev = None
    for n_x in LADDER:
        est = rss_estimate_gb(n_x, 101)
        if est > RSS_CAP_GB:
            print(f"  n_x {n_x:>5}  SKIPPED (RSS rule {est:.2f} GB > {RSS_CAP_GB})", flush=True)
            continue
        t0 = time.time()
        v = price(n_x, 101)
        print(f"  n_x {n_x:>5}  {v!r:>22}  {'' if prev is None else f'd {v - prev:+11.4f}'}   "
              f"({est:.2f} GB rule, {time.time() - t0:.0f} s)", flush=True)
        prev = v
