"""E01: reproduce d04's patched n_x ladder (n_v 101, substeps 1) on this worktree, production solver; and check the
probe's "nodal" arm is bit-identical to production (so later probe arms differ ONLY in the event treatment).

d04 (2026-10-10, before T11/T12/T13): 301 -97286.0115, 601 -93698.0363, 1201 -93695.6189, 2401 -93281.8252.
"""
import time

from common import price, rss_estimate_gb, show_ladder

D04 = {301: -97286.0115, 601: -93698.0363, 1201: -93695.6189, 2401: -93281.8252}

if __name__ == "__main__":
    rows = []
    for n_x in (301, 601, 1201, 2401):
        est = rss_estimate_gb(n_x, 101)
        t0 = time.time()
        v = price(n_x, 101)
        rows.append((n_x, v))
        print(f"n_x {n_x} (RSS rule {est:.2f} GB): {v!r}  d04 {D04[n_x]:.4f}  diff {v - D04[n_x]:+.4f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
    show_ladder("production, n_v 101, substeps 1", rows)
    for n_x in (301, 601):
        a, b = price(n_x, 101), price(n_x, 101, events="nodal")
        print(f"probe nodal vs production at n_x {n_x}: {a!r} {b!r} "
              f"{'bit-identical' if a.hex() == b.hex() else 'DIFFERS'}", flush=True)
