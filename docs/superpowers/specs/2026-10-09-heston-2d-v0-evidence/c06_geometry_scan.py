"""C06 (Q3): how often does an n_x put the live node nearest the KI barrier a small fraction of a cell from it?

Geometry only (the ADI core's S grid; no solve) for the reverse continuous-KI Heston snowball KO 97 / KI 125, every
n_x from 100 to 1200, on the default concentrated grid (n_v 48, n_t 192: the S grid does not depend on them) and on
the production controls (C05). C03 measured: as-is diverges at 0.0039 / 0.0046 / 0.0182 cells and is finite but off by
up to 6.6e-4 at 0.027-0.037 cells. Also the n_x the production profile resolves (200 PV, 300 Greeks, 600 floor).
"""
from c02_q3_common import barrier_geometry
from c05_production_and_inertness import PROD

if __name__ == "__main__":
    for label, kw in (("default", {}), ("production controls", PROD)):
        rows = [(n_x, barrier_geometry((n_x, 48, 192), **kw)["cells"]) for n_x in range(100, 1201)]
        for thr in (0.02, 0.05, 0.10):
            hit = [n for n, c in rows if c < thr]
            print(f"{label:<20} n_x in [100, 1200]: {len(hit):>3} of {len(rows)} within {thr:.2f} cells"
                  + (f"  e.g. {hit[:12]}" if hit else ""), flush=True)
        for n_x in (200, 300, 600):
            print(f"{label:<20} n_x={n_x}: nearest live node {dict(rows)[n_x]:.4f} cells below the barrier", flush=True)
