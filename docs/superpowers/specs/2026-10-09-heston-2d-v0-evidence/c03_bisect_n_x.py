"""C03 (Q3): scan n_x from 288 to 384 at (n_x, 48, 192) on the reverse continuous-KI Heston snowball KO 97 / KI 125.

A plain bisection assumes the divergence is monotone in n_x; an alignment effect is not, so every n_x is priced.
Per n_x: the live node nearest the KI barrier (below it: a reverse product's live side), its log distance in cells
of the barrier cell, the cell the library's half-cell floor measures (a[live[1]] - a[live[0]]; negative = the floor
removes nothing), the price as-is, with the floor measured at the barrier (C02), and with the crossing correction off.
"""
import sys

from c02_q3_common import barrier_cell_floor, barrier_geometry, correction_off, px

if __name__ == "__main__":
    n_v, n_t = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (48, 192)
    print(f"reverse continuous KI 125 / KO 97, n_v={n_v}, n_t={n_t}", flush=True)
    for n_x in range(288, 385):
        n = (n_x, n_v, n_t)
        g = barrier_geometry(n)
        a = px(n)
        with barrier_cell_floor():
            b = px(n)
        with correction_off():
            c = px(n)
        print(f"n_x={n_x}  nearest live S={g['nearest']:.6f} at {g['cells']:.4f} cells (cell {g['cell']:.3e}, "
              f"library floor cell {g['lib_cell']:+.3e})  as-is {a:>22}  barrier-cell floor {b:>12}  "
              f"correction off {c:>12}", flush=True)
