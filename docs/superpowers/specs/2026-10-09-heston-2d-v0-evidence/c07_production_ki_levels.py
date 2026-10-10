"""C07 (Q3): does the production profile land in the divergent regime on another KI level?

Reverse continuous-KI Heston snowball KO 97, KI from 110 to 140 in steps of 0.05: the live node nearest the barrier,
in cells, on the production PV grid (n_x 200) and the production Greek grid (n_x 300; the 600 barrier floor does not
apply under continuous KI, C05). For the KI level nearest a node on each, price through the production path (PV on
(200, 60, 800), ``calculate_greeks`` on its resolved (300, 135, 3200)) as-is and with the half-cell floor measured at
the barrier (C02).
"""
import numpy as np

from c02_q3_common import REV_C, barrier_cell_floor, barrier_geometry, px
from c05_production_and_inertness import MEDIUM, PROD, fmt
from common import HP, env, snowball, solver


def greeks(spec):
    try:
        g = solver(HP, *MEDIUM, **PROD).calculate_greeks(snowball(**spec), env())
        return "  ".join(f"{k} {fmt(float(v))}" for k, v in g.items())
    except Exception as exc:  # noqa: BLE001 - probe
        return f"{type(exc).__name__}: {str(exc)[:100]}"


if __name__ == "__main__":
    levels = np.round(np.arange(110.0, 140.0001, 0.05), 2)
    worst = {}
    for n_x in (200, 300):
        cells = [(barrier_geometry((n_x, 60, 800), spec=dict(REV_C, ki=float(k)), **PROD)["cells"], float(k))
                 for k in levels]
        cells.sort()
        print(f"n_x={n_x}: {sum(c < 0.02 for c, _ in cells)} of {len(cells)} KI levels within 0.02 cells; nearest five "
              + ", ".join(f"KI {k:.2f} at {c:.4f}" for c, k in cells[:5]), flush=True)
        worst[n_x] = cells[0][1]
    for n_x, ki in worst.items():
        spec = dict(REV_C, ki=ki)
        a_pv, a_g = px(MEDIUM, spec=spec, **PROD), greeks(spec)
        with barrier_cell_floor():
            b_pv, b_g = px(MEDIUM, spec=spec, **PROD), greeks(spec)
        print(f"KI {ki:.2f} (nearest on n_x={n_x}): as-is PV {a_pv}; Greeks {a_g}", flush=True)
        print(f"KI {ki:.2f} (nearest on n_x={n_x}): floor PV {b_pv}; Greeks {b_g}", flush=True)
