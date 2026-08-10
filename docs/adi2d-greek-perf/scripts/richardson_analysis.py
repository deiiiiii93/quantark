"""Is Richardson extrapolation viable on this product family? Test from banked ladders.

For each Heston cell and axis, take the three ladder rows (coarse/target/fine,
one axis varying), fit the observed convergence order p from successive
differences, and compare:
    err(target row)                     what production pays for today
    err(fine row)                       the expensive row
    err(Richardson, two finest, fit p)  extrapolant accuracy
    err(coarse+target pair, p=2)        the practical cheap recipe
all in hedge contracts vs the banked MC reference. Extrapolation is only
valid when the approach is monotone (D1, D2 same sign); oscillatory axes are
flagged — that is the event-kink failure mode.
"""
from __future__ import annotations

import json

import numpy as np

CKPT = "/private/tmp/quant-ark-adi-greek-certification/output/adi_greek_certification/checkpoints"
ORDER = ["ordinary_full", "ordinary_decayed", "near_ko", "near_ki",
         "low_feller", "sigma_collapse", "near_expiry"]


def fit_p(h, d_ratio):
    """Solve (h2^p - h1^p)/(h1^p - h0^p) = d_ratio for p by grid search."""
    ps = np.linspace(0.2, 4.0, 3801)
    h0, h1, h2 = h
    model = (h2 ** ps - h1 ** ps) / (h1 ** ps - h0 ** ps)
    return float(ps[np.argmin(np.abs(model - d_ratio))])


for name in ORDER:
    doc = json.load(open(f"{CKPT}/heston__{name}.json"))
    ev = doc["evidence"]
    q = ev["economic_scale"]["delta_quantum_per_contract"]
    ref = ev["certifications"]["delta"]["reference"]
    se_c = ev["certifications"]["delta"]["reference_standard_error"] / q
    print(f"=== {name}   (reference SE = {se_c:.4f} contracts) ===")
    for axis in ("n_x", "n_v", "n_t"):
        rows = sorted(ev["pde_ladders"]["axes"][axis], key=lambda r: r["grid"][axis])
        n = [r["grid"][axis] for r in rows]
        d = [r["delta"] for r in rows]
        e = [(x - ref) / q for x in d]
        D1, D2 = d[1] - d[0], d[2] - d[1]
        line = (f"  {axis}: n={n}  err(c/t/f)=({e[0]:+.4f}, {e[1]:+.4f}, {e[2]:+.4f})c")
        if D1 == 0.0 or D2 == 0.0 or np.sign(D1) != np.sign(D2):
            print(line + "   OSCILLATORY/FLAT -> Richardson invalid")
            continue
        if abs(D1) / q < 0.5 * se_c and abs(D2) / q < 0.5 * se_c:
            print(line + "   moves below reference noise -> nothing to extrapolate")
            continue
        h = tuple(1.0 / x for x in n)
        p = fit_p(h, D2 / D1)
        # two-finest extrapolant with fitted p
        u_inf = d[2] + D2 * (h[2] ** p) / (h[1] ** p - h[2] ** p)
        e_inf = (u_inf - ref) / q
        # practical pair recipe: coarse+target, assume p=2
        r01 = (n[1] / n[0]) ** 2
        u_ct = d[1] + D1 / (r01 - 1.0)
        e_ct = (u_ct - ref) / q
        print(line + f"   fit p={p:.2f}  extrap(fit)={e_inf:+.4f}c  "
                     f"pair(c+t,p=2)={e_ct:+.4f}c")
    print()
