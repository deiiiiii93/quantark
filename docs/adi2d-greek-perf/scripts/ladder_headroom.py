"""Per-axis floor headroom from the finished Heston certification checkpoints.

For each cell and each grid axis, print the delta/gamma at every ladder row as
an error vs the paired RQMC reference IN CONTRACTS, plus the solve seconds.
If the coarse row already sits far inside the 0.5-contract cell bound, the
production floor on that axis is paying for accuracy nobody consumes.
"""
from __future__ import annotations

import glob
import json
import os

CKPT = "/private/tmp/quant-ark-adi-greek-certification/output/adi_greek_certification/checkpoints"
BOUND = 0.5  # DELTA_CELL_BOUND_CONTRACTS / GAMMA_CELL_BOUND_CONTRACTS

ORDER = ["ordinary_full", "ordinary_decayed", "near_ko", "near_ki",
         "low_feller", "sigma_collapse", "near_expiry"]

for name in ORDER:
    path = os.path.join(CKPT, f"heston__{name}.json")
    if not os.path.exists(path):
        continue
    doc = json.load(open(path))
    ev = doc["evidence"]
    quantum = ev["economic_scale"]["delta_quantum_per_contract"]
    ref_d = ev["certifications"]["delta"]["reference"]
    ref_g = ev["certifications"]["gamma"]["reference"]
    ladders = ev["pde_ladders"]["axes"]
    print(f"=== {name}  (ref delta {ref_d:+.6f}, gamma {ref_g:+.6f}; quantum {quantum:.6f}) ===")
    for axis in ("n_x", "n_v", "n_t"):
        rows = ladders[axis]
        print(f"  axis {axis}:")
        for row in rows:
            keys = {k: row[k] for k in ("n_x", "n_v", "n_t") if k in row}
            d = row.get("delta"); g = row.get("gamma"); secs = row.get("seconds")
            ed = (d - ref_d) / quantum if d is not None else float("nan")
            eg = (g - ref_g) / quantum if g is not None else float("nan")
            tag = ""
            if abs(ed) < 0.5 * BOUND and abs(eg) < 0.5 * BOUND:
                tag = "  <- passes with >2x margin"
            print(f"    {keys}  dErr={ed:+7.4f}c  gErr={eg:+7.4f}c"
                  f"  ({secs:.1f}s){tag}" if secs is not None else
                  f"    {keys}  dErr={ed:+7.4f}c  gErr={eg:+7.4f}c{tag}")
    print()
