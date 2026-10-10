"""W10b: per observation-date cell, the PDE routes before / after against the MC (4096 Sobol paths, unchanged).

    python compare_obsdate.py base_0d1e4133.txt after_w10b.txt
"""
from __future__ import annotations

import sys

from summarize import load


def main(old_path, new_path):
    old, new = load(old_path), load(new_path)
    cells = sorted({key.rsplit(" ", 1)[0] for (section, key) in new if section == "obsdate"})
    print("cell | MC = GBM DCNMCEngine, 4096 Sobol paths (the Heston PDE differs from it by the model) | "
          "route: PDE before -> after (after - MC)")
    for cell in cells:
        mc = float(new[("obsdate", f"{cell} DCNMCEngine.price")])
        parts = []
        for route in ("DCNPDEEngine", "LocalVolDCNPDEEngine", "HestonDCNPDESolver"):
            key = ("obsdate", f"{cell} {route}.price")
            a, b = float(old[key]), float(new[key])
            parts.append(f"{route}: {a:.2f} -> {b:.2f} ({b - mc:+.2f})")
        print(f"{cell} | MC {mc:.2f} | " + " | ".join(parts))


if __name__ == "__main__":
    main(*sys.argv[1:3])
