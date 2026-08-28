"""Reconcile FINDING-2026-08-26's reported surface slopes.

The FINDING records dsigma/dlnS = -0.371, dsigma/dt = -0.082 for 2024-02-08.
Direct measurement gives -0.056 / -0.269 on the Dupire surface and
-0.031 / -0.081 on the implied surface. The implied TERM slope reproduces almost
exactly while the skew is off by an order of magnitude, which points to a
definitional mismatch -- which surface, which slice, which moneyness window --
rather than a data problem.

This does not change the case list: 2024-02-08 is unambiguously the worst cell
empirically (-1.2726 contracts, -2.88 sigma). But "bias scales with surface
steepness" is the mechanism the calm-surface contrast is selected on, so the
definition has to be pinned down or explicitly recorded as unreconciled.

Run:
  PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
    docs/modelvalidation/pilot-localvol-1d/probe_steepness.py
"""

import numpy as np

from quantark.modelvalidation.builders.equity_snowball_localvol import load_surface

TARGET_SKEW, TARGET_TERM = -0.371, -0.082
PATH = "example/modelvalidation/data/iv_surface_20240208.json"

surface = load_surface(PATH, 0.02)
art, grid, lv = surface.artifact, surface.grid, surface.local_vol
s0 = float(art.s0)
K = np.array(art.strikes, float)
T = np.array(art.maturities, float)
G = np.array(art.iv_grid, float)

print(f"surface {art.trade_date}  s0={s0:.3f}  max_listed_T={art.max_listed_T:.4f}")
print(f"FINDING targets:  dsigma/dlnS = {TARGET_SKEW}   dsigma/dt = {TARGET_TERM}")
print()

# ---- skew, swept over definition space --------------------------------------
rows = []
slices = (0.02, 0.05, 0.10, 0.25, 0.5, 0.75, float(art.max_listed_T), 1.0)
for which in ("implied", "dupire"):
    for slice_t in slices:
        for width in (0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50):
            x = np.linspace(-width, width, 21)
            if which == "dupire":
                sig = np.array([lv.local_vol(s0 * np.exp(xi), slice_t) for xi in x])
            else:
                j = int(np.argmin(np.abs(T - slice_t)))
                xs = np.log(K / s0)
                m = np.abs(xs) <= width
                if m.sum() < 3:
                    continue
                sig = np.interp(x, xs[m], G[j, m])
            if not np.all(np.isfinite(sig)):
                continue
            skew = float(np.polyfit(x, sig, 1)[0])
            rows.append((which, slice_t, width, skew, abs(skew - TARGET_SKEW)))

rows.sort(key=lambda r: r[4])
print("SKEW -- closest 15 definitions to the FINDING's -0.371")
print(f"{'surface':>9} {'slice_t':>8} {'width':>6} {'dsig/dlnS':>11} {'|err|':>8}")
print("-" * 48)
for which, slice_t, width, skew, err in rows[:15]:
    print(f"{which:>9} {slice_t:8.3f} {width:6.2f} {skew:11.4f} {err:8.4f}")

best = rows[0]
print()
print(f"best skew match: {best[0]} @ t={best[1]:.3f}, width={best[2]:.2f} "
      f"-> {best[3]:.4f}  (err {best[4]:.4f})")

# ---- term slope, swept the same way -----------------------------------------
term_rows = []
for which in ("implied", "dupire"):
    for lo, hi in ((0.02, 0.87), (0.05, 0.87), (0.10, 0.87), (0.25, 0.87),
                   (0.02, 0.50), (0.02, 1.00), (0.05, 1.00)):
        ts = np.linspace(lo, hi, 12)
        if which == "dupire":
            atm = np.array([lv.local_vol(s0, t) for t in ts])
        else:
            atm = np.array([
                np.interp(s0, K, G[int(np.argmin(np.abs(T - t)))]) for t in ts
            ])
        if not np.all(np.isfinite(atm)):
            continue
        slope = float(np.polyfit(ts, atm, 1)[0])
        term_rows.append((which, lo, hi, slope, abs(slope - TARGET_TERM)))

term_rows.sort(key=lambda r: r[4])
print()
print("TERM -- closest 10 definitions to the FINDING's -0.082")
print(f"{'surface':>9} {'from':>6} {'to':>6} {'dsig/dt':>10} {'|err|':>8}")
print("-" * 44)
for which, lo, hi, slope, err in term_rows[:10]:
    print(f"{which:>9} {lo:6.2f} {hi:6.2f} {slope:10.4f} {err:8.4f}")

print()
print("VERDICT")
if best[4] < 0.02:
    print(f"  RECONCILED: skew definition is {best[0]} @ t={best[1]:.3f}, "
          f"width={best[2]:.2f}")
else:
    print(f"  UNRECONCILED: no definition lands within 0.02 of {TARGET_SKEW}; "
          f"closest is {best[3]:.4f} ({best[0]}, t={best[1]:.3f}, w={best[2]:.2f})")
    print("  Record as UNRECONCILED in RESULTS.md and justify the crash/calm")
    print("  contrast by the EMPIRICAL per-cell gaps (-1.2726 vs +0.2614)")
    print("  rather than by a slope metric. Do not silently adopt either number.")
