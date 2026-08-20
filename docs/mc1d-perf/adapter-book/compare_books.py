"""Compare two book_probe.py dumps: exact field equality + timing."""

import json
import sys
from pathlib import Path

NUMERIC_FIELDS = [
    "pv", "pv_total", "pv_premium", "pv_interest", "pv_rebate",
    "pv_margin", "pv_minimum_return",
    "delta", "gamma", "vega", "theta", "rho", "rhoQ",
    "delta_total", "gamma_total", "vega_total", "theta_total",
    "rho_total", "rhoQ_total",
]

a = json.loads(Path(sys.argv[1]).read_text())
b = json.loads(Path(sys.argv[2]).read_text())

print(f"A: {a['label']:12} model={a['model']} gbm={a['gbm_backend']:8} "
      f"total={a['total_seconds']:.2f}s")
print(f"B: {b['label']:12} model={b['model']} gbm={b['gbm_backend']:8} "
      f"total={b['total_seconds']:.2f}s")
print()

ids_a, ids_b = set(a["rows"]), set(b["rows"])
if ids_a != ids_b:
    print(f"!! membership differs: only-A={ids_a - ids_b} only-B={ids_b - ids_a}")

mismatches = []
status_diffs = []
compared = 0
for tid in sorted(ids_a & ids_b):
    ra, rb = a["rows"][tid], b["rows"][tid]
    if ra["pricing_status"] != rb["pricing_status"]:
        status_diffs.append(f"{tid}: {ra['pricing_status']} -> {rb['pricing_status']}")
    for f in NUMERIC_FIELDS:
        va, vb = ra.get(f), rb.get(f)
        if va is None and vb is None:
            continue
        compared += 1
        if va != vb:   # hex strings: exact bit comparison
            fa = float.fromhex(va) if va else float("nan")
            fb = float.fromhex(vb) if vb else float("nan")
            rel = abs(fb - fa) / max(abs(fa), 1e-300) if fa == fa else float("nan")
            mismatches.append(f"{tid} {f}: {fa!r} -> {fb!r}  (rel {rel:.3e})")

print(f"rows compared      : {len(ids_a & ids_b)}")
print(f"numeric fields cmp : {compared}")
print(f"pricing_status diff: {len(status_diffs)}")
for s in status_diffs[:10]:
    print(f"    {s}")
print(f"BIT-EXACT mismatches: {len(mismatches)}")
for m in mismatches[:20]:
    print(f"    {m}")

if not mismatches and not status_diffs:
    print("\nRESULT: every field bit-identical across the two trees.")

# timing: per-structure aggregation
print("\nper-structure wall time (sum of row seconds):")
agg = {}
for tid in sorted(ids_a & ids_b):
    s = a["rows"][tid].get("structure", "?")
    ta, tb = a["rows"][tid]["seconds"], b["rows"][tid]["seconds"]
    cur = agg.setdefault(s, [0.0, 0.0, 0])
    cur[0] += ta
    cur[1] += tb
    cur[2] += 1
print(f"  {'structure':28} {'n':>3} {'A (s)':>9} {'B (s)':>9} {'speedup':>8}")
for s, (ta, tb, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
    sp = ta / tb if tb > 0 else float("nan")
    print(f"  {s[:28]:28} {n:3d} {ta:9.2f} {tb:9.2f} {sp:7.2f}x")
tot_a = sum(v[0] for v in agg.values())
tot_b = sum(v[1] for v in agg.values())
print(f"  {'TOTAL':28} {len(ids_a & ids_b):3d} {tot_a:9.2f} {tot_b:9.2f} "
      f"{tot_a / tot_b if tot_b else float('nan'):7.2f}x")
