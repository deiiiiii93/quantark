"""Gate E's economics, replayed with the barrier alignment on.

The fleet audit re-run cleared the identity failures but left Gate E's P&L
standing on the old engine. This replays all 406 cells with
align_cell_stretch=0.02 and the coupon RE-SOLVED under the same engine, then
compares the two fleets at the (inception, cell) level -- cell means can hide
offsetting per-inception moves, so the distribution of the paired difference
is reported, not just the difference of the means.

Both fleets go through the same committed aggregator. The banked arm keeps
its daily carry audit; the aligned arm runs with carry_audit_mode=none,
because the identity was already re-run over all 7,558 states and re-audited
at the re-solved coupon to the sixth decimal. That is why the aligned arm
reports audit_coverage="not_measured" and audits_passed=False on every row:
all_measured_passed is bool(measured and ...) by design, so a run that
measured nothing can never be reused as a passing gate. It is NOT 406 audit
failures.

Produced the "Gate E, replayed" section of ../gates.md.
"""
import importlib
import math
import os
import statistics
import sys
from pathlib import Path

REPO = Path("/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge")
STUDY = REPO / "example/snowball_q_term_structure"
os.environ["PYTHONPATH"] = str(REPO)
for _k in [k for k in list(sys.modules) if k == "quantark" or k.startswith("quantark.")]:
    del sys.modules[_k]
sys.path[:0] = [str(STUDY), str(REPO)]

import quantark

assert str(REPO) in quantark.__file__, quantark.__file__

import _common as C

agg = importlib.import_module("03_aggregate_and_report")

BANKED = STUDY / "data/bucket_hedge_v2/gate_e"
ALIGNED = REPO / "output/snowball_q_term_structure/bucket_hedge_v3"
MODELS = ("term_flat_fwd", "term_flat_q")
POLICIES = ("buckets_nodes", "buckets_far", "buckets_spot_parallel")
MEASURES = ("terminal_pnl_bp", "daily_pnl_std_bp", "delta_churn",
            "variance_reduction_r2")


def stars(desc):
    """The aggregator's own thresholds (03_aggregate_and_report._sig)."""
    t = desc.get("t_stat")
    if t is None:
        return ""
    return " ***" if abs(t) > 3.3 else " **" if abs(t) > 2.6 else " *" if abs(t) > 2.0 else ""


def load(run_dir):
    manifest, rows = agg.load_fleet(run_dir)
    return manifest, {(r["cell"], r["inception"]): r for r in rows}


def paired(index, variant, base, measure="terminal_pnl_bp"):
    """variant - base, one value per inception they share."""
    out = []
    for (cell, inception), row in index.items():
        if cell != variant:
            continue
        other = index.get((base, inception))
        if other is None:
            continue
        a, b = row.get(measure), other.get(measure)
        if a is None or b is None:
            continue
        a, b = float(a), float(b)
        if math.isfinite(a) and math.isfinite(b):
            out.append(a - b)
    return out


man_off, off = load(BANKED)
man_on, on = load(ALIGNED)
print(f"banked  {BANKED}\n        align={man_off['config'].get('align_cell_stretch')!r} "
      f"audit={man_off['config']['risk']['carry_audit_mode']} "
      f"{man_off['elapsed_seconds'] / 3600:.1f} h")
print(f"aligned {ALIGNED}\n        align={man_on['config'].get('align_cell_stretch')!r} "
      f"audit={man_on['config']['risk']['carry_audit_mode']} "
      f"{man_on['elapsed_seconds'] / 3600:.1f} h")

shared = sorted(set(off) & set(on))
print(f"\npaired rows: {len(shared)} of {len(off)} banked and {len(on)} aligned\n")

print("how far each cell-run moved")
print(f"{'measure':24} {'mean |d|':>10} {'p95':>10} {'max':>10}")
for measure in MEASURES:
    d = sorted(abs(float(on[k][measure]) - float(off[k][measure])) for k in shared
               if off[k].get(measure) is not None and on[k].get(measure) is not None)
    p95 = d[min(len(d) - 1, int(0.95 * len(d)))]
    print(f"{measure:24} {statistics.fmean(d):10.4f} {p95:10.4f} {max(d):10.4f}")

print("\nworst five cell-runs by |d terminal pnl|")
moves = sorted(shared, key=lambda k: -abs(float(on[k]["terminal_pnl_bp"])
                                          - float(off[k]["terminal_pnl_bp"])))
for cell, inception in moves[:5]:
    a = float(off[(cell, inception)]["terminal_pnl_bp"])
    b = float(on[(cell, inception)]["terminal_pnl_bp"])
    print(f"  {inception}  {cell:36s} {a:+9.1f} -> {b:+9.1f} bp  ({b - a:+.1f})")

print("\nthe Gate E table, both arms: terminal P&L difference in bp of notional")
print(f"{'model':15} {'policy':24} {'vs front off':>16} {'vs front on':>16} "
      f"{'vs far off':>16} {'vs far on':>16}")
for model in MODELS:
    for policy in POLICIES:
        cells = []
        for base in ("front", "far"):
            for index in (off, on):
                d = C.describe(paired(index, f"{model}__{policy}", f"{model}__{base}"))
                cells.append(f"{d['mean']:.1f}{stars(d)}")
        print(f"{model:15} {policy:24} {cells[0]:>16} {cells[1]:>16} "
              f"{cells[2]:>16} {cells[3]:>16}")

print("\nthe hedge-contract pair, both arms")
for model in MODELS:
    d_off = C.describe(paired(off, f"{model}__far", f"{model}__front"))
    d_on = C.describe(paired(on, f"{model}__far", f"{model}__front"))
    print(f"  {model:15} far - front  {d_off['mean']:8.1f}{stars(d_off):4} -> "
          f"{d_on['mean']:8.1f}{stars(d_on):4}")

print("\naudit columns, as recorded")
for label, index in (("banked", off), ("aligned", on)):
    cov = {}
    passed = {}
    for k in index:
        cov[index[k].get("audit_coverage")] = cov.get(index[k].get("audit_coverage"), 0) + 1
        passed[index[k].get("audits_passed")] = passed.get(index[k].get("audits_passed"), 0) + 1
    print(f"  {label:8} coverage {cov}  passed {passed}")
