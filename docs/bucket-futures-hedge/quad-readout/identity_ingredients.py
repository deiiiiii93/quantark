"""Which ingredient of the carry identity carries the residual?

    identity_residual = (delta_q - delta_f_direct - sum_i (F_i/S) B_i) / m_ref

`delta_q` is the engine's OWN delta greek, taken at the engine's BumpConfig
spot bump, so it must not move when the audit's bump changes -- but it must
move when the readout changes.  `delta_f_direct` is the audit's own central
difference, so it moves with both.  Comparing the runs I already have on
disk separates the two without pricing anything.
"""
import csv
import pathlib
import statistics

ROOT = pathlib.Path(
    "/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge/"
    "example/snowball_q_term_structure/data/bucket_hedge_v2"
)
CELL = "runs/20230504/term_flat_q__front/hedge_attribution.csv"

RUNS = {
    "legacy   bump=0.01 ": ROOT / "subset" / CELL,
    "transit  bump=0.01 ": ROOT / "subset_transition" / CELL,
    "transit  bump=0.001": ROOT / "ladder_tr_b001" / CELL,
}


def load(path):
    out = {}
    if not path.exists():
        return out
    for r in csv.DictReader(open(path)):
        row = {}
        for k in ("delta_f_derived_hands", "delta_f_direct_hands",
                  "identity_residual_hands", "product_delta_hands"):
            v = r.get(k)
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                row[k] = None
        out[r["date"]] = row
    return out


data = {name: load(p) for name, p in RUNS.items()}
for name, p in RUNS.items():
    print(f"{name}  rows={len(data[name]):4d}  {'MISSING' if not data[name] else ''}")

names = [n for n in RUNS if data[n]]
common = set.intersection(*(set(data[n]) for n in names)) if names else set()
print(f"\ncommon dates: {len(common)}\n")


def diff(a, b, key):
    vals = []
    for d in common:
        x, y = data[a][d][key], data[b][d][key]
        if x is not None and y is not None:
            vals.append(abs(x - y))
    if not vals:
        return None
    return statistics.fmean(vals), max(vals)


PAIRS = [
    ("legacy   bump=0.01 ", "transit  bump=0.01 ", "same bump, readout changes"),
    ("transit  bump=0.01 ", "transit  bump=0.001", "same readout, bump changes"),
]
for a, b, label in PAIRS:
    if a not in names or b not in names:
        print(f"-- {label}: run missing")
        continue
    print(f"-- {label}")
    for key in ("delta_f_derived_hands", "delta_f_direct_hands",
                "identity_residual_hands"):
        r = diff(a, b, key)
        if r:
            print(f"     {key:28s} mean|d|={r[0]:.6f}  max|d|={r[1]:.6f}")
    print()
