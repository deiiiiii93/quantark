"""Does the residual floor live entirely in the pre-knock-in leg?

The sampled rows collapse from ~0.03-0.09 hands to ~1e-4 in January 2024.
`states.csv` carries the lifecycle flag, so the split can be taken exactly
rather than by eye, and separated from "the product is simply near
maturity".
"""
import csv
import pathlib
import statistics

ROOT = pathlib.Path(
    "/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge/"
    "example/snowball_q_term_structure/data/bucket_hedge_v2"
)
CELL = "runs/20230504/term_flat_q__front"


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


for run in ("subset", "subset_transition", "ladder_tr_b001"):
    d = ROOT / run / CELL
    if not (d / "hedge_attribution.csv").exists():
        continue
    ki = {}
    spot = {}
    for r in csv.DictReader(open(d / "states.csv")):
        ki[r["date"]] = r.get("knocked_in", "").strip().lower() in ("true", "1")
        spot[r["date"]] = _f(r.get("spot"))

    before, after = [], []
    first_ki = None
    for r in csv.DictReader(open(d / "hedge_attribution.csv")):
        resid = _f(r["identity_residual_hands"])
        if resid is None:
            continue
        if ki.get(r["date"]):
            after.append(abs(resid))
            if first_ki is None:
                first_ki = r["date"]
        else:
            before.append(abs(resid))

    def stat(v):
        if not v:
            return "n=0"
        return (f"n={len(v):3d} mean={statistics.fmean(v):.6f} "
                f"med={statistics.median(v):.6f} max={max(v):.6f}")

    print(f"{run}")
    print(f"   first knocked-in date: {first_ki}  spot {spot.get(first_ki)}")
    print(f"   before KI  {stat(before)}")
    print(f"   after  KI  {stat(after)}")
    if before and after:
        print(f"   ratio of means: {statistics.fmean(before)/max(statistics.fmean(after), 1e-18):,.0f}x")
    print()
