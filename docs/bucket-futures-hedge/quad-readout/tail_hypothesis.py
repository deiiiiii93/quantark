"""Is the residual floor the part of the product that outlives the curve?

    D|_q = D|_F + sum_i (F_i/S) B_i

sums over LISTED contracts only. A 1Y snowball outlives a futures strip that
runs a few months, so the pricing q beyond the last listed node comes from
the extrapolation. If moving spot at a frozen curve also moves that
extrapolated tail in a way no listed bucket carries, the identity is
structurally short a term, and the residual should track how much product
life sits beyond the last contract.
"""
import csv
import datetime as dt
import pathlib
import statistics

ROOT = pathlib.Path(
    "/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge/"
    "example/snowball_q_term_structure/data/bucket_hedge_v2/subset_transition/"
    "runs/20230504/term_flat_q__front"
)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


legs: dict[str, dict] = {}
for r in csv.DictReader(open(ROOT / "hedge_legs.csv")):
    d = legs.setdefault(r["date"], {"max_tenor": 0.0, "tail": set(), "n": 0})
    t = _f(r.get("tenor_years"))
    if t is not None:
        d["max_tenor"] = max(d["max_tenor"], t)
    d["tail"].add(r.get("has_product_tail"))
    d["n"] += 1

rows = []
for r in csv.DictReader(open(ROOT / "hedge_attribution.csv")):
    resid = _f(r["identity_residual_hands"])
    if resid is None or r["date"] not in legs:
        continue
    rows.append((r["date"], abs(resid), legs[r["date"]]))

# remaining product life: the run is 243 dates from 2023-05-04 to maturity
last = dt.date.fromisoformat(rows[-1][0])
print(f"{'date':12s} {'|resid|':>9s} {'legs':>5s} {'max leg T':>10s} "
      f"{'life left':>10s} {'uncovered':>10s} {'tail flags':>12s}")
sample = rows[::20] + rows[-3:]
for date, resid, d in sample:
    life = (last - dt.date.fromisoformat(date)).days / 365.0
    print(f"{date:12s} {resid:9.4f} {d['n']:5d} {d['max_tenor']:10.4f} "
          f"{life:10.4f} {life - d['max_tenor']:10.4f} "
          f"{','.join(sorted(str(x) for x in d['tail'])):>12s}")

# does the residual track the uncovered span?
pairs = []
for date, resid, d in rows:
    life = (last - dt.date.fromisoformat(date)).days / 365.0
    pairs.append((life - d["max_tenor"], resid))
covered = [r for u, r in pairs if u <= 0]
uncovered = [r for u, r in pairs if u > 0]
print(f"\ncurve spans the product ({len(covered):3d} dates): "
      f"mean |resid| = {statistics.fmean(covered) if covered else float('nan'):.6f}")
print(f"product outlives it  ({len(uncovered):3d} dates): "
      f"mean |resid| = {statistics.fmean(uncovered) if uncovered else float('nan'):.6f}")
if len(pairs) > 2:
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    print(f"correlation(uncovered span, |resid|) = "
          f"{statistics.correlation(xs, ys):+.4f}")
