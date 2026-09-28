"""Falsifiable test of the alignment-crossover root cause.

The grid snaps to the NEAREST barrier in log space, so the alignment target
switches where spot is equidistant between them, at the geometric mean

    crossover = sqrt(KI * KO) = s0 * sqrt(0.75 * 1.03) = 0.878919 * s0

Prediction: residual and hedge-gap spikes occur ONLY where spot is within
about one audit bump of that crossover, and only while BOTH barriers are
live, i.e. before knock-in. Away from it, both should be quiet.

If large residuals appear far from the crossover, the explanation is wrong.
"""
import json
from pathlib import Path

import pandas as pd

RUNS = Path("/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge"
            "/example/snowball_q_term_structure/data/bucket_hedge_v2/gate_e/runs")
CROSS = (0.75 * 1.03) ** 0.5

frames = []
for inc in sorted(p for p in RUNS.iterdir() if p.is_dir()):
    for cell in sorted(p for p in inc.iterdir() if p.is_dir()):
        att, sts, summ = (cell / "hedge_attribution.csv", cell / "states.csv",
                          cell / "run_summary.json")
        if not (att.exists() and sts.exists() and summ.exists()):
            continue
        d = pd.read_csv(att)
        s = pd.read_csv(sts).set_index("date")
        s0 = json.loads(summ.read_text())["s0"]
        d["spot"] = d.date.map(s.spot)
        d["knocked_in"] = d.date.map(s.knocked_in).astype("boolean")
        # distance from the alignment crossover, in units of the audit bump
        d["dist_bumps"] = (d.spot / (CROSS * s0) - 1.0).abs() / 0.00025
        d["inception"], d["cell"] = inc.name, cell.name
        frames.append(d)

a = pd.concat(frames, ignore_index=True)
a["R"] = a.identity_residual_hands.abs()
a["gap"] = a.pricing_delta_hedge_gap_hands.abs()
live = a[a.R.notna() & (a.knocked_in == False)].copy()   # noqa: E712
print(f"{a.groupby(['inception','cell']).ngroups} cells; "
      f"{len(live)} pre-knock-in date-rows (both barriers live)\n")

bands = [(0, 2), (2, 10), (10, 50), (50, 200), (200, 1e9)]
print(f"{'distance from crossover':28} {'rows':>7} {'max R':>10} {'max gap':>10} {'R>0.01':>7}")
for lo, hi in bands:
    b = live[(live.dist_bumps >= lo) & (live.dist_bumps < hi)]
    if b.empty:
        continue
    label = f"{lo:.0f}-{hi:.0f} bumps" if hi < 1e9 else f">{lo:.0f} bumps"
    print(f"{label:28} {len(b):7d} {b.R.max():10.6f} {b.gap.max():10.4f} "
          f"{int((b.R > 0.01).sum()):7d}")

print("\nthe worst residuals, and how far each sits from the crossover:")
w = live.nlargest(6, "R")[["inception", "cell", "date", "spot", "dist_bumps", "R", "gap"]]
print(w.to_string(index=False))

print("\nthe worst hedge gaps:")
w = live.nlargest(6, "gap")[["inception", "cell", "date", "spot", "dist_bumps", "R", "gap"]]
print(w.to_string(index=False))

post = a[a.R.notna() & (a.knocked_in == True)]   # noqa: E712
print(f"\ncontrol, AFTER knock-in ({len(post)} rows, KI barrier gone so no crossover):")
print(f"  max R {post.R.max():.6f}   max gap {post.gap.max():.4f}")
