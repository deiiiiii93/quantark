"""Gate E: do the two single-inception numbers survive 29 inceptions?

Both were measured on 2023-05-04 alone and gates.md says so:

  1. the audit tolerance's margin over the noise floor. 0.01 hands sat 2.3x
     above the worst abs(R)+E seen. If that worst grows past ~0.005 on other
     inceptions, the margin is gone and the tolerance needs revisiting.
  2. the 1%-vs-0.25% secant decision. The 1% pricing bump was kept because
     its error against the desk's own hedge secant never reached half a
     contract. That was 242 dates of one inception.

Reads only what the run already recorded. Prices nothing.
"""
from pathlib import Path
import sys

import pandas as pd

RUNS = Path("/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge"
            "/example/snowball_q_term_structure/data/bucket_hedge_v2/gate_e/runs")
TOLERANCE = 0.01

frames = []
for inception in sorted(p for p in RUNS.iterdir() if p.is_dir()):
    for cell in sorted(p for p in inception.iterdir() if p.is_dir()):
        f = cell / "hedge_attribution.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d["inception"], d["cell"] = inception.name, cell.name
        d["model"] = cell.name.split("__")[0]
        frames.append(d)

if not frames:
    sys.exit("no cells yet")
a = pd.concat(frames, ignore_index=True)
cells = a.groupby(["inception", "cell"]).ngroups
print(f"{cells} cells, {len(a)} date-rows, {a.inception.nunique()} inceptions\n")

# --- 1. does the tolerance margin survive? --------------------------------
a["R"] = a.identity_residual_hands.abs()
a["E"] = a.identity_spot_refinement_error_hands.abs()
a["gated"] = a.R + a.E
m = a[a.R.notna()]
print("1. TOLERANCE MARGIN  (0.01-hand budget against the worst correct-book residual)")
print(f"   max abs(R)      {m.R.max():.6f}")
print(f"   max abs(R)+E    {m.gated.max():.6f}   <- the noise floor")
print(f"   margin at 0.01  {TOLERANCE / m.gated.max():.2f}x   (was 2.3x on one inception)")
print(f"   identity_status: {m.identity_status.value_counts().to_dict()}")
worst = m.loc[m.gated.idxmax()]
print(f"   worst: {worst.inception}/{worst.cell} on {worst.date}, "
      f"R={worst.R:.6f} E={worst.E:.6f}")
by_inc = m.groupby("inception").gated.max().sort_values(ascending=False)
print(f"   worst 5 inceptions: {', '.join(f'{i} {v:.5f}' for i, v in by_inc.head(5).items())}")

# --- 2. does the secant decision survive? ---------------------------------
h = a[a.hedge_gap_status == "measured"].copy()
h["gap"] = h.pricing_delta_hedge_gap_hands.abs()
print(f"\n2. 1% BUMP vs THE 0.25% HEDGE SECANT  ({len(h)} measured date-rows)")
print(f"   mean {h.gap.mean():.4f}   p95 {h.gap.quantile(.95):.4f}   max {h.gap.max():.4f} hands")
print(f"   dates over 0.5 contracts: {int((h.gap > .5).sum())}")
print(f"   dates over 1.0 contracts: {int((h.gap > 1).sum())}")
w = h.loc[h.gap.idxmax()]
print(f"   worst: {w.inception}/{w.cell} on {w.date}, gap {w.gap:.4f}")
print("   worst 5 inceptions:")
for i, v in h.groupby("inception").gap.max().sort_values(ascending=False).head(5).items():
    print(f"     {i}  {v:.4f}")

# --- 3. the audit itself, across the whole grid ----------------------------
print(f"\n3. AUDIT VERDICTS over {cells} cells")
print(f"   {a.audit_status.value_counts().to_dict()}")
print(f"   net delta audit error, max {a.net_delta_audit_error_hands.abs().max():.3e} hands")
