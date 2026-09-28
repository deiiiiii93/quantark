"""Gate E's actual question: does the bucket hedge beat one contract?

Paired by inception, since every cell of an inception shares the contract,
the spot path, the vol channel, the rate and the cost model. The pairing is
what makes the difference attributable to the hedge policy.

Reports terminal P&L, daily tracking error, turnover and realised costs, so
a P&L gap can be attributed rather than just observed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge"
            "/example/snowball_q_term_structure/data/bucket_hedge_v2/gate_e/runs")
NOTIONAL = 50_000_000.0

rows = []
for inc in sorted(p for p in RUNS.iterdir() if p.is_dir()):
    for cell in sorted(p for p in inc.iterdir() if p.is_dir()):
        f = cell / "run_summary.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        st = pd.read_csv(cell / "states.csv")
        pnl_col = "total_pnl" if "total_pnl" in st.columns else None
        daily = st[pnl_col].diff().std() if pnl_col else np.nan
        rows.append({
            "inception": inc.name, "cell": cell.name,
            "model": cell.name.split("__")[0], "policy": cell.name.split("__")[1],
            "pnl_bp": s["final_total_pnl"] / NOTIONAL * 1e4,
            "cost_bp": s.get("transaction_costs", np.nan) / NOTIONAL * 1e4,
            "days": s.get("days_replayed", np.nan),
            "daily_bp": (daily / NOTIONAL * 1e4) if daily == daily else np.nan,
        })
f = pd.DataFrame(rows)
print(f"{len(f)} cells, {f.inception.nunique()} inceptions\n")

CONTROLS = ("front", "far")
BUCKETS = ("buckets_nodes", "buckets_far", "buckets_spot_parallel")
print("PAIRED BY INCEPTION: bucket policy minus its single-contract control")
print(f"{'model':14} {'bucket policy':22} {'vs':6} {'d pnl bp':>10} {'t':>7} "
      f"{'d cost bp':>10} {'d daily bp':>11}")
for model in sorted(f.model.unique()):
    m = f[f.model == model]
    for b in BUCKETS:
        for c in CONTROLS:
            x = m[m.policy == b].set_index("inception")
            y = m[m.policy == c].set_index("inception")
            common = x.index.intersection(y.index)
            if len(common) < 5:
                continue
            d = x.loc[common, "pnl_bp"] - y.loc[common, "pnl_bp"]
            dc = x.loc[common, "cost_bp"] - y.loc[common, "cost_bp"]
            dd = x.loc[common, "daily_bp"] - y.loc[common, "daily_bp"]
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 0 else np.nan
            star = "***" if abs(t) > 2.76 else "** " if abs(t) > 2.05 else "   "
            print(f"{model:14} {b:22} {c:6} {d.mean():10.1f} {t:6.2f}{star} "
                  f"{dc.mean():10.2f} {dd.mean():11.2f}")

print("\nrealised transaction cost by policy, bp of notional:")
print(f.groupby("policy").cost_bp.agg(["mean", "max"]).round(2).to_string())
print("\nterminal P&L by policy, bp:")
print(f.groupby("policy").pnl_bp.agg(["mean", "std", "min", "max"]).round(1).to_string())
