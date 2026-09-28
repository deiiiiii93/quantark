"""How much distinct work is a fleet audit re-run, really?

The carry identity is holdings-free: it compares three spot derivatives of
the PRICE, and no hedge quantity enters. So two cells that differ only in
hedge policy audit the same state and must produce the same residual. What
actually has to be re-priced is the set of distinct (inception, model,
date) states, not the 406 cells.

Checks that claim against the banked Gate E output before relying on it.
"""
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd

RUNS = Path("/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge"
            "/example/snowball_q_term_structure/data/bucket_hedge_v2/gate_e/runs")

cells = []
for inc in sorted(p for p in RUNS.iterdir() if p.is_dir()):
    for cell in sorted(p for p in inc.iterdir() if p.is_dir()):
        if (cell / "hedge_attribution.csv").exists():
            model, policy = cell.name.split("__", 1)
            cells.append((inc.name, model, policy, cell))

print(f"{len(cells)} cells, {len({c[0] for c in cells})} inceptions, "
      f"models {sorted({c[1] for c in cells})}, "
      f"policies {sorted({c[2] for c in cells})}")

# configs that could change what the audit prices
configs = defaultdict(set)
for inc, model, policy, cell in cells:
    cfg = json.loads((cell / "run_config.json").read_text())
    for key in ("quad_grid", "quad_readout", "extrapolation", "dividend_source"):
        configs[key].add(json.dumps(cfg.get(key)))
for key, values in configs.items():
    print(f"  {key}: {sorted(values)}")

# does the residual actually depend only on (inception, model, date)?
print("\nchecking the identity really is holdings-free ...")
groups = defaultdict(dict)
rows_total = 0
for inc, model, policy, cell in cells:
    frame = pd.read_csv(cell / "hedge_attribution.csv")
    if "identity_residual_hands" not in frame.columns:
        continue
    series = frame.set_index("date")["identity_residual_hands"]
    rows_total += int(series.notna().sum())
    groups[(inc, model)][policy] = series

mismatch = 0
compared = 0
for key, by_policy in groups.items():
    policies = sorted(by_policy)
    ref = by_policy[policies[0]]
    for other in policies[1:]:
        joined = pd.concat([ref, by_policy[other]], axis=1, join="inner").dropna()
        compared += len(joined)
        differ = (joined.iloc[:, 0] - joined.iloc[:, 1]).abs() > 1e-12
        mismatch += int(differ.sum())

print(f"  compared {compared} paired rows across policies: {mismatch} differ")
distinct = sum(len(next(iter(v.values()))) for v in groups.values())
print(f"\naudit rows banked across all cells : {rows_total}")
print(f"distinct (inception, model, date)  : {distinct}")
print(f"work saved                          : {rows_total / max(distinct, 1):.1f}x")
