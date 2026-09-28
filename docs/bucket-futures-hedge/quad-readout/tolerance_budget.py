"""Derive a numerical error budget for the carry identity, per date.

The residual is R = [e_D - e_DF - sum_i (F_i/S) e_i] / m_ref, so a budget is
the SUM of the component magnitudes, not their difference. Each spot
component is estimated from its own ladder: for an h^2 estimator,
D(h) = D_true + C h^2, so D(2h) - D(h) = 3 C h^2 and the error at the finest
bump is |D(2h) - D(h)| / 3.

The bucket component is NOT in the ladder, because the quote bump is held at
+/-1 point by design. It is measured separately by quote-bump ladder and
passed in here as a constant.

Nothing is re-priced: the three levels were recorded by the production run.
"""
import json
from pathlib import Path

import pandas as pd

B = Path("/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge"
         "/example/snowball_q_term_structure/data/bucket_hedge_v2")
RUNS = B / "subset_matched/runs/20230504"
M_REF, NOTIONAL = 200.0, 50_000_000.0
# Measured at 2024-01-19 by Richardson on +/-1 and +/-0.5 point quote bumps.
E_BUCKET_HANDS = 4.86e-05


def components(row):
    """(E_D, E_DF) at the finest bump, in hands, from the recorded ladder."""
    lad = json.loads(row)
    if len(lad) < 2:
        return float("nan"), float("nan")
    fine, coarse = lad[-1], lad[-2]
    return (abs(fine["delta_q_hands"] - coarse["delta_q_hands"]) / 3.0,
            abs(fine["delta_f_hands"] - coarse["delta_f_hands"]) / 3.0)


for curve, cell in (("flat zero q", "term_flat_q__front"),
                    ("flat forward carry", "term_flat_fwd__front")):
    d = pd.read_csv(RUNS / cell / "hedge_attribution.csv")
    d = d[d["identity_ladder"].notna() & (d["identity_ladder"] != "[]")].copy()
    d[["E_D", "E_DF"]] = d["identity_ladder"].apply(
        lambda s: pd.Series(components(s)))
    d["budget"] = d["E_D"] + d["E_DF"] + E_BUCKET_HANDS
    d["absR"] = d["identity_residual_hands"].abs()
    d["E_obs"] = d["identity_spot_refinement_error_hands"].abs()
    ok = d["absR"] <= d["budget"]

    print(f"\n=== {curve} ({cell}, {len(d)} dates) ===")
    print(f"{'quantity':34} {'mean':>12} {'p95':>12} {'max':>12}")
    for name, col in (("E_D, frozen-curve delta", "E_D"),
                      ("E_DF, pinned-futures delta", "E_DF"),
                      ("derived budget (sum)", "budget"),
                      ("observed |R|", "absR"),
                      ("reported allowance E", "E_obs")):
        s = d[col]
        print(f"{name:34} {s.mean():12.3e} {s.quantile(.95):12.3e} {s.max():12.3e}")
    print(f"|R| within its own derived budget on {int(ok.sum())}/{len(d)} dates")
    print(f"max derived budget                 {d['budget'].max():.3e} hands")
    print(f"current fixture tolerance          {1e-2:.3e} hands"
          f"   ({1e-2 / d['budget'].max():.0f}x the worst derived budget)")

    worst = d.loc[d["budget"].idxmax()]
    print(f"widest-budget date {worst['date']}: E_D={worst['E_D']:.3e} "
          f"E_DF={worst['E_DF']:.3e} |R|={worst['absR']:.3e}")

# What a tolerance MEANS economically, at the worst date's spot.
print("\n=== economic reading, hedge P&L error per 1% index move ===")
spot = 5306.99
print(f"{'tolerance, hands':>18} {'CNY per 1% move':>18} {'bp of notional':>16}")
for t in (1e-2, 1e-3, 3e-4, 1e-4):
    cny = t * M_REF * 0.01 * spot
    print(f"{t:18.0e} {cny:18.2f} {cny / NOTIONAL * 1e4:16.4f}")
