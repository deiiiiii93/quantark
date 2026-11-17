"""
Stage 3 — rebalancing: the gap only lives in the current (stub) period.

A bump to S_i moves only the stub; every later period restarts from the
target weights.  So the per-name gap scales with the stub length, and the
rebalance frequency switches it on and off.

Sweep 2: rule none / quarterly / monthly / weekly / daily, 1Y ATM call, at the
start, middle and end (95%) of the current period, no drift since the last
rebalance.

Sweep 3: quarterly, mid-period, TECH -30% / 0 / +30% since the last
rebalance, others flat.  Current weights now differ from target weights, so
splitting Delta_I by *target* weights (M2) stops being the whole-index order.

Writes results/03_rebalance.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _common as cm  # noqa: E402

RESULTS = HERE / "results"
RULES = ["none", "quarterly", "monthly", "weekly", "daily"]
POSITIONS = [("start", 0.0), ("mid", 0.5), ("end", 0.95)]
TECH_MOVES = [0.7, 1.0, 1.3]


def sweep_frequency() -> list:
    rows = []
    option = cm.CSIOption(1.0, 1.0)
    for rule in RULES:
        basket = cm.base_basket(rule)
        period = cm.period_length(rule)
        positions = [("start", 0.0)] if period is None else POSITIONS
        for pos, frac in positions:
            elapsed = 0.0 if period is None else frac * period
            state = cm.default_state(basket, elapsed=elapsed)
            rec = cm.evaluate_scenario(basket, state, option, f"{rule}, {pos} of period")
            rec["position"] = pos
            rec["stub_years"] = float(cm.schedule_times(basket, state, 1.0)[0])
            rows.append(rec)
    return rows


def sweep_drift() -> list:
    rows = []
    basket = cm.base_basket("quarterly")
    option = cm.CSIOption(1.0, 1.0)
    tech = basket.names.index("TECH")
    for move in TECH_MOVES:
        moves = np.ones(5)
        moves[tech] = move
        state = cm.default_state(basket, elapsed=0.5 * 0.25, moves=moves)
        rec = cm.evaluate_scenario(
            basket, state, option, f"quarterly, mid-period, TECH {100 * (move - 1):+.0f}%"
        )
        rec["tech_move"] = move
        rows.append(rec)
    return rows


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    out = {
        "stage": "03_rebalance",
        "sweep2_frequency": sweep_frequency(),
        "sweep3_drift": sweep_drift(),
    }
    path = RESULTS / "03_rebalance.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path} in {time.time() - t0:.0f}s")
    for rec in out["sweep2_frequency"]:
        print(f"  {rec['label']:28s} M1 residual sigma {rec['methods']['M1']['residual_sigma_cny']:10,.0f} CNY/day")
    for rec in out["sweep3_drift"]:
        print(
            f"  {rec['label']:36s} M1 {rec['methods']['M1']['residual_sigma_cny']:9,.0f}  "
            f"M2 {rec['methods']['M2']['residual_sigma_cny']:9,.0f} CNY/day"
        )


if __name__ == "__main__":
    main()
