"""
Stage 2 — fixed-share basket (rule A): how big is the per-name gap, and where?

Sweep 1: moneyness x maturity x call/put on the base 5-name CSI with no
rebalance during the option's life.  Weights drift with prices, so the
basket's vol drifts too, and splitting Delta_I by weight (M1) misallocates
delta toward low-vol names.

Sweep 6: dividend-yield dispersion.  On a basket with identical vols, a deep
in-the-money call is almost a forward, and its per-name delta is
n_i exp(-q_i T) — not proportional to n_i.  So even the linear part of the
delta splits differently when yields differ.

Writes results/02_fixed_shares.json.
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

MONEYNESS = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
MATURITIES = [("1M", 1.0 / 12.0), ("3M", 0.25), ("6M", 0.5), ("1Y", 1.0)]


def sweep_moneyness_maturity() -> list:
    basket = cm.base_basket("none")
    state = cm.default_state(basket)
    rows = []
    for tenor, mat in MATURITIES:
        for k in MONEYNESS:
            for is_call in (True, False):
                label = f"{tenor} {'call' if is_call else 'put'} K/I={k:.2f}"
                rec = cm.evaluate_scenario(basket, state, cm.CSIOption(k, mat, is_call), label)
                rec["tenor"] = tenor
                rows.append(rec)
    return rows


def sweep_dividend_dispersion() -> list:
    rows = []
    variants = [
        ("equal yields 1.5%", np.full(5, 0.015)),
        ("base yields (0-3%)", cm.BASE_DIVS),
    ]
    for q_label, divs in variants:
        basket = cm.CSIBasket(
            names=cm.BASE_NAMES,
            target_weights=cm.BASE_WEIGHTS,
            vols=np.full(5, 0.25),
            div_yields=divs,
            corr=cm.uniform_corr(5, 0.5),
            rate=cm.BASE_RATE,
        )
        state = cm.default_state(basket)
        for k in (0.5, 1.0):
            label = f"{q_label}, K/I={k:.1f}, 1Y call"
            rec = cm.evaluate_scenario(basket, state, cm.CSIOption(k, 1.0), label)
            rec["yield_variant"] = q_label
            rows.append(rec)
    return rows


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    out = {
        "stage": "02_fixed_shares",
        "sweep1_moneyness_maturity": sweep_moneyness_maturity(),
        "sweep6_dividend_dispersion": sweep_dividend_dispersion(),
    }
    path = RESULTS / "02_fixed_shares.json"
    path.write_text(json.dumps(out, indent=1))
    atm = next(
        r for r in out["sweep1_moneyness_maturity"] if r["label"] == "1Y call K/I=1.00"
    )
    tech = atm["names"].index("TECH")
    print(f"wrote {path} in {time.time() - t0:.0f}s")
    print(
        "1Y ATM call, M1 gap on TECH: "
        f"{atm['methods']['M1']['gap_cash_per_pct'][tech]:,.0f} CNY per 1% "
        f"({100 * atm['methods']['M1']['gap_rel_to_ref'][tech]:+.1f}% of its delta)"
    )


if __name__ == "__main__":
    main()
