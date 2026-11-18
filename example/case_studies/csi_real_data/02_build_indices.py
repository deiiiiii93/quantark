"""
Stage 2 — the same basket, five rebalance rules, one set of real prices.

Builds the CSI from the cached back-adjusted closes (base 1000 on the first
session of 2023, target weights) under none / quarterly / monthly / weekly /
daily rebalancing, and records each rule's daily level, current weights and
rebalance dates plus summary statistics: total return, realized vol (full
range and the 2024-01 → now evaluation period) and the largest weight drift.

Share counts are not written (spec D14).  Writes results/02_indices.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _data  # noqa: E402
import _index  # noqa: E402
from quantark.util.exceptions import MarketDataError  # noqa: E402

RESULTS = HERE / "results"
BASE_LEVEL = 1000.0
EVAL_START = pd.Timestamp("2024-01-01")


def annualised_vol(levels: pd.Series) -> float:
    """Spec D13: demeaned daily log returns, ddof = 1, ×√244."""
    r = np.diff(np.log(levels.to_numpy(dtype=float)))
    return float(np.std(r, ddof=1) * np.sqrt(_index.TRADING_DAYS))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    prices = _data.load_prices()
    meta = _data.load_meta()
    fills = _data.load_fill_counts()
    w = _data.target_weights()
    base = prices.index[0]
    cal = _data.load_calendar()
    first_session = cal[cal >= pd.Timestamp(_data.FETCH_START)][0]
    if base != first_session:
        raise MarketDataError(f"base date {base.date()} is not the first session {first_session.date()} "
                              "(spec D8): a name starts late or the cache is truncated")
    dates = prices.index
    out_rules = {}
    for rule in _index.RULES:
        path = _index.build_index(prices, w, rule, base, BASE_LEVEL)
        level = path.level
        ev = level[level.index >= EVAL_START]
        drift = (path.weights - w.to_numpy()).abs().to_numpy()
        out_rules[rule] = {
            "level": [round(float(x), 6) for x in level],
            "weights": {c: [round(float(x), 8) for x in path.weights[c]] for c in w.index},
            "rebalance_dates": [d.strftime("%Y-%m-%d") for d in path.rebalance_dates],
            "n_rebalances": int(len(path.rebalance_dates)),
            "total_return": float(level.iloc[-1] / level.iloc[0] - 1.0),
            "eval_return": float(ev.iloc[-1] / level[level.index < EVAL_START].iloc[-1] - 1.0),
            "realized_vol_full": annualised_vol(level),
            "realized_vol_eval": annualised_vol(level[level.index >= level[level.index < EVAL_START].index[-1]]),
            "max_abs_weight_drift": float(drift.max()),
            "final_weights": {c: float(path.weights[c].iloc[-1]) for c in w.index},
        }
    names = []
    for n in _data.BASKET:
        px = prices[n.code]
        names.append({
            "code": n.code, "name": n.name, "kind": n.kind, "weight": n.weight,
            "total_return": float(px.iloc[-1] / px.iloc[0] - 1.0),
            "realized_vol_full": annualised_vol(px),
            "suspension_fills": int(fills.get(n.code, 0)),
            "source": meta.get("sources", {}).get(n.code),
        })
    finals = {r: out_rules[r]["level"][-1] for r in _index.RULES}
    out = {
        "stage": "02_build_indices",
        "meta": {
            "akshare": meta.get("akshare"), "fetched_on": meta.get("fetched_on"),
            "data_first": dates[0].strftime("%Y-%m-%d"), "data_last": dates[-1].strftime("%Y-%m-%d"),
            "n_sessions": int(len(dates)), "base_date": base.strftime("%Y-%m-%d"), "base_level": BASE_LEVEL,
            "eval_start": EVAL_START.strftime("%Y-%m-%d"), "calendar_horizon": meta.get("calendar_horizon"),
            "vol_convention": "demeaned daily log returns, ddof=1, x sqrt(244)",
        },
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "names": names,
        "rules": out_rules,
        "terminal_spread": float(max(finals.values()) / min(finals.values()) - 1.0),
    }
    target = RESULTS / "02_indices.json"
    target.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {target}  ({len(dates)} sessions {out['meta']['data_first']} → {out['meta']['data_last']})")
    for rule in _index.RULES:
        r = out_rules[rule]
        print(f"  {rule:9s} final {r['level'][-1]:9.2f}  total {r['total_return']:+7.1%}  "
              f"vol(full) {r['realized_vol_full']:6.2%}  vol(2024→) {r['realized_vol_eval']:6.2%}  "
              f"rebalances {r['n_rebalances']:4d}  max drift {r['max_abs_weight_drift']:6.2%}")
    print(f"  terminal spread across rules: {out['terminal_spread']:.2%}")


if __name__ == "__main__":
    main()
