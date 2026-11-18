"""
Stage 3 (Stage A) — the gap between the test model and the reference through time.

On the 10th session of each month from 2024-01 (after that day's close and any
rebalance at it), for each rebalance rule and for 1Y calls struck at 90 / 100 /
110% of the index level:

* **test model** — Black on the index, vol = the index's own trailing 244-session
  historical vol; per-name delta = Δ_I split by current weights;
* **reference** — per-name RQMC with per-name trailing vols and correlation,
  the real remaining rebalance schedule, CRN bump per name (pathwise kept for
  the exact decomposition, spec D11).

Each record carries PVs, Δ_I, per-name deltas (CNY per 1% move on 1e8 CNY of
index notional), the decomposition of the per-name gap into a total-delta part
and a split part, the first-order closed form for the split, the stub τ, and
the daily P&L σ the gap implies.  Writes results/03_gap.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _sibling(alias: str, stem: str):
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, HERE / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_models = _sibling("csi_real_data_models", "_models")
_data = _sibling("csi_real_data_data", "_data")
_index = _models._index

RESULTS = HERE / "results"
NOTIONAL = 1e8
RATE = 0.018
WINDOW = 244
MONEYNESS = (0.9, 1.0, 1.1)
EVAL_START = pd.Timestamp("2024-01-01")
EVAL_SESSION = 10            # the 10th session of each month (spec D8)
N_PATHS, N_BATCHES = 2**13, 32
SEED = 20261118
SEED_STRIDE = 1000           # per date; each call uses N_BATCHES consecutive scramble seeds, so stride ≥ N_BATCHES
WORKERS = 4                  # memory: never wide pools on this machine (spec C7)


def eval_dates(sessions: pd.DatetimeIndex, start: pd.Timestamp) -> pd.DatetimeIndex:
    """The ``EVAL_SESSION``-th session of every month from ``start`` that exists in ``sessions``."""
    s = pd.DatetimeIndex(sessions)
    s = s[s >= start]
    out = []
    for _, group in pd.Series(s, index=s).groupby([s.year, s.month]):
        if len(group) >= EVAL_SESSION:
            out.append(group.iloc[EVAL_SESSION - 1])
    return pd.DatetimeIndex(out)


def evaluate_date(prices: pd.DataFrame, level: pd.Series, shares: pd.DataFrame, weights: pd.DataFrame,
                  calendar: pd.DatetimeIndex, rule: str, asof: pd.Timestamp, moneyness, rate: float,
                  n_paths: int, n_batches: int, seed: int, window: int = WINDOW) -> list:
    """Stage A records for one (rule, date): one per moneyness."""
    asof = pd.Timestamp(asof)
    cal = _models.calibrate(prices, level, asof, window)
    expiry, fallback = _models.expiry_date(calendar, asof)
    sched = _models.remaining_schedule(calendar, rule, asof, expiry)
    spots = prices.loc[asof].to_numpy(dtype=float)
    n = shares.loc[asof].to_numpy(dtype=float)
    w_now = weights.loc[asof].to_numpy(dtype=float)
    w_tgt = _data.target_weights().to_numpy()
    lvl = float(level.loc[asof])
    units = NOTIONAL / lvl
    scale = units * 0.01                                     # per index unit → CNY per 1% move
    sigma = cal.vols[:, None] * cal.corr * cal.vols[None, :]
    cov_daily = sigma / _models.TRADING_DAYS
    vol_model = float(np.sqrt(w_now @ sigma @ w_now))
    tau = float(sched[0])
    out = []
    for m in moneyness:
        k = m * lvl
        pv_t, di_t, _ = _models.black_call(lvl, k, float(sched[-1]), cal.index_vol, rate)
        ref = _models.reference_call(spots, n, w_tgt, cal, rule, k, sched, rate, n_paths, n_batches, seed)
        test_cash = w_now * lvl * di_t * scale
        bump = spots * ref.deltas_bump * scale
        bump_se = spots * ref.deltas_bump_se * scale
        pw = spots * ref.deltas_pathwise * scale
        pw_se = spots * ref.deltas_pathwise_se * scale
        total, split = _models.gap_decomposition(spots * ref.deltas_pathwise, di_t, ref.delta_index, w_now, lvl, units)
        fo, fo_valid = _models.first_order_split(cal, w_now, lvl, k, tau, 1.0, rate, units, rule)
        gap = test_cash - bump
        g_unit = gap * 100.0                                   # CNY per unit return of each name
        out.append({
            "date": asof.strftime("%Y-%m-%d"), "rule": rule, "moneyness": float(m), "strike": float(k),
            "index_level": lvl, "units": units, "tau": tau, "tau_sessions": int(round(tau * _models.TRADING_DAYS)),
            "n_rebalances_left": int(len(sched) - 1), "expiry": expiry.strftime("%Y-%m-%d"),
            "calendar_fallback": bool(fallback),
            "index_vol_hist": float(cal.index_vol), "index_vol_model": vol_model,
            "name_vols": [float(v) for v in cal.vols], "psd_repaired": bool(cal.psd_repaired),
            "pv_test_cny": float(pv_t * units), "pv_ref_cny": float(ref.pv * units),
            "pv_ref_se_cny": float(ref.pv_se * units),
            "dindex_test": float(di_t), "dindex_ref": float(ref.delta_index), "dindex_ref_se": float(ref.delta_index_se),
            "weights": [float(x) for x in w_now],
            "test_cash": [float(x) for x in test_cash],
            "ref_bump": [float(x) for x in bump], "ref_bump_se": [float(x) for x in bump_se],
            "ref_pathwise": [float(x) for x in pw], "ref_pathwise_se": [float(x) for x in pw_se],
            "gap_bump": [float(x) for x in gap],
            "gap_bump_resolved": [bool(abs(g) > 3.0 * s) for g, s in zip(gap, bump_se)],
            "gap_total_part": [float(x) for x in total], "gap_split_part": [float(x) for x in split],
            "bump_minus_pathwise": [float(x) for x in bump - pw],
            "first_order": [float(x) for x in fo], "first_order_valid": bool(fo_valid),
            "gap_sigma_cny_day": float(np.sqrt(max(g_unit @ cov_daily @ g_unit, 0.0))),
        })
    return out


def _task(args):
    prices, level, shares, weights, calendar, rule, asof, seed = args
    return evaluate_date(prices, level, shares, weights, calendar, rule, asof, MONEYNESS, RATE,
                         N_PATHS, N_BATCHES, seed)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    prices = _data.load_prices()
    calendar = _data.load_calendar()
    w = _data.target_weights()
    dates = eval_dates(prices.index, EVAL_START)
    tasks = []
    for rule in _index.RULES:
        path = _index.build_index(prices, w, rule, prices.index[0])
        for k, d in enumerate(dates):
            tasks.append((prices, path.level, path.shares, path.weights, calendar, rule, d, SEED + SEED_STRIDE * k))
    # Daily-rule tasks are the slowest; submit them first so the pool drains evenly.
    tasks.sort(key=lambda t: {"daily": 0, "weekly": 1}.get(t[5], 2))
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context("spawn")) as pool:
        chunks = list(pool.map(_task, tasks))
    records = sorted((r for c in chunks for r in c), key=lambda r: (r["rule"], r["date"], r["moneyness"]))
    out = {
        "stage": "03_gap_through_time",
        "meta": {"rate": RATE, "window": WINDOW, "moneyness": list(MONEYNESS), "eval_session": EVAL_SESSION,
                 "n_paths": N_PATHS, "n_batches": N_BATCHES, "seed_base": SEED, "seed_stride": SEED_STRIDE, "notional_cny": NOTIONAL,
                 "dates": [d.strftime("%Y-%m-%d") for d in dates], "rules": list(_index.RULES),
                 "names": [n.code for n in _data.BASKET], "seconds": round(time.time() - t0, 1),
                 "calendar_horizon": _data.load_meta().get("calendar_horizon")},
        "records": records,
    }
    target = RESULTS / "03_gap.json"
    target.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    n_fb = sum(r["calendar_fallback"] for r in records)
    print(f"wrote {target}: {len(records)} records, {len(dates)} dates, {time.time() - t0:.0f}s; "
          f"{n_fb} records use the weekday calendar fallback")
    tech = [n.code for n in _data.BASKET].index("002230")
    for rule in _index.RULES:
        atm = [r for r in records if r["rule"] == rule and r["moneyness"] == 1.0]
        pv = np.array([r["pv_test_cny"] / r["pv_ref_cny"] - 1.0 for r in atm])
        sig = np.array([r["gap_sigma_cny_day"] for r in atm])
        tg = np.array([r["gap_bump"][tech] for r in atm])
        print(f"  {rule:9s} ATM PV test/ref−1 mean {pv.mean():+6.2%} [{pv.min():+6.2%}, {pv.max():+6.2%}]  "
              f"gap σ mean {sig.mean():8,.0f} CNY/day  TECH gap mean {tg.mean():+9,.0f} CNY/1%")


if __name__ == "__main__":
    main()
