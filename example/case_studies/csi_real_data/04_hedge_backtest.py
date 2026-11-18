"""
Stage 4 (Stage B) — hedge each model's deltas along the real path.

For every start date (the 10th session of each month from 2024-01 whose 1Y
option has expired by the last close) and every rebalance rule, the desk sells
one ATM 1Y call on 1e8 CNY of index notional.  Strike K and units u are fixed
at the start (spec D9).  At each close until expiry both models are re-marked
on the trailing 244-session window and three hedges are set (spec D5):

1. **test**        — Black Δ_I · nᵢ · u  (the whole-index order)
2. **reference**   — per-name CRN bump deltas · u
3. **attribution** — the reference MC's own Δ_I · nᵢ · u

Every arm's daily P&L is marked to the reference PV and measured in excess of
financing (spec D3):

    P&Lᴬ(t) = Σᵢ hᵢᴬ(t)·(Sᵢ(t+1) − g·Sᵢ(t)) − (V(t+1) − g·V(t)),   g = e^{r/244}

with V(expiry) = the payoff, so the paired difference of two arms is exactly
their hedge difference.  Terminal P&L per start (sold at the arm's own PV) is
descriptive.  Paired daily differences get a calendar-session block bootstrap
(spec AC12); an independent-seed probe measures the MC-noise share.

Writes results/04_hedge.json.
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

from quantark.util.exceptions import ValidationError

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
_stage_a = _sibling("csi_real_data_stage_a", "03_gap_through_time")
_index = _models._index

RESULTS = HERE / "results"
NOTIONAL = 1e8
RATE = 0.018
WINDOW = 244
SESSIONS = 244               # 1Y option life in sessions (spec D2)
N_PATHS, N_BATCHES = 2**12, 16
SEED = 20261119
WORKERS = 4                  # spec C7
BLOCK = 20                   # bootstrap block, sessions
N_BOOT = 2000
PROBE_RULE = "monthly"
ARMS = ("test", "reference", "attribution")
PAIRS = (("reference", "test"), ("attribution", "test"), ("reference", "attribution"))


# ---------------------------------------------------------------------------
# Accounting and statistics
# ---------------------------------------------------------------------------


def growth(rate: float) -> float:
    return float(np.exp(rate / _models.TRADING_DAYS))


def daily_pnl(holdings: np.ndarray, spots: np.ndarray, marks: np.ndarray, rate: float) -> np.ndarray:
    """Spec D3: Σᵢ hᵢ(t)·(Sᵢ(t+1) − g·Sᵢ(t)) − (V(t+1) − g·V(t)) for t = 0 … N−1."""
    g = growth(rate)
    h = np.asarray(holdings, dtype=float)
    s = np.asarray(spots, dtype=float)
    v = np.asarray(marks, dtype=float)
    return (h * (s[1:] - g * s[:-1])).sum(axis=1) - (v[1:] - g * v[:-1])


def terminal_pnl(premium: float, mark0: float, pnl: np.ndarray, rate: float) -> float:
    """Self-financing terminal P&L of the short option sold at ``premium`` (identity with ``daily_pnl``)."""
    g = growth(rate)
    n = len(pnl)
    return float((premium - mark0) * g**n + np.sum(np.asarray(pnl) * g ** (n - 1 - np.arange(n))))


def block_bootstrap_std(values: np.ndarray, sessions: np.ndarray, block: int = BLOCK, n_boot: int = N_BOOT,
                        seed: int = 0) -> tuple:
    """90% CI of the pooled std, resampling blocks of consecutive calendar sessions (all starts kept together)."""
    values = np.asarray(values, dtype=float)
    keys, inverse = np.unique(np.asarray(sessions), return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(keys))]
    n_blocks = int(np.ceil(len(keys) / block))
    starts_max = max(len(keys) - block, 0)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        firsts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = np.concatenate([members[k] for f in firsts for k in range(f, min(f + block, len(keys)))])
        stats[b] = np.std(values[idx], ddof=1)
    return float(np.quantile(stats, 0.05)), float(np.quantile(stats, 0.95))


def pair_noise_var(noise_cov: np.ndarray, x: np.ndarray, shares: np.ndarray, units: float) -> dict:
    """MC-noise variance of each paired daily difference on one step.

    ``noise_cov`` is the covariance of the means of [bump deltas..., Δ_I] (per index unit); ``x`` is
    S(t+1) − g·S(t).  The test arm is deterministic, so (2)−(1) carries only the bump noise, (3)−(1) only the
    Δ_I noise, and (2)−(3) their joint noise.
    """
    c = np.asarray(noise_cov, dtype=float)
    x = np.asarray(x, dtype=float)
    nx = float(np.asarray(shares, dtype=float) @ x)
    v = np.append(x, -nx)
    return {
        "reference-test": float(units**2 * x @ c[:-1, :-1] @ x),
        "attribution-test": float(units**2 * nx * nx * c[-1, -1]),
        "reference-attribution": float(units**2 * v @ c @ v),
    }


def block_bootstrap_corrected_std(values: np.ndarray, noise_var: np.ndarray, sessions: np.ndarray,
                                  block: int = BLOCK, n_boot: int = N_BOOT, seed: int = 0) -> tuple:
    """Noise-corrected std √max(Var(d) − mean(noise var), 0) with a calendar-block 90% CI: (point, lo, hi)."""
    values = np.asarray(values, dtype=float)
    noise_var = np.asarray(noise_var, dtype=float)

    def corrected(idx):
        return float(np.sqrt(max(np.var(values[idx], ddof=1) - np.mean(noise_var[idx]), 0.0)))

    keys, inverse = np.unique(np.asarray(sessions), return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(keys))]
    n_blocks = int(np.ceil(len(keys) / block))
    starts_max = max(len(keys) - block, 0)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        firsts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = np.concatenate([members[k] for f in firsts for k in range(f, min(f + block, len(keys)))])
        stats[b] = corrected(idx)
    return corrected(np.arange(values.size)), float(np.quantile(stats, 0.05)), float(np.quantile(stats, 0.95))


def mc_noise_share(d_n: np.ndarray, d_2n: np.ndarray) -> float:
    """Spec AC12: σ_mc(N) = sqrt(Var(d_N − d_2N)/1.5) (independent seeds); share = σ_mc(N) / std(d_N)."""
    d_n, d_2n = np.asarray(d_n, dtype=float), np.asarray(d_2n, dtype=float)
    sigma_mc = np.sqrt(np.var(d_n - d_2n, ddof=1) / 1.5)
    return float(sigma_mc / np.std(d_n, ddof=1))


# ---------------------------------------------------------------------------
# One start date, one rule
# ---------------------------------------------------------------------------


def run_start(prices: pd.DataFrame, level: pd.Series, shares: pd.DataFrame, calendar: pd.DatetimeIndex,
              rule: str, start, rate: float, n_paths: int, n_batches: int, seed: int,
              window: int = WINDOW, sessions: int = SESSIONS) -> dict:
    """Sell an ATM call at ``start`` and hedge it with the three arms until expiry."""
    start = pd.Timestamp(start)
    days = prices.index[prices.index >= start][: sessions + 1]
    if len(days) < sessions + 1 or days[0] != start:
        raise ValidationError(f"{start.date()}: need {sessions} sessions of prices after the start")
    expiry = days[-1]
    k = float(level.loc[start])
    units = NOTIONAL / k
    w_tgt = _data.target_weights().to_numpy()
    spots = prices.loc[days].to_numpy(dtype=float)
    hold = {a: np.empty((sessions, spots.shape[1])) for a in ARMS}
    n_hist = np.empty((sessions, spots.shape[1]))
    covs = np.zeros((sessions, spots.shape[1] + 1, spots.shape[1] + 1))
    marks = np.empty(sessions + 1)
    mark_se = np.empty(sessions)
    di_test, di_ref, own_test = np.empty(sessions), np.empty(sessions), 0.0
    for j, t in enumerate(days[:-1]):
        cal = _models.calibrate(prices, level, t, window)
        sched = _models.remaining_schedule(calendar, rule, t, expiry)
        lvl = float(level.loc[t])
        n_t = shares.loc[t].to_numpy(dtype=float)
        pv_t, d_t, _ = _models.black_call(lvl, k, float(sched[-1]), cal.index_vol, rate)
        ref = _models.reference_call(spots[j], n_t, w_tgt, cal, rule, k, sched, rate, n_paths, n_batches,
                                     seed + j)
        hold["test"][j] = d_t * n_t * units
        hold["reference"][j] = ref.deltas_bump * units
        hold["attribution"][j] = ref.delta_index * n_t * units
        marks[j] = ref.pv * units
        mark_se[j] = ref.pv_se * units
        n_hist[j] = n_t
        if ref.noise_cov is not None:
            covs[j] = ref.noise_cov
        di_test[j], di_ref[j] = d_t, ref.delta_index
        if j == 0:
            own_test = pv_t * units
    marks[-1] = max(float(level.loc[expiry]) - k, 0.0) * units
    own = {"test": own_test, "reference": marks[0], "attribution": marks[0]}
    x = spots[1:] - growth(rate) * spots[:-1]
    noise = [pair_noise_var(covs[j], x[j], n_hist[j], units) for j in range(sessions)]
    noise_var = {f"{a}-{b}": [round(nv[f"{a}-{b}"], 2) for nv in noise] for a, b in PAIRS}
    arms = {}
    for a in ARMS:
        pnl = daily_pnl(hold[a], spots, marks, rate)
        arms[a] = {
            "daily_pnl": [round(float(x), 2) for x in pnl],
            "premium_own_cny": float(own[a]),
            "premium_gap_cny": float(own[a] - marks[0]),
            "terminal_cny": terminal_pnl(own[a], marks[0], pnl, rate),
        }
    return {
        "rule": rule, "start": start.strftime("%Y-%m-%d"), "expiry": expiry.strftime("%Y-%m-%d"),
        "strike": k, "units": units, "dates": [d.strftime("%Y-%m-%d") for d in days[:-1]],
        "payoff_cny": float(marks[-1]), "ref_mark_cny": [round(float(x), 2) for x in marks[:-1]],
        "ref_mark_se_cny": [round(float(x), 2) for x in mark_se],
        "dindex_test": [round(float(v), 6) for v in di_test], "dindex_ref": [round(float(v), 6) for v in di_ref],
        "noise_var": noise_var,
        "arms": arms,
    }


def _task(args):
    prices, level, shares, calendar, rule, start, n_paths, n_batches, seed = args
    return run_start(prices, level, shares, calendar, rule, start, RATE, n_paths, n_batches, seed)


def summarize(rule_runs: list) -> dict:
    """Per-rule pooled daily statistics, paired differences with block-bootstrap CIs, terminal P&L."""
    sessions = np.concatenate([np.array(r["dates"], dtype="datetime64[D]") for r in rule_runs])
    pnl = {a: np.concatenate([r["arms"][a]["daily_pnl"] for r in rule_runs]) for a in ARMS}
    out = {"n_starts": len(rule_runs), "n_obs": int(sessions.size),
           "arm_daily_std_cny": {a: float(np.std(pnl[a], ddof=1)) for a in ARMS}, "pairs": {}}
    for x, y in PAIRS:
        key = f"{x}-{y}"
        d = pnl[x] - pnl[y]
        nv = np.concatenate([r["noise_var"][key] for r in rule_runs])
        lo, hi = block_bootstrap_std(d, sessions, seed=7)
        point, clo, chi = block_bootstrap_corrected_std(d, nv, sessions, seed=7)
        out["pairs"][key] = {"std_cny": float(np.std(d, ddof=1)), "ci90": [lo, hi], "mean_cny": float(np.mean(d)),
                             "noise_std_cny": float(np.sqrt(np.mean(nv))),
                             "corrected_std_cny": point, "corrected_ci90": [clo, chi],
                             "resolved": bool(clo > 0.0)}
    out["terminal_cny"] = {a: [r["arms"][a]["terminal_cny"] for r in rule_runs] for a in ARMS}
    out["premium_gap_cny"] = [r["arms"]["test"]["premium_gap_cny"] for r in rule_runs]
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    prices = _data.load_prices()
    calendar = _data.load_calendar()
    w = _data.target_weights()
    last = prices.index[-1]
    pos = {d: i for i, d in enumerate(prices.index)}
    starts = [d for d in _stage_a.eval_dates(prices.index, _stage_a.EVAL_START) if pos[d] + SESSIONS <= pos[last]]
    paths = {rule: _index.build_index(prices, w, rule, prices.index[0]) for rule in _index.RULES}
    tasks = [(prices, paths[r].level, paths[r].shares, calendar, r, s, N_PATHS, N_BATCHES, SEED + 1000 * k)
             for r in _index.RULES for k, s in enumerate(starts)]
    probe_starts = [starts[0], starts[len(starts) // 2], starts[-1]]
    probe = [(prices, paths[PROBE_RULE].level, paths[PROBE_RULE].shares, calendar, PROBE_RULE, s,
              2 * N_PATHS, N_BATCHES, SEED + 500_000 + 1000 * k) for k, s in enumerate(probe_starts)]
    order = {"daily": 0, "weekly": 1}
    tasks.sort(key=lambda t: order.get(t[4], 2))
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=get_context("spawn")) as pool:
        runs = list(pool.map(_task, tasks + probe))
    main_runs, probe_runs = runs[: len(tasks)], runs[len(tasks):]
    by_rule = {r: sorted([x for x in main_runs if x["rule"] == r], key=lambda x: x["start"]) for r in _index.RULES}
    d_n, d_2n, nv_n = [], [], []
    for pr in probe_runs:
        base = next(x for x in by_rule[PROBE_RULE] if x["start"] == pr["start"])
        d_n.append(np.subtract(base["arms"]["reference"]["daily_pnl"], base["arms"]["test"]["daily_pnl"]))
        d_2n.append(np.subtract(pr["arms"]["reference"]["daily_pnl"], pr["arms"]["test"]["daily_pnl"]))
        nv_n.append(base["noise_var"]["reference-test"])
    d_n, d_2n, nv_n = np.concatenate(d_n), np.concatenate(d_2n), np.concatenate(nv_n)
    share = mc_noise_share(d_n, d_2n)
    measured = float(np.sqrt(np.var(d_n - d_2n, ddof=1) / 1.5))
    predicted = float(np.sqrt(np.mean(nv_n)))
    out = {
        "stage": "04_hedge_backtest",
        "meta": {"rate": RATE, "window": WINDOW, "sessions": SESSIONS, "n_paths": N_PATHS, "n_batches": N_BATCHES,
                 "seed_base": SEED, "notional_cny": NOTIONAL, "block": BLOCK, "n_boot": N_BOOT,
                 "starts": [s.strftime("%Y-%m-%d") for s in starts], "rules": list(_index.RULES),
                 "names": [n.code for n in _data.BASKET], "seconds": round(time.time() - t0, 1)},
        "noise_probe": {"rule": PROBE_RULE, "starts": [s.strftime("%Y-%m-%d") for s in probe_starts],
                        "n_paths": N_PATHS, "n_paths_2n": 2 * N_PATHS, "share": share,
                        "noise_std_measured_cny": measured, "noise_std_predicted_cny": predicted,
                        "predicted_over_measured": predicted / measured,
                        "validates": bool(0.75 <= predicted / measured <= 1.33)},
        "summary": {r: summarize(by_rule[r]) for r in _index.RULES},
        "runs": by_rule,
    }
    target = RESULTS / "04_hedge.json"
    target.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {target}: {len(starts)} starts × {len(_index.RULES)} rules, {time.time() - t0:.0f}s; "
          f"MC-noise share {share:.3f}; noise σ predicted {predicted:,.0f} vs measured {measured:,.0f} CNY/day "
          f"({'validates' if out['noise_probe']['validates'] else 'DOES NOT VALIDATE'})")
    for r in _index.RULES:
        s = out["summary"][r]
        line = f"  {r:9s} arm std test {s['arm_daily_std_cny']['test']:8,.0f} ref {s['arm_daily_std_cny']['reference']:8,.0f} |"
        for key in ("reference-test", "attribution-test", "reference-attribution"):
            p = s["pairs"][key]
            line += (f" {key} {p['corrected_std_cny']:7,.0f} [{p['corrected_ci90'][0]:6,.0f},{p['corrected_ci90'][1]:7,.0f}]"
                     f" (raw {p['std_cny']:6,.0f}, noise {p['noise_std_cny']:5,.0f})")
        print(line)


if __name__ == "__main__":
    main()
