"""Per-run statistics, paired variance ratios, Holm, basis regression and the verdict (design 4.5-4.6).

Input is the ``run_one`` dict (see ``runner.py``).  Everything is in basis points of ``NOTIONAL``.
Books are keyed ``"a"`` and ``"b"``.  The vega-adjusted residual is a DIAGNOSTIC only: it is reported
(``std_vega_resid_bp`` per run, a ``vega_resid`` block in ``paired``) and never enters ``verdict``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy import stats

from quantark.util.numerical import is_zero

from . import NOTIONAL

BOOKS = ("a", "b")
HOLD_FORWARD = "hold_forward"
#: Verdict (c) needs at least this many completed (non-truncated) inceptions.
MIN_COMPLETE = 5
NAN = float("nan")


def _bp(x: float) -> float:
    return 1e4 * float(x) / NOTIONAL


def _std(x: Sequence[float]) -> float:
    return float(np.std(np.asarray(x, dtype=float), ddof=1)) if len(x) >= 2 else NAN


def _isnan(x: Optional[float]) -> bool:
    return x is None or not math.isfinite(x)


def _basis_regression(days: Sequence[Mapping[str, Any]], book: str) -> Dict[str, float]:
    """OLS (with intercept) of hedged bp on (d_far, d_mo) in % points, non-roll days; betas in bp per 1% basis."""
    keep = [d for d in days if not d[f"roll_{book}"] and d["d_far_basis"] is not None and d["d_mo_basis"] is not None]
    if len(keep) < 4:
        return {"beta_far": NAN, "beta_mo": NAN, "basis_r2": NAN, "n": len(keep)}
    y = np.array([_bp(d[f"hedged_{book}"]) for d in keep])
    x = np.column_stack([np.ones(len(keep)), [100.0 * d["d_far_basis"] for d in keep],
                         [100.0 * d["d_mo_basis"] for d in keep]])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    sst = float(np.sum((y - y.mean()) ** 2))
    ssr = float(np.sum((y - x @ coef) ** 2))
    r2 = NAN if is_zero(sst) else 1.0 - ssr / sst
    return {"beta_far": float(coef[1]), "beta_mo": float(coef[2]), "basis_r2": r2, "n": len(keep)}


def _vega_resid_std(run: Mapping[str, Any], book: str) -> float:
    """std of hedged_t + vega_{t-1}*d_vol_t (short side); day 1 uses ``day0_vega``."""
    resid: List[float] = []
    prev_vega = run["day0_vega"]
    for d in run["days"]:
        if d["d_vol"] is not None and prev_vega is not None:
            resid.append(_bp(d[f"hedged_{book}"] + prev_vega * d["d_vol"]))
        prev_vega = d["vega"]
    return _std(resid)


def _max_drawdown(h: Sequence[float]) -> float:
    cum = np.concatenate([[0.0], np.cumsum(np.asarray(h, dtype=float))])
    return float(np.max(np.maximum.accumulate(cum) - cum))


def per_run(run: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per book statistics of one run dict: ``{"a": {...}, "b": {...}}``."""
    days = run["days"]
    out: Dict[str, Dict[str, Any]] = {}
    for k in BOOKS:
        hedged = [_bp(d[f"hedged_{k}"]) for d in days]
        note = [_bp(d["note_pnl"]) for d in days]
        var_h, var_n = (np.var(hedged, ddof=1), np.var(note, ddof=1)) if len(days) >= 2 else (NAN, NAN)
        roll = [abs(h) for h, d in zip(hedged, days) if d[f"roll_{k}"]]
        out[k] = {
            "std_bp": _std(hedged),
            "r2": NAN if _isnan(var_n) or is_zero(var_n) else 1.0 - float(var_h) / float(var_n),
            "turnover_lots_per_day": float(np.mean([d[f"turn_{k}"] for d in days])) if days else NAN,
            "roll_day_abs_bp": float(np.mean(roll)) if roll else NAN,
            "cost_bp": _bp(run[f"cost_total_{k}"]),
            "terminal_bp": _bp(run[f"terminal_{k}"]),
            "terminal_own_bp": _bp(run[f"terminal_own_{k}"]),
            "max_dd_bp": _max_drawdown(hedged) if hedged else NAN,
            "basis": _basis_regression(days, k),
            "std_vega_resid_bp": _vega_resid_std(run, k),
        }
    return out


def holm(p: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values, returned in the input order (NaN counts as 1)."""
    m = len(p)
    clean = [1.0 if _isnan(x) else float(x) for x in p]
    order = sorted(range(m), key=lambda i: clean[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * clean[i]))
        adj[i] = running
    return adj


def _pooled(stds: Sequence[float]) -> float:
    return float(math.sqrt(np.mean(np.square(stds)))) if len(stds) else NAN


def _by_inception(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {r["inception"]: r for r in runs}


def _is_converging(tail: str) -> bool:
    return tail.startswith("converge")


def _paired_block(tail_std: Dict[str, float], hf_std: Dict[str, float]) -> Dict[str, Any]:
    """Pooled std and paired variance-ratio statistics from per-inception stds (common inceptions only)."""
    common = sorted(i for i in tail_std if i in hf_std and not _isnan(tail_std[i]) and not _isnan(hf_std[i]))
    ratios = [tail_std[i] ** 2 / hf_std[i] ** 2 for i in common if not is_zero(hf_std[i])]
    logr = np.log([r for r in ratios if r > 0.0])
    n = len(ratios)
    if len(logr) >= 2 and not is_zero(float(np.std(logr, ddof=1))):
        t = float(np.mean(logr) / (np.std(logr, ddof=1) / math.sqrt(len(logr))))
        p_raw = float(2.0 * stats.t.sf(abs(t), len(logr) - 1))
    else:
        t, p_raw = NAN, NAN
    return {
        "pooled_std_bp": _pooled([tail_std[i] for i in common]),
        "hf_pooled_std_bp": _pooled([hf_std[i] for i in common]),
        "ratios": ratios, "n_lower": int(sum(r < 1.0 for r in ratios)), "n": n,
        "t_log_ratio": t, "p_raw": p_raw, "p_holm": None,
    }


def paired(runs_by_tail: Mapping[str, Sequence[Mapping[str, Any]]], book: str) -> Dict[str, Dict[str, Any]]:
    """Per non-hold-forward tail vs ``hold_forward``, paired by inception; Holm over the converging tails."""
    stats_by_tail = {t: {r["inception"]: per_run(r)[book] for r in runs} for t, runs in runs_by_tail.items()}
    hf = stats_by_tail[HOLD_FORWARD]
    out: Dict[str, Dict[str, Any]] = {}
    for tail, rows in stats_by_tail.items():
        if tail == HOLD_FORWARD:
            continue
        blk = _paired_block({i: v["std_bp"] for i, v in rows.items()}, {i: v["std_bp"] for i, v in hf.items()})
        blk["vega_resid"] = _paired_block({i: v["std_vega_resid_bp"] for i, v in rows.items()},
                                          {i: v["std_vega_resid_bp"] for i, v in hf.items()})
        out[tail] = blk
    conv = [t for t in out if _is_converging(t)]
    for key in (None, "vega_resid"):
        blocks = [out[t] if key is None else out[t][key] for t in conv]
        for b, adj in zip(blocks, holm([b["p_raw"] for b in blocks])):
            b["p_holm"] = adj
    return out


def paired_complete(runs_by_tail: Mapping[str, Sequence[Mapping[str, Any]]], book: str) -> Dict[str, Dict[str, Any]]:
    """``paired`` restricted to inceptions complete (non-truncated) in BOTH the tail and hold-forward."""
    hf_done = {r["inception"] for r in runs_by_tail[HOLD_FORWARD] if not r["truncated"]}
    keep = {t: [r for r in runs if not r["truncated"] and r["inception"] in hf_done] for t, runs in runs_by_tail.items()}
    done = set(hf_done)
    for t, runs in keep.items():
        if t != HOLD_FORWARD:
            done &= {r["inception"] for r in runs}
    keep = {t: [r for r in runs if r["inception"] in done] for t, runs in keep.items()}
    return paired(keep, book)


def terminal_paired(runs_by_tail: Mapping[str, Sequence[Mapping[str, Any]]], book: str) -> Dict[str, Dict[str, Any]]:
    """Terminal (traded-price) comparison on completed runs only, vs hold-forward on the same inceptions."""
    term = {t: {r["inception"]: per_run(r)[book]["terminal_bp"] for r in runs if not r["truncated"]}
            for t, runs in runs_by_tail.items()}
    hf = term[HOLD_FORWARD]
    out: Dict[str, Dict[str, Any]] = {}
    for tail, rows in term.items():
        if tail == HOLD_FORWARD:
            continue
        common = sorted(i for i in rows if i in hf)
        tv, hv = [rows[i] for i in common], [hf[i] for i in common]
        out[tail] = {
            "terminal_std_bp": _std(tv), "hf_terminal_std_bp": _std(hv),
            "terminal_mean_bp": float(np.mean(tv)) if tv else NAN,
            "n_abs_lower": int(sum(abs(a) < abs(b) for a, b in zip(tv, hv))), "n_complete": len(common),
        }
    return out


def _daily_ok(p: Mapping[str, Any]) -> bool:
    return bool(p["n"] > 0 and p["pooled_std_bp"] < p["hf_pooled_std_bp"] and 2 * p["n_lower"] > p["n"])


def _terminal_ok(t: Mapping[str, Any]) -> bool:
    return bool(t["n_complete"] >= MIN_COMPLETE and t["terminal_std_bp"] < t["hf_terminal_std_bp"]
                and 2 * t["n_abs_lower"] > t["n_complete"])


def verdict(paired_a: Mapping[str, Mapping[str, Any]], paired_b: Mapping[str, Mapping[str, Any]],
            terminal_a: Mapping[str, Mapping[str, Any]], terminal_b: Mapping[str, Mapping[str, Any]],
            paired_a_complete: Optional[Mapping[str, Mapping[str, Any]]] = None,
            paired_b_complete: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    """Design 4.6: per converging tail and book, (a) pooled std lower, (b) strict-majority paired ratio < 1,
    (c) completed-run terminal std lower and |terminal| lower in a strict majority; < 5 completed => not_decidable."""
    out: Dict[str, Dict[str, Any]] = {}
    for tail in (t for t in paired_a if _is_converging(t)):
        v: Dict[str, Any] = {}
        for k, pb, tb in (("a", paired_a, terminal_a), ("b", paired_b, terminal_b)):
            v[f"daily_ok_{k}"] = _daily_ok(pb[tail])
            v[f"terminal_ok_{k}"] = _terminal_ok(tb[tail])
            v[f"book_{k}"] = ("not_decidable" if tb[tail]["n_complete"] < MIN_COMPLETE
                              else v[f"daily_ok_{k}"] and v[f"terminal_ok_{k}"])
        v["beats_hold_forward"] = v["book_a"] is True and v["book_b"] is True
        if paired_a_complete is not None and paired_b_complete is not None:
            v["complete_only"] = {"daily_ok_a": _daily_ok(paired_a_complete[tail]),
                                  "daily_ok_b": _daily_ok(paired_b_complete[tail])}
        else:
            v["complete_only"] = None
        out[tail] = v
    return out


def qualifying_tails(verdicts: Mapping[str, Mapping[str, Any]], paired_a: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Tails that beat hold-forward, ordered by Book A pooled std (design 4.6)."""
    return sorted((t for t, v in verdicts.items() if v["beats_hold_forward"]),
                  key=lambda t: paired_a[t]["pooled_std_bp"])


def _nanmean(x: Sequence[float]) -> float:
    ok = [v for v in x if not _isnan(v)]
    return float(np.mean(ok)) if ok else NAN


def basis_mitigation(per_run_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per tail: mean |beta_far|, |beta_mo|, basis R2, cost for A and B, and the B-A differences.

    Rows are ``{"tail": ..., **per_run(run)}``.
    """
    tails = sorted({r["tail"] for r in per_run_rows})
    out: Dict[str, Dict[str, Any]] = {}
    for tail in tails:
        rows = [r for r in per_run_rows if r["tail"] == tail]
        agg: Dict[str, Dict[str, float]] = {}
        for k in BOOKS:
            agg[k] = {
                "abs_beta_far": _nanmean([abs(r[k]["basis"]["beta_far"]) for r in rows]),
                "abs_beta_mo": _nanmean([abs(r[k]["basis"]["beta_mo"]) for r in rows]),
                "basis_r2": _nanmean([r[k]["basis"]["basis_r2"] for r in rows]),
                "cost_bp": _nanmean([r[k]["cost_bp"] for r in rows]),
            }
        agg["b_minus_a"] = {m: agg["b"][m] - agg["a"][m] for m in agg["a"]}
        out[tail] = agg
    return out
