"""Stage 4: aggregate the backtest runs and render the report (design 4.5-4.6).

Reads ``OUTPUT_DIR/runs/<inception>__<tail>.json`` (``03_backtest.py``; ``*.error.json`` are failed runs,
``*.tmp`` are ignored) and ``results/stability.json`` (``02_stability.py``); computes everything through
``s2ts.measures``; writes ``results/backtest.json`` (per-run summaries, paired tables, verdict, basis
mitigation, parity flags, missing/truncated lists) and ``results/report.html`` (Bloomberg x Swiss,
``docs/UI_LANGUAGE.md``: frozen navigator, light/dark switch, inline SVG, self-contained).

It works on a partial set of runs: a missing tail or inception is listed as missing, never an error, so it
can be re-run while the backtest is still going.

Run:
    python example/case_studies/snowball_2y_tail_stability/04_report.py
    python example/case_studies/snowball_2y_tail_stability/04_report.py --runs-dir <dir> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from quantark.util.numerical import is_zero

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s2ts import ANCHOR_WINDOW, NOTIONAL, OUTPUT_DIR, RATE, RESULTS_DIR  # noqa: E402
from s2ts import html as s2html  # noqa: E402
from s2ts.measures import (  # noqa: E402
    HOLD_FORWARD,
    basis_mitigation,
    paired,
    paired_complete,
    per_run,
    qualifying_tails,
    terminal_paired,
    verdict,
)
from s2ts.tails import TAILS  # noqa: E402

BOOKS = ("a", "b")
CURVE_GRID = np.linspace(0.05, 2.0, 40)
NAN = float("nan")


def _bp(x: float) -> float:
    return 1e4 * float(x) / NOTIONAL


def _isnan(x: Optional[float]) -> bool:
    return x is None or not math.isfinite(x)


def _nanmean(x: Sequence[float]) -> float:
    ok = [v for v in x if not _isnan(v)]
    return float(np.mean(ok)) if ok else NAN


def clean(obj: Any) -> Any:
    """Recursively turn NaN/inf into ``None`` and numpy scalars into Python ones (JSON has no NaN)."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_runs(runs_dir: Path) -> tuple[List[Dict[str, Any]], List[str]]:
    """Completed run dicts and the ``<inception>__<tail>`` names of failed runs."""
    runs: List[Dict[str, Any]] = []
    failed: List[str] = []
    for path in sorted(Path(runs_dir).glob("*.json")):
        if path.name.endswith(".error.json"):
            failed.append(path.name[: -len(".error.json")])
        else:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
    return runs, failed


def known_inceptions(runs_dir: Path, runs: Sequence[Mapping[str, Any]]) -> List[str]:
    """Inceptions the backtest is meant to cover: ``coupons.json`` next to ``runs/`` plus any seen in runs."""
    found = {r["inception"] for r in runs}
    coupons = Path(runs_dir).parent / "coupons.json"
    if coupons.exists():
        found |= set(json.loads(coupons.read_text(encoding="utf-8")))
    return sorted(found)


def compute_curve() -> Optional[Dict[str, Any]]:
    """q(T) to 2Y for the six tails on the latest day with an anchor; ``None`` if the chain pickle is absent."""
    from s2ts import CHAIN_PATH
    from s2ts.chain import load_chain
    from s2ts.tails import anchor_series, build_curve

    if not CHAIN_PATH.exists():
        return None
    chains = load_chain(CHAIN_PATH)
    anchors = anchor_series(chains, ANCHOR_WINDOW)
    dc = next(c for c in reversed(chains) if c.day in anchors)
    q = {t: [RATE - float(build_curve(dc, t, anchors[dc.day]).carry(float(T))) / float(T) for T in CURVE_GRID]
         for t in TAILS}
    return {"day": dc.day.isoformat(), "grid": [float(T) for T in CURVE_GRID], "q": q}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _ratio_labels(tail_runs: Sequence[Mapping[str, Any]], hf_runs: Sequence[Mapping[str, Any]], book: str) -> List[str]:
    """Inception labels aligned with ``paired()[tail]['ratios']`` (same filtering, same order)."""
    t = {r["inception"]: per_run(r)[book]["std_bp"] for r in tail_runs}
    h = {r["inception"]: per_run(r)[book]["std_bp"] for r in hf_runs}
    common = sorted(i for i in t if i in h and not _isnan(t[i]) and not _isnan(h[i]))
    return [i for i in common if not is_zero(h[i])]


def _terminal_stats(rows: Sequence[Mapping[str, Any]], book: str) -> Dict[str, float]:
    done = [r for r in rows if not r["truncated"]]
    tv = [_bp(r[f"terminal_{book}"]) for r in done]
    own = [_bp(r[f"terminal_own_{book}"]) for r in done]
    return {"std": float(np.std(tv, ddof=1)) if len(tv) >= 2 else NAN, "mean": float(np.mean(tv)) if tv else NAN,
            "own_mean": float(np.mean(own)) if own else NAN, "n": len(tv)}


def _run_means(rows: Sequence[Mapping[str, Any]], book: str) -> Dict[str, float]:
    stats = [per_run(r)[book] for r in rows]
    return {
        "r2": _nanmean([s["r2"] for s in stats]), "turn": _nanmean([s["turnover_lots_per_day"] for s in stats]),
        "roll_abs": _nanmean([s["roll_day_abs_bp"] for s in stats]), "max_dd": _nanmean([s["max_dd_bp"] for s in stats]),
        "cost": _nanmean([s["cost_bp"] for s in stats]),
        "vr_pooled_std": float(math.sqrt(_nanmean([s["std_vega_resid_bp"] ** 2 for s in stats]))) if stats else NAN,
        "pooled_all": float(math.sqrt(_nanmean([s["std_bp"] ** 2 for s in stats]))) if stats else NAN,
    }


def build_data(stability: Optional[Mapping[str, Any]], runs: Sequence[Mapping[str, Any]], failed: Sequence[str],
               inceptions: Sequence[str], curve: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Everything ``s2ts.html.render`` and ``backtest.json`` need, from loaded inputs only."""
    by_tail: Dict[str, List[Mapping[str, Any]]] = {}
    for r in runs:
        by_tail.setdefault(r["tail"], []).append(r)
    present = [t for t in TAILS if t in by_tail]
    incs = sorted(set(inceptions) | {r["inception"] for r in runs})
    failed_keys = set(failed)
    coverage: Dict[str, str] = {}
    cells_missing: List[List[str]] = []
    by_cell = {(r["inception"], r["tail"]): r for r in runs}
    for i in incs:
        for t in TAILS:
            run = by_cell.get((i, t))
            if run is not None:
                coverage[f"{i}|{t}"] = "trunc" if run["truncated"] else "done"
            else:
                coverage[f"{i}|{t}"] = "failed" if f"{i}__{t}" in failed_keys else "missing"
                cells_missing.append([i, t])
    pairing = HOLD_FORWARD in by_tail and bool(by_tail[HOLD_FORWARD])
    data: Dict[str, Any] = {
        "meta": {"n_runs": len(runs), "n_expected": len(incs) * len(TAILS), "notional": NOTIONAL, "rate": RATE,
                 "chain_last_day": (stability or {}).get("meta", {}).get("chain_last_day", ""),
                 "runs_inceptions": sorted({r["inception"] for r in runs})},
        "tails": present, "all_tails": list(TAILS), "inceptions": incs, "coverage": coverage,
        "pairing_available": pairing, "stability": stability, "curve": curve,
        "missing": {"tails": [t for t in TAILS if t not in by_tail], "cells": cells_missing},
        "truncated": sorted(f"{r['inception']}__{r['tail']}" for r in runs if r["truncated"]),
        "failed": sorted(failed), "per_run": [], "hedge": {"a": {}, "b": {}}, "verdict": {}, "crit": {},
        "qualifying": [], "gap": {}, "basis": {},
        "end_reasons": dict(Counter(r["end_reason"] for r in by_tail.get(HOLD_FORWARD, []))),
    }
    # per-run summaries and parity
    parity_rows = []
    for r in sorted(runs, key=lambda x: (x["tail"], x["inception"])):
        pr = per_run(r)
        data["per_run"].append({"tail": r["tail"], "inception": r["inception"], "truncated": r["truncated"],
                                "end_reason": r["end_reason"], "live_days": r["live_days"], "coupon": r["coupon"],
                                "day0_pv_bp": _bp(r["day0_pv"]), "a": pr["a"], "b": pr["b"]})
        if r.get("parity"):
            p = r["parity"]
            parity_rows.append({"inception": r["inception"], "tail": r["tail"], "flag": bool(p["flag"]),
                                "pv_diff_bp": abs(float(p["pv_diff_bp"])), "max_lot_diff": float(p["max_lot_diff"])})
    data["parity"] = {"rows": parity_rows, "n_flag": sum(1 for p in parity_rows if p["flag"])}
    # basis mitigation (works without the baseline)
    data["basis"] = basis_mitigation([{"tail": r["tail"], **per_run(r)} for r in runs]) if runs else {}
    if not pairing:
        data["verdict"], data["qualifying"] = {}, []
        return clean(data)
    # inception-day model gap vs hold-forward on the same inception
    hf_pv = {r["inception"]: r["day0_pv"] for r in by_tail[HOLD_FORWARD]}
    for t in present:
        if t == HOLD_FORWARD:
            continue
        gaps = [_bp(r["day0_pv"] - hf_pv[r["inception"]]) for r in by_tail[t] if r["inception"] in hf_pv]
        if gaps:
            data["gap"][t] = {"mean": float(np.mean(gaps)), "min": float(min(gaps)), "max": float(max(gaps)), "n": len(gaps)}
    pair, term, pair_c = {}, {}, {}
    for k in BOOKS:
        pair[k], term[k], pair_c[k] = paired(by_tail, k), terminal_paired(by_tail, k), paired_complete(by_tail, k)
        for t in present:
            means = _run_means(by_tail[t], k)
            ts = _terminal_stats(by_tail[t], k)
            row = {"r2": means["r2"], "turn": means["turn"], "roll_abs": means["roll_abs"], "max_dd": means["max_dd"],
                   "cost": means["cost"], "term_std": ts["std"], "term_mean": ts["mean"], "term_own_mean": ts["own_mean"],
                   "n_complete": ts["n"], "n_abs_lower": None, "vr_pooled_std": means["vr_pooled_std"], "vr_ratio": None,
                   "ratios": [], "ratio_incs": []}
            if t == HOLD_FORWARD:
                row.update({"pooled_std": means["pooled_all"], "hf_pooled_std": means["pooled_all"], "n_lower": None,
                            "n": len(by_tail[t]), "p_raw": None, "p_holm": None})
            else:
                p, tm = pair[k][t], term[k][t]
                labels = _ratio_labels(by_tail[t], by_tail[HOLD_FORWARD], k)
                row.update({"pooled_std": p["pooled_std_bp"], "hf_pooled_std": p["hf_pooled_std_bp"], "n_lower": p["n_lower"],
                            "n": p["n"], "p_raw": p["p_raw"], "p_holm": p["p_holm"], "n_abs_lower": tm["n_abs_lower"],
                            "n_complete": tm["n_complete"], "term_std": tm["terminal_std_bp"],
                            "term_mean": tm["terminal_mean_bp"],
                            "vr_ratio": (p["vega_resid"]["pooled_std_bp"] / p["vega_resid"]["hf_pooled_std_bp"]
                                         if p["vega_resid"]["hf_pooled_std_bp"] else NAN),
                            "vr_pooled_std": p["vega_resid"]["pooled_std_bp"],
                            "ratios": list(p["ratios"]),
                            "ratio_incs": labels if len(labels) == len(p["ratios"]) else [f"#{n + 1}" for n in range(len(p["ratios"]))]})
            data["hedge"][k][t] = row
    verd = verdict(pair["a"], pair["b"], term["a"], term["b"], pair_c["a"], pair_c["b"])
    data["verdict"] = verd
    data["qualifying"] = qualifying_tails(verd, pair["a"])
    for t, v in verd.items():
        data["crit"][t] = {}
        for k in BOOKS:
            p = pair[k][t]
            data["crit"][t][k] = {
                "a": bool(p["n"] > 0 and p["pooled_std_bp"] < p["hf_pooled_std_bp"]),
                "b": bool(2 * p["n_lower"] > p["n"]) if p["n"] else False,
                "c": None if v[f"book_{k}"] == "not_decidable" else bool(v[f"terminal_ok_{k}"]),
            }
    return clean(data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", type=Path, default=OUTPUT_DIR / "runs", help="directory of <inception>__<tail>.json")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="where backtest.json and report.html are written")
    args = ap.parse_args(argv)
    stab_path = RESULTS_DIR / "stability.json"
    stability = json.loads(stab_path.read_text(encoding="utf-8")) if stab_path.exists() else None
    runs, failed = load_runs(args.runs_dir)
    data = build_data(stability, runs, failed, known_inceptions(args.runs_dir, runs), compute_curve())
    write_json(args.out_dir / "backtest.json", {k: v for k, v in data.items() if k != "stability"})
    out = args.out_dir / "report.html"
    out.write_text(s2html.render(data), encoding="utf-8")
    print(f"{len(runs)} runs, {len(failed)} failed, missing tails: {data['missing']['tails'] or 'none'}")
    print(f"qualifying tails: {data['qualifying'] or 'none'}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
