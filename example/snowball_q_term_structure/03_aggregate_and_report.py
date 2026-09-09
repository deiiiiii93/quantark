"""Stage 03 - Aggregate the fleet and write the study report.

Consumes the per-run artifacts of stage 02 (and the static outputs of stage
01 when present) and produces, under ``--data-dir`` (persisted):

    fleet_per_run.csv            one row per inception x cell with every hedge measure
    fleet_cells.json             per-cell distributions of each measure
    fleet_paired.csv             paired (same inception) differences vs the baseline cell
    fleet_summary.json           everything the report is built from
    q_term_structure_report.html the study report (self-contained; charts inline)

Why paired: every cell of one inception sells the SAME contract on the SAME
spot path with the SAME vol channel, so the difference between two cells is
attributable to the q model / hedge contract alone.  Pooled distributions
would be dominated by which inceptions happened to knock out.

Run:
    .venv/bin/python example/snowball_q_term_structure/03_aggregate_and_report.py
    .venv/bin/python example/snowball_q_term_structure/03_aggregate_and_report.py --run-dir output/snowball_q_term_structure
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402

MEASURES: Tuple[Tuple[str, str, str], ...] = (
    # key, label, unit
    ("terminal_pnl_bp", "Terminal hedged P&L", "bp"),
    ("daily_pnl_std_bp", "Daily hedged P&L std", "bp"),
    ("variance_reduction_r2", "Hedge variance reduction R²", ""),
    ("max_drawdown_bp", "Max drawdown of hedged P&L", "bp"),
    ("turnover", "Turnover (× notional)", "x"),
    ("cost_bp", "Transaction cost", "bp"),
    ("roll_day_mtm_jump_bp", "|ΔMTM| on hedge-roll days", "bp"),
    ("other_day_mtm_jump_bp", "|ΔMTM| on other days", "bp"),
    ("roll_day_q_jump", "|Δq| on hedge-roll days", ""),
    ("delta_churn", "Contracts traded per day", "hands"),
)
PAIRED_KEYS = tuple(k for k, _, _ in MEASURES)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_fleet(run_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    manifest_path = run_dir / "fleet_manifest.json"
    if not manifest_path.exists():
        raise C.StudyDataError(f"no fleet manifest at {manifest_path}; run stage 02 first")
    manifest = json.loads(manifest_path.read_text())
    rows: List[Dict[str, Any]] = []
    for entry in manifest.get("runs", []):
        if entry.get("status") not in ("ok", "resumed"):
            continue
        task = entry["task"]
        run = C.load_run(C.run_dir_for(run_dir, task["inception_tag"], task["model"], task["hedge"]))
        summary = run["summary"]
        measures = C.hedge_measures(run["states"], run["trades"], notional=float(summary["notional"]))
        rows.append(
            {
                "inception": task["inception"],
                "model": task["model"],
                "hedge": task["hedge"],
                "cell": C.cell_name(task["model"], task["hedge"]),
                "coupon": summary["coupon"],
                "s0": summary["s0"],
                "termination_reason": summary.get("termination_reason"),
                "days_replayed": summary.get("days_replayed"),
                "knocked_in": summary.get("knocked_in"),
                "knocked_out": summary.get("knocked_out"),
                "matured": summary.get("matured"),
                "censored": task.get("censored"),
                "final_product_pnl_bp": summary["final_product_pnl"] / summary["notional"] * 1e4,
                "final_hedge_pnl_bp": summary["final_hedge_pnl"] / summary["notional"] * 1e4,
                "elapsed_seconds": summary.get("elapsed_seconds"),
                **measures,
            }
        )
    if not rows:
        raise C.StudyDataError("no completed runs in the manifest")
    return manifest, rows


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def lifecycle_consistency(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Every cell of an inception must terminate identically: the lifecycle
    depends on the spot path only.  Anything else is a defect, not a result."""
    by_inception: Dict[str, set] = {}
    for r in rows:
        by_inception.setdefault(r["inception"], set()).add(
            (r["termination_reason"], r["days_replayed"], bool(r["knocked_in"]))
        )
    mismatched = sorted(k for k, v in by_inception.items() if len(v) > 1)
    return {"inceptions": len(by_inception), "mismatched": mismatched, "consistent": not mismatched}


def cell_summaries(rows: Sequence[Dict[str, Any]], cells: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cell in cells:
        sub = [r for r in rows if r["cell"] == cell]
        if not sub:
            continue
        entry: Dict[str, Any] = {"n": len(sub), "model": sub[0]["model"], "hedge": sub[0]["hedge"]}
        for key, _, _ in MEASURES:
            entry[key] = C.describe([r[key] for r in sub])
        entry["outcomes"] = dict(pd.Series([r["termination_reason"] for r in sub]).value_counts())
        entry["knocked_in_share"] = float(np.mean([bool(r["knocked_in"]) for r in sub]))
        out[cell] = entry
    return out


def paired_rows(rows: Sequence[Dict[str, Any]], pairs: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """One row per (pair, inception) with variant - base for every measure."""
    index = {(r["cell"], r["inception"]): r for r in rows}
    out: List[Dict[str, Any]] = []
    for variant, base in pairs:
        for r in rows:
            if r["cell"] != variant:
                continue
            b = index.get((base, r["inception"]))
            if b is None:
                continue
            row = {"variant": variant, "base": base, "inception": r["inception"],
                   "termination_reason": r["termination_reason"]}
            for key in PAIRED_KEYS:
                a, c = r.get(key), b.get(key)
                row[f"d_{key}"] = (
                    float(a) - float(c)
                    if a is not None and c is not None and math.isfinite(float(a)) and math.isfinite(float(c))
                    else np.nan
                )
            out.append(row)
    return out


def paired_summary(paired: Sequence[Dict[str, Any]], pairs: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for variant, base in pairs:
        sub = [p for p in paired if p["variant"] == variant and p["base"] == base]
        if not sub:
            continue
        entry: Dict[str, Any] = {"n": len(sub), "base": base}
        for key in PAIRED_KEYS:
            entry[key] = C.describe([p[f"d_{key}"] for p in sub])
        out[variant] = entry
    return out


def default_pairs(cells: Sequence[str]) -> List[Tuple[str, str]]:
    base = C.cell_name(C.BASELINE_MODEL, "front")
    pairs = [(c, base) for c in cells if c != base and base in cells]
    # same model, far vs front: isolates the hedge contract for the term
    # models.  NOT for flat_from_hedge, whose q is inverted from the hedged
    # contract, so its far-vs-front pair moves the carry model too.
    for model in C.Q_MODEL_ORDER:
        far, front = C.cell_name(model, "far"), C.cell_name(model, "front")
        if far in cells and front in cells and (far, front) not in pairs:
            pairs.append((far, front))
    return pairs


def aggregate(run_dir: Path, data_dir: Path) -> Dict[str, Any]:
    manifest, rows = load_fleet(run_dir)
    cells = [c for c in manifest.get("cells", []) if any(r["cell"] == c for r in rows)]
    pairs = default_pairs(cells)
    paired = paired_rows(rows, pairs)
    static_path = data_dir / "static_summary.json"
    static = json.loads(static_path.read_text()) if static_path.exists() else None
    failed = [r for r in manifest.get("runs", []) if r.get("status") == "failed"]
    return {
        "schema_version": C.SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "config": manifest.get("config", {}),
        "models": manifest.get("models", {}),
        "cells": cells,
        "inceptions": sorted({r["inception"] for r in rows}),
        "runs_ok": len(rows),
        "runs_failed": [
            {"inception": f["task"]["inception"], "cell": C.cell_name(f["task"]["model"], f["task"]["hedge"]), "error": f.get("error")}
            for f in failed
        ],
        "lifecycle_consistency": lifecycle_consistency(rows),
        "cell_summaries": cell_summaries(rows, cells),
        "pairs": [{"variant": v, "base": b} for v, b in pairs],
        "paired_summary": paired_summary(paired, pairs),
        "per_run": rows,
        "paired": paired,
        "static": static,
    }


def write_tables(agg: Dict[str, Any], data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(agg["per_run"]).to_csv(data_dir / "fleet_per_run.csv", index=False)
    pd.DataFrame(agg["paired"]).to_csv(data_dir / "fleet_paired.csv", index=False)
    C.write_json(data_dir / "fleet_cells.json", {"cells": agg["cell_summaries"], "paired": agg["paired_summary"]})
    C.write_json(data_dir / "fleet_summary.json", {k: v for k, v in agg.items() if k not in ("per_run", "paired")})


# ---------------------------------------------------------------------------
# Charts (matplotlib -> inline PNG)
# ---------------------------------------------------------------------------


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.figsize": (9, 3.6)})
    return plt


CELL_COLORS = {
    "flat_from_hedge__front": "#7f7f7f", "term_flat_q__front": "#1f77b4", "term_flat_fwd__front": "#17becf",
    "surface_fwd__front": "#9467bd", "flat_from_hedge__far": "#bcbd22", "term_flat_q__far": "#2ca02c",
    "term_flat_fwd__far": "#8c564b", "surface_fwd__far": "#e377c2",
    "term_opt_tail__front": "#d62728", "term_opt_tail__far": "#e377c2",
}


def chart_curve_anatomy(data_dir: Path) -> Optional[str]:
    path = data_dir / "curve_anatomy.csv"
    if not path.exists():
        return None
    plt = _plt()
    df = pd.read_csv(path, parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(df["date"], df["q_flat_front"] * 100, color="#7f7f7f", lw=0.8, label="flat q from the front hedge contract (engine default)")
    # the flat-q tail makes q(1Y) equal the longest contract's yield whenever
    # the chain ends inside a year, so that series is drawn once (blue)
    ax.plot(df["date"], df["q1y_term_flat_q"] * 100, color="#1f77b4", lw=1.1, label="term q(1Y), flat-q tail (= longest listed contract)")
    ax.plot(df["date"], df["q1y_term_flat_fwd"] * 100, color="#ff7f0e", lw=0.8, alpha=0.8, label="term q(1Y), flat-forward-carry tail")
    if "q1y_surface_fwd" in df:
        ax.plot(df["date"], df["q1y_surface_fwd"] * 100, color="#9467bd", lw=0.9, ls="--", label="MO option-implied q(1Y)")
    if "q1y_term_opt_tail" in df:
        ax.plot(df["date"], df["q1y_term_opt_tail"] * 100, color="#d62728", lw=0.9, label="term q(1Y), log-forward chain + option-forward tail")
    lo, hi = np.nanpercentile(df["q_flat_front"] * 100, [0.5, 99.5])
    ax.set_ylim(min(lo, -5), max(hi, 30))
    ax.set_ylabel("implied dividend / carry yield, % p.a.")
    ax.set_title("What the pricer is told the carry is")
    ax.legend(loc="upper left", fontsize=7.5)
    return _png(fig)


def chart_forward_error(data_dir: Path) -> Optional[str]:
    path = data_dir / "curve_anatomy.csv"
    if not path.exists():
        return None
    plt = _plt()
    df = pd.read_csv(path, parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.plot(df["date"], df["fwd_err_rms_flat_front"], color="#7f7f7f", lw=0.8, label="flat q from front contract")
    if "fwd_err_rms_flat_far" in df:
        ax.plot(df["date"], df["fwd_err_rms_flat_far"], color="#bcbd22", lw=0.8, label="flat q from longest contract")
    if "fwd_err_rms_surface" in df:
        ax.plot(df["date"], df["fwd_err_rms_surface"], color="#9467bd", lw=0.8, ls="--", label="MO option forwards vs IM marks")
    ax.axhline(0.0, color="#1f77b4", lw=1.2, label="term q (reprices every contract)")
    ax.set_ylabel("RMS log error vs listed IM marks, bp")
    ax.set_title("Forward-pricing error of each carry model at the listed tenors")
    ax.legend(fontsize=7.5)
    return _png(fig)


def chart_ki_probe(data_dir: Path) -> Optional[str]:
    path = data_dir / "ki_probe_roll_window.csv"
    if not path.exists():
        return None
    plt = _plt()
    df = pd.read_csv(path, parse_dates=["date"])
    colours = {"flat_from_hedge": "#7f7f7f", "term_flat_q": "#1f77b4", "term_flat_fwd": "#ff7f0e", "surface_fwd": "#9467bd",
               "term_opt_tail": "#d62728"}
    # trading-day positions on the x axis: a holiday gap must not stretch the lines
    days = sorted(df["date"].unique())
    pos = {d: i for i, d in enumerate(days)}
    fig, (ax_p, ax_q) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for name, g in df.groupby("model", sort=False):
        g = g.sort_values("date")
        c = colours.get(name, None)
        x = [pos[d] for d in g["date"]]
        ax_p.plot(x, g["p_ki"], marker="o", ms=3.5, lw=1.2, color=c, label=name)
        ax_q.plot(x, g["q_T"] * 100, marker="o", ms=3.5, lw=1.2, color=c, label=name)
    for r in df.loc[df["roll_front"], "date"].unique():
        ax_p.axvline(pos[r], color="#d62728", lw=0.8, ls=":")
        ax_q.axvline(pos[r], color="#d62728", lw=0.8, ls=":")
    ax_q.set_xticks(range(len(days)))
    ax_q.set_xticklabels([pd.Timestamp(d).strftime("%m-%d") for d in days], fontsize=8)
    ax_q.set_xlabel(f"trading day ({pd.Timestamp(days[0]).year})")
    ax_p.set_ylim(-0.02, 1.02)
    ax_p.set_ylabel("P(KI ever), pricing measure")
    ax_p.set_title("Same product, same spot, same vol: only the carry input changes day by day")
    ax_p.legend(fontsize=7.5, loc="upper left")
    ax_q.set_ylabel("q(T) handed to the pricer, % p.a.")
    ax_q.axhline(0.0, color="black", lw=0.5)
    return _png(fig)


def chart_paired_strip(agg: Dict[str, Any], key: str, title: str, unit: str) -> Optional[str]:
    paired = [p for p in agg["paired"] if p["base"] == C.cell_name(C.BASELINE_MODEL, "front")]
    if not paired:
        return None
    plt = _plt()
    variants = [p["variant"] for p in agg["pairs"] if p["base"] == C.cell_name(C.BASELINE_MODEL, "front")]
    fig, ax = plt.subplots(figsize=(9, 0.6 + 0.55 * len(variants) + 1.5))
    for i, v in enumerate(variants):
        vals = [p[f"d_{key}"] for p in paired if p["variant"] == v and math.isfinite(p[f"d_{key}"])]
        if not vals:
            continue
        y = np.full(len(vals), i) + np.random.default_rng(1).uniform(-0.12, 0.12, len(vals))
        ax.scatter(vals, y, s=14, color=CELL_COLORS.get(v, "#333"), alpha=0.75)
        ax.plot([np.mean(vals)] * 2, [i - 0.3, i + 0.3], color="black", lw=1.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)
    ax.invert_yaxis()
    ax.set_xlabel(f"variant − {C.cell_name(C.BASELINE_MODEL, 'front')}, {unit} (one dot per inception; bar = mean)")
    ax.set_title(title)
    return _png(fig)


def chart_example_path(run_dir: Path, agg: Dict[str, Any]) -> Optional[str]:
    """Product MTM and hedge contract under the baseline and the term model
    for the inception with the most hedge-roll days (the most visible case)."""
    base, term = C.cell_name(C.BASELINE_MODEL, "front"), C.cell_name(C.REFERENCE_MODEL, "front")
    candidates = [r for r in agg["per_run"] if r["cell"] == base]
    if not candidates or not any(r["cell"] == term for r in agg["per_run"]):
        return None
    pick = max(candidates, key=lambda r: (r["roll_days"], r["days_replayed"]))
    tag = pick["inception"].replace("-", "")
    try:
        rb = C.load_run(C.run_dir_for(run_dir, tag, C.BASELINE_MODEL, "front"))
        rt = C.load_run(C.run_dir_for(run_dir, tag, C.REFERENCE_MODEL, "front"))
    except C.StudyDataError:
        return None
    plt = _plt()
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    sb, st = rb["states"], rt["states"]
    n = float(pick.get("s0", 1.0))
    axes[0].plot(sb.index, sb["product_mtm"] / C.NOTIONAL * 1e4, color="#7f7f7f", lw=0.9, label=base)
    axes[0].plot(st.index, st["product_mtm"] / C.NOTIONAL * 1e4, color="#1f77b4", lw=0.9, label=term)
    rolls = sb.index[sb["active_contract"].astype(str).ne(sb["active_contract"].astype(str).shift()).to_numpy() & (np.arange(len(sb)) > 0)]
    for d in rolls:
        for ax in axes:
            ax.axvline(d, color="#d62728", lw=0.6, alpha=0.5)
    axes[0].set_ylabel("product MTM, bp of notional")
    axes[0].set_title(f"Inception {pick['inception']}: the same contract priced under two carry models (red = hedge roll)")
    axes[0].legend(fontsize=7.5)
    axes[1].plot(sb.index, sb["pricing_q"] * 100, color="#7f7f7f", lw=0.9, label="pricing q (flat, active contract)")
    axes[1].plot(st.index, st["pricing_q"] * 100, color="#1f77b4", lw=0.9, label="pricing q (term, at remaining maturity)")
    axes[1].set_ylabel("q handed to the pricer, %")
    axes[1].legend(fontsize=7.5)
    axes[2].plot(sb.index, sb["total_pnl"] / C.NOTIONAL * 1e4, color="#7f7f7f", lw=0.9, label=base)
    axes[2].plot(st.index, st["total_pnl"] / C.NOTIONAL * 1e4, color="#1f77b4", lw=0.9, label=term)
    axes[2].set_ylabel("hedged P&L, bp of notional")
    axes[2].legend(fontsize=7.5)
    del n
    return _png(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(v: Any, digits: int = 1, pct: bool = False) -> str:
    if v is None:
        return "–"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    if not math.isfinite(f):
        return "–"
    return f"{f * 100:.{digits}f}%" if pct else f"{f:,.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], caption: str = "") -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    return f'<div class="scroll"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _figure(src: Optional[str], caption: str) -> str:
    if not src:
        return ""
    return f'<figure><img src="{src}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def _sig(desc: Dict[str, Any]) -> str:
    t = desc.get("t_stat")
    if t is None:
        return ""
    return " ***" if abs(t) > 3.3 else " **" if abs(t) > 2.6 else " *" if abs(t) > 2.0 else ""


def _verdict(agg: Dict[str, Any]) -> str:
    base = C.cell_name(C.BASELINE_MODEL, "front")
    term = C.cell_name(C.REFERENCE_MODEL, "front")
    ps = agg["paired_summary"].get(term)
    cs = agg["cell_summaries"]
    if not ps or base not in cs or term not in cs:
        return "<p>The baseline and reference cells are not both present; no verdict.</p>"
    n = ps["n"]
    parts = []
    std = ps["daily_pnl_std_bp"]
    jump = ps["roll_day_mtm_jump_bp"]
    pnl = ps["terminal_pnl_bp"]
    churn = ps["delta_churn"]
    r2 = ps["variance_reduction_r2"]
    parts.append(
        f"Across {n} paired inceptions, replacing the flat active-contract yield with the IM-chain term structure "
        f"changed the daily hedged P&L standard deviation by {_fmt(std['mean'], 2)} bp on average "
        f"(median {_fmt(std['median'], 2)} bp, positive in {_fmt(std['share_positive'], 0, pct=True)} of inceptions{_sig(std)})."
    )
    parts.append(
        f"The re-mark on hedge-roll days, |ΔMTM|, moved by {_fmt(jump['mean'], 2)} bp on average "
        f"({_fmt(cs[base]['roll_day_mtm_jump_bp']['mean'], 2)} bp under the flat model vs "
        f"{_fmt(cs[term]['roll_day_mtm_jump_bp']['mean'], 2)} bp under the term model, against "
        f"{_fmt(cs[base]['other_day_mtm_jump_bp']['mean'], 2)} / {_fmt(cs[term]['other_day_mtm_jump_bp']['mean'], 2)} bp on ordinary days)."
    )
    parts.append(
        f"Terminal hedged P&L differed by {_fmt(pnl['mean'], 1)} bp of notional on average "
        f"(std {_fmt(pnl['std'], 1)} bp{_sig(pnl)}); hedge variance reduction R² by {_fmt(r2['mean'], 3)}; "
        f"contracts traded per day by {_fmt(churn['mean'], 2)}."
    )
    parts.append(
        "Read the terminal P&L as the noisy, path-dependent number it is: one inception's knock-out timing "
        "dominates it. The day-to-day measures (std, roll-day re-mark, churn) are where a carry model shows "
        "up, because they are averaged over hundreds of hedging days per inception."
    )
    return "<p>" + " ".join(parts) + "</p>"


def _dl(items: Sequence[Tuple[str, str]]) -> str:
    """Definition list: (term, meaning) pairs, meanings may carry inline HTML."""
    return "<dl class='terms'>" + "".join(
        f"<dt>{t}</dt><dd>{d}</dd>" for t, d in items
    ) + "</dl>"


def _appendix(agg: Dict[str, Any]) -> str:
    """Section 6: what every column of every table means.

    Driven by the fleet manifest config so the stated notional, cost and
    grid can never drift from the run the report was built from.
    """
    cfg = agg.get("config", {})
    notional = float(cfg.get("notional", C.NOTIONAL))
    cost_bp = cfg.get("cost_bp", 1.0)
    mult = C.FUTURES_MULTIPLIER
    ref = C.REFERENCE_MODEL
    one_bp = notional / 1e4

    conventions = _dl([
        ("bp",
         f"One basis point of the product notional ({notional:,.0f}), so 1 bp = {one_bp:,.0f} of currency. "
         "Every column labelled bp is a currency amount divided by the notional."),
        ("hands",
         f"One IM futures contract. A contract pays {mult:,.0f} of currency per index point, so "
         f"hands = currency delta / {mult:,.0f}. The fleet trades whole hands only."),
        ("Whose book",
         "Section 2 prices the <strong>long holder</strong> (the buyer): a positive PV is value to the buyer. "
         "The fleet in section 3 books the <strong>seller</strong> (product quantity &minus;1), who is short the "
         "index and hedges by going long futures. A +24-hand delta in section 2 is the seller buying 24 hands."),
        ("q, carry, dividend yield",
         "One object throughout: the continuously-compounded zero yield <code>q(T)</code> that reproduces the "
         "forward, <code>F(T) = S&middot;exp((r &minus; q(T))&middot;T)</code>. For an index futures chain trading at a "
         "discount it is mostly the discount, not a dividend."),
        ("Averaging",
         "Section 1 averages over trading days, section 2 over the monthly grid dates, section 3 over inceptions. "
         "Each table's caption names its sample."),
    ])

    stats = _dl([
        ("n", "Sample size: completed inceptions in a cell, or matched inception pairs in the paired table."),
        ("mean, median, std, min, max",
         "Over the sample the caption names. <code>std</code> is the sample standard deviation (n&minus;1)."),
        ("&ldquo;x &gt; 0&rdquo;",
         "Share of the sample strictly above zero, printed under the mean in the paired table. It says how "
         "consistent an effect is, which a mean alone does not."),
        ("*, **, ***",
         "|t| &gt; 2.0, 2.6, 3.3 where t = mean / (std / &radic;n). <strong>These overstate significance</strong>: "
         "the inception windows overlap, so the paired samples are not independent."),
    ])

    sec1 = _dl([
        ("front / longest contract implied q (cont.)",
         "<code>q_i = r &minus; ln(F_i/S)/T_i</code> for the nearest and the last <em>eligible</em> listed contract. "
         f"Continuous compounding, signed, no floor; T is ACT/365 calendar. Eligible = at least "
         f"{C.FUTURES_CURVE_MIN_TENOR_DAYS} days to expiry, because a contract inside its delivery week has no "
         "measurable annualised carry."),
        ("flat q the engine uses, front / far hedge",
         "What the replay default hands the pricer: <code>max(0, r &minus; basis)</code> with "
         "<code>basis = (F &minus; S)/S/T</code> read off the <em>hedge contract that policy holds</em>. Simple "
         "compounding, floored at zero. This is the object the study is testing."),
        ("term q(1Y), flat-q / flat-forward-carry tail",
         "The zero yield to one year under each term model, i.e. what a fresh 1Y product would be priced with. "
         "The two differ only in how the curve is continued past the last listed tenor."),
        ("last listed tenor, years",
         "T of the longest eligible contract that day. Beyond it nobody quotes carry and the curve is "
         "extrapolated by the model's tail convention."),
        ("Forward-pricing error, RMS bp",
         "How far a model's own forwards sit from the marks it should reprice: "
         "<code>err_i = 10<sup>4</sup>&middot;ln(F_model(T_i) / F_market(T_i))</code> over every live contract, "
         "root-mean-squared within the day, then averaged over days. Equivalently "
         "<code>10<sup>4</sup>&middot;(q_market(T_i) &minus; q_model(T_i))&middot;T_i</code>. Zero for the term models "
         "by construction, since their nodes are inverted from those same marks. A calibration error, not a P&amp;L."),
        ("floored days, hedge rolls, |&Delta;q|",
         "How often the zero floor binds, how often the hedge contract changes, and the mean absolute daily move "
         "of the q handed to the pricer on roll days against ordinary days."),
    ])

    sec2_model = _dl([
        ("PV bp",
         f"Long-holder PV of the fresh contract, bp of notional. Zero by construction under <code>{ref}</code> "
         "(the fair coupon was solved there), so every other row <em>is</em> that model's pricing gap."),
        ("PV gap vs ref bp", f"PV under this model minus PV under <code>{ref}</code>, on the same contract and the same day."),
        ("delta, hands",
         f"The <strong>hedging delta</strong>: <code>&part;PV/&part;S</code> holding <code>q(T)</code> fixed, divided "
         f"by {mult:,.0f}. This is the only delta the fleet trades, rounded to whole contracts."),
        ("delta gap vs ref", "The same delta minus the reference model's, in hands. How much a model over- or under-hedges."),
        ("rhoq bp per +1% q",
         "PV change for a parallel +1 percentage-point shift of the <em>whole</em> q curve, bp of notional. "
         "Measured as a one-sided up bump and rescaled to one point."),
        ("q at maturity",
         "The model's zero yield to the product's maturity: the single number a flat pricer would have to use to "
         "agree with this model on the terminal forward."),
        ("P(KI) mean, P(KO) mean",
         "Probability under the pricing measure that the knock-in barrier is ever breached, and that any monthly "
         "observation knocks out, from the QUAD event recursion on the same contract. Pricing-measure numbers: "
         "they move one-for-one with the forward the carry model implies, and are not a real-world forecast."),
        ("P(KI) gap vs ref, P(KI) gap range", f"This model's P(KI) minus <code>{ref}</code>'s on the same day: mean, then min .. max over the grid."),
        ("corr(gap, q(T) gap)", "Correlation across grid dates between the P(KI) gap and the q-at-maturity gap. Near 1 means the KI error is the carry error and nothing else."),
        ("q-only probe columns",
         "<code>ki_probe_roll_window.csv</code>: one frozen product repriced with each day's carry input only. "
         "q(T) range and P(KI) range are min .. max over the window; |&Delta;| per day is the mean absolute "
         "day-to-day change; PV range is the long-holder mark in bp of notional."),
    ])

    sec2_bucket = _dl([
        ("contract rank (0 = nearest eligible)",
         "Position in that day's listed IM chain, counting from the shortest eligible contract. Ranks are assigned "
         "per date, so the higher ranks have fewer observations: a day with three eligible contracts contributes "
         "no rank 3."),
        ("bucket hands",
         f"<code>&minus;(&part;PV/&part;F_i)/{mult:,.0f}</code> with <strong>spot held fixed</strong>: bump one IM mark by "
         "an index point, rebuild <code>q(T)</code> from the chain, reprice. Spot is held, so the only channel is "
         "the carry node and <code>&part;PV/&part;F_i = &minus;(1/(T_i&middot;F_i))&middot;&part;PV/&part;q(T_i)</code>. "
         "<strong>Carry risk in futures-price units, analysis only.</strong> It is a partial in a different "
         "coordinate system from the hedging delta above, does not decompose it and does not sum to it, and no "
         "cell trades it."),
    ])

    sec3_cell = _dl([
        ("cell", "<code>&lt;carry model&gt;__&lt;hedge contract policy&gt;</code>. Every cell of one inception sells the "
                 "same contract on the same spot path with the same vol channel and the same fair coupon. "
                 "The two fields are independent for the term models, which read the whole chain. They are NOT "
                 "independent for <code>flat_from_hedge</code>: the engine inverts whichever contract the roll "
                 "policy holds, so <code>flat_from_hedge__far</code> prices off the longest listed contract and "
                 "<code>flat_from_hedge__front</code> off the front month."),
        ("P&amp;L mean bp, P&amp;L std bp",
         "Mean and cross-inception standard deviation of the <strong>terminal hedged P&amp;L</strong>: product + hedge "
         "&minus; costs at termination, seller's book, bp of notional. Every run is booked at the traded price of "
         "zero, so this is hedge error, not a margin. Path-dependent and dominated by knock-out timing."),
        ("daily std bp",
         "Standard deviation of the <strong>daily increments</strong> of that same hedged P&amp;L within a run, then "
         "averaged over inceptions. The day-to-day error a desk actually lives with, and where a carry model shows."),
        ("R&sup2;",
         "<code>1 &minus; var(daily hedged P&amp;L) / var(daily product P&amp;L)</code> per run: the share of the "
         "product's daily variance the futures hedge removes. Not a regression R&sup2;; it can go negative if a "
         "hedge adds variance."),
        ("max DD bp", "Largest peak-to-trough fall of the cumulative hedged P&amp;L inside a run."),
        ("turnover &times;",
         "&Sigma;|traded notional| over the run divided by the product notional, counting both rebalance and roll "
         "legs. Reported in multiples of notional, not currency."),
        ("cost bp",
         f"Cumulative transaction cost charged over the run, at {cost_bp} bp per side proportional to traded notional."),
        ("|&Delta;MTM| roll days bp / other days bp",
         "Mean absolute day-on-day change in the <em>product's</em> mark, split by whether the active hedge contract "
         "changed that day. A pricer whose q is tied to the contract it holds re-marks the whole book on every "
         "roll for no economic reason, and the gap between these two columns is that artefact."),
        ("hands/day",
         "Delta churn: mean absolute contracts traded per replay day, <strong>rebalance legs only</strong>. Roll "
         "legs are excluded because a roll closes and reopens the whole position for calendar reasons, which would "
         "swamp the hedging signal."),
        ("|&Delta;q| on hedge-roll days",
         "In <code>fleet_per_run.csv</code>: mean absolute jump of the q handed to the pricer on roll days. The "
         "mechanism behind the roll-day re-mark."),
    ])

    sec3_paired = _dl([
        ("variant, base",
         "The two cells being differenced. Differences are taken on <strong>matched inceptions</strong>, so contract, "
         "spot path, vol channel and fair coupon are identical and the difference is attributable to the carry "
         "model or the hedge contract alone. Pooled distributions would instead be dominated by which inceptions "
         "happened to knock out."),
        ("n", "Inceptions present in both cells."),
        ("&Delta; &lt;measure&gt;",
         "Variant minus base of the identically-named column above, averaged over the matched inceptions. The "
         "small print under each mean is the share of inceptions where the difference was positive."),
    ])

    return (
        "<h2>6. Appendix: what the columns mean</h2>"
        "<p>Every column of every table above, in the order the tables appear. All of it is persisted: "
        "<code>fleet_per_run.csv</code> (one row per inception &times; cell), <code>fleet_cells.json</code> "
        "(per-cell and paired distributions), <code>curve_anatomy.csv</code>, <code>static_risk.csv</code> and "
        "<code>static_buckets.csv</code> and <code>ki_probe_roll_window.csv</code>.</p>"
        "<h3>Conventions</h3>" + conventions
        + "<h3>Statistics (every table)</h3>" + stats
        + "<h3>Section 1 &mdash; carry anatomy</h3>" + sec1
        + "<h3>Section 2 &mdash; static risk, per model</h3>" + sec2_model
        + "<h3>Section 2 &mdash; futures-tenor carry buckets</h3>" + sec2_bucket
        + "<h3>Section 3 &mdash; per-cell hedging measures</h3>" + sec3_cell
        + "<h3>Section 3 &mdash; paired comparisons</h3>" + sec3_paired
    )

def _ki_section(s: Dict[str, Any], data_dir: Path) -> str:
    """Section 2, second half: the knock-in probability each carry model implies."""
    if not any("p_ki" in m for m in s["models"].values()):
        return ""
    ki_rows = []
    for name in C.Q_MODEL_ORDER:
        m = s["models"].get(name)
        if not m or "p_ki" not in m:
            continue
        gap = m.get("p_ki_gap_vs_reference")
        corr = m.get("p_ki_gap_vs_q_gap_corr")
        ki_rows.append(
            [html.escape(name), _fmt(m["p_ki"]["mean"], 3), _fmt(m["p_ko"]["mean"], 3),
             _fmt(gap["mean"], 3) if gap else "&mdash;",
             f"{_fmt(gap['min'], 3)} .. {_fmt(gap['max'], 3)}" if gap else "&mdash;",
             _fmt(corr, 2) if corr is not None else "&mdash;"]
        )
    out = (
        "<h3>2b. The knock-in probability each carry model implies</h3>"
        "<p>Under the pricing measure P(KI) is fixed by the forward curve, not by a yield as a number. A flat q "
        "extrapolates one contract's forward to the whole life, so its error scales with 1/tenor of that "
        "contract; the term curve pins every listed forward and extrapolates only past the last tenor. "
        "P(KI) is the probability that the 75% barrier is ever breached, P(KO) that any monthly observation "
        "knocks out, both read from the QUAD recursion on the same contract and the same day.</p>"
        + _table(["model", "P(KI) mean", "P(KO) mean", "P(KI) gap vs ref, mean", "P(KI) gap range", "corr(gap, q(T) gap)"],
                 ki_rows, f"Knock-in and knock-out probabilities on the monthly grid ({s['grid_dates']} dates), "
                 f"same fair coupon as above.")
    )
    kp = s.get("ki_probe")
    if kp:
        probe_rows = []
        for name in C.Q_MODEL_ORDER:
            m = kp["models"].get(name)
            if not m:
                continue
            probe_rows.append(
                [html.escape(name), f"{_fmt(m['q_T']['min'], 1, True)} .. {_fmt(m['q_T']['max'], 1, True)}",
                 f"{_fmt(m['p_ki']['min'], 3)} .. {_fmt(m['p_ki']['max'], 3)}", _fmt(m["p_ki_range"], 3),
                 _fmt(m["p_ki_daily_abs_change_mean"], 3), _fmt(m["q_T_daily_abs_change_mean"], 2, True),
                 f"{_fmt(m['pv_bp']['min'])} .. {_fmt(m['pv_bp']['max'])}"]
            )
        out += (
            f"<p><strong>The q-only probe.</strong> One product is frozen at its {kp['inception']} inception "
            f"(spot, vol, terms and the {_fmt(kp['coupon'], 2, True)} fair coupon) and repriced with nothing but "
            f"the carry read off each of the {kp['n_days']} trading days from {kp['first_date']} to {kp['last_date']}, "
            "the window around the largest day-to-day jump of the front contract's annualised basis. Every move in "
            "P(KI) below is caused by the carry input alone.</p>"
            + _table(["model", "q(T) range", "P(KI) range", "P(KI) max &minus; min", "mean |&Delta;P(KI)| per day",
                      "mean |&Delta;q(T)| per day", "PV range bp"], probe_rows,
                     "Day-to-day movement of the frozen product's knock-in probability, by carry model.")
            + _figure(chart_ki_probe(data_dir), "P(KI) of the frozen product (top) and the q(T) each model handed the "
                      "pricer (bottom); dotted lines mark front-hedge roll days.")
        )
    return out


def build_report(agg: Dict[str, Any], run_dir: Path, data_dir: Path) -> str:
    cfg = agg["config"]
    cells = agg["cells"]
    cs = agg["cell_summaries"]
    base = C.cell_name(C.BASELINE_MODEL, "front")

    # --- cell table -------------------------------------------------------
    cell_rows = []
    for cell in cells:
        e = cs.get(cell)
        if not e:
            continue
        cell_rows.append(
            [html.escape(cell), str(e["n"]),
             _fmt(e["terminal_pnl_bp"]["mean"]), _fmt(e["terminal_pnl_bp"]["std"]),
             _fmt(e["daily_pnl_std_bp"]["mean"], 2), _fmt(e["variance_reduction_r2"]["mean"], 3),
             _fmt(e["max_drawdown_bp"]["mean"]), _fmt(e["turnover"]["mean"], 2), _fmt(e["cost_bp"]["mean"], 2),
             _fmt(e["roll_day_mtm_jump_bp"]["mean"], 2), _fmt(e["other_day_mtm_jump_bp"]["mean"], 2),
             _fmt(e["delta_churn"]["mean"], 2)]
        )
    cell_table = _table(
        ["cell", "n", "P&L mean bp", "P&L std bp", "daily std bp", "R²", "max DD bp", "turnover ×", "cost bp",
         "|ΔMTM| roll days bp", "|ΔMTM| other days bp", "hands/day"],
        cell_rows, "Per-cell means over inceptions (bp = basis points of notional).",
    )

    # --- paired table -----------------------------------------------------
    paired_rows_html = []
    for p in agg["pairs"]:
        e = agg["paired_summary"].get(p["variant"])
        if not e or e["base"] != p["base"]:
            continue
        paired_rows_html.append(
            [html.escape(p["variant"]), html.escape(p["base"]), str(e["n"])]
            + [
                f"{_fmt(e[k]['mean'], 2)}{_sig(e[k])}<br><small>{_fmt(e[k]['share_positive'], 0, pct=True)} &gt; 0</small>"
                for k in ("terminal_pnl_bp", "daily_pnl_std_bp", "variance_reduction_r2", "max_drawdown_bp",
                          "turnover", "roll_day_mtm_jump_bp", "delta_churn")
            ]
        )
    paired_table = _table(
        ["variant", "base", "n", "Δ P&L bp", "Δ daily std bp", "Δ R²", "Δ max DD bp", "Δ turnover", "Δ |ΔMTM| roll bp", "Δ hands/day"],
        paired_rows_html,
        "Paired differences (variant − base) on matched inceptions; mean, share positive; * |t|>2, ** |t|>2.6, *** |t|>3.3.",
    )

    # --- static section ---------------------------------------------------
    static_html = ""
    st = agg.get("static")
    if st:
        a = st["anatomy"]
        anatomy_rows = [
            ["front contract implied q (cont.)", _fmt(a["q_front"]["mean"], 2, True), _fmt(a["q_front"]["std"], 2, True), _fmt(a["q_front"]["min"], 1, True), _fmt(a["q_front"]["max"], 1, True)],
            ["longest contract implied q (cont.)", _fmt(a["q_far"]["mean"], 2, True), _fmt(a["q_far"]["std"], 2, True), _fmt(a["q_far"]["min"], 1, True), _fmt(a["q_far"]["max"], 1, True)],
            ["flat q the engine uses, front hedge", _fmt(a["front"]["q_flat"]["mean"], 2, True), _fmt(a["front"]["q_flat"]["std"], 2, True), _fmt(a["front"]["q_flat"]["min"], 1, True), _fmt(a["front"]["q_flat"]["max"], 1, True)],
            ["flat q the engine uses, far hedge", _fmt(a["far"]["q_flat"]["mean"], 2, True), _fmt(a["far"]["q_flat"]["std"], 2, True), _fmt(a["far"]["q_flat"]["min"], 1, True), _fmt(a["far"]["q_flat"]["max"], 1, True)],
            ["term q(1Y), flat-q tail", _fmt(a["q1y_term_flat_q"]["mean"], 2, True), _fmt(a["q1y_term_flat_q"]["std"], 2, True), _fmt(a["q1y_term_flat_q"]["min"], 1, True), _fmt(a["q1y_term_flat_q"]["max"], 1, True)],
            ["term q(1Y), flat-forward-carry tail", _fmt(a["q1y_term_flat_fwd"]["mean"], 2, True), _fmt(a["q1y_term_flat_fwd"]["std"], 2, True), _fmt(a["q1y_term_flat_fwd"]["min"], 1, True), _fmt(a["q1y_term_flat_fwd"]["max"], 1, True)],
            ["last listed tenor, years", _fmt(a["t_last"]["mean"], 2), _fmt(a["t_last"]["std"], 2), _fmt(a["t_last"]["min"], 2), _fmt(a["t_last"]["max"], 2)],
        ]
        if "q1y_term_opt_tail" in a:
            anatomy_rows.insert(
                6, ["term q(1Y), log-forward chain + option-forward tail", _fmt(a["q1y_term_opt_tail"]["mean"], 2, True),
                    _fmt(a["q1y_term_opt_tail"]["std"], 2, True), _fmt(a["q1y_term_opt_tail"]["min"], 1, True), _fmt(a["q1y_term_opt_tail"]["max"], 1, True)],
            )
        facts = (
            f"<ul>"
            f"<li>{a['days']} trading days, {a['first_date']} to {a['last_date']}.</li>"
            f"<li>Front-hedge flat q floored at zero on {a['front']['floored_days']} days ({_fmt(a['front']['floored_share'], 0, True)}); "
            f"{a['front']['roll_days']} hedge rolls. Mean |Δq| on roll days {_fmt(a['front']['q_flat_abs_change_on_roll_days'], 2, True)} "
            f"vs {_fmt(a['front']['q_flat_abs_change_on_other_days'], 2, True)} on other days; the term q(1Y) moves {_fmt(a['q1y_term_flat_q_daily_abs_change_mean'], 2, True)} a day.</li>"
            f"<li>Forward-pricing error at the listed tenors, RMS: front-hedge flat q {_fmt(a['front']['fwd_err_rms_flat']['mean'], 0)} bp, "
            f"far-hedge flat q {_fmt(a['far']['fwd_err_rms_flat']['mean'], 0)} bp, term q 0 bp by construction"
            + (f", MO option forwards {_fmt(a['surface']['fwd_err_rms_surface_vs_futures']['mean'], 0)} bp (a genuine cross-market basis)" if "surface" in a else "")
            + ".</li></ul>"
        )
        static_html = (
            "<h2>1. What the market says about carry</h2>"
            + facts
            + _table(["series", "mean", "std", "min", "max"], anatomy_rows, "Daily implied-yield statistics over the history.")
            + _figure(chart_curve_anatomy(data_dir), "The flat yield from the front hedge contract (grey) against the chain's far yield and the term q(1Y).")
            + _figure(chart_forward_error(data_dir), "How far each carry model's forwards sit from the listed IM marks.")
        )
        if "static" in st:
            s = st["static"]
            model_rows = []
            for name in C.Q_MODEL_ORDER:
                m = s["models"].get(name)
                if not m:
                    continue
                model_rows.append(
                    [html.escape(name), _fmt(m["pv_bp"]["mean"]), _fmt(m.get("pv_gap_vs_reference_bp", {}).get("mean")),
                     _fmt(m["delta_hands"]["mean"]), _fmt(m.get("delta_hands_gap_vs_reference", {}).get("mean")),
                     _fmt(m["rhoq_1pct_bp"]["mean"]), _fmt(m["q_at_maturity"]["mean"], 2, True)]
                )
            static_html += (
                "<h2>2. Static risk of a fresh 1Y snowball</h2>"
                f"<p>On the first trading day of each month ({s['grid_dates']} dates) a fresh 1Y snowball is struck with the fair coupon "
                f"solved under <code>{C.REFERENCE_MODEL}</code>, then priced under every model. Because PV is zero under the reference "
                f"model, the PV under any other model is its pricing gap. On average {_fmt(s['tail_time_share']['mean'], 0, True)} of the "
                f"product's life and {_fmt(s['ko_obs_beyond_last']['mean'], 1)} of {C.MATURITY_MONTHS - C.LOCKOUT_MONTHS + 1} KO observations "
                f"lie beyond the last listed IM tenor: the tail convention is not a detail.</p>"
                + _table(["model", "PV bp", "PV gap vs ref bp", "delta, hands", "delta gap vs ref", "rhoq bp per +1% q", "q at maturity"],
                         model_rows, "Long-holder PV and Greeks per unit product (one unit = the notional). "
                         "Delta is the hedging delta dPV/dS at fixed q(T); hands = delta / 200, rounded to whole "
                         "contracts by the fleet.")
            )
            if "hands_by_rank" in s:
                ranks = s["hands_by_rank"]
                static_html += _table(
                    ["contract rank (0 = nearest eligible)", "mean bucket hands", "median", "min", "max"],
                    [[str(r), _fmt(v["mean"]), _fmt(v["median"]), _fmt(v["min"]), _fmt(v["max"])] for r, v in ranks.items()],
                    f"Futures-tenor carry buckets under {C.REFERENCE_MODEL}: dPV/dF_i holding spot fixed, in hands, "
                    f"i.e. where the product's carry sensitivity sits along the IM chain. Analysis only - the hedge is "
                    f"sized off the spot delta above, and these buckets are a different partial that does not sum to it. "
                    f"The last bucket carries the extrapolated tail (share of |hands| in it: {_fmt(s['last_bucket_hands_share']['mean'], 0, True)}).",
                )
            static_html += _ki_section(s, data_dir)

    # --- backtest section -------------------------------------------------
    lc = agg["lifecycle_consistency"]
    consistency = (
        "<p class='ok'>Lifecycle check passed: every cell of every inception terminated identically (the lifecycle depends on the spot path only).</p>"
        if lc["consistent"]
        else f"<p class='bad'>Lifecycle MISMATCH on inceptions {html.escape(', '.join(lc['mismatched']))}: a defect, not a result.</p>"
    )
    failed_html = ""
    if agg["runs_failed"]:
        failed_html = "<p class='bad'>Failed cells: " + "; ".join(
            html.escape(f"{f['inception']} {f['cell']}: {f['error']}") for f in agg["runs_failed"]
        ) + "</p>"
    outcomes = cs.get(base, {}).get("outcomes", {})
    outcome_text = ", ".join(f"{html.escape(str(k))}: {v}" for k, v in outcomes.items())
    backtest_html = (
        "<h2>3. Hedging backtest</h2>"
        f"<p>{len(agg['inceptions'])} monthly inceptions ({agg['inceptions'][0]} to {agg['inceptions'][-1]}), "
        f"{agg['runs_ok']} completed runs over {len(cells)} cells. Outcomes (baseline cell): {outcome_text}. "
        f"QUAD grid {cfg.get('quad_grid')}, cost {cfg.get('cost_bp')} bp per side, daily delta hedge in whole contracts, "
        f"rate {_fmt(cfg.get('rate'), 1, True)} flat.</p>"
        + consistency + failed_html
        + cell_table
        + "<h3>Paired comparisons</h3>"
        + paired_table
        + _figure(chart_paired_strip(agg, "daily_pnl_std_bp", "Daily hedged P&L std: variant − baseline", "bp"),
                  "Negative = the variant's day-to-day hedge error is smaller than the flat active-contract model's.")
        + _figure(chart_paired_strip(agg, "roll_day_mtm_jump_bp", "|ΔMTM| on hedge-roll days: variant − baseline", "bp"),
                  "Negative = the variant re-marks the book less when the hedge contract rolls.")
        + _figure(chart_paired_strip(agg, "terminal_pnl_bp", "Terminal hedged P&L: variant − baseline", "bp"),
                  "Path-dependent and noisy by nature; shown for completeness.")
        + _figure(chart_example_path(run_dir, agg), "One inception, day by day: product MTM, the q handed to the pricer, and hedged P&L under the two models.")
    )

    verdict_html = "<h2>4. Verdict</h2>" + _verdict(agg)
    caveats = (
        "<h2>5. Caveats</h2><ul>"
        "<li>One product, one underlying, one 3.3-year window; the fleet's inceptions overlap, so the paired samples are not independent and the t-statistics overstate significance.</li>"
        "<li>The vol channel is a single ATM 1Y IV for every model; the study isolates carry, it does not test vol models (see <code>example/mo_volmodels</code> for that).</li>"
        "<li>Beyond the last listed IM tenor no market quotes carry; both tail conventions are assumptions, and the difference between them is the honest width of that uncertainty.</li>"
        "<li>The hedge is one contract sized off the spot delta. The carry buckets of section 2 are a diagnostic, not a hedge; a multi-contract hedge across them is a further step this study does not take.</li>"
        "<li>Transaction costs are a flat proportional rate; roll costs, margin funding and the spot-futures basis on unwind are not modelled.</li>"
        "</ul>"
    )
    models_html = "<ul>" + "".join(
        f"<li><code>{html.escape(k)}</code> — {html.escape(v.get('description', ''))}</li>" for k, v in agg["models"].items()
    ) + "</ul>"
    intro = (
        "<h1>Does a q term structure help hedge a snowball?</h1>"
        f"<p class='meta'>Generated {html.escape(agg['generated_at'])} from <code>{html.escape(agg['run_dir'])}</code>.</p>"
        "<p><strong>Question.</strong> A desk that sells a CSI 1000 snowball and delta-hedges it with IM futures has to tell its pricer "
        "what the index's dividend/carry yield is. The replay engine's default reads ONE flat yield off the hedge contract it happens to hold. "
        "The alternative reads the whole listed chain as a term structure q(T). Same contract, same spot path, same vol: does the term structure hedge better?</p>"
        "<p><strong>Design.</strong> A 1Y standard snowball on 000852.SH (KO 103% monthly from month 3, KI 75% daily, seller short 50 mio, "
        f"fair coupon solved once per inception under <code>{C.REFERENCE_MODEL}</code>), monthly inceptions, QUAD pricing, daily delta hedge in whole IM contracts. Carry models:</p>"
        + models_html
        + "<p>Hedge contracts: <em>front</em> (front month, rolled five days before expiry, the engine default) and <em>far</em> (longest listed contract, same roll rule).</p>"
    )
    body = intro + static_html + backtest_html + verdict_html + caveats + _appendix(agg)
    return f"<!doctype html><html><head><meta charset='utf-8'><title>Snowball q term structure study</title><style>{_css()}</style></head><body><main>{body}</main></body></html>"


def _css() -> str:
    return """
    body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1c1c1c;background:#fafafa;margin:0}
    main{max-width:1100px;margin:0 auto;padding:24px 28px 60px}
    h1{font-size:1.7em;margin:.2em 0}h2{font-size:1.25em;margin-top:1.6em;border-bottom:1px solid #ddd;padding-bottom:.2em}
    h3{font-size:1.05em}p,li{line-height:1.5}.meta{color:#666;font-size:.9em}code{background:#eee;padding:0 3px;border-radius:3px}
    .scroll{overflow-x:auto}table{border-collapse:collapse;font-size:.85em;margin:.6em 0}th,td{border:1px solid #ddd;padding:4px 7px;text-align:right;white-space:nowrap}
    th:first-child,td:first-child{text-align:left}caption{caption-side:bottom;color:#555;font-size:.9em;text-align:left;padding:4px 0}
    figure{margin:1em 0}figcaption{color:#555;font-size:.9em}img{max-width:100%}.ok{color:#2a7a2a}.bad{color:#b00020;font-weight:600}
    dl.terms{margin:.4em 0 1.1em}dl.terms dt{font-weight:600;margin-top:.7em;color:#111}
    dl.terms dd{margin:.2em 0 0 1.5em;color:#333;line-height:1.5;max-width:78ch}
    """


# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=C.DEFAULT_OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    agg = aggregate(args.run_dir, args.data_dir)
    write_tables(agg, args.data_dir)
    base = C.cell_name(C.BASELINE_MODEL, "front")
    print(f"{agg['runs_ok']} runs, {len(agg['inceptions'])} inceptions, cells: {', '.join(agg['cells'])}")
    print(f"lifecycle consistent: {agg['lifecycle_consistency']['consistent']}")
    for cell, e in agg["cell_summaries"].items():
        print(
            f"  {cell:24s} n={e['n']:3d}  pnl {_fmt(e['terminal_pnl_bp']['mean']):>8s} bp  "
            f"daily std {_fmt(e['daily_pnl_std_bp']['mean'], 2):>6s} bp  R2 {_fmt(e['variance_reduction_r2']['mean'], 3):>6s}  "
            f"roll |dMTM| {_fmt(e['roll_day_mtm_jump_bp']['mean'], 2):>6s} vs other {_fmt(e['other_day_mtm_jump_bp']['mean'], 2):>6s} bp  "
            f"hands/day {_fmt(e['delta_churn']['mean'], 2):>5s}"
        )
    for variant, e in agg["paired_summary"].items():
        print(
            f"  paired {variant:24s} - {e['base']:24s} n={e['n']:3d}  d pnl {_fmt(e['terminal_pnl_bp']['mean']):>7s}{_sig(e['terminal_pnl_bp']):4s} "
            f"d std {_fmt(e['daily_pnl_std_bp']['mean'], 2):>6s}{_sig(e['daily_pnl_std_bp']):4s} "
            f"d roll|dMTM| {_fmt(e['roll_day_mtm_jump_bp']['mean'], 2):>6s}{_sig(e['roll_day_mtm_jump_bp']):4s} "
            f"d hands/day {_fmt(e['delta_churn']['mean'], 2):>5s}"
        )
    if not args.no_report:
        report = args.data_dir / "q_term_structure_report.html"
        report.write_text(build_report(agg, args.run_dir, args.data_dir), encoding="utf-8")
        print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
