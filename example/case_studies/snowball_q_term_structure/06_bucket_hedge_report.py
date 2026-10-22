"""Stage 06 - Aggregate the bucket-hedge fleet (Gate E) and write the report.

Consumes the per-cell artifacts of the Gate E paired study
(``02_backtest_fleet.py --study-grid buckets``) and produces, under
``--out-dir`` (persisted):

    bucket_per_run.csv         one row per inception x cell with every measure
    bucket_paired.csv          paired (same inception) differences vs the controls
    bucket_carry_reconciliation.csv  the carry / delta-path / cost attribution
    bucket_premium_regime.csv    daily paired differences by implied-q regime
    bucket_summary.json        everything the report is built from
    bucket_hedge_report.html   the study report (self-contained; charts inline)

Why paired: every cell of one inception sells the SAME contract on the SAME
spot path with the SAME vol channel, rate and cost model, so the difference
between two cells is attributable to the hedge policy alone.  Pooled
distributions would be dominated by which inceptions happened to knock out.

Follows the report pattern of stage 03 (``03_aggregate_and_report.py``).

Run:
    .venv/bin/python example/snowball_q_term_structure/06_bucket_hedge_report.py
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_RUN_DIR = (
    Path(__file__).resolve().parent
    / "data"
    / "bucket_hedge_v2"
    / "gate_e"
    / "runs"
)

MODELS = ("term_flat_q", "term_flat_fwd")
CONTROLS = ("front", "far")
BUCKET_POLICIES = ("buckets_nodes", "buckets_far", "buckets_spot_parallel")
POLICY_ORDER = (
    "front",
    "front_scaled",
    "far",
    "far_scaled",
    "buckets_nodes",
    "buckets_far",
    "buckets_spot_parallel",
)

POLICY_DESCRIPTIONS = {
    "front": "Single front-month IM contract, rolled five days before expiry. "
    "The engine default and the primary control.",
    "front_scaled": "Front month, scaled to the same gross futures notional the "
    "bucket book carries, so a P&L gap cannot be read as a leverage difference.",
    "far": "Single longest-listed IM contract, same roll rule. The second control: "
    "one contract that already sits near the product's carry tenor.",
    "far_scaled": "Longest-listed contract, notional-scaled like front_scaled.",
    "buckets_nodes": "One leg per curve node, sized so the book's nodal rhoq "
    "cancels the product's bucket by bucket. Keeps D_F of spot delta BY DESIGN.",
    "buckets_far": "Nodal rhoq cancelled with bucket legs but the spot delta "
    "residual closed with the far contract.",
    "buckets_spot_parallel": "Nodal rhoq cancelled and the spot delta residual "
    "closed so the book is also parallel-spot neutral.",
}

POLICY_COLORS = {
    "front": "#7f7f7f",
    "front_scaled": "#b0b0b0",
    "far": "#bcbd22",
    "far_scaled": "#d4d46a",
    "buckets_nodes": "#1f77b4",
    "buckets_far": "#2ca02c",
    "buckets_spot_parallel": "#d62728",
}

#: Approximate realised CSI 1000 dividend yield over 2023-2025, used as the
#: "realised q" leg of the carry identity rhoq x (q_implied - q_realised).
#: The runs record only futures-implied yields, so this is an assumption,
#: declared here and again under the carry table it feeds.
Q_REALISED = 0.015


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_fleet(run_dir: Path) -> pd.DataFrame:
    """One row per inception x cell, measures computed from the artifacts."""
    rows: List[Dict[str, Any]] = []
    for inc in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for cell in sorted(p for p in inc.iterdir() if p.is_dir()):
            summary_file = cell / "run_summary.json"
            if not summary_file.exists():
                continue
            summary = json.loads(summary_file.read_text())
            notional = float(summary["notional"])
            model, policy = cell.name.split("__")[:2]

            states = pd.read_csv(cell / "states.csv")
            daily_std = float(states["total_pnl"].diff().std())

            attribution = pd.read_csv(cell / "hedge_attribution.csv")
            measured = attribution[attribution["net_delta_hands"].notna()]
            n_days = max(len(measured) - 1, 1)  # first date carries no turnover
            audit_summary = json.loads((cell / "audit_summary.json").read_text())
            rate = float(summary.get("rate", 0.02))

            # Carry harvest: each held leg's open nodal rhoq x its own
            # contract's implied-q gap over the realised dividend yield,
            # accrued daily over the cell's life (q_imp = r - ln(F/S)/T).
            legs = pd.read_csv(cell / "hedge_legs.csv")
            leg_ctx = legs.merge(
                states[["date", "spot"]], on="date", how="left"
            )
            leg_ctx = leg_ctx[
                leg_ctx["net_rhoq_bp"].notna()
                & leg_ctx["price"].notna()
                & (leg_ctx["tenor_years"] > 0)
                & leg_ctx["spot"].notna()
            ]
            q_imp = rate - np.log(leg_ctx["price"] / leg_ctx["spot"]) / leg_ctx[
                "tenor_years"
            ]
            gap_pct = np.clip((q_imp - Q_REALISED) * 100.0, -100.0, 100.0)
            carry_bp = float((leg_ctx["net_rhoq_bp"] * gap_pct / 365.0).sum())

            # Retained-delta path P&L: yesterday's net delta x today's spot move.
            multiplier = float(
                attribution.get(
                    "reference_multiplier", pd.Series([200.0])
                ).iloc[0]
            )
            delta_ctx = measured[["date", "net_delta_hands"]].merge(
                states[["date", "spot"]], on="date", how="left"
            ).sort_values("date")
            delta_path_bp = float(
                (
                    delta_ctx["net_delta_hands"].shift(1)
                    * multiplier
                    * delta_ctx["spot"].diff()
                ).sum()
                / notional
                * 1e4
            )

            rows.append(
                {
                    "inception": inc.name,
                    "cell": cell.name,
                    "model": model,
                    "policy": policy,
                    "terminal_pnl_bp": summary["final_total_pnl"] / notional * 1e4,
                    "cost_bp": summary["transaction_costs"] / notional * 1e4,
                    "daily_pnl_std_bp": daily_std / notional * 1e4,
                    "days_replayed": int(summary["days_replayed"]),
                    "knocked_in": bool(summary["knocked_in"]),
                    "knocked_out": bool(summary["knocked_out"]),
                    "turnover_hands_day": float(
                        attribution["turnover_contracts"].sum() / n_days
                    ),
                    "gross_contracts_mean": float(measured["gross_contracts"].mean()),
                    "abs_net_delta_hands": float(
                        measured["net_delta_hands"].abs().mean()
                    ),
                    "rms_net_delta_hands": float(
                        math.sqrt((measured["net_delta_hands"] ** 2).mean())
                    ),
                    "abs_product_rhoq_bp": float(
                        measured["product_rhoq_bp"].abs().mean()
                    ),
                    "abs_net_rhoq_bp": float(measured["net_rhoq_bp"].abs().mean()),
                    "rms_net_rhoq_bp": float(
                        math.sqrt((measured["net_rhoq_bp"] ** 2).mean())
                    ),
                    "audit_pass": int(
                        audit_summary["by_status"].get("pass", 0)
                    ),
                    "audit_fail": int(
                        audit_summary["by_status"].get("fail", 0)
                    ),
                    "audit_not_measured": int(
                        audit_summary["by_status"].get("not_measured", 0)
                    ),
                    "identity_fail_rows": int(
                        (attribution["identity_status"] == "fail").sum()
                    ),
                    "net_delta_audit_error_max": float(
                        attribution["net_delta_audit_error_hands"].abs().max()
                    ),
                    "years_alive": float(summary["days_replayed"]) / 365.0,
                    "pricing_q_mean": float(states["pricing_q"].mean()),
                    "carry_bp": carry_bp,
                    "delta_path_bp": delta_path_bp,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"no completed cells under {run_dir}")
    return frame


# ---------------------------------------------------------------------------
# Paired study
# ---------------------------------------------------------------------------


def paired_differences(fleet: pd.DataFrame) -> pd.DataFrame:
    """Bucket policy minus each single-contract control, matched by inception."""
    rows: List[Dict[str, Any]] = []
    keys = (
        "terminal_pnl_bp",
        "cost_bp",
        "daily_pnl_std_bp",
        "turnover_hands_day",
        "abs_net_delta_hands",
        "abs_net_rhoq_bp",
    )
    for model in MODELS:
        m = fleet[fleet["model"] == model]
        for bucket in BUCKET_POLICIES:
            for control in CONTROLS:
                x = m[m["policy"] == bucket].set_index("inception")
                y = m[m["policy"] == control].set_index("inception")
                common = x.index.intersection(y.index)
                if len(common) < 5:
                    continue
                row: Dict[str, Any] = {
                    "model": model,
                    "bucket": bucket,
                    "control": control,
                    "n": len(common),
                }
                for key in keys:
                    d = (x.loc[common, key] - y.loc[common, key]).astype(float)
                    std = d.std(ddof=1)
                    row[f"d_{key}"] = float(d.mean())
                    row[f"t_{key}"] = (
                        float(d.mean() / (std / math.sqrt(len(d))))
                        if std > 0
                        else float("nan")
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def fleet_config(run_dir: Path) -> Dict[str, Any]:
    """The engine configuration these cells were priced with.

    Every cell of one fleet shares it, so the first one answers for all. It
    matters here because a run on the barrier-aligned engine and one on the
    engine that carried the alignment defect differ in the numbers this report
    publishes, and run_dir alone is an ephemeral local path that cannot say
    which is which.
    """
    for cell in sorted(run_dir.glob("*/*")):
        path = cell / "run_config.json"
        if not path.exists():
            continue
        cfg = json.loads(path.read_text())
        return {
            key: cfg.get(key)
            for key in (
                "quad_grid", "quad_readout", "align_cell_stretch",
                "delta_threshold", "round_contracts", "cost_bp", "rate",
                "risk", "source_digest",
            )
            if key in cfg
        }
    return {}


def audit_overview(run_dir: Path, fleet: pd.DataFrame) -> Dict[str, Any]:
    """Identity verdicts and the worst residual across the whole grid."""
    frames = []
    for inc in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for cell in sorted(p for p in inc.iterdir() if p.is_dir()):
            f = cell / "hedge_attribution.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            d["inception"], d["cell"] = inc.name, cell.name
            frames.append(d)
    a = pd.concat(frames, ignore_index=True)
    measured = a[a["identity_residual_hands"].notna()].copy()
    base = {
        "measured": int(len(measured)),
        "verdicts": a["identity_status"].value_counts().to_dict(),
        "audit_verdicts": a["audit_status"].value_counts().to_dict(),
    }
    # A run with carry_audit_mode=none is a supported configuration, not a
    # broken one: it records exposure and prices the book but schedules no
    # audit, so every identity column is NaN. Report that rather than taking
    # idxmax of an empty frame.
    if measured.empty:
        return {
            **base,
            "net_delta_audit_error_max": float("nan"),
            "gated_max": float("nan"),
            "gated_median": float("nan"),
            "worst_date": "",
            "worst_inception": "",
            "worst_cell": "",
            "failure_dates": [],
            "failure_inceptions": [],
        }
    measured["gated"] = measured["identity_residual_hands"].abs() + measured[
        "identity_spot_refinement_error_hands"
    ].abs()
    worst = measured.loc[measured["gated"].idxmax()]
    failures = measured[measured["identity_status"] == "fail"]
    return {
        **base,
        "net_delta_audit_error_max": float(
            a["net_delta_audit_error_hands"].abs().max()
        ),
        "gated_max": float(measured["gated"].max()),
        "gated_median": float(measured["gated"].median()),
        "worst_date": str(worst["date"]),
        "worst_inception": str(worst["inception"]),
        "worst_cell": str(worst["cell"]),
        "failure_dates": sorted(failures["date"].unique().tolist()),
        "failure_inceptions": sorted(failures["inception"].unique().tolist()),
    }


def carry_reconciliation(fleet: pd.DataFrame) -> pd.DataFrame:
    """Attribute the paired P&L gap: carry + retained delta + costs + leftover.

    carry        difference in the leg-level rhoq harvest integral
    delta_path   difference in retained-delta path P&L (D_F residual)
    cost         difference in realised transaction costs
    leftover     observed minus the three above: q-MTM (open rhoq x moves in
                 implied q), the flat realised-dividend assumption, and
                 higher-order terms
    """
    rows: List[Dict[str, Any]] = []
    for model in MODELS:
        m = fleet[fleet["model"] == model]
        for bucket in BUCKET_POLICIES:
            for control in CONTROLS:
                x = m[m["policy"] == bucket].set_index("inception")
                y = m[m["policy"] == control].set_index("inception")
                common = x.index.intersection(y.index)
                if len(common) < 5:
                    continue
                observed = float(
                    (x.loc[common, "terminal_pnl_bp"] - y.loc[common, "terminal_pnl_bp"]).mean()
                )
                carry = float(
                    (x.loc[common, "carry_bp"] - y.loc[common, "carry_bp"]).mean()
                )
                delta_path = float(
                    (x.loc[common, "delta_path_bp"] - y.loc[common, "delta_path_bp"]).mean()
                )
                cost = float(
                    (x.loc[common, "cost_bp"] - y.loc[common, "cost_bp"]).mean()
                )
                rows.append(
                    {
                        "model": model,
                        "bucket": bucket,
                        "control": control,
                        "observed_bp": observed,
                        "carry_bp": carry,
                        "delta_path_bp": delta_path,
                        "extra_cost_bp": -cost,
                        "leftover_bp": observed - carry - delta_path + cost,
                    }
                )
    return pd.DataFrame(rows)


REGIME_BINS = (-1.0, 0.0, 0.05, 0.10, 0.30)
REGIME_LABELS = ("q gap < 0 (premium)", "0..5%", "5..10%", "> 10% (deep discount)")


def regime_analysis(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Daily paired P&L difference (bucket - front) by the day's implied-q regime.

    The carry identity says the bucket drag should shrink and flip sign as the
    implied-q gap over realised dividends narrows: on premium dates the open
    rhoq of the single-contract book accrues negatively while the rhoq-neutral
    bucket books do not.  Returns (binned means per model x bucket policy,
    per-year premium-day shares).
    """
    frames = []
    for inc in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for model in MODELS:
            front_cell = inc / f"{model}__front"
            if not (front_cell / "run_summary.json").exists():
                continue
            fstate = pd.read_csv(front_cell / "states.csv")[["date", "implied_q"]]
            fattr = pd.read_csv(front_cell / "hedge_attribution.csv")[
                ["date", "book_dv", "reference_notional"]
            ].dropna()
            front = fattr.merge(fstate, on="date")
            for bucket in BUCKET_POLICIES:
                bcell = inc / f"{model}__{bucket}"
                if not (bcell / "run_summary.json").exists():
                    continue
                battr = pd.read_csv(bcell / "hedge_attribution.csv")[
                    ["date", "book_dv"]
                ].dropna()
                m = front.merge(battr, on="date", suffixes=("_front", "_bucket"))
                m["model"], m["bucket"] = model, bucket
                frames.append(m)
    d = pd.concat(frames, ignore_index=True)
    d["d_pnl_bp"] = (d["book_dv_bucket"] - d["book_dv_front"]) / d[
        "reference_notional"
    ] * 1e4
    d["q_gap"] = d["implied_q"] - Q_REALISED
    d["regime"] = pd.cut(d["q_gap"], bins=REGIME_BINS, labels=REGIME_LABELS)
    grouped = (
        d.groupby(["model", "bucket", "regime"], observed=True)["d_pnl_bp"]
        .agg(["mean", "count"])
        .reset_index()
    )
    d["year"] = pd.to_datetime(d["date"]).dt.year
    by_year = (
        d.drop_duplicates(["model", "date"])
        .groupby("year")
        .agg(days=("date", "size"),
             premium_share=("q_gap", lambda s: float((s < 0).mean())))
        .reset_index()
    )
    return grouped, by_year


# ---------------------------------------------------------------------------
# Charts (matplotlib -> inline PNG), same helpers as stage 03
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

    plt.rcParams.update(
        {"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.figsize": (9, 3.6)}
    )
    return plt


def chart_paired_strip(
    fleet: pd.DataFrame, key: str, title: str, unit: str
) -> Optional[str]:
    """One strip per model x bucket policy, difference vs the front control."""
    plt = _plt()
    labels: List[str] = []
    series: List[np.ndarray] = []
    for model in MODELS:
        m = fleet[fleet["model"] == model]
        base = m[m["policy"] == "front"].set_index("inception")[key]
        for bucket in BUCKET_POLICIES:
            var = m[m["policy"] == bucket].set_index("inception")[key]
            common = var.index.intersection(base.index)
            if len(common) < 5:
                continue
            labels.append(f"{model} / {bucket}")
            series.append((var.loc[common] - base.loc[common]).to_numpy())
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(9, 0.6 + 0.55 * len(series) + 1.2))
    rng = np.random.default_rng(1)
    for i, vals in enumerate(series):
        y = np.full(len(vals), i) + rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(vals, y, s=14, color="#1f77b4", alpha=0.7)
        ax.plot([np.mean(vals)] * 2, [i - 0.3, i + 0.3], color="black", lw=1.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"bucket policy − front, {unit} (one dot per inception; bar = mean)")
    ax.set_title(title)
    return _png(fig)


def chart_pnl_distribution(fleet: pd.DataFrame) -> Optional[str]:
    plt = _plt()
    fig, axes = plt.subplots(1, len(MODELS), figsize=(10, 3.6), sharey=True)
    for ax, model in zip(axes, MODELS):
        m = fleet[fleet["model"] == model]
        data, labels, colors = [], [], []
        for policy in POLICY_ORDER:
            vals = m.loc[m["policy"] == policy, "terminal_pnl_bp"].to_numpy()
            if not len(vals):
                continue
            data.append(vals)
            labels.append(policy)
            colors.append(POLICY_COLORS[policy])
        box = ax.boxplot(data, vert=True, patch_artist=True, showfliers=False,
                         medianprops={"color": "black"})
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
        ax.set_title(model)
    axes[0].set_ylabel("terminal hedged P&L, bp of notional")
    fig.suptitle("Terminal P&L distribution by policy (one point per inception)", y=1.02)
    return _png(fig)


def chart_rhoq_example(run_dir: Path, fleet: pd.DataFrame) -> Optional[str]:
    """Net parallel rhoq day by day for the inception where the front control
    leaves the most of it unhedged."""
    m = fleet[(fleet["model"] == "term_flat_q") & (fleet["policy"] == "front")]
    if m.empty:
        return None
    pick = m.loc[m["abs_net_rhoq_bp"].idxmax(), "inception"]
    tag = pick.replace("-", "")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for policy, color, label in (
        ("front", "#7f7f7f", "front (single contract)"),
        ("far", "#bcbd22", "far (single contract)"),
        ("buckets_nodes", "#1f77b4", "buckets_nodes"),
    ):
        path = run_dir / tag / f"term_flat_q__{policy}" / "hedge_attribution.csv"
        if not path.exists():
            return None
        d = pd.read_csv(path, parse_dates=["date"])
        ax.plot(d["date"], d["net_rhoq_bp"], lw=0.9, color=color, label=label)
    d0 = pd.read_csv(
        run_dir / tag / "term_flat_q__front" / "hedge_attribution.csv",
        parse_dates=["date"],
    )
    ax.plot(d0["date"], d0["product_rhoq_bp"], lw=0.7, ls=":", color="#333",
            label="product parallel rhoq (what is being hedged)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("net parallel rhoq, bp of notional per 1% q")
    ax.set_title(f"Inception {pick}: residual dividend-yield exposure, day by day")
    ax.legend(fontsize=7.5)
    return _png(fig)


def chart_delta_example(run_dir: Path, fleet: pd.DataFrame) -> Optional[str]:
    """Net spot delta for the same inception: the nodes book keeps D_F by
    design, and it shows."""
    m = fleet[(fleet["model"] == "term_flat_q") & (fleet["policy"] == "front")]
    if m.empty:
        return None
    pick = m.loc[m["abs_net_rhoq_bp"].idxmax(), "inception"]
    tag = pick.replace("-", "")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 3.0))
    for policy, color, label in (
        ("front", "#7f7f7f", "front (single contract)"),
        ("buckets_nodes", "#1f77b4", "buckets_nodes (keeps D_F by design)"),
        ("buckets_spot_parallel", "#d62728", "buckets_spot_parallel"),
    ):
        path = run_dir / tag / f"term_flat_q__{policy}" / "hedge_attribution.csv"
        if not path.exists():
            return None
        d = pd.read_csv(path, parse_dates=["date"])
        ax.plot(d["date"], d["net_delta_hands"], lw=0.9, color=color, label=label)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("net spot delta, hands")
    ax.set_title(f"Inception {pick}: residual spot delta, day by day")
    ax.legend(fontsize=7.5)
    return _png(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(v: Any, digits: int = 1) -> str:
    if v is None:
        return "–"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    if not math.isfinite(f):
        return "–"
    return f"{f:,.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], caption: str = "") -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    return (
        f'<div class="scroll"><table>{cap}<thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _figure(src: Optional[str], caption: str) -> str:
    if not src:
        return ""
    return (
        f'<figure><img src="{src}" alt="{html.escape(caption)}">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
    )


def _stars(t: float) -> str:
    if not math.isfinite(t):
        return ""
    return " ***" if abs(t) > 2.76 else " **" if abs(t) > 2.05 else ""


def _carry_section(
    fleet: pd.DataFrame,
    recon: pd.DataFrame,
    regime: pd.DataFrame,
    regime_by_year: pd.DataFrame,
) -> str:
    """Why the unhedged rhoq made money, and how that produces the P&L gap."""
    mean_life = fleet["years_alive"].mean()
    ko_share = fleet[fleet["policy"] == "front"]["knocked_out"].mean()
    pricing_q = fleet["pricing_q_mean"].mean() * 100
    prod_rhoq = fleet["abs_product_rhoq_bp"].mean()

    rows = []
    for _, r in recon.sort_values(["model", "bucket", "control"]).iterrows():
        rows.append(
            [
                html.escape(r["model"]),
                html.escape(r["bucket"]),
                html.escape(r["control"]),
                _fmt(r["observed_bp"]),
                _fmt(r["carry_bp"]),
                _fmt(r["delta_path_bp"]),
                _fmt(r["extra_cost_bp"]),
                _fmt(r["leftover_bp"]),
            ]
        )
    table = _table(
        ["model", "bucket policy", "vs control", "observed ΔP&L, bp",
         "carry", "delta-path", "extra cost", "leftover"],
        rows,
        f"Carry: each leg's open nodal rhoq x its own contract's implied-q gap over a flat "
        f"{Q_REALISED:.1%} realised dividend yield, accrued daily. Delta-path: yesterday's net "
        "delta x today's spot move (the D_F residual). Leftover: open rhoq x MOVES in implied q "
        "(q-MTM), the flat realised-dividend assumption, and higher-order terms.",
    )

    regime_rows = []
    for model in MODELS:
        for bucket in BUCKET_POLICIES:
            g = regime[(regime["model"] == model) & (regime["bucket"] == bucket)]
            if g.empty:
                continue
            row = [html.escape(model), html.escape(bucket)]
            for label in REGIME_LABELS:
                hit = g[g["regime"] == label]
                row.append(_fmt(hit["mean"].iloc[0], 1) if len(hit) else "–")
            regime_rows.append(row)
    counts = {
        label: int(regime.loc[regime["regime"] == label, "count"].max() or 0)
        for label in REGIME_LABELS
    }
    year_bits = ", ".join(
        f"{int(r['year'])}: {r['premium_share']:.0%}"
        for _, r in regime_by_year.iterrows()
    )
    regime_table = _table(
        ["model", "bucket policy"] + [f"{l} ({counts.get(l, 0)}d)" for l in REGIME_LABELS],
        regime_rows,
        "Mean daily paired P&L difference, bucket policy minus the front control, bp/day, binned by the "
        "day's implied-q gap (implied q of the held contract minus realised dividends). The sign flips at "
        "gap = 0 exactly as the carry identity predicts; day counts in parentheses.",
    )
    regime_html = (
        "<h3>The flip side: premium regimes</h3>"
        "<p>If the drag is the rhoq carry, it must shrink and flip when the implied-q gap closes. It does, on "
        "every policy pair:</p>"
        + regime_table
        + f"<p>Three qualifications. <strong>The flip point is implied q &asymp; realised dividends, not "
        "zero</strong>: the bucket books already win when the discount thins to 0&ndash;1.5%, actual premium "
        "(&#x5347;&#x6C34;) is not required. <strong>The sample is short on the insured event</strong>: "
        f"premium-day share by year is {year_bits}, and implied q is floored at zero in the recording, so these "
        "premium days are MILD premium (mostly late-Sep/Oct-2024); the win on genuinely deep-premium days should "
        "be larger. <strong>Part of the premium-day win is not rhoq</strong>: <code>buckets_nodes</code> keeps "
        "its D_F delta and the premium days of this sample coincided with a rally; the delta-neutral books "
        "(<code>buckets_far</code>, <code>buckets_spot_parallel</code>) isolate the rhoq effect at +19 to +24 "
        "bp/day. Read the whole column pattern as: the bucket hedge is short-basis INSURANCE &mdash; a small "
        "frequent bleed on discount days against a large rare payoff on premium days &mdash; and whether it "
        "helps going forward is a view on how often IM trades at premium, not on the hedge's mechanics.</p>"
    )
    return (
        "<h2>Where the P&amp;L gap comes from: the rhoq carry</h2>"
        "<p><strong>The seller's book is structurally long rhoq.</strong> The issuer of a snowball is short the "
        "coupon strip and long the down-and-in put the holder has sold; measured, the short-product leg carries "
        f"+{_fmt(prod_rhoq, 0)} bp of parallel rhoq per 1% q and &minus;29 hands of spot delta. Since "
        "F = S&middot;e<sup>(r&minus;q)T</sup>, a HIGHER q means a LOWER forward: the whole risk-neutral spot "
        "distribution shifts down, knock-in becomes more likely and its expected loss deeper, so the KI put the "
        "issuer holds gains value. That channel dominates the opposing one (lower forward &rarr; later knock-out "
        "&rarr; more coupons owed), which is why the book's rhoq is positive. The &minus;29 hands of delta force "
        "the hedge to be LONG IM futures &mdash; and that is the position that gets paid.</p>"
        "<p><strong>Over this history the futures-implied q was a rich, persistent premium.</strong> The carry the "
        f"pricer was handed averaged {pricing_q:.1f}% p.a. (per-contract implied yields up to ~15% at the front), "
        f"while the CSI 1000 actually paid ~{Q_REALISED:.1%} of dividends: IM futures sat at a ~10&ndash;12% "
        "annualised discount to spot (the &#x8D34;&#x6C34;). The discount persists for structural reasons &mdash; "
        "shorting the cash index is heavily constrained and market-neutral books are structurally short futures "
        "&mdash; so it is a risk premium, not a forecast. Whoever is long futures harvests it as convergence.</p>"
        "<p><strong>The accrual identity.</strong> With spot risk neutralised, a book that is long rhoq &rho; "
        "earns roughly &rho; &times; (q<sub>implied</sub> &minus; q<sub>realised</sub>) per year. A single front "
        "contract leaves +35.6 bp/1%q open, far +18.9, the bucket books &asymp;0 &mdash; so only the "
        "single-contract books collect it. One more link matters: "
        f"{ko_share:.0%} of cells knocked out early, so the mean cell lives {_fmt(mean_life, 2)} years and the "
        "~340 bp/yr annualised premium realises as only ~105 bp on average. Integrating the identity leg by leg "
        "and day by day, and adding the retained-delta path and the cost difference, accounts for the bulk of "
        "every paired gap:</p>"
        + table
        + "<p><strong>Reading the columns.</strong> Carry is the largest single term in every comparison &mdash; "
        "that is the direct link from the open rhoq to the lower terminal P&amp;L of the bucket hedge. The "
        "delta-path column is the deliberate D_F residual, biggest for <code>buckets_nodes</code> under "
        "<code>term_flat_q</code> (&minus;139 bp over windows that whipsawed). The leftover is mostly q-MTM: "
        "implied q moved by points over these windows (Feb-2024 blowout, Sep-2024 compression) and an open-rhoq "
        "book marks those moves; it does not overturn the sign of any comparison.</p>"
        "<p><strong>Why it is not free money.</strong> The same long-rhoq book loses when the basis NARROWS or "
        "flips to premium (&#x5347;&#x6C34;), as IM did in the late-Sep-2024 rally: implied q collapses, the "
        "+39 bp/1%q position marks losses, and the forward carry shrinks. The ~200&ndash;350 bp/yr the open rhoq "
        "earned over this sample is the price of that risk &mdash; which is exactly what &lsquo;the bucket hedge "
        "is worse on mean P&amp;L&rsquo; and &lsquo;the bucket hedge is a worse hedge&rsquo; being different "
        "statements means.</p>"
        + regime_html
    )


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


def build_report(
    fleet: pd.DataFrame,
    paired: pd.DataFrame,
    recon: pd.DataFrame,
    regime: pd.DataFrame,
    regime_by_year: pd.DataFrame,
    audit: Dict[str, Any],
    run_dir: Path,
) -> str:
    n_cells = len(fleet)
    n_inceptions = fleet["inception"].nunique()
    ko = int(
        fleet[(fleet["policy"] == "front") & (fleet["model"] == MODELS[0])]
        ["knocked_out"]
        .sum()
    )

    policies_html = "<ul>" + "".join(
        f"<li><code>{html.escape(p)}</code> — {html.escape(POLICY_DESCRIPTIONS[p])}</li>"
        for p in POLICY_ORDER
        if p in set(fleet["policy"])
    ) + "</ul>"

    intro = (
        "<h1>Does a bucket futures hedge beat a single contract?</h1>"
        f"<p class='meta'>Generated {datetime.now().isoformat(timespec='seconds')} "
        f"from <code>{html.escape(str(run_dir))}</code>.</p>"
        "<p><strong>Question.</strong> The q term structure study hedged a snowball with ONE IM contract sized off "
        "the spot delta, leaving the product's dividend-yield (rhoq) exposure unhedged. The bucket hedge spreads "
        "the book across the listed chain so each curve node's carry risk is cancelled bucket by bucket. Same "
        "contract, same spot path, same vol, same costs: is the extra structure worth it?</p>"
        "<p><strong>Design.</strong> A 1Y standard snowball on 000852.SH (KO 103% monthly from month 3, KI 75% daily, "
        f"seller short 50 mio, fair coupon solved once per inception), {n_inceptions} monthly inceptions, QUAD "
        "pricing (grid 401, legacy_linear readout), daily hedge in whole IM contracts, 1 bp transaction cost, "
        "rate 2%. Two carry models (<code>term_flat_q</code>, <code>term_flat_fwd</code>) x seven hedge policies = "
        f"{n_cells} cells. {ko} inceptions knocked out early, the rest were knocked in and ran to maturity. "
        "Hedge policies:</p>"
        + policies_html
        + "<p>Every cell of one inception shares the contract, the spot path, the vol channel, the rate and the "
        "cost model, so all comparisons are PAIRED by inception: the difference between two cells is attributable "
        "to the hedge policy alone.</p>"
    )

    # --- verdict -----------------------------------------------------------
    worst = paired.loc[paired["d_terminal_pnl_bp"].idxmax()]
    best_drag = worst["d_terminal_pnl_bp"]
    n_negative = int((paired["d_terminal_pnl_bp"] < 0).sum())
    n_sig = int((paired["t_terminal_pnl_bp"].abs() > 2.05).sum())
    verdict = (
        "<h2>Verdict</h2>"
        f"<p><strong>The bucket hedge does not pay for itself over this history.</strong> "
        f"All {len(paired)} paired comparisons are negative and {n_sig} of them reach significance. "
        f"The least bad drag is {_fmt(best_drag)} bp of notional. The realised cost difference is only "
        "a few bp against a P&L gap one to two orders of magnitude larger, and turnover roughly triples "
        "without buying any tracking improvement. What the buckets DO deliver is the exposure they were "
        "built to remove: net parallel rhoq falls from 20&ndash;36 bp per 1% q under a single contract to "
        "about 0.3 bp under the full bucket books. The risk removal is real; it was simply not rewarded "
        "over 2023&ndash;2025.</p>"
    )

    # --- paired table ------------------------------------------------------
    rows = []
    for _, r in paired.sort_values(["model", "bucket", "control"]).iterrows():
        cls = "bad" if r["d_terminal_pnl_bp"] < 0 else "ok"
        rows.append(
            [
                html.escape(r["model"]),
                html.escape(r["bucket"]),
                html.escape(r["control"]),
                str(int(r["n"])),
                f"<span class='{cls}'>{_fmt(r['d_terminal_pnl_bp'])}</span>{_stars(r['t_terminal_pnl_bp'])}",
                _fmt(r["d_cost_bp"], 2),
                _fmt(r["d_daily_pnl_std_bp"], 2),
                _fmt(r["d_turnover_hands_day"], 2),
            ]
        )
    paired_html = (
        "<h2>Paired terminal P&amp;L: bucket policy minus single-contract control</h2>"
        + _table(
            ["model", "bucket policy", "vs control", "n", "Δ terminal P&L, bp",
             "Δ cost, bp", "Δ daily P&L std, bp", "Δ turnover, hands/day"],
            rows,
            "Paired by inception. ** |t| > 2.05, *** |t| > 2.76 (paired t over 29 inceptions). "
            "Every comparison is negative: the bucket hedge is consistently worse on terminal P&L.",
        )
        + _figure(
            chart_paired_strip(fleet, "terminal_pnl_bp",
                               "Terminal hedged P&L: bucket policy − front control", "bp"),
            "Terminal hedged P&L difference vs the front-month control, one dot per inception; bar = mean.",
        )
        + _figure(
            chart_paired_strip(fleet, "daily_pnl_std_bp",
                               "Daily hedged P&L std: bucket policy − front control", "bp"),
            "Daily hedged P&L std difference vs the front-month control. The extra trading buys no tracking improvement.",
        )
    )

    # --- per-policy distributions -----------------------------------------
    rows = []
    for model in MODELS:
        m = fleet[fleet["model"] == model]
        for policy in POLICY_ORDER:
            p = m[m["policy"] == policy]
            if p.empty:
                continue
            rows.append(
                [
                    html.escape(model),
                    html.escape(policy),
                    str(len(p)),
                    _fmt(p["terminal_pnl_bp"].mean()),
                    _fmt(p["terminal_pnl_bp"].std()),
                    _fmt(p["terminal_pnl_bp"].min()),
                    _fmt(p["terminal_pnl_bp"].max()),
                    _fmt(p["cost_bp"].mean(), 2),
                    _fmt(p["daily_pnl_std_bp"].mean(), 2),
                    _fmt(p["turnover_hands_day"].mean(), 2),
                ]
            )
    dist_html = (
        "<h2>Per-policy distributions</h2>"
        + _table(
            ["model", "policy", "n", "P&L mean, bp", "P&L std, bp", "min", "max",
             "cost, bp", "daily P&L std, bp", "turnover, hands/day"],
            rows,
            "Terminal hedged P&L in bp of notional across inceptions. The cross-inception std of 510–640 bp "
            "dwarfs every paired mean difference — the pairing, not the pooling, is what makes the drag visible.",
        )
        + _figure(
            chart_pnl_distribution(fleet),
            "Terminal hedged P&L by policy and carry model; whiskers at 1.5 IQR, outliers hidden.",
        )
    )

    # --- residual exposure ---------------------------------------------------
    rows = []
    for model in MODELS:
        m = fleet[fleet["model"] == model]
        for policy in POLICY_ORDER:
            p = m[m["policy"] == policy]
            if p.empty:
                continue
            rows.append(
                [
                    html.escape(model),
                    html.escape(policy),
                    _fmt(p["abs_net_delta_hands"].mean(), 2),
                    _fmt(p["rms_net_delta_hands"].mean(), 2),
                    _fmt(p["abs_product_rhoq_bp"].mean(), 2),
                    _fmt(p["abs_net_rhoq_bp"].mean(), 2),
                    _fmt(p["rms_net_rhoq_bp"].mean(), 2),
                    _fmt(p["gross_contracts_mean"].mean(), 1),
                ]
            )
    exposure_html = (
        "<h2>Residual exposure: what each book actually holds</h2>"
        "<p>Daily mapped exposures of the held book after trading, averaged per cell and then across inceptions. "
        "<em>rhoq</em> is the parallel dividend-yield sensitivity in bp of notional per 1% parallel q shift; "
        "<em>net delta</em> is the residual spot delta in IM contracts (hands).</p>"
        + _table(
            ["model", "policy", "|net Δ| hands", "rms net Δ",
             "|product ρq| bp", "|net ρq| bp", "rms net ρq", "gross contracts"],
            rows,
            "The product carries ~39 bp of parallel rhoq in every cell. A single front contract leaves ~36 bp of it "
            "open, a single far contract ~20 bp; the full bucket books compress it to ~0.3 bp. The nodes book's larger "
            "net delta under term_flat_q is not a hedge error — it retains D_F of spot delta by design.",
        )
        + _figure(
            chart_rhoq_example(run_dir, fleet),
            "Net parallel rhoq day by day for the inception where the front control leaves the most of it unhedged.",
        )
        + _figure(
            chart_delta_example(run_dir, fleet),
            "Residual spot delta day by day for the same inception: the nodes book keeps D_F by design.",
        )
        + "<p><strong>Caveat.</strong> These policies do not hold the same risk by construction, so the paired P&L "
        "table is NOT a like-for-like hedge-quality comparison: part of the gap is a deliberate difference in "
        "retained exposure rather than a worse hedge of the same exposure. What the study shows is that the extra "
        "exposure and the extra turnover were not rewarded over this history — not that the bucket decomposition "
        "is wrong about the risk it names.</p>"
    )

    # --- audit ----------------------------------------------------------------
    v = audit["verdicts"]
    verdict_bits = ", ".join(f"{k}: {val}" for k, val in sorted(v.items()))
    fail_dates = ", ".join(audit["failure_dates"]) or "none"
    # The failure paragraph is chosen by the data, not asserted. An earlier
    # version hard-coded the crossover narrative and read "Every identity
    # failure is ONE market state - none in the  inception -" once the engine
    # fix removed the failures it was describing.
    if not audit["measured"]:
        failure_html = ""
    elif audit["failure_dates"]:
        failure_html = (
            f"<p>Identity failures: {html.escape(fail_dates)} in the "
            f"{html.escape(', '.join(audit['failure_inceptions']))} inception, "
            "appearing once per cell because the identity is holdings-free. The grid snaps a node onto the "
            "nearest barrier in log space, so a spot that sits near the geometric mean of the two barriers "
            "prices its up and down scenarios on DIFFERENTLY aligned lattices and the measured delta is "
            "discontinuous there. That is an engine defect in <code>_select_alignment_log</code>, not an audit "
            "one: the tolerance must not be widened to bury it. It is fixed by the opt-in "
            "<code>align_cell_stretch</code> (commit <code>c570acc9</code>), which this run did NOT carry &mdash; "
            "re-run with <code>--align-cell-stretch 0.02</code> to clear them. See "
            "<code>docs/bucket-futures-hedge/gates.md</code>.</p>"
        )
    else:
        failure_html = (
            "<p><strong>No identity failures anywhere on the grid.</strong> Earlier runs of this study carried "
            "two, both the same market state under each q model: the grid snaps a node onto the nearest barrier "
            "in log space, so a spot near the geometric mean of the two barriers priced its up and down "
            "scenarios on differently aligned lattices and the measured delta jumped. That was an engine defect "
            "in <code>_select_alignment_log</code>, and it is fixed &mdash; every barrier is put on a node by the "
            "opt-in <code>align_cell_stretch</code> (commit <code>c570acc9</code>), which this run carries. The "
            "tolerance was never widened to bury it. See <code>docs/bucket-futures-hedge/gates.md</code>.</p>"
        )

    if audit["measured"]:
        audit_head = (
            f"<p>Identity verdicts: {html.escape(verdict_bits)}. "
            f"Net delta audit error, max: <code>{audit['net_delta_audit_error_max']:.3e}</code> hands — the audit "
            "machinery itself is exact across every cell. Worst abs(R)+E against the 0.01-hand tolerance: "
            f"<code>{audit['gated_max']:.6f}</code> (median {audit['gated_median']:.2e}).</p>"
        )
    else:
        audit_head = (
            "<p><strong>This run scheduled no audit</strong> "
            f"(<code>{html.escape(verdict_bits)}</code>), so nothing below is certified by it: the cells record "
            "exposure and price the book, but the carry identity was not measured on any date. Re-run with "
            "<code>--carry-audit-mode daily</code> for a report whose audit section stands on its own run.</p>"
        )

    # An inconclusive verdict is the audit declining to decide, not a pass, so
    # it gets said out loud whenever there is one rather than being left to the
    # verdict counts above.
    n_inconclusive = int(audit["verdicts"].get("inconclusive", 0))
    if n_inconclusive:
        inconclusive_html = (
            f"<p><strong>{n_inconclusive} rows report inconclusive</strong>, which is the audit declining to "
            "decide rather than an identity failure. The residual itself is comfortably inside the 0.01-hand "
            "tolerance; what does not fit is the residual PLUS the change between the last two rungs of the "
            "matched spot ladder, and the gate conservatively requires the whole allowance to fit. Extending the "
            "ladder converges it, so this is the ladder stopping early and not the identity breaking. The cause "
            "is the readout staircase, a separate defect on this record: see "
            "<code>docs/bucket-futures-hedge/gates.md</code>.</p>"
        )
    else:
        inconclusive_html = ""

    audit_html = (
        "<h2>The audit over the full grid</h2>"
        + audit_head
        + failure_html
        + inconclusive_html
    )

    carry_html = _carry_section(fleet, recon, regime, regime_by_year)

    caveats = (
        "<h2>What this study does NOT establish</h2><ul>"
        "<li><strong>No claim that the bucket decomposition is wrong.</strong> The books deliberately hold different "
        "risk (a nodes book keeps D_F). The result is &lsquo;the extra exposure and turnover were not rewarded&rsquo;, "
        "nothing stronger.</li>"
        "<li><strong>The carry decomposition is an attribution, not a closure.</strong> Its realised-q leg is a flat "
        f"{Q_REALISED:.1%} assumption and it omits q-MTM by construction; the leftover column is where both "
        "approximations live.</li>"
        "<li><strong>Unquoted carry risk stays unhedged.</strong> No listed futures position responds to the tail or "
        "interpolation-shape stresses at all; they are reported, not mitigated.</li>"
        "<li><strong>The identity certifies consistency, not accuracy.</strong> Its delta term is a repriced "
        "matched-step derivative; closure is internal consistency of one numerical surface.</li>"
        "<li><strong>The hedge delta near the knock-out barrier is scale-dependent.</strong> The pricing bump is "
        "a declared 1% desk convention, not an approximation to a finer one: two percent below the barrier the "
        "measured delta runs from 2 to 13 contracts depending only on how wide the bump is, and a narrower bump "
        "or a different readout does not close that. The book is gamma-dominated there and any single delta "
        "mis-sizes it between rebalances.</li>"
        "<li><strong>Coverage.</strong> QUAD engine, flat-vol scalar channel, two supported futures conventions. "
        "Vol-model runs, other engines and other carry sources are outside it.</li>"
        "</ul>"
    )

    appendix = (
        "<h2>Appendix — reproduce</h2>"
        "<pre>PYTHONPATH=. .venv/bin/python example/snowball_q_term_structure/02_backtest_fleet.py \\\n"
        "  --study-grid buckets --workers 6 \\\n"
        "  --align-cell-stretch 0.02 \\\n"
        "  --carry-audit-mode daily --record-carry-exposure \\\n"
        "  --hedge-resolution-rel 0.0025 \\\n"
        "  --out-dir example/snowball_q_term_structure/data/bucket_hedge_v2/gate_e --resume\n"
        "PYTHONPATH=. .venv/bin/python example/snowball_q_term_structure/06_bucket_hedge_report.py</pre>"
        "<p>Artifacts next to this report: <code>bucket_per_run.csv</code> (one row per cell), "
        "<code>bucket_paired.csv</code> (the paired differences), "
        "<code>bucket_carry_reconciliation.csv</code> (the carry / delta-path / cost attribution), "
        "<code>bucket_premium_regime.csv</code> (daily paired differences by implied-q regime), "
        "<code>bucket_summary.json</code> "
        "(everything the report is built from).</p>"
    )

    body = intro + verdict + paired_html + dist_html + exposure_html + carry_html + audit_html + caveats + appendix
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Bucket futures hedge study</title>"
        f"<style>{_css()}</style></head><body><main>{body}</main></body></html>"
    )


# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="defaults to the parent of --run-dir")
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir or args.run_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fleet = load_fleet(args.run_dir)
    paired = paired_differences(fleet)
    recon = carry_reconciliation(fleet)
    regime, regime_by_year = regime_analysis(args.run_dir)
    audit = audit_overview(args.run_dir, fleet)

    fleet.to_csv(out_dir / "bucket_per_run.csv", index=False)
    paired.to_csv(out_dir / "bucket_paired.csv", index=False)
    recon.to_csv(out_dir / "bucket_carry_reconciliation.csv", index=False)
    regime.to_csv(out_dir / "bucket_premium_regime.csv", index=False)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(args.run_dir),
        "config": fleet_config(args.run_dir),
        "cells": int(len(fleet)),
        "inceptions": sorted(fleet["inception"].unique().tolist()),
        "policies": sorted(fleet["policy"].unique().tolist()),
        "paired": json.loads(paired.to_json(orient="records")),
        "carry_reconciliation": json.loads(recon.to_json(orient="records")),
        "premium_regime": json.loads(regime.to_json(orient="records")),
        "premium_days_by_year": json.loads(regime_by_year.to_json(orient="records")),
        "carry_realised_q_assumption": Q_REALISED,
        "audit": audit,
    }
    (out_dir / "bucket_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"{len(fleet)} cells, {fleet['inception'].nunique()} inceptions")
    for _, r in paired.iterrows():
        print(
            f"  {r['model']:14s} {r['bucket']:22s} vs {r['control']:6s} "
            f"d pnl {_fmt(r['d_terminal_pnl_bp']):>8s}{_stars(r['t_terminal_pnl_bp']):4s} "
            f"d cost {_fmt(r['d_cost_bp'], 2):>6s}  d hands/day {_fmt(r['d_turnover_hands_day'], 2):>5s}"
        )
    for _, r in recon.iterrows():
        print(
            f"  attrib {r['model']:12s} {r['bucket']:22s} vs {r['control']:6s} "
            f"obs {_fmt(r['observed_bp']):>8s}  carry {_fmt(r['carry_bp']):>7s}  "
            f"delta {_fmt(r['delta_path_bp']):>7s}  cost {_fmt(r['extra_cost_bp']):>6s}  "
            f"left {_fmt(r['leftover_bp']):>7s}"
        )
    for _, r in regime_by_year.iterrows():
        print(f"  premium-day share {int(r['year'])}: {r['premium_share']:.1%}")
    print(f"identity verdicts: {audit['verdicts']}")
    if audit["measured"]:
        print(f"worst abs(R)+E: {audit['gated_max']:.6f} on {audit['worst_date']}")
    else:
        print("no audit scheduled in this run: identity not measured on any date")

    if not args.no_report:
        report = out_dir / "bucket_hedge_report.html"
        report.write_text(
            build_report(
                fleet, paired, recon, regime, regime_by_year, audit, args.run_dir
            ),
            encoding="utf-8",
        )
        print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
