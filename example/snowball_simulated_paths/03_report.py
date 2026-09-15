"""Stage 03: distributions, paired differences, stress, the engine check, and the report.

    .venv/bin/python example/snowball_simulated_paths/03_report.py

Reads every cell under ``<out>/cells``, writes the tables to the study's
``data/`` directory and the self-contained ``simulated_paths_report.html``.
The historical study's realised runs (``output/snowball_q_term_structure``)
are located inside the simulated distribution when that directory exists.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _sim_common as C  # noqa: E402

from quantark.backtest.simulation.measures import path_measures  # noqa: E402
from quantark.backtest.simulation.results import EnsembleResults  # noqa: E402
from quantark.util.exceptions import ValidationError  # noqa: E402

HEADLINE_MEASURES = (
    "terminal_pnl_bp", "daily_pnl_std_bp", "variance_reduction_r2", "max_drawdown_bp", "turnover", "cost_bp",
    "roll_day_mtm_jump_bp", "other_day_mtm_jump_bp", "delta_churn",
)
#: Measures whose bad tail is the upper one (cost-like); the rest are P&L-like.
UPPER_TAIL = ("daily_pnl_std_bp", "max_drawdown_bp", "turnover", "cost_bp",
              "roll_day_mtm_jump_bp", "other_day_mtm_jump_bp", "delta_churn")
SUFFIXES = ("__stress", "__ladder_quad", "__exact_quad")


def _is_bootstrap_cell(name: str) -> bool:
    return not any(name.endswith(s) for s in SUFFIXES)


def load_cells(out_dir) -> Dict[str, EnsembleResults]:
    cells_dir = Path(out_dir) / "cells"
    if not cells_dir.exists():
        raise ValidationError(f"no cells under {cells_dir}; run 02_ensemble_fleet.py first")
    out = {}
    for path in sorted(p for p in cells_dir.iterdir() if (p / "manifest.json").exists()):
        out[path.name] = EnsembleResults.from_dir(path)
    if not out:
        raise ValidationError(f"no persisted cell under {cells_dir}")
    return out


def _distributions(results: EnsembleResults, es_level: float) -> Dict[str, Any]:
    return {measure: results.distribution(measure, es_level=es_level, tail="upper" if measure in UPPER_TAIL else "lower")
            for measure in HEADLINE_MEASURES}


def _paired(variant: EnsembleResults, base: EnsembleResults) -> Dict[str, Any]:
    comparison = variant.paired(base)
    return {measure: comparison.describe(measure) for measure in HEADLINE_MEASURES}


def _stress_rows(name: str, results: EnsembleResults) -> List[Dict[str, Any]]:
    names = results.manifest.get("path_meta", {}).get("scenario_names") or [f"scenario_{i}" for i in range(results.n_paths)]
    rows = []
    for i, row in results.summary.iterrows():
        rows.append({"cell": name, "scenario": names[int(i)], "terminal_pnl_bp": row["terminal_pnl_bp"],
                     "daily_pnl_std_bp": row["daily_pnl_std_bp"], "max_drawdown_bp": row["max_drawdown_bp"],
                     "termination": row["termination_reason"], "knocked_in": bool(row["knocked_in"]), "days": int(row["days"])})
    return rows


def _historical(historical_dir, cells: Dict[str, EnsembleResults]) -> Dict[str, Any]:
    if historical_dir is None:
        return {"available": False, "reason": "no historical directory given", "rows": []}
    runs_dir = Path(historical_dir) / "runs"
    if not runs_dir.exists():
        return {"available": False, "reason": f"{runs_dir} does not exist (run the q term-structure study first)", "rows": []}
    rows = []
    for summary_path in sorted(runs_dir.glob("*/*/run_summary.json")):
        summary = C.read_json(summary_path)
        cell = C.cell_name(summary["model"], summary["hedge"])
        if cell not in cells:
            continue
        run = C.Q.load_run(summary_path.parent)
        measures = path_measures(run["states"], run["trades"], notional=float(summary["notional"]))
        simulated = cells[cell].summary
        row = {"cell": cell, "inception": summary["inception"], **{m: measures[m] for m in HEADLINE_MEASURES}}
        for measure in ("terminal_pnl_bp", "daily_pnl_std_bp"):
            values = simulated[measure].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{measure.replace('_bp', '')}_percentile"] = (
                float((values <= measures[measure]).mean() * 100.0) if values.size else None
            )
        rows.append(row)
    if not rows:
        return {"available": False, "reason": f"no run under {runs_dir} matches a simulated cell", "rows": []}
    return {"available": True, "reason": None, "rows": rows}


def _portable(value: Any) -> Any:
    """Paths relative to the project root where possible: the tables ship in a public repository."""
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_portable(v) for v in value]
    if isinstance(value, str) and value.startswith("/"):
        try:
            return str(Path(value).relative_to(C.PROJECT_ROOT))
        except ValueError:
            return Path(value).name
    return value


def _provider_of(out_dir, name: str) -> str:
    """The provider a run was configured with, from its ``config.json``."""
    path = Path(out_dir) / "cells" / name / "config.json"
    if not path.exists():
        raise ValidationError(f"{path} is missing; every persisted run writes its config")
    return str(C.read_json(path)["metadata"]["provider"])


def aggregate(out_dir, *, es_level: float, historical_dir) -> Dict[str, Any]:
    """Everything the report shows, as plain dicts and lists."""
    out_dir = Path(out_dir)
    cells = load_cells(out_dir)
    bootstrap = {name: r for name, r in cells.items() if _is_bootstrap_cell(name)}
    agg: Dict[str, Any] = {
        "out_dir": _portable(str(out_dir)), "es_level": float(es_level),
        "fleet": _portable(C.read_json(out_dir / "fleet_manifest.json")) if (out_dir / "fleet_manifest.json").exists() else {},
        "coupon": C.read_json(out_dir / "coupon.json") if (out_dir / "coupon.json").exists() else {},
        "cells": {name: {"n_paths": r.n_paths, "distributions": _distributions(r, es_level), "manifest_mode": r.manifest.get("mode"),
                         "gate": r.manifest.get("gate"), "solves": r.manifest.get("solves"), "engine_calls": r.manifest.get("engine_calls"),
                         "seconds": r.manifest.get("seconds"),
                         "provider": _provider_of(out_dir, name),
                         "day0_book_mark_bp": float(r.cube.product_mtm[0, 0]) / r.notional * 1e4}
                  for name, r in bootstrap.items()},
        "paired": [], "stress": [], "engine_check": [], "gates": {}, "historical": None,
    }
    base = bootstrap.get(C.BASELINE_CELL)
    for name, r in bootstrap.items():
        if base is not None and name != C.BASELINE_CELL:
            agg["paired"].append({"variant": name, "base": C.BASELINE_CELL, "measures": _paired(r, base)})
    for model in C.MODELS:
        front, far = bootstrap.get(C.cell_name(model, "front")), bootstrap.get(C.cell_name(model, "far"))
        if front is not None and far is not None:
            agg["paired"].append({"variant": C.cell_name(model, "far"), "base": C.cell_name(model, "front"),
                                  "measures": _paired(far, front)})
    for name, r in cells.items():
        if name.endswith("__stress"):
            agg["stress"] += _stress_rows(name[: -len("__stress")], r)
    for name, r in cells.items():
        for suffix in ("__ladder_quad", "__exact_quad"):
            cell = name[: -len(suffix)]
            if name.endswith(suffix) and cell in bootstrap:
                same_paths = bootstrap[cell].take(range(r.n_paths))
                agg["engine_check"].append({
                    "cell": cell, "check": suffix[2:], "n": r.n_paths,
                    "pair": f"{_provider_of(out_dir, cell)} minus {_provider_of(out_dir, name)}",
                    "measures": _paired(same_paths, r),
                })
    for run_path in sorted((out_dir / "cells").glob("*/run.json")):     # failed runs have a run.json and no results
        run = C.read_json(run_path)
        agg["gates"][run_path.parent.name] = {**run["gate"], "oracle": run.get("oracle", []),
                                              "seconds": run.get("seconds"), "failed": bool(run.get("failed"))}
    agg["historical"] = _historical(historical_dir, bootstrap)
    return agg


def write_tables(agg: Dict[str, Any], data_dir) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    C.write_json(data_dir / "fleet_cells.json", {"cells": agg["cells"], "paired": agg["paired"]})
    paired_rows = [{"variant": p["variant"], "base": p["base"], "measure": m, **d}
                   for p in agg["paired"] for m, d in p["measures"].items()]
    pd.DataFrame(paired_rows).to_csv(data_dir / "fleet_paired.csv", index=False)
    pd.DataFrame(agg["stress"]).to_csv(data_dir / "stress_table.csv", index=False)
    check_rows = [{"cell": e["cell"], "check": e["check"], "pair": e["pair"], "n": e["n"], "measure": m, **d}
                  for e in agg["engine_check"] for m, d in e["measures"].items()]
    pd.DataFrame(check_rows).to_csv(data_dir / "engine_check.csv", index=False)
    if agg["historical"]["available"]:
        pd.DataFrame(agg["historical"]["rows"]).to_csv(data_dir / "historical_location.csv", index=False)
    C.write_json(data_dir / "fleet_summary.json", {k: v for k, v in agg.items() if k not in ("stress",)})


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 1, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{value * 100:.{digits}f}%" if pct else f"{value:,.{digits}f}"
    return html.escape(str(value))


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], caption: str = "") -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
    return f"<table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _histograms(cells: Dict[str, EnsembleResults]) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # matplotlib is optional here, as in the q study
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, r in cells.items():
        values = r.summary["terminal_pnl_bp"].to_numpy(dtype=float)
        ax.hist(values[np.isfinite(values)], bins=40, histtype="step", label=name)
    ax.set_xlabel("terminal hedged P&L (bp of notional)")
    ax.legend(fontsize=7)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _setup_section(agg: Dict[str, Any]) -> str:
    paths = agg.get("fleet", {}).get("paths", {}) or {}
    coupon = agg.get("coupon", {}) or agg.get("fleet", {}).get("coupon", {}) or {}
    terms = agg.get("fleet", {}).get("terms", {}) or {}
    rows = [
        ("history", f"{_fmt(paths.get('history_first_day'))} to {_fmt(paths.get('history_last_day'))}"),
        ("history fingerprint", _fmt(paths.get("history_fingerprint"))),
        ("bootstrap paths x days", f"{_fmt(paths.get('n_paths'))} x {_fmt(paths.get('n_days'))}"),
        ("calendar", f"{_fmt(paths.get('calendar_first_day'))} to {_fmt(paths.get('calendar_last_day'))}"),
        ("stress scenarios", ", ".join((paths.get("stress_meta") or {}).get("scenario_names", [])) or "–"),
        ("product", f"maturity {_fmt(terms.get('maturity_years'), 3)}y, KO {_fmt(terms.get('ko_pct'), 2)}, "
                    f"KI {_fmt(terms.get('ki_pct'), 2)}, {_fmt(terms.get('n_ko'))} KO observations"),
        ("fair coupon", f"{_fmt((coupon.get('coupon') or {}).get('coupon') if isinstance(coupon.get('coupon'), dict) else coupon.get('coupon'), 2, pct=True)} "
                        f"under {_fmt(coupon.get('reference_model'))}"),
        ("expected-shortfall level", _fmt(agg["es_level"], 2, pct=True)),
    ]
    return "<h2>Setup</h2>" + _table(["item", "value"], [(html.escape(k), v) for k, v in rows])


def _distribution_section(agg: Dict[str, Any], image: Optional[str]) -> str:
    parts = ["<h2>Hedge-cost distributions per cell</h2>",
             "<p>One row per cell and measure over the bootstrap paths; expected shortfall is the mean of the "
             "stated tail (the loss tail of a P&L measure, the upper tail of a cost-like one). KO and KI "
             "frequencies are over every path of the cell.</p>"]
    headers = ["cell", "measure", "n", "mean", "std", "q05", "q50", "q95", "expected shortfall", "share > 0",
               "KO freq", "KI freq"]
    rows = []
    for cell, entry in agg["cells"].items():
        for measure, d in entry["distributions"].items():
            q = d.get("quantiles") or {}
            rows.append([html.escape(cell), html.escape(measure), _fmt(d["n"]), _fmt(d.get("mean"), 2), _fmt(d.get("std"), 2),
                         _fmt(q.get("q05"), 2), _fmt(q.get("q50"), 2), _fmt(q.get("q95"), 2),
                         _fmt(d.get("expected_shortfall"), 2), _fmt(d.get("share_positive"), 1, pct=True),
                         _fmt(d.get("ko_frequency"), 1, pct=True), _fmt(d.get("ki_frequency"), 1, pct=True)])
    parts.append(_table(headers, rows))
    if image:
        parts.append(f"<p><img src='{image}' alt='terminal P&amp;L histograms' style='max-width:100%'></p>")
    return "\n".join(parts)


def _paired_section(agg: Dict[str, Any]) -> str:
    parts = ["<h2>Paired differences on matched paths</h2>",
             "<p>Variant minus base on the same simulated paths; the t-statistic is the paired one over "
             "independent paths.</p>"]
    headers = ["variant", "base", "measure", "n", "mean", "median", "share > 0", "t-stat"]
    rows = []
    for p in agg["paired"]:
        for measure, d in p["measures"].items():
            rows.append([html.escape(p["variant"]), html.escape(p["base"]), html.escape(measure), _fmt(d["n"]),
                         _fmt(d.get("mean"), 2), _fmt(d.get("median"), 2), _fmt(d.get("share_positive"), 1, pct=True),
                         _fmt(d.get("t_stat"), 2)])
    parts.append(_table(headers, rows) if rows else "<p>No paired comparison: the baseline cell was not run.</p>")
    return "\n".join(parts)


def _stress_section(agg: Dict[str, Any]) -> str:
    headers = ["cell", "scenario", "terminal P&L bp", "daily std bp", "max drawdown bp", "termination", "knocked in", "days"]
    rows = [[html.escape(r["cell"]), html.escape(r["scenario"]), _fmt(r["terminal_pnl_bp"]), _fmt(r["daily_pnl_std_bp"]),
             _fmt(r["max_drawdown_bp"]), html.escape(str(r["termination"])), _fmt(r["knocked_in"]), _fmt(r["days"])]
            for r in agg["stress"]]
    return "<h2>Stress scenarios</h2>" + (_table(headers, rows) if rows else "<p>No stress run persisted.</p>")


def _engine_section(agg: Dict[str, Any]) -> str:
    parts = ["<h2>Engine check: each cell against QUAD on the same paths</h2>",
             "<p>A cell's first paths paired against the same paths repriced on QUAD: the cell's provider minus "
             "the check's, per measure. Reported, not gated: exact PDE and exact QUAD are both exact engines "
             "with different numerics, so no threshold is claimed.</p>"]
    headers = ["cell", "pair", "n", "measure", "mean difference", "median", "std"]
    rows = [[html.escape(e["cell"]), html.escape(e["pair"]), _fmt(e["n"]), html.escape(m), _fmt(d.get("mean"), 2),
             _fmt(d.get("median"), 2), _fmt(d.get("std"), 2)]
            for e in agg["engine_check"] for m, d in e["measures"].items()]
    parts.append(_table(headers, rows) if rows else "<p>No QUAD check run persisted.</p>")
    return "\n".join(parts)


def _day0_section(agg: Dict[str, Any]) -> str:
    headers = ["cell", "provider", "day-0 book mark bp"]
    rows = [[html.escape(name), html.escape(entry["provider"]), _fmt(entry["day0_book_mark_bp"], 2)]
            for name, entry in agg["cells"].items()]
    return ("<h2>Day-0 book marks</h2><p>Every cell books the traded price (0) at inception and the fair coupon "
            "is solved on QUAD, so a cell's day-0 mark is its provider's price of the traded contract: the "
            "engine gap at inception, carried in every terminal P&amp;L of that cell.</p>" + _table(headers, rows))


def _gate_section(agg: Dict[str, Any]) -> str:
    headers = ["run", "mode", "sampled states", "max PV gap bp", "max delta gap hands", "gate", "oracle paths", "oracle", "seconds"]
    rows = []
    for name, g in agg["gates"].items():
        oracle = g.get("oracle") or []
        rows.append([html.escape(name), html.escape(str(g.get("mode"))), _fmt(g.get("sampled")), _fmt(g.get("max_pv_gap_bp"), 2),
                     _fmt(g.get("max_delta_gap_hands"), 2), "pass" if g.get("passed") else "FAIL (no results)", _fmt(len(oracle)),
                     ("pass" if all(r.get("passed") for r in oracle) else "FAIL") if oracle else "–", _fmt(g.get("seconds"), 0)])
    return ("<h2>Gates and oracle spot checks</h2>"
            "<p>The gate reprices a reservoir of visited states exactly and reports the worst gap; the oracle "
            "runs single paths through the replay engine.</p>" + _table(headers, rows))


def _historical_section(agg: Dict[str, Any]) -> str:
    hist = agg["historical"] or {"available": False, "reason": "not computed", "rows": []}
    if not hist["available"]:
        return f"<h2>Historical runs inside the simulated distribution</h2><p>Not available: {html.escape(str(hist['reason']))}.</p>"
    headers = ["cell", "inception", "terminal P&L bp", "percentile", "daily std bp", "percentile", "R²", "cost bp"]
    rows = [[html.escape(r["cell"]), html.escape(str(r["inception"])), _fmt(r["terminal_pnl_bp"]), _fmt(r.get("terminal_pnl_percentile")),
             _fmt(r["daily_pnl_std_bp"]), _fmt(r.get("daily_pnl_std_percentile")), _fmt(r["variance_reduction_r2"], 2), _fmt(r["cost_bp"])]
            for r in hist["rows"]]
    return ("<h2>Historical runs inside the simulated distribution</h2>"
            "<p>Each realised inception of the q term-structure study, measured with the same library "
            "function, and the share of simulated paths at or below it.</p>" + _table(headers, rows))


def _caveats_section() -> str:
    return ("<h2>Caveats</h2><ul>"
            "<li>Every simulated path starts from one state, the history's last day, so a historical inception's "
            "percentile is indicative: its own start state differs.</li>"
            "<li>Under the life surface and the spot ladder the term dividend models enter only through the scalar "
            "yield at the remaining maturity, flat at the bucket centre; under per_date and exact every state "
            "hands the engine that date's term dividend object.</li>"
            "<li>Paired t-statistics treat the simulated paths as independent draws, unlike the historical study's "
            "overlapping inceptions.</li>"
            "<li>The stress paths are designed, not sampled; they show mechanics, not probabilities.</li>"
            "</ul>")


def build_report(agg: Dict[str, Any], cells: Optional[Dict[str, EnsembleResults]] = None) -> str:
    """The self-contained HTML report, one section per part of ``agg``."""
    image = _histograms(cells) if cells else None
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Snowball hedging on simulated paths</title>",
             "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}"
             "table{border-collapse:collapse;margin:1em 0;font-size:13px}th,td{border:1px solid #ccc;padding:3px 8px;text-align:right}"
             "th:first-child,td:first-child{text-align:left}caption{text-align:left;font-weight:600}</style></head><body>",
             "<h1>Snowball hedging on simulated paths</h1>",
             _setup_section(agg), _day0_section(agg), _distribution_section(agg, image), _paired_section(agg), _stress_section(agg),
             _engine_section(agg), _gate_section(agg), _historical_section(agg), _caveats_section(),
             "</body></html>"]
    return "\n".join(parts)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=C.DEFAULT_OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    parser.add_argument("--es-level", type=float, default=0.05)
    parser.add_argument("--historical-dir", type=Path, default=C.DEFAULT_HISTORICAL_DIR)
    parser.add_argument("--no-historical", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cells = load_cells(args.out_dir)
    agg = aggregate(args.out_dir, es_level=args.es_level, historical_dir=None if args.no_historical else args.historical_dir)
    write_tables(agg, args.data_dir)
    report = build_report(agg, {n: r for n, r in cells.items() if _is_bootstrap_cell(n)})
    (Path(args.data_dir) / "simulated_paths_report.html").write_text(report)
    print(f"{len(agg['cells'])} cells, {len(agg['paired'])} paired comparisons, {len(agg['stress'])} stress rows, "
          f"historical {'located' if agg['historical']['available'] else 'not available: ' + str(agg['historical']['reason'])}; "
          f"report at {Path(args.data_dir) / 'simulated_paths_report.html'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
