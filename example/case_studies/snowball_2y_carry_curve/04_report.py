"""Stage 04 — render the study as one self-contained HTML report.

Reads ``data/market_20261105.json``, ``results/pricing.json`` and
``results/greeks.json`` (stages 01–03) and writes ``results/report.html``:
the question, the chain and the curve per tail convention, the misprice, the
event statistics, the static hedge, the scenarios, the findings and the
method. Every number in the prose is computed from those files here.

The page follows ``docs/UI_LANGUAGE.md`` (Bloomberg × Swiss): a frozen
navigator (function bar + numbered menu), a light/dark switcher, inline SVG
charts styled by the same CSS tokens, a table under every chart, true minus
signs. No network assets: it opens offline.

Run:
    python example/case_studies/snowball_2y_carry_curve/04_report.py
"""

from __future__ import annotations

import html
import importlib.util
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent


def _common():
    name = "snowball_2y_cc_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _HERE / "_common.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


REPORT_NAME = "report.html"
MINUS = "−"
TAIL_ORDER = ("hold_forward", "hold_yield", "zero_carry")
#: Series slots in fixed order (docs/UI_LANGUAGE.md tokens), one per tail.
SERIES = {"hold_forward": "var(--s1)", "hold_yield": "var(--s2)", "zero_carry": "var(--s3)"}
#: Curve chart x range and sampling.
CURVE_X_MAX = 2.1
CURVE_GRID = np.linspace(0.005, CURVE_X_MAX, 421)
CURVE_TABLE_T = (0.25, 0.5, 0.75, 1.0, 1.5)
#: Stage-03 cost profile measured while building it (engine-side, out of scope).
QUAD_RESPOT_SECONDS = 13.0
QUAD_CACHED_SECONDS = 0.8


# ---------------------------------------------------------------------------
# Number formatting: true minus, never a negative zero
# ---------------------------------------------------------------------------


def _m(text: str) -> str:
    return text.replace("-", MINUS)


def _r(v: float, d: int) -> float:
    return round(float(v), d) + 0.0


def num(v: Optional[float], d: int = 2) -> str:
    return "–" if v is None else _m(f"{_r(v, d):,.{d}f}")


def pct(v: Optional[float], d: int = 2) -> str:
    """A fraction shown as a percentage."""
    return "–" if v is None else num(100.0 * float(v), d) + "%"


def snum(v: float, d: int = 2) -> str:
    """Signed text (no colour): for tooltips and prose. A value that rounds to zero carries no sign."""
    v = _r(v, d)
    return _m(f"{v:,.{d}f}" if v == 0 else f"{v:+,.{d}f}")


def signed(v: float, d: int = 2, unit: str = "") -> str:
    """A signed change: sign printed AND coloured (colour never carries it alone)."""
    v = _r(v, d)
    cls = "up" if v > 0 else "dn" if v < 0 else "flat"
    return f'<span class="{cls}">{snum(v, d)}{unit}</span>'


def esc(s: str) -> str:
    return html.escape(str(s))


def short_label(label: str) -> str:
    """``MO2709 C-P`` → ``MO2709``; IM codes unchanged."""
    return label.split()[0]


# ---------------------------------------------------------------------------
# Inputs and derived quantities
# ---------------------------------------------------------------------------


def load_inputs() -> Dict[str, Any]:
    """The three study files plus the curves rebuilt from the snapshot (for the chart)."""
    c = _common()
    snap = c.load_snapshot()
    pr = c.read_json(c.RESULTS_DIR / "pricing.json")
    gk = c.read_json(c.RESULTS_DIR / "greeks.json")
    t_mat = float(pr["terms"]["maturity_years"])
    curves: Dict[str, Dict[str, Any]] = {}
    for tail in TAIL_ORDER:
        curve = c.build_curve(snap, tail)
        view = curve.to_dividend_yield(c.rate_curve())
        nodes_q = [float(view.get_yield(t)) for t in curve.tenors]
        for got, rec in zip(nodes_q, pr["tails"][tail]["nodes"]):
            if not math.isclose(got, rec["q"], rel_tol=1e-12, abs_tol=1e-15):
                raise c.StudyDataError("results/pricing.json is stale against the snapshot: rerun 02_pricing.py")
        curves[tail] = {
            "grid_q": np.asarray(view.get_yield(CURVE_GRID), dtype=float),
            "table_q": [float(view.get_yield(t)) for t in CURVE_TABLE_T] + [float(view.get_yield(t_mat))],
            "q_at": lambda t, _v=view: float(_v.get_yield(t)),
        }
    return {"c": c, "snap": snap, "pr": pr, "gk": gk, "t_mat": t_mat, "curves": curves}


def headline(d: Dict[str, Any]) -> Dict[str, float]:
    """The handful of numbers the hero, the findings and the README quote."""
    pr, gk = d["pr"], d["gk"]
    hf = pr["tails"]["hold_forward"]
    ghf = gk["tails"]["hold_forward"]
    pvs = [pr["tails"][t]["curve"]["pv_bp"] for t in TAIL_ORDER]
    gaps_mark = [hf["gap_bp"]] + [pr["marks"][m]["tails"]["hold_forward"]["gap_bp"] for m in ("mo_close", "all_settle")]
    t_last = hf["nodes"][-1]["tenor"]
    return {
        "coupon": pr["coupon"]["value"],
        "gap_bp": hf["gap_bp"],
        "coupon_equiv": hf["coupon_equiv_gap"],
        "q_flat": hf["q_flat"],
        "tail_share": ghf["forward_interval_rho"][-1]["rho_q"] / ghf["curve"]["parallel_rho_q"],
        "pv_min": min(pvs),
        "pv_max": max(pvs),
        "pv_spread": max(pvs) - min(pvs),
        "max_abs_gap": max(abs(pr["tails"][t]["gap_bp"]) for t in TAIL_ORDER),
        "mark_gap_min": min(gaps_mark),
        "mark_gap_max": max(gaps_mark),
        "unquoted_share": 1.0 - t_last / d["t_mat"],
        "t_last": t_last,
    }


# ---------------------------------------------------------------------------
# SVG charts (inline, styled by the CSS tokens so the theme switch repaints them)
# ---------------------------------------------------------------------------


def _nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    span = hi - lo
    raw = span / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    ticks, t = [], start
    while t <= hi + step * 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks


def _legend(items: Sequence[Tuple[str, str, str]]) -> str:
    """``items``: [(name, colour, css border style)]."""
    spans = "".join(f'<span><i style="border-top:3px {style} {colour}"></i>{esc(name)}</span>'
                    for name, colour, style in items)
    return f'<div class="legend">{spans}</div>'


def _svg(width: int, height: int, body: List[str], label: str) -> str:
    return (f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}">'
            + "".join(body) + "</svg>")


def xy_chart(series: Sequence[Dict[str, Any]], *, xdomain: Tuple[float, float], xticks: Sequence[float],
             xfmt: Callable[[float], str], yfmt: Callable[[float], str], ylabel: str, xlabel: str, label: str,
             zero_anchor: bool = False, pillars: Sequence[Tuple[float, str]] = (),
             shade_from: Optional[float] = None, shade_label: str = "",
             vlines: Sequence[Tuple[float, str]] = (), width: int = 760, height: int = 330) -> str:
    """Numeric-x line chart. ``series``: dicts with ``name, xs, ys, colour, dash, end_label`` and
    optional ``points`` [(x, y, tip)] drawn as ringed markers with an enlarged hover target.
    The first series is drawn on top."""
    left, right, top, bottom = 62, 150, 46, 42
    ys = [y for s in series for y in s["ys"]]
    lo, hi = min(ys), max(ys)
    if zero_anchor:
        lo = 0.0
    pad = (hi - lo) * 0.08 or 1e-4
    lo_p = lo if zero_anchor else lo - pad
    yt = _nice_ticks(lo_p, hi + pad)
    if zero_anchor:
        yt = [t for t in yt if t >= 0.0]
    ylo, yhi = min(yt[0], lo_p), max(yt[-1], hi + pad)
    xlo, xhi = xdomain

    def X(v: float) -> float:
        return left + (v - xlo) / (xhi - xlo) * (width - left - right)

    def Y(v: float) -> float:
        return top + (yhi - v) / (yhi - ylo) * (height - top - bottom)

    base = height - bottom
    body: List[str] = []
    if shade_from is not None:
        xs0 = X(shade_from)
        body.append(f'<rect class="shade" x="{xs0:.1f}" y="{top}" width="{X(xhi) - xs0:.1f}" height="{base - top}"/>')
        body.append(f'<text class="annot" x="{xs0 + 8:.1f}" y="{top + 14}">{esc(shade_label)}</text>')
    for v in yt:
        y = Y(v)
        body.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
                    f'<text class="tick" x="{left - 8}" y="{y + 3.5:.1f}" text-anchor="end">{yfmt(v)}</text>')
    if ylo < 0 < yhi:
        body.append(f'<line class="zero" x1="{left}" x2="{width - right}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"/>')
    for k, (x, lab) in enumerate(pillars):
        xp = X(x)
        body.append(f'<line class="pgrid" x1="{xp:.1f}" x2="{xp:.1f}" y1="{top}" y2="{base}"/>'
                    f'<line class="pillar" x1="{xp:.1f}" x2="{xp:.1f}" y1="{base}" y2="{base + 6}"/>'
                    f'<text class="plab" x="{xp:.1f}" y="{top - 18 + 11 * (k % 2)}" text-anchor="middle">{esc(lab)}</text>')
    for x, lab in vlines:
        xv = X(x)
        body.append(f'<line class="vline" x1="{xv:.1f}" x2="{xv:.1f}" y1="{top}" y2="{base}"/>'
                    f'<text class="annot" x="{xv - 4:.1f}" y="{base - 6}" text-anchor="end">{esc(lab)}</text>')
    body.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{base}" y2="{base}"/>')
    for v in xticks:
        body.append(f'<text class="tick" x="{X(v):.1f}" y="{base + 18}" text-anchor="middle">{xfmt(v)}</text>')
    body.append(f'<text class="axlab" x="{left}" y="{height - 4}">{esc(xlabel)}</text>')
    body.append(f'<text class="axlab" x="6" y="12">{esc(ylabel)}</text>')
    ends = []
    # drawn last-to-first so the first (default) series sits on top where curves coincide
    for s in reversed(series):
        pts =" ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(s["xs"], s["ys"]))
        dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        body.append(f'<polyline class="ln" points="{pts}" style="stroke:{s["colour"]}"{dash}/>')
        for x, y, tip in s.get("points", ()):
            body.append(f'<circle class="mk" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.5" style="fill:{s["colour"]}"/>'
                        f'<circle class="hit" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="10" data-tip="{esc(tip)}" '
                        f'aria-label="{esc(tip)}"/>')
        if s.get("end_label"):
            ends.append([Y(s["ys"][-1]), s["end_label"], X(s["xs"][-1])])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 13)
    for y, name, x in ends:
        body.append(f'<text class="dlab" x="{x + 8:.1f}" y="{y + 3.5:.1f}">{esc(name)}</text>')
    legend = _legend([(s["name"], s["colour"], "dashed" if s.get("dash") else "solid") for s in series])
    return legend + _svg(width, height, body, label)


def grouped_bars(cats: Sequence[str], series: Sequence[Tuple[str, str, Sequence[float]]], *,
                 yfmt: Callable[[float], str], tipfmt: Callable[[float], str], ylabel: str, xlabel: str,
                 label: str, shade_last: str = "", width: int = 760, height: int = 300) -> str:
    """Signed grouped bars around a zero line. ``series``: [(name, colour, values)]."""
    left, right, top, bottom = 62, 16, 26, 46
    vals = [v for _, _, vs in series for v in vs]
    lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
    pad = (hi - lo) * 0.08 or 1.0
    lo_p, hi_p = (lo - pad if lo < 0 else 0.0), (hi + pad if hi > 0 else 0.0)
    yt = _nice_ticks(lo_p, hi_p)
    ylo, yhi = min(yt[0], lo_p), max(yt[-1], hi_p)

    def Y(v: float) -> float:
        return top + (yhi - v) / (yhi - ylo) * (height - top - bottom)

    n, k = len(cats), len(series)
    slot = (width - left - right) / n
    gw = slot * 0.72
    bw = (gw - 2 * (k - 1)) / k
    base = height - bottom
    body: List[str] = []
    if shade_last:
        x0 = left + slot * (n - 1)
        body.append(f'<rect class="shade" x="{x0:.1f}" y="{top}" width="{slot:.1f}" height="{base - top}"/>')
    for v in yt:
        y = Y(v)
        body.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
                    f'<text class="tick" x="{left - 8}" y="{y + 3.5:.1f}" text-anchor="end">{yfmt(v)}</text>')
    y0 = Y(0.0)
    for i, cat in enumerate(cats):
        gx = left + slot * i + (slot - gw) / 2
        for j, (name, colour, vs) in enumerate(series):
            v = vs[i]
            yv = Y(v)
            tip = f"{name} · {cat} · {tipfmt(v)}"
            body.append(f'<rect class="bar" x="{gx + j * (bw + 2):.1f}" y="{min(yv, y0):.1f}" width="{bw:.1f}" '
                        f'height="{max(abs(yv - y0), 0.5):.1f}" style="fill:{colour}" data-tip="{esc(tip)}" '
                        f'aria-label="{esc(tip)}"/>')
        body.append(f'<text class="tick" x="{left + slot * (i + 0.5):.1f}" y="{base + 18}" '
                    f'text-anchor="middle">{esc(cat)}</text>')
    if shade_last:
        # annotate on the side of the zero line the bars leave free
        ya = base - 8 if hi <= 0 else top + 12
        body.append(f'<text class="annot" x="{left + slot * (n - 1) + 6:.1f}" y="{ya}">{esc(shade_last)}</text>')
    body.append(f'<line class="zero" x1="{left}" x2="{width - right}" y1="{y0:.1f}" y2="{y0:.1f}"/>')
    body.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{base}" y2="{base}"/>')
    body.append(f'<text class="axlab" x="{left}" y="{height - 4}">{esc(xlabel)}</text>')
    body.append(f'<text class="axlab" x="6" y="12">{esc(ylabel)}</text>')
    legend = _legend([(name, colour, "solid") for name, colour, _ in series])
    return legend + _svg(width, height, body, label)


# ---------------------------------------------------------------------------
# HTML building blocks
# ---------------------------------------------------------------------------


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], note: str = "", text: bool = False) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    foot = f'<p class="note">{note}</p>' if note else ""
    cls = ' class="text"' if text else ""
    return f'<div class="scroll"><table{cls}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{foot}'


def figure(chart: str, title: str, caption: str) -> str:
    return (f'<figure><figcaption><b>{esc(title)}</b><span>{esc(caption)}</span></figcaption>'
            f"{chart}</figure>")


def section(no: int, code: str, title: str, content: str) -> str:
    return (f'<section id="s{no}"><div class="rail"><div class="numeral">{no:02d}</div>'
            f'<div class="code">{code}</div><h2>{title}</h2></div><div class="body">{content}</div></section>')


def tail_name(d: Dict[str, Any], tail: str) -> str:
    return d["c"].TAILS[tail][1]


def tail_head(d: Dict[str, Any], tail: str) -> str:
    """Table label of a tail (the enum names are listed in the setup and the leverage table)."""
    return esc(tail_name(d, tail))


def total_diffs(d: Dict[str, Any], tail: str) -> Tuple[float, float]:
    """|curve/flat − 1| of the spot delta and of the parallel rho, in %."""
    g = d["gk"]["tails"][tail]
    return (100 * abs(g["curve"]["spot_delta"] / g["flat"]["spot_delta"] - 1),
            100 * abs(g["curve"]["parallel_rho_q"] / g["flat"]["parallel_rho_q"] - 1))


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section_setup(d: Dict[str, Any]) -> str:
    c, snap, pr = d["c"], d["snap"], d["pr"]
    terms = pr["terms"]
    pair = snap["option_pair"]
    t_of = lambda iso: c.year_fraction(date.fromisoformat(iso))  # noqa: E731
    rows = []
    for f in snap["futures"]:
        rows.append([esc(f["contract"]), f["expiry"], num(t_of(f["expiry"]), 4), num(f["price"], 1),
                     num(f["close_cffex"], 1), num(f["settle"], 1), "close (dashboard)"])
    for leg in ("call", "put"):
        rows.append([esc(pair[f"{leg}_contract"]), pair["expiry"], num(t_of(pair["expiry"]), 4), "–",
                     num(pair[f"{leg}_close"], 1), num(pair[f"{leg}_settle"], 1), "settlement (CFFEX)"])
    syn = snap["dashboard_syn_fut"]
    f_mo = snap["mo_forward"]
    ko = ", ".join(terms["ko_dates"])
    vol = snap["vol"]
    return section(1, "SETUP", "The question, the trade, the market", f"""
<p class="lead">How much does a single flat dividend yield misprice and mis-hedge a 2Y CSI 1000 snowball, against the
term structure the IM futures and MO options chain actually quotes on {esc(snap["valuation_date"])}? Both models price the
same note with the same coupon, vol and rate; they differ only in the carry.</p>
<dl class="spec">
<dt>Underlying</dt><dd>CSI 1000 ({esc(snap["index_code"])}) · close {num(snap["spot"], 4)} on {esc(snap["valuation_date"])}
· inception = valuation date · numeric ACT/365F clock</dd>
<dt>Trade</dt><dd>2Y standard snowball · notional {num(c.NOTIONAL / 1e6, 0)} mio CNY · strike = initial = spot ·
maturity {esc(terms["maturity_date"])} (T = {num(terms["maturity_years"], 4)}Y) · principal excluded</dd>
<dt>Knock-out</dt><dd>{num(100 * terms["ko_pct"], 0)}% of initial, monthly from month {c.LOCKOUT_MONTHS}
({len(terms["ko_dates"])} SSE dates): <span class="dates">{esc(ko)}</span></dd>
<dt>Knock-in</dt><dd>{num(100 * terms["ki_pct"], 0)}% of initial, discrete on every SSE trading day
({terms["n_ki_observations"]} observations)</dd>
<dt>Coupon</dt><dd><b>{pct(pr["coupon"]["value"], 2)} p.a.</b> = KO coupon = rebate if neither event · solved to PV = 0
under the curve with the hold-forward tail ({esc(pr["coupon"]["solved_under"])}) and held fixed in every cell</dd>
<dt>Rate · vol</dt><dd>flat r = {pct(snap["rate"], 2)} (the dashboard's synthetic-forward convention) · flat vol
{pct(vol["value"], 2)} = Black-76 IV of the MO2709 settlement pair at K = {num(vol["strike"], 0)}, nearest the pillar
forward (call {pct(vol["call_iv"], 2)}, put {pct(vol["put_iv"], 2)})</dd>
<dt>Curve model</dt><dd>CarryCurve through the four IM futures and the MO2709 C−P synthetic forward, LOG_LINEAR_DF
(piecewise-constant forward carry), one of three tail conventions beyond the last pillar:
{"; ".join(f"{esc(lab)} = <code>{ext.name}</code>" for ext, lab in (c.TAILS[t] for t in TAIL_ORDER))}</dd>
<dt>Flat model</dt><dd>q_flat = q(T_mat) of the same curve and tail, so both models agree on F(T_mat) by construction</dd>
</dl>
<h3 class="sub">Chain on {esc(snap["valuation_date"])}</h3>
{table(["contract", "expiry", "T (Y)", "close (dashboard)", "close (CFFEX)", "settle (CFFEX)", "base mark"], rows,
       note="IM pillars at the dashboard close (equal to the CFFEX close); the MO pair at the CFFEX settlement. "
            "The mark-sensitivity rows in section 03 swap in the alternatives.")}
<p>MO2709 pillar forward <code>K + (C − P)·e<sup>rT</sup></code> = {num(f_mo, 2)} against the dashboard synthetic
forward {esc(syn["contract"])} {num(syn["price"], 2)} (live mids): {signed(100 * (f_mo / syn["price"] - 1), 2, "%")}.</p>
""")


def section_curve(d: Dict[str, Any]) -> str:
    pr, t_mat, h = d["pr"], d["t_mat"], headline(d)
    nodes = pr["tails"]["hold_forward"]["nodes"]
    pillars = [(n["tenor"], short_label(n["label"])[-4:] if n["label"].startswith("IM") else short_label(n["label"]))
               for n in nodes]
    series = []
    for tail in TAIL_ORDER:
        col = SERIES[tail]
        q_flat = pr["tails"][tail]["q_flat"]
        name = tail_name(d, tail)
        pts = [(n["tenor"], 100 * n["q"], f"{short_label(n['label'])} · T {num(n['tenor'], 3)} · q {pct(n['q'], 3)}")
               for n in nodes] if tail == "hold_forward" else []
        pts.append((t_mat, 100 * q_flat, f"{name} · q(T_mat) {pct(q_flat, 3)}"))
        series.append({"name": f"curve · {name}", "xs": list(CURVE_GRID), "ys": list(100 * d["curves"][tail]["grid_q"]),
                       "colour": col, "end_label": name, "points": pts})
    for tail in TAIL_ORDER:
        q_flat = pr["tails"][tail]["q_flat"]
        series.append({"name": f"flat q(T_mat) · {tail_name(d, tail)}", "xs": [0.0, CURVE_X_MAX],
                       "ys": [100 * q_flat] * 2, "colour": SERIES[tail], "dash": "6 4"})
    chart = xy_chart(series, xdomain=(0.0, CURVE_X_MAX), xticks=[0, 0.5, 1.0, 1.5, 2.0],
                     xfmt=lambda v: f"{v:g}", yfmt=lambda v: num(v, 0), ylabel="IMPLIED q(T), % P.A.",
                     xlabel="T, YEARS (ACT/365F) · TICKS AT PILLARS", label="Implied dividend yield by tail convention",
                     pillars=pillars, shade_from=h["t_last"],
                     shade_label=f"unquoted: {pct(h['unquoted_share'], 0)} of the life",
                     vlines=[(t_mat, "T_mat")])
    grid_rows = []
    for tail in TAIL_ORDER:
        tq = d["curves"][tail]["table_q"]
        grid_rows.append([tail_head(d, tail)] + [pct(v, 3) for v in tq] + [pct(pr["tails"][tail]["q_flat"], 3)])
    node_rows = [[esc(n["label"]), num(n["tenor"], 4), num(n["forward"], 2), num(n["carry"], 6), pct(n["q"], 3)]
                 for n in nodes]
    lev_rows = []
    for tail in TAIL_ORDER:
        lev = pr["tails"][tail]["tail_leverage"]
        lev_rows.append([f'{tail_head(d, tail)} <span class="dim">{d["c"].TAILS[tail][0].name}</span>']
                        + [num(x, 2) for x in lev] + [num(sum(lev), 2)])
    labels = [short_label(n["label"]) for n in nodes]
    return section(2, "CURVE", "The implied q curve and its unquoted tail", f"""
<p class="lead">The chain pins q only out to the MO2709 expiry, T = {num(h["t_last"], 3)}Y. The note lives to
{num(t_mat, 3)}Y, so {pct(h["unquoted_share"], 0)} of its life sits beyond the last quote, where the curve is whatever
the tail convention says.</p>
{figure(chart, "Implied q(T) per tail convention, with each tail's flat q(T_mat)",
        "Solid: the curve. Dashed: the flat model's q, equal to the curve at T_mat. Dots: the pillars.")}
{table(["tail"] + [f"q({t:g}Y)" for t in CURVE_TABLE_T] + ["q(T_mat)", "flat q"], grid_rows,
       note="The same curves as the chart, read at fixed tenors; q(T_mat) = flat q by construction.", )}
<h3 class="sub">Pillars (shared by all tails)</h3>
{table(["pillar", "T (Y)", "forward F", "carry B = ln F/S", "zero q"], node_rows,
       note="Inside the last pillar the three tails are identical.")}
<h3 class="sub">Tail leverage ∂B(T_mat)/∂B_i</h3>
{table(["tail"] + labels + ["sum"], lev_rows,
       note="How a move in pillar i's carry reaches the maturity carry. Hold-forward extends the last segment's slope, "
            "so the 2Y carry is a levered IM2706/MO2709 spread; hold-yield scales the last node by T_mat/T_n; "
            "zero-carry holds it.")}
""")


def section_price(d: Dict[str, Any]) -> str:
    pr, h = d["pr"], headline(d)
    rows, val_rows = [], []
    for tail in TAIL_ORDER:
        t = pr["tails"][tail]
        rows.append([tail_head(d, tail), num(t["curve"]["pv_bp"], 2), num(t["flat"]["pv_bp"], 2),
                     signed(t["gap_bp"], 2), signed(-100 * t["coupon_equiv_gap"], 2, "%"), pct(t["q_flat"], 3)])
        for model in ("curve", "flat"):
            m = t[model]
            val_rows.append([tail_head(d, tail), model, num(m["pv_bp"], 3), num(m["pv_bp_fine_grid"], 3),
                             signed(m["pv_bp_fine_grid"] - m["pv_bp"], 3), num(m["pv_bp_pde"], 3),
                             signed(m["pv_bp_pde"] - m["pv_bp"], 3)])
        val_rows.append([tail_head(d, tail), "<b>gap</b>", num(t["gap_bp"], 3), num(t["gap_bp_fine_grid"], 3),
                         signed(t["gap_bp_fine_grid"] - t["gap_bp"], 3), num(t["gap_bp_pde"], 3),
                         signed(t["gap_bp_pde"] - t["gap_bp"], 3)])
    mark_rows = []
    marks = [("base", d["c"].MARKS["base"], {tl: pr["tails"][tl] for tl in TAIL_ORDER})]
    marks += [(m, pr["marks"][m]["description"], pr["marks"][m]["tails"]) for m in ("mo_close", "all_settle")]
    for key, desc, tails in marks:
        for tail in TAIL_ORDER:
            t = tails[tail]
            curve_bp = t["curve"]["pv_bp"] if key == "base" else t["curve_bp"]
            flat_bp = t["flat"]["pv_bp"] if key == "base" else t["flat_bp"]
            mark_rows.append([f"<b>{esc(key)}</b> <span class='dim'>{esc(desc)}</span>", tail_name(d, tail),
                              pct(t["q_flat"], 3), num(curve_bp, 2), num(flat_bp, 2), signed(t["gap_bp"], 2)])
    eng = pr["engine"]
    gaps = [pr["tails"][t]["gap_bp"] for t in TAIL_ORDER]
    return section(3, "PRICE", "The misprice", f"""
<p class="lead">Coupon fixed at {pct(h["coupon"], 2)} p.a. Under the default hold-forward tail the flat q(T_mat) =
{pct(h["q_flat"], 2)} values the note {snum(h["gap_bp"], 2)} bp of notional {"above" if h["gap_bp"] > 0 else "below"}
the curve; a flat-q desk would solve a coupon {num(100 * abs(h["coupon_equiv"]), 2)}% p.a.
{"lower" if h["gap_bp"] > 0 else "higher"}. The gap has the same sign under every tail
({", ".join(snum(g, 1) for g in gaps)} bp) but its size depends on the tail.</p>
{table(["tail", "curve PV, bp", "flat PV, bp", "gap flat − curve, bp", "flat-desk coupon, % p.a.", "q_flat"], rows,
       note="PV of the note holder in bp of notional. Flat-desk coupon: the coupon change that zeroes the gap, "
            "gap / ∂PV_curve/∂coupon (that tail's own annuity).")}
<h3 class="sub">Validation: QUAD grid doubling and PDE</h3>
{table(["tail", "model", f"QUAD {eng['quad_grid_resolved']}", f"QUAD {eng['quad_grid_fine_resolved']}", "Δ grid",
        f"PDE {eng['pde_accuracy']}", "Δ PDE"], val_rows,
       note=f"bp of notional. QUAD readout '{esc(eng['quad_readout'])}', resolved grid {eng['quad_grid_resolved']} "
            f"doubled to {eng['quad_grid_fine_resolved']}; PDE accuracy '{esc(eng['pde_accuracy'])}'.")}
<h3 class="sub">Mark sensitivity (coupon held)</h3>
{table(["mark", "tail", "q(T_mat)", "curve PV, bp", "flat PV, bp", "gap, bp"], mark_rows,
       note="The MO pillar is levered into F(T_mat) under hold-forward, so its mark matters. No fully consistent mark "
            "exists (the index has no settlement price): the all-settlement chain mixes settlements with the spot close.")}
""")


EVENT_KEYS = (("p_ko_total", "P(KO)"), ("p_ko_first_year", "P(KO ≤ 1Y)"), ("p_ki_ever", "P(KI ever)"),
              ("p_ki_survive", "P(KI & survive)"))


def section_events(d: Dict[str, Any]) -> str:
    pr = d["pr"]
    rows = []
    for tail in TAIL_ORDER:
        t = pr["tails"][tail]
        cur, fl = t["curve"]["events"], t["flat"]["events"]
        for model, ev in (("curve", cur), ("flat", fl)):
            rows.append([tail_head(d, tail), model] + [pct(ev[k], 2) for k, _ in EVENT_KEYS]
                        + [num(ev["expected_life_years"], 3)])
        rows.append([tail_head(d, tail), "<b>curve − flat</b>"]
                    + [signed(100 * (cur[k] - fl[k]), 2, " pp") for k, _ in EVENT_KEYS]
                    + [signed(cur["expected_life_years"] - fl["expected_life_years"], 3, "Y")])
    hf = pr["tails"]["hold_forward"]
    dates = pr["terms"]["ko_dates"]
    ev_c, ev_f = hf["curve"]["events"], hf["flat"]["events"]
    times = ev_c["ko_times"]
    series = []
    for model, ev, col, dash in (("curve", ev_c, SERIES["hold_forward"], ""), ("flat", ev_f, "var(--ink-2)", "6 4")):
        ys = [100 * p for p in ev["ko_probability_by_obs"]]
        pts = [(x, y, f"{model} · {dt} · {num(y, 2)}%") for x, y, dt in zip(times, ys, dates)]
        series.append({"name": f"{model} · hold forward", "xs": times, "ys": ys, "colour": col, "dash": dash,
                       "end_label": model, "points": pts})
    chart = xy_chart(series, xdomain=(0.0, CURVE_X_MAX), xticks=[0, 0.5, 1.0, 1.5, 2.0], xfmt=lambda v: f"{v:g}",
                     yfmt=lambda v: num(v, 0), ylabel="P(KO AT THIS DATE), %", xlabel="KO OBSERVATION, YEARS",
                     label="KO probability by observation, hold-forward tail", zero_anchor=True, height=300)
    obs_rows = [[dt, num(t, 3), pct(a, 2), pct(b, 2), signed(100 * (a - b), 2, " pp")]
                for dt, t, a, b in zip(dates, times, ev_c["ko_probability_by_obs"], ev_f["ko_probability_by_obs"])]
    return section(4, "EVENTS", "Event statistics", f"""
<p class="lead">Same coupon, same vol: only the forward path differs. The table reads the QUAD event statistics of every
cell; the chart breaks the default tail's P(KO) down by observation.</p>
{table(["tail", "model"] + [lab for _, lab in EVENT_KEYS] + ["E[life], Y"], rows,
       note="P(KI & survive): knocked in and never knocked out (the loss scenario). E[life]: expected KO time, maturity "
            "if no KO.")}
{figure(chart, "P(KO) by observation date · hold-forward tail", "Curve vs flat q(T_mat); the first date carries the lockout mass.")}
{table(["KO date", "T (Y)", "curve", "flat", "curve − flat"], obs_rows)}
""")


def _unit(d: Dict[str, Any], label: str) -> str:
    if label.startswith("IM"):
        return f'<span class="wrap">IM lots (×{num(d["c"].IM_MULTIPLIER, 0)} CNY/pt)</span>'
    return (f'<span class="wrap">{esc(short_label(label))} C−P pairs (×{num(d["c"].MO_MULTIPLIER, 0)}·DF CNY/pt) — '
            "not hedgeable with IM futures</span>")


def section_hedge(d: Dict[str, Any]) -> str:
    c, gk = d["c"], d["gk"]
    notional, spot = gk["notional"], gk["spot"]
    greek_rows, rho_rows, pillar_tables, resid_items = [], [], [], []
    for tail in TAIL_ORDER:
        g = gk["tails"][tail]
        cur, fl = g["curve"], g["flat"]
        sum_fwd = sum(b["forward"] / spot * b["delta"] for b in cur["buckets"])
        greek_rows.append([tail_head(d, tail), num(cur["spot_delta"], 1), num(fl["spot_delta"], 1),
                           signed(100 * (cur["spot_delta"] / fl["spot_delta"] - 1), 1, "%"),
                           num(g["curve_spot_lots"], 2), num(fl["spot_lots"], 2), num(cur["pinned_forward_delta"], 1),
                           num(sum_fwd, 1)])
        tail_rho = g["forward_interval_rho"][-1]["rho_q"]
        pillar_sum = sum(b["rho_q"] for b in cur["buckets"])
        rho_rows.append([tail_head(d, tail), num(c.bp(fl["parallel_rho_q"]), 2), num(c.bp(cur["parallel_rho_q"]), 2),
                         signed(100 * (cur["parallel_rho_q"] / fl["parallel_rho_q"] - 1), 1, "%"),
                         num(c.bp(pillar_sum), 2), num(c.bp(tail_rho), 2), pct(tail_rho / cur["parallel_rho_q"], 1)])
        rows = []
        for b in cur["buckets"]:
            rows.append([esc(b["label"]), num(b["tenor"], 4), num(b["forward"], 2), num(b["delta"], 2),
                         num(b["hedge_lots"], 3), _unit(d, b["label"]), num(c.bp(b["rho_q"]), 3),
                         "yes" if b["extrapolated_tail"] else "–"])
        pillar_tables.append(f'<h3 class="sub">Pillar hedge · {tail_head(d, tail)}</h3>'
                             + table(["pillar", "T (Y)", "F", "∂PV/∂F, CNY/pt", "hedge", "unit", "rho_q, bp/+1%",
                                      "read by tail"], rows))
        resid_items.append(
            f"<li><b>{tail_head(d, tail)}.</b> delta identity residual {snum(cur['delta_identity_residual'], 3)} CNY/pt "
            f"({pct(abs(cur['delta_identity_residual'] / cur['spot_delta']), 3)} of spot delta); rho-sum residual "
            f"{snum(cur['rho_sum_residual'], 2)} CNY "
            f"({pct(abs(cur['rho_sum_residual'] / cur['parallel_rho_q']), 2)} of parallel rho)"
            + (" — equal to the tail interval by construction: a flat-B tail cannot carry a parallel shift."
               if tail == "zero_carry" else ".") + "</li>")
    labels = [short_label(b["label"]) for b in gk["tails"]["hold_forward"]["curve"]["buckets"]]
    delta_series = [(tail_name(d, t), SERIES[t], [b["delta"] for b in gk["tails"][t]["curve"]["buckets"]])
                    for t in TAIL_ORDER]
    rho_series = [(tail_name(d, t), SERIES[t], [c.bp(b["rho_q"]) for b in gk["tails"][t]["curve"]["buckets"]])
                  for t in TAIL_ORDER]
    edges = ["0"] + [lab[2:] if lab.startswith("IM") else lab for lab in labels] + ["T_mat"]
    int_labels = [f"({edges[i]}, {edges[i + 1]}]" for i in range(len(edges) - 1)]
    int_series = [(tail_name(d, t), SERIES[t], [c.bp(x["rho_q"]) for x in gk["tails"][t]["forward_interval_rho"]])
                  for t in TAIL_ORDER]

    def cat_table(cats, ser, d_):
        return table(["tail"] + list(cats), [[esc(n)] + [num(v, d_) for v in vs] for n, _, vs in ser])

    delta_chart = grouped_bars(labels, delta_series, yfmt=lambda v: num(v, 0), tipfmt=lambda v: f"{snum(v, 1)} CNY/pt",
                               ylabel="∂PV/∂F_i, CNY PER POINT", xlabel="PILLAR (S FIXED)", label="Pillar delta")
    rho_chart = grouped_bars(labels, rho_series, yfmt=lambda v: num(v, 0), tipfmt=lambda v: f"{snum(v, 2)} bp",
                             ylabel="RHO_q PER PILLAR, BP PER +1% q", xlabel="PILLAR (NODE BUMP)", label="Pillar rho")
    int_chart = grouped_bars(int_labels, int_series, yfmt=lambda v: num(v, 0), tipfmt=lambda v: f"{snum(v, 2)} bp",
                             ylabel="RHO_q PER FORWARD INTERVAL, BP PER +1% q", xlabel="FORWARD-q INTERVAL",
                             label="Forward-interval rho", shade_last="unquoted")
    h = headline(d)
    hf = gk["tails"]["hold_forward"]["curve"]["buckets"]
    dd_hf, dr_hf = total_diffs(d, "hold_forward")
    pinned = {
        "hold_forward": "every F_i held; beyond T_n the tail is read off the last two nodes, so it is pinned too. Only "
                        "the short end [0, T_1] moves with S, through the (0, 0) anchor.",
        "hold_yield": "every F_i held, but beyond T_n the curve holds B(t)/t = B_n/T_n, so re-solving B_n at a new S "
                      "moves every forward past the last pillar: the pinned-forward delta is that tail's spot exposure.",
        "zero_carry": "every F_i held, and beyond T_n the forward stays at F_n, so the tail is pinned as well.",
    }
    pinned_items = "".join(
        f"<li><b>{tail_head(d, t)}</b>: {pinned[t]} Measured {snum(gk['tails'][t]['curve']['pinned_forward_delta'], 1)} "
        "CNY/pt.</li>" for t in TAIL_ORDER)
    return section(5, "HEDGE", "The static hedge", f"""
<p class="lead">Under hold-forward the totals are close (spot delta {num(dd_hf, 1)}%, parallel rho {num(dr_hf, 1)}%
apart); where the q risk sits is not. {pct(h["tail_share"], 0)} of the parallel rho lives beyond the last quote, and the pillar hedge is a calendar spread:
{snum(hf[3]["hedge_lots"], 2)} {esc(short_label(hf[3]["label"]))} lots against {snum(hf[4]["hedge_lots"], 2)}
{esc(short_label(hf[4]["label"]))} C−P pairs.</p>
{table(["tail", "Δ curve, CNY/pt", "Δ flat, CNY/pt", "curve vs flat", "curve, IM lots", "flat, IM lots",
        "pinned-F Δ", "Σ (F_i/S)·∂PV/∂F_i"], greek_rows,
       note=f"Spot delta ±{num(100 * gk['bumps']['spot_rel'], 0)}% of S: curve with the carry nodes B_i held "
            "(forwards move with spot), flat with q_flat held. IM-lot equivalent = −Δ/200. "
            "Positions as priced: long one note; a desk short the note trades the opposite sign.")}
{table(["tail", "rho flat, bp", "parallel rho curve, bp", "curve vs flat", "Σ pillar rho, bp", "tail-interval rho, bp",
        "tail share"], rho_rows,
       note=f"Rho_q per +1% q, central ±{num(1e4 * gk['bumps']['q'], 0)} bp. Tail interval: forward q bumped on "
            "(T_n, T_mat] only.")}
<h3 class="sub">Pinned-forward delta, per tail</h3>
<ul class="points">{pinned_items}</ul>
{"".join(pillar_tables)}
<p class="note">Hedge = −(∂PV/∂F_i)/m_i: IM m = 200 CNY/pt per lot; the MO pillar m = 100·DF(T_n) per C−P pair (one
synthetic forward, paid at expiry). Pillar delta at fixed S, ±{num(100 * gk["bumps"]["forward_rel"], 1)}% of F_i.</p>
{figure(delta_chart, "Pillar delta ∂PV/∂F_i", "Hold-forward turns the 2Y exposure into opposite IM2706 / MO2709 legs.")}
{cat_table(labels, delta_series, 2)}
{figure(rho_chart, "Pillar rho_q", "Node bump: q at pillar i +1%, other pillars held.")}
{cat_table(labels, rho_series, 2)}
{figure(int_chart, "Forward-interval rho_q", "The six intervals partition (0, T_mat]; the shaded one is unquoted.")}
{cat_table(int_labels, int_series, 2)}
<h3 class="sub">Identity residuals</h3>
<ul class="points">{"".join(resid_items)}</ul>
<p class="note">Delta identity: spot Δ − pinned-F Δ − Σ (F_i/S)·∂PV/∂F_i (exact chain rule, central-difference error
only). Rho sum: parallel rho − Σ pillar rho.</p>
""")


def section_scenarios(d: Dict[str, Any]) -> str:
    gk = d["gk"]
    blocks, worst = [], []
    for tail in TAIL_ORDER:
        sc = gk["tails"][tail]["scenarios"]
        rows = [[esc(s["scenario"].replace("-", "−")), signed(100 * s["dq_at_maturity"], 3, "%"),
                 signed(s["dpv_curve_bp"], 2), signed(s["dpv_flat_bp"], 2), signed(s["flat_greek_prediction_bp"], 2),
                 signed(s["missed_bp"], 2)] for s in sc]
        w = max(sc, key=lambda s: abs(s["missed_bp"]))
        worst.append(f"{tail_name(d, tail)}: {esc(w['scenario'])} ({snum(w['missed_bp'], 1)} bp)")
        blocks.append(f'<h3 class="sub">{tail_head(d, tail)}</h3>'
                      + table(["scenario", "Δq(T_mat)", "ΔPV curve", "ΔPV flat", "flat-Greek prediction", "missed"],
                              rows))
    return section(6, "SCEN", "Scenarios: what the flat model misses", f"""
<p class="lead">Each scenario shocks the curve; the curve model re-prices in full, the flat model re-reads q_flat =
q(T_mat) off the shocked curve and re-prices, and the flat Greek predicts rho_flat · Δq(T_mat)/1%. "Missed" =
ΔPV curve − ΔPV flat. Largest miss per tail: {"; ".join(worst)}.</p>
{"".join(blocks)}
<p class="note">bp of notional. A pillar shock that does not reach T_mat leaves the flat model blind (Δq(T_mat) = 0).</p>
""")


def findings(d: Dict[str, Any]) -> List[str]:
    """Plain-language conclusions, each computed from the results."""
    pr, gk, h = d["pr"], d["gk"], headline(d)
    hf = pr["tails"]["hold_forward"]
    nodes = hf["nodes"]
    ev_c, ev_f = hf["curve"]["events"], hf["flat"]["events"]
    out = []
    word = "higher" if h["gap_bp"] > 0 else "lower"
    out.append(
        f"<b>The flat q misprices the note.</b> Under the default hold-forward tail the flat q(T_mat) = "
        f"{pct(h['q_flat'], 2)} values the note {snum(h['gap_bp'], 2)} bp of notional {word} than the curve. A flat-q "
        f"desk would solve a coupon {num(100 * abs(h['coupon_equiv']), 2)}% p.a. "
        f"{'lower' if h['gap_bp'] > 0 else 'higher'} than {pct(h['coupon'], 2)}.")
    q_by = {short_label(n["label"]): n["q"] for n in nodes}
    peak = max(nodes, key=lambda n: n["q"])
    ko_year1 = [t for t in ev_c["ko_times"] if t <= 1.0]
    q_curve_y1 = [d["curves"]["hold_forward"]["q_at"](t) for t in ko_year1]
    above = sum(q > h["q_flat"] for q in q_curve_y1)
    out.append(
        f"<b>The curve puts q in the 2027 H1 dividend window.</b> Zero q rises from {pct(nodes[0]['q'], 1)} "
        f"(IM2611) to {pct(q_by['IM2703'], 1)} (IM2703) and {pct(q_by['IM2706'], 1)} (IM2706), peaks at "
        f"{esc(short_label(peak['label']))} and falls back to {pct(nodes[-1]['q'], 1)} at the last pillar. At "
        f"{above} of the {len(ko_year1)} first-year KO dates the curve's q is above the flat {pct(h['q_flat'], 2)}, so "
        f"its forwards are lower there: P(KO ≤ 1Y) {pct(ev_c['p_ko_first_year'], 1)} vs {pct(ev_f['p_ko_first_year'], 1)}, "
        f"P(KI) {pct(ev_c['p_ki_ever'], 1)} vs {pct(ev_f['p_ki_ever'], 1)}, expected life "
        f"{num(ev_c['expected_life_years'], 3)}Y vs {num(ev_f['expected_life_years'], 3)}Y (curve vs flat). "
        f"The note is worth {'less' if h['gap_bp'] > 0 else 'more'} on the curve.")
    dd = [100 * abs(gk["tails"][t]["curve"]["spot_delta"] / gk["tails"][t]["flat"]["spot_delta"] - 1) for t in TAIL_ORDER]
    dr = [100 * abs(gk["tails"][t]["curve"]["parallel_rho_q"] / gk["tails"][t]["flat"]["parallel_rho_q"] - 1)
          for t in TAIL_ORDER]
    shares = [gk["tails"][t]["forward_interval_rho"][-1]["rho_q"] / gk["tails"][t]["curve"]["parallel_rho_q"]
              for t in TAIL_ORDER]
    b = gk["tails"]["hold_forward"]["curve"]["buckets"]
    out.append(
        f"<b>The mis-hedge is where, not how much.</b> Spot delta differs from the flat model by "
        f"{num(min(dd), 1)}–{num(max(dd), 1)}% and parallel rho by {num(min(dr), 1)}–{num(max(dr), 1)}% across the tails "
        f"(the hold-forward pair: {num(dd[0], 1)}% and {num(dr[0], 1)}%). But {pct(shares[0], 0)} of the hold-forward "
        f"parallel rho ({pct(min(shares), 0)}–{pct(max(shares), 0)} across tails) lives beyond the last quote, and the "
        f"hold-forward pillar hedge is a calendar spread: {snum(b[3]['hedge_lots'], 2)} {esc(short_label(b[3]['label']))} "
        f"lots against {snum(b[4]['hedge_lots'], 2)} {esc(short_label(b[4]['label']))} C−P pairs, which IM futures "
        "alone cannot hold.")
    pvs = {t: pr["tails"][t]["curve"]["pv_bp"] for t in TAIL_ORDER}
    lo_t, hi_t = min(pvs, key=pvs.get), max(pvs, key=pvs.get)
    out.append(
        f"<b>The unquoted tail is the dominant model risk.</b> With the coupon held, the tail convention alone moves the "
        f"curve PV from {snum(pvs[lo_t], 1)} bp ({tail_name(d, lo_t)}) to {snum(pvs[hi_t], 1)} bp ({tail_name(d, hi_t)}): "
        f"a {num(h['pv_spread'], 0)} bp range, "
        f"{'larger' if h['pv_spread'] > h['max_abs_gap'] else 'smaller'} than the largest flat-vs-curve gap "
        f"({num(h['max_abs_gap'], 1)} bp).")
    out.append(
        f"<b>The marks matter through the lever.</b> Swapping the MO pair to its close, or the whole chain to CFFEX "
        f"settlements, moves the hold-forward gap between {num(h['mark_gap_min'], 1)} and {num(h['mark_gap_max'], 1)} bp "
        f"(base {num(h['gap_bp'], 1)} bp): the sign "
        f"{'holds' if h['mark_gap_min'] * h['mark_gap_max'] > 0 else 'does not hold'}, the size is uncertain by up to "
        f"{num(max(h['gap_bp'] - h['mark_gap_min'], h['mark_gap_max'] - h['gap_bp']), 1)} bp.")
    return out


def section_read(d: Dict[str, Any]) -> str:
    items = "".join(f"<li>{f}</li>" for f in findings(d))
    return section(7, "READ", "Findings", f"""
<p class="lead">What the numbers say, in order of what a desk would act on.</p>
<ul class="points">{items}</ul>
<h3 class="sub">Caveats</h3>
<ul class="points dim">
<li>One flat vol for every cell (no smile, no local or stochastic vol); r flat; vol is not the subject.</li>
<li>The coupon equivalent divides by each tail's own curve annuity.</li>
<li>Static Greeks only: the historical curve replay / hedging backtest is deferred.</li>
<li>Mark-inconsistency: IM closes and the MO settlement are mixed across the levered last segment (see 03).</li>
</ul>
""")


def section_method(d: Dict[str, Any]) -> str:
    snap, pr, gk = d["snap"], d["pr"], d["gk"]
    prov = [[esc(f["contract"]), str(f["revision_id"]), f'<code>{esc(f["content_hash"][:16])}…</code>']
            for f in snap["futures"]]
    syn = snap["dashboard_syn_fut"]
    prov.append([esc(syn["contract"]) + " (cross-check)", str(syn["revision_id"]),
                 f'<code>{esc(syn["content_hash"][:16])}…</code>'])
    csv = snap["cffex_csv"]
    eng, b = pr["engine"], gk["bumps"]
    study = "example/case_studies/snowball_2y_carry_curve"
    g_min = gk["runtime_seconds"] / 60.0
    return section(8, "METHOD", "Method and reproducibility", f"""
<dl class="spec">
<dt>Spot · IM</dt><dd>{esc(snap["sources"]["spot_and_im"])}, opened read-only</dd>
<dt>MO pair</dt><dd>{esc(snap["sources"]["mo_pair"])} · <code>{esc(csv["url"])}</code> · sha256
<code>{esc(csv["sha256"])}</code> · {csv["bytes"]:,} bytes (not committed; the snapshot freezes the values)</dd>
<dt>Engines</dt><dd>SnowballQuadEngine, readout '{esc(eng["quad_readout"])}', resolved grid {eng["quad_grid_resolved"]}
(doubling check {eng["quad_grid_fine_resolved"]}) for every price and Greek · PDE accuracy '{esc(eng["pde_accuracy"])}'
cross-check on the six PVs</dd>
<dt>Bumps</dt><dd>spot ±{num(100 * b["spot_rel"], 0)}% of S · pillar forward ±{num(100 * b["forward_rel"], 1)}% of F_i ·
q ±{num(1e4 * b["q"], 0)} bp (rho reported per {esc(b["rho_unit"])}) · scenarios {num(100 * b["scenario_shock"], 0)}%</dd>
<dt>Runtime</dt><dd>stage 02 {num(pr["runtime_seconds"], 0)} s · stage 03 {num(gk["runtime_seconds"], 0)} s
(≈{num(g_min, 0)} min): a QUAD re-price at a new spot rebuilds the engine's spot-keyed cache, ≈{num(QUAD_RESPOT_SECONDS, 0)} s
against ≈{num(QUAD_CACHED_SECONDS, 1)} s at the cached spot (engine-side; open item)</dd>
</dl>
<h3 class="sub">Dashboard rows</h3>
{table(["row", "revision id", "content hash"], prov, text=True)}
<h3 class="sub">Rerun</h3>
<pre>python {study}/01_snapshot.py   # live: dashboard DB + CFFEX CSV (--db, --cffex-csv)
python {study}/02_pricing.py    # ≈{num(pr["runtime_seconds"] / 60.0, 1)} min → results/pricing.json
python {study}/03_greeks.py     # ≈{num(g_min, 0)} min → results/greeks.json
python {study}/04_report.py     # → results/report.html</pre>
<p class="note">Stages 02–04 run offline from data/market_20261105.json. In a worktree prefix every command with
<code>env PYTHONPATH=&lt;worktree&gt;</code>.</p>
""")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

NAV = [(1, "SETUP"), (2, "CURVE"), (3, "PRICE"), (4, "EVENTS"), (5, "HEDGE"), (6, "SCEN"), (7, "READ"), (8, "METHOD")]


def masthead(d: Dict[str, Any]) -> Tuple[str, str]:
    snap, h = d["snap"], headline(d)
    dd_hf, dr_hf = total_diffs(d, "hold_forward")
    kpis = [
        (signed(h["gap_bp"], 1), "BP", "Flat − curve PV · hold-forward tail"),
        (signed(-100 * h["coupon_equiv"], 2), "% P.A.", "Coupon a flat-q desk would solve"),
        (pct(h["tail_share"], 0), "", "Parallel rho beyond the last quote"),
        (num(h["pv_spread"], 0), "BP", "Curve PV range across tail conventions"),
    ]
    tiles = "".join(f'<div class="kpi"><div class="kv">{v}<small>{u}</small></div><div class="kl">{esc(lab)}</div></div>'
                    for v, u, lab in kpis)
    nav = "".join(f'<a href="#s{n}"><b>{n})</b> {t}</a>' for n, t in NAV)
    chrome = f"""
<div class="top">
<header class="fbar">
  <span class="brand">QUANTARK</span><span class="cmd">SNBL CARRY &lt;GO&gt;</span>
  <span class="fld">STUDY <b>snowball_2y_carry_curve</b></span><span class="fld">UL <b>{esc(snap["index_code"])}</b></span>
  <span class="fld">ENGINES <b>QUAD · PDE</b></span><span class="fld date">{esc(snap["valuation_date"])}</span>
  <button class="theme" id="theme" type="button" aria-label="Switch colour theme">◐ <b>LIGHT</b></button>
</header>
<nav class="menu" id="menu">{nav}</nav>
</div>
"""
    hero = f"""
<div class="hero">
  <div class="kicker">SNOWBALL · CARRY CURVE · 2Y</div>
  <h1>A flat q misprices the 2Y snowball, and hides where its carry risk sits.</h1>
  <p class="answer">On the {esc(snap["valuation_date"])} CSI 1000 chain the flat q(T_mat) values the note
  <b>{snum(h["gap_bp"], 1)} bp</b> of notional above the curve (a coupon {num(100 * abs(h["coupon_equiv"]), 2)}% p.a.
  lower). Delta and rho totals differ by only {num(dd_hf, 1)}% and {num(dr_hf, 1)}%, but
  <b>{pct(h["tail_share"], 0)}</b> of the rho sits beyond the last quote, and the
  unquoted tail convention alone moves the PV by <b>{num(h["pv_spread"], 0)} bp</b>.</p>
</div>
<div class="kpis">{tiles}</div>
"""
    return chrome, hero


CSS = """
/* Dark (terminal) is the base; light is the Swiss paper variant.  The OS preference picks the
   default, the switcher stamps data-theme on <html> and wins both ways. */
:root{
  --bg:#000;--panel:#0b0b0b;--line:#262626;--hair:#1a1a1a;--row:#141414;
  --ink:#e8e6df;--ink-2:#a3a199;--ink-3:#6f6d66;
  --amber:#ffa028;--amber-dim:#b37018;
  --up:#3ecf6e;--dn:#ff5a4f;
  --s1:#d07a00;--s2:#2b95cf;--s3:#e0569b;
  color-scheme:dark}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){
  --bg:#f4f2ec;--panel:#ebe8e0;--line:#cbc7bb;--hair:#dedad0;--row:#e8e4da;
  --ink:#111110;--ink-2:#46443e;--ink-3:#77746b;
  --amber:#b35a00;--amber-dim:#9a5b10;
  --up:#13803a;--dn:#c42b20;
  --s1:#c46a00;--s2:#1f7fc0;--s3:#d0457f;
  color-scheme:light}}
:root[data-theme="light"]{
  --bg:#f4f2ec;--panel:#ebe8e0;--line:#cbc7bb;--hair:#dedad0;--row:#e8e4da;
  --ink:#111110;--ink-2:#46443e;--ink-3:#77746b;
  --amber:#b35a00;--amber-dim:#9a5b10;
  --up:#13803a;--dn:#c42b20;
  --s1:#c46a00;--s2:#1f7fc0;--s3:#d0457f;
  color-scheme:light}
:root{
  /* the frozen terminal strip stays dark in both themes */
  --bar:#000;--bar-2:#0d0d0d;--bar-line:#2a2a2a;--bar-ink:#e8e6df;--bar-ink-2:#a3a199;--bar-ink-3:#77756e;
  --bar-amber:#ffa028;
  --sans:"Helvetica Neue",Helvetica,Inter,Arial,sans-serif;
  --mono:"SF Mono","JetBrains Mono","IBM Plex Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box}
html{background:var(--bg);scroll-padding-top:84px}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
code{font:0.88em var(--mono);color:var(--amber);word-break:break-all}
em{font-style:normal;color:var(--amber)}
a{color:inherit;text-decoration:none}

/* Bloomberg function bar + numbered menu, frozen together */
.top{position: sticky;top:0;z-index:5;background:var(--bar)}
.fbar{display:flex;flex-wrap:wrap;align-items:stretch;gap:0;background:var(--bar);
  border-bottom:1px solid var(--bar-amber);font:12px/1 var(--mono);letter-spacing:.04em}
.fbar>span{padding:9px 12px;border-right:1px solid var(--bar-line);white-space:nowrap;display:flex;align-items:center}
.brand{color:#000;background:var(--bar-amber);font-weight:700}
.cmd{color:var(--bar-amber);font-weight:700}
.fld{color:var(--bar-ink-3)}.fld b{color:var(--bar-ink);font-weight:500;margin-left:6px}
.fld.date{margin-left:auto;color:var(--bar-ink-2)}
.theme{font:12px/1 var(--mono);letter-spacing:.06em;color:var(--bar-ink-2);background:transparent;border:0;
  padding:9px 14px;cursor:pointer}
.theme b{color:var(--bar-amber);font-weight:600}
.theme:hover{background:var(--bar-amber);color:#000}.theme:hover b{color:#000}
.theme:focus-visible,.menu a:focus-visible{outline:1px solid var(--bar-amber);outline-offset:-1px}
.menu{display:flex;flex-wrap:wrap;gap:0;border-bottom:1px solid var(--bar-line);font:11px/1 var(--mono);
  letter-spacing:.06em;background:var(--bar-2)}
.menu a{padding:9px 14px;color:var(--bar-ink-2);border-right:1px solid var(--bar-line)}
.menu a b{color:var(--bar-amber);font-weight:500}
.menu a:hover,.menu a.on{background:var(--bar-amber);color:#000}.menu a:hover b,.menu a.on b{color:#000}

/* Swiss grid */
main{max-width:1180px;margin:0 auto;padding:0 32px 96px}
.hero{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));column-gap:24px;padding:72px 0 40px;border-bottom:1px solid var(--line)}
.kicker{grid-column:1/4;font:11px/1.4 var(--mono);letter-spacing:.14em;color:var(--amber);padding-top:12px}
.hero h1{grid-column:4/13;margin:0;font:700 clamp(32px,4.8vw,60px)/1.04 var(--sans);letter-spacing:-.025em}
.answer{grid-column:4/12;margin:28px 0 0;font-size:18px;line-height:1.5;color:var(--ink-2)}
.answer b{color:var(--ink);font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border-bottom:1px solid var(--line)}
.kpi{padding:22px 20px 22px 0;border-right:1px solid var(--line)}
.kpi+.kpi{padding-left:20px}.kpi:last-child{border-right:0}
.kv{font:600 clamp(26px,3.4vw,40px)/1 var(--mono);letter-spacing:-.02em;color:var(--ink)}
.kv small{font-size:12px;margin-left:6px;color:var(--ink-3);letter-spacing:.08em}
.kl{margin-top:10px;font:11px/1.4 var(--mono);letter-spacing:.06em;color:var(--ink-2);text-transform:uppercase}

section{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));column-gap:24px;padding:56px 0;border-bottom:1px solid var(--line)}
.rail{grid-column:1/4}
.numeral{font:700 88px/0.85 var(--sans);letter-spacing:-.05em;color:var(--amber)}
.code{margin-top:18px;font:11px/1 var(--mono);letter-spacing:.16em;color:var(--ink-3)}
.rail h2{margin:8px 0 0;font:700 22px/1.2 var(--sans);letter-spacing:-.01em}
.body{grid-column:4/13;min-width:0}
.lead{margin:0 0 28px;font-size:17px;max-width:68ch}
.body p{max-width:72ch}
h3.sub{margin:36px 0 8px;font:11px/1.3 var(--mono);letter-spacing:.16em;color:var(--ink-3);text-transform:uppercase}
pre{font:12px/1.6 var(--mono);color:var(--ink);border-top:1px solid var(--line);border-bottom:1px solid var(--line);
  padding:12px 0;margin:8px 0;overflow-x:auto}

dl.spec{display:grid;grid-template-columns:160px minmax(0,1fr);margin:0 0 24px;border-top:1px solid var(--line)}
dl.spec dt,dl.spec dd{margin:0;padding:10px 0;border-bottom:1px solid var(--hair)}
dl.spec dt{font:11px/1.6 var(--mono);letter-spacing:.1em;color:var(--amber);text-transform:uppercase}
.dates{font:11.5px/1.6 var(--mono);color:var(--ink-2)}

/* tables: terminal density */
.scroll{overflow-x:auto;margin:8px 0}
table{border-collapse:collapse;width:100%;font:12.5px/1.3 var(--mono);font-variant-numeric:tabular-nums}
th{padding:8px 12px 8px 0;text-align:right;font-weight:500;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--amber-dim);border-bottom:1px solid var(--amber-dim);white-space:nowrap}
td{padding:6px 12px 6px 0;text-align:right;border-bottom:1px solid var(--hair);white-space:nowrap}
th:first-child,td:first-child,table.text th,table.text td{text-align:left}
tbody tr:hover td{background:var(--row)}
td code{color:var(--ink)}
td .wrap{display:inline-block;white-space:normal;min-width:24ch;max-width:30ch;text-align:left}
.up{color:var(--up)}.dn{color:var(--dn)}.flat{color:var(--ink-2)}.dim{color:var(--ink-3)}
.note{margin:6px 0 0;font:11px/1.5 var(--mono);color:var(--ink-3);max-width:none}

/* figures and charts */
figure{margin:28px 0 0;border-top:1px solid var(--line);padding-top:12px}
figcaption{display:flex;flex-wrap:wrap;gap:4px 16px;align-items:baseline;margin-bottom:10px}
figcaption b{font:600 13px/1.3 var(--sans)}
figcaption span{font:11px/1.3 var(--mono);color:var(--ink-3)}
.legend{display:flex;flex-wrap:wrap;gap:6px 20px;font:11px/1 var(--mono);color:var(--ink-2);margin:2px 0 6px}
.legend i{display:inline-block;width:16px;height:0;margin-right:7px;vertical-align:middle}
svg.chart{width:100%;height:auto;display:block;overflow:visible}
.chart .grid{stroke:var(--hair);stroke-width:1}
.chart .axis{stroke:var(--line);stroke-width:1}
.chart .zero{stroke:var(--ink-3);stroke-width:1}
.chart .pgrid{stroke:var(--line);stroke-width:1;stroke-dasharray:2 3}
.chart .pillar{stroke:var(--ink-2);stroke-width:1.5}
.chart .vline{stroke:var(--ink-3);stroke-width:1;stroke-dasharray:4 3}
.chart .shade{fill:var(--amber);fill-opacity:.08}
.chart .annot{font:10px var(--mono);fill:var(--ink-2);letter-spacing:.04em}
.chart .plab{font:9.5px var(--mono);fill:var(--ink-3)}
.chart .tick{font:10.5px var(--mono);fill:var(--ink-3)}
.chart .axlab{font:9.5px var(--mono);letter-spacing:.1em;fill:var(--ink-3)}
.chart .ln{fill:none;stroke-width:2;stroke-linejoin:round}
.chart .mk{stroke:var(--bg);stroke-width:2;pointer-events:none}
.chart .hit{fill:transparent;cursor:crosshair}
.chart .bar{cursor:crosshair}.chart .bar:hover{stroke:var(--ink);stroke-width:1.5}
.tip{position:fixed;z-index:10;pointer-events:none;padding:6px 9px;background:var(--panel);border:1px solid var(--amber);
  color:var(--ink);font:11.5px/1.3 var(--mono);white-space:nowrap;display:none}
.chart .dlab{font:11px var(--mono);fill:var(--ink-2)}

ul.points{padding:0;list-style:none;margin:20px 0 0;max-width:76ch}
ul.points li{padding:10px 0 10px 18px;border-top:1px solid var(--hair);position:relative}
ul.points li::before{content:"";position:absolute;left:0;top:19px;width:8px;height:2px;background:var(--amber)}
ul.points li b{color:var(--ink);font-weight:600}
ul.points.dim li{color:var(--ink-2)}
footer{padding:24px 0;font:11px/1.5 var(--mono);color:var(--ink-3);letter-spacing:.04em}

@media (max-width:820px){
  main{padding:0 16px 64px}
  .hero,section{grid-template-columns:minmax(0,1fr)}
  .kicker,.hero h1,.answer,.rail,.body{grid-column:1/-1}
  .rail{margin-bottom:20px}.numeral{font-size:56px}
  .kpis{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.kpi:nth-child(2){border-right:0}
  .kpi:nth-child(3){padding-left:0}.kpi:nth-child(n+3){border-top:1px solid var(--line)}
  dl.spec{grid-template-columns:minmax(0,1fr)}dl.spec dt{border-bottom:0;padding-bottom:0}
  .fld{display:none}
  .menu{flex-wrap:nowrap;overflow-x:auto;scrollbar-width:none}.menu a{white-space:nowrap}
}
"""

# Runs in <head> so a stored theme applies before first paint (no flash).
THEME_INIT_JS = """<script>
try{const t=localStorage.getItem('qa-theme');if(t==='light'||t==='dark')document.documentElement.dataset.theme=t}catch(e){}
</script>"""

PAGE_JS = """<div class="tip" id="tip"></div><script>
(()=>{
const root=document.documentElement,btn=document.getElementById('theme'),os=matchMedia('(prefers-color-scheme: light)');
const current=()=>root.dataset.theme||(os.matches?'light':'dark');
const label=()=>{btn.querySelector('b').textContent=current()==='dark'?'LIGHT':'DARK'};
btn.addEventListener('click',()=>{const next=current()==='dark'?'light':'dark';root.dataset.theme=next;
  try{localStorage.setItem('qa-theme',next)}catch(e){}label()});
os.addEventListener('change',label);label();

const t=document.getElementById('tip');
document.addEventListener('pointermove',e=>{const s=e.target.closest&&e.target.closest('[data-tip]');
  if(!s){t.style.display='none';return}t.textContent=s.dataset.tip;t.style.display='block';
  const w=t.offsetWidth;t.style.left=Math.min(e.clientX+14,innerWidth-w-8)+'px';t.style.top=(e.clientY+14)+'px'});

const links=[...document.querySelectorAll('#menu a')];
const io=new IntersectionObserver(es=>es.forEach(en=>{if(!en.isIntersecting)return;
  links.forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+en.target.id))}),
  {rootMargin:'-90px 0px -70% 0px'});
document.querySelectorAll('section[id]').forEach(s=>io.observe(s));
})();
</script>"""


def render(d: Dict[str, Any]) -> str:
    chrome, hero = masthead(d)
    content = (
        hero + section_setup(d) + section_curve(d) + section_price(d) + section_events(d) + section_hedge(d)
        + section_scenarios(d) + section_read(d) + section_method(d)
        + "<footer>QUANTARK · example/case_studies/snowball_2y_carry_curve · data/market_20261105.json → "
          "02_pricing → 03_greeks → 04_report · exchange closes and settlements only, no desk data</footer>"
    )
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>2Y snowball · flat q vs CarryCurve</title>{THEME_INIT_JS}<style>{CSS}</style></head>"
            f"<body>{chrome}<main>{content}</main>{PAGE_JS}</body></html>\n")


def headline_markdown(d: Dict[str, Any]) -> str:
    """The README's headline table, printed by ``main`` so the README copies the results."""
    pr, gk = d["pr"], d["gk"]
    c = d["c"]
    lines = ["| tail | curve PV | flat PV | gap | flat-desk coupon | q_flat | Δ curve / flat (CNY/pt) "
             "| parallel rho curve / flat (bp per +1% q) | rho beyond last quote |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for tail in TAIL_ORDER:
        t, g = pr["tails"][tail], gk["tails"][tail]
        share = g["forward_interval_rho"][-1]["rho_q"] / g["curve"]["parallel_rho_q"]
        lines.append(
            f"| {tail_name(d, tail)} | {num(t['curve']['pv_bp'], 2)} bp | {num(t['flat']['pv_bp'], 2)} bp | "
            f"{snum(t['gap_bp'], 2)} bp | {snum(-100 * t['coupon_equiv_gap'], 2)}% p.a. | {pct(t['q_flat'], 2)} | "
            f"{num(g['curve']['spot_delta'], 1)} / {num(g['flat']['spot_delta'], 1)} | "
            f"{num(c.bp(g['curve']['parallel_rho_q']), 2)} / {num(c.bp(g['flat']['parallel_rho_q']), 2)} | "
            f"{pct(share, 1)} |")
    return "\n".join(lines)


def build_report(out_path: Optional[Path] = None) -> Path:
    """Render the report from the snapshot and the two results files; returns the path written."""
    d = load_inputs()
    out = Path(out_path) if out_path is not None else d["c"].RESULTS_DIR / REPORT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(d), encoding="utf-8")
    return out


def main() -> int:
    out = build_report()
    print(f"wrote {out}")
    print(headline_markdown(load_inputs()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
