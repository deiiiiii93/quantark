"""Curve construction demo: bootstrapped discount / projection / carry curves.

Docs: docs/curve-construction.md
Spec: .ship/2026-11-05-curve-construction/spec.md

Numeric mode throughout (times in years on the ACT/365F axis, no dates).

Sections:
  A  the quote set: deposits 1M/3M/6M + quarterly SHIBOR_3M swaps 1Y..10Y
  B  the discount curve bootstrapped under all FIVE interpolation schemes:
     pillar zero rate, DF, instantaneous forward and the max |reprice| residual
  C  a dense grid (0.01..12Y) of zero rates and instantaneous forwards per
     scheme (the chart data; schemes agree at pillars, differ in between)
  D  dual curve: a SHIBOR_3M projection curve bootstrapped off a discount curve
     shifted by -50 bp, with the basis per pillar
  E  the carry curve from index futures, the implied dividend-yield view at the
     pillars and under the three carry extrapolations beyond the last node

Also writes a self-contained Bloomberg x Swiss HTML report (frozen navigator,
light/dark switcher, inline SVG charts with a table under each chart) to
example/output/curve_construction_report.html (git-ignored).

Run:
  python example/param/curve_construction_demo.py [--output PATH]
"""
from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from quantark.param import (
    BootstrapConfig,
    CurveInterpolation,
    DepositQuote,
    FuturesQuote,
    SwapQuote,
    bootstrap_discount_curve,
    bootstrap_projection_curve,
    build_carry_curve,
    reprice_quotes,
)
from quantark.param.extrapolation import CarryExtrapolation
from quantark.param.index import SHIBOR_3M

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "example" / "output" / "curve_construction_report.html"

MINUS = "−"
SCHEMES = list(CurveInterpolation)
SCHEME_NAMES = [s.name for s in SCHEMES]
GRID = np.linspace(0.01, 12.0, 300)
TABLE_GRID = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0]
CARRY_GRID = np.linspace(0.05, 4.0, 160)
CARRY_TABLE_GRID = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
BASIS_SHIFT = -0.005          # discount curve for the dual-curve study: -50 bp
SPOT = 4000.0
FLAT_Q = 0.012                # dividend yield the futures prices are built from
FUTURES_TENORS = [(0.25, "F3M"), (0.5, "F6M"), (0.75, "F9M"), (1.0, "F1Y"), (1.5, "F18M"), (2.0, "F2Y")]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_quotes() -> list:
    deposits = [
        DepositQuote(rate=0.0180, tenor_years=1 / 12, label="1M"),
        DepositQuote(rate=0.0190, tenor_years=0.25, label="3M"),
        DepositQuote(rate=0.0205, tenor_years=0.5, label="6M"),
    ]
    swap_rates = [(1.0, 0.0220), (2.0, 0.0245), (3.0, 0.0262), (5.0, 0.0285), (7.0, 0.0298), (10.0, 0.0306)]
    swaps = [SwapQuote(fixed_rate=r, index=SHIBOR_3M, tenor_years=t, label=f"{t:g}Y") for t, r in swap_rates]
    return deposits + swaps


def quote_row(q) -> tuple:
    if isinstance(q, DepositQuote):
        return ("Deposit", q.label, q.tenor_years, q.rate)
    return ("Swap " + q.index.name, q.label, q.tenor_years, q.fixed_rate)


def compute() -> dict:
    """Everything both the stdout tables and the HTML report show."""
    quotes = build_quotes()
    d: dict = {"quotes": quotes}

    # B/C: discount curve under every scheme
    curves, schemes = {}, {}
    for scheme in SCHEMES:
        curve = bootstrap_discount_curve(quotes, BootstrapConfig(interpolation=scheme))
        curves[scheme.name] = curve
        resid = reprice_quotes(quotes, curve)
        T = np.asarray(curve.tenors, dtype=float)
        schemes[scheme.name] = {
            "T": T,
            "zero": np.asarray(curve.get_rate(T), dtype=float),
            "df": np.asarray(curve.get_discount_factor(T), dtype=float),
            "fwd": np.asarray(curve.instantaneous_forward(T), dtype=float),
            "residuals": resid,
            "max_resid": max(abs(r) for r in resid),
            "grid_zero": np.asarray(curve.get_rate(GRID), dtype=float),
            "grid_fwd": np.asarray(curve.instantaneous_forward(GRID), dtype=float),
            "table_zero": np.asarray(curve.get_rate(np.array(TABLE_GRID)), dtype=float),
            "table_fwd": np.asarray(curve.instantaneous_forward(np.array(TABLE_GRID)), dtype=float),
        }
    d["curves"], d["schemes"] = curves, schemes
    base = curves["LOG_LINEAR_DF"]
    d["base"] = base

    # D: dual curve
    shifted = base.parallel_shifted(BASIS_SHIFT)
    projection = bootstrap_projection_curve(quotes, shifted)
    resid_p = reprice_quotes(quotes, shifted, projection_curve=projection)
    Tp = np.asarray(projection.tenors, dtype=float)
    d["dual"] = {
        "T": Tp,
        "disc_zero": np.asarray(shifted.get_rate(Tp), dtype=float),
        "proj_zero": np.asarray(projection.get_rate(Tp), dtype=float),
        "disc_fwd": np.asarray(shifted.instantaneous_forward(Tp), dtype=float),
        "proj_fwd": np.asarray(projection.instantaneous_forward(Tp), dtype=float),
        "residuals": resid_p,
        "max_resid": max(abs(r) for r in resid_p),
        "grid_zero_basis": np.asarray(projection.get_rate(GRID), dtype=float) - np.asarray(shifted.get_rate(GRID), dtype=float),
        "grid_fwd_basis": np.asarray(projection.instantaneous_forward(GRID), dtype=float)
        - np.asarray(shifted.instantaneous_forward(GRID), dtype=float),
    }

    # E: carry curve from index futures priced at a flat q on the bootstrapped curve
    fut_quotes = []
    for T, label in FUTURES_TENORS:
        price = SPOT * math.exp((float(base.get_rate(T)) - FLAT_Q) * T)
        fut_quotes.append(FuturesQuote(price=price, tenor_years=T, label=label))
    carry = {}
    for ext in CarryExtrapolation:
        curve = build_carry_curve(SPOT, fut_quotes, base, extrapolation=ext)
        view = curve.to_dividend_yield(base)
        Tn = np.asarray(curve.tenors, dtype=float)
        resid_c = [SPOT * math.exp(float(curve.carry(q.tenor_years))) - q.price for q in fut_quotes]
        carry[ext.name] = {
            "curve": curve,
            "T": Tn,
            "carry": np.asarray(curve.carries, dtype=float),
            "yield": np.asarray(view.get_yield(Tn), dtype=float),
            "residuals": resid_c,
            "max_resid": max(abs(r) for r in resid_c),
            "grid_yield": np.asarray(view.get_yield(CARRY_GRID), dtype=float),
            "table_yield": np.asarray(view.get_yield(np.array(CARRY_TABLE_GRID)), dtype=float),
            "table_fwd_carry": np.asarray(curve.forward_carry(np.array(CARRY_TABLE_GRID)), dtype=float),
        }
    d["futures"] = fut_quotes
    d["carry"] = carry
    d["grid_rate"] = np.asarray(base.get_rate(CARRY_GRID), dtype=float)
    d["table_rate"] = np.asarray(base.get_rate(np.array(CARRY_TABLE_GRID)), dtype=float)
    return d


# ---------------------------------------------------------------------------
# Number formatting: one formatter, true minus, never a negative zero
# ---------------------------------------------------------------------------


def _m(text: str) -> str:
    return text.replace("-", MINUS)


def num(v: float, d: int = 4) -> str:
    v = round(float(v), d) + 0.0          # no "-0.000"
    return _m(f"{v:,.{d}f}")


def pct(v: float, d: int = 3) -> str:
    return num(float(v) * 100.0, d) + "%"


def bp(v: float, d: int = 2) -> str:
    return num(float(v) * 1e4, d)


def sci(v: float) -> str:
    return _m(f"{float(v):.1e}")


# ---------------------------------------------------------------------------
# Stdout tables
# ---------------------------------------------------------------------------


def _print_stdout(d: dict, say: Callable[[str], None]) -> None:
    def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
        w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        say("  " + "  ".join(h.rjust(w[i]) for i, h in enumerate(headers)))
        for r in rows:
            say("  " + "  ".join(c.rjust(w[i]) for i, c in enumerate(r)))

    def plain(s: str) -> str:
        return s

    say("A. Quote set (numeric mode, ACT/365F years)")
    table(["kind", "label", "tenor_y", "rate"],
          [[k, lab, f"{t:.4f}", f"{r * 100:.3f}%"] for k, lab, t, r in map(quote_row, d["quotes"])])

    say("\nB. Discount curve bootstrapped under every scheme")
    for name in SCHEME_NAMES:
        s = d["schemes"][name]
        say(f"\n  {name}   max |reprice residual| = {s['max_resid']:.2e}")
        table(["T", "zero %", "DF", "inst. fwd %", "residual"],
              [[f"{t:.4f}", f"{z * 100:.4f}", f"{df:.6f}", f"{f * 100:.4f}", f"{r:.1e}"]
               for t, z, df, f, r in zip(s["T"], s["zero"], s["df"], s["fwd"], s["residuals"])])

    say("\nC. Dense-grid curves (grid 0.01..12Y, selected points): zero % / inst. forward %")
    table(["T"] + SCHEME_NAMES,
          [[f"{t:g}"] + [f"{d['schemes'][n]['table_zero'][i] * 100:.3f}/{d['schemes'][n]['table_fwd'][i] * 100:.3f}"
                         for n in SCHEME_NAMES] for i, t in enumerate(TABLE_GRID)])

    dual = d["dual"]
    say(f"\nD. Dual curve: SHIBOR_3M projection off the discount curve shifted by {BASIS_SHIFT * 1e4:.0f} bp "
        f"(max |residual| {dual['max_resid']:.1e})")
    table(["T", "disc zero %", "proj zero %", "basis bp", "disc fwd %", "proj fwd %", "fwd basis bp", "residual"],
          [[f"{t:.4f}", f"{dz * 100:.4f}", f"{pz * 100:.4f}", f"{(pz - dz) * 1e4:.2f}", f"{df * 100:.4f}",
            f"{pf * 100:.4f}", f"{(pf - df) * 1e4:.2f}", f"{r:.1e}"]
           for t, dz, pz, df, pf, r in zip(dual["T"], dual["disc_zero"], dual["proj_zero"], dual["disc_fwd"],
                                           dual["proj_fwd"], dual["residuals"])])

    say(f"\nE. Carry curve: spot {SPOT:g}, futures built from flat q = {FLAT_Q * 100:.2f}% on the LOG_LINEAR_DF discount curve")
    for ext, c in d["carry"].items():
        say(f"\n  extrapolation {ext}   max |futures residual| = {c['max_resid']:.1e}")
        table(["T", "futures", "carry B", "implied q %", "residual"],
              [[f"{t:.2f}", f"{q.price:.4f}", f"{b:.6f}", f"{y * 100:.5f}", f"{r:.1e}"]
               for t, q, b, y, r in zip(c["T"], d["futures"], c["carry"], c["yield"], c["residuals"])])
    say("\n  implied q % beyond the last node (T=2Y):")
    table(["T", "r % (zero)"] + list(d["carry"]),
          [[f"{t:g}", f"{d['table_rate'][i] * 100:.4f}"] + [f"{c['table_yield'][i] * 100:.4f}" for c in d["carry"].values()]
           for i, t in enumerate(CARRY_TABLE_GRID)])


# ---------------------------------------------------------------------------
# SVG charts (numeric x axis, any number of series)
# ---------------------------------------------------------------------------

# (colour var, svg dash array, css border style for the legend swatch)
SERIES = [
    ("var(--s1)", "", "solid"),
    ("var(--s2)", "7 4", "dashed"),
    ("var(--s3)", "2 3", "dotted"),
    ("var(--s4)", "10 3 2 3", "dashed"),
    ("var(--s5)", "", "double"),
]


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


def line_chart(xs: Sequence[float], series: Sequence[tuple], *, xlabel: str, ylabel: str,
               yfmt: Callable[[float], str], label: str, markers: Optional[Sequence[float]] = None,
               width: int = 760, height: int = 300) -> str:
    """``series``: [(name, ys)] drawn over the shared ``xs``; ``markers``: x positions of pillars."""
    left, right, top, bottom = 62, 24, 16, 42
    ys_all = [v for _, ys in series for v in ys]
    lo, hi = min(ys_all), max(ys_all)
    pad = (hi - lo) * 0.06 or 1e-4
    yt = _nice_ticks(lo - pad, hi + pad)
    ylo, yhi = min(yt[0], lo - pad), max(yt[-1], hi + pad)
    xlo, xhi = float(xs[0]), float(xs[-1])
    xt = _nice_ticks(xlo, xhi, 6)
    xt = [t for t in xt if xlo - 1e-9 <= t <= xhi + 1e-9]

    def X(v: float) -> float:
        return left + (v - xlo) / (xhi - xlo) * (width - left - right)

    def Y(v: float) -> float:
        return top + (yhi - v) / (yhi - ylo) * (height - top - bottom)

    base = height - bottom
    body: List[str] = []
    for v in yt:
        y = Y(v)
        body.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
                    f'<text class="tick" x="{left - 8}" y="{y + 3.5:.1f}" text-anchor="end">{yfmt(v)}</text>')
    if ylo < 0 < yhi:
        body.append(f'<line class="zero" x1="{left}" x2="{width - right}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"/>')
    body.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{base}" y2="{base}"/>')
    for v in xt:
        body.append(f'<text class="tick" x="{X(v):.1f}" y="{base + 16}" text-anchor="middle">{num(v, 1 if v % 1 else 0)}</text>')
    for m in (markers if markers is not None else []):
        body.append(f'<line class="pillar" x1="{X(m):.1f}" x2="{X(m):.1f}" y1="{base}" y2="{base + 5}"/>')
    body.append(f'<text class="axlab" x="{left}" y="{height - 4}">{html.escape(xlabel)}</text>')
    body.append(f'<text class="axlab" x="6" y="{top - 4}">{html.escape(ylabel)}</text>')
    for k, (name, ys) in enumerate(series):
        colour, dash, _ = SERIES[k]
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        body.append(f'<polyline class="ln" points="{pts}" style="stroke:{colour}"{da}/>')
    legend = "".join(
        f'<span><i style="border-top:3px {SERIES[k][2]} {SERIES[k][0]}"></i>{html.escape(name)}</span>'
        for k, (name, _) in enumerate(series))
    return (f'<div class="legend">{legend}</div>'
            f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)}">'
            + "".join(body) + "</svg>")


def table_html(headers: Sequence[str], rows: Sequence[Sequence[str]], note: str = "", text: bool = False) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    foot = f'<p class="note">{note}</p>' if note else ""
    cls = ' class="text"' if text else ""
    return f'<div class="scroll"><table{cls}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{foot}'


def figure(chart: str, title: str, caption: str, data_table: str) -> str:
    return (f'<figure><figcaption><b>{html.escape(title)}</b><span>{html.escape(caption)}</span></figcaption>'
            f"{chart}</figure>{data_table}")


def section(no: int, code: str, title: str, content: str) -> str:
    return (f'<section id="s{no}"><div class="rail"><div class="numeral">{no:02d}</div>'
            f'<div class="code">{code}</div><h2>{html.escape(title)}</h2></div><div class="body">{content}</div></section>')


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def sec_quotes(d: dict) -> str:
    rows = [[html.escape(k), html.escape(lab), num(t, 4), pct(r, 3)] for k, lab, t, r in map(quote_row, d["quotes"])]
    return section(1, "QUOTES", "The quote set", f"""
<p class="lead">Three deposits and six quarterly SHIBOR_3M swaps, numeric mode: tenors are year fractions on the ACT/365F
axis, no dates. One quote per pillar; every scheme is calibrated to the same nine instruments.</p>
{table_html(["instrument", "label", "tenor, y", "rate"], rows, text=False)}
""")


def sec_pillars(d: dict) -> str:
    summary = [[html.escape(n), sci(d["schemes"][n]["max_resid"])] for n in SCHEME_NAMES]
    cards = []
    for n in SCHEME_NAMES:
        s = d["schemes"][n]
        rows = [[num(t, 4), pct(z), num(df, 6), pct(f), sci(abs(r))]
                for t, z, df, f, r in zip(s["T"], s["zero"], s["df"], s["fwd"], s["residuals"])]
        cards.append(f'<div class="card"><h3><span>{html.escape(n)}</span><b>{sci(s["max_resid"])}</b></h3>'
                     + table_html(["T", "zero", "DF", "inst. fwd", "|residual|"], rows) + "</div>")
    return section(2, "PILLARS", "Discount curve under five schemes", f"""
<p class="lead">The same quotes bootstrapped under all five interpolation schemes. Each scheme reprices every deposit and
swap to rounding error, so the pillar zero rates and discount factors agree; the schemes differ only between pillars and
in the forward they imply at each pillar.</p>
{table_html(["scheme", "max |residual|"], summary, "Max over the nine quotes of the reprice residual (unit-notional PV).")}
<div class="cards">{"".join(cards)}</div>
""")


def sec_zero(d: dict) -> str:
    S = d["schemes"]
    chart = line_chart(GRID, [(n, S[n]["grid_zero"] * 100) for n in SCHEME_NAMES], xlabel="MATURITY T, YEARS",
                       ylabel="ZERO RATE z(T), %", yfmt=lambda v: num(v, 2), label="Zero rate by scheme",
                       markers=S[SCHEME_NAMES[0]]["T"])
    rows = [[num(t, 2)] + [pct(S[n]["table_zero"][i]) for n in SCHEME_NAMES] for i, t in enumerate(TABLE_GRID)]
    tbl = table_html(["T"] + SCHEME_NAMES, rows, "Continuously compounded zero rate. Grid 0.01 to 12 years; the last "
                                                  "pillar is 10 years, so 12 years is extrapolated (flat forward).")
    stack = np.array([S[n]["grid_zero"] for n in SCHEME_NAMES])
    spread_bp = (stack.max(axis=0) - stack.min(axis=0)) * 1e4
    k = int(np.argmax(spread_bp))
    return section(3, "ZERO", "Zero rates on a dense grid", f"""
<p class="lead">Zero rates are close: the curves share nine pillars and the zero is an average of the forward, so
the schemes stay within {num(float(spread_bp[k]), 1)} bp of each other on this grid (widest at T = {num(float(GRID[k]), 2)}).
The forward chart is where they separate.</p>
{figure(chart, "Zero rate z(T) by scheme", "Ticks under the axis mark the pillars.", tbl)}
""")


def sec_forward(d: dict) -> str:
    S = d["schemes"]
    chart = line_chart(GRID, [(n, S[n]["grid_fwd"] * 100) for n in SCHEME_NAMES], xlabel="MATURITY T, YEARS",
                       ylabel="INSTANTANEOUS FORWARD f(T), %", yfmt=lambda v: num(v, 2),
                       label="Instantaneous forward by scheme", markers=S[SCHEME_NAMES[0]]["T"])
    rows = [[num(t, 2)] + [pct(S[n]["table_fwd"][i]) for n in SCHEME_NAMES] for i, t in enumerate(TABLE_GRID)]
    tbl = table_html(["T"] + SCHEME_NAMES, rows, "Right-continuous instantaneous forward d(zT)/dT.")
    return section(4, "FORWARD", "Instantaneous forwards: where schemes differ", f"""
<p class="lead">The forward is the derivative of <code>z(T)·T</code>, so it magnifies interpolation. LOG_LINEAR_DF has a
piecewise-constant forward (a step at every pillar), LINEAR_ZERO has a sawtooth, the cubic schemes are smooth but can
oscillate, and MONOTONE_CONVEX keeps the forward continuous and positive.</p>
{figure(chart, "Instantaneous forward f(T) by scheme", "All five reprice the same nine quotes.", tbl)}
""")


def sec_basis(d: dict) -> str:
    dual = d["dual"]
    chart = line_chart(GRID, [("zero basis, bp", dual["grid_zero_basis"] * 1e4),
                              ("forward basis, bp", dual["grid_fwd_basis"] * 1e4)],
                       xlabel="MATURITY T, YEARS", ylabel="PROJECTION MINUS DISCOUNT, BP",
                       yfmt=lambda v: num(v, 0), label="Projection minus discount basis",
                       markers=dual["T"])
    rows = [[num(t, 4), pct(dz), pct(pz), bp(pz - dz), pct(df), pct(pf), bp(pf - df), sci(abs(r))]
            for t, dz, pz, df, pf, r in zip(dual["T"], dual["disc_zero"], dual["proj_zero"], dual["disc_fwd"],
                                            dual["proj_fwd"], dual["residuals"])]
    tbl = table_html(["T", "disc zero", "proj zero", "basis bp", "disc fwd", "proj fwd", "fwd basis bp", "|residual|"],
                     rows, f"Max |residual| {sci(dual['max_resid'])}: deposits and swaps reprice on the pair "
                           f"(discount, projection).")
    return section(5, "BASIS", "Dual curve: SHIBOR_3M projection", f"""
<p class="lead">The discount curve is the LOG_LINEAR_DF bootstrap shifted by {num(BASIS_SHIFT * 1e4, 0)} bp. The same swap
quotes are then bootstrapped as a SHIBOR_3M projection curve off it: the floating leg is projected on the new curve and
discounted on the shifted one, so the projection curve must sit above the discount curve to reprice the par rates.</p>
{figure(chart, "Projection minus discount, zero and forward", "Basis in basis points.", tbl)}
""")


def sec_carry(d: dict) -> str:
    c_ff, c_zf = d["carry"]["FLAT_FORWARD_CARRY"], d["carry"]["ZERO_FORWARD_CARRY"]
    c_fy = d["carry"]["FLAT_CARRY_YIELD"]
    chart = line_chart(CARRY_GRID, [
        ("r(T) zero rate", d["grid_rate"] * 100),
        ("implied q, FLAT_FORWARD_CARRY", c_ff["grid_yield"] * 100),
        ("implied q, ZERO_FORWARD_CARRY", c_zf["grid_yield"] * 100),
        ("implied q, FLAT_CARRY_YIELD", c_fy["grid_yield"] * 100),
    ], xlabel="MATURITY T, YEARS", ylabel="RATE / IMPLIED YIELD, %", yfmt=lambda v: num(v, 2),
        label="Carry-implied dividend yield and discount zero rate", markers=c_ff["T"])
    rows = [[num(t, 2), pct(d["table_rate"][i]), pct(c_ff["table_yield"][i]), pct(c_zf["table_yield"][i]),
             pct(c_fy["table_yield"][i]),
             pct(c_ff["table_fwd_carry"][i]), pct(c_zf["table_fwd_carry"][i])]
            for i, t in enumerate(CARRY_TABLE_GRID)]
    tbl = table_html(["T", "r(T)", "q flat-fwd", "q zero-fwd", "q flat-yield", "fwd carry flat-fwd", "fwd carry zero-fwd"], rows,
                     "The last futures node is 2 years; the rows at 2.5 to 4 years are extrapolated.")
    pillar_rows = [[html.escape(q.label), num(q.tenor_years, 2), num(q.price, 4), num(b, 6), pct(y, 4), sci(abs(r))]
                   for q, b, y, r in zip(d["futures"], c_ff["carry"], c_ff["yield"], c_ff["residuals"])]
    tail_ff = c_ff["grid_yield"][CARRY_GRID > c_ff["T"][-1]]
    return section(6, "CARRY", "Carry curve from index futures", f"""
<p class="lead">Spot {num(SPOT, 0)}; futures priced from a flat dividend yield of {pct(FLAT_Q, 2)} on the discount curve, so
the carry curve's implied yield returns {pct(FLAT_Q, 2)} at every futures tenor. Beyond the last node the carry
extrapolation decides the yield: FLAT_FORWARD_CARRY keeps the last forward carry, so <code>q</code> follows the
discount curve's shape from {pct(float(tail_ff.min()), 2)} to {pct(float(tail_ff.max()), 2)} out to
{num(float(CARRY_GRID[-1]), 0)} years; ZERO_FORWARD_CARRY freezes the cumulative carry, so <code>q</code> drifts up
toward <code>r</code>; FLAT_CARRY_YIELD holds the last zero carry yield <code>B/T</code>, so <code>q</code> moves only
with <code>r</code>.</p>
{table_html(["future", "T", "price", "carry B", "implied q", "|residual|"], pillar_rows,
            f"Residual: spot·exp(B) minus the quoted futures price; max {sci(c_ff['max_resid'])}.")}
{figure(chart, "Implied dividend yield q(T) and r(T)", "Ticks under the axis mark the futures nodes.", tbl)}
""")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

NAV = [(1, "QUOTES"), (2, "PILLARS"), (3, "ZERO"), (4, "FORWARD"), (5, "BASIS"), (6, "CARRY")]

CSS = """
/* Dark (terminal) is the base; light is the Swiss paper variant.  The OS preference picks the
   default, the switcher stamps data-theme on <html> and wins both ways. */
:root{
  --bg:#000;--panel:#0b0b0b;--line:#262626;--hair:#1a1a1a;--row:#141414;
  --ink:#e8e6df;--ink-2:#a3a199;--ink-3:#6f6d66;
  --amber:#ffa028;--amber-dim:#b37018;
  --s1:#d07a00;--s2:#2b95cf;--s3:#e0569b;--s4:#3ecf6e;--s5:#a68cf0;
  color-scheme:dark}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){
  --bg:#f4f2ec;--panel:#ebe8e0;--line:#cbc7bb;--hair:#dedad0;--row:#e8e4da;
  --ink:#111110;--ink-2:#46443e;--ink-3:#77746b;
  --amber:#b35a00;--amber-dim:#9a5b10;
  --s1:#c46a00;--s2:#1f7fc0;--s3:#d0457f;--s4:#13803a;--s5:#6b4fc8;
  color-scheme:light}}
:root[data-theme="light"]{
  --bg:#f4f2ec;--panel:#ebe8e0;--line:#cbc7bb;--hair:#dedad0;--row:#e8e4da;
  --ink:#111110;--ink-2:#46443e;--ink-3:#77746b;
  --amber:#b35a00;--amber-dim:#9a5b10;
  --s1:#c46a00;--s2:#1f7fc0;--s3:#d0457f;--s4:#13803a;--s5:#6b4fc8;
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
code{font:0.88em var(--mono);color:var(--amber)}
a{color:inherit;text-decoration:none}

/* Bloomberg function bar + numbered menu, frozen together */
.top{position: sticky; top: 0; z-index:5;background:var(--bar)}
.fbar{display:flex;flex-wrap:wrap;align-items:stretch;gap:0;background:var(--bar);
  border-bottom:1px solid var(--bar-amber);font:12px/1 var(--mono);letter-spacing:.04em}
.fbar>span{padding:9px 12px;border-right:1px solid var(--bar-line);white-space:nowrap;display:flex;align-items:center}
.brand{color:#000;background:var(--bar-amber);font-weight:700}
.cmd{color:var(--bar-amber);font-weight:700}
.fld{color:var(--bar-ink-3)}.fld b{color:var(--bar-ink);font-weight:500;margin-left:6px}
.fld.last{margin-left:auto}
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
.hero{display:grid;grid-template-columns:repeat(12,1fr);column-gap:24px;padding:72px 0 40px;border-bottom:1px solid var(--line)}
.kicker{grid-column:1/4;font:11px/1.4 var(--mono);letter-spacing:.14em;color:var(--amber);padding-top:12px}
.hero h1{grid-column:4/13;margin:0;font:700 clamp(34px,5vw,62px)/1.04 var(--sans);letter-spacing:-.025em}
.answer{grid-column:4/12;margin:28px 0 0;font-size:18px;line-height:1.5;color:var(--ink-2)}
.answer b{color:var(--ink);font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border-bottom:1px solid var(--line)}
.kpi{padding:22px 20px 22px 0;border-right:1px solid var(--line)}
.kpi+.kpi{padding-left:20px}.kpi:last-child{border-right:0}
.kv{font:600 clamp(24px,3vw,36px)/1 var(--mono);letter-spacing:-.02em;color:var(--ink)}
.kv small{font-size:12px;margin-left:6px;color:var(--ink-3);letter-spacing:.08em}
.kl{margin-top:10px;font:11px/1.4 var(--mono);letter-spacing:.06em;color:var(--ink-2);text-transform:uppercase}

section{display:grid;grid-template-columns:repeat(12,1fr);column-gap:24px;padding:56px 0;border-bottom:1px solid var(--line)}
.rail{grid-column:1/4}
.numeral{font:700 88px/0.85 var(--sans);letter-spacing:-.05em;color:var(--amber)}
.code{margin-top:18px;font:11px/1 var(--mono);letter-spacing:.16em;color:var(--ink-3)}
.rail h2{margin:8px 0 0;font:700 22px/1.2 var(--sans);letter-spacing:-.01em}
.body{grid-column:4/13;min-width:0}
.lead{margin:0 0 28px;font-size:17px;max-width:68ch}

/* tables: terminal density */
.scroll{overflow-x:auto;margin:8px 0 24px}
table{border-collapse:collapse;width:100%;font:12.5px/1.3 var(--mono);font-variant-numeric:tabular-nums}
th{padding:8px 12px 8px 0;text-align:right;font-weight:500;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--amber-dim);border-bottom:1px solid var(--amber-dim);white-space:nowrap}
td{padding:6px 12px 6px 0;text-align:right;border-bottom:1px solid var(--hair);white-space:nowrap}
th:first-child,td:first-child,table.text th,table.text td{text-align:left}
tbody tr:hover td{background:var(--row)}
.note{margin:6px 0 0;font:11px/1.5 var(--mono);color:var(--ink-3);max-width:none}

/* figures and charts */
figure{margin:0 0 8px;border-top:1px solid var(--line);padding-top:12px}
figcaption{display:flex;flex-wrap:wrap;gap:4px 16px;align-items:baseline;margin-bottom:10px}
figcaption b{font:600 13px/1.3 var(--sans)}
figcaption span{font:11px/1.3 var(--mono);color:var(--ink-3)}
.legend{display:flex;flex-wrap:wrap;gap:6px 20px;font:11px/1 var(--mono);color:var(--ink-2);margin:2px 0 6px}
.legend i{display:inline-block;width:22px;height:0;margin-right:7px;vertical-align:middle}
svg.chart{width:100%;height:auto;display:block;overflow:visible;margin-bottom:12px}
.chart .grid{stroke:var(--hair);stroke-width:1}
.chart .axis{stroke:var(--line);stroke-width:1}
.chart .zero{stroke:var(--ink-3);stroke-width:1}
.chart .pillar{stroke:var(--amber);stroke-width:1.5}
.chart .tick{font:10.5px var(--mono);fill:var(--ink-3)}
.chart .axlab{font:9.5px var(--mono);letter-spacing:.1em;fill:var(--ink-3)}
.chart .ln{fill:none;stroke-width:2;stroke-linejoin:round}

.cards{display:grid;grid-template-columns:1fr 1fr;gap:0 24px;margin-top:8px}
.card{border-top:1px solid var(--line);padding-top:10px;min-width:0}
.card h3{display:flex;justify-content:space-between;align-items:baseline;margin:0 0 4px;
  font:11px/1.3 var(--mono);letter-spacing:.1em;color:var(--ink-2)}
.card h3 b{font:600 16px/1 var(--mono);color:var(--amber);letter-spacing:0}
footer{padding:24px 0;font:11px/1.5 var(--mono);color:var(--ink-3);letter-spacing:.04em}

@media (max-width:820px){
  main{padding:0 16px 64px}
  .hero,section{grid-template-columns:minmax(0,1fr)}
  .kicker,.hero h1,.answer,.rail,.body{grid-column:1/-1}
  .rail{margin-bottom:20px}.numeral{font-size:56px}
  .kpis{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.kpi:nth-child(2){border-right:0}
  .kpi:nth-child(3){padding-left:0}.kpi:nth-child(n+3){border-top:1px solid var(--line)}
  .cards{grid-template-columns:minmax(0,1fr)}
  .fld{display:none}
  .menu{flex-wrap:nowrap;overflow-x:auto;scrollbar-width:none}.menu a{white-space:nowrap}
}
"""

# Runs in <head> so a stored theme applies before first paint (no flash).
THEME_INIT_JS = """<script>
try{const t=localStorage.getItem('qa-theme');if(t==='light'||t==='dark')document.documentElement.dataset.theme=t}catch(e){}
</script>"""

PAGE_JS = """<script>
(()=>{
const root=document.documentElement,btn=document.getElementById('theme'),os=matchMedia('(prefers-color-scheme: light)');
const current=()=>root.dataset.theme||(os.matches?'light':'dark');
const label=()=>{btn.querySelector('b').textContent=current()==='dark'?'LIGHT':'DARK'};
btn.addEventListener('click',()=>{const next=current()==='dark'?'light':'dark';root.dataset.theme=next;
  try{localStorage.setItem('qa-theme',next)}catch(e){}label()});
os.addEventListener('change',label);label();
const bar=document.querySelector('.top'),pad=()=>{root.style.scrollPaddingTop=(bar.offsetHeight+8)+'px'};pad();addEventListener('resize',pad);

const links=[...document.querySelectorAll('#menu a')];
const io=new IntersectionObserver(es=>es.forEach(en=>{if(!en.isIntersecting)return;
  links.forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+en.target.id))}),
  {rootMargin:'-90px 0px -70% 0px'});
document.querySelectorAll('section[id]').forEach(s=>io.observe(s));
})();
</script>"""


def render_page(d: dict) -> str:
    S = d["schemes"]
    worst = max(S[n]["max_resid"] for n in SCHEME_NAMES)
    long_fwd = [S[n]["table_fwd"][TABLE_GRID.index(5.0)] for n in SCHEME_NAMES]
    kpis = [
        (str(len(SCHEMES)), "", "Interpolation schemes calibrated"),
        (sci(worst), "", "Worst reprice residual, any scheme"),
        (bp(max(long_fwd) - min(long_fwd), 1), "BP", "Spread of the 5Y forward across schemes"),
        (sci(d["dual"]["max_resid"]), "", "Dual curve reprice residual"),
    ]
    tiles = "".join(f'<div class="kpi"><div class="kv">{v}<small>{u}</small></div>'
                    f'<div class="kl">{html.escape(l)}</div></div>' for v, u, l in kpis)
    nav = "".join(f'<a href="#s{n}"><b>{n})</b> {t}</a>' for n, t in NAV)
    chrome = f"""
<div class="top">
<header class="fbar">
  <span class="brand">QUANTARK</span><span class="cmd">CURV BUILD &lt;GO&gt;</span>
  <span class="fld">MODE <b>NUMERIC · ACT/365F</b></span><span class="fld">SCHEMES <b>5</b></span>
  <span class="fld last">INDEX <b>SHIBOR_3M</b></span>
  <button class="theme" id="theme" type="button" aria-label="Switch colour theme">◐ <b>LIGHT</b></button>
</header>
<nav class="menu" id="menu">{nav}</nav>
</div>
"""
    hero = f"""
<div class="hero">
  <div class="kicker">CURVES · BOOTSTRAP · CARRY</div>
  <h1>Five ways to join the same nine quotes.</h1>
  <p class="answer">A bootstrapped curve reprices its instruments exactly, whatever the interpolation. <b>Pillars agree;
  forwards do not.</b> The schemes differ between pillars, and most visibly in the instantaneous forward. A SHIBOR_3M
  projection curve and an index-futures carry curve are built on the same machinery.</p>
</div>
<div class="kpis">{tiles}</div>
"""
    content = (hero + sec_quotes(d) + sec_pillars(d) + sec_zero(d) + sec_forward(d) + sec_basis(d) + sec_carry(d)
               + "<footer>QUANTARK · example/param/curve_construction_demo.py · synthetic quotes, numeric mode, "
                 "no desk data</footer>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Curve construction</title>{THEME_INIT_JS}<style>{CSS}</style></head>"
            f"<body>{chrome}<main>{content}</main>{PAGE_JS}</body></html>")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(output_path: "str | Path | None" = None, quiet: bool = False) -> Path:
    out = Path(output_path) if output_path is not None else DEFAULT_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)

    def say(line: str = "") -> None:
        if not quiet:
            print(line)

    data = compute()
    _print_stdout(data, say)
    out.write_text(render_page(data), encoding="utf-8")
    say(f"\nwrote {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default=None, help="HTML report path (default example/output/curve_construction_report.html)")
    args = parser.parse_args()
    main(output_path=args.output)
