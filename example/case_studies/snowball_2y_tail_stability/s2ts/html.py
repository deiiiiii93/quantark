"""HTML rendering for the tail-stability report (Bloomberg x Swiss, ``docs/UI_LANGUAGE.md``).

Pure functions: ``render(data)`` turns the dict built by ``04_report.build_data`` into one self-contained
page (frozen function bar + numbered menu, light/dark switch, inline SVG, no external assets, a table under
every chart, true minus signs).  Every missing input renders as an explicit "missing" note, never an error.
The scaffolding (tokens, CSS, navigator, theme switch) follows
``example/case_studies/snowball_2y_carry_curve/04_report.py``.
"""

from __future__ import annotations

import html as _html
import math
from datetime import date
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import ANCHOR_WINDOW, ANCHOR_WINDOW_LONG, HALF_LIVES, IM_MULT, MO_MULT, NOTIONAL, RATE

MINUS = "−"
HF = "hold_forward"
BOOKS = ("a", "b")
BOOK_NAME = {"a": "Book A · single far IM", "b": "Book B · pillar lots (IM + MO C−P)"}
TAIL_LABEL = {
    "hold_forward": "hold forward carry", "hold_yield": "hold carry yield", "zero_carry": "zero carry",
    "converge_hl025": "converge hl 0.25Y", "converge_hl050": "converge hl 0.50Y", "converge_hl100": "converge hl 1.00Y",
}
#: One fixed series slot per tail (docs/UI_LANGUAGE.md tokens).
TAIL_COLOUR = {
    "hold_forward": "var(--s1)", "hold_yield": "var(--s2)", "zero_carry": "var(--s3)",
    "converge_hl025": "var(--s4)", "converge_hl050": "var(--s5)", "converge_hl100": "var(--s6)",
}
BOOK_COLOUR = {"a": "var(--s1)", "b": "var(--s2)"}
VARIANT_LABEL = {"base": "base", "multi_strike": "multi-strike MO", "all_settlement": "all settlement",
                 "anchor_252": "anchor 252d"}
SECTIONS = (("verdict", "VERDICT"), ("stability", "STABILITY"), ("hedging", "HEDGING"), ("basis", "BASIS"),
            ("curve", "CURVE"), ("method", "METHOD"), ("parity", "PARITY"))


# ---------------------------------------------------------------------------
# Formatting: true minus, never a negative zero, None -> en dash
# ---------------------------------------------------------------------------


def _m(text: str) -> str:
    return text.replace("-", MINUS)


def _ok(v: Optional[float]) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def _r(v: float, d: int) -> float:
    return round(float(v), d) + 0.0


def num(v: Optional[float], d: int = 2) -> str:
    return "–" if not _ok(v) else _m(f"{_r(v, d):,.{d}f}")


def snum(v: float, d: int = 2) -> str:
    v = _r(v, d)
    return _m(f"{v:,.{d}f}" if v == 0 else f"{v:+,.{d}f}")


def signed(v: Optional[float], d: int = 2, unit: str = "") -> str:
    """Signed change, sign printed AND coloured."""
    if not _ok(v):
        return "–"
    v = _r(v, d)
    cls = "up" if v > 0 else "dn" if v < 0 else "flat"
    return f'<span class="{cls}">{snum(v, d)}{unit}</span>'


def esc(s: Any) -> str:
    return _html.escape(str(s))


def tname(tail: str) -> str:
    return TAIL_LABEL.get(tail, tail)


def frac(a: Optional[int], b: Optional[int]) -> str:
    return "–" if a is None or b is None else f"{a}/{b}"


def yesno(v: Optional[bool]) -> str:
    if v is None:
        return '<span class="dim">n/d</span>'
    return '<span class="up">yes</span>' if v else '<span class="dn">no</span>'


def _ordinal_label(v: float) -> str:
    return date.fromordinal(int(round(v))).isoformat()[:7]


# ---------------------------------------------------------------------------
# Charts (inline SVG, styled by CSS tokens)
# ---------------------------------------------------------------------------


def _nice_ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    span = hi - lo
    raw = span / max(n, 1) if span > 0 else 1.0
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    ticks, t = [], math.floor(lo / step) * step
    while t <= hi + step * 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks


def _legend(items: Sequence[Tuple[str, str, str]]) -> str:
    spans = "".join(f'<span><i style="border-top:3px {style} {colour}"></i>{esc(name)}</span>'
                    for name, colour, style in items)
    return f'<div class="legend">{spans}</div>'


def _svg(width: int, height: int, body: List[str], label: str) -> str:
    return (f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}">'
            + "".join(body) + "</svg>")


def xy_chart(series: Sequence[Dict[str, Any]], *, xdomain: Tuple[float, float], xticks: Sequence[float],
             xfmt: Callable[[float], str], yfmt: Callable[[float], str], ylabel: str, xlabel: str, label: str,
             width: int = 760, height: int = 320) -> str:
    """Numeric-x line chart. ``series``: dicts ``name, xs, ys, colour, dash``; the first is drawn on top."""
    left, right, top, bottom = 62, 20, 16, 42
    ys = [y for s in series for y in s["ys"]]
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.08 or 1e-4
    yt = _nice_ticks(lo - pad, hi + pad)
    ylo, yhi = min(yt[0], lo - pad), max(yt[-1], hi + pad)
    xlo, xhi = xdomain

    def X(v: float) -> float:
        return left + (v - xlo) / (xhi - xlo) * (width - left - right)

    def Y(v: float) -> float:
        return top + (yhi - v) / (yhi - ylo) * (height - top - bottom)

    base = height - bottom
    body: List[str] = []
    for v in yt:
        body.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                    f'<text class="tick" x="{left - 8}" y="{Y(v) + 3.5:.1f}" text-anchor="end">{yfmt(v)}</text>')
    body.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{base}" y2="{base}"/>')
    for v in xticks:
        body.append(f'<text class="tick" x="{X(v):.1f}" y="{base + 18}" text-anchor="middle">{xfmt(v)}</text>')
    body.append(f'<text class="axlab" x="{left}" y="{height - 4}">{esc(xlabel)}</text>')
    body.append(f'<text class="axlab" x="6" y="12">{esc(ylabel)}</text>')
    for s in reversed(series):
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(s["xs"], s["ys"]))
        dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        body.append(f'<polyline class="ln" points="{pts}" style="stroke:{s["colour"]}"{dash}/>')
    legend = _legend([(s["name"], s["colour"], "dashed" if s.get("dash") else "solid") for s in series])
    return legend + _svg(width, height, body, label)


def grouped_bars(cats: Sequence[str], series: Sequence[Tuple[str, str, Sequence[Optional[float]]]], *,
                 yfmt: Callable[[float], str], tipfmt: Callable[[float], str], ylabel: str, xlabel: str,
                 label: str, width: int = 760, height: int = 300) -> str:
    """Grouped bars from zero. ``series``: [(name, colour, values)]; ``None`` leaves a gap."""
    left, right, top, bottom = 62, 16, 16, 46
    vals = [v for _, _, vs in series for v in vs if _ok(v)]
    hi = max(vals) if vals else 1.0
    yt = _nice_ticks(0.0, hi * 1.08)
    yhi = max(yt[-1], hi * 1.08)

    def Y(v: float) -> float:
        return top + (yhi - v) / yhi * (height - top - bottom)

    n, k = len(cats), len(series)
    slot = (width - left - right) / n
    gw = slot * 0.74
    bw = (gw - 2 * (k - 1)) / k
    base = height - bottom
    body: List[str] = []
    for v in yt:
        body.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                    f'<text class="tick" x="{left - 8}" y="{Y(v) + 3.5:.1f}" text-anchor="end">{yfmt(v)}</text>')
    for i, cat in enumerate(cats):
        gx = left + slot * i + (slot - gw) / 2
        for j, (name, colour, vs) in enumerate(series):
            if not _ok(vs[i]):
                continue
            tip = f"{name} · {cat} · {tipfmt(vs[i])}"
            body.append(f'<rect class="bar" x="{gx + j * (bw + 2):.1f}" y="{Y(vs[i]):.1f}" width="{bw:.1f}" '
                        f'height="{max(base - Y(vs[i]), 0.5):.1f}" style="fill:{colour}" data-tip="{esc(tip)}" '
                        f'aria-label="{esc(tip)}"/>')
        body.append(f'<text class="tick" x="{left + slot * (i + 0.5):.1f}" y="{base + 18}" '
                    f'text-anchor="middle">{esc(cat)}</text>')
    body.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{base}" y2="{base}"/>')
    body.append(f'<text class="axlab" x="{left}" y="{height - 4}">{esc(xlabel)}</text>')
    body.append(f'<text class="axlab" x="6" y="12">{esc(ylabel)}</text>')
    return _legend([(name, colour, "solid") for name, colour, _ in series]) + _svg(width, height, body, label)


def strip_chart(rows: Sequence[Dict[str, Any]], *, label: str, xlabel: str, width: int = 760) -> str:
    """Log-x strip chart. ``rows``: dicts ``name, colour, points [(value, tip)]``; a unit line marks ratio 1."""
    left, right, top, bottom = 190, 70, 14, 40
    pts_all = [v for r in rows for v, _ in r["points"] if _ok(v) and v > 0]
    lo_e = math.floor(math.log2(min(pts_all + [1.0]))) - 0.0
    hi_e = math.ceil(math.log2(max(pts_all + [1.0])))
    if hi_e == lo_e:
        hi_e += 1
    step = 22
    height = top + bottom + step * max(len(rows), 1)

    def X(v: float) -> float:
        return left + (math.log2(v) - lo_e) / (hi_e - lo_e) * (width - left - right)

    body: List[str] = []
    base = height - bottom
    for e in range(int(lo_e), int(hi_e) + 1):
        v = 2.0 ** e
        body.append(f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{top}" y2="{base}"/>'
                    f'<text class="tick" x="{X(v):.1f}" y="{base + 16}" text-anchor="middle">{num(v, 2 if v < 1 else 0)}</text>')
    body.append(f'<line class="zero" x1="{X(1.0):.1f}" x2="{X(1.0):.1f}" y1="{top}" y2="{base}"/>')
    body.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{base}" y2="{base}"/>')
    body.append(f'<text class="axlab" x="{left}" y="{height - 4}">{esc(xlabel)}</text>')
    for i, r in enumerate(rows):
        y = top + step * (i + 0.5)
        body.append(f'<text class="dlab" x="{left - 10}" y="{y + 3.5:.1f}" text-anchor="end">{esc(r["name"])}</text>')
        below = 0
        for v, tip in r["points"]:
            if not _ok(v) or v <= 0:
                continue
            below += v < 1.0
            body.append(f'<circle class="mk" cx="{X(v):.1f}" cy="{y:.1f}" r="4" style="fill:{r["colour"]}"/>'
                        f'<circle class="hit" cx="{X(v):.1f}" cy="{y:.1f}" r="9" data-tip="{esc(tip)}" '
                        f'aria-label="{esc(tip)}"/>')
        body.append(f'<text class="annot" x="{width - right + 8}" y="{y + 3.5:.1f}">{below}/{len(r["points"])} &lt; 1</text>')
    return _svg(width, height, body, label)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], note: str = "", text: bool = False,
          row_cls: Optional[Sequence[str]] = None) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = ""
    for i, r in enumerate(rows):
        cls = f' class="{row_cls[i]}"' if row_cls and row_cls[i] else ""
        body += f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    foot = f'<p class="note">{note}</p>' if note else ""
    cls = ' class="text"' if text else ""
    return f'<div class="scroll"><table{cls}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{foot}'


def figure(chart: str, title: str, caption: str) -> str:
    return f'<figure><figcaption><b>{esc(title)}</b><span>{esc(caption)}</span></figcaption>{chart}</figure>'


def section(no: int, sid: str, code: str, title: str, content: str) -> str:
    return (f'<section id="{sid}"><div class="rail"><div class="numeral">{no:02d}</div>'
            f'<div class="code">{code}</div><h2>{esc(title)}</h2></div><div class="body">{content}</div></section>')


def missing_note(text: str) -> str:
    return f'<p class="note missing">MISSING · {esc(text)}</p>'


def sub(text: str) -> str:
    return f'<h3 class="sub">{esc(text)}</h3>'


def points(items: Sequence[str], dim: bool = False) -> str:
    return f'<ul class="points{" dim" if dim else ""}">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _conv(data: Mapping[str, Any]) -> List[str]:
    return [t for t in data["tails"] if t.startswith("converge")]


def _hedge(data: Mapping[str, Any], book: str, tail: str) -> Optional[Mapping[str, Any]]:
    return data["hedge"].get(book, {}).get(tail)


def headline_text(data: Mapping[str, Any]) -> Tuple[str, str]:
    """(headline, answer paragraph) from the verdict, computed."""
    if not data["pairing_available"]:
        return ("Waiting for hold-forward runs.",
                "No hold-forward run is on disk, so no tail can be paired against the baseline yet.")
    verdict = data["verdict"]
    done = [t for t in verdict if verdict[t]["book_a"] != "not_decidable" and verdict[t]["book_b"] != "not_decidable"]
    qual = data["qualifying"]
    n_conv = len(verdict)
    if qual:
        names = ", ".join(tname(t) for t in qual)
        return (f"{len(qual)} of {n_conv} converging tails beat hold-forward in both books.",
                f"Lowest pooled daily hedged-P&L std in Book A: <b>{esc(names)}</b>. Each qualifier is lower on the pooled "
                f"std, in a strict majority of paired inceptions, and on the terminal P&L, in Book A and Book B.")
    if n_conv and len(done) == n_conv:
        return ("No converging tail beats hold-forward.",
                "Every converging tail is decidable and none passes (a) pooled std, (b) paired majority and (c) terminal "
                "in both books. The anchor does not buy a lower hedging variance on this history.")
    return ("Not shown yet that a converging tail beats hold-forward.",
            f"{len(done)} of {n_conv} converging tails are decidable (at least 5 completed inceptions in the terminal "
            "measure); none that is decidable passes in both books. Re-run the report as runs complete.")


def section_verdict(data: Mapping[str, Any]) -> str:
    head, _ = headline_text(data)
    parts = [f'<p class="lead"><b>{esc(head)}</b> Rule per book: (a) pooled daily hedged-P&amp;L std lower than '
             "hold-forward, (b) paired variance ratio below 1 in a strict majority of inceptions, (c) terminal "
             "P&amp;L std lower and |terminal| lower in a strict majority, over completed (non-truncated) "
             "inceptions; fewer than 5 completed is <i>not decidable</i>. A tail qualifies only if it passes in "
             "both books.</p>"]
    if not data["pairing_available"]:
        parts.append(missing_note("hold_forward runs: pairing unavailable, no verdict can be computed"))
    rows, cls = [], []
    for tail in ("converge_hl025", "converge_hl050", "converge_hl100"):
        v = data["verdict"].get(tail)
        cells = [esc(tname(tail))]
        if v is None:
            reason = "missing" if tail in data["missing"]["tails"] else "unavailable"
            cells += [f'<span class="dim">{reason}</span>'] * 7 + ['<span class="dim">–</span>']
        else:
            for k in BOOKS:
                h = _hedge(data, k, tail)
                cells.append(num(h["pooled_std"]) if h else "–")
            for k in BOOKS:
                h = _hedge(data, k, tail)
                cells.append(frac(h["n_lower"], h["n"]) if h else "–")
            for k in BOOKS:
                c = data["crit"][tail][k]
                cells.append(f'(a) {yesno(c["a"])} (b) {yesno(c["b"])} (c) {yesno(c["c"])}')
            hf_a, hf_b = _hedge(data, "a", HF), _hedge(data, "b", HF)
            cells.append(f'{num(hf_a["pooled_std"]) if hf_a else "–"} / {num(hf_b["pooled_std"]) if hf_b else "–"}')
            nd = "not_decidable" in (v["book_a"], v["book_b"])
            cells.append('<b class="up">BEATS</b>' if v["beats_hold_forward"] else '<b class="dim">NOT DECIDABLE</b>'
                         if nd else '<b class="dn">DOES NOT BEAT</b>')
        rows.append(cells)
        cls.append("vrow")
    parts.append(table(["Tail", "Book A std (bp)", "Book B std (bp)", "A n_lower/n", "B n_lower/n",
                        "Book A criteria", "Book B criteria", "Hold-fwd std A / B", "Verdict"], rows,
                       note="Std = sqrt(mean per-inception daily variance), bp of 10 mio notional, gross of costs. "
                            "Terminal is measured at the traded price; the vega-adjusted residual is a diagnostic "
                            "and never enters the verdict.", row_cls=cls))
    comp = [(t, data["verdict"][t].get("complete_only")) for t in data["verdict"]]
    if any(c for _, c in comp):
        rows2 = [[esc(tname(t)), yesno(c["daily_ok_a"]), yesno(c["daily_ok_b"])] for t, c in comp if c]
        parts.append(sub("Daily criterion restricted to completed inceptions"))
        parts.append(table(["Tail", "(a)+(b) Book A", "(a)+(b) Book B"], rows2,
                           note="Same daily rule on inceptions complete in both the tail and hold-forward "
                                "(truncated runs removed), a robustness check on the headline."))
    if data["missing"]["tails"]:
        parts.append(missing_note("tails with no runs: " + ", ".join(data["missing"]["tails"])))
    return section(1, "verdict", "VERDICT", "Does a converging tail beat hold-forward?", "".join(parts))


def section_stability(data: Mapping[str, Any]) -> str:
    st = data.get("stability")
    if not st:
        return section(2, "stability", "STABILITY", "q(2Y) day-to-day stability", missing_note("results/stability.json"))
    meta = st["meta"]
    variants = [v for v in VARIANT_LABEL if v in st["variants"]]
    tails = [t for t in TAIL_LABEL if t in st["variants"]["base"]["tails"]]
    series = [(VARIANT_LABEL[v], f"var(--s{i + 1})", [st["variants"][v]["tails"][t]["summary"]["all"]["std"] for t in tails])
              for i, v in enumerate(variants)]
    chart = grouped_bars([tname(t) for t in tails], series, yfmt=lambda v: num(v, 0), tipfmt=lambda v: f"{num(v, 1)} bp",
                         ylabel="STD OF DAILY Δq(2Y), BP", xlabel="TAIL", label="Std of daily change in q(2Y) per tail and variant")
    rows = []
    for t in tails:
        cells = [esc(tname(t))]
        for v in variants:
            s = st["variants"][v]["tails"][t]["summary"]
            cells.append(f'{num(s["all"]["std"], 1)} <span class="dim">/ {num(s["all"]["mad"], 1)}</span>')
        base = st["variants"]["base"]["tails"][t]["summary"]
        cells += [num(base["ordinary"]["std"], 1), num(base["atm_switch"]["std"], 1), num(base["mo_roll"]["std"], 1)]
        rows.append(cells)
    tbl = table(["Tail"] + [f"{VARIANT_LABEL[v]} std / mad" for v in variants]
                + ["base · ordinary", "base · ATM switch", "base · MO roll"], rows,
                note="Bp of yield, change of q(2Y) = r − B(2Y)/2 day over day. Variants share a date window per "
                     "variant, so tails compare paired within a column.")
    floor = None
    for t in tails:
        floor = st["variants"]["base"]["tails"][t].get("switch_floor")
        if floor:
            break
    rank = [esc(tname(t)) for t in st["variants"]["base"]["ranking"]]
    prose = [
        f'<b>Ranking by std of Δq(2Y), most stable first (base):</b> {" · ".join(rank)}.',
        "<b>zero_carry ranks most stable only because q(2Y) = r − B<sub>n</sub>/2 halves the pillar noise "
        "(its carry past the last pillar is zero, so only the pillar level B<sub>n</sub> is left in q(2Y)). "
        "This is a mechanical property of the formula, not a recommendation: a tail that ignores the curve "
        "beyond the last pillar is stable because it is uninformative.</b>",
    ]
    if floor:
        prose.append(f"<b>ATM-switch floor.</b> Re-pricing today's MO pillar at yesterday's ATM strike moves q(2Y) by "
                     f"median {num(floor['median_abs'], 1)} bp (p90 {num(floor['p90_abs'], 1)}, "
                     f"n = {floor['n']}): the part of the daily noise that is the pillar re-selection, not the market.")
    ch = ""
    base_v = st["variants"]["base"]
    dates = [date.fromisoformat(x).toordinal() for x in base_v["dates"]]
    if dates:
        years = range(date.fromordinal(dates[0]).year + 1, date.fromordinal(dates[-1]).year + 1)
        ticks = [date(y, 1, 1).toordinal() for y in years]
        ser = [dict(name=tname(t), xs=dates, ys=[100 * q for q in base_v["tails"][t]["q2y"]], colour=TAIL_COLOUR[t],
                    dash="4 3" if t in ("hold_yield", "zero_carry") else None) for t in tails]
        ch = figure(xy_chart(ser, xdomain=(dates[0], dates[-1]), xticks=ticks, xfmt=lambda v: str(date.fromordinal(int(v)).year),
                             yfmt=lambda v: num(v, 1) + "%", ylabel="q(2Y), % P.A.", xlabel="DATE",
                             label="Daily q(2Y) under six tails"),
                    "Daily q(2Y), six tails", f"base variant, {len(dates)} days with an anchor")
    return section(2, "stability", "STABILITY", "q(2Y) day-to-day stability",
                   f'<p class="lead">Rebuilt daily from the chain ({esc(meta["chain_first_day"])} → '
                   f'{esc(meta["chain_last_day"])}, {meta["chain_days"]} days). Smaller std of the daily change '
                   f'means a steadier input for the 2Y snowball.</p>'
                   + figure(chart, "Std of daily Δq(2Y)", "by tail and variant") + tbl + points(prose) + ch)


def _fmt_p(p: Optional[float]) -> str:
    return "–" if not _ok(p) else (num(p, 3) if p >= 0.001 else "&lt;0.001")


def _hedge_table(data: Mapping[str, Any], book: str) -> str:
    rows = []
    for tail in data["tails"]:
        h = _hedge(data, book, tail)
        if h is None:
            continue
        isf = tail == HF
        rows.append([
            esc(tname(tail)), num(h["pooled_std"]), "–" if isf else num(h["hf_pooled_std"]),
            "–" if isf else frac(h["n_lower"], h["n"]), "–" if isf else _fmt_p(h["p_raw"]),
            "–" if isf else _fmt_p(h["p_holm"]), num(h["r2"], 3), num(h["turn"], 2), num(h["roll_abs"], 1),
            num(h["max_dd"], 0), num(h["cost"], 1),
            num(h["term_std"], 1), num(h["term_mean"], 1), num(h["term_own_mean"], 1),
            frac(h["n_abs_lower"], h["n_complete"]) if not isf else str(h["n_complete"]),
            num(h["vr_pooled_std"]), "–" if isf else num(h["vr_ratio"], 2),
        ])
    return table(["Tail", "Pooled std", "HF std", "n_lower/n", "p raw", "p Holm", "R² vs note", "Lots/day", "Roll-day |bp|",
                  "Max DD", "Cost bp", "Term. std", "Term. mean", "Term. own-PV mean", "|term| lower/n compl.",
                  "Vega-adj. std (diag.)", "Vega-adj. / raw"], rows,
                 note="Bp of notional. Terminal: completed runs only, at the traded price (net of each tail's day-0 model "
                      "gap); the own-PV mean is shown alongside and not used by the verdict. Holm p over the three "
                      "converging tails. <b>Vega-adjusted std is a diagnostic</b>: it removes the unhedged daily vega "
                      "P&amp;L (common to all tails) and never enters the verdict.")


def section_hedging(data: Mapping[str, Any]) -> str:
    parts = ['<p class="lead">Short 2Y snowball, hedged daily, 13 quarterly inceptions × six tails on '
             "QUAD V2. Daily hedged P&amp;L is gross of costs.</p>"]
    # coverage
    incs = data["inceptions"]
    cov = data["coverage"]
    rows = [[esc(i)] + [cov.get(f"{i}|{t}", "missing") for t in data["all_tails"]] for i in incs]
    parts.append(sub("Run coverage"))
    parts.append(table(["Inception"] + [tname(t) for t in data["all_tails"]], rows,
                       note="done = completed run, trunc = hit the end of the data before KO or maturity, "
                            "missing = no run file, failed = .error.json (see #parity).") if incs else
                 missing_note("no inceptions known (no coupons.json and no runs)"))
    if not data["pairing_available"]:
        parts.append(missing_note("hold_forward runs: pairing unavailable, per-book tables cannot be built"))
    else:
        for k in BOOKS:
            parts.append(sub(BOOK_NAME[k]))
            parts.append(_hedge_table(data, k))
        strips = []
        for tail in _conv(data) + [t for t in data["tails"] if t not in (HF,) and not t.startswith("converge")]:
            for k in BOOKS:
                h = _hedge(data, k, tail)
                if h and h["ratios"]:
                    pts = [(r, f"{tname(tail)} · {BOOK_NAME[k].split(' ·')[0]} · {lab} · variance ratio {num(r, 2)}")
                           for r, lab in zip(h["ratios"], h["ratio_incs"])]
                    strips.append(dict(name=f"{tname(tail)} · {k.upper()}", colour=BOOK_COLOUR[k], points=pts))
        if strips:
            parts.append(figure(strip_chart(strips, label="Paired variance ratios per inception",
                                            xlabel="PAIRED VARIANCE RATIO vs HOLD-FORWARD (LOG), LEFT OF 1 = TAIL LOWER"),
                                "Paired variance ratio per inception", "one dot per inception; Book A amber, Book B blue"))
            parts.append(table(["Row", "Dots left of 1"],
                               [[esc(s["name"]), frac(sum(1 for v, _ in s["points"] if v < 1), len(s["points"]))] for s in strips]))
    gap = data["gap"]
    parts.append(sub("Inception-day model gap (static)"))
    if gap:
        rows = [[esc(tname(t)), num(g["mean"], 1), num(g["min"], 1), num(g["max"], 1), str(g["n"])] for t, g in gap.items()]
        parts.append(table(["Tail", "Mean gap bp", "Min", "Max", "n"], rows,
                           note="day0_pv − day0_pv under hold_forward, same inception, bp of notional. A valuation "
                                "difference at inception, not a hedging outcome: every tail books at its own day-0 PV "
                                "and the terminal measure is net of it."))
    else:
        parts.append(missing_note("no tail with a day-0 PV and a hold-forward run on the same inception"))
    return section(3, "hedging", "HEDGING", "Hedging variance by tail", "".join(parts))


def section_basis(data: Mapping[str, Any]) -> str:
    b = data["basis"]
    intro = ('<p class="lead">The calendar-spread question: does holding the pillar lots (Book B: IM plus MO C−P pair) '
             "instead of a single far IM (Book A) mitigate the IM-basis oscillation? Betas are OLS of daily hedged "
             "P&amp;L (bp) on the daily change in the far-IM and MO basis (percentage points), non-roll days; mean over "
             "inceptions.</p>")
    if not b:
        return section(4, "basis", "BASIS", "Does the calendar spread mitigate basis risk?",
                       intro + missing_note("no run with basis regressions"))
    rows = []
    for tail, agg in b.items():
        rows.append([esc(tname(tail)),
                     num(agg["a"]["abs_beta_far"], 1), num(agg["b"]["abs_beta_far"], 1), signed(agg["b_minus_a"]["abs_beta_far"], 1),
                     num(agg["a"]["abs_beta_mo"], 1), num(agg["b"]["abs_beta_mo"], 1),
                     num(agg["a"]["basis_r2"], 3), num(agg["b"]["basis_r2"], 3), signed(agg["b_minus_a"]["basis_r2"], 3),
                     num(agg["a"]["cost_bp"], 1), num(agg["b"]["cost_bp"], 1), signed(agg["b_minus_a"]["cost_bp"], 1)])
    tbl = table(["Tail", "A |β far|", "B |β far|", "B−A", "A |β MO|", "B |β MO|", "A basis R²", "B basis R²", "B−A R²",
                 "A cost bp", "B cost bp", "B−A cost"], rows,
                note="β in bp per 1 percentage-point move of the basis. Negative B−A in |β far| or R² means Book B "
                     "is less exposed to basis. B−A cost is the extra transaction cost of holding pillar lots.")
    return section(4, "basis", "BASIS", "Does the calendar spread mitigate basis risk?", intro + tbl)


def section_curve(data: Mapping[str, Any]) -> str:
    parts = ['<p class="lead">The converging tail pulls the carry from the last pillar toward a trailing '
             f"{ANCHOR_WINDOW}-day mean of the far-IM carry, with half-life {', '.join(f'{h:g}' for h in HALF_LIVES)}Y.</p>"]
    st = data.get("stability")
    if st and st["variants"]["base"]["anchor"]:
        base = st["variants"]["base"]
        d = [date.fromisoformat(x).toordinal() for x in base["dates"]]
        years = range(date.fromordinal(d[0]).year + 1, date.fromordinal(d[-1]).year + 1)
        ser = [dict(name=f"{ANCHOR_WINDOW}d anchor", xs=d, ys=[100 * a for a in base["anchor"]], colour="var(--s1)")]
        parts.append(figure(xy_chart(ser, xdomain=(d[0], d[-1]), xticks=[date(y, 1, 1).toordinal() for y in years],
                                     xfmt=lambda v: str(date.fromordinal(int(v)).year), yfmt=lambda v: num(v, 1) + "%",
                                     ylabel="ANCHOR B∞, % P.A.", xlabel="DATE", label="Trailing far-IM carry anchor"),
                            "Anchor series", f"trailing {ANCHOR_WINDOW}-day mean of ln(F_far/S)/T_far, point-in-time"))
        parts.append(table(["First", "Last", "Min %", "Max %", "Latest %"],
                           [[base["dates"][0], base["dates"][-1], num(100 * min(base["anchor"]), 2),
                             num(100 * max(base["anchor"]), 2), num(100 * base["anchor"][-1], 2)]]))
    else:
        parts.append(missing_note("anchor series (results/stability.json)"))
    c = data.get("curve")
    if c:
        tails = [t for t in TAIL_LABEL if t in c["q"]]
        ser = [dict(name=tname(t), xs=c["grid"], ys=[100 * q for q in c["q"][t]], colour=TAIL_COLOUR[t],
                    dash="4 3" if t in ("hold_yield", "zero_carry") else None) for t in tails]
        parts.append(figure(xy_chart(ser, xdomain=(min(c["grid"]), max(c["grid"])), xticks=[0.5, 1.0, 1.5, 2.0],
                                     xfmt=lambda v: f"{v:g}Y", yfmt=lambda v: num(v, 1) + "%", ylabel="q(T), % P.A.",
                                     xlabel="TENOR T", label="q(T) out to 2Y under six tails"),
                            f"q(T) to 2Y on {c['day']}", "one illustrative day, the six tails on the same pillars"))
        idx = [min(range(len(c["grid"])), key=lambda i: abs(c["grid"][i] - T)) for T in (0.5, 1.0, 1.5, 2.0)]
        parts.append(table(["Tail"] + [f"q({c['grid'][i]:g}Y) %" for i in idx],
                           [[esc(tname(t))] + [num(100 * c["q"][t][i], 2) for i in idx] for t in tails]))
    else:
        parts.append(missing_note("illustrative-day curve (needs OUTPUT_DIR/daily_chain.pkl)"))
    return section(5, "curve", "CURVE", "The anchor and the tails", "".join(parts))


def section_method(data: Mapping[str, Any]) -> str:
    n_runs = data["meta"]["n_runs"]
    ends = data["end_reasons"]
    ends_txt = ", ".join(f"{k} {v}" for k, v in sorted(ends.items())) if ends else "no hold-forward run yet"
    spec = [
        ("Trade", f"Short 2Y standard snowball, notional {NOTIONAL:,.0f} CNY, strike = KO ref = inception spot, KO 100% "
                  "monthly from month 3 to 24, KI 75% discrete daily, coupon = KO coupon = rebate, principal excluded."),
        ("Inceptions", "Quarterly, one coupon per inception solved to PV = 0 under hold-forward and shared by all tails."),
        ("Booking", "Each tail books at its own day-0 PV; the day-0 model gap is excluded from P&amp;L (see #hedging)."),
        ("Tails", f"hold forward, hold carry yield, zero carry, converge to a trailing {ANCHOR_WINDOW}-day far-IM anchor with "
                  f"half-lives {', '.join(f'{h:g}' for h in HALF_LIVES)}Y ({ANCHOR_WINDOW_LONG}-day anchor as a sensitivity)."),
        ("Marks", f"Spot = CSI 1000 close; IM = CFFEX close; MO = CFFEX settlement; flat r = {100 * RATE:g}%, ACT/365F."),
        ("Hedges", f"Book A: far IM only (follows the longest IM). Book B: IM pillar lots plus MO C−P pair at its opened "
                   f"strike and the pinned-forward delta in the far IM. Multipliers IM {IM_MULT:g}, MO pair 100·DF(T_n)."),
        ("Costs", "IM 1 bp of traded notional; MO 0.2 pt per option leg per side (40 CNY per pair). Variance is gross of costs."),
        ("Engine", "QUAD V2 (defaults), flat vol per day from the chain; V1 parity check on inception day."),
        ("Std", "Pooled std = sqrt(mean per-inception daily variance), equal weight per inception."),
        ("Terminal", "At the traded price: 0 − realised note cash + realised hedge P&amp;L + cash accrual; own-PV version alongside."),
    ]
    dl = '<dl class="spec">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in spec) + "</dl>"
    cav = [
        "<b>Overlapping inceptions.</b> 2Y notes struck a quarter apart share most of their calendar, so daily hedged P&amp;L is "
        "strongly correlated across inceptions and the paired tests overstate the independent evidence.",
        "<b>Truncation.</b> Notes not yet knocked out or matured at the end of the data are truncated; daily statistics use their "
        "live days, the terminal measure uses completed runs only (under 5 completed is not decidable).",
        f"<b>Early knock-outs.</b> Hold-forward run endings: {esc(ends_txt)}. A knock-out ends the hedge early, so those daily "
        "samples are short and the terminal sample is small.",
        "<b>Unhedged vega.</b> Neither book hedges vega; its daily P&amp;L is common to all tails and dominates the raw hedged "
        "variance, which is why a vega-adjusted residual is shown as a diagnostic.",
        "<b>One flat vol.</b> No smile, local or stochastic vol; the vol used each day is the one the chain implies.",
        "<b>Basis is not modelled.</b> The curve is built from marks; the realised basis is in the P&amp;L, which is the point.",
    ]
    return section(6, "method", "METHOD", "Setup, marks, costs and caveats",
                   f'<p class="lead">{n_runs} run files read.</p>' + dl + sub("Caveats") + points(cav, dim=True))


def section_parity(data: Mapping[str, Any]) -> str:
    p = data["parity"]
    parts = ['<p class="lead">On inception day (fresh product) the V2 PV and pillar lots are checked against the V1 '
             "transition-readout engine; a breach is a PV difference above 0.5 bp of notional or a lot difference above 0.01.</p>"]
    if p["rows"]:
        rows = [[esc(r["inception"]), esc(tname(r["tail"])), num(r["pv_diff_bp"], 3), num(r["max_lot_diff"], 4),
                 '<span class="dn">FLAG</span>' if r["flag"] else '<span class="up">ok</span>'] for r in p["rows"]]
        parts.append(table(["Inception", "Tail", "|PV V2 − V1| bp", "Max lot diff", "Flag"], rows,
                           note=f"{p['n_flag']} flagged of {len(p['rows'])} checked."))
    else:
        parts.append(missing_note("no run carries a parity record"))
    parts.append(sub("Failed runs"))
    if data["failed"]:
        parts.append(table(["Run (inception__tail)"], [[f"<code>{esc(f)}</code>"] for f in data["failed"]],
                           note="Each left an .error.json under runs/; they are excluded from every statistic above.", text=True))
    else:
        parts.append('<p class="note">No failed run files.</p>')
    return section(7, "parity", "PARITY", "V1 parity and failed runs", "".join(parts))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def masthead(data: Mapping[str, Any]) -> Tuple[str, str]:
    meta = data["meta"]
    headline, answer = headline_text(data)
    hf_a = _hedge(data, "a", HF)
    hf_b = _hedge(data, "b", HF)
    kpis = [
        (f"{meta['n_runs']}<small>/{meta['n_expected']}</small>", "Runs found / expected"),
        (f"{len(data['qualifying'])}<small>/{len(data['verdict'])}</small>", "Converging tails beating hold-forward"),
        (num(hf_a["pooled_std"], 0) + "<small>BP</small>" if hf_a else "–", "Hold-forward pooled std · Book A"),
        (num(hf_b["pooled_std"], 0) + "<small>BP</small>" if hf_b else "–", "Hold-forward pooled std · Book B"),
    ]
    tiles = "".join(f'<div class="kpi"><div class="kv">{v}</div><div class="kl">{esc(lab)}</div></div>' for v, lab in kpis)
    nav = "".join(f'<a href="#{sid}"><b>{i + 1})</b> {lab}</a>' for i, (sid, lab) in enumerate(SECTIONS))
    chrome = f"""
<div class="top">
<header class="fbar">
  <span class="brand">QUANTARK</span><span class="cmd">SNBL TAIL &lt;GO&gt;</span>
  <span class="fld">STUDY <b>snowball_2y_tail_stability</b></span><span class="fld">UL <b>000852.SH</b></span>
  <span class="fld">ENGINE <b>QUAD V2</b></span><span class="fld date">{esc(meta.get("chain_last_day", ""))}</span>
  <button class="theme" id="theme" type="button" aria-label="Switch colour theme">◐ <b>LIGHT</b></button>
</header>
<nav class="menu" id="menu">{nav}</nav>
</div>
"""
    hero = f"""
<div class="hero">
  <div class="kicker">SNOWBALL · CARRY TAIL · 2Y</div>
  <h1>{esc(headline)}</h1>
  <p class="answer">{answer}</p>
</div>
<div class="kpis">{tiles}</div>
"""
    return chrome, hero


CSS = """
:root{
  --bg:#000;--panel:#0b0b0b;--line:#262626;--hair:#1a1a1a;--row:#141414;
  --ink:#e8e6df;--ink-2:#a3a199;--ink-3:#6f6d66;
  --amber:#ffa028;--amber-dim:#b37018;
  --up:#3ecf6e;--dn:#ff5a4f;
  --s1:#d07a00;--s2:#2b95cf;--s3:#e0569b;--s4:#7bc96f;--s5:#a78bfa;--s6:#3fc1b0;
  color-scheme:dark}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){
  --bg:#f4f2ec;--panel:#ebe8e0;--line:#cbc7bb;--hair:#dedad0;--row:#e8e4da;
  --ink:#111110;--ink-2:#46443e;--ink-3:#77746b;
  --amber:#b35a00;--amber-dim:#9a5b10;
  --up:#13803a;--dn:#c42b20;
  --s1:#c46a00;--s2:#1f7fc0;--s3:#d0457f;--s4:#3d8b2f;--s5:#6d4fd0;--s6:#168a7c;
  color-scheme:light}}
:root[data-theme="light"]{
  --bg:#f4f2ec;--panel:#ebe8e0;--line:#cbc7bb;--hair:#dedad0;--row:#e8e4da;
  --ink:#111110;--ink-2:#46443e;--ink-3:#77746b;
  --amber:#b35a00;--amber-dim:#9a5b10;
  --up:#13803a;--dn:#c42b20;
  --s1:#c46a00;--s2:#1f7fc0;--s3:#d0457f;--s4:#3d8b2f;--s5:#6d4fd0;--s6:#168a7c;
  color-scheme:light}
:root{
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

main{max-width:1180px;margin:0 auto;padding:0 32px 96px}
.hero{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));column-gap:24px;padding:72px 0 40px;border-bottom:1px solid var(--line)}
.kicker{grid-column:1/4;font:11px/1.4 var(--mono);letter-spacing:.14em;color:var(--amber);padding-top:12px}
.hero h1{grid-column:4/13;margin:0;font:700 clamp(30px,4.4vw,54px)/1.06 var(--sans);letter-spacing:-.025em}
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

dl.spec{display:grid;grid-template-columns:160px minmax(0,1fr);margin:0 0 24px;border-top:1px solid var(--line)}
dl.spec dt,dl.spec dd{margin:0;padding:10px 0;border-bottom:1px solid var(--hair)}
dl.spec dt{font:11px/1.6 var(--mono);letter-spacing:.1em;color:var(--amber);text-transform:uppercase}

.scroll{overflow-x:auto;margin:8px 0}
table{border-collapse:collapse;width:100%;font:12.5px/1.3 var(--mono);font-variant-numeric:tabular-nums}
th{padding:8px 12px 8px 0;text-align:right;font-weight:500;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--amber-dim);border-bottom:1px solid var(--amber-dim);white-space:nowrap}
td{padding:6px 12px 6px 0;text-align:right;border-bottom:1px solid var(--hair);white-space:nowrap}
th:first-child,td:first-child,table.text th,table.text td{text-align:left}
tbody tr:hover td{background:var(--row)}
td code{color:var(--ink)}
.up{color:var(--up)}.dn{color:var(--dn)}.flat{color:var(--ink-2)}.dim{color:var(--ink-3)}
.note{margin:6px 0 0;font:11px/1.5 var(--mono);color:var(--ink-3);max-width:none}
.note.missing{color:var(--amber)}

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
.chart .annot{font:10px var(--mono);fill:var(--ink-2);letter-spacing:.04em}
.chart .tick{font:10.5px var(--mono);fill:var(--ink-3)}
.chart .axlab{font:9.5px var(--mono);letter-spacing:.1em;fill:var(--ink-3)}
.chart .ln{fill:none;stroke-width:2;stroke-linejoin:round}
.chart .mk{stroke:var(--bg);stroke-width:2;pointer-events:none}
.chart .hit{fill:transparent;cursor:crosshair}
.chart .bar{cursor:crosshair}.chart .bar:hover{stroke:var(--ink);stroke-width:1.5}
.chart .dlab{font:11px var(--mono);fill:var(--ink-2)}
.tip{position:fixed;z-index:10;pointer-events:none;padding:6px 9px;background:var(--panel);border:1px solid var(--amber);
  color:var(--ink);font:11.5px/1.3 var(--mono);white-space:nowrap;display:none}

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


def render(data: Mapping[str, Any]) -> str:
    """One self-contained HTML page from the report data (see ``04_report.build_data``)."""
    chrome, hero = masthead(data)
    content = (hero + section_verdict(data) + section_stability(data) + section_hedging(data) + section_basis(data)
               + section_curve(data) + section_method(data) + section_parity(data)
               + "<footer>QUANTARK · example/case_studies/snowball_2y_tail_stability · 01_data → 02_stability → "
                 "03_backtest → 04_report · exchange closes and settlements only, no desk data</footer>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>2Y snowball · carry-tail stability</title>{THEME_INIT_JS}<style>{CSS}</style></head>"
            f"<body>{chrome}<main>{content}</main>{PAGE_JS}</body></html>\n")
