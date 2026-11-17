"""Stage 05 - Render the CSI basket-delta study as one self-contained HTML report.

Reads ``results/01..04_*.json`` (run stages 01-04 first) and writes
``results/csi_basket_delta_report.html``.  Every figure on the page is computed
from those JSON files when the page is built; nothing is re-priced here and no
finding is typed in by hand.  A gap is called nonzero only when its JSON
``resolved`` flag is set (|gap| > 3 paired SE, spec D7/AC12); everything else
prints as "not resolved".

The page uses the house UI language (``docs/UI_LANGUAGE.md``): a Bloomberg
terminal surface on a Swiss grid, a frozen function bar + section menu, a
light/dark switcher and hand-built inline SVG charts.

    python example/case_studies/csi_basket_delta/05_report.py
"""

from __future__ import annotations

import html
import json
import math
import re
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
REPORT_PATH = RESULTS_DIR / "csi_basket_delta_report.html"
STAGES = ("01_sanity", "02_fixed_shares", "03_rebalance", "04_dispersion_correlation")

TRADING_DAYS = 244          # spec D4: period lengths on a 244-trading-day year
SE_MULTIPLE = 3.0           # spec D7: a difference counts only above 3 SE
METHODS = ("M1", "M2", "M3", "M4", "M5")
METHOD_NOTES = {
    "REF": ("MC pathwise", "Multi-name GBM, per-name vols and correlation, exact sampling at rebalance dates; "
                           "the per-name pathwise derivative. The truth every method is scored against."),
    "M1": ("Δ_I × current shares", "The whole-index (combination) order. Also exactly what bumping one name gives "
                                   "on an index-only pricer."),
    "M2": ("Δ_I × target weights", "The shortcut: split Δ_I by the target weights instead of today's shares."),
    "M3": ("per-name FD on MC", "Central 1% bump of one name on the multi-name MC, common random numbers, strike "
                                "held at the unbumped level."),
    "M4": ("M1 on the MM model", "Δ_I from the moment-matched lognormal, split by current shares."),
    "M5": ("per-name MM", "Analytic per-name derivative of the moment-matched lognormal (chain rule through "
                          "the stub moments)."),
}
SERIES = ("var(--s1)", "var(--s2)", "var(--s3)")
MINUS = "−"


def load_results() -> dict:
    """Read the four stage JSON files; nothing else is read."""
    out = {}
    for stage in STAGES:
        path = RESULTS_DIR / f"{stage}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}: run {stage}.py first")
        out[stage] = json.loads(path.read_text())
    return out


# ---------------------------------------------------------------------------
# Number formatting: round first, then sign, true minus
# ---------------------------------------------------------------------------


def _m(text: str) -> str:
    return text.replace("-", MINUS)


def lab(text: str) -> str:
    """A JSON label for HTML: a minus sign before a figure becomes the true minus."""
    return html.escape(re.sub(r"(?<![\w.])-(?=\d)", MINUS, text))


def _r(v: float, d: int) -> float:
    return round(v, d) + 0.0          # + 0.0 turns -0.0 into 0.0


def num(v, d: int = 0) -> str:
    return "–" if v is None else _m(f"{_r(v, d):,.{d}f}")


def sgn(v, d: int = 2) -> str:
    """Signed plain text (statistics such as z-scores): no colour."""
    v = _r(v, d)
    return f"{0:.{d}f}" if v == 0 else _m(f"{v:+,.{d}f}")


def signed(v, d: int = 0, unit: str = "") -> str:
    """A signed difference: sign printed AND coloured."""
    if v is None:
        return "–"
    v = _r(v, d)
    if v == 0:
        return f'<span class="flat">{0:.{d}f}{unit}</span>'
    cls = "up" if v > 0 else "dn"
    return f'<span class="{cls}">{_m(f"{v:+,.{d}f}")}{unit}</span>'


def spct(v, d: int = 1) -> str:
    return signed(v * 100.0, d, "%")


def pct(v, d: int = 1) -> str:
    return num(v * 100.0, d) + "%"


def sci(v) -> str:
    return _m(f"{v:.1e}")


def compact(v: float) -> str:
    a = abs(v)
    if a >= 1e6:
        return num(v / 1e6, 1) + "m"
    if a >= 1e3:
        d = 0 if a >= 1e4 or _r(a / 1e3, 1) == round(a / 1e3) else 1
        return num(v / 1e3, d) + "k"
    return num(v, 0)


NR = '<span class="nr">not resolved</span>'


def gap_cell(m: dict, i: int, rel: bool = False) -> str:
    """One per-name gap: signed CNY per 1% ± paired SE, or 'not resolved'."""
    if not m["resolved"][i]:
        return NR
    if rel:
        return spct(m["gap_rel_to_ref"][i])
    return f'{signed(m["gap_cash_per_pct"][i])} <span class="dim">±{num(m["gap_se"][i])}</span>'


def sigma_ub(m: dict) -> float:
    """3-SE upper bound of the residual sigma (used only for unresolved values)."""
    return math.sqrt(max(m["residual_q"], 0.0) + SE_MULTIPLE * m["residual_q_se"])


def sigma_text(m: dict) -> str:
    if m["residual_resolved"]:
        return num(m["residual_sigma_cny"])
    return f"not resolved (&lt; {num(sigma_ub(m))})"


def sigma_cell(m: dict) -> str:
    if m["residual_resolved"]:
        return num(m["residual_sigma_cny"])
    return f'<span class="nr">not resolved · &lt; {num(sigma_ub(m))}</span>'


def sigma_point(m: dict):
    """(value, hollow) for charts: unresolved points sit at their 3-SE upper bound."""
    if m["residual_resolved"]:
        return m["residual_sigma_cny"], False
    return sigma_ub(m), True


def pv(rec: dict) -> dict:
    """PV of the option on the record's notional (CNY): MC ± SE, MM, MM - MC."""
    p, u = rec["price"], rec["units"]
    return {
        "mc": p["mc"] * u, "se": p["mc_se"] * u, "mm": p["mm"] * u,
        "err": p["mm_minus_mc"] * u, "rel": p["mm_minus_mc"] / p["mc"],
        "resolved": abs(p["mm_minus_mc"]) > SE_MULTIPLE * p["mc_se"],
    }


def name_idx(rec: dict, name: str) -> int:
    return rec["names"].index(name)


def max_abs_resolved(m: dict):
    """(max |gap| over resolved names, hollow) — hollow when no name is resolved."""
    vals = [abs(g) for g, ok in zip(m["gap_cash_per_pct"], m["resolved"]) if ok]
    if vals:
        return max(vals), False
    return max(abs(g) for g in m["gap_cash_per_pct"]), True


# ---------------------------------------------------------------------------
# SVG charts (from the snowball_ftp_study reference, plus gaps/hollow/log-y)
# ---------------------------------------------------------------------------


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list:
    span = hi - lo
    if span <= 0:
        lo, hi, span = lo - 1.0, hi + 1.0, 2.0
    raw = span / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    ticks, t = [], start
    while t <= hi + step * 1e-9:
        ticks.append(round(t, 10) + 0.0)
        t += step
    return ticks


class _Frame:
    """Plot-area geometry."""

    def __init__(self, width, height, ylo, yhi, *, ticks=None, left=56, right=92, top=28, bottom=40):
        self.w, self.h = width, height
        self.l, self.r, self.t, self.b = left, right, top, bottom
        ticks = ticks or _nice_ticks(ylo, yhi)
        self.yticks = ticks
        self.ylo, self.yhi = min(ticks[0], ylo), max(ticks[-1], yhi)

    def y(self, v):
        return self.t + (self.yhi - v) / (self.yhi - self.ylo) * (self.h - self.t - self.b)

    def xslot(self, i, n):
        inner = self.w - self.l - self.r
        return self.l + inner * (i + 0.5) / n

    def axes(self, xlabels, ytick_fmt, ylabel, xlabel, zero=False):
        out = []
        for v in self.yticks:
            y = self.y(v)
            out.append(f'<line class="grid" x1="{self.l}" x2="{self.w - self.r}" y1="{y:.1f}" y2="{y:.1f}"/>'
                       f'<text class="tick" x="{self.l - 8}" y="{y + 3.5:.1f}" text-anchor="end">{ytick_fmt(v)}</text>')
        if zero and self.ylo < 0 < self.yhi:
            y0 = self.y(0)
            out.append(f'<line class="zero" x1="{self.l}" x2="{self.w - self.r}" y1="{y0:.1f}" y2="{y0:.1f}"/>')
        base = self.h - self.b
        out.append(f'<line class="axis" x1="{self.l}" x2="{self.w - self.r}" y1="{base}" y2="{base}"/>')
        for i, xl_text in enumerate(xlabels):
            out.append(f'<text class="tick" x="{self.xslot(i, len(xlabels)):.1f}" y="{base + 16}" '
                       f'text-anchor="middle">{html.escape(xl_text)}</text>')
        out.append(f'<text class="axlab" x="{self.l}" y="{self.h - 4}">{html.escape(xlabel)}</text>')
        out.append(f'<text class="axlab" x="{self.l - 48}" y="{self.t - 16}">{html.escape(ylabel)}</text>')
        return out


def _svg(frame, body, label):
    return (f'<svg class="chart" viewBox="0 0 {frame.w} {frame.h}" role="img" aria-label="{html.escape(label)}">'
            + "".join(body) + "</svg>")


def _legend(names):
    items = "".join(f'<span><i style="background:{SERIES[k]}"></i>{html.escape(n)}</span>'
                    for k, n in enumerate(names))
    return f'<div class="legend">{items}</div>'


def line_chart(xlabels, series, *, ylabel, xlabel, yfmt, tipfmt, zero=False, ydomain=None, hollow=None,
               logy=False, width=760, height=290, label=""):
    """``series``: [(name, ys)], fixed slot order, ``None`` leaves a gap.

    ``hollow``: per-series list of bools; a hollow marker is a value that is not
    resolved at 3 SE.  ``logy`` plots log10 with decade ticks.
    """
    hollow = hollow or [[False] * len(ys) for _, ys in series]
    tr = (lambda v: math.log10(v)) if logy else (lambda v: v)
    vals = [tr(v) for _, ys in series for v in ys if v is not None and (not logy or v > 0)]
    if logy:
        lo, hi = math.floor(min(vals)), math.ceil(max(vals))
        f = _Frame(width, height, lo, hi, ticks=list(range(lo, hi + 1)))
    else:
        lo, hi = ydomain or (min(vals), max(vals))
        lo, hi = min(lo, min(vals)), max(hi, max(vals))
        pad = (hi - lo) * 0.06 or 1.0
        f = _Frame(width, height, lo if lo == 0 else lo - pad, hi + pad)
    body = f.axes(xlabels, (lambda t: yfmt(10 ** t)) if logy else yfmt, ylabel, xlabel, zero=zero)
    n = len(xlabels)
    ends = []
    for k, (name, ys) in enumerate(series):
        pts = [(f.xslot(i, n), f.y(tr(v)), v, xl, h) if v is not None and (not logy or v > 0) else None
               for i, (v, xl, h) in enumerate(zip(ys, xlabels, hollow[k]))]
        segs, cur = [], []
        for p in pts:
            if p is None:
                if cur:
                    segs.append(cur)
                cur = []
            else:
                cur.append(p)
        if cur:
            segs.append(cur)
        for seg in segs:
            if len(seg) > 1:
                path = " ".join(f"{x:.1f},{y:.1f}" for x, y, *_ in seg)
                body.append(f'<polyline class="ln" points="{path}" style="stroke:{SERIES[k]}"/>')
        for p in pts:
            if p is None:
                continue
            x, y, v, xl, h = p
            tip = html.escape(f"{name} · {xl} · {tipfmt(v)}" + (" · not resolved" if h else ""))
            style = f"stroke:{SERIES[k]};fill:var(--bg)" if h else f"fill:{SERIES[k]}"
            body.append(f'<circle class="mk{" hollow" if h else ""}" cx="{x:.1f}" cy="{y:.1f}" r="4" '
                        f'style="{style}"/>'
                        f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="12" data-tip="{tip}" '
                        f'aria-label="{tip}"/>')
        last = [p for p in pts if p is not None]
        if last:
            ends.append([last[-1][1], name, last[-1][0]])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 13)
    for y, name, x in ends:
        body.append(f'<text class="dlab" x="{x + 10:.1f}" y="{y + 3.5:.1f}">{html.escape(name)}</text>')
    return _legend([s[0] for s in series]) + _svg(f, body, label)


# ---------------------------------------------------------------------------
# HTML building blocks
# ---------------------------------------------------------------------------


def gk(text: str) -> str:
    """Keep lower-case Greek (σ, λ, ρ) lower case inside upper-cased labels: σ must not read as Σ."""
    return re.sub("([α-ω])", r'<span class="gk">\1</span>', text)


def table(headers, rows, note="", text=False):
    head = "".join(f"<th>{gk(html.escape(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    foot = f'<p class="note">{note}</p>' if note else ""
    cls = ' class="text"' if text else ""
    return f'<div class="scroll"><table{cls}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{foot}'


def figure(svg_html, title, caption):
    return (f'<figure><figcaption><b>{html.escape(title)}</b><span>{html.escape(caption)}</span></figcaption>'
            f"{svg_html}</figure>")


def multiples(title, caption, panels):
    inner = "".join(f'<div><div class="ptitle">{gk(html.escape(t))}</div>{c}</div>' for t, c in panels)
    return (f'<figure><figcaption><b>{html.escape(title)}</b><span>{html.escape(caption)}</span></figcaption>'
            f'<div class="multiples">{inner}</div></figure>')


def section(no: int, code: str, title: str, content: str) -> str:
    return (f'<section id="s{no}"><div class="rail"><div class="numeral">{no:02d}</div>'
            f'<div class="code">{code}</div><h2>{title}</h2></div><div class="body">{content}</div></section>')


def less_more(v: float) -> str:
    return "less" if v < 0 else "more"


HOLLOW_GAP = "Hollow marker: not resolved at 3 SE, drawn at its point estimate."
HOLLOW_SIGMA = "Hollow marker: not resolved at 3 SE, drawn at its 3-SE upper bound."


# ---------------------------------------------------------------------------
# Derived views of the JSON
# ---------------------------------------------------------------------------


class Study:
    """Indexes the four result files; every lookup reads the JSON as written."""

    def __init__(self, res: dict):
        self.s1 = res["01_sanity"]
        self.s2 = res["02_fixed_shares"]
        self.s3 = res["03_rebalance"]
        self.s4 = res["04_dispersion_correlation"]
        self.sweep1 = self.s2["sweep1_moneyness_maturity"]
        self.tenors = list(dict.fromkeys(r["tenor"] for r in self.sweep1))
        self.moneyness = sorted({r["moneyness"] for r in self.sweep1})
        self.base = self.find1("1Y", True, 1.0)
        self.five = self.s4["sweep5_corr_five_names"]
        self.two = self.s4["sweep5_corr_two_names"]
        self.lam = self.s4["sweep4_vol_dispersion"]
        self.freq = self.s3["sweep2_frequency"]
        self.drift = self.s3["sweep3_drift"]

    def find1(self, tenor, is_call, k):
        return next(r for r in self.sweep1 if r["tenor"] == tenor and r["is_call"] == is_call
                    and r["moneyness"] == k)

    def freq_row(self, rule, position):
        return next((r for r in self.freq if r["rebalance"] == rule and r["position"] == position), None)

    def rules(self):
        return list(dict.fromkeys(r["rebalance"] for r in self.freq))

    def records(self):
        """Every scenario record with a stage tag (96 in the committed run)."""
        out = [("01 control", self.s1["control"]), ("01 concentration", self.s1["concentration"])]
        out += [("02 sweep 1", r) for r in self.sweep1]
        out += [("02 sweep 6", r) for r in self.s2["sweep6_dividend_dispersion"]]
        out += [("03 sweep 2", r) for r in self.freq]
        out += [("03 sweep 3", r) for r in self.drift]
        out += [("04 sweep 4", r) for r in self.lam]
        out += [("04 sweep 5 · 5 names", r) for r in self.five]
        out += [("04 sweep 5 · 2 names", r) for r in self.two]
        return out


def vol_tilt(rec: dict) -> str:
    """Which way M1 moves delta between the highest- and lowest-vol names (from the JSON)."""
    vols, m1 = rec["vols"], rec["methods"]["M1"]
    hi = max(range(len(vols)), key=lambda i: vols[i])
    lo = min(range(len(vols)), key=lambda i: vols[i])
    if (m1["resolved"][hi] and m1["resolved"][lo]
            and m1["gap_cash_per_pct"][hi] < 0 < m1["gap_cash_per_pct"][lo]):
        return (f"M1 moves delta from the highest-vol name, {rec['names'][hi]} (σ {pct(vols[hi], 0)}), "
                f"to the lowest-vol name, {rec['names'][lo]} (σ {pct(vols[lo], 0)})")
    return "M1's split does not order cleanly by vol here"


def pv_skew(st: Study) -> dict:
    """Counts and the strike pattern of resolved MM-vs-MC PV gaps, all from the JSON."""
    allpv = [(tag, r, pv(r)) for tag, r in st.records()]
    res = [x for x in allpv if x[2]["resolved"]]
    above = [x for x in res if x[2]["err"] > 0]
    below = [x for x in res if x[2]["err"] < 0]
    below_k = sorted({r["moneyness"] for _, r, _ in below})
    below_types = sorted({"calls" if r["is_call"] else "puts" for _, r, _ in below})
    atm = [x for x in res if x[1]["moneyness"] == 1.0]
    return {
        "all": allpv, "res": res, "above": above, "below": below,
        "below_high_only": bool(below) and all(k > 1.0 for k in below_k),
        "below_kmin": below_k[0] if below_k else None,
        "below_types": below_types,
        "low_all_above": all(p["err"] > 0 for _, r, p in res if r["moneyness"] <= 1.0),
        "worst_atm": max(atm, key=lambda x: abs(x[2]["rel"])) if atm else None,
        # largest in CNY: relative errors on far-OTM wings are huge but worth a few CNY
        "max_over": max(res, key=lambda x: x[2]["err"]) if res else None,
        "max_under": min(res, key=lambda x: x[2]["err"]) if res else None,
    }


def pv_skew_text(sk: dict) -> tuple:
    """(count line, skew reading) — the reading appears only if the JSON shows the pattern."""
    line = (f"Of {len(sk['all'])} scenarios, {len(sk['res'])} PV gaps are resolved: MM is above MC in "
            f"{len(sk['above'])} and below in {len(sk['below'])}.")
    if sk["below_high_only"]:
        line += (f" Every case where MM is below MC is a high strike (K/I ≥ {num(sk['below_kmin'], 2)}, "
                 f"{' and '.join(sk['below_types'])}; a call and a put at one strike share the PV gap by parity, up to MC noise in the "
                 f"forward).")
    if sk["low_all_above"]:
        line += " At or below ATM, MM is above MC in every resolved case."
    read = ""
    if sk["below_high_only"] and sk["low_all_above"]:
        read = ("<p>So the error is a <em>skew</em>, not a level: a basket of lognormals with dispersed vols is "
                "more right-skewed than the lognormal matched to its first two moments. The fast model is rich "
                "on low strikes and downside puts and cheap on upside calls.</p>")
    return line, read


def pv_text(p: dict) -> str:
    if not p["resolved"]:
        return "not resolved"
    return f"{spct(p['rel'], 2)} ({signed(p['err'])} CNY)"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section_summary(st: Study) -> str:
    b = st.base
    m = b["methods"]
    tech, gold = name_idx(b, "TECH"), name_idx(b, "GOLD")
    starts = [(rule, st.freq_row(rule, "start")) for rule in st.rules()]
    drift_rows = [r for r in st.drift if r["tech_move"] != 1.0]
    interior = [r for r in st.five if r["corr_band_cash_per_pct"] is not None]
    band_max = max(max(r["corr_band_cash_per_pct"]) for r in interior)
    m1_gap_min = min(max(abs(g) for g in r["methods"]["M1"]["gap_cash_per_pct"]) for r in interior)
    m1_gap_max = max(max(abs(g) for g in r["methods"]["M1"]["gap_cash_per_pct"]) for r in interior)

    delta_rows = [
        ["M1 gap · TECH", f"{gap_cell(m['M1'], tech, rel=True)} of its delta",
         f"{gap_cell(m['M1'], tech)} CNY/1%"],
        ["M1 gap · GOLD", f"{gap_cell(m['M1'], gold, rel=True)} of its delta",
         f"{gap_cell(m['M1'], gold)} CNY/1%"],
        ["Σ M1 gaps over names", "total delta notional", f"{signed(sum(m['M1']['gap_cash_per_pct']))} CNY/1%"],
    ] + [[f"residual σ · {k}", METHOD_NOTES[k][0], f"{sigma_cell(m[k])} CNY/day"] for k in ("M1", "M3", "M4", "M5")]
    delta_tbl = table(["base 5-name, fixed shares, 1Y ATM call", "", "value"], delta_rows,
                      note="Gaps are method − REF in CNY per 1% move of the name, ± paired SE, on 1e8 CNY of "
                           "index notional. Residual σ: daily P&L σ of the mis-split, CNY.")

    reb_rows = [[rule, num(r["stub_years"] * TRADING_DAYS, 1), f"{sigma_cell(r['methods']['M1'])}"]
                for rule, r in starts if r is not None]
    reb_tbl = table(["rebalance", "stub, trading days", "M1 residual σ, CNY/day"], reb_rows,
                    note="Base basket, 1Y ATM call, at the start of the current period (sweep 2).")

    drift_tbl = table(["quarterly, mid-period", "M1 σ, CNY/day", "M2 σ, CNY/day", "M2 gap · TECH, CNY/1%"], [
        [lab(r["label"].split(", ")[-1]), sigma_cell(r["methods"]["M1"]), sigma_cell(r["methods"]["M2"]),
         gap_cell(r["methods"]["M2"], name_idx(r, "TECH"))] for r in drift_rows],
        note="Sweep 3: TECH has drifted since the last rebalance, so target weights ≠ current weights.")

    sk = pv_skew(st)
    bp = pv(b)
    pv_rows = [["base 5-name 1Y ATM call", num(bp["mc"]) + f' <span class="dim">±{num(bp["se"])}</span>',
                num(bp["mm"]), pv_text(bp)]]
    picks = []
    if sk["worst_atm"]:
        picks.append(("worst ATM case", sk["worst_atm"]))
    neg5 = min(st.five, key=lambda r: r["rho"])
    neg2 = min(st.two, key=lambda r: r["rho"])
    picks += [("5 names, lowest ρ", ("", neg5, pv(neg5))), ("2 names, lowest ρ", ("", neg2, pv(neg2)))]
    long_t = st.tenors[-1]
    lo_put = st.find1(long_t, False, st.moneyness[0])
    hi_call = st.find1(long_t, True, st.moneyness[-1])
    picks += [("lowest-strike put", ("", lo_put, pv(lo_put))), ("highest-strike call", ("", hi_call, pv(hi_call)))]
    if sk["max_over"]:
        picks.append(("largest MM overprice, CNY", sk["max_over"]))
    if sk["max_under"]:
        picks.append(("largest MM underprice, CNY", sk["max_under"]))
    seen = {b["label"]}
    for tag, (_, r, p) in picks:
        if r["label"] in seen:
            continue
        seen.add(r["label"])
        pv_rows.append([f"{tag} · {lab(r['label'])}",
                        num(p["mc"]) + f' <span class="dim">±{num(p["se"])}</span>', num(p["mm"]), pv_text(p)])
    pv_tbl = table(["PV on 1e8 CNY notional", "MC ± SE, CNY", "MM, CNY", "MM − MC"], pv_rows,
                   note="MM − MC is resolved only when |MM − MC| > 3 MC SE; CNY = (MM − MC) × units.")

    skew_line, skew_read = pv_skew_text(sk)
    band_verdict = (f" The smallest of those gaps is {num(m1_gap_min / band_max, 0)}× the widest band: the method gap "
                    f"is not a correlation-assumption artefact." if m1_gap_min > band_max else "")

    m1_none = st.freq_row("none", "start")["methods"]["M1"]
    m1_daily = st.freq_row("daily", "start")["methods"]["M1"]
    m1_weekly = st.freq_row("weekly", "start")["methods"]["M1"]
    worst_m5 = _worst_m5(st)
    m2_sig = [r["methods"]["M2"]["residual_sigma_cny"] for r in drift_rows if r["methods"]["M2"]["residual_resolved"]]
    m1_sig = [r["methods"]["M1"]["residual_sigma_cny"] for r in drift_rows if r["methods"]["M1"]["residual_resolved"]]
    p2 = pv(neg2)
    desk = f"""
<ul class="points">
<li><b>(a) Never split by target weights mid-period.</b> After TECH drifts ±30% inside a quarterly period, the
target-weight split (M2) carries a residual σ of {num(min(m2_sig)) if m2_sig else "–"}–{num(max(m2_sig)) if m2_sig else "–"}
CNY/day against {num(min(m1_sig)) if m1_sig else "–"}–{num(max(m1_sig)) if m1_sig else "–"} for the whole-index order (M1).</li>
<li><b>(b) The whole-index order (M1) is adequate when the stub is short.</b> M1's residual σ is
{sigma_text(m1_daily)} CNY/day under daily and {sigma_text(m1_weekly)} under weekly rebalance, against
{sigma_text(m1_none)} on fixed shares.</li>
<li><b>(c) For fixed-share or long-period CSIs, hedge per name from a multi-name model (M3 or REF).</b> On the base
1Y ATM call {vol_tilt(b)}: M1 puts {num(abs(m['M1']['gap_cash_per_pct'][tech]))} CNY/1% {less_more(m['M1']['gap_cash_per_pct'][tech])}
delta on TECH than the reference. M3's residual is {sigma_text(m['M3'])} CNY/day. The total delta notional is the
same under M1, M2 and REF (Euler; the M1 gaps sum to {signed(sum(m['M1']['gap_cash_per_pct']))} CNY/1%): only the split differs.</li>
<li><b>(d) The moment-matched per-name model (M5) is a cheap middle ground.</b> Its residual σ on the base case is
{sigma_text(m['M5'])} CNY/day against M1's {sigma_text(m['M1'])}. It inherits the MM model's own Δ_I error (section 8).
Among resolved M5 gaps in sweep 1, {worst_m5}.</li>
<li><b>(e) Quote PV off the multi-name MC.</b> The fast single-index (moment-matched) model is not safe for quoting
baskets that mix equities with hedge-like or negatively correlated assets: it is {pv_text(bp)} off on the base ATM
call and {pv_text(p2)} on the EQ/HEDGE basket at ρ = {sgn(neg2['rho'], 1)}.</li>
</ul>"""

    return section(1, "SUMMARY", "Desk read", f"""
<p class="lead">On a fixed-share CSI the whole-index order mis-splits delta between names: on the base 1Y ATM call it
puts {num(abs(m['M1']['gap_cash_per_pct'][tech]))} CNY per 1% {less_more(m['M1']['gap_cash_per_pct'][tech])} delta on
TECH and {num(abs(m['M1']['gap_cash_per_pct'][gold]))} {less_more(m['M1']['gap_cash_per_pct'][gold])} on GOLD than the
reference, a residual of {sigma_text(m['M1'])} CNY a day. The gap lives only in the current stub: daily rebalancing
cuts it to {sigma_text(m1_daily)}.</p>
<h3 class="sub">Delta split · base case</h3>
{delta_tbl}
<h3 class="sub">Rebalance frequency</h3>
{reb_tbl}
<h3 class="sub">Target-weight shortcut after drift</h3>
{drift_tbl}
<h3 class="sub">Correlation model risk vs method gap</h3>
<p>Across the interior uniform-ρ points (5 names), moving ρ by ±{num(st.s4['band_shift'], 1)} moves the reference
per-name delta by at most {num(band_max)} CNY/1%, while the largest M1 gap per scenario runs
{num(m1_gap_min)}–{num(m1_gap_max)} CNY/1%.{band_verdict}</p>
<h3 class="sub">PV · fast model vs multi-name MC</h3>
<p>{skew_line}</p>
{skew_read}
{pv_tbl}
<h3 class="sub">What the desk should do</h3>
{desk}
""")


def _worst_m5(st: Study) -> str:
    """Largest resolved M5 gap in sweep 1, relative and in CNY."""
    cand = [(r, i) for r in st.sweep1 for i, ok in enumerate(r["methods"]["M5"]["resolved"]) if ok]
    if not cand:
        return "not resolved anywhere"

    def say(r, i):
        m5 = r["methods"]["M5"]
        return (f"{spct(m5['gap_rel_to_ref'][i])} of the reference delta on {r['names'][i]} "
                f"({signed(m5['gap_cash_per_pct'][i])} CNY/1%) on the {lab(r['label'])}")

    rel = max(cand, key=lambda x: abs(x[0]["methods"]["M5"]["gap_rel_to_ref"][x[1]]))
    cny = max(cand, key=lambda x: abs(x[0]["methods"]["M5"]["gap_cash_per_pct"][x[1]]))
    return f"the largest relative gap is {say(*rel)}; the largest in cash is {say(*cny)}"


def section_setup(st: Study) -> str:
    b = st.base
    basket = table(["name", "target weight", "σ", "q"], [
        [n, pct(w, 0), pct(v, 0), pct(q, 1)]
        for n, w, v, q in zip(b["names"], b["target_weights"], b["vols"], b["div_yields"])])
    corr = table([""] + b["names"], [[n] + [num(c, 2) for c in row] for n, row in zip(b["names"], b["corr"])])
    methods = table(["method", "what it is", "how"], [[k, html.escape(METHOD_NOTES[k][0]), html.escape(METHOD_NOTES[k][1])]
                                                     for k in ("REF",) + METHODS], text=True)
    notional = b["units"] * b["index_level"]
    return section(2, "SETUP", "Basket, methods and metrics", f"""
<p class="lead">A stylised 5-name customized strategy index (synthetic, no desk data), European options struck today on
{num(notional)} CNY of index notional. Prices and deltas from {b['n_paths']:,} paths × {b['n_batches']} RQMC batches per
scenario.</p>
<dl class="spec">
<dt>Index</dt><dd>Price return. Between rebalances I(t) = Σ nᵢSᵢ(t); at a rebalance nᵢ resets to wᵢ·I/Sᵢ. Rule A
(fixed shares) has no rebalance in the option's life.</dd>
<dt>Dynamics</dt><dd>Multi-name GBM, per-name vol σᵢ and yield qᵢ, flat rate, correlation ρ below. Exact sampling at
the stub end, each rebalance date and expiry.</dd>
<dt>Product</dt><dd>European call or put on I(T), K = moneyness × I(now), units = notional / I(now).</dd>
</dl>
{basket}
<h3 class="sub">Correlation, base basket</h3>
{corr}
<p class="note">Large caps and the ETF are the equity block; GOLD is the hedge-like name.</p>
<h3 class="sub">Methods</h3>
{methods}
<h3 class="sub">Metrics</h3>
<dl class="spec">
<dt>Gap, CNY/1%</dt><dd>(Δᵢᵐ − Δᵢʳᵉᶠ) · Sᵢ · units · 1%: the hedge error per 1% move of name i.</dd>
<dt>Paired SE</dt><dd>SE across RQMC batches of the difference itself (same paths), not root-sum-of-squares.</dd>
<dt>3-SE rule</dt><dd>A gap is resolved only if |gap| &gt; 3 paired SE (plus the FD bias allowance for M3).
Everything else prints as <span class="nr">not resolved</span>.</dd>
<dt>Residual σ</dt><dd>√(eᵀΣe) with eᵢ the cash-delta error and Σ the daily return covariance (244-day year):
the daily P&amp;L σ of using method m instead of REF. Bias-corrected (Q = ēᵀΣē − tr(Σ·Cov ē)), resolved only if
Q &gt; 3·SE(Q); unresolved values show their 3-SE upper bound.</dd>
<dt>PV error</dt><dd>(MM − MC) × units in CNY, resolved when |MM − MC| &gt; 3 MC SE (MM is deterministic).</dd>
</dl>
""")


def section_sanity(st: Study) -> str:
    s1 = st.s1
    sn = s1["single_name"]
    sn_rows = [[lab(r["case"]), "call" if r["is_call"] else "put", num(r["bs_price"], 3), sci(r["mm_price_rel_err"]),
                num(r["mc_price"], 3) + f' <span class="dim">±{num(r["mc_price_se"], 4)}</span>', sgn(r["mc_price_z"]),
                num(r["bs_delta"], 4), num(r["mm_delta"], 4), num(r["mc_delta"], 4), sgn(r["mc_delta_z"])] for r in sn]
    max_rel = max(abs(r["mm_price_rel_err"]) for r in sn)
    max_z = max(max(abs(r["mc_price_z"]), abs(r["mc_delta_z"])) for r in sn)

    mo = s1["moments"]
    mo_rows = []
    for r in mo:
        reps = "; ".join(f"seed {x['seed']}: {sgn(x['m1_z'])} / {sgn(x['m2_z'])}" for x in r["replicates"])
        mo_rows.append([lab(r["case"]), num(r["m1_closed"], 3), num(r["m1_mc"], 3), sgn(r["m1_z"]),
                        num(r["m2_closed"], 0), num(r["m2_mc"], 0), sgn(r["m2_z"]), sci(r["euler_max_rel_err"]),
                        reps or '<span class="dim">–</span>'])
    flagged = [r for r in mo if max(abs(r["m1_z"]), abs(r["m2_z"])) > SE_MULTIPLE]
    rep_z = [max(abs(x["m1_z"]), abs(x["m2_z"])) for r in flagged for x in r["replicates"]]
    flag_txt = (f"{len(flagged)} of {len(mo)} rows exceed 3 SE; re-run on {len(flagged[0]['replicates'])} independent "
                f"seeds each, the largest replicate |z| is {num(max(rep_z), 2)}, so the flag reads as chance."
                if flagged and rep_z else f"No row exceeds 3 SE ({len(mo)} rows).")

    eu = s1["euler"]
    ctrl, conc = s1["control"], s1["concentration"]
    ctrl_resolved = sum(sum(ctrl["methods"][k]["resolved"]) for k in METHODS)
    ctrl_rows = [[n, num(ctrl["ref_cash_per_pct"][i])] + [gap_cell(ctrl["methods"][k], i) for k in METHODS]
                 for i, n in enumerate(ctrl["names"])]
    heavy = max(range(len(conc["names"])), key=lambda i: conc["target_weights"][i])
    light = min(range(len(conc["names"])), key=lambda i: conc["target_weights"][i])
    cm1 = conc["methods"]["M1"]
    conc_rows = [[n, pct(conc["target_weights"][i], 0), num(conc["ref_cash_per_pct"][i]), gap_cell(cm1, i),
                  gap_cell(cm1, i, rel=True), gap_cell(conc["methods"]["M3"], i)] for i, n in enumerate(conc["names"])]
    return section(3, "SANITY", "The pricer reproduces what it must", f"""
<p class="lead">On one name the moment-matched model equals Black–Scholes to {sci(max_rel)} relative, and the MC sits
within |z| ≤ {num(max_z, 2)} of it. Per path, Σ Sᵢ·∂I_T/∂Sᵢ = I_T to {sci(eu['euler_max_rel_err_per_path'])}.</p>
<h3 class="sub">Single name vs Black–Scholes · per index unit</h3>
{table(["case", "type", "BS price", "MM/BS − 1", "MC price ± SE", "z", "BS Δ", "MM Δ", "MC Δ", "z"], sn_rows)}
<h3 class="sub">Moments of I_T · MC sample vs closed form</h3>
<p>{flag_txt}</p>
{table(["case", "E[I] closed", "E[I] MC", "z", "E[I²] closed", "E[I²] MC", "z", "Euler / path", "replicates z₁ / z₂"], mo_rows)}
<h3 class="sub">Euler identity · {lab(eu['case'])}</h3>
{table(["check", "value"], [["max over paths |Σ Sᵢ∂I_T/∂Sᵢ − I_T| / I_T", sci(eu["euler_max_rel_err_per_path"])],
                           ["Σ Sᵢ·Δᵢʳᵉᶠ", num(eu["sum_S_delta"], 6)], ["I·Δ_I", num(eu["I_delta_index"], 6)]])}
<h3 class="sub">Exchangeable control · all methods agree</h3>
<p>{lab(ctrl['label'])}: {ctrl_resolved} of {5 * len(ctrl['names'])} method gaps resolved.</p>
{table(["name", "REF, CNY/1%"] + list(METHODS), ctrl_rows)}
<h3 class="sub">Concentration · equal vols, unequal weights</h3>
<p>{lab(conc['label'])}. Identical vols, but M1 puts {num(abs(cm1['gap_cash_per_pct'][heavy]))} CNY/1%
{less_more(cm1['gap_cash_per_pct'][heavy])} delta on the heaviest name, {conc['names'][heavy]}
(w {pct(conc['target_weights'][heavy], 0)}), and {num(abs(cm1['gap_cash_per_pct'][light]))}
{less_more(cm1['gap_cash_per_pct'][light])} on the lightest, {conc['names'][light]}: a heavier name carries more delta
per share. M1 residual σ {sigma_text(cm1)} CNY/day.</p>
{table(["name", "w", "REF, CNY/1%", "M1 gap", "M1 gap / REF", "M3 gap"], conc_rows)}
""")


def section_fixed(st: Study) -> str:
    ks = st.moneyness
    klabels = [f"{k:.2f}" for k in ks]
    b = st.base
    tech, gold = name_idx(b, "TECH"), name_idx(b, "GOLD")
    allv = [100 * r["methods"]["M1"]["gap_rel_to_ref"][i] for r in st.sweep1 if r["is_call"] for i in (tech, gold)]
    dom = (min(allv), max(allv))
    panels = []
    for t in st.tenors:
        rows = [st.find1(t, True, k) for k in ks]
        ser, hol = [], []
        for nm, i in (("TECH", tech), ("GOLD", gold)):
            ser.append((nm, [100 * r["methods"]["M1"]["gap_rel_to_ref"][i] for r in rows]))
            hol.append([not r["methods"]["M1"]["resolved"][i] for r in rows])
        panels.append((f"{t} CALL", line_chart(klabels, ser, hollow=hol, ylabel="M1 GAP, % OF REF Δ",
                                                xlabel="K / I(NOW)", yfmt=lambda v: num(v, 0),
                                                tipfmt=lambda v: sgn(v, 1) + "%", zero=True, ydomain=dom,
                                                width=380, height=260, label=f"M1 gap {t}")))
    grid_rows = []
    for t in st.tenors:
        for k in ks:
            c, p = st.find1(t, True, k), st.find1(t, False, k)
            grid_rows.append([t, f"{k:.2f}"] + [gap_cell(r["methods"]["M1"], i, rel=True)
                                                for r in (c, p) for i in (tech, gold)]
                             + [gap_cell(c["methods"]["M1"], tech), gap_cell(p["methods"]["M1"], tech)])
    one_y = [st.find1("1Y", True, k) for k in ks]
    sig_ser, sig_hol = [], []
    for key in ("M1", "M4", "M5"):
        pts = [sigma_point(r["methods"][key]) for r in one_y]
        sig_ser.append((key, [v for v, _ in pts]))
        sig_hol.append([h for _, h in pts])
    sig_chart = line_chart(klabels, sig_ser, hollow=sig_hol, ylabel="RESIDUAL σ, CNY/DAY", xlabel="K / I(NOW) · 1Y CALL",
                           yfmt=compact, tipfmt=lambda v: num(v) + " CNY/day", ydomain=(0, max(max(s) for _, s in sig_ser)),
                           label="Residual sigma by method, 1Y")
    sig_rows = [[f"{k:.2f}"] + [sigma_cell(r["methods"][key]) for key in METHODS] for k, r in zip(ks, one_y)]
    m3_res = sum(sum(r["methods"]["M3"]["resolved"]) for r in st.sweep1)
    m3_tot = sum(len(r["methods"]["M3"]["resolved"]) for r in st.sweep1)
    bt = b["methods"]["M1"]
    atm_by_tenor = ", ".join(f"{t} {gap_cell(st.find1(t, True, 1.0)['methods']['M1'], tech, rel=True)}"
                             for t in st.tenors)
    return section(4, "FIXED", "Fixed shares: where the split goes wrong", f"""
<p class="lead">With no rebalance in the option's life the weights drift with prices, and splitting Δ_I by current
shares misallocates: on the 1Y ATM call M1 is {gap_cell(bt, tech, rel=True)} off on TECH and
{gap_cell(bt, gold, rel=True)} on GOLD ({gap_cell(bt, tech)} and {gap_cell(bt, gold)} CNY/1%). At the money the TECH
gap by tenor is {atm_by_tenor}. Per-name FD on the multi-name MC (M3) is resolved in {m3_res} of {m3_tot}
name-scenarios.</p>
{multiples("M1 gap as % of the reference delta, calls", "Shared y-axis. " + HOLLOW_GAP, panels)}
{table(["tenor", "K/I", "TECH call", "GOLD call", "TECH put", "GOLD put", "TECH call, CNY/1%", "TECH put, CNY/1%"],
       grid_rows, "M1 gap / REF delta, sweep 1, base basket, rule A. A put's reference delta is negative, so its % "
       "carries the opposite sign to its cash gap.")}
{figure(sig_chart, "Residual daily P&L σ by method, 1Y calls", "M2 = M1 here (no drift). " + HOLLOW_SIGMA)}
{table(["K/I"] + [f"{k} σ, CNY/day" for k in METHODS], sig_rows)}
""")


def section_rebalance(st: Study) -> str:
    rules = st.rules()
    positions = ["start", "mid", "end"]
    ser, hol = [], []
    for pos in positions:
        pts = [st.freq_row(rule, pos) for rule in rules]
        vals = [sigma_point(r["methods"]["M1"]) if r else (None, False) for r in pts]
        ser.append((pos, [v for v, _ in vals]))
        hol.append([h for _, h in vals])
    chart = line_chart(rules, ser, hollow=hol, logy=True, ylabel="M1 RESIDUAL σ, CNY/DAY (LOG)",
                       xlabel="REBALANCE RULE · POSITION IN CURRENT PERIOD", yfmt=compact,
                       tipfmt=lambda v: num(v) + " CNY/day", label="M1 residual sigma vs rebalance")
    per_day = [r["methods"]["M1"]["residual_sigma_cny"] / (r["stub_years"] * TRADING_DAYS)
               for r in st.freq if r["methods"]["M1"]["residual_resolved"]]
    rows = []
    for r in st.freq:
        tech = name_idx(r, "TECH")
        rows.append([r["rebalance"], r["position"], num(r["stub_years"] * TRADING_DAYS, 2),
                     sigma_cell(r["methods"]["M1"]), gap_cell(r["methods"]["M1"], tech),
                     sigma_cell(r["methods"]["M3"]), sigma_cell(r["methods"]["M5"])])
    drows = []
    for r in st.drift:
        tech = name_idx(r, "TECH")
        drows.append([sgn(100 * (r["tech_move"] - 1), 0) + "%", pct(r["current_weights"][tech], 1),
                      pct(r["target_weights"][tech], 0), sigma_cell(r["methods"]["M1"]),
                      sigma_cell(r["methods"]["M2"]), gap_cell(r["methods"]["M1"], tech),
                      gap_cell(r["methods"]["M2"], tech), gap_cell(r["methods"]["M2"], tech, rel=True)])
    return section(5, "REBAL", "Rebalancing switches the gap off", f"""
<p class="lead">A bump to one name moves only the current stub; every later period restarts from the target weights.
So M1's residual σ scales with the stub: {num(min(per_day))}–{num(max(per_day))} CNY/day per trading day of stub across
every resolved row, from {sigma_text(st.freq_row('none', 'start')['methods']['M1'])} on fixed shares to
{sigma_text(st.freq_row('daily', 'start')['methods']['M1'])} under daily rebalance.</p>
{figure(chart, "M1 residual σ vs rebalance rule", "Base basket, 1Y ATM call. Log scale. " + HOLLOW_SIGMA)}
{table(["rule", "position", "stub, days", "M1 σ, CNY/day", "M1 gap · TECH, CNY/1%", "M3 σ", "M5 σ"], rows)}
<h3 class="sub">Drift since the last rebalance · target-weight shortcut</h3>
<p>Quarterly rule, mid-period, TECH moved since the last rebalance. Current weights now differ from target weights, so
the target-weight split (M2) stops being the whole-index order.</p>
{table(["TECH move", "TECH weight now", "target", "M1 σ", "M2 σ", "M1 gap · TECH", "M2 gap · TECH", "M2 / REF"], drows,
       "σ in CNY/day; gaps in CNY/1%.")}
""")


def section_drivers(st: Study) -> str:
    lam = st.lam
    ser, hol = [], []
    for key in ("M1", "M4", "M5"):
        pts = [sigma_point(r["methods"][key]) for r in lam]
        ser.append((key, [v for v, _ in pts]))
        hol.append([h for _, h in pts])
    lam_chart = line_chart([num(r["lambda"], 1) for r in lam], ser, hollow=hol, ylabel="RESIDUAL σ, CNY/DAY",
                           xlabel="VOL DISPERSION λ (LOG-SPACE SCALE)", yfmt=compact,
                           tipfmt=lambda v: num(v) + " CNY/day", ydomain=(0, max(max(s) for _, s in ser)),
                           label="Residual sigma vs vol dispersion")
    lam_rows = [[num(r["lambda"], 1), f"{pct(min(r['vols']), 1)}–{pct(max(r['vols']), 1)}",
                 num(max_abs_resolved(r["methods"]["M1"])[0])] + [sigma_cell(r["methods"][k]) for k in METHODS]
                for r in lam]

    def rho_chart(rows, idx_name, xl, aria):
        s, h = [], []
        for key in ("M1", "M5"):
            if idx_name is None:
                pts = [max_abs_resolved(r["methods"][key]) for r in rows]
            else:
                pts = []
                for r in rows:
                    i = name_idx(r, idx_name)
                    pts.append((abs(r["methods"][key]["gap_cash_per_pct"][i]), not r["methods"][key]["resolved"][i]))
            s.append((f"|{key} gap|", [v for v, _ in pts]))
            h.append([x for _, x in pts])
        band = []
        for r in rows:
            bnd = r["corr_band_cash_per_pct"]
            band.append(None if bnd is None else (max(bnd) if idx_name is None else bnd[name_idx(r, idx_name)]))
        s.append(("ρ ± band", band))
        h.append([False] * len(rows))
        top = max(v for _, vs in s for v in vs if v is not None)
        return line_chart([sgn(r["rho"], 2) for r in rows], s, hollow=h, ylabel="CNY PER 1%", xlabel=xl,
                          yfmt=compact, tipfmt=lambda v: num(v) + " CNY/1%", ydomain=(0, top), label=aria)

    five_chart = rho_chart(st.five, None, "UNIFORM ρ · 5 NAMES", "Gap vs correlation band, five names")
    two_chart = rho_chart(st.two, "EQ", "ρ · EQ/HEDGE", "Gap vs correlation band, two names")

    band_rows = []
    for r in st.five + st.two:
        bnd = r["corr_band_cash_per_pct"]
        for i, n in enumerate(r["names"]):
            m1 = r["methods"]["M1"]
            if bnd is None:
                verdict = '<span class="dim">no band (ρ ± shift infeasible)</span>'
            elif not m1["resolved"][i]:
                verdict = NR
            elif m1["within_corr_band"][i]:
                verdict = "within correlation model risk"
            else:
                verdict = "outside the band"
            band_rows.append([lab(r["label"]), n, gap_cell(m1, i), gap_cell(r["methods"]["M3"], i),
                              gap_cell(r["methods"]["M5"], i), "–" if bnd is None else num(bnd[i]), verdict])
    five_rows = [[sgn(r["rho"], 2), sigma_cell(r["methods"]["M1"]), num(max_abs_resolved(r["methods"]["M1"])[0]),
                  num(max_abs_resolved(r["methods"]["M5"])[0]),
                  "–" if r["corr_band_cash_per_pct"] is None else num(max(r["corr_band_cash_per_pct"]))]
                 for r in st.five]
    two_rows = [[sgn(r["rho"], 1), sigma_cell(r["methods"]["M1"]), gap_cell(r["methods"]["M1"], 0),
                 gap_cell(r["methods"]["M1"], 1), gap_cell(r["methods"]["M5"], 0),
                 "–" if r["corr_band_cash_per_pct"] is None else num(r["corr_band_cash_per_pct"][0])]
                for r in st.two]
    lo_l, hi_l = lam[0], lam[-1]
    two_lo, two_hi = min(st.two, key=lambda r: r["rho"]), max(st.two, key=lambda r: r["rho"])
    eq = name_idx(two_lo, "EQ")
    five_lo, five_hi = min(st.five, key=lambda r: r["rho"]), max(st.five, key=lambda r: r["rho"])
    five_sig = [r["methods"]["M1"]["residual_sigma_cny"] for r in st.five if r["methods"]["M1"]["residual_resolved"]]
    within = sum(sum(r["methods"]["M1"].get("within_corr_band") or []) for r in st.five + st.two)
    return section(6, "DRIVERS", "Vol dispersion and correlation", f"""
<p class="lead">Vol dispersion drives the mis-split: M1's residual σ goes from {sigma_text(lo_l['methods']['M1'])}
CNY/day at λ = {num(lo_l['lambda'], 1)} (only concentration and yield dispersion left) to
{sigma_text(hi_l['methods']['M1'])} at λ = {num(hi_l['lambda'], 1)}. For 5 names under uniform ρ from
{sgn(five_lo['rho'], 2)} to {sgn(five_hi['rho'], 2)} it stays within {num(min(five_sig))}–{num(max(five_sig))} CNY/day.
On a two-name equity/hedge basket the M1 gap on EQ is {gap_cell(two_hi['methods']['M1'], eq)} CNY/1% at
ρ = {sgn(two_hi['rho'], 1)} and {gap_cell(two_lo['methods']['M1'], eq)} at ρ = {sgn(two_lo['rho'], 1)}.</p>
{figure(lam_chart, "Residual σ vs vol dispersion", "Base basket, rule A, 1Y ATM call; λ = 1 is the base vols. " + HOLLOW_SIGMA)}
{table(["λ", "vol range", "max |M1 gap|, CNY/1%"] + [f"{k} σ" for k in METHODS], lam_rows, "σ in CNY/day.")}
<h3 class="sub">Correlation, including the extremes</h3>
<p>The band is how far the reference delta itself moves when ρ shifts by ±{num(st.s4['band_shift'], 1)} on the same
paths (|REF(ρ+) − REF(ρ−)| / 2). A gap inside it is within correlation model risk. {within} of the M1 per-name gaps
fall inside their band.</p>
{multiples("Method gap vs correlation band", "Left: max over names. Right: the EQ leg. " + HOLLOW_GAP,
           [("5 NAMES · MAX OVER NAMES", five_chart), ("EQ/HEDGE · EQ", two_chart)])}
{table(["ρ (5 names)", "M1 σ, CNY/day", "max |M1 gap|", "max |M5 gap|", "max band"], five_rows,
       "Gaps and band in CNY/1%. ρ = −0.25 is the PSD floor and ρ = 1 is singular: no feasible ± shift, no band.")}
{table(["ρ (EQ/HEDGE)", "M1 σ, CNY/day", "M1 gap · EQ", "M1 gap · HEDGE", "M5 gap · EQ", "band · EQ"], two_rows)}
<h3 class="sub">Band next to the method gaps, per name</h3>
{table(["scenario", "name", "M1 gap (= M2)", "M3 gap", "M5 gap", "band", "M1 verdict"], band_rows,
       "CNY/1%. M2 equals M1 at a fresh state (no drift).")}
""")


def section_yields(st: Study) -> str:
    rows6 = st.s2["sweep6_dividend_dispersion"]
    names = rows6[0]["names"]
    rows = [[html.escape(r["yield_variant"]), num(r["moneyness"], 1)]
            + [gap_cell(r["methods"]["M1"], i) for i in range(len(names))]
            + [sigma_cell(r["methods"]["M1"]), sigma_cell(r["methods"]["M3"])] for r in rows6]
    deep = [r for r in rows6 if r["moneyness"] == min(x["moneyness"] for x in rows6)]
    eq = next(r for r in deep if r["yield_variant"].startswith("equal"))
    dis = next(r for r in deep if not r["yield_variant"].startswith("equal"))
    return section(7, "YIELDS", "Dividend-yield dispersion", f"""
<p class="lead">A deep in-the-money call (K/I = {num(eq['moneyness'], 1)}) is almost a forward, and a forward's
per-name delta is nᵢe^(−qᵢT), not proportional to nᵢ. With identical vols and equal yields M1's residual is
{sigma_text(eq['methods']['M1'])} CNY/day; with the base yields it is {sigma_text(dis['methods']['M1'])}. Even the
linear part of the delta splits differently when yields differ.</p>
{table(["yields", "K/I"] + [f"M1 · {n}" for n in names] + ["M1 σ", "M3 σ"], rows,
       "Equal vols 25%, uniform ρ 0.5, base weights, rule A, 1Y call. Gaps CNY/1% ± paired SE; σ CNY/day.")}
""")


def section_mm(st: Study) -> str:
    sk = pv_skew(st)
    ks = st.moneyness
    klabels = [f"{k:.2f}" for k in ks]
    errs = [pv(r)["err"] for r in st.sweep1]
    dom = (min(errs), max(errs))
    panels = []
    for t in st.tenors:
        ser, hol = [], []
        for kind, call in (("call", True), ("put", False)):
            ps = [pv(st.find1(t, call, k)) for k in ks]
            ser.append((kind, [p["err"] for p in ps]))
            hol.append([not p["resolved"] for p in ps])
        panels.append((f"{t}", line_chart(klabels, ser, hollow=hol, ylabel="MM − MC PV, CNY", xlabel="K / I(NOW)",
                                          yfmt=compact, tipfmt=lambda v: sgn(v, 0) + " CNY", zero=True, ydomain=dom,
                                          width=380, height=260, label=f"PV error {t}")))

    def rel_chart(rows, xlabs, name, xl, aria):
        ps = [pv(r) for r in rows]
        return line_chart(xlabs, [(name, [100 * p["rel"] for p in ps])], hollow=[[not p["resolved"] for p in ps]],
                          ylabel="(MM − MC) / MC, %", xlabel=xl, yfmt=lambda v: num(v, 1),
                          tipfmt=lambda v: sgn(v, 2) + "%", zero=True, width=380, height=260, label=aria)

    rho_panels = [("5 NAMES · UNIFORM ρ", rel_chart(st.five, [sgn(r["rho"], 2) for r in st.five], "5 names", "ρ",
                                                     "PV error vs rho, five names")),
                  ("EQ/HEDGE", rel_chart(st.two, [sgn(r["rho"], 1) for r in st.two], "EQ/HEDGE", "ρ",
                                         "PV error vs rho, two names"))]
    one_y = [(st.find1("1Y", True, k), st.find1("1Y", False, k)) for k in ks]
    one_y_chart = line_chart(klabels, [("call", [100 * pv(c)["rel"] for c, _ in one_y]),
                                       ("put", [100 * pv(p)["rel"] for _, p in one_y])],
                             hollow=[[not pv(c)["resolved"] for c, _ in one_y], [not pv(p)["resolved"] for _, p in one_y]],
                             ylabel="(MM − MC) / MC, %", xlabel="K / I(NOW) · 1Y", yfmt=lambda v: num(v, 1),
                             tipfmt=lambda v: sgn(v, 2) + "%", zero=True, width=380, height=260, label="PV error 1Y")
    lam_panels = [("VOL DISPERSION λ · 1Y ATM CALL", rel_chart(st.lam, [num(r["lambda"], 1) for r in st.lam], "MM vs MC",
                                                               "λ", "PV error vs dispersion")),
                  ("MONEYNESS · 1Y", one_y_chart)]

    rows = []
    for tag, r in st.records():
        p = pv(r)
        di = r["delta_index"]
        dd = di["mm"] - di["mc"]
        tc = r["total_delta_cash_per_pct"]
        td = tc["mm"] - tc["mc"]
        rows.append([
            html.escape(tag), lab(r["label"]),
            num(p["mc"]) + f' <span class="dim">±{num(p["se"])}</span>', num(p["mm"]),
            spct(p["rel"], 2) if p["resolved"] else NR, signed(p["err"]) if p["resolved"] else NR,
            num(100 * di["mc"], 2) + f' <span class="dim">±{num(100 * di["mc_se"], 2)}</span>', num(100 * di["mm"], 2),
            signed(100 * dd, 2) if abs(dd) > SE_MULTIPLE * di["mc_se"] else NR,
            signed(td) if abs(td) > SE_MULTIPLE * tc["mc_se"] else NR,
        ])
    five_lo, five_hi = min(st.five, key=lambda r: r["rho"]), max(st.five, key=lambda r: r["rho"])
    lam_lo, lam_hi = st.lam[0], st.lam[-1]
    b = st.base
    btd = b["total_delta_cash_per_pct"]
    trend = (f"On the 1Y ATM call: 5 names {pv_text(pv(five_hi))} at ρ = {sgn(five_hi['rho'], 2)} and "
             f"{pv_text(pv(five_lo))} at ρ = {sgn(five_lo['rho'], 2)}; vol dispersion {pv_text(pv(lam_lo))} at "
             f"λ = {num(lam_lo['lambda'], 1)} and {pv_text(pv(lam_hi))} at λ = {num(lam_hi['lambda'], 1)}.")
    skew, skew_read = pv_skew_text(sk)
    return section(8, "MM", "PV and Δ_I: fast model vs multi-name MC", f"""
<p class="lead">The moment-matched lognormal is the desk-style fast single-index model. On the base 1Y ATM call it
misprices PV by {pv_text(pv(b))} and its total delta by {signed(btd['mm'] - btd['mc'])} CNY per 1% index move.
{skew} {trend}</p>
{skew_read}
{multiples("MM − MC PV by moneyness and tenor", "CNY on the scenario notional; shared y-axis. Base basket, rule A. Call and put lines "
           "overlap: MM matches the forward, so by put–call parity they share the PV gap. " + HOLLOW_GAP, panels)}
{multiples("Relative PV error vs correlation", "1Y ATM call, rule A. " + HOLLOW_GAP, rho_panels)}
{multiples("Relative PV error vs vol dispersion and moneyness", "Rule A. " + HOLLOW_GAP, lam_panels)}
<h3 class="sub">Every scenario</h3>
{table(["stage", "scenario", "MC PV ± SE, CNY", "MM PV, CNY", "PV err %", "PV err, CNY", "MC Δ_I ± SE, %", "MM Δ_I, %",
        "Δ_I MM − MC, pts", "total Δ cash MM − MC, CNY/1%"], rows,
       "PV on the scenario notional. Δ_I is per index unit, in %. A difference is shown only when it clears 3 MC SE; MM is deterministic.")}
""")


def section_caveats(st: Study) -> str:
    total = resolved = 0
    for _, r in st.records():
        for k in METHODS:
            total += len(r["methods"][k]["resolved"])
            resolved += sum(r["methods"][k]["resolved"])
    p_null = math.erfc(SE_MULTIPLE / math.sqrt(2.0))
    dec_note = (" The decrement is zero in every scenario record; only the sanity checks use one."
                if all(r["decrement"] == 0.0 for _, r in st.records()) else "")
    return section(9, "CAVEATS", "Caveats", f"""
<ul class="points dim">
<li><b>Multiplicity.</b> The report tests {total:,} per-name method gaps at 3 SE ({resolved:,} resolved). If every
true gap were zero, about {num(p_null * total, 1)} would still clear 3 SE by chance (two-sided rate
{num(100 * p_null, 2)}%). A lone resolved gap of a few CNY is weak evidence; the findings rest on patterns across
scenarios.</li>
<li><b>Stylised, synthetic basket.</b> No desk data. Weights, vols, yields and correlations are illustrative.</li>
<li><b>GBM only.</b> No smile, local or stochastic vol, no jumps. A smile would change both the reference and the
fast model.</li>
<li><b>Price-return index.</b> No total-return or net-total-return variants.{dec_note}</li>
<li><b>Correlation band.</b> The band measures how sensitive the reference is to a ±shift of a uniform ρ. It is a
model-risk yardstick, not a confidence interval, and it is not defined at the PSD floor or at ρ = 1.</li>
<li><b>Static deltas.</b> This compares deltas at one instant; it is not a hedge backtest and carries no costs.</li>
</ul>
""")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

NAV = [(1, "SUMMARY"), (2, "SETUP"), (3, "SANITY"), (4, "FIXED"), (5, "REBAL"), (6, "DRIVERS"), (7, "YIELDS"),
       (8, "PV · MM"), (9, "CAVEATS")]


def masthead(st: Study) -> tuple:
    b = st.base
    m1 = b["methods"]["M1"]
    tech, gold = name_idx(b, "TECH"), name_idx(b, "GOLD")
    daily = st.freq_row("daily", "start")["methods"]["M1"]
    bp = pv(b)

    def rel_tile(m, i):
        return spct(m["gap_rel_to_ref"][i]) if m["resolved"][i] else NR

    kpis = [
        (rel_tile(m1, tech), "", f"M1 gap on TECH · {sgn(m1['gap_cash_per_pct'][tech], 0) if m1['resolved'][tech] else 'not resolved'} CNY/1% · base 1Y ATM call"),
        (rel_tile(m1, gold), "", f"M1 gap on GOLD · {sgn(m1['gap_cash_per_pct'][gold], 0) if m1['resolved'][gold] else 'not resolved'} CNY/1%"),
        (num(m1["residual_sigma_cny"]) if m1["residual_resolved"] else NR, "CNY/DAY",
         f"M1 residual σ fixed shares · daily rebalance {sigma_text(daily).replace('&lt;', '<')}"),
        (spct(bp["rel"], 2) if bp["resolved"] else NR, "PV",
         f"Fast-model PV error · base ATM call · {sgn(bp['err'], 0) if bp['resolved'] else 'not resolved'} CNY"),
    ]
    tiles = "".join(f'<div class="kpi"><div class="kv">{v}<small>{u}</small></div><div class="kl">{gk(html.escape(l))}</div></div>'
                    for v, u, l in kpis)
    nav = "".join(f'<a href="#s{n}"><b>{n})</b> {t}</a>' for n, t in NAV)
    chrome = f"""
<div class="top">
<header class="fbar">
  <span class="brand">QUANTARK</span><span class="cmd">CSI DELTA &lt;GO&gt;</span>
  <span class="fld">STUDY <b>csi_basket_delta</b></span><span class="fld">MODEL <b>MULTI-NAME GBM · MC + MM</b></span>
  <span class="fld">PATHS <b>{b['n_paths']:,} × {b['n_batches']}</b></span><span class="fld date">{date.today().isoformat()}</span>
  <button class="theme" id="theme" type="button" aria-label="Switch colour theme">◐ <b>LIGHT</b></button>
</header>
<nav class="menu" id="menu">{nav}</nav>
</div>
"""
    hero = f"""
<div class="hero">
  <div class="kicker">CSI · BASKET · DELTA</div>
  <h1>Hedge a custom index as one order, or name by name?</h1>
  <p class="answer">The total delta is the same either way; only the <b>split</b> differs. On a fixed-share basket the
  whole-index order puts {num(abs(m1['gap_cash_per_pct'][tech]))} CNY per 1% {less_more(m1['gap_cash_per_pct'][tech])}
  delta on TECH than a multi-name model, a residual of {sigma_text(m1)} CNY a day. Daily rebalancing cuts that to
  {sigma_text(daily)}. Never split by target weights after drift, and quote PV off the multi-name MC.</p>
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
  --bar:#000;--bar-2:#0d0d0d;--bar-line:#2a2a2a;--bar-ink:#e8e6df;--bar-ink-2:#a3a199;--bar-ink-3:#77756e;
  --bar-amber:#ffa028;
  --sans:"Helvetica Neue",Helvetica,Inter,Arial,sans-serif;
  --mono:"SF Mono","JetBrains Mono","IBM Plex Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box}
html{background:var(--bg);scroll-padding-top:84px}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
code{font:0.88em var(--mono);color:var(--amber)}
em{font-style:normal;color:var(--amber)}
a{color:inherit;text-decoration:none}

.top{position:sticky;top:0;z-index:5;background:var(--bar)}
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
.hero h1{grid-column:4/13;margin:0;font:700 clamp(34px,5vw,62px)/1.04 var(--sans);letter-spacing:-.025em}
.answer{grid-column:4/12;margin:28px 0 0;font-size:18px;line-height:1.5;color:var(--ink-2)}
.answer b{color:var(--ink);font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border-bottom:1px solid var(--line)}
.kpi{padding:22px 20px 22px 0;border-right:1px solid var(--line);min-width:0}
.kpi+.kpi{padding-left:20px}.kpi:last-child{border-right:0}
.kv{font:600 clamp(24px,3.2vw,38px)/1 var(--mono);letter-spacing:-.02em;color:var(--ink)}
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
h3.sub{margin:36px 0 8px;font:11px/1 var(--mono);letter-spacing:.16em;color:var(--ink-3);text-transform:uppercase}

dl.spec{display:grid;grid-template-columns:160px minmax(0,1fr);margin:0 0 24px;border-top:1px solid var(--line)}
dl.spec dt,dl.spec dd{margin:0;padding:10px 0;border-bottom:1px solid var(--hair)}
dl.spec dt{font:11px/1.6 var(--mono);letter-spacing:.1em;color:var(--amber);text-transform:uppercase}

.scroll{overflow-x:auto;margin:8px 0}
table{border-collapse:collapse;width:100%;font:12.5px/1.3 var(--mono);font-variant-numeric:tabular-nums}
th{padding:8px 12px 8px 0;text-align:right;font-weight:500;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--amber-dim);border-bottom:1px solid var(--amber-dim);white-space:nowrap}
td{padding:6px 12px 6px 0;text-align:right;border-bottom:1px solid var(--hair);white-space:nowrap}
th:first-child,td:first-child,table.text th,table.text td{text-align:left}
table.text td{white-space:normal}
tbody tr:hover td{background:var(--row)}
td code{color:var(--ink)}
.up{color:var(--up)}.dn{color:var(--dn)}.flat{color:var(--ink-2)}.dim{color:var(--ink-3)}
.nr{color:var(--ink-3);font-size:.92em;letter-spacing:.02em}
.gk{text-transform:none}
.note{margin:6px 0 0;font:11px/1.5 var(--mono);color:var(--ink-3);max-width:none}

figure{margin:0 0 28px;border-top:1px solid var(--line);padding-top:12px}
figcaption{display:flex;flex-wrap:wrap;gap:4px 16px;align-items:baseline;margin-bottom:10px}
figcaption b{font:600 13px/1.3 var(--sans)}
figcaption span{font:11px/1.3 var(--mono);color:var(--ink-3)}
.legend{display:flex;flex-wrap:wrap;gap:6px 20px;font:11px/1 var(--mono);color:var(--ink-2);margin:2px 0 6px}
.legend i{display:inline-block;width:14px;height:3px;margin-right:7px;vertical-align:middle}
svg.chart{width:100%;height:auto;display:block;overflow:visible}
.chart .grid{stroke:var(--hair);stroke-width:1}
.chart .axis{stroke:var(--line);stroke-width:1}
.chart .zero{stroke:var(--ink-3);stroke-width:1}
.chart .tick{font:10.5px var(--mono);fill:var(--ink-3)}
.chart .axlab{font:9.5px var(--mono);letter-spacing:.1em;fill:var(--ink-3)}
.chart .ln{fill:none;stroke-width:2;stroke-linejoin:round}
.chart .mk{stroke:var(--bg);stroke-width:2;pointer-events:none}
.chart .mk.hollow{stroke-width:2}
.chart .hit{fill:transparent;cursor:crosshair}
.tip{position:fixed;z-index:10;pointer-events:none;padding:6px 9px;background:var(--panel);border:1px solid var(--amber);
  color:var(--ink);font:11.5px/1.3 var(--mono);white-space:nowrap;display:none}
.chart .dlab{font:11px var(--mono);fill:var(--ink-2)}
.multiples{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px}
.ptitle{font:11px/1 var(--mono);letter-spacing:.12em;color:var(--amber);margin-bottom:6px}

ul.points{padding:0;list-style:none;margin:20px 0 0;max-width:72ch}
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
  .multiples{grid-template-columns:minmax(0,1fr)}
  dl.spec{grid-template-columns:minmax(0,1fr)}dl.spec dt{border-bottom:0;padding-bottom:0}
  .fbar .fld{display:none}
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


def build_html() -> str:
    """Read results/*.json and return the full report page."""
    st = Study(load_results())
    chrome, hero = masthead(st)
    content = (
        hero + section_summary(st) + section_setup(st) + section_sanity(st) + section_fixed(st)
        + section_rebalance(st) + section_drivers(st) + section_yields(st) + section_mm(st) + section_caveats(st)
        + "<footer>QUANTARK · example/case_studies/csi_basket_delta · stages 01–04 → results/*.json → 05_report.py · "
          "synthetic basket, no desk data</footer>"
    )
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>CSI Basket Delta</title>{THEME_INIT_JS}<style>{CSS}</style></head>"
            f"<body>{chrome}<main>{content}</main>{PAGE_JS}</body></html>")


def main() -> None:
    page = build_html()
    REPORT_PATH.write_text(page, encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
