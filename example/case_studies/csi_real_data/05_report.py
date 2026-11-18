"""Stage 05 - Render the CSI real-data study as one self-contained HTML report.

Reads ``results/02_indices.json``, ``results/03_gap.json`` and
``results/04_hedge.json`` (run stages 02-04 first) and writes
``results/csi_real_data_report.html``.  Every figure on the page is computed
from those three files when the page is built; nothing is re-priced and no
number is typed in by hand.

The page follows the house UI language (``docs/UI_LANGUAGE.md``: Bloomberg x
Swiss, frozen function bar + section menu, light/dark switcher, inline SVG).
CSS, number formatters, tables and the MathML typesetter are reused from the
synthetic sibling study's report, loaded by file path under a unique module
name (other studies ship modules with the same file name).

    python example/case_studies/csi_real_data/05_report.py
"""

from __future__ import annotations

import html
import importlib.util
import json
import math
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
REPORT_PATH = RESULTS_DIR / "csi_real_data_report.html"
STAGES = ("02_indices", "03_gap", "04_hedge")
SYNTH_REPORT = "../../csi_basket_delta/results/csi_basket_delta_report.html"


def _helpers():
    """The synthetic study's report module, loaded by path under a name no other study uses."""
    alias = "csi_real_data_synthetic_report_helpers"
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, HERE.parent / "csi_basket_delta" / "05_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


H = _helpers()
num, sgn, signed, spct, pct, compact = H.num, H.sgn, H.signed, H.spct, H.pct, H.compact
table, figure, multiples, section, gk, mx, mxb = H.table, H.figure, H.multiples, H.section, H.gk, H.mx, H.mxb
NR = H.NR

TECH_CODE, GOLD_CODE = "002230", "518880"       # the basket's high-beta tech name and its hedge-like name
COLOURS = ("var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)")
ARMS = ("test", "reference", "attribution")
ARM_NOTES = {
    "test": "Black on the index, vol from the index's own trailing history; Δ_I × current shares.",
    "reference": "Per-name MC (per-name vols + correlation); per-name CRN central bump deltas.",
    "attribution": "The reference MC's own Δ_I × current shares: the reference model with the test model's split.",
}
PAIRS = ("reference-test", "attribution-test", "reference-attribution")
PAIR_NOTES = {
    "reference-test": "reference − test · the whole gap",
    "attribution-test": "attribution − test · total-delta (index vol) part",
    "reference-attribution": "reference − attribution · per-name split part",
}


def load_results(results_dir: Path = RESULTS_DIR) -> dict:
    """Read the three stage JSON files; nothing else is read."""
    out = {}
    for stage in STAGES:
        path = Path(results_dir) / f"{stage}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}: run the stage that writes it first")
        out[stage] = json.loads(path.read_text(encoding="utf-8"))
    return out


# ---------------------------------------------------------------------------
# Derived views of the JSON
# ---------------------------------------------------------------------------


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def _rms(xs) -> float:
    xs = list(xs)
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def pair_view(p: dict) -> dict:
    """A Stage B paired statistic: noise-corrected when the JSON carries it, else the raw std."""
    if "corrected_std_cny" in p:
        lo, hi = p["corrected_ci90"]
        return {"v": p["corrected_std_cny"], "lo": lo, "hi": hi, "resolved": bool(p["resolved"]),
                "corrected": True, "raw": p["std_cny"], "noise": p["noise_std_cny"], "mean": p["mean_cny"]}
    lo, hi = p["ci90"]
    return {"v": p["std_cny"], "lo": lo, "hi": hi, "resolved": None, "corrected": False,
            "raw": p["std_cny"], "noise": None, "mean": p["mean_cny"]}


def pair_text(pv: dict) -> str:
    """σ in CNY/day for prose: 'not resolved' when the corrected CI reaches zero."""
    if pv["resolved"] is False:
        return f"not resolved (90% CI below {num(pv['hi'])} CNY/day)"
    tag = "" if pv["corrected"] else " (raw)"
    return f"{num(pv['v'])} CNY/day{tag}"


def pair_cell(pv: dict) -> str:
    if pv["resolved"] is False:
        return f'<span class="nr">not resolved · &lt; {num(pv["hi"])}</span>'
    return num(pv["v"])


class Study:
    """Indexes the three result files; every lookup reads the JSON as written."""

    def __init__(self, res: dict):
        self.ix, self.gap, self.hedge = res["02_indices"], res["03_gap"], res["04_hedge"]
        self.rules = list(self.gap["meta"]["rules"])
        self.names = self.ix["names"]
        self.codes = [n["code"] for n in self.names]
        self.label = {n["code"]: n["name"] for n in self.names}
        self.tech, self.gold = self.codes.index(TECH_CODE), self.codes.index(GOLD_CODE)
        self.recs = self.gap["records"]
        self.dates = list(self.gap["meta"]["dates"])
        self.moneyness = list(self.gap["meta"]["moneyness"])
        self.summary = self.hedge["summary"]
        self.runs = self.hedge["runs"]
        self._by = {(r["rule"], r["date"], r["moneyness"]): r for r in self.recs}

    def rec(self, rule, d, m=1.0):
        return self._by.get((rule, d, m))

    def atm(self, rule):
        return [self._by[(rule, d, 1.0)] for d in self.dates if (rule, d, 1.0) in self._by]

    def rule_recs(self, rule, m=None):
        return [r for r in self.recs if r["rule"] == rule and (m is None or r["moneyness"] == m)]

    def pair(self, rule, key):
        return pair_view(self.summary[rule]["pairs"][key])

    def dominance(self) -> tuple:
        """Rules where the split part (reference − attribution) or the total-delta part (attribution − test)
        dominates — its 90% CI lies wholly above the other's — then rules where the CIs overlap
        (comparable), then rules whose larger part is not resolved from MC noise."""
        split, vol, comparable, unres = [], [], [], []
        for r in self.rules:
            s, v = self.pair(r, "reference-attribution"), self.pair(r, "attribution-test")
            big, small = (s, v) if s["v"] > v["v"] else (v, s)
            if big["resolved"] is False:
                unres.append(r)
            elif big["lo"] > small["hi"]:
                (split if big is s else vol).append(r)
            else:
                comparable.append(r)
        return split, vol, comparable, unres

    def corrected(self) -> bool:
        return all("corrected_std_cny" in p for s in self.summary.values() for p in s["pairs"].values())

    # Stage A statistics -----------------------------------------------------

    @staticmethod
    def pv_gap(r) -> float:
        return r["pv_test_cny"] / r["pv_ref_cny"] - 1.0

    @staticmethod
    def dindex_cny(r) -> float:
        """Total-delta gap in CNY per 1% index move: I·u·(Δ_I^test − Δ_I^ref)·1%."""
        return r["index_level"] * r["units"] * (r["dindex_test"] - r["dindex_ref"]) * 0.01

    def split_fit(self, rule) -> dict:
        """Least-squares slope through the origin of the split part on the first-order closed form."""
        rs = self.rule_recs(rule)
        x = [v for r in rs for v in r["first_order"]]
        y = [v for r in rs for v in r["gap_split_part"]]
        slope = sum(a * b for a, b in zip(x, y)) / sum(a * a for a in x)
        return {"n": len(x), "slope": slope, "rms_resid": _rms(b - slope * a for a, b in zip(x, y)),
                "rms_split": _rms(y), "valid": all(r["first_order_valid"] for r in rs)}

    def resolved_share(self, rule, i=None) -> float:
        rs = self.rule_recs(rule)
        flags = [f for r in rs for k, f in enumerate(r["gap_bump_resolved"]) if i is None or k == i]
        return sum(flags) / len(flags)

    # Stage B statistics -----------------------------------------------------

    def premium(self, rule) -> float:
        return _mean(r["arms"]["reference"]["premium_own_cny"] for r in self.runs[rule])

    def session_rms(self, rule, window, a="reference", b="test"):
        """Trailing ``window``-session RMS of the paired daily difference a − b over every live start
        (raw, incl. MC noise), one value per calendar session."""
        acc = {}
        for run in self.runs[rule]:
            for d, x, y in zip(run["dates"], run["arms"][a]["daily_pnl"], run["arms"][b]["daily_pnl"]):
                acc.setdefault(d, []).append(x - y)
        days = sorted(acc)
        return days, [_rms(v for d in days[max(0, i - window + 1):i + 1] for v in acc[d]) for i in range(len(days))]


# ---------------------------------------------------------------------------
# SVG charts on a date axis (the helpers' charts use categorical slots)
# ---------------------------------------------------------------------------


def _ord(d: str) -> int:
    return date.fromisoformat(d).toordinal()


def _date_ticks(lo: int, hi: int, pixels: float):
    """Month-start ticks (every 1/3/6/12 months), at least ~56 px apart."""
    a, b = date.fromordinal(lo), date.fromordinal(hi)
    px_per_month = pixels / max((hi - lo) / 30.44, 1.0)
    step = next((s for s in (1, 3, 6) if s * px_per_month >= 56), 12)
    out, y, m = [], a.year, 1
    while date(y, m, 1) <= b:
        if date(y, m, 1) >= a:
            out.append((date(y, m, 1).toordinal(), f"{y}-{m:02d}"))
        m += step
        if m > 12:
            y, m = y + 1, m - 12
    return out


def _legend(items) -> str:
    spans = "".join(f'<span><i style="background:{c}"></i>{html.escape(n)}</span>' for n, c in items)
    return f'<div class="legend">{spans}</div>'


def _end_labels(ends, body):
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 13)
    for y, name, x in ends:
        body.append(f'<text class="dlab" x="{x + 10:.1f}" y="{y + 3.5:.1f}">{html.escape(name)}</text>')


def ts_chart(series, *, ylabel, yfmt, tipfmt, xlabel="DATE", zero=False, logy=False, markers=False, hollow=None,
             ydomain=None, width=760, height=300, label=""):
    """Date-axis line chart.  ``series``: [(name, colour, dates, ys)]; ``None`` in ys leaves a gap.

    ``hollow``: per-series bools, a hollow marker is a value not resolved at 3 SE.  Dense series get
    a thinned hover layer; ``markers`` draws and hovers every point.
    """
    hollow = hollow or [[False] * len(s[3]) for s in series]
    tr = (lambda v: math.log10(v)) if logy else (lambda v: v)
    ok = (lambda v: v is not None and v > 0) if logy else (lambda v: v is not None)
    vals = [tr(v) for s in series for v in s[3] if ok(v)]
    if logy:
        lo, hi = math.floor(min(vals)), math.ceil(max(vals))
        hi = max(hi, lo + 1)
        f = H._Frame(width, height, lo, hi, ticks=list(range(lo, hi + 1)), left=64, right=96)
        tick_fmt = (lambda t: yfmt(10 ** t))
    else:
        lo, hi = ydomain or (min(vals), max(vals))
        lo, hi = min(lo, min(vals)), max(hi, max(vals))
        pad = (hi - lo) * 0.06 or 1.0
        f = H._Frame(width, height, lo if lo == 0 else lo - pad, hi + pad, left=64, right=96)
        tick_fmt = yfmt
    body = f.axes([], tick_fmt, ylabel, xlabel, zero=zero)
    xs_all = [_ord(d) for s in series for d in s[2]]
    xlo, xhi = min(xs_all), max(xs_all)

    def X(o):
        return f.l + (o - xlo) / max(xhi - xlo, 1) * (f.w - f.l - f.r)

    base = f.h - f.b
    for o, text in _date_ticks(xlo, xhi, f.w - f.l - f.r):
        body.append(f'<line class="grid" x1="{X(o):.1f}" x2="{X(o):.1f}" y1="{f.t}" y2="{base}"/>'
                    f'<text class="tick" x="{X(o):.1f}" y="{base + 16}" text-anchor="middle">{text}</text>')
    ends = []
    for k, (name, colour, dates, ys) in enumerate(series):
        pts = [(X(_ord(d)), f.y(tr(v)), v, d, h) if ok(v) else None for d, v, h in zip(dates, ys, hollow[k])]
        seg = []
        for p in pts + [None]:
            if p is None:
                if len(seg) > 1:
                    path = " ".join(f"{x:.1f},{y:.1f}" for x, y, *_ in seg)
                    body.append(f'<polyline class="ln" points="{path}" style="stroke:{colour}"/>')
                seg = []
            else:
                seg.append(p)
        live = [p for p in pts if p is not None]
        step = 1 if markers else max(1, len(live) // 120)
        for j, (x, y, v, d, h) in enumerate(live):
            if markers:
                style = f"stroke:{colour};fill:var(--bg)" if h else f"fill:{colour}"
                body.append(f'<circle class="mk{" hollow" if h else ""}" cx="{x:.1f}" cy="{y:.1f}" r="3.5" '
                            f'style="{style}"/>')
            if j % step == 0 or j == len(live) - 1:
                tip = html.escape(f"{name} · {d} · {tipfmt(v)}" + (" · not resolved" if h else ""))
                body.append(f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="9" data-tip="{tip}" '
                            f'aria-label="{tip}"/>')
        if live:
            ends.append([live[-1][1], name, live[-1][0]])
    _end_labels(ends, body)
    return _legend([(s[0], s[1]) for s in series]) + H._svg(f, body, label)


def ci_chart(cats, series, *, ylabel, xlabel, yfmt, tipfmt, width=760, height=300, label=""):
    """Point + 90% CI per category on a log axis.  ``series``: [(name, colour, [pair_view])].

    A CI whose lower end is 0 (not resolved) runs to the bottom of the axis and gets a hollow marker.
    """
    vals = [math.log10(p[key]) for _, _, ps in series for p in ps for key in ("v", "lo", "hi") if p[key] > 0]
    lo, hi = math.floor(min(vals)), math.ceil(max(vals))
    hi = max(hi, lo + 1)
    f = H._Frame(width, height, lo, hi, ticks=list(range(lo, hi + 1)), left=64, right=40)
    body = f.axes(cats, lambda t: yfmt(10 ** t), ylabel, xlabel)
    k = len(series)
    for j, (name, colour, ps) in enumerate(series):
        for i, p in enumerate(ps):
            x = f.xslot(i, len(cats)) + (j - (k - 1) / 2) * 16
            bottom = f.y(f.ylo)
            y_hi = f.y(math.log10(p["hi"])) if p["hi"] > 0 else bottom
            y_lo = f.y(math.log10(p["lo"])) if p["lo"] > 0 else bottom
            body.append(f'<line class="wh" x1="{x:.1f}" x2="{x:.1f}" y1="{y_lo:.1f}" y2="{y_hi:.1f}" '
                        f'style="stroke:{colour}"/>')
            nr = p["resolved"] is False or p["v"] <= 0
            y = f.y(math.log10(p["v"])) if p["v"] > 0 else bottom
            style = f"stroke:{colour};fill:var(--bg)" if nr else f"fill:{colour}"
            tip = html.escape(f"{name} · {cats[i]} · {tipfmt(p['v'])} · CI {num(p['lo'])}–{num(p['hi'])}"
                              + (" · not resolved" if nr else ""))
            body.append(f'<circle class="mk{" hollow" if nr else ""}" cx="{x:.1f}" cy="{y:.1f}" r="4" style="{style}"/>'
                        f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="11" data-tip="{tip}" aria-label="{tip}"/>')
    return _legend([(s[0], s[1]) for s in series]) + H._svg(f, body, label)


def scatter_chart(points, *, xlabel, ylabel, fmt, width=380, height=300, label=""):
    """y against x on a shared symmetric scale with the y = x diagonal.  ``points``: [(colour, x, y, tip)]."""
    m = max(max(abs(x), abs(y)) for _, x, y, _ in points) * 1.06
    pos = [t for t in H._nice_ticks(0.0, m, 2) if t >= 0]
    if pos[-1] < m:
        pos.append(pos[-1] + (pos[1] - pos[0]))
    ticks = [-t for t in reversed(pos[1:])] + pos
    f = H._Frame(width, height, ticks[0], ticks[-1], ticks=ticks, left=56, right=16)

    def X(v):
        return f.l + (v - f.ylo) / (f.yhi - f.ylo) * (f.w - f.l - f.r)

    body = f.axes([], fmt, ylabel, xlabel, zero=True)
    base = f.h - f.b
    for v in ticks:
        body.append(f'<text class="tick" x="{X(v):.1f}" y="{base + 16}" text-anchor="middle">{fmt(v)}</text>')
    body.append(f'<line class="diag" x1="{X(f.ylo):.1f}" y1="{f.y(f.ylo):.1f}" x2="{X(f.yhi):.1f}" '
                f'y2="{f.y(f.yhi):.1f}"/>')
    for colour, x, y, tip in points:
        t = html.escape(tip)
        body.append(f'<circle class="pt" cx="{X(x):.1f}" cy="{f.y(y):.1f}" r="2.6" style="fill:{colour}"/>'
                    f'<circle class="hit" cx="{X(x):.1f}" cy="{f.y(y):.1f}" r="6" data-tip="{t}" aria-label="{t}"/>')
    return H._svg(f, body, label)


# ---------------------------------------------------------------------------
# Formulas (built outside f-strings: no backslashes in f-string expressions before Python 3.12)
# ---------------------------------------------------------------------------

F_INDEX = mx(r"I(t) = \sum_i n_i S_i(t)")
F_RESET = mx(r"n_i = w_i\, I(t) / S_i(t)")
F_VAR = mx(r"σ^2 = 244 · \up{Var}(\up{ln}\, S(t+1) / S(t))")
F_TEST = mx(r"h_i^{\up{test}} = Δ_I^{\up{Black}}\, n_i\, u")
F_REF = mx(r"h_i^{\up{ref}} = \frac{V(S_i(1 + ε)) - V(S_i(1 - ε))}{2 ε S_i}\, u")
F_ATT = mx(r"h_i^{\up{att}} = Δ_I^{\up{MC}}\, n_i\, u")
F_SPLIT_DEF = mxb(r"\up{total}_i = w_i\, I\, (Δ_I^{\up{test}} - Δ_I^{\up{ref}})\, u · 1%",
                  r"\up{split}_i = (w_i\, I\, Δ_I^{\up{ref}} - S_i\, Δ_i^{\up{pw}})\, u · 1%")
F_FIRST = mxb(r"\up{split}_i ≈ -τ\, w_i\, I\, (β_i - 1)\, σ_I^2\, I\, Γ · u · 1%",
              r"β_i = \frac{{(Σw)}_i}{σ_I^2}, \quad σ_I^2 = w^⊤Σw")
F_PNL = mxb(r"P_A(t) = \sum_i h_i^A(t)\, (S_i(t+1) - g\, S_i(t)) - (V(t+1) - g\, V(t)), \quad g = e^{r/244}",
            r"P_A(t) - P_B(t) = \sum_i (h_i^A(t) - h_i^B(t))\, (S_i(t+1) - g\, S_i(t))")
F_CORR = mx(r"σ_c = \sqrt{\up{Var}(d) - \bar{v}}")
F_TAU = mx("τ")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _rule_series(st: Study, ys_by_rule, dates_by_rule):
    return [(rule, COLOURS[k], dates_by_rule[rule], ys_by_rule[rule]) for k, rule in enumerate(st.rules)]


def section_summary(st: Study) -> str:
    rows = []
    for rule in st.rules:
        atm = st.atm(rule)
        rows.append([rule, spct(_mean(st.pv_gap(r) for r in atm), 2),
                     signed(_mean(r["gap_bump"][st.tech] for r in atm)),
                     num(_mean(r["gap_sigma_cny_day"] for r in atm)),
                     pair_cell(st.pair(rule, "reference-test")),
                     pair_cell(st.pair(rule, "attribution-test")),
                     pair_cell(st.pair(rule, "reference-attribution")),
                     signed(_mean(st.summary[rule]["premium_gap_cny"]))])
    kind = "noise-corrected" if st.corrected() else "raw (MC noise included)"
    tbl = table(["rule", "A · ATM PV test/ref − 1", "A · TECH gap, CNY/1%", "A · gap σ, CNY/day",
                 "B · ref − test σ", "B · att − test σ", "B · ref − att σ", "B · premium gap, CNY"], rows,
                note=f"A = Stage A, means over the {len(st.dates)} evaluation dates, 1Y ATM call on 1e8 CNY. "
                     f"B = Stage B, paired daily hedge P&amp;L σ in CNY/day, {kind}, pooled over "
                     f"{st.summary[st.rules[0]]['n_starts']} starts. Premium gap = test PV − reference PV at "
                     "the start, mean over starts.")
    none, daily = st.rules[0], st.rules[-1]
    split = {r: st.pair(r, "reference-attribution") for r in st.rules}
    split_dom, vol_dom, comparable, unres = st.dominance()
    unres_txt = f" Under {', '.join(unres)} neither part is resolved from MC noise." if unres else ""
    comp_txt = (f" Under {', '.join(comparable)} the two parts are comparable (their 90% CIs overlap)."
                if comparable else "")
    fits = [st.split_fit(r) for r in st.rules if r != none]
    pv_none = [st.pv_gap(r) for r in st.atm(none)]
    pv_daily = [st.pv_gap(r) for r in st.atm(daily)]
    points = f"""
<ul class="points">
<li><b>The rebalance rule decides the split error.</b> The per-name split part of the daily hedge gap (reference −
attribution) is {pair_text(split[none])} on fixed shares ({none}) and {pair_text(split[daily])} under
{daily} rebalancing. In Stage A the mean |split part| per name falls from
{num(_mean(abs(v) for r in st.rule_recs(none) for v in r["gap_split_part"]))} to
{num(_mean(abs(v) for r in st.rule_recs(daily) for v in r["gap_split_part"]))} CNY/1%.</li>
<li><b>Where the split dominates, the desk needs a per-name model.</b> The split part dominates the
total-delta part (90% CIs apart) under {", ".join(split_dom) or "no rule"}; the total-delta (index-vol) part
dominates under {", ".join(vol_dom) or "no rule"}. That part does not need a per-name model, only a better
{mx("Δ_I")}.{comp_txt}{unres_txt}</li>
<li><b>The first-order closed form predicts the split on real calibrations.</b> On the rebalanced rules the fitted
slope of the measured split part on the formula is {num(min(x["slope"] for x in fits), 3)}–{num(max(x["slope"] for x in fits), 3)}
(1 = exact), with an rms residual of {num(min(x["rms_resid"] for x in fits), 1)}–{num(max(x["rms_resid"] for x in fits), 1)} CNY/1%.</li>
<li><b>PV.</b> Black on the index prices the ATM call {spct(_mean(pv_none), 2)} off the reference on average under
{none} (range {spct(min(pv_none), 2)} to {spct(max(pv_none), 2)}) and {spct(_mean(pv_daily), 2)} under {daily}
(range {spct(min(pv_daily), 2)} to {spct(max(pv_daily), 2)}).</li>
</ul>"""
    return section(1, "SUMMARY", "Desk read", f"""
<p class="lead">Same real prices, five index rules, two models. The single-underlying desk model (Black on the index,
{mx("Δ_I")} split by current shares) is compared with a per-name Monte Carlo reference, first as a static gap on
{len(st.dates)} monthly dates (Stage A), then as daily hedge P&amp;L along the real path (Stage B). All figures in CNY
on 1e8 CNY of index notional.</p>
{tbl}
<h3 class="sub">What the desk should take from it</h3>
{points}
""")


def section_setup(st: Study) -> str:
    meta, gm, hm = st.ix["meta"], st.gap["meta"], st.hedge["meta"]
    basket = table(["code", "name", "kind", "target weight", "source", "total return", "realized σ", "suspension fills"],
                   [[n["code"], html.escape(n["name"]), n["kind"], pct(n["weight"], 0), html.escape(n["source"]),
                     spct(n["total_return"]), pct(n["realized_vol_full"]), num(n["suspension_fills"])]
                    for n in st.names],
                   note=f"Total return and realized σ over {meta['data_first']} → {meta['data_last']} "
                        f"({meta['n_sessions']} sessions). σ convention: {html.escape(meta['vol_convention'])}.")
    models = table(["model", "what it is", "per-name hedge"], [
        ["test", "Black on the index as one underlying, σ = the index's own trailing historical vol, q = 0.", F_TEST],
        ["reference", "Per-name GBM Monte Carlo (RQMC), per-name trailing vols + sample correlation, the real "
                      "remaining rebalance schedule. The truth.", F_REF],
        ["attribution", html.escape(ARM_NOTES["attribution"]), F_ATT],
    ], text=True)
    return section(2, "SETUP", "Basket, data, models", f"""
<p class="lead">A public five-name test basket (not a desk CSI), built from real daily closes. Back-adjusted (hfq)
prices make every index a total-return index, so both models use q = 0.</p>
{basket}
<dl class="spec">
<dt>Data</dt><dd>akshare {html.escape(meta['akshare'])}, fetched {meta['fetched_on']}; stocks via
<code>stock_zh_a_hist_em</code>, ETFs via <code>fund_etf_hist_em</code>, adjust = <code>hfq</code> (total return).
Sessions {meta['data_first']} → {meta['data_last']} on the exchange calendar (listed to
{meta['calendar_horizon']}); a suspended name carries its previous close. Raw prices stay in a cache outside the
repo.</dd>
<dt>Index</dt><dd>Base {num(meta['base_level'])} on {meta['base_date']} at target weights. Between rebalances
{F_INDEX}; at a rebalance close {F_RESET}. Rules: none (fixed shares), quarterly, monthly, weekly (first session of
each period) and daily.</dd>
<dt>Derived data</dt><dd>The results JSON holds daily index levels and current weights per rule, not prices. Rule
none's levels and weights together rebuild each name's rebased series; that is allowed for this public test case.</dd>
<dt>Calibration</dt><dd>Both models on the same trailing {gm['window']} daily log returns ending on the date:
{F_VAR}, demeaned, ddof 1. Reference: per-name vols + sample correlation (PSD-checked). Test: the index's own
returns under the same rule.</dd>
<dt>Clock</dt><dd>1Y = {hm['sessions']} sessions; times are session counts / 244. Rate r =
{pct(gm['rate'], 1)} for discounting and hedge financing, both models.</dd>
<dt>Options</dt><dd>European calls on 1e8 CNY of index notional, 1Y. Stage A: struck at
{", ".join(num(m, 1) for m in st.moneyness)} × the index on the 10th session of each month from
{st.dates[0]} ({len(st.dates)} dates). Stage B: the ATM call sold on each of {len(hm['starts'])} starts whose
option has expired, strike and units fixed at the start.</dd>
<dt>MC</dt><dd>Stage A {gm['n_paths']:,} × {gm['n_batches']} RQMC batches; Stage B {hm['n_paths']:,} ×
{hm['n_batches']}. Per-name deltas by CRN central bump; pathwise deltas kept for the exact decomposition.</dd>
</dl>
<h3 class="sub">Models</h3>
{models}
""")


def section_index(st: Study) -> str:
    dates = st.ix["dates"]
    rules = st.ix["rules"]
    lvl = ts_chart([(r, COLOURS[k], dates, rules[r]["level"]) for k, r in enumerate(st.rules)],
                   ylabel="INDEX LEVEL", yfmt=lambda v: num(v, 0), tipfmt=lambda v: num(v, 1), label="Index levels")
    none = st.rules[0]
    wts = rules[none]["weights"]
    drift = ts_chart([(st.label[c], COLOURS[k], dates, [100 * w for w in wts[c]]) for k, c in enumerate(st.codes)],
                     ylabel="WEIGHT, %", yfmt=lambda v: num(v, 0), tipfmt=lambda v: num(v, 1) + "%",
                     label="Weights under rule none")
    # Sampled tables under the daily charts: the first session of each quarter, plus the last session.
    pick, seen = [], set()
    for i, d in enumerate(dates):
        q = (d[:4], (int(d[5:7]) - 1) // 3)
        if q not in seen:
            seen.add(q)
            pick.append(i)
    if pick[-1] != len(dates) - 1:
        pick.append(len(dates) - 1)
    lvl_rows = [[dates[i]] + [num(rules[r]["level"][i], 1) for r in st.rules] for i in pick]
    w_rows = [[dates[i]] + [pct(wts[c][i], 1) for c in st.codes] for i in pick]
    stat_rows = [[r, num(rules[r]["n_rebalances"]), spct(rules[r]["total_return"]), spct(rules[r]["eval_return"]),
                  pct(rules[r]["realized_vol_full"], 2), pct(rules[r]["realized_vol_eval"], 2),
                  pct(rules[r]["max_abs_weight_drift"], 1)] for r in st.rules]
    fw_rows = [[r] + [pct(rules[r]["final_weights"][c], 1) for c in st.codes] for r in st.rules]
    ev = st.ix["meta"]["eval_start"]
    end_lv = {r: rules[r]["level"][-1] for r in st.rules}
    hi_r, lo_r = max(end_lv, key=end_lv.get), min(end_lv, key=end_lv.get)
    return section(3, "INDEX", "Same prices, five rules", f"""
<p class="lead">The rebalance rule alone moves the index. From the same closes the five indices end
{pct(st.ix['terminal_spread'], 2)} apart ({hi_r} {num(end_lv[hi_r], 1)} vs {lo_r} {num(end_lv[lo_r], 1)}), and
under rule {none} the weights drift up to {pct(rules[none]['max_abs_weight_drift'], 1)} from target.</p>
{figure(lvl, "Index level by rebalance rule", "Base 1000 at the first session. Daily closes.")}
{table(["date"] + st.rules, lvl_rows, "Sampled at the first session of each quarter and the last session.")}
<h3 class="sub">Return, realized vol and drift by rule</h3>
{table(["rule", "rebalances", "total return", f"return from {ev}", "σ full", f"σ from {ev}", "max |w − target|"],
       stat_rows, "σ: realized vol of the index's daily log returns, annualised on 244 sessions.")}
{figure(drift, "Current weights under rule none", "Fixed shares: the weights drift with relative prices.")}
{table(["date"] + [html.escape(st.label[c]) for c in st.codes], w_rows, "Rule none, sampled as above.")}
<h3 class="sub">Final weights by rule</h3>
{table(["rule"] + [html.escape(st.label[c]) for c in st.codes], fw_rows)}
""")


def section_gap(st: Study) -> str:
    d_by = {r: [x["date"] for x in st.atm(r)] for r in st.rules}
    pv_by = {r: [100 * st.pv_gap(x) for x in st.atm(r)] for r in st.rules}
    pv_chart = ts_chart(_rule_series(st, pv_by, d_by), ylabel="PV TEST / REF − 1, %", yfmt=lambda v: num(v, 1),
                        tipfmt=lambda v: sgn(v, 2) + "%", zero=True, markers=True, label="ATM PV gap")
    pv_rows = [[d] + [spct(st.pv_gap(st.rec(r, d)), 2) if st.rec(r, d) else "–" for r in st.rules] for d in st.dates]
    di_by = {r: [st.dindex_cny(x) for x in st.atm(r)] for r in st.rules}
    di_chart = ts_chart(_rule_series(st, di_by, d_by), ylabel="Δ_I GAP, CNY PER 1%", yfmt=compact,
                        tipfmt=lambda v: sgn(v, 0) + " CNY/1%", zero=True, markers=True, label="Total delta gap")
    di_rows = [[d] + [f"{num(st.rec(r, d)['dindex_test'], 4)} / {num(st.rec(r, d)['dindex_ref'], 4)}"
                      if st.rec(r, d) else "–" for r in st.rules] for d in st.dates]

    def name_panel(i):
        ys = {r: [x["gap_bump"][i] for x in st.atm(r)] for r in st.rules}
        hol = [[not x["gap_bump_resolved"][i] for x in st.atm(r)] for r in st.rules]
        return ts_chart(_rule_series(st, ys, d_by), ylabel="TEST − REF, CNY PER 1%", yfmt=compact,
                        tipfmt=lambda v: sgn(v, 0) + " CNY/1%", zero=True, markers=True, hollow=hol,
                        width=380, height=280, label=f"Per-name gap {st.codes[i]}")

    panels = [(f"TECH · {st.label[TECH_CODE]}", name_panel(st.tech)), (f"GOLD · {st.label[GOLD_CODE]}", name_panel(st.gold))]

    def gcell(r, i):
        if r is None:
            return "–"
        return signed(r["gap_bump"][i]) if r["gap_bump_resolved"][i] else NR

    nm_rows = [[d] + [gcell(st.rec(r, d), st.tech) for r in st.rules] + [gcell(st.rec(r, d), st.gold) for r in st.rules]
               for d in st.dates]
    sig_by = {r: [x["gap_sigma_cny_day"] for x in st.atm(r)] for r in st.rules}
    sig_chart = ts_chart(_rule_series(st, sig_by, d_by), ylabel="GAP σ, CNY/DAY (LOG)", yfmt=compact,
                         tipfmt=lambda v: num(v) + " CNY/day", logy=True, markers=True, label="Gap sigma")
    sig_rows = [[r, num(_mean(sig_by[r])), num(min(sig_by[r])), num(max(sig_by[r])),
                 pct(st.resolved_share(r), 0), pct(st.resolved_share(r, st.tech), 0), pct(st.resolved_share(r, st.gold), 0)]
                for r in st.rules]
    fb = sum(x["calendar_fallback"] for x in st.recs)
    return section(4, "STAGE A", "The gap through time", f"""
<p class="lead">On each evaluation date both models are calibrated on the same trailing window and priced on the same
index state. The test model's ATM PV sits {spct(_mean(pv_by[st.rules[0]]) / 100, 2)} off the reference on average
under {st.rules[0]} and {spct(_mean(pv_by[st.rules[-1]]) / 100, 2)} under {st.rules[-1]}.</p>
{figure(pv_chart, "ATM PV gap, test / reference − 1", "1Y ATM call, one point per evaluation date and rule.")}
{table(["date"] + st.rules, pv_rows, "Test PV / reference PV − 1, 1Y ATM call.")}
<h3 class="sub">Total delta {mx("Δ_I")}</h3>
{figure(di_chart, "Total-delta gap in CNY per 1% index move", "I · u · (Δ_I test − Δ_I ref) · 1%, ATM.")}
{table(["date"] + [f"{r} Δ_I test / ref" for r in st.rules], di_rows, "Per index unit, ATM.")}
<h3 class="sub">Per-name gap · TECH and GOLD</h3>
{multiples("Per-name delta gap, test − reference (CRN bump)", "CNY per 1% move of the name, ATM. Hollow: |gap| ≤ 3 SE.", panels)}
{table(["date"] + [f"TECH {r}" for r in st.rules] + [f"GOLD {r}" for r in st.rules], nm_rows,
       "CNY per 1% move of the name; resolved only when |gap| &gt; 3 MC SE of the reference bump delta.")}
<h3 class="sub">Daily P&amp;L σ of the gap and resolved share</h3>
<p>The gap σ prices the per-name gap vector with the trailing daily covariance. It is <b>not</b> noise-corrected:
it includes MC noise in the reference deltas, which sets its floor under the frequent rules.</p>
{figure(sig_chart, "Gap σ through time", "ATM, CNY/day, log scale; includes MC noise.")}
{table(["rule", "mean σ, CNY/day", "min", "max", "resolved · all names", "resolved · TECH", "resolved · GOLD"], sig_rows,
       "Resolved share: per-name gaps with |gap| &gt; 3 SE, all dates and strikes.")}
<p class="note">{fb} of {len(st.recs)} records use the weekday calendar past the exchange calendar's horizon
({st.gap['meta']['calendar_horizon']}) for rebalance dates and expiry.</p>
""")


def section_split(st: Study) -> str:
    dec_rows = []
    for r in st.rules:
        for m in st.moneyness:
            rs = st.rule_recs(r, m)
            dec_rows.append([r, num(m, 1), num(_mean(x["tau_sessions"] for x in rs), 1),
                             num(_mean(abs(v) for x in rs for v in x["gap_total_part"])),
                             num(_mean(abs(v) for x in rs for v in x["gap_split_part"])),
                             num(_mean(abs(v) for x in rs for v in x["bump_minus_pathwise"]), 1)])
    fit_rows = []
    for r in st.rules:
        f = st.split_fit(r)
        fit_rows.append([r, num(f["n"]), num(f["slope"], 3), num(f["rms_resid"], 1), num(f["rms_split"], 1),
                         "yes" if f["valid"] else '<span class="nr">no · τ = T</span>'])
    panels = []
    for r in st.rules:
        pts = [(COLOURS[i], x["first_order"][i], x["gap_split_part"][i],
                f"{st.label[c]} · {x['date']} · K/I {num(x['moneyness'], 1)} · formula {sgn(x['first_order'][i], 0)} · "
                f"measured {sgn(x['gap_split_part'][i], 0)}")
               for x in st.atm(r) for i, c in enumerate(st.codes)]
        title = r.upper() + ("" if st.split_fit(r)["valid"] else " · τ = T, NOT VALID")
        panels.append((title, scatter_chart(pts, xlabel="FORMULA, CNY/1%", ylabel="MEASURED SPLIT, CNY/1%",
                                            fmt=compact, label=f"Split vs formula {r}")))
    legend = _legend([(st.label[c], COLOURS[i]) for i, c in enumerate(st.codes)])
    return section(5, "SPLIT", "Total delta vs split", f"""
<p class="lead">The per-name gap against the pathwise reference splits exactly into a total-delta part (the two
models disagree on {mx("Δ_I")}) and a split part (same {mx("Δ_I")}, different allocation across names). The split
parts sum to zero over the names.</p>
{F_SPLIT_DEF}
<h3 class="sub">Mean size of each part · per name</h3>
{table(["rule", "K/I", "stub τ, sessions", "|total part|, CNY/1%", "|split part|, CNY/1%", "|bump − pathwise|"], dec_rows,
       "Means of absolute values over names and evaluation dates. Bump − pathwise: FD bias plus MC noise of the "
       "reference per-name delta.")}
<h3 class="sub">The split against the first-order closed form</h3>
<p>To first order in the stub {F_TAU} to the next rebalance (or to expiry), with q = 0 (total-return prices):</p>
{F_FIRST}
<p>β, {mx("σ_I^2")} from the per-name calibration and current weights, Γ from Black at {mx("σ_I")}. Under rule
none {F_TAU} is the whole year and the expansion is not valid; it is shown for comparison.</p>
{table(["rule", "points", "fitted slope", "rms residual, CNY/1%", "rms split, CNY/1%", "valid"], fit_rows,
       "Least-squares slope through the origin of the measured split part on the formula, all names, strikes and "
       "dates. 1 = the formula is exact.")}
<figure><figcaption><b>Measured split part vs the closed form</b><span>ATM, every name and date (the fit above uses
every strike). Each panel on its own symmetric scale; diagonal = formula exact.</span></figcaption>{legend}<div class="multiples">
{"".join(f'<div><div class="ptitle">{html.escape(t)}</div>{c}</div>' for t, c in panels)}</div></figure>
""")


def section_hedge(st: Study) -> str:
    hm = st.hedge["meta"]
    arms = table(["arm", "per-name hedge", "premium"], [
        ["test", F_TEST, "own PV (Black)"], ["reference", F_REF, "own PV (MC)"], ["attribution", F_ATT, "reference PV"],
    ], text=True)
    arm_rows = [[r] + [num(st.summary[r]["arm_daily_std_cny"][a]) for a in ARMS] + [num(st.summary[r]["n_obs"])]
                for r in st.rules]
    pm_ratio = st.hedge["noise_probe"].get("predicted_over_measured")
    pair_rows = []
    for r in st.rules:
        for key in PAIRS:
            p = st.pair(r, key)
            res = "–" if p["resolved"] is None else ("yes" if p["resolved"] else NR)
            if p["noise"] is None or not pm_ratio:
                scaled = "–"
            else:   # sensitivity: the predicted noise scaled up to the probe's measured level (CP3 review)
                scaled = num(math.sqrt(max(p["raw"] ** 2 - (p["noise"] / pm_ratio) ** 2, 0.0)))
            pair_rows.append([r, html.escape(PAIR_NOTES[key].split(" · ")[0]), num(p["v"]), f"{num(p['lo'])} – {num(p['hi'])}", res,
                              scaled, num(p["raw"]), "–" if p["noise"] is None else num(p["noise"]), signed(p["mean"])])
    ci = ci_chart(st.rules, [(PAIR_NOTES[k].split(" · ")[0], COLOURS[j], [st.pair(r, k) for r in st.rules])
                             for j, k in enumerate(PAIRS)],
                  ylabel="PAIRED DAILY σ, CNY/DAY (LOG)", xlabel="REBALANCE RULE", yfmt=compact,
                  tipfmt=lambda v: num(v) + " CNY/day", label="Paired hedge differences")
    npb = st.hedge["noise_probe"]
    val = npb.get("validates")
    val_text = "–" if val is None else ("validates" if val else "does not validate")
    ratio = npb.get("predicted_over_measured")
    probe = f"""
<dl class="spec">
<dt>Probe</dt><dd>Rule {npb['rule']}, starts {", ".join(npb['starts'])}: the reference re-run with
{npb.get('n_paths_2n', 2 * hm['n_paths']):,} instead of {npb.get('n_paths', hm['n_paths']):,} paths on independent
seeds. Measured MC noise of the reference − test difference: σ²(N) = Var(d_N − d_2N) / 1.5.</dd>
<dt>Noise σ</dt><dd>predicted {num(npb.get('noise_std_predicted_cny'))} vs measured
{num(npb.get('noise_std_measured_cny'))} CNY/day (ratio {num(ratio, 2)}): <b>{val_text}</b> (band 0.75–1.33).</dd>
<dt>Raw share</dt><dd>MC noise σ / std of the raw reference − test difference = {num(npb['share'], 3)}. The
noise correction above removes that variance instead of buying many times the paths.</dd>
</dl>"""
    corr_note = ("" if st.corrected() else
                 '<p class="note">This results file predates the noise correction: σ below is the raw std and the '
                 "CI is the raw block-bootstrap CI.</p>")
    rms_by, d_by = {}, {}
    for r in st.rules:
        d_by[r], rms_by[r] = st.session_rms(r, hm["block"])
    when = ts_chart(_rule_series(st, rms_by, d_by), ylabel="RMS REF − TEST, CNY/DAY (LOG)", yfmt=compact,
                    tipfmt=lambda v: num(v) + " CNY/day", logy=True, label="When the hedge gap happens")
    when_rows = [[r, num(max(rms_by[r])), d_by[r][rms_by[r].index(max(rms_by[r]))], num(min(rms_by[r])),
                  d_by[r][rms_by[r].index(min(rms_by[r]))]] for r in st.rules]
    starts = hm["starts"]
    t_panels, t_rows = [], []
    for r in st.rules:
        tc = st.summary[r]["terminal_cny"]
        t_panels.append((r.upper(), ts_chart([(a, COLOURS[k], starts, tc[a]) for k, a in enumerate(ARMS)],
                                             ylabel="TERMINAL P&L, CNY", xlabel="START", yfmt=compact,
                                             tipfmt=lambda v: sgn(v, 0) + " CNY", zero=True, markers=True,
                                             width=380, height=260, label=f"Terminal P&L {r}")))
    for i, s in enumerate(starts):
        t_rows.append([s] + [compact(st.summary[r]["terminal_cny"][a][i]) for r in st.rules for a in ARMS])
    t_stats = [[r] + [compact(_mean(st.summary[r]["terminal_cny"][a])) for a in ARMS]
               + [compact(_mean(st.summary[r]["terminal_cny"]["reference"][i] - st.summary[r]["terminal_cny"]["test"][i]
                                for i in range(len(starts))))] for r in st.rules]
    pg = ts_chart(_rule_series(st, {r: st.summary[r]["premium_gap_cny"] for r in st.rules}, {r: starts for r in st.rules}),
                  ylabel="TEST PV − REF PV, CNY", xlabel="START", yfmt=compact, tipfmt=lambda v: sgn(v, 0) + " CNY",
                  zero=True, markers=True, label="Premium gap")
    pg_rows = [[s] + [signed(st.summary[r]["premium_gap_cny"][i]) for r in st.rules] for i, s in enumerate(starts)]
    none = st.rules[0]
    return section(6, "STAGE B", "Hedging along the real path", f"""
<p class="lead">On each of {len(starts)} starts the desk sells the 1Y ATM call (strike and units fixed at the start),
re-marks both models on the trailing window at every close and trades to three hedges over
{hm['sessions']} sessions. Every arm is marked to the reference PV, in excess of financing, so the paired difference of
two arms is exactly their hedge difference.</p>
{F_PNL}
{arms}
<p class="note">Day 0 (the premium at the arm's own PV vs the reference PV) is outside the daily series and reported
as the premium gap. No trading costs or lots; fractional shares.</p>
<h3 class="sub">Each arm's daily P&amp;L σ</h3>
{table(["rule", "test, CNY/day", "reference, CNY/day", "attribution, CNY/day", "observations"], arm_rows,
       "Std of the daily P&amp;L over all (start, session). It includes the reference re-mark jumps from daily "
       "recalibration, common to every arm.")}
<h3 class="sub">Paired differences · noise-corrected σ with 90% CI</h3>
<p>The paired difference d of two arms on the same day and start. Its MC-noise variance per observation
{mx("v")} comes from the reference engine's batch covariance; the corrected std is {F_CORR}. The 90% CI resamples
blocks of {hm['block']} calendar sessions ({hm['n_boot']:,} draws), keeping every live start of a session together.
A pair is resolved only when the CI's lower end is above 0.</p>
{corr_note}
{figure(ci, "Paired daily hedge-P&L σ by rule", "Point = σ, whisker = 90% CI, log scale. Hollow: not resolved (CI reaches 0).")}
{table(["rule", "pair", "σ, CNY/day", "90% CI", "resolved", "σ at measured noise", "raw σ", "noise σ", "mean"],
       pair_rows,
       "CNY/day per 1e8 notional. " + "; ".join(html.escape(PAIR_NOTES[k]) for k in PAIRS)
       + ". Raw σ includes MC noise; noise σ = √(mean per-observation noise variance), predicted from the engine's "
       "batch covariance. σ at measured noise: the same correction with the noise scaled by the probe's "
       "measured/predicted ratio — a sensitivity, largest where the noise is a big share of the raw σ (weekly, daily).")}
<h3 class="sub">MC noise probe</h3>
{probe}
<h3 class="sub">When the gap happens</h3>
{figure(when, f"Trailing {hm['block']}-session RMS of reference − test", "Every live start pooled; raw daily differences (MC noise included), log scale.")}
{table(["rule", "largest RMS, CNY/day", "window ending", "smallest RMS, CNY/day", "window ending"], when_rows)}
<h3 class="sub">Terminal P&amp;L per start</h3>
<p>Sold at the arm's own PV, cash at r, payoff paid at expiry. The {len(starts)} starts overlap (1Y options a month
apart), so these are 2–3 independent outcomes, descriptive only. All arms share the same short-gamma exposure to
realized vs model vol, which dominates the level.</p>
{multiples("Terminal P&L by start and arm", "CNY on 1e8 notional; each panel on its own scale.", t_panels)}
{table(["rule", "test mean", "reference mean", "attribution mean", "mean ref − test"], t_stats, "CNY, means over starts.")}
{table(["start"] + [f"{r} {a}" for r in st.rules for a in ARMS], t_rows, "Terminal P&amp;L, CNY.")}
<h3 class="sub">Premium gap</h3>
{figure(pg, "Test PV − reference PV at the start", f"ATM, CNY. Mean under {none}: {sgn(_mean(st.summary[none]['premium_gap_cny']), 0)} CNY.")}
{table(["start"] + st.rules, pg_rows, "CNY on 1e8 notional.")}
""")


def section_business(st: Study) -> str:
    rows = []
    for r in st.rules:
        p = st.pair(r, "reference-test")
        prem = st.premium(r)
        arm = st.summary[r]["arm_daily_std_cny"]["reference"]
        year = p["v"] * math.sqrt(st.hedge["meta"]["sessions"])
        unres = p["resolved"] is False
        rows.append([r, pair_cell(p), NR if unres else pct(p["v"] / arm, 1), NR if unres else compact(year),
                     NR if unres else pct(year / prem, 2), compact(prem),
                     signed(_mean(st.summary[r]["premium_gap_cny"])),
                     pair_cell(st.pair(r, "reference-attribution")), pair_cell(st.pair(r, "attribution-test"))])
    none, daily = st.rules[0], st.rules[-1]
    pn, pd_ = st.pair(none, "reference-test"), st.pair(daily, "reference-test")
    ordered = sorted(st.rules, key=lambda r: st.pair(r, "reference-test")["v"])
    split_dom, vol_dom, comparable, unres_rules = st.dominance()
    if split_dom:
        split_txt = "; ".join(f"{pair_text(st.pair(r, 'reference-attribution'))} under {r}" for r in split_dom)
        gap_txt = "; ".join(f"{signed(_mean(st.summary[r]['premium_gap_cny']))} CNY under {r}" for r in split_dom)
        buys_li = (f"<li><b>What the reference buys for the rest.</b> Under {', '.join(split_dom)} the split "
                   f"dominates. A per-name model (or at least per-name deltas from it) removes split noise of "
                   f"{split_txt}. The quote is affected too: the premium gap averages {gap_txt}.</li>")
    else:
        buys_li = ""
    comp_li = (f"<li><b>In between.</b> Under {', '.join(comparable)} the split and total-delta parts are "
               "comparable (90% CIs overlap): both a per-name split and a better index vol would help.</li>"
               if comparable else "")
    unres_txt = (f"<li><b>Below the MC noise.</b> Under {', '.join(unres_rules)} neither part is resolved: the "
                 "reference cannot tell the models apart at these path counts.</li>" if unres_rules else "")
    return section(7, "BUSINESS", "Business read", f"""
<p class="lead">The cost of running the CSI book on the single-underlying model is the extra hedge P&amp;L noise of its
deltas against the reference. On 1e8 CNY that is {pair_text(pn)} with fixed shares and {pair_text(pd_)} with daily
rebalancing; by point estimate, from smallest to largest: {", ".join(ordered)}.</p>
{table(["rule", "ref − test σ, CNY/day", "÷ ref arm σ", "× √244, CNY", "÷ premium", "mean premium", "premium gap",
        "split σ", "total-Δ σ"], rows,
       "√244 scaling assumes independent days: an order of magnitude for a year's accumulated hedge difference, "
       "not a measurement. Premium: mean reference PV of the ATM call at the starts. Split σ = reference − "
       "attribution; total-Δ σ = attribution − test.")}
<ul class="points">
<li><b>Where the single-underlying model is enough.</b> Under {", ".join(vol_dom) or "no rule"} the per-name split
adds less than the total-delta part: the index-level order is the right structure, and the remaining gap sits in
{mx("Δ_I")} (the index vol and the index-level model), not in the split. Whether a better index-vol estimate closes
it is not measured here.</li>
{buys_li}
{comp_li}
{unres_txt}
<li><b>No pass/fail line.</b> The figures sit next to the premium; what is good enough is the desk's call.</li>
</ul>
<p>The mechanism (why the split error lives only until the next rebalance and is linear in the stub) is derived and
checked on a synthetic basket in the sibling study:
<a class="link" href="{SYNTH_REPORT}">CSI basket delta · synthetic study report</a>.</p>
""")


def section_caveats(st: Study) -> str:
    fb = sum(x["calendar_fallback"] for x in st.recs)
    hm = st.hedge["meta"]
    return section(8, "CAVEATS", "Caveats", f"""
<ul class="points dim">
<li><b>GBM reference.</b> The truth is a per-name GBM with historical vols and correlation. No smile, no jumps, no
stochastic vol; both models would move under implied vols.</li>
<li><b>Historical, not implied, vols.</b> Both models use the same trailing {st.gap['meta']['window']}-session
window; a desk quoting off implied vols would see a different level gap.</li>
<li><b>One period.</b> Evaluation {st.dates[0]} onward only, one public test basket.</li>
<li><b>Overlapping starts.</b> The {len(hm['starts'])} Stage B starts are a month apart with 1Y options: terminal P&amp;L
has only 2–3 independent outcomes. The daily paired statistic carries the evidence.</li>
<li><b>Calendar fallback.</b> {fb} Stage A records need sessions past {st.gap['meta']['calendar_horizon']}; those use
weekdays (no holidays), so their stubs and expiries can be off by holidays.</li>
<li><b>MC noise.</b> The reference deltas carry MC noise. Stage B removes its variance from the paired σ (validated
by the independent-seed probe); the Stage A gap σ and the session RMS chart are raw.</li>
<li><b>No costs or lots.</b> Fractional shares, no transaction costs, no 100-share lots.</li>
</ul>
""")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

NAV = [(1, "SUMMARY"), (2, "SETUP"), (3, "INDEX"), (4, "STAGE A"), (5, "SPLIT"), (6, "STAGE B"), (7, "BUSINESS"),
       (8, "CAVEATS")]


def masthead(st: Study) -> tuple:
    meta, hm = st.ix["meta"], st.hedge["meta"]
    none, daily = st.rules[0], st.rules[-1]
    pn, pd_ = st.pair(none, "reference-test"), st.pair(daily, "reference-test")
    pv_none = _mean(st.pv_gap(r) for r in st.atm(none))

    def tile(p):
        return f"&lt;{num(p['hi'])}" if p["resolved"] is False else num(p["v"])

    def tag(p):
        if p["resolved"] is False:
            return "not resolved · 90% CI upper end"
        return "noise-corrected" if p["corrected"] else "raw"

    def say(p):
        if p["resolved"] is False:
            return f"an amount not resolved from MC noise (90% CI up to {num(p['hi'])} CNY/day)"
        return f"σ <b>{pair_text(p)}</b>"

    kpis = [
        (tile(pn), "CNY/DAY", f"Hedge gap σ ref − test · rule {none} · {tag(pn)}"),
        (tile(pd_), "CNY/DAY", f"Hedge gap σ ref − test · rule {daily} · {tag(pd_)}"),
        (spct(pv_none, 2), "PV", f"Black vs reference ATM PV · rule {none} · mean of {len(st.dates)} dates"),
        (pct(st.ix["terminal_spread"], 2), "", "Spread of the five rules' final index levels · same prices"),
    ]
    tiles = "".join(f'<div class="kpi"><div class="kv">{v}<small>{u}</small></div><div class="kl">{gk(html.escape(l))}</div></div>'
                    for v, u, l in kpis)
    nav = "".join(f'<a href="#s{n}"><b>{n})</b> {t}</a>' for n, t in NAV)
    chrome = f"""
<div class="top">
<header class="fbar">
  <span class="brand">QUANTARK</span><span class="cmd">CSI REAL &lt;GO&gt;</span>
  <span class="fld">STUDY <b>csi_real_data</b></span><span class="fld">DATA <b>AKSHARE {html.escape(meta['akshare'])} · HFQ</b></span>
  <span class="fld">SESSIONS <b>{meta['data_first']} → {meta['data_last']}</b></span>
  <span class="fld">PATHS <b>{hm['n_paths']:,} × {hm['n_batches']}</b></span><span class="fld date">{meta['fetched_on']}</span>
  <button class="theme" id="theme" type="button" aria-label="Switch colour theme">◐ <b>LIGHT</b></button>
</header>
<nav class="menu" id="menu">{nav}</nav>
</div>
"""
    hero = f"""
<div class="hero">
  <div class="kicker">CSI · REAL DATA · HEDGE</div>
  <h1>Is one index-level model enough to hedge a real CSI?</h1>
  <p class="answer">On real {meta['data_first'][:4]}–{meta['data_last'][:4]} prices the five rebalance rules end
  {pct(st.ix['terminal_spread'], 2)} apart. Hedging per name from the reference instead of Black on the index changes the
  daily hedge P&amp;L on 1e8 CNY by {say(pn)} with fixed shares and by {say(pd_)} with daily rebalancing. The rule,
  not the model, decides how much a per-name model is worth.</p>
</div>
<div class="kpis">{tiles}</div>
"""
    return chrome, hero


EXTRA_CSS = """
:root{--s4:#7f9a1c;--s5:#8a6fe0}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){--s4:#5f8a10;--s5:#6d4fd0}}
:root[data-theme="light"]{--s4:#5f8a10;--s5:#6d4fd0}
.chart .wh{stroke-width:2;stroke-linecap:butt}
.chart .diag{stroke:var(--ink-3);stroke-width:1;stroke-dasharray:4 4}
.chart .pt{stroke:var(--bg);stroke-width:1;pointer-events:none}
a.link{color:var(--amber);border-bottom:1px solid var(--amber-dim)}
a.link:hover{background:var(--amber);color:#000}
"""


def build_html(results_dir: Path = RESULTS_DIR) -> str:
    """Read the results JSON in ``results_dir`` and return the full report page."""
    st = Study(load_results(results_dir))
    chrome, hero = masthead(st)
    content = (
        hero + section_summary(st) + section_setup(st) + section_index(st) + section_gap(st) + section_split(st)
        + section_hedge(st) + section_business(st) + section_caveats(st)
        + "<footer>QUANTARK · example/case_studies/csi_real_data · stages 02–04 → results/*.json → 05_report.py · "
          "public test basket, akshare hfq closes, no desk data</footer>"
    )
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>CSI Real Data</title>{H.THEME_INIT_JS}<style>{H.CSS}{EXTRA_CSS}</style></head>"
            f"<body>{chrome}<main>{content}</main>{H.PAGE_JS}</body></html>")


def main() -> None:
    page = build_html()
    REPORT_PATH.write_text(page, encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
