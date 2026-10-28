"""Render the local-vol transition QUAD study as one self-contained HTML page.

Reads the result JSONs in this folder (``reference.json``, ``bakeoff.json``,
``tuning.json``, ``benchmark.json``) plus the prose in ``narrative.json`` and
writes ``report.html``. Every number on the page comes from the JSONs; the
renderer states no conclusion of its own. Missing or partial inputs render as
"pending" notices instead of failing, so the page can be rebuilt while the
tuning / benchmark jobs are still running.

The page follows ``docs/UI_LANGUAGE.md`` (Bloomberg x Swiss): a frozen function
bar + numbered menu, a light/dark switcher, inline SVG charts styled by CSS
tokens with a hover layer, and a table under every chart. Tokens, CSS and JS
are copied from ``example/case_studies/snowball_ftp_study/05_report.py``.

    python docs/research/localvol_transition_quad/render_report.py [--dir <folder>] [--out report.html]
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

MINUS = "−"
# Engine -> fixed series slot, the same on every chart of the page.
ENGINES = ("quad", "pde", "mc")
ENGINE_NAME = {"quad": "QUAD", "pde": "PDE", "mc": "MC"}
SLOT = {"quad": "var(--s1)", "pde": "var(--s2)", "mc": "var(--s3)"}
ARM_SLOTS = ("var(--s1)", "var(--s2)", "var(--s3)")
QUANTITIES = ("pv", "delta", "gamma", "vega", "theta", "rho", "carry")
CASE_ORDER = ("snowball_1y", "snowball_2y", "put_1y", "up_out_1y")
DASH = "–"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load(folder: Path, name: str):
    path = folder / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None        # a file mid-write renders as pending, never as a crash


def _cases(*sources) -> list[str]:
    seen = []
    for src in sources:
        for name in (src or {}).get("cases", {}) or {}:
            if name not in seen:
                seen.append(name)
    return sorted(seen, key=lambda n: (CASE_ORDER.index(n) if n in CASE_ORDER else len(CASE_ORDER), n))


def case_title(name: str) -> str:
    return name.replace("_", " ").upper()


# ---------------------------------------------------------------------------
# Number formatting (true minus; never "nan"/"None" on the page)
# ---------------------------------------------------------------------------


def _ok(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _m(text: str) -> str:
    return text.replace("-", MINUS)


def sig(v, n=4) -> str:
    """Significant figures, compact exponent (``1.23e−5``)."""
    if not _ok(v):
        return DASH
    if v == 0:
        return "0"
    s = f"{v:.{n}g}"
    if "e" in s:
        mant, exp = s.split("e")
        s = f"{mant}e{int(exp)}"
    return _m(s)


def fixed(v, d=3) -> str:
    if not _ok(v):
        return DASH
    v = round(v, d) + 0.0
    return _m(f"{v:,.{d}f}")


def ratio(v) -> str:
    """An error expressed in budgets."""
    if not _ok(v):
        return DASH
    if v == 0:
        return "0"
    return sig(v, 2) if abs(v) < 0.01 else fixed(v, 3)


def secs(v) -> str:
    if not _ok(v):
        return DASH
    return fixed(v, 3) if v < 10 else fixed(v, 1)


def bp(v) -> str:
    """A PV error per unit notional, in basis points of notional (1 bp = 1e-4)."""
    if not _ok(v):
        return DASH
    return signed_plain(v * 1e4, 3)


def signed_plain(v, d=3) -> str:
    if not _ok(v):
        return DASH
    v = round(v, d) + 0.0
    return _m(f"{v:+.{d}f}") if v else f"{0:.{d}f}"


def speedup(v) -> str:
    return DASH if not _ok(v) else (f"{v:.1f}×" if v < 100 else f"{v:.0f}×")


def flag(ok) -> str:
    if ok is None:
        return '<span class="dim">–</span>'
    return '<span class="up">YES</span>' if ok else '<span class="dn">NO</span>'


def mb(nbytes) -> str:
    return DASH if not _ok(nbytes) else f"{nbytes / 2 ** 20:,.0f}"


def config_label(engine: str, config) -> str:
    if not config:
        return DASH
    a, b = config[0], config[1]
    if engine == "quad":
        return f"{a} cells · ±{float(b):g}σ"
    if engine == "pde":
        return f"{a} pts · {b}/day"
    if engine == "mc":
        paths = int(b)
        exp = math.log2(paths) if paths > 0 else 0
        p = f"2^{int(exp)}" if exp == int(exp) else f"{paths:,}"
        return f"{a} sub · {p} paths"
    return "×".join(str(x) for x in config)


def ladder_label(engine: str, key: str) -> str:
    """Reference-ladder keys: PDE ``points x steps/day``, QUAD ``cells x subdivide``."""
    try:
        a, b = key.split("x")
    except ValueError:
        return key
    return f"{a} pts · {b}/day" if engine == "pde" else f"{a} cells · sub {b}"


# ---------------------------------------------------------------------------
# Narrative text: escaped, with `code` and **bold** as the only markup
# ---------------------------------------------------------------------------


def prose(text) -> str:
    if text is None:
        return ""
    s = html.escape(str(text))
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    return s


def paragraphs(text, cls="") -> str:
    if text is None:
        return ""
    parts = text if isinstance(text, list) else [p for p in str(text).split("\n\n") if p.strip()]
    c = f' class="{cls}"' if cls else ""
    return "".join(f"<p{c}>{prose(p)}</p>" for p in parts)


def pending(what: str, how: str = "") -> str:
    tail = f" <code>{html.escape(how)}</code>" if how else ""
    return f'<div class="pending"><b>PENDING</b> {html.escape(what)}{tail}</div>'


# ---------------------------------------------------------------------------
# SVG charts
# ---------------------------------------------------------------------------


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo
    if span <= 0:
        span = abs(hi) or 1.0
        lo, hi = lo - span / 2, hi + span / 2
        span = hi - lo
    raw = span / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.floor(lo / step) * step
    ticks, t = [], start
    while t <= hi + step * 1e-9:
        ticks.append(round(t, 12))
        t += step
    return ticks


class _Scale:
    """Maps a data interval onto pixels ``a -> b``; log scales snap to whole decades."""

    def __init__(self, lo: float, hi: float, a: float, b: float, log: bool = False):
        self.a, self.b, self.log = a, b, log
        if log:
            k0, k1 = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
            if k1 == k0:
                k1 += 1
            self.lo, self.hi = float(k0), float(k1)
            ticks = []
            for k in range(k0, k1 + 1):
                ticks.append(10.0 ** k)
                if k1 - k0 <= 2 and k < k1:
                    ticks += [2 * 10.0 ** k, 5 * 10.0 ** k]
            self.ticks = sorted(ticks)
        else:
            ticks = _nice_ticks(lo, hi)
            self.lo, self.hi = min(ticks[0], lo), max(ticks[-1], hi)
            self.ticks = ticks

    def __call__(self, v: float) -> float:
        t = math.log10(v) if self.log else v
        return self.a + (t - self.lo) / (self.hi - self.lo) * (self.b - self.a)


def logfmt(v: float) -> str:
    if 1e-3 <= v < 1e4:
        return _m(f"{v:g}")
    return _m(f"1e{int(round(math.log10(v)))}") if math.isclose(v, 10 ** round(math.log10(v))) else _m(f"{v:.0e}")


def _legend(items) -> str:
    """``items``: [(name, colour, hollow)]."""
    out = []
    for name, colour, hollow in items:
        style = f"border:1.5px solid {colour};height:8px" if hollow else f"background:{colour}"
        out.append(f'<span><i style="{style}"></i>{html.escape(name)}</span>')
    return f'<div class="legend">{"".join(out)}</div>'


def _svg(width, height, body, label) -> str:
    return (f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)}">'
            + "".join(body) + "</svg>")


def xy_chart(series, *, xlabel, ylabel, xlog=False, ylog=False, xfmt=None, yfmt=None, xticks=None,
             hlines=(), ydomain=None, width=760, height=300, label="", right=96, xpad=None) -> str:
    """Line / scatter chart.

    ``series``: [dict(name, colour, line, pts=[dict(x, y, tip, hi)])]. Points with a non-positive
    coordinate on a log axis are left out (they are in the table). ``hi`` rings a point in amber.
    ``hlines``: [(y, label)] reference lines (budget, 1/5 budget).
    """
    keep = lambda p: _ok(p["x"]) and _ok(p["y"]) and (p["x"] > 0 or not xlog) and (p["y"] > 0 or not ylog)
    series = [dict(s, pts=[p for p in s["pts"] if keep(p)]) for s in series]
    pts = [p for s in series for p in s["pts"]]
    if not pts:
        return '<p class="note">No priced points to chart yet.</p>'
    left, top, bottom = 60, 30, 42
    xs = [p["x"] for p in pts] + [v for v, _ in (xticks or [])]
    ys = [p["y"] for p in pts] + [v for v, _ in hlines]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = ydomain or (min(ys), max(ys))
    if not xlog:
        pad = xpad if xpad is not None else (xhi - xlo) * 0.06 or 0.5
        xlo, xhi = xlo - pad, xhi + pad
    if not ylog and ydomain is None:
        pad = (yhi - ylo) * 0.06 or abs(yhi) * 0.1 or 1.0
        ylo, yhi = (ylo if ylo >= 0 and ylo - pad < 0 else ylo - pad), yhi + pad   # zero-anchored: no pad below 0
    X = _Scale(xlo, xhi, left, width - right, xlog)
    if xticks and not xlog:
        X.lo, X.hi = xlo, xhi            # explicit ticks: the padded data range IS the domain, no nice-tick widening
    Y = _Scale(ylo, yhi, height - bottom, top, ylog)
    xfmt = xfmt or (logfmt if xlog else (lambda v: _m(f"{v:g}")))
    yfmt = yfmt or (logfmt if ylog else (lambda v: _m(f"{v:g}")))
    body = []
    for v in Y.ticks:
        y = Y(v)
        body.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
                    f'<text class="tick" x="{left - 8}" y="{y + 3.5:.1f}" text-anchor="end">{yfmt(v)}</text>')
    base = height - bottom
    body.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{base}" y2="{base}"/>')
    for v, lab in (xticks or [(t, xfmt(t)) for t in X.ticks]):
        x = X(v)
        body.append(f'<line class="axis" x1="{x:.1f}" x2="{x:.1f}" y1="{base}" y2="{base + 4}"/>'
                    f'<text class="tick" x="{x:.1f}" y="{base + 16}" text-anchor="middle">{html.escape(lab)}</text>')
    for v, lab in hlines:
        y = Y(v)
        body.append(f'<line class="ref" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
                    f'<text class="reflab" x="{width - right + 6}" y="{y + 3.5:.1f}">{html.escape(lab)}</text>')
    body.append(f'<text class="axlab" x="{left}" y="{height - 4}">{html.escape(xlabel)}</text>')
    body.append(f'<text class="axlab" x="{left - 52}" y="12">{html.escape(ylabel)}</text>')
    ends = []
    for s in series:
        xy = [(X(p["x"]), Y(p["y"]), p) for p in s["pts"]]
        if s.get("line") and len(xy) > 1:
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in xy)
            body.append(f'<polyline class="ln" points="{path}" style="stroke:{s["colour"]}"/>')
        for x, y, p in xy:
            if p.get("hi"):
                body.append(f'<circle class="hiring" cx="{x:.1f}" cy="{y:.1f}" r="8.5"/>')
            mstyle = (f"fill:var(--bg);stroke:{s['colour']}" if s.get("hollow") else f"fill:{s['colour']}")
            body.append(f'<circle class="mk" cx="{x:.1f}" cy="{y:.1f}" r="4.5" style="{mstyle}"/>')
            tip = html.escape(p.get("tip", ""))
            body.append(f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="12" data-tip="{tip}" aria-label="{tip}"/>')
        if s.get("line") and xy and len(series) >= 2:
            ends.append([xy[-1][1], xy[-1][0], s["name"]])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 13)
    for y, x, name in ends:
        body.append(f'<text class="dlab" x="{x + 10:.1f}" y="{y + 3.5:.1f}">{html.escape(name)}</text>')
    legend = (_legend([(s["name"], s["colour"], s.get("hollow", False)) for s in series])
              if len(series) >= 2 else "")
    return legend + _svg(width, height, body, label)


def bar_chart(categories, series, *, ylabel, xlabel, valfmt, width=760, height=300, label="") -> str:
    """Grouped bars from zero. ``series``: [dict(name, colour, hollow, values=[v|None per category])]."""
    vals = [v for s in series for v in s["values"] if _ok(v)]
    if not vals:
        return '<p class="note">Nothing timed yet.</p>'
    left, right, top, bottom = 60, 24, 34, 42
    Y = _Scale(0.0, max(vals) * 1.12, height - bottom, top)
    body = []
    for v in Y.ticks:
        y = Y(v)
        body.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>'
                    f'<text class="tick" x="{left - 8}" y="{y + 3.5:.1f}" text-anchor="end">{_m(f"{v:g}")}</text>')
    base = height - bottom
    body.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{base}" y2="{base}"/>')
    n, k = len(categories), len(series)
    slot = (width - left - right) / n
    bw = min(slot * 0.8 / k, 46)
    for i, cat in enumerate(categories):
        cx = left + slot * (i + 0.5)
        x0 = cx - bw * k / 2
        for j, s in enumerate(series):
            v = s["values"][i]
            if not _ok(v):
                continue
            x = x0 + j * bw + 1                     # 2px surface gap between neighbours
            y = Y(v)
            tip = html.escape(f"{s['name']} · {case_title(cat)} · {valfmt(v)}")
            fill = (f"fill:var(--bg);stroke:{s['colour']};stroke-width:1.5" if s.get("hollow")
                    else f"fill:{s['colour']}")
            body.append(f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{bw - 2:.1f}" '
                        f'height="{max(base - y, 0.5):.1f}" style="{fill}" data-tip="{tip}" aria-label="{tip}"/>')
        body.append(f'<text class="tick" x="{cx:.1f}" y="{base + 16}" text-anchor="middle">'
                    f'{html.escape(case_title(cat))}</text>')
    body.append(f'<text class="axlab" x="{left}" y="{height - 4}">{html.escape(xlabel)}</text>')
    body.append(f'<text class="axlab" x="{left - 52}" y="12">{html.escape(ylabel)}</text>')
    legend = _legend([(s["name"], s["colour"], s.get("hollow", False)) for s in series])
    return legend + _svg(width, height, body, label)


# ---------------------------------------------------------------------------
# HTML building blocks
# ---------------------------------------------------------------------------


def table(headers, rows, note="", text=False, cls="") -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    foot = f'<p class="note">{note}</p>' if note else ""
    classes = " ".join(c for c in ("text" if text else "", cls) if c)
    attr = f' class="{classes}"' if classes else ""
    return f'<div class="scroll"><table{attr}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{foot}'


def figure(svg_html, title, caption) -> str:
    return (f'<figure><figcaption><b>{html.escape(title)}</b><span>{html.escape(caption)}</span></figcaption>'
            f"{svg_html}</figure>")


def section(no: int, code: str, title: str, content: str) -> str:
    return (f'<section id="s{no}"><div class="rail"><div class="numeral">{no:02d}</div>'
            f'<div class="code">{code}</div><h2>{html.escape(title)}</h2></div><div class="body">{content}</div></section>')


def note_for(narr: dict, code: str) -> str:
    """Optional per-section paragraph from ``narrative.notes[<CODE>]``."""
    notes = narr.get("notes") or {}
    return paragraphs(notes.get(code), "lead") if notes.get(code) else ""


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


def budgets(reference) -> dict:
    return {n: c.get("assessment", {}).get("budget", {}) for n, c in (reference or {}).get("cases", {}).items()}


def worst_ratio(values: dict, ref: dict, budget: dict, keys=QUANTITIES):
    """Worst |value - ref| / budget over ``keys`` and the binding quantity."""
    best, binding = None, None
    for k in keys:
        if k in values and k in ref and _ok(budget.get(k)) and budget[k] > 0:
            r = abs(values[k] - ref[k]) / budget[k]
            if best is None or r > best:
                best, binding = r, k
    return best, binding


def _ladder_sort(key: str):
    try:
        return tuple(float(x) for x in key.split("x"))
    except ValueError:
        return (math.inf,)


def _median(xs):
    xs = [x for x in xs or [] if _ok(x)]
    return statistics.median(xs) if xs else None


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section_verdict(narr, bench) -> str:
    cases = (bench or {}).get("cases", {})

    def tile(value, unit, label, is_pending):
        v = '<span class="dim">PENDING</span>' if is_pending else value
        return (f'<div class="kpi"><div class="kv">{v}<small>{"" if is_pending else unit}</small></div>'
                f'<div class="kl">{html.escape(label)}</div></div>')

    def sp(name):
        v = cases.get(name, {}).get("speedup_vs_pde")
        return tile(speedup(v), "", f"{case_title(name)} · full risk · QUAD vs PDE, cold median", not _ok(v))

    s1 = cases.get("snowball_1y", {})
    qcold = s1.get("quad", {}).get("median_cold")
    bind = {e: str(s1[e]["accuracy"]["binding"]).upper() for e in ENGINES
            if isinstance(s1.get(e), dict) and (s1[e].get("accuracy") or {}).get("binding")}
    others = " · ".join(f"{ENGINE_NAME[e]} {b}" for e, b in bind.items() if e != "quad")
    tiles = (sp("snowball_1y") + sp("snowball_2y")
             + tile(secs(qcold), "S", "QUAD cold full risk · SNOWBALL 1Y · median", not _ok(qcold))
             + tile(html.escape(bind.get("quad", DASH)), html.escape(others),
                    "Binding quantity · QUAD (others small) · timed config · SNOWBALL 1Y", not bind))
    body = paragraphs(narr.get("verdict"), "lead") or pending("narrative.json has no verdict yet.")
    if bench is None:
        body += pending("benchmark.json not produced yet; the KPI tiles fill in once the timing run lands.")
    return section(1, "VERDICT", "Verdict", body + f'<div class="kpis">{tiles}</div>')


def section_method(narr, reference) -> str:
    mkt = (reference or {}).get("market", {})
    env = (reference or {}).get("environment", {})
    spec = []
    if mkt:
        spec.append(("Market", f"S0 {fixed(mkt.get('spot'), 3)} · r {fixed(mkt.get('r', 0) * 100, 2)}% · "
                               f"q {fixed(mkt.get('q', 0) * 100, 3)}% (flat)"))
    if env.get("fixture"):
        spec.append(("Fixture", f"<code>{html.escape(env['fixture'])}</code>"))
    spec.append(("Budget", "PV 1 bp of notional (1e−4 per unit); Greeks max(1% of reference, absolute floor); "
                           "MC meets only if |bias| + 2·SE ≤ budget"))
    spec.append(("Theory", 'Equations, literature review and contract scope: <a class="lnk" href="theory.md">'
                           'theory.md</a> (same folder)'))
    dl = '<dl class="spec">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in spec) + "</dl>"
    body = paragraphs(narr.get("method")) or pending("narrative.json has no method text yet.")
    return section(2, "METHOD", "Method", body + dl)


def section_reference(narr, reference) -> str:
    cases = (reference or {}).get("cases", {})
    if not cases:
        return section(3, "REF", "Reference & convergence", pending("reference.json missing."))
    panels, ladder_rows, case_tables = [], [], []
    all_y = []
    per_case = {}
    for name in _cases(reference):
        c = cases[name]
        a = c.get("assessment", {})
        ref, bud = a.get("reference", {}), a.get("budget", {})
        series = []
        for e in ("pde", "quad"):
            rungs = sorted((c.get(e) or {}).items(), key=lambda kv: _ladder_sort(kv[0]))
            pts = []
            for i, (key, row) in enumerate(rungs, 1):
                w, b = worst_ratio(row.get("values", {}), ref, bud)
                tip = f"{ENGINE_NAME[e]} · {case_title(name)} · {ladder_label(e, key)} · worst {ratio(w)} ({b or DASH})"
                pts.append(dict(x=i, y=w, tip=tip))
                if _ok(w) and w > 0:
                    all_y.append(w)
                ladder_rows.append([case_title(name), ENGINE_NAME[e], f"R{i}", ladder_label(e, key),
                                    secs(row.get("seconds")), str(row.get("cells", DASH)),
                                    "= reference" if w == 0 else ratio(w), (b or DASH).upper() if w else DASH])
            series.append(dict(name=ENGINE_NAME[e], colour=SLOT[e], line=True, pts=pts))
        per_case[name] = series
        rows = []
        for q in QUANTITIES:
            if q not in ref:
                continue
            ps, qg, bq = a.get("pde_step", {}).get(q), a.get("quad_gap", {}).get(q), bud.get(q)
            rows.append([q.upper(), sig(ref[q], 6), sig(bq, 3), sig(ps, 3),
                         ratio(abs(ps) / bq) if _ok(ps) and _ok(bq) and bq else DASH, sig(qg, 3),
                         ratio(abs(qg) / bq) if _ok(qg) and _ok(bq) and bq else DASH])
        flags = (f'<span class="dim">PDE CONVERGED</span> {flag(a.get("pde_converged"))} · '
                 f'<span class="dim">QUAD CONFIRMS</span> {flag(a.get("quad_confirms"))}')
        case_tables.append(f'<h3 class="sub">{case_title(name)} <span class="flags">{flags}</span></h3>'
                           + table(["quantity", "reference", "budget", "PDE step", "step/budget", "QUAD gap",
                                    "gap/budget"], rows))
    ydom = (min(all_y + [0.2]), max(all_y + [1.0])) if all_y else None
    for name, series in per_case.items():
        n = max((len(s["pts"]) for s in series), default=1)
        panels.append(f'<div><div class="ptitle">{case_title(name)}</div>' + xy_chart(
            series, xlabel="REFINEMENT RUNG", ylabel="WORST |ERR| / BUDGET", ylog=True,
            xticks=[(i, f"R{i}") for i in range(1, n + 1)], hlines=[(1.0, "BUDGET"), (0.2, "1/5")], xpad=0.3,
            ydomain=ydom, width=380, height=270, right=52, label=f"Convergence ladder {name}") + "</div>")
    chart = (f'<figure><figcaption><b>Error vs refinement rung, against the reference</b><span>Worst over the seven '
             f'full-risk outputs; log scale; the finest PDE rung IS the reference (zero, not drawn).</span>'
             f'</figcaption><div class="multiples">{"".join(panels)}</div></figure>')
    ladder = table(["case", "engine", "rung", "config", "seconds", "cells", "worst/budget", "binding"], ladder_rows,
                   "Reference = finest PDE rung. Step = last two PDE rungs; gap = finest QUAD minus reference. "
                   "Convergence and confirmation pass at 1/5 of each budget.")
    return section(3, "REF", "Reference & convergence",
                   note_for(narr, "REF") + chart + ladder + "".join(case_tables))


def _bakeoff_budget(bakeoff, reference):
    ref = (bakeoff or {}).get("reference", {})
    for c in (reference or {}).get("cases", {}).values():
        r = c.get("assessment", {}).get("reference", {})
        if r and all(_ok(r.get(k)) and _ok(ref.get(k)) and r[k] == ref[k] for k in ("pv", "delta", "gamma")):
            return c["assessment"].get("budget", {})
    return {}


def section_bakeoff(narr, bakeoff, reference) -> str:
    if not bakeoff:
        return section(4, "BAKEOFF", "Kernel bake-off", pending("bakeoff.json missing."))
    bud = _bakeoff_budget(bakeoff, reference)
    ref = bakeoff.get("reference", {})
    dec = bakeoff.get("decision", {})
    matrix, winner = dec.get("matrix", {}), dec.get("winner")
    arms = list(matrix) or sorted({r["arm"] for r in bakeoff.get("rows", [])})
    colour = {a: ARM_SLOTS[i % 3] for i, a in enumerate(arms)}
    series, rows, refused = [], [], []
    for a in arms:
        pts = []
        for r in bakeoff.get("rows", []):
            if r.get("arm") != a:
                continue
            cfg = f"{r.get('cells_base')} cells · sub {r.get('subdivide')}"
            if r.get("status") != "priced":
                rows.append([a, cfg, DASH, DASH, DASH, DASH, DASH, DASH, '<span class="dn">REFUSED</span>'])
                refused.append([html.escape(a), cfg, html.escape(r.get("reason", ""))])
                continue
            vals = {k: ref[k] + v for k, v in (r.get("error") or {}).items() if k in ref}
            w, b = worst_ratio(vals, ref, bud, ("pv", "delta", "gamma")) if bud else (None, None)
            err = r.get("error") or {}
            pts.append(dict(x=r.get("seconds"), y=w, tip=f"{a} · {cfg} · {secs(r.get('seconds'))} s · worst {ratio(w)}"))
            rows.append([a, cfg, str(r.get("cells", DASH)), secs(r.get("seconds")), bp(err.get("pv")),
                         sig(err.get("delta"), 3), sig(err.get("gamma"), 3), ratio(w),
                         flag(r.get("meets_budget"))])
        series.append(dict(name=a, colour=colour[a], line=False, pts=pts))
    chart = xy_chart(series, xlabel="BASE-SOLVE SECONDS (MEDIAN)", ylabel="WORST |ERR| / BUDGET · PV Δ Γ",
                     xlog=True, ylog=True, hlines=[(1.0, "BUDGET")], label="Kernel bake-off time vs error")
    mrows, indep = [], []
    for a in arms:
        m = matrix.get(a, {})
        cm = m.get("cheapest_meeting") or {}
        err = cm.get("error") or {}
        vals = {k: ref[k] + v for k, v in err.items() if k in ref}
        w, _ = worst_ratio(vals, ref, bud, ("pv", "delta", "gamma")) if bud and cm else (None, None)
        name = f'<b class="win">{html.escape(a)} ◆</b>' if a == winner else html.escape(a)
        cfg = f"{cm.get('cells_base')} cells · sub {cm.get('subdivide')}" if cm else '<span class="dn">none</span>'
        mrows.append([name, cfg, str(cm.get("cells", DASH)), secs(cm.get("seconds")), bp(err.get("pv")),
                      sig(err.get("delta"), 3), sig(err.get("gamma"), 3), ratio(w), str(m.get("refused_configs", DASH))])
        indep.append([name, html.escape(m.get("independence", "") or DASH)])
    matrix_tbl = table(["kernel", "cheapest meeting", "cells", "seconds", "PV err bp", "Δ err", "Γ err",
                        "worst/budget", "refused"], mrows,
                       f"◆ winner: <b>{html.escape(str(winner or DASH))}</b>. Base solve only (PV, Δ, Γ), "
                       "SNOWBALL 1Y. PV errors in bp of notional.")
    indep_tbl = table(["kernel", "independence from the production PDE"], indep, text=True)
    all_tbl = table(["kernel", "config", "cells", "seconds", "PV err bp", "Δ err", "Γ err", "worst/budget", "meets"],
                    rows, "Every bake-off row. PV errors in bp of notional; Δ and Γ errors per unit notional.")
    refused_tbl = (table(["kernel", "config", "refusal reason (engine)"], refused, text=True) if refused else "")
    return section(4, "BAKEOFF", "Kernel bake-off",
                   note_for(narr, "BAKEOFF") + '<h3 class="sub first">Decision matrix</h3>' + matrix_tbl + indep_tbl
                   + figure(chart, "Time vs error, every priced configuration",
                            "Log–log. Below the budget line meets PV, Δ and Γ.") + all_tbl + refused_tbl)


def section_tuning(narr, tuning, reference) -> str:
    if not tuning or not tuning.get("cases"):
        return section(5, "TUNE", "Matched-accuracy tuning",
                       pending("tuning.json missing.",
                               "python -m example.case_studies.localvol_transition_quad.benchmark tune"))
    buds = budgets(reference)
    blocks = []
    for name in _cases(tuning):
        c = tuning["cases"][name]
        bud = buds.get(name, {})
        series, rows, refused = [], [], []
        for e in ENGINES:
            rungs = c.get(e)
            if not isinstance(rungs, dict):
                continue
            chosen = c.get(f"{e}_chosen")
            pts = []
            for key, r in rungs.items():
                cfg = r.get("config")
                is_chosen = chosen is not None and cfg == chosen
                lab = config_label(e, cfg)
                if r.get("status") != "priced":
                    rows.append([ENGINE_NAME[e], html.escape(lab), DASH, DASH, DASH, DASH,
                                 '<span class="dn">REFUSED</span>'])
                    refused.append([ENGINE_NAME[e], html.escape(lab), html.escape(r.get("reason", ""))])
                    continue
                w, b = r.get("worst"), r.get("binding")
                se_share = DASH
                if r.get("stderr") and b and _ok(bud.get(b)) and bud[b] > 0 and _ok(r["stderr"].get(b)):
                    se_share = ratio(2 * r["stderr"][b] / bud[b])
                pts.append(dict(x=r.get("seconds"), y=w, hi=is_chosen,
                                tip=f"{ENGINE_NAME[e]} · {lab} · {secs(r.get('seconds'))} s · worst {ratio(w)} "
                                    f"({(b or DASH).upper()}){' · CHOSEN' if is_chosen else ''}"))
                cfg_cell = f'<b class="win">{html.escape(lab)} ◆</b>' if is_chosen else html.escape(lab)
                rows.append([ENGINE_NAME[e], cfg_cell, secs(r.get("seconds")), ratio(w), (b or DASH).upper(),
                             se_share, flag(r.get("meets"))])
            series.append(dict(name=ENGINE_NAME[e], colour=SLOT[e], line=False, pts=pts))
        status = []
        for e in ENGINES:
            if e not in c:
                continue
            key = f"{e}_chosen"
            if key not in c:
                status.append(f"{ENGINE_NAME[e]} <span class='dim'>ladder running</span>")
            elif c[key] is None:
                status.append(f"{ENGINE_NAME[e]} <span class='dn'>none meets the budget</span>")
            else:
                status.append(f"{ENGINE_NAME[e]} <b>{html.escape(config_label(e, c[key]))}</b>")
        chart = xy_chart(series, xlabel="FULL-RISK SECONDS (ONE RUN, TUNING)", ylabel="WORST |ERR| / BUDGET",
                         xlog=True, ylog=True, hlines=[(1.0, "BUDGET")], label=f"Tuning frontier {name}")
        blocks.append(figure(chart, f"{case_title(name)} · accuracy–cost frontier",
                             "Log–log; amber ring = timed (cheapest meeting) config. MC worst includes 2·SE.")
                      + f'<p class="note chosen">CHOSEN · {" · ".join(status) or DASH}</p>'
                      + table(["engine", "config", "seconds", "worst/budget", "binding", "of which 2·SE",
                               "meets"], rows,
                              "worst/budget = max over the seven outputs of (|err| + 2·SE for MC) / budget. "
                              "\"of which 2·SE\" is the MC statistical share of the binding quantity.")
                      + (table(["engine", "config", "refusal reason (engine)"], refused, text=True) if refused else ""))
    return section(5, "TUNE", "Matched-accuracy tuning", note_for(narr, "TUNE") + "".join(blocks))


def section_benchmark(narr, bench) -> str:
    how = "python -m example.case_studies.localvol_transition_quad.benchmark time --repeats 3"
    if not bench or not bench.get("cases"):
        return section(6, "BENCH", "Full-risk benchmark",
                       note_for(narr, "BENCH")
                       + pending("benchmark.json not produced yet — the cold/warm full-risk timing runs after "
                                 "tuning finishes. Run:", how))
    names = _cases(bench)
    cases = bench["cases"]
    engines = [e for e in ENGINES if any(isinstance(cases[n].get(e), dict) for n in names)]
    series = []
    for e in engines:
        series.append(dict(name=f"{ENGINE_NAME[e]} cold", colour=SLOT[e],
                           values=[(cases[n].get(e) or {}).get("median_cold") for n in names]))
        if e == "quad":
            series.append(dict(name="QUAD warm-moved", colour=SLOT[e], hollow=True,
                               values=[_median((cases[n].get(e) or {}).get("warm_moved")) for n in names]))
    timed = [v for s in series for v in s["values"] if _ok(v) and v > 0]
    log_scale = bool(timed) and max(timed) / min(timed) > 20
    if log_scale:
        # a >20x spread would flatten the fast engine on a linear bar axis: dot plot on a log axis instead
        k = len(series)
        dots = [dict(name=s["name"], colour=s["colour"], hollow=s.get("hollow", False), line=False,
                     pts=[dict(x=i + (j - (k - 1) / 2) * 0.14, y=v,
                               tip=f"{s['name']} · {case_title(n)} · {secs(v)} s")
                          for i, (n, v) in enumerate(zip(names, s["values"]))])
                for j, s in enumerate(series)]
        chart = xy_chart(dots, xlabel="CASE", ylabel="FULL-RISK SECONDS · MEDIAN · LOG", ylog=True,
                         xticks=[(i, case_title(n)) for i, n in enumerate(names)], right=24, xpad=0.5,
                         label="Full-risk benchmark")
    else:
        chart = bar_chart(names, series, ylabel="FULL-RISK SECONDS · MEDIAN", xlabel="CASE",
                          valfmt=lambda v: f"{secs(v)} s", label="Full-risk benchmark")
    rows, vrows, vkeys = [], [], []
    for n in names:
        for e in engines:
            r = cases[n].get(e)
            if not isinstance(r, dict):
                continue
            acc = r.get("accuracy") or {}
            sp = cases[n].get("speedup_vs_pde") if e == "pde" else cases[n].get("speedup_vs_mc") if e == "mc" else None
            cold = " / ".join(secs(x) for x in r.get("cold", []))
            worst = ratio(acc.get("worst"))
            if acc.get("meets") is False:
                worst = f'<span class="dn">{worst}</span>'
            rows.append([case_title(n), ENGINE_NAME[e], html.escape(config_label(e, r.get("config"))),
                         f'<span title="cold runs: {cold}">{secs(r.get("median_cold"))}</span>',
                         secs(_median(r.get("warm_moved"))), speedup(sp) if e != "quad" else DASH, worst,
                         str(acc.get("binding", DASH)).upper(), mb(r.get("peak_rss_after"))])
            per = {}
            for v in r.get("valuations", []):
                for k, s in (v or {}).items():
                    per.setdefault(k, []).append(s)
                    if k not in vkeys:
                        vkeys.append(k)
            vrows.append((case_title(n), ENGINE_NAME[e], {k: _median(s) for k, s in per.items()}))
    vtable = table(["case", "engine"] + [k.upper() for k in vkeys],
                   [[c, e] + [secs(d.get(k)) for k in vkeys] for c, e, d in vrows],
                   "Seconds per valuation, median over repeats (base solve + bumped solves of the full-risk run).")
    env = bench.get("environment", {})
    envs = " · ".join(f"{html.escape(str(k))} {html.escape(str(v))}" for k, v in env.items())
    return section(6, "BENCH", "Full-risk benchmark", note_for(narr, "BENCH") + figure(
        chart, "Median cold full-risk time per case",
        "Cold = fresh engine objects and empty caches (the only basis of a speed claim). Hollow = QUAD warm-moved "
        f"(s0 +0.5%, re-anchored grid).{' Log scale: the engines differ by more than 20×.' if log_scale else ''}")
        + table(["case", "engine", "timed config", "cold s", "warm s", "QUAD ×", "err/bud", "binding", "RSS MB"],
                rows,
                f"cold s = median cold full-risk seconds (hover for the individual runs); warm s = QUAD warm-moved "
                f"median; QUAD × = this engine's median cold / QUAD median cold; err/bud = worst |err|/budget at the "
                f"timed config, red if over budget; RSS = peak process RSS after the arm. Repeats: "
                f"{html.escape(str(bench.get('repeats', DASH)))}. {envs}")
        + '<h3 class="sub">Per-valuation timings</h3>' + vtable)


def section_dupire(narr, reference) -> str:
    gap = (reference or {}).get("dupire_gap") or []
    if not gap:
        return section(7, "DUPIRE", "Dupire repricing gap", pending("reference.json has no dupire_gap rows."))
    mats = sorted({g["maturity"] for g in gap})
    mons = sorted({g["moneyness"] for g in gap})
    by = {(g["maturity"], g["moneyness"]): g for g in gap}
    vols = [g["gap_vol_points"] * 100 for g in gap if _ok(g.get("gap_vol_points"))]
    lo, hi = min(vols + [0.0]), max(vols + [0.0])
    panels = []
    for t in mats:
        pts = [dict(x=m, y=by[(t, m)]["gap_vol_points"] * 100,
                    tip=f"T {t:.3f}y · K/S0 {m:g} · gap {signed_plain(by[(t, m)]['gap_vol_points'] * 100, 2)} vol pts")
               for m in mons if (t, m) in by and _ok(by[(t, m)].get("gap_vol_points"))]
        panels.append(f'<div><div class="ptitle">T = {t:.3f}Y · {t * 365:.0f}D</div>' + xy_chart(
            [dict(name="gap", colour="var(--s1)", line=True, pts=pts)], xlabel="MONEYNESS K/S0",
            ylabel="GAP, VOL PTS", ydomain=(lo, hi), width=300, height=220, right=20,
            xticks=[(m, f"{m:g}") for m in mons[::2]], label=f"Dupire gap T={t:.3f}") + "</div>")
    peak = max(vols) if vols else 1.0
    head = ["T (years)", "T (days)"] + [f"K/S0 {m:g}" for m in mons]
    rows = []
    for t in mats:
        cells = [fixed(t, 3), f"{t * 365:.0f}"]
        for m in mons:
            g = by.get((t, m))
            if not g or not _ok(g.get("gap_vol_points")):
                cells.append(DASH)
                continue
            v = g["gap_vol_points"] * 100
            share = max(0, min(100, round(abs(v) / peak * 60))) if peak else 0
            cells.append(f'<span class="heat" style="--h:{share}%" title="QUAD {sig(g["quad"], 5)} · BS '
                         f'{sig(g["black_scholes"], 5)} · iv {fixed(g["implied_vol"] * 100, 2)}%">'
                         f'{signed_plain(v, 2)}</span>')
        rows.append(cells)
    finds = [f for f in narr.get("findings") or [] if "dupire" in (str(f.get("title", "")) + str(f.get("body", ""))).lower()]
    callout = "".join(f'<p class="lead"><b>{prose(f.get("title"))}.</b> {prose(f.get("body"))}</p>' for f in finds)
    chart = (f'<figure><figcaption><b>QUAD European put on the Dupire surface minus BS(input iv)</b><span>In vol points '
             f'(price gap / BS vega), per maturity pillar; shared y-axis.</span></figcaption>'
             f'<div class="multiples m3">{"".join(panels)}</div></figure>')
    return section(7, "DUPIRE", "Dupire repricing gap", note_for(narr, "DUPIRE") + callout + chart + table(
        head, rows, "Signed gap in vol points (+ = the LV surface prices the put above BS at the input iv). Shading "
                    "scales with |gap|; hover a cell for QUAD / BS prices and the input iv. Reported, not gated."))


def section_findings(narr) -> str:
    items = narr.get("findings") or []
    if not items:
        return section(8, "FINDINGS", "Findings", pending("narrative.json has no findings yet."))
    lis = "".join(f'<li><b>{prose(f.get("title"))}.</b> {prose(f.get("body"))}</li>' for f in items)
    return section(8, "FINDINGS", "Findings", f'<ul class="points">{lis}</ul>')


def section_limitations(narr) -> str:
    items = narr.get("limitations") or []
    if not items:
        return section(9, "LIMITS", "Limitations", pending("narrative.json has no limitations yet."))
    lis = "".join(f"<li>{prose(x)}</li>" for x in items)
    return section(9, "LIMITS", "Limitations", f'<ul class="points dim">{lis}</ul>')


def section_reproduce(narr, reference, tuning, bakeoff, bench) -> str:
    cmds = narr.get("reproduce") or []
    body = ('<ol class="cmds">' + "".join(f"<li><code>{html.escape(c)}</code></li>" for c in cmds) + "</ol>"
            if cmds else pending("narrative.json has no reproduce commands yet."))
    rows = []
    for fname, data in (("reference.json", reference), ("bakeoff.json", bakeoff), ("tuning.json", tuning),
                        ("benchmark.json", bench)):
        if data is None:
            rows.append([fname, '<span class="dim">absent</span>', DASH])
            continue
        src = data.get("sources") or {}
        flat = {f"{k}/{kk}": vv for k, v in src.items() if isinstance(v, dict) for kk, vv in v.items()} or src
        env = data.get("environment") or {}
        rows.append([fname, html.escape(" · ".join(f"{k} {v}" for k, v in env.items()) or DASH),
                     f'<span class="wrap">{html.escape(" ".join(f"{k}:{v}" for k, v in flat.items()) or DASH)}</span>'])
    return section(10, "REPRO", "Reproduce", body + '<h3 class="sub">Provenance</h3>' + table(
        ["file", "environment", "source hashes"], rows, "Hashes as recorded by each run; a mismatch against the "
        "current sources means that file is stale.", text=True))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

NAV = [(1, "VERDICT"), (2, "METHOD"), (3, "REFERENCE"), (4, "BAKE-OFF"), (5, "TUNING"), (6, "BENCHMARK"),
       (7, "DUPIRE"), (8, "FINDINGS"), (9, "LIMITS"), (10, "REPRODUCE")]


def chrome(reference, as_of: str = "") -> str:
    """Top bar; the date is the narrative's ``as_of`` so identical inputs render identical bytes."""
    mkt = (reference or {}).get("market", {})
    nav = "".join(f'<a href="#s{n}"><b>{n})</b> {t}</a>' for n, t in NAV)
    s0 = f"S0 {fixed(mkt['spot'], 1)}" if _ok(mkt.get("spot")) else "LOCAL VOL"
    return f"""
<div class="top">
<header class="fbar">
  <span class="brand">QUANTARK</span><span class="cmd">LVQ &lt;GO&gt;</span>
  <span class="fld">STUDY <b>localvol_transition_quad</b></span><span class="fld">ENGINES <b>QUAD · PDE · MC</b></span>
  <span class="fld">MKT <b>DUPIRE LV · {s0}</b></span><span class="fld date">{html.escape(as_of)}</span>
  <button class="theme" id="theme" type="button" aria-label="Switch colour theme">◐ <b>LIGHT</b></button>
</header>
<nav class="menu" id="menu">{nav}</nav>
</div>
"""


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
a.lnk{color:var(--amber);border-bottom:1px solid var(--amber-dim)}

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
.hero{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));column-gap:24px;padding:72px 0 40px;
  border-bottom:1px solid var(--line)}
.kicker{grid-column:1/4;font:11px/1.4 var(--mono);letter-spacing:.14em;color:var(--amber);padding-top:12px}
.hero h1{grid-column:4/13;margin:0;font:700 clamp(32px,4.8vw,60px)/1.04 var(--sans);letter-spacing:-.025em}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border-top:1px solid var(--line);
  border-bottom:1px solid var(--line);margin-top:8px}
.kpi{padding:22px 20px 22px 0;border-right:1px solid var(--line);min-width:0}
.kpi+.kpi{padding-left:20px}.kpi:last-child{border-right:0}
.kv{font:600 clamp(22px,3vw,36px)/1.05 var(--mono);letter-spacing:-.02em;color:var(--ink);overflow-wrap:anywhere}
.kv small{font-size:12px;margin-left:6px;color:var(--ink-3);letter-spacing:.08em}
.kl{margin-top:10px;font:11px/1.4 var(--mono);letter-spacing:.06em;color:var(--ink-2);text-transform:uppercase}

section{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));column-gap:24px;padding:56px 0;
  border-bottom:1px solid var(--line)}
.rail{grid-column:1/4}
.numeral{font:700 88px/0.85 var(--sans);letter-spacing:-.05em;color:var(--amber)}
.code{margin-top:18px;font:11px/1 var(--mono);letter-spacing:.16em;color:var(--ink-3)}
.rail h2{margin:8px 0 0;font:700 22px/1.2 var(--sans);letter-spacing:-.01em}
.body{grid-column:4/13;min-width:0}
.lead{margin:0 0 28px;font-size:17px;max-width:68ch}
.body p{max-width:72ch}
h3.sub{margin:36px 0 8px;font:11px/1.4 var(--mono);letter-spacing:.16em;color:var(--ink-3);text-transform:uppercase}
h3.sub.first{margin-top:0}
h3.sub .flags{margin-left:14px;letter-spacing:.08em}

dl.spec{display:grid;grid-template-columns:160px minmax(0,1fr);margin:24px 0;border-top:1px solid var(--line)}
dl.spec dt,dl.spec dd{margin:0;padding:10px 0;border-bottom:1px solid var(--hair)}
dl.spec dd{overflow-wrap:anywhere;min-width:0}
dl.spec dt{font:11px/1.6 var(--mono);letter-spacing:.1em;color:var(--amber);text-transform:uppercase}

.scroll{overflow-x:auto;margin:8px 0 24px}
table{border-collapse:collapse;width:100%;font:12.5px/1.3 var(--mono);font-variant-numeric:tabular-nums}
th{padding:8px 12px 8px 0;text-align:right;font-weight:500;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--amber-dim);border-bottom:1px solid var(--amber-dim);white-space:nowrap}
td{padding:6px 12px 6px 0;text-align:right;border-bottom:1px solid var(--hair);white-space:nowrap}
th:first-child,td:first-child,table.text th,table.text td{text-align:left}
tbody tr:hover td{background:var(--row)}
td code{color:var(--ink)}
table.text td{white-space:normal}
td .wrap{white-space:normal;overflow-wrap:anywhere;color:var(--ink-2)}
.up{color:var(--up)}.dn{color:var(--dn)}.flat{color:var(--ink-2)}.dim{color:var(--ink-3)}
b.win{color:var(--amber);font-weight:600}
.note{margin:-16px 0 24px;font:11px/1.5 var(--mono);color:var(--ink-3);max-width:none}
.note.chosen{margin:0 0 4px;color:var(--ink-2)}.note.chosen b{color:var(--amber);font-weight:500}
.heat{display:inline-block;min-width:54px;padding:2px 4px;
  background:color-mix(in srgb,var(--amber) var(--h),transparent)}
.pending{margin:0 0 24px;padding:12px 0 12px 14px;border-left:2px solid var(--amber);font:12.5px/1.5 var(--mono);
  color:var(--ink-2)}
.pending b{color:var(--amber);font-weight:600;letter-spacing:.12em;margin-right:8px}

figure{margin:0 0 12px;border-top:1px solid var(--line);padding-top:12px}
figcaption{display:flex;flex-wrap:wrap;gap:4px 16px;align-items:baseline;margin-bottom:10px}
figcaption b{font:600 13px/1.3 var(--sans)}
figcaption span{font:11px/1.3 var(--mono);color:var(--ink-3)}
.legend{display:flex;flex-wrap:wrap;gap:6px 20px;font:11px/1 var(--mono);color:var(--ink-2);margin:2px 0 6px}
.legend i{display:inline-block;width:14px;height:3px;margin-right:7px;vertical-align:middle}
svg.chart{width:100%;height:auto;display:block;overflow:visible}
.chart .grid{stroke:var(--hair);stroke-width:1}
.chart .axis{stroke:var(--line);stroke-width:1}
.chart .ref{stroke:var(--ink-3);stroke-width:1;stroke-dasharray:4 3}
.chart .reflab{font:9.5px var(--mono);letter-spacing:.08em;fill:var(--ink-3)}
.chart .tick{font:10.5px var(--mono);fill:var(--ink-3)}
.chart .axlab{font:9.5px var(--mono);letter-spacing:.1em;fill:var(--ink-3)}
.chart .ln{fill:none;stroke-width:2;stroke-linejoin:round}
.chart .mk{stroke:var(--bg);stroke-width:2;pointer-events:none}
.chart .hiring{fill:none;stroke:var(--amber);stroke-width:2;pointer-events:none}
.chart .hit{fill:transparent;cursor:crosshair}
.chart .bar{cursor:crosshair}.chart .bar:hover{stroke:var(--ink);stroke-width:1.5}
.tip{position:fixed;z-index:10;pointer-events:none;padding:6px 9px;background:var(--panel);border:1px solid var(--amber);
  color:var(--ink);font:11.5px/1.3 var(--mono);white-space:nowrap;display:none}
.chart .dlab{font:11px var(--mono);fill:var(--ink-2)}
.chart .tot{font:10.5px var(--mono);fill:var(--ink)}
.multiples{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px}
.multiples.m3{grid-template-columns:repeat(3,minmax(0,1fr))}
.ptitle{font:11px/1 var(--mono);letter-spacing:.12em;color:var(--amber);margin-bottom:6px}

ul.points{padding:0;list-style:none;margin:0;max-width:72ch}
ul.points li{padding:10px 0 10px 18px;border-top:1px solid var(--hair);position:relative}
ul.points li::before{content:"";position:absolute;left:0;top:19px;width:8px;height:2px;background:var(--amber)}
ul.points li b{color:var(--ink);font-weight:600}
ul.points.dim li{color:var(--ink-2)}
ol.cmds{margin:0;padding:0 0 0 22px;font:12.5px/1.5 var(--mono);color:var(--ink-3)}
ol.cmds li{padding:6px 0;border-top:1px solid var(--hair);overflow-wrap:anywhere}
footer{padding:24px 0;font:11px/1.5 var(--mono);color:var(--ink-3);letter-spacing:.04em}

@media (max-width:820px){
  main{padding:0 16px 64px}
  .hero,section{grid-template-columns:minmax(0,1fr)}
  .kicker,.hero h1,.rail,.body{grid-column:1/-1}
  .rail{margin-bottom:20px}.numeral{font-size:56px}
  .kpis{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.kpi:nth-child(2){border-right:0}
  .kpi:nth-child(3){padding-left:0}.kpi:nth-child(n+3){border-top:1px solid var(--line)}
  .multiples,.multiples.m3{grid-template-columns:minmax(0,1fr)}
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


def render(folder: Path) -> str:
    """Build the report HTML from the JSONs in ``folder``."""
    folder = Path(folder)
    reference = _load(folder, "reference.json")
    bakeoff = _load(folder, "bakeoff.json")
    tuning = _load(folder, "tuning.json")
    bench = _load(folder, "benchmark.json")
    narr = _load(folder, "narrative.json") or {}
    title = str(narr.get("title") or "Local-vol transition QUAD")
    hero = (f'<div class="hero"><div class="kicker">LOCAL VOL · QUAD · FULL RISK</div>'
            f"<h1>{prose(title)}</h1></div>")
    content = (hero + section_verdict(narr, bench) + section_method(narr, reference)
               + section_reference(narr, reference) + section_bakeoff(narr, bakeoff, reference)
               + section_tuning(narr, tuning, reference) + section_benchmark(narr, bench)
               + section_dupire(narr, reference) + section_findings(narr) + section_limitations(narr)
               + section_reproduce(narr, reference, tuning, bakeoff, bench)
               + "<footer>QUANTARK · example/case_studies/localvol_transition_quad → docs/research/"
                 "localvol_transition_quad/*.json → render_report.py · fixture market, no desk data</footer>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Local-Vol Transition QUAD</title>{THEME_INIT_JS}<style>{CSS}</style></head>"
            f"<body>{chrome(reference, str(narr.get('as_of', '')))}<main>{content}</main>{PAGE_JS}</body></html>")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=str(HERE), help="folder holding the result JSONs (default: this folder)")
    parser.add_argument("--out", default="report.html", help="output file; relative paths resolve against --dir")
    args = parser.parse_args(argv)
    folder = Path(args.dir)
    out = Path(args.out)
    out = out if out.is_absolute() else folder / out
    out.write_text(render(folder), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
