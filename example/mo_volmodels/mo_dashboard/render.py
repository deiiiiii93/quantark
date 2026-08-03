"""Payload -> one self-contained HTML document.

A file:// page cannot fetch() a sibling JSON, so the payload is inlined.
"""
from __future__ import annotations

import html as _html
import json
from typing import Any, Dict, Sequence

PANEL_IDS = ("panel-status", "panel-results", "panel-fleet")

STATE_GLYPH = {
    "fresh": "██",
    "stale": "▒▒",
    "void": "░░",
    "failed": "▓▓",
    "running": "▶▶",
    "unreadable": "!!",
    "missing": "··",
}
STATE_COLOR = {
    "fresh": "var(--pos)",
    "stale": "var(--warn)",
    "void": "var(--neg)",
    "failed": "var(--neg)",
    "running": "var(--info)",
    "unreadable": "var(--neg)",
    "missing": "var(--hairline-2)",
}

_CSS = """
:root {
  --paper:#111110; --paper-2:#1c1b1a; --paper-3:#292725;
  --ink:#f5f2e8; --ink-2:#9b978d;
  --hairline:#3d3830; --hairline-2:#5c554a;
  --font-ui:'Inter Tight',-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  --font-num:'Berkeley Mono','JetBrains Mono','SF Mono',monospace;
  --pos:#3cb371; --neg:#e45756; --warn:#f0ad4e; --info:#4c72b0;
}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:var(--font-ui);background:var(--paper);color:var(--ink);
     line-height:1.5;padding:24px}
.container{max-width:1320px;margin:0 auto}
header{padding:8px 0 24px;border-bottom:1px solid var(--hairline);margin-bottom:24px}
header h1{font-size:1.5rem;font-weight:600;letter-spacing:-0.01em}
header .meta{font-family:var(--font-num);font-size:0.78rem;color:var(--ink-2);margin-top:6px}
.caveat{border-left:2px solid var(--warn);padding:8px 12px;margin:12px 0;
        font-size:0.82rem;color:var(--ink-2);background:var(--paper-3)}
.caveat b{color:var(--ink)}
section{border:1px solid var(--hairline);background:var(--paper-2);
        padding:20px 24px;margin-bottom:24px}
section h2{font-size:1rem;font-weight:600;margin-bottom:14px}
section h3{font-size:0.85rem;font-weight:600;margin:18px 0 8px;color:var(--ink-2);
           text-transform:uppercase;letter-spacing:0.06em}
table{width:100%;border-collapse:collapse;font-family:var(--font-num);font-size:0.8rem}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--hairline)}
th{color:var(--ink-2);font-weight:500}
.badge{font-family:var(--font-num);font-size:0.72rem;padding:1px 6px;
       border:1px solid var(--hairline-2)}
.grid-wrap{overflow-x:auto}
.grid{font-family:var(--font-num);font-size:0.7rem;white-space:pre;line-height:1.35}
.err{color:var(--neg);font-family:var(--font-num);font-size:0.76rem;
     padding:3px 0;border-bottom:1px solid var(--hairline)}
"""


def esc(value: Any) -> str:
    return _html.escape(str(value if value is not None else ""), quote=True)


def verdict_label(passed: bool, confidence: str) -> str:
    """Never a bare PASS -- inferred freshness is not proof (spec 6.3)."""
    return f"{'PASS' if passed else 'FAIL'} ({confidence})"


def _badge(freshness: str, mode: str = "inferred") -> str:
    color = STATE_COLOR.get(freshness, "var(--ink-2)")
    return (
        f'<span class="badge" style="color:{color};border-color:{color}">'
        f"{esc(freshness)} · {esc(mode)}</span>"
    )


def _panel_status(doc: Dict[str, Any]) -> str:
    rows = []
    for row in doc.get("gates", []):
        facets = row.get("facets") or {}
        cells = " ".join(
            f"{esc(name)}: {_badge(f.get('freshness', '?'), f.get('mode', '?'))}"
            for name, f in facets.items()
        ) or esc(row.get("status"))
        satisfied = (row.get("headline") or {}).get("satisfied")
        conf = (
            "exact"
            if facets and all(f.get("mode") == "exact" for f in facets.values())
            else "inferred"
        )
        rows.append(
            f"<tr><td>{esc(row.get('id'))}</td><td>{esc(row.get('title'))}</td>"
            f"<td>{esc(verdict_label(bool(satisfied), conf))}</td>"
            f"<td>{cells}</td>"
            f"<td>{esc(row.get('artifact_mtime') or '—')}</td></tr>"
        )
    action = (doc.get("chain") or {}).get("next_action") or {}
    git = doc.get("git") or {}
    dirty = git.get("dirty_paths") or []
    dirty_html = (
        "<div class='caveat'><b>Dirty working tree</b> — "
        + ", ".join(esc(p) for p in dirty)
        + "</div>"
    ) if dirty else ""
    cohort = doc.get("cohort") or {}
    return f"""
<section id="panel-status">
  <h2>Program status</h2>
  <table><tr><th>gate</th><th>title</th><th>verdict</th><th>facets</th><th>artifact</th></tr>
  {''.join(rows)}</table>
  <div class="caveat"><b>Next action:</b> {esc(action.get('node') or '—')} —
    {esc(action.get('why'))} <i>({esc(action.get('confidence'))})</i></div>
  {dirty_html}
  <div class="caveat">Inferred freshness is evidence <i>against</i> invalidity,
    not evidence <i>of</i> validity: a copied, restored or touched artifact
    reads fresh. Cohort pinned at {esc(cohort.get('asof'))} —
    {esc(cohort.get('n_admitted'))} admitted, {esc(cohort.get('n_excluded'))} excluded.</div>
</section>"""


def _panel_results(doc: Dict[str, Any]) -> str:
    res = doc.get("results") or {}
    evidence = res.get("gate_evidence") or {}
    rows = []
    for name, block in (evidence.get("variants") or {}).items():
        pv, delta = block.get("pv") or {}, block.get("delta") or {}
        rows.append(
            f"<tr><td>{esc(name)}</td><td>{esc(block.get('route'))}</td>"
            f"<td>{esc(pv.get('pass'))}</td>"
            f"<td>{esc(delta.get('pass'))}</td>"
            f"<td>{esc(delta.get('max_abs_contracts'))}</td>"
            f"<td>{esc(delta.get('bound_contracts'))}</td></tr>"
        )
    delta_facet = evidence.get("delta") or {}
    delta_note = ""
    if delta_facet.get("freshness") == "void":
        delta_note = (
            f"<div class='caveat'><b>Delta column VOID</b> by "
            f"{esc(delta_facet.get('invalidated_by'))} — "
            f"{esc(delta_facet.get('invalidation_reason'))}</div>"
        )
    backtest = res.get("backtest") or {}
    rec = backtest.get("reconciliation") or {}
    rec_note = "" if rec.get("agrees", True) else (
        f"<div class='caveat'><b>Denominator reconciliation:</b> "
        f"{esc(rec.get('manifest_runs'))} runs in the manifest, "
        f"{esc(rec.get('tree_total'))} cells on disk, "
        f"{esc(rec.get('unaccounted'))} unaccounted. {esc(rec.get('note'))}</div>"
    )
    calib = res.get("calibration") or {}
    feller = calib.get("feller") or {}
    band_rows = "".join(
        f"<tr><td>{esc(key)}</td><td>{esc(band.get('n'))}</td>"
        f"<td>{esc(round(band['pct'], 1) if band.get('pct') is not None else '—')}</td>"
        f"<td>{esc(band.get('label'))}</td><td>{esc(band.get('citation'))}</td></tr>"
        for key, band in feller.items()
        if isinstance(band, dict)
    )
    cost = calib.get("cost") or {}
    return f"""
<section id="panel-results">
  <h2>Results</h2>
  <h3>Gate evidence</h3>
  <table><tr><th>variant</th><th>route</th><th>PV</th><th>delta</th>
    <th>max |Δ| ct</th><th>bound ct</th></tr>{''.join(rows)}</table>
  {delta_note}
  <h3>Backtest outcomes</h3>
  <div class="caveat">{esc(backtest.get('caveat'))}</div>
  {rec_note}
  <h3>Calibration health — {esc(calib.get('n_records'))} records,
    as of {esc(calib.get('as_of_date') or '—')}</h3>
  <table><tr><th>band</th><th>n</th><th>%</th><th>label</th><th>citation</th></tr>
  {band_rows}</table>
  <div class="caveat">calibration objective (record <code>cost</code>, not bp of IV):
    median {esc(cost.get('median'))} · p90 {esc(cost.get('p90'))} ·
    max {esc(cost.get('max'))}</div>
</section>"""


def _panel_fleet(doc: Dict[str, Any]) -> str:
    fleet = doc.get("fleet") or {}
    variants: Sequence[str] = fleet.get("variants") or []
    inceptions: Sequence[str] = fleet.get("inceptions") or []
    grid = fleet.get("grid") or {}
    counts = fleet.get("counts") or {}

    def state_of(variant: str, tag: str) -> str:
        return ((grid.get(variant) or {}).get(tag) or {}).get("state", "missing")

    lines = []
    for variant in variants:
        pieces = []
        n_fresh = 0
        for tag in inceptions:
            state = state_of(variant, tag)
            n_fresh += state in ("fresh", "stale")
            color = STATE_COLOR.get(state, "var(--ink-2)")
            glyph = STATE_GLYPH.get(state, "··")
            pieces.append(
                f'<span style="color:{color}" title="{esc(tag)} {esc(state)}">'
                f"{glyph}</span>"
            )
        label = esc(variant).ljust(15)
        lines.append(f"{label}{''.join(pieces)}  {n_fresh}/{len(inceptions)}")

    dir_rows = "".join(
        f"<tr><td>{esc(d.get('dir'))}</td><td>{esc(d.get('role'))}</td>"
        f"<td>{esc(d.get('n_cells'))}</td></tr>"
        for d in fleet.get("run_dirs", [])
    )
    legend = " ".join(f"{glyph} {name}" for name, glyph in STATE_GLYPH.items())
    return f"""
<section id="panel-fleet">
  <h2>Fleet coverage — {esc(fleet.get('admitted'))}/{esc(fleet.get('expected_cells'))} admitted
    ({esc(counts.get('fresh', 0))} fresh · {esc(counts.get('stale', 0))} stale)</h2>
  <div class="grid-wrap"><div class="grid">{'<br>'.join(lines)}</div></div>
  <div class="caveat">{esc(legend)}<br>counts: {esc(json.dumps(counts))}</div>
  <table><tr><th>run dir</th><th>role</th><th>cells</th></tr>{dir_rows}</table>
</section>"""


def render(doc: Dict[str, Any]) -> str:
    errors = doc.get("errors") or []
    err_html = "".join(
        f"<div class='err'>{esc(e.get('source'))}: {esc(e.get('path'))} — "
        f"{esc(e.get('message'))}</div>"
        for e in errors
    )
    # json.dumps passes "</script>" through untouched (verified), and the
    # payload carries log tails, exception text and git subjects -- arbitrary
    # text from disk.  An unescaped closer terminates the application/json
    # element and everything after it becomes markup.
    payload_json = json.dumps(doc, default=str).replace("<", "\\u003c")
    git = doc.get("git") or {}
    return f"""<!doctype html>
<html lang="en" data-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Snowball study — progress</title><style>{_CSS}</style></head><body>
<div class="container">
<header><h1>Snowball vol-model study — progress</h1>
<div class="meta">generated {esc(doc.get('generated_at'))} · mode {esc(doc.get('mode'))} ·
 {esc(git.get('branch'))} @ {esc(git.get('head'))} — {esc(git.get('head_subject'))}</div></header>
{_panel_status(doc)}
{_panel_results(doc)}
{_panel_fleet(doc)}
{f'<section><h2>Errors</h2>{err_html}</section>' if err_html else ''}
</div>
<script id="__DASHBOARD_PAYLOAD__" type="application/json">{payload_json}</script>
</body></html>"""
