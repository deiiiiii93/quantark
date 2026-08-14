"""Self-contained HTML certification report.

The markdown report is for reading in a terminal or a diff; this one is for
reviewing and circulating. It shows what a table of numbers cannot: how much of
each bound a cell actually consumed. A cell at 4% of its budget and a cell at
96% both read as "PASS" in markdown, and they are not remotely the same
evidence.

The page is a single file with no external requests -- inline CSS, no scripts,
no fonts, no images -- so it survives being emailed, archived beside the
certificate, or opened years later from a directory with no network.
"""

from __future__ import annotations

import html
from typing import Any, Mapping, Optional, Sequence

_NA = "&mdash;"

#: Verdict and decision colours are semantic, not decorative: they encode state
#: so a reviewer can scan for trouble without reading every number.
_VERDICT_CLASS = {
    "PASS": "pass",
    "FAIL": "fail",
    "UNRESOLVED": "warn",
    "ERROR": "err",
}
_DECISION_CLASS = {
    "ADMITTED": "pass",
    "REJECTED": "fail",
    "INCONCLUSIVE": "warn",
}

_STYLE = """
:root {
  --ground: #eceff1;
  --paper: #ffffff;
  --ink: #16212b;
  --muted: #5d6d79;
  --rule: #ccd6dc;
  --rail: #dfe6ea;
  --accent: #0f7a56;
  --pass: #0f7a56;
  --fail: #b3261e;
  --warn: #a8690f;
  --err: #6b5b95;
  --pass-bg: #e4f1eb;
  --fail-bg: #fae9e7;
  --warn-bg: #f8eeda;
  --err-bg: #ece9f3;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #10171d;
    --paper: #18212a;
    --ink: #e4ecf1;
    --muted: #93a4b0;
    --rule: #31404c;
    --rail: #26323c;
    --accent: #46c391;
    --pass: #46c391;
    --fail: #f08b80;
    --warn: #dfa445;
    --err: #a595c9;
    --pass-bg: #17332a;
    --fail-bg: #3a211f;
    --warn-bg: #35291480;
    --err-bg: #262137;
  }
}
:root[data-theme="dark"] {
  --ground: #10171d;
  --paper: #18212a;
  --ink: #e4ecf1;
  --muted: #93a4b0;
  --rule: #31404c;
  --rail: #26323c;
  --accent: #46c391;
  --pass: #46c391;
  --fail: #f08b80;
  --warn: #dfa445;
  --err: #a595c9;
  --pass-bg: #17332a;
  --fail-bg: #3a211f;
  --warn-bg: #35291480;
  --err-bg: #262137;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: Charter, "Iowan Old Style", "Palatino Linotype", Georgia, serif;
  font-size: 16px;
  line-height: 1.55;
}
.sheet {
  max-width: 1180px;
  margin: 0 auto;
  padding: 40px 24px 80px;
}

.masthead { border-bottom: 3px solid var(--ink); padding-bottom: 18px; }
.eyebrow {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--muted); margin: 0 0 10px;
}
h1 {
  font-size: clamp(26px, 4vw, 36px); line-height: 1.12; margin: 0 0 6px;
  letter-spacing: -0.01em; text-wrap: balance;
}
.subtitle { margin: 0; color: var(--muted); font-size: 16px; }

.banner {
  display: flex; flex-wrap: wrap; gap: 10px 28px; align-items: baseline;
  margin: 22px 0 0; padding: 16px 20px;
  background: var(--paper); border: 1px solid var(--rule); border-radius: 5px;
  border-left: 5px solid var(--rule);
}
.banner.pass { border-left-color: var(--pass); }
.banner.fail { border-left-color: var(--fail); }
.banner.warn { border-left-color: var(--warn); }
.banner .headline { font-size: 19px; font-weight: 600; }
.banner .note { color: var(--muted); font-size: 14.5px; }

.quick {
  margin: 18px 0 0; padding: 12px 18px; border-radius: 5px;
  background: var(--warn-bg); border-left: 4px solid var(--warn);
  font-size: 15px;
}

.stats { display: flex; flex-wrap: wrap; gap: 12px; margin: 22px 0 0; }
.stat {
  flex: 1 1 150px; background: var(--paper); border: 1px solid var(--rule);
  border-radius: 5px; padding: 12px 16px;
}
.stat .k {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 10.5px; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--muted); display: block; margin-bottom: 4px;
}
.stat .v {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 21px; font-variant-numeric: tabular-nums; font-weight: 600;
}
.stat .v.small { font-size: 15px; font-weight: 500; }

section { margin: 40px 0 0; }
h2 {
  font-size: 19px; margin: 0 0 4px; letter-spacing: -0.005em;
  border-bottom: 1px solid var(--rule); padding-bottom: 7px;
}
.lede { color: var(--muted); font-size: 14.5px; margin: 8px 0 14px; max-width: 74ch; }

.scroll { overflow-x: auto; }
table {
  width: 100%; border-collapse: collapse; background: var(--paper);
  border: 1px solid var(--rule); font-size: 13.5px;
}
th, td {
  text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--rule);
  white-space: nowrap;
}
th {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 10.5px; letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--muted); font-weight: 600; background: var(--rail);
}
tbody tr:last-child td { border-bottom: none; }
td.num {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums; text-align: right;
}
td.name { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12.5px; }
td.wrap { white-space: normal; }

.pill {
  display: inline-block; padding: 1px 9px; border-radius: 999px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 10.5px; font-weight: 700; letter-spacing: 0.07em;
}
.pill.pass { background: var(--pass-bg); color: var(--pass); }
.pill.fail { background: var(--fail-bg); color: var(--fail); }
.pill.warn { background: var(--warn-bg); color: var(--warn); }
.pill.err  { background: var(--err-bg);  color: var(--err); }

.gauge { display: flex; align-items: center; gap: 8px; min-width: 132px; }
.track {
  position: relative; flex: 1; height: 7px; border-radius: 4px;
  background: var(--rail); overflow: hidden; min-width: 74px;
}
.fill { position: absolute; inset: 0 auto 0 0; border-radius: 4px; }
.fill.pass { background: var(--pass); }
.fill.fail { background: var(--fail); }
.fill.warn { background: var(--warn); }
.fill.err  { background: var(--err); }
.gauge .pct {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums;
  min-width: 38px; text-align: right;
}

.meta {
  margin: 44px 0 0; padding-top: 14px; border-top: 1px solid var(--rule);
  font-size: 13px; color: var(--muted);
}
.meta dl { display: grid; grid-template-columns: max-content 1fr; gap: 4px 18px; margin: 0; }
.meta dt {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 10.5px; letter-spacing: 0.09em; text-transform: uppercase;
}
.meta dd {
  margin: 0; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 12px; word-break: break-all; color: var(--ink);
}
code {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.92em;
}
.empty { color: var(--muted); font-style: italic; font-size: 14px; }

@media print {
  body { background: #fff; }
  .sheet { max-width: none; padding: 0; }
  section { break-inside: avoid; }
}
"""


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: Any, digits: int = 6) -> str:
    """Format a number for a data cell, keeping non-finite values honest."""
    if value is None:
        return _NA
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value in (float("inf"), float("-inf")):
            return "&infin;" if value > 0 else "-&infin;"
        return f"{value:.{digits}g}"
    return _esc(value)


def _pill(text: str, kind: str) -> str:
    return f'<span class="pill {kind}">{_esc(text)}</span>'


def _gauge(consumed: Optional[float], bound: float, kind: str) -> str:
    """A bar showing how much of a bound this measurement consumed.

    This is the whole reason the HTML report exists: PASS at 4% of budget and
    PASS at 96% of budget are different evidence, and only one of them is
    comfortable.
    """
    if consumed is None or bound <= 0:
        return f'<span class="pct">{_NA}</span>'
    ratio = consumed / bound
    width = min(100.0, max(1.5, ratio * 100.0))
    return (
        f'<div class="gauge"><div class="track">'
        f'<div class="fill {kind}" style="width:{width:.1f}%"></div></div>'
        f'<span class="pct">{ratio * 100:.0f}%</span></div>'
    )


def _first_line(text: str) -> str:
    lines = [line.strip() for line in (text or "").strip().splitlines() if line.strip()]
    return lines[-1] if lines else "&mdash;"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], empty: str) -> str:
    if not rows:
        return f'<p class="empty">{_esc(empty)}</p>'
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(row) + "</tr>" for row in rows)
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _banner(payload: Mapping[str, Any]) -> str:
    decisions = payload["decisions"]
    values = list(decisions.values())
    if values and all(v == "ADMITTED" for v in values):
        kind, headline = "pass", "All candidates admitted"
    elif any(v == "REJECTED" for v in values):
        kind, headline = "fail", "At least one candidate rejected"
    else:
        kind, headline = "warn", "Inconclusive"

    cells = payload["cells"]
    unresolved = sum(1 for c in cells if c["verdict"] == "UNRESOLVED")
    errored = sum(1 for c in cells if c["verdict"] == "ERROR")
    notes = []
    if unresolved:
        notes.append(f"{unresolved} cell(s) UNRESOLVED &mdash; the benchmark never met its budget")
    if errored:
        notes.append(f"{errored} cell(s) ERROR")
    note = "; ".join(notes) if notes else "Every cell was measured against a benchmark that met its budget."

    return (
        f'<div class="banner {kind}"><span class="headline">{_esc(headline)}</span>'
        f'<span class="note">{note}</span></div>'
    )


def _stats(payload: Mapping[str, Any]) -> str:
    cells = payload["cells"]
    bounds = payload["study"]["bounds"]
    gated = [c for c in cells if c["gate"] is not None]
    worst = max((c["gate"]["interval_c"] for c in gated), default=None)
    batches = [
        block["batches"]
        for block in payload["references"].values()
        if "batches" in block
    ]

    tiles = [
        ("cells", str(len(cells)), False),
        ("candidates", str(len(payload["study"]["candidates"])), False),
        (
            "worst cell margin",
            f"{worst / bounds['cell'] * 100:.0f}%" if worst is not None else _NA,
            False,
        ),
        ("benchmark batches", str(sum(batches)) if batches else _NA, False),
        ("cell bound", f"{bounds['cell']:g} c", True),
        ("bias bound", f"{bounds['mean_signed_bias']:g} c", True),
    ]
    return '<div class="stats">' + "".join(
        f'<div class="stat"><span class="k">{_esc(k)}</span>'
        f'<span class="v{" small" if small else ""}">{v}</span></div>'
        for k, v, small in tiles
    ) + "</div>"


def _decisions_section(payload: Mapping[str, Any]) -> str:
    rows = []
    for name, decision in sorted(payload["decisions"].items()):
        own = [c for c in payload["cells"] if c["candidate"] == name]
        counts = {
            verdict: sum(1 for c in own if c["verdict"] == verdict)
            for verdict in ("PASS", "FAIL", "UNRESOLVED", "ERROR")
        }
        tally = " ".join(
            _pill(f"{n} {verdict}", _VERDICT_CLASS[verdict])
            for verdict, n in counts.items()
            if n
        )
        rows.append(
            [
                f'<td class="name">{_esc(name)}</td>',
                f"<td>{_pill(decision, _DECISION_CLASS.get(decision, 'warn'))}</td>",
                f"<td>{tally}</td>",
            ]
        )
    return (
        "<section><h2>Decisions</h2>"
        '<p class="lede">One decision per candidate engine. ADMITTED requires every cell '
        "to pass against a benchmark sharp enough to discriminate; REJECTED requires "
        "confident evidence of disagreement; INCONCLUSIVE means the evidence cannot say.</p>"
        + _table(["candidate", "decision", "cells"], rows, "No candidates.")
        + "</section>"
    )


def _cells_section(payload: Mapping[str, Any]) -> str:
    bound = payload["study"]["bounds"]["cell"]
    rows = []
    for cell in payload["cells"]:
        gate = cell["gate"]
        reference = cell["reference"]
        kind = _VERDICT_CLASS.get(cell["verdict"], "warn")
        rows.append(
            [
                f'<td class="name">{_esc(cell["candidate"])}</td>',
                f'<td>{_esc(cell["case"])}</td>',
                f'<td>{_esc(cell["quantity"])}</td>',
                f'<td class="num">{_fmt(reference["value"]) if reference else _NA}</td>',
                f'<td class="num">{_fmt(reference["se"], 3) if reference else _NA}</td>',
                f'<td class="num">{_fmt(cell["candidate_value"])}</td>',
                f'<td class="num">{_fmt(gate["signed_err_c"], 4) if gate else _NA}</td>',
                f'<td class="num">{_fmt(gate["envelope_c"], 3) if gate else _NA}</td>',
                f'<td>{_gauge(gate["interval_c"] if gate else None, bound, kind)}</td>',
                f'<td>{_pill(cell["verdict"], kind)}</td>',
            ]
        )
    return (
        "<section><h2>Cells</h2>"
        f'<p class="lede">One row per candidate &times; case &times; quantity. The gauge shows '
        f"how much of the {bound:g} c cell bound the disagreement interval "
        f"(|error| + k&middot;SE) consumed &mdash; a pass near 100% is a pass with no room "
        "left. Envelope is the candidate's own discretization error from its refinement "
        "ladder.</p>"
        + _table(
            [
                "candidate",
                "case",
                "quantity",
                "reference",
                "SE",
                "candidate",
                "err (c)",
                "envelope (c)",
                "interval vs bound",
                "verdict",
            ],
            rows,
            "No cells.",
        )
        + "</section>"
    )


def _aggregates_section(payload: Mapping[str, Any]) -> str:
    bound = payload["study"]["bounds"]["mean_signed_bias"]
    interval_k = payload["study"]["bounds"]["interval_k"]
    rows = []
    for aggregate in payload["aggregates"]:
        kind = "pass" if aggregate["passed"] else "fail"
        consumed = abs(aggregate["mean_signed_bias_c"]) + interval_k * aggregate["se_of_mean_c"]
        rows.append(
            [
                f'<td class="name">{_esc(aggregate["candidate"])}</td>',
                f'<td>{_esc(aggregate["quantity"])}</td>',
                f'<td class="num">{aggregate["cells"]}</td>',
                f'<td class="num">{_fmt(aggregate["mean_signed_bias_c"], 4)}</td>',
                f'<td class="num">{_fmt(aggregate["se_of_mean_c"], 3)}</td>',
                f"<td>{_gauge(consumed, bound, kind)}</td>",
                f'<td>{_pill("PASS" if aggregate["passed"] else "FAIL", kind)}</td>',
            ]
        )
    return (
        "<section><h2>Aggregate bias</h2>"
        '<p class="lede">Per-cell bounds are blind to a small error repeated with the same '
        "sign in every case. This gate catches exactly that: it measures the mean "
        f"<em>signed</em> error across cells against a {bound:g} c budget.</p>"
        + _table(
            [
                "candidate",
                "quantity",
                "cells",
                "mean bias (c)",
                "SE (c)",
                "bias vs bound",
                "passed",
            ],
            rows,
            "No aggregate gates ran (every cell errored).",
        )
        + "</section>"
    )


def _benchmark_section(payload: Mapping[str, Any]) -> str:
    sampling = payload["study"]["sampling"]
    rows = []
    for case, block in sorted(payload["references"].items()):
        if "error" in block:
            rows.append(
                [
                    f"<td>{_esc(case)}</td>",
                    f'<td class="num">{_NA}</td>',
                    f'<td>{_pill("ERROR", "err")}</td>',
                    f'<td class="wrap">{_esc(_first_line(block["error"]))}</td>',
                ]
            )
            continue
        stopped = block["stopped_reason"]
        kind = "pass" if stopped == "se_budget_met" else "warn"
        ses = ", ".join(
            f"{_esc(q)} {_fmt(se, 3)}" for q, se in sorted(block["std_errors"].items())
        )
        rows.append(
            [
                f"<td>{_esc(case)}</td>",
                f'<td class="num">{block["batches"]}</td>',
                f"<td>{_pill(stopped, kind)}</td>",
                f'<td class="name">{ses}</td>',
            ]
        )
    return (
        "<section><h2>Benchmark sampling</h2>"
        '<p class="lede">Sampling continues until the benchmark is sharp enough for the '
        "gate that will judge it. A case that stopped at <code>max_batches</code> never "
        "reached that point, and its cells cannot certify anything.</p>"
        + _table(
            ["case", "batches", "stopped because", "standard errors (raw units)"],
            rows,
            "No benchmark ran.",
        )
        + f'<p class="lede">Policy: {sampling["paths_per_batch"]:,} paths per batch, '
        f'{sampling["min_batches"]}&ndash;{sampling["max_batches"]} batches, seed '
        f'{sampling["seed"]}, relative bump {sampling["bump"]:g}.</p>'
        "</section>"
    )


def _errors_section(payload: Mapping[str, Any]) -> str:
    errored = [cell for cell in payload["cells"] if cell.get("error")]
    if not errored:
        return ""
    rows = [
        [
            f'<td class="name">{_esc(cell["candidate"])}</td>',
            f'<td>{_esc(cell["case"])}</td>',
            f'<td>{_esc(cell["quantity"])}</td>',
            f'<td class="wrap">{_esc(_first_line(cell["error"]))}</td>',
        ]
        for cell in errored
    ]
    return (
        "<section><h2>Errors</h2>"
        '<p class="lede">Full tracebacks are in <code>certificate.json</code>. An errored '
        "cell makes ADMITTED unreachable for that candidate.</p>"
        + _table(["candidate", "case", "quantity", "exception"], rows, "")
        + "</section>"
    )


def _amendment_section(payload: Mapping[str, Any]) -> str:
    amendment = payload.get("amendment")
    if not amendment:
        return ""
    return (
        "<section><h2>Amendment</h2>"
        '<p class="lede">This certificate re-measured part of a parent and carried the '
        "rest forward, so its evidence is a hash chain rather than an independent "
        "claim.</p>"
        '<div class="meta"><dl>'
        f'<dt>parent</dt><dd>{_esc(amendment["parent"])}</dd>'
        f'<dt>parent digest</dt><dd>{_esc(amendment["parent_projected_sha256"])}</dd>'
        f'<dt>reason</dt><dd>{_esc(amendment["reason"])}</dd>'
        f'<dt>replaced</dt><dd>{len(amendment["replaced_cells"])} cell(s)</dd>'
        f'<dt>carried</dt><dd>{len(amendment["carried_cells"])} cell(s)</dd>'
        "</dl></div></section>"
    )


def _provenance_section(payload: Mapping[str, Any]) -> str:
    runtime = payload["runtime"]
    return (
        '<div class="meta"><dl>'
        f'<dt>evidence digest</dt><dd>{_esc(payload["projected_sha256"])}</dd>'
        f'<dt>machine</dt><dd>{_esc(runtime["machine"])} &middot; {_esc(runtime["platform"])}</dd>'
        f'<dt>python</dt><dd>{_esc(runtime["python"])} &middot; numpy {_esc(runtime["numpy"])}</dd>'
        f'<dt>quantark</dt><dd>{_esc(runtime.get("quantark_git_sha") or "unknown")}</dd>'
        f'<dt>schema</dt><dd>{payload["schema"]}</dd>'
        "</dl></div>"
    )


def render_html(payload: Mapping[str, Any]) -> str:
    """Render a certificate as a self-contained HTML report.

    Carries no timestamps, so the page is a pure function of the certified
    evidence: two identical certifications produce identical reports.
    """
    study = payload["study"]
    name = _esc(study["name"])

    quick = ""
    if study.get("quick"):
        quick = (
            '<div class="quick"><strong>Quick mode.</strong> Sampling was shrunk for a '
            "wiring check. This run is not bankable evidence: the benchmark is not "
            "expected to meet its standard-error budget.</div>"
        )

    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Certification: {name}</title>",
        f"<style>{_STYLE}</style>",
        "</head><body><div class='sheet'>",
        '<header class="masthead">',
        '<p class="eyebrow">quantark &middot; engine release certification</p>',
        f"<h1>{name}</h1>",
        '<p class="subtitle">Deterministic engines measured against a statistically '
        "controlled stochastic benchmark, in hedge contracts.</p>",
        "</header>",
        _banner(payload),
        quick,
        _stats(payload),
        _decisions_section(payload),
        _benchmark_section(payload),
        _cells_section(payload),
        _aggregates_section(payload),
        _errors_section(payload),
        _amendment_section(payload),
        _provenance_section(payload),
        "</div></body></html>",
    ]
    return "\n".join(part for part in parts if part)
