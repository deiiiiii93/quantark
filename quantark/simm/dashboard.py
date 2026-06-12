"""
SIMM Dashboard — interactive HTML report for SIMM v2.6 results.

Generates a self-contained HTML page with Plotly interactive charts
that visualise every component of a SIMM initial margin calculation.

Design follows the project UI style guide (colour tokens, numeric font, etc.).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ══════════════════════════════════════════════════════════════════════════
# Colour palette (token-aligned — see UI_STYLE_GUIDE.md)
# ══════════════════════════════════════════════════════════════════════════

_NEUTRAL = {
    "paper": "#111110",  # dark surface
    "paper2": "#1c1b1a",
    "paper3": "#292725",
    "ink": "#f5f2e8",  # cream primary text
    "ink2": "#9b978d",  # secondary text
    "hairline": "#3d3830",
    "hairline2": "#5c554a",
    "white": "#f5f2e8",
}

_PALETTE = {
    "RatesFX": "#4C72B0",
    "Equity": "#55A868",
    "Credit": "#C44E52",
    "Commodity": "#8172B2",
    "IR": "#4C72B0",
    "Equity_rc": "#55A868",
    "CreditQ": "#C44E52",
    "CreditNQ": "#DD8452",
    "Commodity_rc": "#8172B2",
    "FX": "#937860",
    "Delta": "#4C72B0",
    "Vega": "#55A868",
    "Curvature": "#DD8452",
    "BaseCorr": "#C44E52",
}

_CURRENCY = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CHF": "Fr"}


def _normalise_product_class_dict(d: dict) -> dict[str, float]:
    """Convert enum-keyed product class dict to string-keyed."""
    return {
        k.value if hasattr(k, "value") else str(k): v
        for k, v in d.items()
    }


def _normalise_risk_class_dict(d: dict) -> dict[str, float]:
    """Convert enum-keyed risk class dict to string-keyed."""
    return {
        k.value if hasattr(k, "value") else str(k): v
        for k, v in d.items()
    }


def _money(amount: float, ccy: str = "USD") -> str:
    """Format a monetary value with currency symbol."""
    sym = _CURRENCY.get(ccy, ccy + " ")
    if abs(amount) >= 1e6:
        return f"{sym}{amount / 1e6:,.2f}M"
    return f"{sym}{amount:,.0f}"


def _pct(value: float, total: float) -> str:
    if total == 0:
        return "0.0%"
    return f"{value / total * 100:.1f}%"


# ══════════════════════════════════════════════════════════════════════════
# Dashboard class
# ══════════════════════════════════════════════════════════════════════════


class SIMMDashboard:
    """Interactive HTML dashboard for a SIMMAggregationResult.

    Parameters
    ----------
    result : SIMMAggregationResult
        Output of SIMMCalculator.calculate().
    equity_portfolio : optional
        The equity portfolio used to generate sensitivities (for context).
    fi_portfolio : optional
        The fixed-income portfolio used (for context).
    """

    def __init__(
        self,
        result: Any,
        equity_portfolio: Optional[Any] = None,
        fi_portfolio: Optional[Any] = None,
    ):
        if not HAS_PLOTLY:
            raise ImportError("plotly is required for SIMMDashboard")
        self._r = result
        self._ccy = getattr(result, "calculation_currency", "USD")
        self._total = result.total_margin
        self._eq = equity_portfolio
        self._fi = fi_portfolio

        # --- pre-compute rich taxonomy ---
        self._risk_class_names = {
            rc: rc.value for rc in self._r.by_risk_class
        }
        self._flattened = self._flatten_buckets()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(self, path: str | Path) -> str:
        """Write the dashboard to *path* (HTML). Returns *path*."""
        html = self._build_html()
        Path(path).write_text(html, encoding="utf-8")
        return str(path)

    # ------------------------------------------------------------------
    # HTML assembly
    # ------------------------------------------------------------------

    def _build_html(self) -> str:
        charts = "\n".join(
            [
                self._kpi_cards(),
                self._product_class_bar(),
                self._risk_class_sunburst(),
                self._margin_type_stacked(),
                self._bucket_table(),
                self._waterfall(),
                self._portfolio_context(),
            ]
        )

        return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SIMM Dashboard — v{getattr(self._r,'simm_version','2.6')}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root {{
    --paper: {_NEUTRAL['paper']}; --paper-2: {_NEUTRAL['paper2']}; --paper-3: {_NEUTRAL['paper3']};
    --ink: {_NEUTRAL['ink']}; --ink-2: {_NEUTRAL['ink2']};
    --hairline: {_NEUTRAL['hairline']}; --hairline-2: {_NEUTRAL['hairline2']};
    --font-ui: 'Inter Tight', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    --font-numeric: 'Berkeley Mono', 'JetBrains Mono', 'SF Mono', monospace;
    --pos: #3cb371; --neg: #e45756; --warn: #f0ad4e; --info: #4c72b0;
    --gap-2: 12px; --gap-4: 24px; --panel-padding: 24px;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: var(--font-ui); background: var(--paper); color: var(--ink);
    line-height: 1.5; padding: var(--gap-4);
  }}
  .container {{ max-width: 1320px; margin:0 auto; }}
  header {{ text-align:center; padding: 32px 0 24px; }}
  header h1 {{ font-size: 1.6rem; font-weight: 600; letter-spacing: -0.01em; }}
  header p {{ font-size: 0.85rem; color: var(--ink-2); margin-top: 4px; }}
  .row {{ display:flex; flex-wrap:wrap; gap: var(--gap-4); margin-bottom: var(--gap-4); }}
  .card {{
    flex:1 1 220px; min-width:0; border:1px solid var(--hairline);
    background: var(--paper-2); padding: 20px 24px; border-radius:0;
  }}
  .card__label {{ font-size: 0.65rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--ink-2); }}
  .card__value {{ font-family:var(--font-numeric); font-size:1.5rem; font-weight:600; margin-top:6px; }}
  .card__sub {{ font-family:var(--font-numeric); font-size:0.78rem; color:var(--ink-2); margin-top:2px; }}
  .card--primary {{ border-color: var(--hairline-2); background: var(--paper-3); }}
  .section-title {{
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--ink-2); padding: 20px 0 8px;
  }}
  .chart-box {{
    background: var(--paper-2); border: 1px solid var(--hairline);
    border-radius: 0; margin-bottom: var(--gap-4);
  }}
  .table-wrap {{
    overflow-x:auto; background:var(--paper-2); border:1px solid var(--hairline);
    padding:var(--panel-padding);
  }}
  table {{ width:100%; border-collapse:collapse; }}
  thead th {{
    font-size:0.65rem; text-transform:uppercase; letter-spacing:0.05em; color:var(--ink-2);
    text-align:left; padding:8px 12px; border-bottom:1px solid var(--hairline-2);
  }}
  tbody td {{
    font-family:var(--font-numeric); font-size:0.85rem; padding:8px 12px;
    border-bottom:1px solid var(--hairline); tabular-nums:;
  }}
  tbody td:not(:first-child) {{ text-align:right; }}
  tbody tr:hover {{ background:var(--paper-3); }}
  .num-pos {{ color:var(--pos); }} .num-neg {{ color:var(--neg); }}
  footer {{ text-align:center; padding:32px 0 16px; color:var(--ink-2); font-size:0.75rem; }}
  .btn-back {{ position:fixed; top:20px; right:24px; background:var(--paper-3); color:var(--ink); border:1px solid var(--hairline-2); padding:8px 16px; cursor:pointer; font-family:var(--font-ui); font-size:0.8rem; z-index:100; }}
</style>
</head>
<body>
<button class="btn-back" onclick="window.scrollTo({{top:0,behavior:'smooth'}})">↑ Top</button>
<div class="container">
<header>
  <h1>SIMM Initial Margin Dashboard</h1>
  <p>ISDA SIMM v{getattr(self._r,'simm_version','2.6')} &middot; {getattr(self._r,'calculation_timestamp','')[:10]}</p>
</header>
{charts}
<footer>Generated by QuantArk SIMM v{getattr(self._r,'simm_version','2.6')}</footer>
</div>
</body>
</html>"""

    # ------------------------------------------------------------------
    # KPI summary cards
    # ------------------------------------------------------------------

    def _kpi_cards(self) -> str:
        m = self._r
        by_pc = _normalise_product_class_dict(m.by_product_class)

        rf_im = by_pc.get("RatesFX", 0.0)
        eq_im = by_pc.get("Equity", 0.0)
        cr_im = by_pc.get("Credit", 0.0)

        cards = [
            ("Total SIMM", _money(self._total, self._ccy), "", "primary"),
            ("RatesFX IM", _money(rf_im, self._ccy), _pct(rf_im, self._total), ""),
            ("Equity IM", _money(eq_im, self._ccy), _pct(eq_im, self._total), ""),
            ("Credit IM", _money(cr_im, self._ccy), _pct(cr_im, self._total), ""),
            ("Add-Ons", _money(m.addon.total_addon if m.addon else 0, self._ccy), "", ""),
            ("Risk Classes", str(len(m.by_risk_class)), "", ""),
        ]
        parts = [
            f'<div class="card {"card--primary" if cls else ""}" '
            f'style="{"border-left:4px solid var(--info)" if cls else ""}">'
            f'<div class="card__label">{label}</div>'
            f'<div class="card__value">{value}</div>'
            f'<div class="card__sub">{sub}</div>'
            f"</div>"
            for label, value, sub, cls in cards
        ]
        return f'<div class="section-title">SUMMARY</div><div class="row">{"".join(parts)}</div>'

    # ------------------------------------------------------------------
    # Product class bar chart
    # ------------------------------------------------------------------

    def _product_class_bar(self) -> str:
        by_pc = _normalise_product_class_dict(self._r.by_product_class)
        labels = list(by_pc)
        values = list(by_pc.values())
        colours = [_PALETTE.get(l, "#7f7f7f") for l in labels]

        fig = go.Figure(
            go.Bar(
                x=labels,
                y=values,
                marker_color=colours,
                text=[_money(v, self._ccy) for v in values],
                textposition="outside",
                hovertemplate="<b>%{{x}}</b><br>%{{text}}<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=340,
            margin=dict(l=40, r=40, t=30, b=40),
            paper_bgcolor=_NEUTRAL["paper2"],
            plot_bgcolor=_NEUTRAL["paper2"],
            font=dict(family="Inter Tight, sans-serif", color=_NEUTRAL["ink2"]),
            title=dict(text="SIMM by Product Class", font_size=13, x=0.02),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor=_NEUTRAL["hairline"], title=""),
        )
        return f'<div class="chart-box">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'

    # ------------------------------------------------------------------
    # Risk class sunburst (hierarchical donut)
    # ------------------------------------------------------------------

    def _risk_class_sunburst(self) -> str:
        by_rc = _normalise_risk_class_dict(self._r.by_risk_class)
        labels = list(by_rc)
        values = list(by_rc.values())
        colours = [_PALETTE.get(f"{l}_rc", _PALETTE.get(l, "#7f7f7f")) for l in labels]

        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker_colors=colours,
                textinfo="label+percent",
                hovertemplate="<b>%{{label}}</b><br>%{{value:,.0f}} USD<br>%{{percent}}<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor=_NEUTRAL["paper2"],
            font=dict(family="Inter Tight, sans-serif", color=_NEUTRAL["ink2"]),
            title=dict(text="SIMM by Risk Class", font_size=13, x=0.02),
            legend=dict(orientation="h", y=-0.1),
        )
        return f'<div class="chart-box">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'

    # ------------------------------------------------------------------
    # Margin type stacked bar (risk class × margin type)
    # ------------------------------------------------------------------

    def _margin_type_stacked(self) -> str:
        by_type: dict[str, dict[str, float]] = {}
        for rc, mt_dict in self._r.by_margin_type.items():
            rc_name = rc.value if hasattr(rc, "value") else str(rc)
            for mt, val in mt_dict.items():
                mt_name = mt.value if hasattr(mt, "value") else str(mt)
                by_type.setdefault(mt_name, {})[rc_name] = val

        margin_types = list(by_type)
        risk_classes = sorted(
            {rc for d in by_type.values() for rc in d},
            key=lambda rc: sum(d.get(rc, 0) for d in by_type.values()),
            reverse=True,
        )

        traces = []
        for mt in margin_types:
            traces.append(
                go.Bar(
                    name=mt,
                    x=risk_classes,
                    y=[by_type[mt].get(rc, 0) for rc in risk_classes],
                    marker_color=_PALETTE.get(mt, "#7f7f7f"),
                    hovertemplate="<b>%{{x}}</b> %{{data.name}}<br>%{{y:,.0f}} USD<extra></extra>",
                )
            )

        fig = go.Figure(traces)
        fig.update_layout(
            barmode="stack",
            template="plotly_dark",
            height=380,
            margin=dict(l=40, r=40, t=30, b=40),
            paper_bgcolor=_NEUTRAL["paper2"],
            plot_bgcolor=_NEUTRAL["paper2"],
            font=dict(family="Inter Tight, sans-serif", color=_NEUTRAL["ink2"]),
            title=dict(text="Margin Type Breakdown by Risk Class", font_size=13, x=0.02),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor=_NEUTRAL["hairline"], title=""),
            legend=dict(orientation="h", y=1.08, xanchor="left", x=0),
        )
        return f'<div class="chart-box">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'

    # ------------------------------------------------------------------
    # Bucket detail table
    # ------------------------------------------------------------------

    def _flatten_buckets(self) -> list[dict]:
        """Extract bucket-level data from SIMMAggregationResult."""
        rows: list[dict] = []
        bd = getattr(self._r, "bucket_details", {}) or {}
        for pc, rc_dict in bd.items():
            pc_name = pc.value if hasattr(pc, "value") else str(pc)
            for rc, mt_dict in rc_dict.items():
                rc_name = rc.value if hasattr(rc, "value") else str(rc)
                for mt, buckets in mt_dict.items():
                    mt_name = mt.value if hasattr(mt, "value") else str(mt)
                    for bk, br in buckets.items():
                        rows.append(
                            {
                                "product_class": pc_name,
                                "risk_class": rc_name,
                                "margin_type": mt_name,
                                "bucket": str(bk),
                                "k": getattr(br, "k", 0),
                                "ws_sum": getattr(br, "ws_sum", 0),
                                "s_sum": getattr(br, "s_sum", 0),
                            }
                        )
        rows.sort(key=lambda r: abs(r["k"]), reverse=True)
        return rows

    def _bucket_table(self) -> str:
        rows = self._flattened[:30]  # top 30
        if not rows:
            return ""

        thead = (
            "<thead><tr>"
            + "".join(
                f"<th>{h}</th>"
                for h in ["Product", "Risk", "Margin", "Bucket", "K", "WS Sum", "S Sum"]
            )
            + "</tr></thead>"
        )
        tbody = "<tbody>"
        for r in rows:
            k_cls = "num-pos" if r["k"] > 0 else "num-neg" if r["k"] < 0 else ""
            tbody += (
                f"<tr>"
                f"<td>{r['product_class']}</td>"
                f"<td>{r['risk_class']}</td>"
                f"<td>{r['margin_type']}</td>"
                f"<td>{r['bucket']}</td>"
                f'<td class="{k_cls}">{r["k"]:,.0f}</td>'
                f'<td>{r["ws_sum"]:,.0f}</td>'
                f'<td>{r["s_sum"]:,.0f}</td>'
                f"</tr>"
            )
        tbody += "</tbody>"

        return (
            f'<div class="section-title">BUCKET DETAIL</div>'
            f'<div class="table-wrap"><table>{thead}{tbody}</table></div>'
        )

    # ------------------------------------------------------------------
    # Waterfall — how total SIMM is built from components
    # ------------------------------------------------------------------

    def _waterfall(self) -> str:
        pc_margins = _normalise_product_class_dict(self._r.by_product_class)

        labels = list(pc_margins)
        values = list(pc_margins.values())

        addon = self._r.addon.total_addon if self._r.addon else 0.0
        if addon:
            labels.append("Add-Ons")
            values.append(addon)

        cumulative = 0.0
        measure: list[str] = ["relative"] * len(values)
        waterfall_x: list[float] = []
        for v in values:
            waterfall_x.append(v)

        fig = go.Figure(
            go.Waterfall(
                name="",
                orientation="v",
                measure=measure,
                x=labels,
                y=waterfall_x,
                text=[_money(v, self._ccy) for v in values],
                textposition="outside",
                connector=dict(line=dict(color=_NEUTRAL["hairline2"], width=1)),
                decreasing=dict(marker_color=_PALETTE.get("RatesFX", "#7f7f7f")),
                increasing=dict(marker_color=_PALETTE.get("Equity", "#7f7f7f")),
                totals=dict(
                    marker=dict(color=_NEUTRAL["paper3"], line=dict(color=_NEUTRAL["hairline2"], width=1))
                ),
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=380,
            margin=dict(l=40, r=40, t=30, b=40),
            paper_bgcolor=_NEUTRAL["paper2"],
            plot_bgcolor=_NEUTRAL["paper2"],
            font=dict(family="Inter Tight, sans-serif", color=_NEUTRAL["ink2"]),
            title=dict(text="SIMM Composition Waterfall", font_size=13, x=0.02),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor=_NEUTRAL["hairline"], title=""),
        )
        return f'<div class="chart-box">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'

    # ------------------------------------------------------------------
    # Portfolio context section
    # ------------------------------------------------------------------

    def _portfolio_context(self) -> str:
        parts = []

        if self._eq and hasattr(self._eq, "positions"):
            n = len(self._eq.positions)
            val = getattr(self._eq, "get_portfolio_value", lambda *a: 0)()
            parts.append(
                f'<div class="card">'
                f'<div class="card__label">Equity Portfolio</div>'
                f'<div class="card__value">{n} positions</div>'
                f'<div class="card__sub">MV {_money(val, self._ccy)}</div>'
                f"</div>"
            )

        if self._fi and hasattr(self._fi, "positions"):
            n = len(self._fi.positions)
            val = getattr(self._fi, "get_portfolio_value", lambda *a: 0)()
            parts.append(
                f'<div class="card">'
                f'<div class="card__label">Fixed Income Portfolio</div>'
                f'<div class="card__value">{n} positions</div>'
                f'<div class="card__sub">MV {_money(val, self._ccy)}</div>'
                f"</div>"
            )

        # Sensitivity counts
        n_rc = len(self._r.by_risk_class)
        n_bk = len(self._flattened)
        parts.append(
            f'<div class="card">'
            f'<div class="card__label">Sensitivities</div>'
            f'<div class="card__value">{n_rc} risk classes</div>'
            f'<div class="card__sub">{n_bk} bucket results</div>'
            f"</div>"
        )

        if not parts:
            return ""

        return (
            f'<div class="section-title">PORTFOLIO CONTEXT</div>'
            f'<div class="row">{"".join(parts)}</div>'
        )
