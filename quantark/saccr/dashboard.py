"""SA-CCR Dashboard — interactive HTML report for SA-CCR EAD results.

Generates a self-contained HTML page with Plotly interactive charts that
visualise every component of a Basel SA-CCR Exposure-at-Default calculation
(``EAD = alpha * (RC + PFE)``, ``PFE = multiplier * AddOn_aggregate``).

Design mirrors :mod:`quantark.simm.dashboard` (same colour tokens, numeric
font, layout) so the two regulatory dashboards look consistent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ══════════════════════════════════════════════════════════════════════════
# Colour palette (token-aligned — shared with the SIMM dashboard)
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

#: Colour per asset class, keyed by ``AssetClass.name``.
_ASSET_COLORS = {
    "INTEREST_RATE": "#4C72B0",
    "FX": "#937860",
    "CREDIT": "#C44E52",
    "EQUITY": "#55A868",
    "COMMODITY": "#8172B2",
}

#: Human-readable asset-class labels, keyed by ``AssetClass.name``.
_ASSET_LABELS = {
    "INTEREST_RATE": "Interest Rate",
    "FX": "FX",
    "CREDIT": "Credit",
    "EQUITY": "Equity",
    "COMMODITY": "Commodity",
}

#: Hedging-set label prefix -> asset-class name (for colouring hedging sets).
_HS_PREFIX_TO_ASSET = {
    "IR": "INTEREST_RATE",
    "FX": "FX",
    "CREDIT": "CREDIT",
    "EQUITY": "EQUITY",
    "COMMODITY": "COMMODITY",
}

_CURRENCY = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CHF": "Fr"}


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


def _asset_label(name: str) -> str:
    return _ASSET_LABELS.get(name, name.replace("_", " ").title())


def _hs_color(label: str) -> str:
    """Colour a hedging-set label by its asset-class prefix (e.g. ``IR:USD``)."""
    prefix = label.split(":", 1)[0]
    asset = _HS_PREFIX_TO_ASSET.get(prefix)
    if asset is None:
        return "#7f7f7f"
    return _ASSET_COLORS.get(asset, "#7f7f7f")


# ══════════════════════════════════════════════════════════════════════════
# Dashboard class
# ══════════════════════════════════════════════════════════════════════════


class SACCRDashboard:
    """Interactive HTML dashboard for a :class:`SACCRResult`.

    Parameters
    ----------
    result : SACCRResult
        Output of ``SACCRCalculator.calculate()``.
    netting_set : optional
        The :class:`SACCRNettingSet` used (for trade-count / id context).
    currency : str
        Reporting currency symbol used for formatting (SA-CCR uses a single
        reporting currency; the result carries no currency of its own).
    version : str
        Calibration version string shown in the header/footer.
    """

    def __init__(
        self,
        result: Any,
        netting_set: Optional[Any] = None,
        currency: str = "USD",
        version: str = "Basel SA-CCR (BCBS d291)",
    ):
        if not HAS_PLOTLY:
            raise ImportError("plotly is required for SACCRDashboard")
        self._r = result
        self._ns = netting_set
        self._ccy = currency
        self._version = version

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
                self._ead_waterfall(),
                self._asset_class_bar(),
                self._asset_class_donut(),
                self._hedging_set_bar(),
                self._hedging_set_table(),
                self._netting_set_context(),
            ]
        )

        ns_id = getattr(self._ns, "netting_set_id", "") if self._ns else ""
        subtitle = self._version + (f" &middot; {ns_id}" if ns_id else "")

        return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SA-CCR Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" integrity="sha384-7TVmlZWH60iKX5Uk7lSvQhjtcgw2tkFjuwLcXoRSR4zXTyWFJRm9aPAguMh7CIra" crossorigin="anonymous"></script>
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
  .badge {{ display:inline-block; font-family:var(--font-ui); font-size:0.7rem;
    padding:2px 8px; border:1px solid var(--hairline-2); color:var(--warn); margin-top:6px; }}
  footer {{ text-align:center; padding:32px 0 16px; color:var(--ink-2); font-size:0.75rem; }}
  .btn-back {{ position:fixed; top:20px; right:24px; background:var(--paper-3); color:var(--ink); border:1px solid var(--hairline-2); padding:8px 16px; cursor:pointer; font-family:var(--font-ui); font-size:0.8rem; z-index:100; }}
</style>
</head>
<body>
<button class="btn-back" onclick="window.scrollTo({{top:0,behavior:'smooth'}})">↑ Top</button>
<div class="container">
<header>
  <h1>SA-CCR Exposure-at-Default Dashboard</h1>
  <p>{subtitle}</p>
</header>
{charts}
<footer>Generated by QuantArk SA-CCR &middot; EAD = α × (RC + PFE)</footer>
</div>
</body>
</html>"""

    # ------------------------------------------------------------------
    # KPI summary cards
    # ------------------------------------------------------------------

    def _kpi_cards(self) -> str:
        r = self._r
        ead_sub = ""
        if getattr(r, "ead_capped", False):
            ead_sub = f"capped (uncapped {_money(r.ead_uncapped, self._ccy)})"

        cards = [
            ("EAD", _money(r.ead, self._ccy), ead_sub, "primary"),
            ("Replacement Cost", _money(r.rc, self._ccy), _pct(r.rc, r.rc + r.pfe), ""),
            ("PFE", _money(r.pfe, self._ccy), _pct(r.pfe, r.rc + r.pfe), ""),
            ("Add-On (aggregate)", _money(r.addon_aggregate, self._ccy), "", ""),
            ("Multiplier", f"{r.multiplier:.4f}", "PFE = mult × AddOn", ""),
            ("Alpha", f"{r.alpha:.2f}", "margined" if r.is_margined else "unmargined", ""),
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
    # EAD composition waterfall (RC + PFE + alpha uplift -> EAD)
    # ------------------------------------------------------------------

    def _ead_waterfall(self) -> str:
        r = self._r
        base = r.rc + r.pfe
        # EAD = alpha*(RC+PFE) = RC + PFE + (alpha-1)*(RC+PFE). Compute the final
        # step from the actual EAD so the bars always sum to it — exact even when
        # the margined cap (para 129) makes EAD != alpha*(RC+PFE).
        uplift = r.ead - base
        uplift_label = "α (capped)" if getattr(r, "ead_capped", False) else f"α uplift (×{r.alpha:g})"

        labels = ["RC", "PFE", uplift_label, "EAD"]
        values = [r.rc, r.pfe, uplift, r.ead]
        measure = ["relative", "relative", "relative", "total"]

        fig = go.Figure(
            go.Waterfall(
                name="",
                orientation="v",
                measure=measure,
                x=labels,
                y=values,
                text=[_money(v, self._ccy) for v in values],
                textposition="outside",
                connector=dict(line=dict(color=_NEUTRAL["hairline2"], width=1)),
                decreasing=dict(marker_color=_NEUTRAL["ink2"]),
                increasing=dict(marker_color=_ASSET_COLORS["INTEREST_RATE"]),
                totals=dict(
                    marker=dict(
                        color=_NEUTRAL["paper3"],
                        line=dict(color=_NEUTRAL["hairline2"], width=1),
                    )
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
            title=dict(text="EAD Composition Waterfall", font_size=13, x=0.02),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor=_NEUTRAL["hairline"], title=""),
        )
        return (
            '<div class="section-title">EAD COMPOSITION</div>'
            f'<div class="chart-box">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'
        )

    # ------------------------------------------------------------------
    # Add-On by asset class — bar chart
    # ------------------------------------------------------------------

    def _asset_class_bar(self) -> str:
        by_ac = dict(self._r.addon_by_asset_class)
        if not by_ac:
            return ""
        names = list(by_ac)
        labels = [_asset_label(n) for n in names]
        values = list(by_ac.values())
        colours = [_ASSET_COLORS.get(n, "#7f7f7f") for n in names]

        fig = go.Figure(
            go.Bar(
                x=labels,
                y=values,
                marker_color=colours,
                text=[_money(v, self._ccy) for v in values],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>%{text}<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=340,
            margin=dict(l=40, r=40, t=30, b=40),
            paper_bgcolor=_NEUTRAL["paper2"],
            plot_bgcolor=_NEUTRAL["paper2"],
            font=dict(family="Inter Tight, sans-serif", color=_NEUTRAL["ink2"]),
            title=dict(text="Add-On by Asset Class", font_size=13, x=0.02),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor=_NEUTRAL["hairline"], title=""),
        )
        return f'<div class="chart-box">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'

    # ------------------------------------------------------------------
    # Add-On by asset class — donut
    # ------------------------------------------------------------------

    def _asset_class_donut(self) -> str:
        by_ac = dict(self._r.addon_by_asset_class)
        if not by_ac:
            return ""
        names = list(by_ac)
        labels = [_asset_label(n) for n in names]
        values = list(by_ac.values())
        colours = [_ASSET_COLORS.get(n, "#7f7f7f") for n in names]

        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker_colors=colours,
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>%{value:,.0f} "
                + self._ccy
                + "<br>%{percent}<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor=_NEUTRAL["paper2"],
            font=dict(family="Inter Tight, sans-serif", color=_NEUTRAL["ink2"]),
            title=dict(text="Add-On Share by Asset Class", font_size=13, x=0.02),
            legend=dict(orientation="h", y=-0.1),
        )
        return f'<div class="chart-box">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'

    # ------------------------------------------------------------------
    # Add-On by hedging set — horizontal bar
    # ------------------------------------------------------------------

    def _hedging_set_bar(self) -> str:
        by_hs = self._r.addon_by_hedging_set
        if not by_hs:
            return ""
        # Sort ascending so the largest bar renders at the top of a horizontal bar.
        items = sorted(by_hs.items(), key=lambda kv: kv[1])
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        colours = [_hs_color(k) for k in labels]

        fig = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=colours,
                text=[_money(v, self._ccy) for v in values],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>%{text}<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=max(300, 40 * len(labels) + 80),
            margin=dict(l=120, r=60, t=30, b=40),
            paper_bgcolor=_NEUTRAL["paper2"],
            plot_bgcolor=_NEUTRAL["paper2"],
            font=dict(family="Inter Tight, sans-serif", color=_NEUTRAL["ink2"]),
            title=dict(text="Add-On by Hedging Set", font_size=13, x=0.02),
            xaxis=dict(showgrid=True, gridcolor=_NEUTRAL["hairline"], title=""),
            yaxis=dict(showgrid=False),
        )
        return f'<div class="chart-box">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'

    # ------------------------------------------------------------------
    # Hedging-set detail table
    # ------------------------------------------------------------------

    def _hedging_set_table(self) -> str:
        by_hs = self._r.addon_by_hedging_set
        if not by_hs:
            return ""
        total = sum(by_hs.values())
        rows = sorted(by_hs.items(), key=lambda kv: abs(kv[1]), reverse=True)

        thead = (
            "<thead><tr>"
            + "".join(f"<th>{h}</th>" for h in ["Hedging Set", "Asset Class", "Add-On", "Share"])
            + "</tr></thead>"
        )
        tbody = "<tbody>"
        for label, value in rows:
            prefix = label.split(":", 1)[0]
            asset = _HS_PREFIX_TO_ASSET.get(prefix) or prefix
            tbody += (
                "<tr>"
                f"<td>{label}</td>"
                f"<td>{_asset_label(asset)}</td>"
                f"<td>{_money(value, self._ccy)}</td>"
                f"<td>{_pct(value, total)}</td>"
                "</tr>"
            )
        tbody += "</tbody>"

        return (
            '<div class="section-title">HEDGING SET DETAIL</div>'
            f'<div class="table-wrap"><table>{thead}{tbody}</table></div>'
        )

    # ------------------------------------------------------------------
    # Netting-set context
    # ------------------------------------------------------------------

    def _netting_set_context(self) -> str:
        r = self._r
        parts = [
            '<div class="card">'
            '<div class="card__label">Market Value (V)</div>'
            f'<div class="card__value">{_money(r.v, self._ccy)}</div>'
            f'<div class="card__sub">Collateral (C) {_money(r.c, self._ccy)}</div>'
            "</div>",
            '<div class="card">'
            '<div class="card__label">NICA</div>'
            f'<div class="card__value">{_money(r.nica, self._ccy)}</div>'
            f'<div class="card__sub">{"margined" if r.is_margined else "unmargined"}</div>'
            "</div>",
        ]

        if self._ns is not None:
            trades = getattr(self._ns, "trades", []) or []
            mpor = getattr(self._ns, "mpor_days", None)
            mpor_sub = f"MPOR {mpor}d" if (r.is_margined and mpor is not None) else ""
            parts.append(
                '<div class="card">'
                '<div class="card__label">Netting Set</div>'
                f'<div class="card__value">{len(trades)} trades</div>'
                f'<div class="card__sub">{mpor_sub}</div>'
                "</div>"
            )

        if getattr(r, "ead_capped", False):
            parts.append(
                '<div class="card">'
                '<div class="card__label">Margined Cap</div>'
                f'<div class="card__value">{_money(r.ead, self._ccy)}</div>'
                f'<div class="card__sub">uncapped {_money(r.ead_uncapped, self._ccy)}</div>'
                '<div class="badge">EAD capped at unmargined (para 129)</div>'
                "</div>"
            )

        return (
            '<div class="section-title">NETTING SET CONTEXT</div>'
            f'<div class="row">{"".join(parts)}</div>'
        )
