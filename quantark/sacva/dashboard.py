"""SA-CVA dashboard — interactive HTML report for a SACVAResult.

Single-file HTML page; Plotly is loaded from the same pinned CDN as the
quantark.simm / quantark.saccr dashboards (shared colour tokens, fonts).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

_NEUTRAL = {
    "paper": "#111110", "paper2": "#1c1b1a", "paper3": "#292725",
    "ink": "#f5f2e8", "ink2": "#9b978d", "hairline": "#3d3830", "hairline2": "#5c554a",
}
_RC_COLORS = {
    "INTEREST_RATE": "#4C72B0", "FX": "#937860", "COUNTERPARTY_CREDIT": "#C44E52",
    "REFERENCE_CREDIT": "#DD8452", "EQUITY": "#55A868", "COMMODITY": "#8172B2",
}


def _money(x: float, ccy: str = "USD") -> str:
    if abs(x) >= 1e6:
        return f"{ccy} {x / 1e6:,.2f}M"
    return f"{ccy} {x:,.0f}"


class SACVADashboard:
    """Interactive HTML dashboard for a :class:`SACVAResult`."""

    def __init__(self, result: Any, portfolio: Optional[Any] = None,
                 currency: str = "USD", version: str = "Basel SA-CVA (MAR50)"):
        if not HAS_PLOTLY:
            raise ImportError("plotly is required for SACVADashboard")
        self._r = result
        self._pf = portfolio
        self._ccy = currency
        self._version = version

    def generate(self, path: str | Path) -> str:
        Path(path).write_text(self._build_html(), encoding="utf-8")
        return str(path)

    def _build_html(self) -> str:
        r = self._r
        kpis = "".join(
            f'<div class="card"><div class="lbl">{lbl}</div>'
            f'<div class="val">{val}</div></div>'
            for lbl, val in [
                ("SA-CVA Capital", _money(r.total_capital, self._ccy)),
                ("Delta", _money(r.delta_capital, self._ccy)),
                ("Vega", _money(r.vega_capital, self._ccy)),
                ("m_CVA", f"{r.m_cva:.2f}"),
            ]
        )
        delta_vega = self._delta_vega_bar()
        class_bar = self._class_bar()
        table = self._bucket_table()
        return f"""<!DOCTYPE html><html lang="en" data-theme="dark"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SA-CVA Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" integrity="sha384-7TVmlZWH60iKX5Uk7lSvQhjtcgw2tkFjuwLcXoRSR4zXTyWFJRm9aPAguMh7CIra" crossorigin="anonymous"></script>
<style>
body{{background:{_NEUTRAL['paper']};color:{_NEUTRAL['ink']};font-family:Inter Tight,Arial,sans-serif;padding:24px;}}
.row{{display:flex;gap:24px;flex-wrap:wrap;margin-bottom:24px;}}
.card{{flex:1 1 200px;border:1px solid {_NEUTRAL['hairline']};background:{_NEUTRAL['paper2']};padding:20px;}}
.lbl{{font-size:.7rem;text-transform:uppercase;color:{_NEUTRAL['ink2']};letter-spacing:.06em;}}
.val{{font-size:1.5rem;font-weight:600;margin-top:6px;font-family:monospace;}}
table{{width:100%;border-collapse:collapse;}}
th,td{{padding:8px 12px;border-bottom:1px solid {_NEUTRAL['hairline']};text-align:right;font-family:monospace;font-size:.85rem;}}
th{{color:{_NEUTRAL['ink2']};text-transform:uppercase;font-size:.65rem;}}
th:first-child,td:first-child{{text-align:left;}}
h1{{font-size:1.5rem;}} h3{{margin-top:24px;}} p{{color:{_NEUTRAL['ink2']};}}
</style></head><body>
<h1>SA-CVA Capital Dashboard</h1><p>{self._version}</p>
<div class="row">{kpis}</div>
{delta_vega}
{class_bar}
{table}
<p>SA-CVA = K_delta + K_vega &middot; K = m_CVA&middot;&radic;(&Sigma;K_b&sup2; + &Sigma;&Sigma;&gamma;&middot;S_b&middot;S_c)</p>
</body></html>"""

    def _delta_vega_bar(self) -> str:
        r = self._r
        fig = go.Figure(go.Bar(
            x=["Delta", "Vega"], y=[r.delta_capital, r.vega_capital],
            marker_color=["#4C72B0", "#DD8452"],
            text=[_money(r.delta_capital, self._ccy), _money(r.vega_capital, self._ccy)],
            textposition="outside"))
        fig.update_layout(template="plotly_dark", height=320,
                          paper_bgcolor=_NEUTRAL["paper2"], plot_bgcolor=_NEUTRAL["paper2"],
                          title="Delta vs Vega Capital", margin=dict(l=40, r=40, t=40, b=40))
        return f'<div>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'

    def _class_bar(self) -> str:
        by = self._r.by_risk_class
        if not by:
            return ""
        labels = list(by.keys())
        values = list(by.values())
        colors = [_RC_COLORS.get(k.split(":")[0], "#7f7f7f") for k in labels]
        fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors,
                               text=[_money(v, self._ccy) for v in values],
                               textposition="outside"))
        fig.update_layout(template="plotly_dark", height=360,
                          paper_bgcolor=_NEUTRAL["paper2"], plot_bgcolor=_NEUTRAL["paper2"],
                          title="Capital by Risk Class x Type",
                          margin=dict(l=40, r=40, t=40, b=90))
        return f'<div>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'

    def _bucket_table(self) -> str:
        by = self._r.by_bucket
        if not by:
            return ""
        rows = "".join(
            f"<tr><td>{k}</td><td>{_money(v, self._ccy)}</td>"
            f"<td>{_money(self._r.bucket_s_b.get(k, 0.0), self._ccy)}</td>"
            f"<td>{self._r.hedge_disallowance.get(k, 0.0):,.2f}</td></tr>"
            for k, v in sorted(by.items(), key=lambda kv: -abs(kv[1]))
        )
        return ("<h3>Bucket detail</h3><table><thead><tr>"
                "<th>Bucket</th><th>K_b</th><th>S_b</th><th>Hedge disallow.</th>"
                f"</tr></thead><tbody>{rows}</tbody></table>")
