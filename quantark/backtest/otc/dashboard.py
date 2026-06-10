"""
Static HTML dashboard for OTC autocallable backtest results.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any, Sequence

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.report.autocallable_risk_report import (
    build_snowball_risk_snapshot,
)
from quantark.asset.equity.report.surfaces import GridSpec
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

from .engine_factory import create_surface_engine
from .market import ImpliedBasisYield, SignedDividendYield


DEFAULT_ROLE_LAYOUTS = ("executive", "trader", "risk_manager", "quant")


@dataclass
class AutocallableDashboardConfig:
    """Configuration for static OTC autocallable dashboard generation."""

    title: str = "OTC Autocallable Backtest Dashboard"
    include_plotlyjs: bool = True
    role_layouts: tuple[str, ...] = DEFAULT_ROLE_LAYOUTS
    full_surface_snapshots: bool = True
    high_accuracy_surfaces: bool = False
    snapshot_dates: Sequence[Any] | None = None
    snapshot_grid_spec: GridSpec = field(
        default_factory=lambda: GridSpec(spot_nodes=9, q_nodes=7, vol_nodes=7)
    )


class AutocallableBacktestDashboard:
    """Render a self-contained Plotly HTML dashboard for OTC backtest results."""

    def __init__(
        self,
        results: Any,
        config: AutocallableDashboardConfig | None = None,
    ) -> None:
        self.results = results
        self.config = config or AutocallableDashboardConfig()

    def write_html(self, path: str | Path) -> Path:
        """Write the dashboard HTML file and return its path."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._build_payload()
        output_path.write_text(self._render_html(payload), encoding="utf-8")
        return output_path

    def _build_payload(self) -> dict[str, Any]:
        states = self.results.states_df
        greeks = self.results.greeks_df
        rebalances = self.results.rebalance_df
        trades = self.results.trades_df
        actions = self.results.actions_df
        daily_events = self.results.daily_event_summary_df
        event_probabilities = self.results.event_probability_df
        surfaces = self.results.surfaces_df
        summary = self.results.get_summary()

        risk_snapshots = []
        snapshot_errors = []
        if self.config.full_surface_snapshots:
            risk_snapshots, snapshot_errors = self._build_risk_snapshots(states)

        payload = {
            "metadata": {
                "title": self.config.title,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "underlying": getattr(self.results.config, "underlying", "underlying"),
                "role_layouts": list(self.config.role_layouts),
                "surface_mode": (
                    "point-greeks"
                    if self.config.high_accuracy_surfaces
                    else "finite-difference"
                ),
            },
            "summary": summary,
            "warnings": self._risk_warnings(states, greeks, daily_events, risk_snapshots),
            "tables": {
                "states": _frame_records(states),
                "greeks": _frame_records(greeks),
                "rebalances": _frame_records(rebalances),
                "trades": _frame_records(trades),
                "actions": _frame_records(actions),
                "daily_events": _frame_records(daily_events),
                "event_probabilities": _frame_records(event_probabilities),
                "surfaces": _frame_records(surfaces),
            },
            "risk_snapshots": risk_snapshots,
            "snapshot_errors": snapshot_errors,
        }
        return _to_jsonable(payload)

    def _build_risk_snapshots(
        self, states: pd.DataFrame
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        product = self.results.config.product
        if not isinstance(product, SnowballOption):
            return [], [
                {
                    "message": (
                        "Full risk-suite snapshots are currently implemented for "
                        f"SnowballOption, got {type(product).__name__}."
                    )
                }
            ]

        dates = self._snapshot_dates(states)
        snapshots: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        surface_engine = create_surface_engine(
            self.results.config.product, self.results.config.engine_config
        )

        for date in dates:
            try:
                env = self._pricing_env_for_date(date, states)
                snapshot_product = self._product_for_date(date, env, states)
                snapshot = build_snowball_risk_snapshot(
                    product=snapshot_product,
                    pricing_env=env,
                    engine=surface_engine,
                    label=str(date.date()),
                    grid_spec=self.config.snapshot_grid_spec,
                    high_accuracy_surfaces=self.config.high_accuracy_surfaces,
                )
                self._add_cash_snapshot_surfaces(snapshot)
                snapshots.append(snapshot)
            except Exception as exc:
                errors.append({"date": str(date.date()), "message": str(exc)})
        return snapshots, errors

    def _add_cash_snapshot_surfaces(self, snapshot: dict[str, Any]) -> None:
        base = snapshot.get("base", {})
        spot = _finite_float(base.get("spot"))
        delta = _finite_float(base.get("delta"))
        gamma = _finite_float(base.get("gamma"))
        if spot is not None and delta is not None:
            base["delta_cash_1pct"] = delta * spot * 0.01
        if spot is not None and gamma is not None:
            base["gamma_cash_1pct"] = gamma * spot**2 / 100.0

        surfaces = snapshot.get("surfaces")
        if not isinstance(surfaces, list):
            return
        by_key = {surface.get("key"): surface for surface in surfaces}
        cash_surfaces = []
        delta_surface = by_key.get("delta_spot_div")
        if delta_surface is not None:
            cash_surfaces.append(
                _cash_surface_payload(
                    delta_surface,
                    key="delta_cash_1pct_spot_div",
                    label="Delta Cash 1% vs Spot x Dividend",
                    z_label="Delta cash 1%",
                )
            )
        gamma_surface = by_key.get("gamma_spot_div")
        if gamma_surface is not None:
            cash_surfaces.append(
                _cash_surface_payload(
                    gamma_surface,
                    key="gamma_cash_1pct_spot_div",
                    label="Gamma Cash 1% vs Spot x Dividend",
                    z_label="Gamma cash 1%",
                    gamma=True,
                )
            )
        for cash_surface in reversed([s for s in cash_surfaces if s is not None]):
            surfaces.insert(1, cash_surface)

    def _snapshot_dates(self, states: pd.DataFrame) -> list[pd.Timestamp]:
        if states.empty:
            return []
        selected: set[pd.Timestamp] = {
            pd.Timestamp(states.index.min()).normalize(),
            pd.Timestamp(states.index.max()).normalize(),
        }

        actions = self.results.actions_df
        if not actions.empty:
            selected.update(pd.Timestamp(d).normalize() for d in actions.index)

        trades = self.results.trades_df
        if not trades.empty and "trade_type" in trades.columns:
            roll_trades = trades[
                trades["trade_type"].astype(str).str.startswith("roll_", na=False)
            ]
            selected.update(pd.Timestamp(d).normalize() for d in roll_trades.index)

        if self.config.snapshot_dates:
            selected.update(
                pd.Timestamp(d).normalize() for d in self.config.snapshot_dates
            )

        available = {pd.Timestamp(d).normalize() for d in states.index}
        return sorted(date for date in selected if date in available)

    def _pricing_env_for_date(
        self, date: pd.Timestamp, states: pd.DataFrame
    ) -> PricingEnvironment:
        date = pd.Timestamp(date).normalize()
        market = self.results.config.market_data.get_market_row(date)
        state = states.loc[date]
        implied_q = float(_row_value(state, "implied_q", 0.0))
        basis_yield = float(_row_value(state, "basis_yield", 0.0))
        return PricingEnvironment(
            spot_quote=SpotQuote(
                spot=float(market["spot"]),
                asset_name=getattr(self.results.config, "underlying", "underlying"),
            ),
            vol_surface=FlatVolSurface(volatility=float(market["volatility"])),
            rate_curve=FlatRateCurve(rate=float(market["rate"])),
            div_yield=SignedDividendYield(implied_q),
            basis_yield=ImpliedBasisYield(basis_yield),
            valuation_date=date.to_pydatetime(),
        )

    def _product_for_date(
        self,
        date: pd.Timestamp,
        pricing_env: PricingEnvironment,
        states: pd.DataFrame,
    ) -> SnowballOption:
        date = pd.Timestamp(date).normalize()
        product = copy.deepcopy(self.results.config.product)
        state = states.loc[date]
        setattr(product, "_otc_lifecycle_knocked_in", bool(state.get("knocked_in", False)))

        start_date = self.results.config.start_date
        if start_date is None and not states.empty:
            start_date = pd.Timestamp(states.index.min()).to_pydatetime()
        elapsed = 0.0
        if start_date is not None:
            elapsed = max(0.0, (date - pd.Timestamp(start_date).normalize()).days / 365.0)

        if (
            getattr(product, "exercise_date", None) is None
            and getattr(product, "maturity", None) is not None
        ):
            product.maturity = max(float(product.maturity) - elapsed, 1e-8)

        barrier_config = getattr(product, "barrier_config", None)
        if barrier_config is not None and hasattr(barrier_config, "time_shift"):
            shifted_config, dropped_all = barrier_config.time_shift(
                elapsed,
                date.to_pydatetime(),
                pricing_env,
            )
            if shifted_config is not None and not dropped_all:
                product.barrier_config = shifted_config
        return product

    def _risk_warnings(
        self,
        states: pd.DataFrame,
        greeks: pd.DataFrame,
        daily_events: pd.DataFrame,
        snapshots: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        warnings: list[dict[str, Any]] = []
        if not states.empty and "portfolio_value" in states.columns:
            values = states["portfolio_value"].astype(float)
            running_peak = values.cummax()
            drawdown = values - running_peak
            worst_drawdown = float(drawdown.min())
            warnings.append(
                {
                    "label": "Worst drawdown",
                    "value": worst_drawdown,
                    "severity": "alert" if worst_drawdown < 0 else "ok",
                    "detail": "Portfolio value drop from running peak.",
                }
            )

        if not daily_events.empty and "ki_probability_to_maturity" in daily_events.columns:
            latest_ki = float(daily_events["ki_probability_to_maturity"].iloc[-1])
            warnings.append(
                {
                    "label": "Latest KI probability",
                    "value": latest_ki,
                    "severity": "alert" if latest_ki >= 0.35 else "warn" if latest_ki >= 0.15 else "ok",
                    "detail": "Risk-neutral KI probability to maturity.",
                }
            )

        if not greeks.empty and "post_hedge_delta_cash_1pct" in greeks.columns:
            max_delta_cash = float(greeks["post_hedge_delta_cash_1pct"].abs().max())
            warnings.append(
                {
                    "label": "Peak |Delta cash 1%|",
                    "value": max_delta_cash,
                    "severity": "warn" if max_delta_cash > 0 else "ok",
                    "detail": "Maximum absolute post-hedge delta cash exposure.",
                }
            )

        if not greeks.empty and "post_hedge_gamma_cash_1pct" in greeks.columns:
            max_gamma = float(greeks["post_hedge_gamma_cash_1pct"].abs().max())
            warnings.append(
                {
                    "label": "Peak |Gamma cash 1%|",
                    "value": max_gamma,
                    "severity": "warn" if max_gamma > 0 else "ok",
                    "detail": "Maximum absolute post-hedge gamma cash exposure.",
                }
            )
        elif not greeks.empty and "gamma" in greeks.columns:
            max_gamma = float(greeks["gamma"].abs().max())
            warnings.append(
                {
                    "label": "Peak |Gamma|",
                    "value": max_gamma,
                    "severity": "warn" if max_gamma > 1.0 else "ok",
                    "detail": "Maximum absolute local gamma during replay.",
                }
            )

        if snapshots:
            stress_rows = snapshots[-1].get("stress_table", [])
            stress_pnls = [
                float(row["pnl"])
                for row in stress_rows
                if row.get("pnl") is not None and math.isfinite(float(row["pnl"]))
            ]
            if stress_pnls:
                worst_stress = min(stress_pnls)
                warnings.append(
                    {
                        "label": "Worst stress PnL",
                        "value": worst_stress,
                        "severity": "alert" if worst_stress < 0 else "ok",
                        "detail": "Worst named stress scenario from latest snapshot.",
                    }
                )
        return warnings

    def _render_html(self, payload: dict[str, Any]) -> str:
        payload_json = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace(
            "</", "<\\/"
        )
        if self.config.include_plotlyjs:
            plotly_loader = f"<script>{get_plotlyjs()}</script>"
        else:
            plotly_loader = (
                '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
            )
        template = Template(_HTML_TEMPLATE)
        return template.substitute(
            title=str(payload["metadata"]["title"]),
            payload_json=payload_json,
            plotly_loader=plotly_loader,
        )


def _row_value(row: pd.Series, key: str, default: Any) -> Any:
    if key not in row:
        return default
    value = row[key]
    if pd.isna(value):
        return default
    return value


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _cash_surface_payload(
    surface: dict[str, Any],
    *,
    key: str,
    label: str,
    z_label: str,
    gamma: bool = False,
) -> dict[str, Any] | None:
    spots = surface.get("y")
    values = surface.get("z")
    if not isinstance(spots, list) or not isinstance(values, list):
        return None
    z_cash: list[list[float | None]] = []
    for spot, row in zip(spots, values):
        spot_value = _finite_float(spot)
        if spot_value is None or not isinstance(row, list):
            z_cash.append([])
            continue
        one_percent_move = spot_value * 0.01
        cash_row: list[float | None] = []
        for raw in row:
            raw_value = _finite_float(raw)
            if raw_value is None:
                cash_row.append(None)
            elif gamma:
                cash_row.append(raw_value * spot_value**2 / 100.0)
            else:
                cash_row.append(raw_value * one_percent_move)
        z_cash.append(cash_row)
    cash_surface = dict(surface)
    cash_surface.update({"key": key, "label": label, "z": z_cash, "z_label": z_label})
    return cash_surface


def _frame_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return df.reset_index(drop=False).to_dict(orient="records")


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int | str | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "name") and not isinstance(value, str):
        return str(value.name)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_to_jsonable(v) for v in value]
    return str(value)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>$title</title>
  <style>
    :root {
      --bg: #f7f5f0;
      --surface: #ffffff;
      --surface-2: #f0f5f2;
      --ink: #1b2420;
      --muted: #667069;
      --line: #d8ddd6;
      --accent: #0f7c72;
      --accent-2: #a46318;
      --danger: #b33a3a;
      --warn: #b7781f;
      --ok: #2f7b4f;
      --shadow: 0 18px 42px rgba(37, 43, 38, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      padding: 24px 28px 16px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #ffffff 0%, #f7f5f0 100%);
    }
    h1 {
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.2;
      font-weight: 720;
    }
    h2 {
      margin: 0 0 14px;
      font-size: 19px;
      line-height: 1.25;
      font-weight: 700;
    }
    h3 {
      margin: 0 0 10px;
      font-size: 14px;
      line-height: 1.25;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
    }
    .subhead {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
    }
    .role-tabs {
      display: flex;
      gap: 8px;
      padding: 14px 28px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.78);
      position: sticky;
      top: 0;
      z-index: 5;
      backdrop-filter: blur(8px);
    }
    .role-tabs button {
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--ink);
      border-radius: 6px;
      padding: 9px 12px;
      font-size: 13px;
      font-weight: 650;
      cursor: pointer;
    }
    .role-tabs button.active {
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
    }
    main { padding: 22px 28px 36px; }
    .role-section { display: none; }
    .role-section.active { display: block; }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 14px;
      align-items: start;
    }
    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
      min-width: 0;
    }
    .span-3 { grid-column: span 3; }
    .span-4 { grid-column: span 4; }
    .span-5 { grid-column: span 5; }
    .span-6 { grid-column: span 6; }
    .span-7 { grid-column: span 7; }
    .span-8 { grid-column: span 8; }
    .span-12 { grid-column: span 12; }
    .kpi-value {
      font-size: 24px;
      line-height: 1.1;
      font-weight: 760;
      margin-bottom: 6px;
      overflow-wrap: anywhere;
    }
    .kpi-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    .chart {
      width: 100%;
      min-height: 360px;
    }
    .chart.short { min-height: 270px; }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
    }
    label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    select {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 8px 10px;
      font-size: 13px;
      min-width: 190px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      padding: 8px 9px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-weight: 700;
      background: var(--surface-2);
    }
    .warning-row {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr 2fr;
      gap: 10px;
      align-items: center;
      padding: 10px 0;
      border-bottom: 1px solid var(--line);
      font-size: 13px;
    }
    .severity {
      width: max-content;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 750;
      text-transform: uppercase;
    }
    .severity.ok { color: var(--ok); background: rgba(47, 123, 79, 0.11); }
    .severity.warn { color: var(--warn); background: rgba(183, 120, 31, 0.13); }
    .severity.alert { color: var(--danger); background: rgba(179, 58, 58, 0.12); }
    .empty {
      color: var(--muted);
      padding: 18px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: var(--surface-2);
      font-size: 13px;
    }
    @media (max-width: 980px) {
      header, .role-tabs, main { padding-left: 16px; padding-right: 16px; }
      .span-3, .span-4, .span-5, .span-6, .span-7, .span-8, .span-12 {
        grid-column: span 12;
      }
      .role-tabs { overflow-x: auto; }
      .role-tabs button { white-space: nowrap; }
      .warning-row { grid-template-columns: 1fr; }
    }
  </style>
  $plotly_loader
</head>
<body>
  <header>
    <h1>$title</h1>
    <div class="subhead">
      <span id="underlyingLabel"></span>
      <span id="dateRangeLabel"></span>
      <span id="surfaceModeLabel"></span>
    </div>
  </header>
  <nav class="role-tabs" aria-label="Dashboard role selector">
    <button id="role-executive" data-role="executive">Executive Report</button>
    <button id="role-trader" data-role="trader">Trader Workstation</button>
    <button id="role-risk-manager" data-role="risk_manager">Risk Manager</button>
    <button id="role-quant" data-role="quant">Quant Explorer</button>
  </nav>
  <main>
    <section id="section-executive" class="role-section" data-role-section="executive">
      <div class="grid">
        <div class="panel span-3"><div class="kpi-value" id="kpi-final-pnl"></div><div class="kpi-label">Final PnL</div></div>
        <div class="panel span-3"><div class="kpi-value" id="kpi-final-value"></div><div class="kpi-label">Final portfolio value</div></div>
        <div class="panel span-3"><div class="kpi-value" id="kpi-trades"></div><div class="kpi-label">Trades</div></div>
        <div class="panel span-3"><div class="kpi-value" id="kpi-actions"></div><div class="kpi-label">Lifecycle actions</div></div>
        <div class="panel span-8"><h2>Portfolio PnL and Components</h2><div id="executive-pnl-chart" class="chart"></div></div>
        <div class="panel span-4"><h2>Major Events</h2><div id="executive-actions-table"></div></div>
        <div class="panel span-12"><h2>Lifecycle Probability Watch</h2><div id="executive-event-chart" class="chart short"></div></div>
      </div>
    </section>
    <section id="section-trader" class="role-section" data-role-section="trader">
      <div class="grid">
        <div class="panel span-3"><div class="kpi-value" id="trader-delta-cash-before"></div><div class="kpi-label">Delta cash 1% before hedge</div></div>
        <div class="panel span-3"><div class="kpi-value" id="trader-delta-cash-after"></div><div class="kpi-label">Delta cash 1% after hedge</div></div>
        <div class="panel span-3"><div class="kpi-value" id="trader-gamma-cash-before"></div><div class="kpi-label">Gamma cash 1% before hedge</div></div>
        <div class="panel span-3"><div class="kpi-value" id="trader-gamma-cash-after"></div><div class="kpi-label">Gamma cash 1% after hedge</div></div>
        <div class="panel span-7"><h2>Hedge Tracking</h2><div id="trader-hedge-chart" class="chart"></div></div>
        <div class="panel span-5"><h2>Trades and Rolls</h2><div id="trader-trades-table"></div></div>
        <div class="panel span-6"><h2>Delta Cash Before / After Hedging</h2><div id="trader-delta-cash-chart" class="chart"></div></div>
        <div class="panel span-6"><h2>Gamma Cash Before / After Hedging</h2><div id="trader-gamma-cash-chart" class="chart"></div></div>
      </div>
    </section>
    <section id="section-risk_manager" class="role-section" data-role-section="risk_manager">
      <div class="grid">
        <div class="panel span-5"><h2>Limit-Style Warning Bands</h2><div id="risk-manager-warnings"></div></div>
        <div class="panel span-7"><h2>Drawdown</h2><div id="risk-drawdown-chart" class="chart short"></div></div>
        <div class="panel span-6"><h2>Stress Losses</h2><div id="risk-stress-table"></div></div>
        <div class="panel span-6"><h2>Scenario Ladder</h2><div id="risk-scenario-ladder-chart" class="chart short"></div></div>
        <div class="panel span-12"><h2>KI / KO Risk</h2><div id="risk-event-probability-chart" class="chart short"></div></div>
      </div>
    </section>
    <section id="section-quant" class="role-section" data-role-section="quant">
      <div class="grid">
        <div class="panel span-12">
          <h2>Full Risk-Suite Snapshot Surfaces</h2>
          <div class="toolbar">
            <label for="snapshotSelect">Snapshot</label>
            <select id="snapshotSelect"></select>
            <label for="surfaceMetric">Metric</label>
            <select id="surfaceMetric"></select>
          </div>
          <div id="quant-snapshot-surface-chart" class="chart"></div>
        </div>
        <div class="panel span-7">
          <h2>Daily Backtest Spot x Implied-q Surface</h2>
          <div class="toolbar">
            <label for="dailySurfaceDate">Date</label>
            <select id="dailySurfaceDate"></select>
            <label for="dailySurfaceMetric">Metric</label>
            <select id="dailySurfaceMetric"></select>
          </div>
          <div id="quant-daily-surface-chart" class="chart"></div>
        </div>
        <div class="panel span-5"><h2>Snapshot Tables</h2><div id="quant-snapshot-table"></div></div>
      </div>
    </section>
  </main>
  <script>
    const DATA = $payload_json;
    const COLORS = {
      ink: '#1b2420',
      muted: '#667069',
      accent: '#0f7c72',
      accent2: '#a46318',
      danger: '#b33a3a',
      warn: '#b7781f',
      ok: '#2f7b4f',
      line: '#d8ddd6'
    };

    const states = DATA.tables.states || [];
    const greeks = DATA.tables.greeks || [];
    const rebalances = DATA.tables.rebalances || [];
    const trades = DATA.tables.trades || [];
    const actions = DATA.tables.actions || [];
    const dailyEvents = DATA.tables.daily_events || [];
    const surfaces = DATA.tables.surfaces || [];
    let activeRole = 'executive';

    function num(v) {
      if (v === null || v === undefined || v === '') return null;
      const n = Number(v);
      return Number.isFinite(n) ? n : null;
    }
    function fmtNumber(v, digits = 2) {
      const n = num(v);
      if (n === null) return 'n/a';
      return n.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
    }
    function fmtPct(v, digits = 2) {
      const n = num(v);
      if (n === null) return 'n/a';
      return (n * 100).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits }) + '%';
    }
    function dateKey(row) { return row.date || row.index || row.valuation_date; }
    function metricLabel(metric) {
      const labels = {
        delta_cash_1pct: 'Delta cash 1%',
        gamma_cash_1pct: 'Gamma cash 1%',
        price: 'Price',
        delta: 'Raw delta',
        gamma: 'Raw gamma'
      };
      return labels[metric] || metric;
    }
    function tableHtml(rows, columns, limit = 10) {
      if (!rows || rows.length === 0) return '<div class="empty">No records available.</div>';
      const visible = rows.slice(0, limit);
      const head = '<tr>' + columns.map(c => '<th>' + c.label + '</th>').join('') + '</tr>';
      const body = visible.map(row => '<tr>' + columns.map(c => '<td>' + (c.format ? c.format(row[c.key], row) : (row[c.key] ?? '')) + '</td>').join('') + '</tr>').join('');
      return '<table>' + head + body + '</table>';
    }
    function baseLayout(title, yTitle) {
      return {
        title: { text: title, font: { size: 14 } },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: '#ffffff',
        margin: { l: 58, r: 22, t: 42, b: 46 },
        hovermode: 'x unified',
        xaxis: { gridcolor: '#edf0ec', zerolinecolor: COLORS.line },
        yaxis: { title: yTitle, gridcolor: '#edf0ec', zerolinecolor: COLORS.line },
        legend: { orientation: 'h', y: -0.18 },
        font: { family: 'Inter, system-ui, sans-serif', color: COLORS.ink }
      };
    }
    function setHeader() {
      const summary = DATA.summary || {};
      document.getElementById('underlyingLabel').textContent = 'Underlying: ' + (DATA.metadata.underlying || 'n/a');
      document.getElementById('dateRangeLabel').textContent = 'Range: ' + (summary.start_date || 'n/a') + ' → ' + (summary.end_date || 'n/a');
      document.getElementById('surfaceModeLabel').textContent = 'Snapshot surfaces: ' + (DATA.metadata.surface_mode || 'finite-difference');
      document.getElementById('kpi-final-pnl').textContent = fmtNumber(summary.total_pnl, 2);
      document.getElementById('kpi-final-value').textContent = fmtNumber(summary.final_portfolio_value, 2);
      document.getElementById('kpi-trades').textContent = String(summary.num_trades ?? trades.length);
      document.getElementById('kpi-actions').textContent = String(summary.num_actions ?? actions.length);
    }
    function renderExecutive() {
      Plotly.react('executive-pnl-chart', [
        { x: states.map(dateKey), y: states.map(r => num(r.total_pnl)), type: 'scatter', mode: 'lines', name: 'Total PnL', line: { color: COLORS.accent, width: 2 } },
        { x: states.map(dateKey), y: states.map(r => num(r.product_pnl)), type: 'scatter', mode: 'lines', name: 'Product PnL', line: { color: COLORS.accent2, width: 2 } },
        { x: states.map(dateKey), y: states.map(r => num(r.hedge_pnl)), type: 'scatter', mode: 'lines', name: 'Hedge PnL', line: { color: COLORS.muted, width: 2 } }
      ], baseLayout('PnL replay', 'PnL'), { responsive: true });
      document.getElementById('executive-actions-table').innerHTML = tableHtml(actions, [
        { key: 'date', label: 'Date' },
        { key: 'action_type', label: 'Action' },
        { key: 'spot', label: 'Spot', format: v => fmtNumber(v, 2) },
        { key: 'barrier', label: 'Barrier', format: v => fmtNumber(v, 2) },
        { key: 'cashflow', label: 'Cashflow', format: v => fmtNumber(v, 2) }
      ]);
      Plotly.react('executive-event-chart', [
        { x: dailyEvents.map(dateKey), y: dailyEvents.map(r => num(r.next_ko_probability)), type: 'scatter', mode: 'lines+markers', name: 'Next KO probability', line: { color: COLORS.ok } },
        { x: dailyEvents.map(dateKey), y: dailyEvents.map(r => num(r.ki_probability_to_maturity)), type: 'scatter', mode: 'lines+markers', name: 'KI to maturity', line: { color: COLORS.danger } },
        { x: dailyEvents.map(dateKey), y: dailyEvents.map(r => num(r.survival_probability)), type: 'scatter', mode: 'lines+markers', name: 'Survival', line: { color: COLORS.accent } }
      ], baseLayout('Lifecycle probabilities', 'Probability'), { responsive: true });
    }
    function renderTrader() {
      const latestGreek = greeks.length ? greeks[greeks.length - 1] : {};
      document.getElementById('trader-delta-cash-before').textContent = fmtNumber(latestGreek.pre_hedge_delta_cash_1pct, 2);
      document.getElementById('trader-delta-cash-after').textContent = fmtNumber(latestGreek.post_hedge_delta_cash_1pct ?? latestGreek.delta_cash_1pct, 2);
      document.getElementById('trader-gamma-cash-before').textContent = fmtNumber(latestGreek.pre_hedge_gamma_cash_1pct, 2);
      document.getElementById('trader-gamma-cash-after').textContent = fmtNumber(latestGreek.post_hedge_gamma_cash_1pct ?? latestGreek.gamma_cash_1pct, 2);
      Plotly.react('trader-hedge-chart', [
        { x: rebalances.map(dateKey), y: rebalances.map(r => num(r.current_contracts)), type: 'scatter', mode: 'lines+markers', name: 'Current contracts', line: { color: COLORS.accent } },
        { x: rebalances.map(dateKey), y: rebalances.map(r => num(r.target_contracts)), type: 'scatter', mode: 'lines', name: 'Target contracts', line: { color: COLORS.accent2, dash: 'dash' } },
        { x: states.map(dateKey), y: states.map(r => num(r.futures_price)), type: 'scatter', mode: 'lines', name: 'Futures price', yaxis: 'y2', line: { color: COLORS.muted } }
      ], {
        ...baseLayout('Contracts and futures price', 'Contracts'),
        yaxis2: { title: 'Futures', overlaying: 'y', side: 'right', gridcolor: 'rgba(0,0,0,0)' }
      }, { responsive: true });
      document.getElementById('trader-trades-table').innerHTML = tableHtml(trades, [
        { key: 'date', label: 'Date' },
        { key: 'trade_type', label: 'Type' },
        { key: 'contract', label: 'Contract' },
        { key: 'quantity', label: 'Qty', format: v => fmtNumber(v, 2) },
        { key: 'transaction_cost', label: 'Cost', format: v => fmtNumber(v, 2) }
      ], 14);
      const traceCashSeries = (series, colors) => series.filter(([name]) => greeks.some(r => num(r[name]) !== null)).map(([name, label], i) => ({
        x: greeks.map(dateKey),
        y: greeks.map(r => num(r[name])),
        type: 'scatter',
        mode: 'lines',
        name: label,
        line: { width: 2, color: colors[i % colors.length] }
      }));
      const deltaCashSeries = [
        ['pre_hedge_delta_cash_1pct', 'Delta cash 1% before hedge'],
        ['post_hedge_delta_cash_1pct', 'Delta cash 1% after hedge']
      ];
      const gammaCashSeries = [
        ['pre_hedge_gamma_cash_1pct', 'Gamma cash 1% before hedge'],
        ['post_hedge_gamma_cash_1pct', 'Gamma cash 1% after hedge']
      ];
      Plotly.react('trader-delta-cash-chart', traceCashSeries(deltaCashSeries, [COLORS.accent2, COLORS.accent]), baseLayout('Delta cash over time', 'Delta cash 1%'), { responsive: true });
      Plotly.react('trader-gamma-cash-chart', traceCashSeries(gammaCashSeries, [COLORS.warn, COLORS.danger]), baseLayout('Gamma cash over time', 'Gamma cash 1%'), { responsive: true });
    }
    function renderRiskManager() {
      const warningsHtml = (DATA.warnings || []).map(w => (
        '<div class="warning-row"><strong>' + w.label + '</strong><span class="severity ' + w.severity + '">' + w.severity + '</span><span>' + fmtNumber(w.value, 4) + ' · ' + w.detail + '</span></div>'
      )).join('') || '<div class="empty">No risk warnings were generated.</div>';
      document.getElementById('risk-manager-warnings').innerHTML = warningsHtml;
      let peak = -Infinity;
      const dd = states.map(r => {
        const v = num(r.portfolio_value);
        if (v === null) return null;
        peak = Math.max(peak, v);
        return v - peak;
      });
      Plotly.react('risk-drawdown-chart', [
        { x: states.map(dateKey), y: dd, type: 'scatter', mode: 'lines', fill: 'tozeroy', name: 'Drawdown', line: { color: COLORS.danger } }
      ], baseLayout('Portfolio drawdown', 'Drawdown'), { responsive: true });
      const snap = latestSnapshot();
      document.getElementById('risk-stress-table').innerHTML = snap ? tableHtml(snap.stress_table || [], [
        { key: 'scenario', label: 'Scenario' },
        { key: 'spot_shock', label: 'Spot', format: v => fmtPct(v, 1) },
        { key: 'vol_shock', label: 'Vol', format: v => fmtPct(v, 1) },
        { key: 'q_shift', label: 'q shift', format: v => fmtPct(v, 2) },
        { key: 'pnl', label: 'PnL', format: v => fmtNumber(v, 4) }
      ]) : '<div class="empty">No full risk snapshot available.</div>';
      if (snap && snap.scenario_ladder) {
        Plotly.react('risk-scenario-ladder-chart', [{
          x: snap.scenario_ladder.columns,
          y: snap.scenario_ladder.rows,
          z: snap.scenario_ladder.z,
          type: 'heatmap',
          colorscale: 'RdBu',
          reversescale: true,
          colorbar: { title: 'PnL' }
        }], baseLayout('Spot x vol scenario ladder', 'Spot shock'), { responsive: true });
      }
      Plotly.react('risk-event-probability-chart', [
        { x: dailyEvents.map(dateKey), y: dailyEvents.map(r => num(r.total_remaining_ko_probability)), type: 'scatter', mode: 'lines', name: 'Remaining KO probability', line: { color: COLORS.ok } },
        { x: dailyEvents.map(dateKey), y: dailyEvents.map(r => num(r.ki_probability_to_maturity)), type: 'scatter', mode: 'lines', name: 'KI probability', line: { color: COLORS.danger } }
      ], baseLayout('KI / KO risk trend', 'Probability'), { responsive: true });
    }
    function latestSnapshot() {
      const snaps = DATA.risk_snapshots || [];
      return snaps.length ? snaps[snaps.length - 1] : null;
    }
    function setupQuantControls() {
      const snaps = DATA.risk_snapshots || [];
      const snapshotSelect = document.getElementById('snapshotSelect');
      const surfaceMetric = document.getElementById('surfaceMetric');
      snapshotSelect.innerHTML = snaps.map((s, i) => '<option value="' + i + '">' + s.label + '</option>').join('');
      if (snaps.length) {
        surfaceMetric.innerHTML = (snaps[0].surfaces || []).concat(snaps[0].barrier_surfaces || []).map(s => '<option value="' + s.key + '">' + s.label + '</option>').join('');
      }
      snapshotSelect.onchange = renderQuantSnapshot;
      surfaceMetric.onchange = renderQuantSnapshot;

      const dates = Array.from(new Set(surfaces.map(r => String((r.date || '').slice(0, 10))))).filter(Boolean);
      const metrics = ['delta_cash_1pct', 'gamma_cash_1pct', 'price', 'delta', 'gamma'].filter(m => surfaces.some(r => num(r[m]) !== null));
      document.getElementById('dailySurfaceDate').innerHTML = dates.map(d => '<option value="' + d + '">' + d + '</option>').join('');
      document.getElementById('dailySurfaceMetric').innerHTML = metrics.map(m => '<option value="' + m + '">' + metricLabel(m) + '</option>').join('');
      document.getElementById('dailySurfaceDate').onchange = renderDailySurface;
      document.getElementById('dailySurfaceMetric').onchange = renderDailySurface;
    }
    function renderQuantSnapshot() {
      const snaps = DATA.risk_snapshots || [];
      if (!snaps.length) {
        document.getElementById('quant-snapshot-surface-chart').innerHTML = '<div class="empty">No full risk snapshots were generated.</div>';
        document.getElementById('quant-snapshot-table').innerHTML = '<div class="empty">No snapshot tables available.</div>';
        return;
      }
      const snap = snaps[Number(document.getElementById('snapshotSelect').value || 0)];
      const allSurfaces = (snap.surfaces || []).concat(snap.barrier_surfaces || []);
      const selected = allSurfaces.find(s => s.key === document.getElementById('surfaceMetric').value) || allSurfaces[0];
      Plotly.react('quant-snapshot-surface-chart', [{
        x: selected.x,
        y: selected.y,
        z: selected.z,
        type: 'heatmap',
        colorscale: 'Viridis',
        colorbar: { title: selected.z_label }
      }], baseLayout(selected.label, selected.y_label), { responsive: true });
      document.getElementById('quant-snapshot-table').innerHTML =
        '<h3>Base Metrics</h3>' + tableHtml(Object.entries(snap.base || {}).map(([k, v]) => ({ metric: k, value: v })), [
          { key: 'metric', label: 'Metric' },
          { key: 'value', label: 'Value', format: v => fmtNumber(v, 6) }
        ], 20) +
        '<h3>Bucketed Greeks</h3>' + tableHtml(snap.bucketed_greeks || [], [
          { key: 'bucket', label: 'Bucket' },
          { key: 'bucket_vega', label: 'Vega', format: v => fmtNumber(v, 6) },
          { key: 'bucket_rho_q', label: 'RhoQ', format: v => fmtNumber(v, 6) },
          { key: 'bucket_rho_b', label: 'RhoB', format: v => fmtNumber(v, 6) }
        ], 10);
    }
    function renderDailySurface() {
      if (!surfaces.length) {
        document.getElementById('quant-daily-surface-chart').innerHTML = '<div class="empty">No daily backtest surfaces available.</div>';
        return;
      }
      const d = document.getElementById('dailySurfaceDate').value;
      const metric = document.getElementById('dailySurfaceMetric').value;
      const rows = surfaces.filter(r => String((r.date || '').slice(0, 10)) === d);
      const spots = Array.from(new Set(rows.map(r => num(r.spot_node)).filter(v => v !== null))).sort((a,b) => a-b);
      const qs = Array.from(new Set(rows.map(r => num(r.q_node)).filter(v => v !== null))).sort((a,b) => a-b);
      const z = spots.map(s => qs.map(q => {
        const row = rows.find(r => num(r.spot_node) === s && num(r.q_node) === q);
        return row ? num(row[metric]) : null;
      }));
      Plotly.react('quant-daily-surface-chart', [{
        x: qs,
        y: spots,
          z,
          type: 'heatmap',
          colorscale: 'Viridis',
          colorbar: { title: metricLabel(metric) }
      }], baseLayout('Daily ' + metricLabel(metric) + ' surface · ' + d, 'Spot'), { responsive: true });
    }
    function switchRole(role) {
      activeRole = role;
      document.querySelectorAll('.role-tabs button').forEach(btn => btn.classList.toggle('active', btn.dataset.role === role));
      document.querySelectorAll('.role-section').forEach(sec => sec.classList.toggle('active', sec.dataset.roleSection === role));
      if (role === 'executive') renderExecutive();
      if (role === 'trader') renderTrader();
      if (role === 'risk_manager') renderRiskManager();
      if (role === 'quant') { setupQuantControls(); renderQuantSnapshot(); renderDailySurface(); }
    }
    document.querySelectorAll('.role-tabs button').forEach(btn => btn.addEventListener('click', () => switchRole(btn.dataset.role)));
    setHeader();
    switchRole('executive');
  </script>
</body>
</html>
"""
