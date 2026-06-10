from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantark.asset.equity.param import MCParams, QuadParams
from quantark.asset.equity.product.option import create_standard_snowball
from quantark.backtest.otc import (
    AutocallableBacktestConfig,
    AutocallableBacktestDashboard,
    AutocallableBacktestEngine,
    AutocallableDashboardConfig,
    AutocallableDeltaHedgeStrategy,
    AutocallableEngineConfig,
    AutocallableMarketDataSet,
    SurfaceGridConfig,
)
from quantark.backtest.otc.results import AutocallableBacktestResults
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import EngineType


def _snowball_product():
    return create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=6.0 / 365.0,
        contract_multiplier=100.0,
        ko_barrier=103.0,
        ki_barrier=97.0,
        ko_rate=0.02,
        num_observations=2,
        ko_observation_dates=[2.0 / 365.0, 5.0 / 365.0],
        ki_observation_type=ObservationType.DISCRETE,
        ki_continuous=False,
        ki_observation_dates=[1.0 / 365.0, 3.0 / 365.0],
        include_principal=True,
    )


def _market_data() -> AutocallableMarketDataSet:
    dates = pd.date_range("2024-01-02", periods=5, freq="D")
    spot_data = pd.DataFrame(
        {"date": dates, "spot": [100.0, 96.0, 104.0, 105.0, 106.0]}
    )
    vol_data = pd.DataFrame({"date": dates, "volatility": [0.22] * len(dates)})
    rate_data = pd.DataFrame({"date": dates, "rate": [0.02] * len(dates)})
    futures_rows = []
    for date, spot in zip(dates, spot_data["spot"]):
        futures_rows.extend(
            [
                {
                    "date": date,
                    "contract": "IF2401",
                    "futures_price": spot * 1.004,
                    "expiry_date": pd.Timestamp("2024-01-07"),
                    "multiplier": 300.0,
                },
                {
                    "date": date,
                    "contract": "IF2402",
                    "futures_price": spot * 1.01,
                    "expiry_date": pd.Timestamp("2024-02-16"),
                    "multiplier": 300.0,
                },
            ]
        )
    return AutocallableMarketDataSet.from_dataframes(
        spot_data=spot_data,
        vol_data=vol_data,
        rate_data=rate_data,
        futures_data=pd.DataFrame(futures_rows),
    )


def _backtest_config() -> AutocallableBacktestConfig:
    return AutocallableBacktestConfig(
        product=_snowball_product(),
        market_data=_market_data(),
        engine_config=AutocallableEngineConfig(
            pricing_engine_type=EngineType.MONTE_CARLO,
            mc_params=MCParams(num_paths=64, time_steps=8, seed=7),
            quad_params=QuadParams(grid_points=101, num_std_devs=4.0),
        ),
        strategy=AutocallableDeltaHedgeStrategy(
            delta_threshold=0.0,
            round_contracts=False,
        ),
        surface_config=SurfaceGridConfig(spot_nodes=3, q_nodes=3),
        calculate_surfaces=True,
        calculate_event_probabilities=True,
        product_quantity=-1.0,
        underlying="CSI500",
    )


def test_dashboard_html_contains_role_controls_and_surfaces(tmp_path: Path):
    results = AutocallableBacktestEngine(_backtest_config()).run()
    html_path = AutocallableBacktestDashboard(
        results,
        AutocallableDashboardConfig(full_surface_snapshots=False),
    ).write_html(tmp_path / "dashboard.html")

    content = html_path.read_text(encoding="utf-8")

    assert "Executive Report" in content
    assert "Trader Workstation" in content
    assert "Risk Manager" in content
    assert "Quant Explorer" in content
    assert 'id="role-risk-manager"' in content
    assert 'id="risk-manager-warnings"' in content
    assert 'id="surfaceMetric"' in content
    assert 'id="quant-daily-surface-chart"' in content
    assert "Delta Cash Before / After Hedging" in content
    assert "Gamma Cash Before / After Hedging" in content
    assert 'id="trader-delta-cash-chart"' in content
    assert 'id="trader-gamma-cash-chart"' in content
    assert "Delta cash 1% before hedge" in content
    assert "Gamma cash 1% after hedge" in content
    assert "post_hedge_delta_cash_1pct" in content
    assert "delta_cash_1pct" in content
    assert "gamma_cash_1pct" in content
    assert "Lifecycle Probability Watch" in content


def test_dashboard_handles_empty_result_tables(tmp_path: Path):
    config = _backtest_config()
    results = AutocallableBacktestResults(
        config=config,
        states=[],
        greeks=[],
        rebalances=[],
        trades=[],
        actions=[],
        surfaces=[],
        daily_event_summary=[],
        event_probabilities=[],
    )

    html_path = AutocallableBacktestDashboard(
        results,
        AutocallableDashboardConfig(full_surface_snapshots=False),
    ).write_html(tmp_path / "empty_dashboard.html")

    content = html_path.read_text(encoding="utf-8")
    assert "No records available." in content
    assert "No daily backtest surfaces available." in content
