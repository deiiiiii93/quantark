from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from example import otc_autocallable_backtest_demo as demo
from quantark.util.enum.engine_enums import EngineType


class FakeAKShare:
    def __init__(self) -> None:
        self.futures_calls: list[str] = []

    def index_zh_a_hist(
        self,
        symbol: str,
        period: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        assert symbol == "000905"
        assert period == "daily"
        return pd.DataFrame(
            {
                "日期": pd.date_range("2024-01-02", periods=5, freq="D"),
                "收盘": [5400.0, 5412.0, 5380.0, 5425.0, 5440.0],
            }
        )

    def futures_zh_daily_sina(self, symbol: str) -> pd.DataFrame:
        self.futures_calls.append(symbol)
        base = {
            "IC2401": 5396.0,
            "IC2402": 5412.0,
            "IC2403": 5424.0,
        }.get(symbol, 5430.0)
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-02", periods=5, freq="D"),
                "close": [base, base + 6.0, base - 3.0, base + 8.0, base + 11.0],
            }
        )


def test_normalize_csi500_spot_accepts_akshare_chinese_columns():
    raw = pd.DataFrame(
        {
            "日期": ["2024-01-03", "2024-01-02"],
            "收盘": [5412.0, 5400.0],
        }
    )

    spot = demo.normalize_csi500_spot(raw)

    assert list(spot.columns) == ["date", "spot"]
    assert list(spot["date"]) == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert list(spot["spot"]) == [5400.0, 5412.0]


def test_ic_futures_normalization_and_expiry_inference():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "close": [5396.0, 5402.0],
        }
    )

    futures = demo.normalize_ic_futures(raw, "IC2401")

    assert list(futures["contract"].unique()) == ["IC2401"]
    assert futures["expiry_date"].iloc[0] == pd.Timestamp("2024-01-19")
    assert futures["multiplier"].iloc[0] == 200.0
    assert list(futures["futures_price"]) == [5396.0, 5402.0]


def test_cache_read_write_round_trip(tmp_path: Path):
    frames = demo.CachedMarketFrames(
        spot_data=pd.DataFrame(
            {"date": [pd.Timestamp("2024-01-02")], "spot": [5400.0]}
        ),
        futures_data=pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02")],
                "contract": ["IC2401"],
                "futures_price": [5396.0],
                "expiry_date": [pd.Timestamp("2024-01-19")],
                "multiplier": [200.0],
            }
        ),
    )

    demo.write_cache(tmp_path, frames)
    cached = demo.read_cache(tmp_path)

    assert cached.spot_data["spot"].iloc[0] == 5400.0
    assert cached.futures_data["contract"].iloc[0] == "IC2401"
    assert cached.futures_data["expiry_date"].iloc[0] == pd.Timestamp("2024-01-19")


def test_cache_only_missing_cache_raises_clear_error(tmp_path: Path):
    with pytest.raises(demo.DemoDataError, match="Cache is missing"):
        demo.load_market_cache_or_fetch(
            start_date=pd.Timestamp("2024-01-02"),
            end_date=pd.Timestamp("2024-01-05"),
            cache_dir=tmp_path / "missing",
            refresh_data=False,
            cache_only=True,
        )


def test_cli_engine_config_selection():
    quad_args = demo.parse_args(["--engine", "quad", "--quad-grid", "101"])
    pde_args = demo.parse_args(["--engine", "pde", "--pde-grid", "80"])
    mc_args = demo.parse_args(
        ["--engine", "mc", "--mc-paths", "64", "--mc-steps", "8"]
    )

    quad_config = demo.create_engine_config(quad_args)
    pde_config = demo.create_engine_config(pde_args)
    mc_config = demo.create_engine_config(mc_args)

    assert quad_config.pricing_engine_type == EngineType.QUADRATURE
    assert isinstance(quad_config.quad_params, QuadParams)
    assert pde_config.pricing_engine_type == EngineType.PDE
    assert isinstance(pde_config.pde_params, PDEParams)
    assert mc_config.pricing_engine_type == EngineType.MONTE_CARLO
    assert isinstance(mc_config.mc_params, MCParams)


def test_demo_smoke_run_creates_expected_outputs(monkeypatch, tmp_path: Path):
    fake_ak = FakeAKShare()
    monkeypatch.setattr(demo, "load_akshare", lambda: fake_ak)

    args = demo.parse_args(
        [
            "--engine",
            "quad",
            "--start-date",
            "2024-01-02",
            "--end-date",
            "2024-01-05",
            "--refresh-data",
            "--realized-vol-window",
            "3",
            "--quad-grid",
            "101",
            "--output-dir",
            str(tmp_path),
        ]
    )

    results, summary = demo.run_demo(args)

    expected = {
        "states.csv",
        "greeks.csv",
        "rebalance.csv",
        "trades.csv",
        "actions.csv",
        "daily_event_summary.csv",
        "event_probability.csv",
        "surfaces.parquet",
        "results.xlsx",
        "summary.json",
        "dashboard.html",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    assert (tmp_path / "cache" / "csi500_spot.csv").exists()
    assert (tmp_path / "cache" / "ic_futures.csv").exists()
    assert len(results.states_df) == 4
    assert not results.greeks_df.empty
    assert not results.daily_event_summary_df.empty
    assert summary["engine"] == "quad"
    assert summary["dashboard_html"] == str(tmp_path / "dashboard.html")
    assert "IC2401" in fake_ak.futures_calls

    workbook = pd.ExcelFile(tmp_path / "results.xlsx")
    expected_sheets = {
        "OTC_Contract_Terms",
        "OTC_Observation_Schedule",
        "Trade_Position",
        "Pricing_Model_Config",
        "Market_Data_Assumptions",
        "Hedge_Futures_Config",
        "Listed_Futures",
        "Active_Futures",
    }
    assert expected_sheets.issubset(set(workbook.sheet_names))
    greeks_sheet = pd.read_excel(workbook, sheet_name="Greeks")
    assert {
        "pre_hedge_delta_cash_1pct",
        "post_hedge_delta_cash_1pct",
        "pre_hedge_gamma_cash_1pct",
        "post_hedge_gamma_cash_1pct",
    }.issubset(greeks_sheet.columns)
    surfaces_sheet = pd.read_excel(workbook, sheet_name="Surfaces")
    assert {"delta_cash_1pct", "gamma_cash_1pct"}.issubset(
        surfaces_sheet.columns
    )

    terms = pd.read_excel(workbook, sheet_name="OTC_Contract_Terms").set_index("term")
    assert terms.loc["product_type", "value"] == "SnowballOption"
    assert "roll_days_before_expiry" not in set(terms.index)

    trade_position = pd.read_excel(workbook, sheet_name="Trade_Position").set_index("term")
    assert trade_position.loc["desk_perspective", "value"] == "short_note"

    market_data = pd.read_excel(
        workbook, sheet_name="Market_Data_Assumptions"
    ).set_index("term")
    assert market_data.loc["implied_q_formula", "value"] == "max(0, r - basis_yield)"

    hedge_config = pd.read_excel(
        workbook, sheet_name="Hedge_Futures_Config"
    ).set_index("term")
    assert hedge_config.loc["futures_roll_days_before_expiry", "value"] == 5

    listed_futures = pd.read_excel(workbook, sheet_name="Listed_Futures")
    ic2401 = listed_futures[listed_futures["contract"] == "IC2401"].iloc[0]
    assert ic2401["multiplier"] == 200.0
    assert pd.Timestamp(ic2401["expiry_date"]) == pd.Timestamp("2024-01-19")
