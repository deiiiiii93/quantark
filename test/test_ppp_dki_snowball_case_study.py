from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from asset.equity.param import QuadParams
from asset.equity.product.option import AccrualConfig, BarrierConfig, PayoffConfig, SnowballOption
from backtest.otc import (
    AutocallableBacktestConfig,
    AutocallableBacktestEngine,
    AutocallableDeltaHedgeStrategy,
    AutocallableEngineConfig,
    AutocallableMarketDataSet,
)
from example import ppp_dki_snowball_backtest_case_study as case_study
from param import FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.enum import CouponPayType, ObservationType, ProtectionType
from util.enum.engine_enums import EngineType


class RecordingEngine:
    def __init__(self, delta_before: float = 100.0, delta_after_ki: float = 200.0):
        self.delta_before = delta_before
        self.delta_after_ki = delta_after_ki
        self.price_calls: list[tuple[pd.Timestamp, bool]] = []
        self.greek_calls: list[tuple[pd.Timestamp, bool]] = []

    def price(self, product, env) -> float:
        knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
        self.price_calls.append((pd.Timestamp(env.valuation_date).normalize(), knocked_in))
        return 20.0 if knocked_in else 10.0

    def calculate_greeks(self, product, env) -> dict[str, float]:
        knocked_in = bool(getattr(product, "_otc_lifecycle_knocked_in", False))
        self.greek_calls.append((pd.Timestamp(env.valuation_date).normalize(), knocked_in))
        delta = self.delta_after_ki if knocked_in else self.delta_before
        return {"price": 20.0 if knocked_in else 10.0, "delta": delta, "gamma": 0.0}


def _market_data(spots: list[float]) -> AutocallableMarketDataSet:
    dates = pd.date_range("2024-01-02", periods=len(spots), freq="D")
    futures_rows = [
        {
            "date": date,
            "contract": "IM2402",
            "futures_price": spot,
            "expiry_date": pd.Timestamp("2024-02-16"),
            "multiplier": 200.0,
        }
        for date, spot in zip(dates, spots)
    ]
    return AutocallableMarketDataSet.from_dataframes(
        spot_data=pd.DataFrame({"date": dates, "spot": spots}),
        vol_data=pd.DataFrame({"date": dates, "volatility": [0.25] * len(spots)}),
        rate_data=pd.DataFrame({"date": dates, "rate": [0.02] * len(spots)}),
        futures_data=pd.DataFrame(futures_rows),
    )


def _product(
    *,
    ko_barrier=200.0,
    ki_barrier=50.0,
    maturity=2.0 / 365.0,
    ko_dates=None,
    ki_dates=None,
    disable_ko_after_ki=False,
) -> SnowballOption:
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        maturity=maturity,
        initial_date=datetime(2024, 1, 2),
        contract_multiplier=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=ko_barrier,
            ko_rate=0.10,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=ko_dates or [maturity],
            ki_barrier=ki_barrier,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=ki_dates or [maturity],
            ki_continuous=False,
            disable_ko_after_ki=disable_ko_after_ki,
        ),
        payoff_config=PayoffConfig(
            rebate_rate=0.10,
            include_principal=False,
            participation_rate=1.0,
            protection_type=ProtectionType.NONE,
        ),
        accrual_config=AccrualConfig(
            coupon_pay_type=CouponPayType.INSTANT,
            is_annualized=True,
        ),
    )


def _config(product: SnowballOption, market_data: AutocallableMarketDataSet, **kwargs):
    return AutocallableBacktestConfig(
        product=product,
        market_data=market_data,
        engine_config=AutocallableEngineConfig(
            pricing_engine_type=EngineType.QUADRATURE,
            quad_params=QuadParams(grid_points=101),
        ),
        strategy=AutocallableDeltaHedgeStrategy(
            delta_threshold=0.0,
            round_contracts=True,
        ),
        product_quantity=-1.0,
        underlying="CSI1000",
        initial_product_price=0.0,
        calculate_surfaces=False,
        calculate_event_probabilities=False,
        **kwargs,
    )


def _run_with_recording_engine(config: AutocallableBacktestConfig, engine: RecordingEngine):
    backtest = AutocallableBacktestEngine(config)
    backtest.pricing_engine = engine
    return backtest.run()


def test_case_study_product_construction_matches_terms():
    dates = pd.Series(pd.date_range("2024-01-02", periods=12, freq="B"))
    terms = case_study.CaseStudyTerms(notional=10_000_000.0, ki_ratio=0.75, ppp_protection_rate=0.25)
    products = case_study.build_case_study_products(
        initial_spot=100.0,
        issue_date=pd.Timestamp("2024-01-02"),
        dates=dates,
        terms=terms,
        coupons={"PPP-DKI": 0.10, "NPP-DKI": 0.20, "PPP-EKI-Parachute": 0.15},
    )

    assert products["PPP-DKI"].contract_multiplier == pytest.approx(100_000.0)
    assert products["PPP-DKI"].payoff_config.protection_type == ProtectionType.PARTIAL
    assert products["PPP-DKI"].payoff_config.protection_rate == pytest.approx(0.25)
    assert products["NPP-DKI"].payoff_config.protection_type == ProtectionType.NONE
    assert len(products["PPP-DKI"].barrier_config.ki_observation_dates) > 1
    assert products["PPP-EKI-Parachute"].barrier_config.ki_observation_dates == [
        products["PPP-EKI-Parachute"].maturity
    ]
    assert products["PPP-EKI-Parachute"].barrier_config.ko_barrier[-1] == pytest.approx(75.0)


def test_fixed_dividend_yield_overrides_pricing_q_but_keeps_implied_q():
    market_data = _market_data([100.0, 101.0])
    results = _run_with_recording_engine(
        _config(
            _product(maturity=1.0 / 365.0),
            market_data,
            fixed_dividend_yield=0.08,
        ),
        RecordingEngine(),
    )

    assert results.states_df["pricing_q"].iloc[0] == pytest.approx(0.08)
    assert results.states_df["implied_q"].iloc[0] == pytest.approx(0.02)


def test_same_day_ki_reprices_and_hedges_in_knocked_in_state():
    market_data = _market_data([100.0, 90.0, 95.0])
    product = _product(
        ko_barrier=200.0,
        ki_barrier=95.0,
        maturity=2.0 / 365.0,
        ko_dates=[2.0 / 365.0],
        ki_dates=[1.0 / 365.0],
    )
    engine = RecordingEngine(delta_before=40.0, delta_after_ki=240.0)

    results = _run_with_recording_engine(_config(product, market_data), engine)

    ki_date = pd.Timestamp("2024-01-03")
    assert ("KI" in set(results.actions_df["action_type"]))
    assert (ki_date, True) in engine.price_calls
    assert (ki_date, True) in engine.greek_calls
    assert results.greeks_df.loc[ki_date, "product_delta"] == pytest.approx(240.0)


def test_ko_terminates_product_and_closes_hedge():
    market_data = _market_data([100.0, 104.0, 105.0])
    product = _product(
        ko_barrier=103.0,
        ki_barrier=50.0,
        maturity=2.0 / 365.0,
        ko_dates=[1.0 / 365.0],
        ki_dates=[2.0 / 365.0],
    )
    results = _run_with_recording_engine(
        _config(product, market_data),
        RecordingEngine(delta_before=400.0, delta_after_ki=400.0),
    )
    inception_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.25),
        rate_curve=FlatRateCurve(rate=0.02),
        valuation_date=datetime(2024, 1, 2),
    )
    expected_ko_payoff = product.resolve_ko_observations(inception_env)[0].payoff

    assert list(results.actions_df["action_type"]) == ["KO"]
    assert results.actions_df["payoff"].iloc[0] == pytest.approx(expected_ko_payoff)
    assert results.actions_df["cashflow"].iloc[0] == pytest.approx(-expected_ko_payoff)
    assert "hedge_close" in set(results.trades_df["trade_type"])
    assert bool(results.states_df["alive"].iloc[-1]) is False
    assert results.states_df["futures_contracts"].iloc[-1] == pytest.approx(0.0)


def test_post_ki_ko_remains_possible_for_parachute_rescue_logic():
    market_data = _market_data([100.0, 90.0, 103.0])
    product = _product(
        ko_barrier=103.0,
        ki_barrier=95.0,
        maturity=2.0 / 365.0,
        ko_dates=[2.0 / 365.0],
        ki_dates=[1.0 / 365.0],
    )
    results = _run_with_recording_engine(_config(product, market_data), RecordingEngine())

    actions = results.actions_df
    assert list(actions["action_type"]) == ["KI", "KO"]
    assert bool(actions.iloc[1]["knocked_in_before"]) is True


def test_final_parachute_tie_resolves_as_ko_before_ki_or_maturity():
    dates = pd.Series(pd.date_range("2024-01-02", periods=2, freq="D"))
    terms = case_study.CaseStudyTerms(notional=10_000_000.0)
    product = case_study.build_case_study_products(
        initial_spot=100.0,
        issue_date=pd.Timestamp("2024-01-02"),
        dates=dates,
        terms=terms,
        coupons={"PPP-EKI-Parachute": 0.10},
    )["PPP-EKI-Parachute"]
    market_data = _market_data([100.0, 75.0])

    results = _run_with_recording_engine(_config(product, market_data), RecordingEngine())

    assert list(results.actions_df["action_type"]) == ["KO"]
    assert bool(results.states_df["knocked_out"].iloc[-1]) is True
    assert bool(results.states_df["matured"].iloc[-1]) is False


def test_surviving_product_settles_maturity_and_closes_hedge():
    market_data = _market_data([100.0, 100.0])
    product = _product(
        ko_barrier=200.0,
        ki_barrier=50.0,
        maturity=1.0 / 365.0,
        ko_dates=[1.0 / 365.0],
        ki_dates=[1.0 / 365.0],
    )
    results = _run_with_recording_engine(
        _config(product, market_data),
        RecordingEngine(delta_before=400.0),
    )

    assert list(results.actions_df["action_type"]) == ["MATURITY"]
    assert "hedge_close" in set(results.trades_df["trade_type"])
    assert bool(results.states_df["matured"].iloc[-1]) is True
    assert results.states_df["futures_contracts"].iloc[-1] == pytest.approx(0.0)


def test_roll_closes_held_contract_even_when_old_contract_missing_from_slice():
    dates = pd.date_range("2024-01-02", periods=2, freq="D")
    market_data = AutocallableMarketDataSet.from_dataframes(
        spot_data=pd.DataFrame({"date": dates, "spot": [100.0, 101.0]}),
        vol_data=pd.DataFrame({"date": dates, "volatility": [0.25, 0.25]}),
        rate_data=pd.DataFrame({"date": dates, "rate": [0.02, 0.02]}),
        futures_data=pd.DataFrame(
            [
                {
                    "date": dates[0],
                    "contract": "IM2401",
                    "futures_price": 100.0,
                    "expiry_date": pd.Timestamp("2024-01-05"),
                    "multiplier": 200.0,
                },
                {
                    "date": dates[1],
                    "contract": "IM2402",
                    "futures_price": 101.0,
                    "expiry_date": pd.Timestamp("2024-02-16"),
                    "multiplier": 200.0,
                },
            ]
        ),
    )
    product = _product(
        ko_barrier=200.0,
        ki_barrier=50.0,
        maturity=5.0 / 365.0,
        ko_dates=[5.0 / 365.0],
        ki_dates=[5.0 / 365.0],
    )

    results = _run_with_recording_engine(
        _config(product, market_data),
        RecordingEngine(delta_before=400.0),
    )

    trades = results.trades_df.reset_index()
    assert "roll_close" in set(trades["trade_type"])
    assert "roll_open" in set(trades["trade_type"])
    assert "futures_roll_missing_old_contract" in set(trades["reason"])


def test_fair_coupon_solver_returns_coupon_with_small_initial_pv():
    dates = pd.Series(pd.date_range("2024-01-02", periods=252, freq="B"))
    terms = case_study.CaseStudyTerms(notional=1_000_000.0)
    engine_config = AutocallableEngineConfig(
        pricing_engine_type=EngineType.QUADRATURE,
        quad_params=QuadParams(grid_points=101),
    )
    env = case_study.pricing_environment(100.0, pd.Timestamp("2024-01-02"), terms)

    def builder(coupon: float):
        return case_study.build_case_study_products(
            initial_spot=100.0,
            issue_date=pd.Timestamp("2024-01-02"),
            dates=dates,
            terms=terms,
            coupons={"PPP-DKI": coupon},
        )["PPP-DKI"]

    coupon = case_study.solve_fair_coupon(
        product_builder=builder,
        engine_config=engine_config,
        env=env,
        tolerance=1e-2,
    )
    product = builder(coupon)
    price = case_study.create_pricing_engine(product, engine_config).price(product, env)

    assert 0.0 <= coupon <= 5.0
    assert abs(price) < 1.0


def test_case_study_runner_smoke_creates_all_final_artifacts(tmp_path: Path):
    args = case_study.parse_args(
        [
            "--synthetic-only",
            "--scenario-days",
            "8",
            "--quad-grid",
            "101",
            "--output-dir",
            str(tmp_path / "case_study"),
        ]
    )

    manifest = case_study.run_case_study(args)

    assert manifest["num_runs"] == 15
    assert Path(manifest["workbook"]).exists()
    assert Path(manifest["docx"]).exists()
    assert Path(manifest["html"]).exists()
    dashboard = Path(manifest["html"]).read_text(encoding="utf-8")
    assert "General Manager View" in dashboard
    assert "Trader View" in dashboard
    assert "Risk Manager View" in dashboard
    assert "Scenario Price Paths" in dashboard
    assert "Total PnL K-Line" in dashboard
    assert "trade-chart-data" in dashboard
    assert "freq-button" in dashboard
    assert "chart-kind-button" in dashboard
    assert "multi-select-button" in dashboard
    assert 'data-role="scenario" multiple' in dashboard
    assert 'data-role="product" multiple' in dashboard
    assert "Daily Detail" in dashboard
    assert "daily-detail-data" in dashboard
    assert "Lifecycle Event Timeline" in dashboard
    assert "Final PnL Matrix" in dashboard
    assert (tmp_path / "case_study" / "charts" / "scenario_price_paths.png").exists()
    assert (tmp_path / "case_study" / "data" / "daily_greeks.csv").exists()
    assert (tmp_path / "case_study" / "data" / "daily_pnl.csv").exists()
    assert (tmp_path / "case_study" / "data" / "hedge_actions.csv").exists()
