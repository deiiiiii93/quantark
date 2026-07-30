from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde import GridConfig
from quantark.asset.equity.engine.pde_engine import PDEEngine
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import (
    BumpConfig,
    EngineParams,
    MCParams,
    PDEParams,
    QuadParams,
)
from quantark.asset.equity.product.option import (
    create_standard_phoenix,
    create_standard_snowball,
)
from quantark.backtest.otc import (
    AKShareAutocallableDataAdapter,
    AutocallableBacktestConfig,
    AutocallableBacktestEngine,
    AutocallableDeltaHedgeStrategy,
    AutocallableEngineConfig,
    AutocallableMarketDataSet,
    FuturesRollPolicy,
    SignedDividendYield,
    SurfaceGridConfig,
    derive_implied_dividend_yield,
)
from quantark.backtest.otc.engine_factory import create_pricing_engine, create_surface_engine
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import EngineType, PDEMethod


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


def _phoenix_product():
    return create_standard_phoenix(
        initial_price=100.0,
        strike=100.0,
        maturity=0.25,
        contract_multiplier=100.0,
        ko_barrier=103.0,
        ki_barrier=75.0,
        coupon_barrier=95.0,
        ko_rate=0.02,
        coupon_rate=0.01,
        num_observations=2,
        ko_observation_dates=[0.125, 0.25],
        ki_observation_type=ObservationType.DISCRETE,
        ki_continuous=False,
        ki_observation_dates=[0.125, 0.25],
        memory_coupon=True,
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


class QuadraticRecordingEngine(BaseEngine):
    def __init__(self, params: EngineParams | None = None) -> None:
        super().__init__(params or EngineParams(bump_size=0.001))
        self.spot_calls: list[float] = []

    def price(self, product, pricing_env: PricingEnvironment) -> float:
        spot = float(pricing_env.spot)
        self.spot_calls.append(spot)
        return spot**2


def test_basis_implied_q_uses_simple_basis_and_q_floor():
    basis_yield, implied_q = derive_implied_dividend_yield(
        rate=0.02,
        spot=100.0,
        futures_price=104.0,
        time_to_maturity=0.25,
    )

    assert basis_yield == pytest.approx(0.04 / 0.25)
    assert implied_q == 0.0
    assert SignedDividendYield(implied_q).get_yield(1.0) == pytest.approx(0.0)


def test_futures_roll_policy_selects_next_valid_contract():
    valuation_date = pd.Timestamp("2024-01-02")
    futures_slice = pd.DataFrame(
        [
            {
                "contract": "IF2401",
                "expiry_date": pd.Timestamp("2024-01-06"),
                "futures_price": 100.0,
                "multiplier": 300.0,
            },
            {
                "contract": "IF2402",
                "expiry_date": pd.Timestamp("2024-01-20"),
                "futures_price": 101.0,
                "multiplier": 300.0,
            },
        ]
    )

    selected = FuturesRollPolicy(roll_days_before_expiry=5).select_contract(
        futures_slice, valuation_date, current_contract="IF2401"
    )

    assert selected["contract"] == "IF2402"


def test_engine_factory_selects_snowball_and_phoenix_engines():
    snowball = _snowball_product()
    phoenix = _phoenix_product()

    pde_config = AutocallableEngineConfig(
        pricing_engine_type=EngineType.PDE,
        method=EngineType.PDE(PDEMethod.CRANK_NICOLSON),
        pde_params=PDEParams(grid=GridConfig(points=80)),
    )
    mc_config = AutocallableEngineConfig(
        pricing_engine_type=EngineType.MONTE_CARLO,
        mc_params=MCParams(num_paths=64, time_steps=8, seed=11),
        quad_params=QuadParams(grid_points=101, num_std_devs=4.0),
    )
    quad_config = AutocallableEngineConfig(
        pricing_engine_type=EngineType.QUADRATURE,
        quad_params=QuadParams(grid_points=101, num_std_devs=4.0),
    )

    assert isinstance(create_pricing_engine(snowball, pde_config), PDEEngine)
    assert isinstance(create_pricing_engine(snowball, mc_config), SnowballMCEngine)
    assert isinstance(create_pricing_engine(phoenix, mc_config), PhoenixMCEngine)
    assert isinstance(create_pricing_engine(snowball, quad_config), SnowballQuadEngine)
    assert isinstance(create_pricing_engine(phoenix, quad_config), PhoenixQuadEngine)
    assert isinstance(create_surface_engine(snowball, mc_config), SnowballQuadEngine)


def test_gamma_bump_config_is_independent_from_delta_bump():
    product = _snowball_product()
    config = AutocallableBacktestConfig(
        product=product,
        market_data=_market_data(),
        engine_config=AutocallableEngineConfig(
            pricing_engine_type=EngineType.QUADRATURE,
            quad_params=QuadParams(grid_points=101),
        ),
        gamma_bump_size=0.01,
        calculate_surfaces=False,
        calculate_event_probabilities=False,
    )
    backtest = AutocallableBacktestEngine(config)
    # Post-consolidation contract: bump sizes live on the ENGINE's BumpConfig
    # (config-level overrides flow into factory-built engines; an injected
    # engine carries its own). The engine reprices its own base (100.0 first).
    recording_engine = QuadraticRecordingEngine(
        EngineParams(bump_config=BumpConfig(spot_bump=0.001, gamma_spot_bump=0.01))
    )
    backtest.pricing_engine = recording_engine
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0, asset_name="CSI500"),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=SignedDividendYield(0.0),
        valuation_date=datetime(2024, 1, 2),
    )

    greeks = backtest._calculate_greeks(product, env, price=10_000.0)

    assert greeks["delta"] == pytest.approx(200.0)
    assert greeks["gamma"] == pytest.approx(2.0)
    assert recording_engine.spot_calls == pytest.approx(
        [100.0, 100.1, 99.9, 101.0, 99.0]
    )

    # Config-level overrides reach engines the factory builds.
    factory_engine = create_pricing_engine(
        product,
        config.engine_config,
        delta_bump_size=config.delta_bump_size,
        gamma_bump_size=config.gamma_bump_size,
    )
    factory_bumps = factory_engine.params.get_effective_bump_config()
    assert factory_bumps.gamma_spot_bump == pytest.approx(0.01)


def test_delta_hedge_contract_sizing_uses_futures_multiplier():
    strategy = AutocallableDeltaHedgeStrategy(round_contracts=False)

    target = strategy.target_contracts(
        product_delta=1000.0,
        product_quantity=-2.0,
        futures_multiplier=200.0,
    )

    assert target == pytest.approx(10.0)


def test_akshare_adapter_normalizes_mocked_raw_frames():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02"],
            "spot": [101.0, 100.0],
        }
    )

    normalized = AKShareAutocallableDataAdapter.normalize_spot(raw)

    assert list(normalized["date"]) == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert list(normalized["spot"]) == [100.0, 101.0]


def test_mc_engine_accepts_negative_signed_dividend_yield():
    product = _snowball_product()
    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=SignedDividendYield(-0.01),
        valuation_date=datetime(2024, 1, 2),
    )
    engine = SnowballMCEngine(params=MCParams(num_paths=64, time_steps=8, seed=3))

    price = engine.price(product, env)

    assert np.isfinite(price)


def test_synthetic_snowball_backtest_outputs_daily_records_and_events():
    config = AutocallableBacktestConfig(
        product=_snowball_product(),
        market_data=_market_data(),
        engine_config=AutocallableEngineConfig(
            pricing_engine_type=EngineType.MONTE_CARLO,
            mc_params=MCParams(num_paths=128, time_steps=8, seed=7),
            quad_params=QuadParams(grid_points=101, num_std_devs=4.0),
        ),
        strategy=AutocallableDeltaHedgeStrategy(
            delta_threshold=0.0,
            round_contracts=False,
        ),
        surface_config=SurfaceGridConfig(spot_nodes=3, q_nodes=3),
        calculate_surfaces=True,
        calculate_event_probabilities=True,
        # Records over the full window: pin the pre-0.5 termination behavior.
        terminate_on_lifecycle_end=False,
        product_quantity=-1.0,
        underlying="CSI500",
    )

    results = AutocallableBacktestEngine(config).run()

    assert len(results.states_df) == 5
    assert len(results.greeks_df) == 5
    assert len(results.rebalance_df) == 5
    assert not results.daily_event_summary_df.empty
    assert not results.surfaces_df.empty
    assert results.surfaces_df["q_node"].min() >= 0.0
    assert {"delta_cash_1pct", "gamma_cash_1pct"}.issubset(
        results.surfaces_df.columns
    )
    expected_surface_delta_cash = (
        results.surfaces_df["delta"]
        * results.surfaces_df["spot_node"]
        * 0.01
    )
    expected_surface_gamma_cash = (
        results.surfaces_df["gamma"]
        * results.surfaces_df["spot_node"] ** 2
        / 100.0
    )
    assert np.allclose(
        results.surfaces_df["delta_cash_1pct"],
        expected_surface_delta_cash,
        equal_nan=True,
    )
    assert np.allclose(
        results.surfaces_df["gamma_cash_1pct"],
        expected_surface_gamma_cash,
        equal_nan=True,
    )
    assert {"KO", "KI"}.issubset(set(results.event_probability_df["event_type"]))
    assert set(results.actions_df["action_type"]) == {"KI", "KO"}
    assert results.states_df["implied_q"].iloc[0] == 0.0
    assert results.states_df["active_contract"].iloc[0] == "IF2402"

    expected_greek_columns = {
        "pre_hedge_delta",
        "post_hedge_delta",
        "pre_hedge_gamma",
        "post_hedge_gamma",
        "pre_hedge_delta_cash_1pct",
        "post_hedge_delta_cash_1pct",
        "pre_hedge_gamma_cash_1pct",
        "post_hedge_gamma_cash_1pct",
        "delta_cash_1pct",
        "gamma_cash_1pct",
    }
    assert expected_greek_columns.issubset(results.greeks_df.columns)
    spot = results.states_df.loc[results.greeks_df.index, "spot"]
    expected_delta_cash = results.greeks_df["post_hedge_delta"] * spot * 0.01
    expected_gamma_cash = results.greeks_df["post_hedge_gamma"] * spot**2 / 100.0
    assert np.allclose(
        results.greeks_df["post_hedge_delta_cash_1pct"],
        expected_delta_cash,
        equal_nan=True,
    )
    assert np.allclose(
        results.greeks_df["post_hedge_gamma_cash_1pct"],
        expected_gamma_cash,
        equal_nan=True,
    )
