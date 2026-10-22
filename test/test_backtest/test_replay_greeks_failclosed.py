"""Fail-closed greeks and engine-side BumpConfig (plan Task 10)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from replay_golden import fixtures  # noqa: E402

from quantark.asset.equity.engine.base_engine import BaseEngine  # noqa: E402
from quantark.asset.equity.param import BumpConfig, EngineParams  # noqa: E402
from quantark.backtest.replay.single import AutocallableBacktestEngine  # noqa: E402
from quantark.util.exceptions import PricingError  # noqa: E402


class RaisingEngine(BaseEngine):
    def __init__(self) -> None:
        super().__init__(EngineParams())

    def price(self, product, pricing_env) -> float:
        raise PricingError("synthetic pricing failure")


class RecordingEngine(BaseEngine):
    """Quadratic payoff: delta = 2S, gamma = 2 under exact bumping."""

    def __init__(self, params: EngineParams) -> None:
        super().__init__(params)
        self.spots: list[float] = []

    def price(self, product, pricing_env) -> float:
        spot = float(pricing_env.spot)
        self.spots.append(spot)
        return spot**2


def test_greeks_failure_propagates_no_zero_fallback():
    """A failed price must raise — never a silent zero-delta day."""
    engine = AutocallableBacktestEngine(fixtures.make_scalar_bsm_config())
    engine.pricing_engine = RaisingEngine()
    with pytest.raises(PricingError, match="synthetic pricing failure"):
        engine.run()


def test_gamma_spot_bump_drives_second_difference():
    params = EngineParams(
        bump_config=BumpConfig(spot_bump=0.01, gamma_spot_bump=0.02)
    )
    engine = RecordingEngine(params)

    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.priceenv import PricingEnvironment
    from quantark.backtest.replay.market import SignedDividendYield
    from datetime import datetime

    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0, asset_name="X"),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=SignedDividendYield(0.0),
        valuation_date=datetime(2024, 1, 2),
    )
    greeks = engine.calculate_greeks(None, env)
    assert engine.spots == pytest.approx([100.0, 101.0, 99.0, 102.0, 98.0])
    assert greeks["delta"] == pytest.approx(200.0)
    assert greeks["gamma"] == pytest.approx(2.0)


def test_gamma_bump_defaults_to_spot_bump_no_extra_prices():
    engine = RecordingEngine(EngineParams(bump_config=BumpConfig(spot_bump=0.01)))

    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.priceenv import PricingEnvironment
    from quantark.backtest.replay.market import SignedDividendYield
    from datetime import datetime

    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0, asset_name="X"),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=SignedDividendYield(0.0),
        valuation_date=datetime(2024, 1, 2),
    )
    engine.calculate_greeks(None, env)
    assert engine.spots == pytest.approx([100.0, 101.0, 99.0])


def test_config_bump_overrides_flow_into_factory_engines():
    from quantark.backtest.replay.engine_factory import create_pricing_engine

    cfg = fixtures.make_scalar_bsm_config()
    engine = create_pricing_engine(
        cfg.product, cfg.engine_config, delta_bump_size=0.005, gamma_bump_size=0.02
    )
    bumps = engine.params.get_effective_bump_config()
    assert bumps.spot_bump == pytest.approx(0.005)
    assert bumps.gamma_spot_bump == pytest.approx(0.02)

    # Shared params objects must not be mutated by the override.
    plain = create_pricing_engine(cfg.product, cfg.engine_config)
    assert plain.params.get_effective_bump_config().gamma_spot_bump is None


def test_delta_only_override_preserves_configured_gamma_bump():
    """Override matrix (review finding): delta-only must not erase a
    configured gamma bump; gamma-only must not touch delta."""
    from dataclasses import replace as dc_replace

    from quantark.asset.equity.param import PDEParams
    from quantark.backtest.replay.engine_factory import create_pricing_engine

    cfg = fixtures.make_scalar_bsm_config()
    base_params = PDEParams(
        bump_config=BumpConfig(spot_bump=0.01, gamma_spot_bump=0.02)
    )
    engine_config = dc_replace(cfg.engine_config, pde_params=base_params)

    delta_only = create_pricing_engine(
        cfg.product, engine_config, delta_bump_size=0.005
    ).params.get_effective_bump_config()
    assert delta_only.spot_bump == pytest.approx(0.005)
    assert delta_only.gamma_spot_bump == pytest.approx(0.02)

    gamma_only = create_pricing_engine(
        cfg.product, engine_config, gamma_bump_size=0.03
    ).params.get_effective_bump_config()
    assert gamma_only.spot_bump == pytest.approx(0.01)
    assert gamma_only.gamma_spot_bump == pytest.approx(0.03)

    neither = create_pricing_engine(
        cfg.product, engine_config
    ).params.get_effective_bump_config()
    assert neither.spot_bump == pytest.approx(0.01)
    assert neither.gamma_spot_bump == pytest.approx(0.02)

    both = create_pricing_engine(
        cfg.product, engine_config, delta_bump_size=0.004, gamma_bump_size=0.04
    ).params.get_effective_bump_config()
    assert both.spot_bump == pytest.approx(0.004)
    assert both.gamma_spot_bump == pytest.approx(0.04)
