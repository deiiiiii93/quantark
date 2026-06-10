"""
Pricing engine factory for OTC autocallable backtests.
"""

from typing import Any

from quantark.asset.equity.engine.analytical import (
    AsianOptionAnalyticalEngine,
    BarrierAnalyticalEngine,
    BlackScholesEngine,
    DoubleBarrierOptionAnalyticalEngine,
    DoubleSharkfinOptionAnalyticalEngine,
    OneTouchAnalyticalEngine,
    SingleSharkfinOptionAnalyticalEngine,
)
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde_engine import PDEEngine
from quantark.asset.equity.engine.quad.european_quad_engine import EuropeanQuadEngine
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import EngineParams, MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import (
    AsianOption,
    BarrierOption,
    DoubleBarrierOption,
    DoubleSharkfinOption,
    EuropeanVanillaOption,
    OneTouchOption,
    SingleSharkfinOption,
)
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.util.enum.engine_enums import EngineType
from quantark.util.exceptions import ValidationError

from .config import AutocallableEngineConfig


def _create_analytical_engine(product: Any, config: AutocallableEngineConfig) -> BaseEngine:
    params = EngineParams()
    if isinstance(product, EuropeanVanillaOption):
        return BlackScholesEngine(params=params)
    if isinstance(product, AsianOption):
        return AsianOptionAnalyticalEngine(params=params, method=config.method)
    if isinstance(product, BarrierOption):
        return BarrierAnalyticalEngine(params=params)
    if isinstance(product, DoubleBarrierOption):
        return DoubleBarrierOptionAnalyticalEngine(params=params)
    if isinstance(product, OneTouchOption):
        return OneTouchAnalyticalEngine(params=params)
    if isinstance(product, SingleSharkfinOption):
        return SingleSharkfinOptionAnalyticalEngine(params=params)
    if isinstance(product, DoubleSharkfinOption):
        return DoubleSharkfinOptionAnalyticalEngine(params=params)
    raise ValidationError(f"Unsupported analytical product type: {type(product).__name__}")


def create_autocallable_engine(
    *,
    product: Any,
    engine_type: EngineType,
    config: AutocallableEngineConfig,
    method: Any = None,
) -> BaseEngine:
    """Create a product-compatible pricing engine for backtest replay."""
    if engine_type == EngineType.ANALYTICAL:
        return _create_analytical_engine(product, config)

    if engine_type == EngineType.PDE:
        return PDEEngine(
            params=config.pde_params or PDEParams(),
            method=method if method is not None else config.method,
        )

    if engine_type == EngineType.MONTE_CARLO:
        mc_params = config.mc_params or MCParams()
        selected_method = method if method is not None else config.method
        if isinstance(product, EuropeanVanillaOption):
            return EuropeanMCEngine(params=mc_params, method=selected_method)
        if isinstance(product, PhoenixOption):
            return PhoenixMCEngine(params=mc_params, method=selected_method)
        if isinstance(product, SnowballOption):
            return SnowballMCEngine(params=mc_params, method=selected_method)
        raise ValidationError(f"Unsupported MC product type: {type(product).__name__}")

    if engine_type == EngineType.QUADRATURE:
        quad_params = config.quad_params or QuadParams()
        if isinstance(product, EuropeanVanillaOption):
            selected_method = method if method is not None else config.method
            return EuropeanQuadEngine(params=quad_params, method=selected_method)
        if isinstance(product, PhoenixOption):
            return PhoenixQuadEngine(params=quad_params)
        if isinstance(product, SnowballOption):
            return SnowballQuadEngine(params=quad_params)
        raise ValidationError(f"Unsupported Quad product type: {type(product).__name__}")

    raise ValidationError(f"Unsupported engine type: {engine_type}")


def create_pricing_engine(product: Any, config: AutocallableEngineConfig) -> BaseEngine:
    return create_autocallable_engine(
        product=product,
        engine_type=config.pricing_engine_type,
        config=config,
    )


def create_surface_engine(product: Any, config: AutocallableEngineConfig) -> BaseEngine:
    return create_autocallable_engine(
        product=product,
        engine_type=config.resolve_surface_engine_type(),
        config=config,
        method=None,
    )


def create_event_stats_engine(product: Any, config: AutocallableEngineConfig) -> BaseEngine:
    return create_autocallable_engine(
        product=product,
        engine_type=config.resolve_event_stats_engine_type(),
        config=config,
        method=None,
    )


def create_mc_event_stats_engine(product: Any, config: AutocallableEngineConfig) -> BaseEngine:
    fallback = AutocallableEngineConfig(
        pricing_engine_type=EngineType.MONTE_CARLO,
        mc_params=config.mc_params or MCParams(num_paths=5000, seed=42),
    )
    return create_autocallable_engine(
        product=product,
        engine_type=EngineType.MONTE_CARLO,
        config=fallback,
    )
