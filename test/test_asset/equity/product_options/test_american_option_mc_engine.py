"""
Unit tests for American option Monte Carlo pricing engine.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantark.asset.equity.product.option import AmericanOption
from quantark.asset.equity.engine.analytical import AmericanOptionAnalyticalEngine
from quantark.asset.equity.engine.mc import AmericanOptionMCEngine
from quantark.asset.equity.engine.pde import AmericanPDESolver
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import MonteCarloMethod, AmericanAnalyticalMethod
from quantark.util.numerical import is_close


def create_pricing_env(
    spot: float = 100.0,
    vol: float = 0.20,
    rate: float = 0.05,
    div: float = 0.02,
) -> PricingEnvironment:
    """Helper to create standard pricing environment."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


def create_mc_engine(
    num_paths: int = 20000,
    time_steps: int = 200,
    method: MonteCarloMethod = MonteCarloMethod.QUASI,
) -> AmericanOptionMCEngine:
    """Helper to create American MC engine with standard parameters."""
    params = MCParams(num_paths=num_paths, time_steps=time_steps, seed=42)
    return AmericanOptionMCEngine(params=params, method=method)


def _assert_mc_close_to_reference(
    mc_price: float,
    ref_price: float,
    std_error: float,
    label: str,
    min_abs_tol: float = 0.25,
) -> None:
    """Assert MC price is within tolerance of reference price."""
    tolerance = max(3.0 * std_error, min_abs_tol)
    assert is_close(mc_price, ref_price, rel_tol=0.0, abs_tol=tolerance), (
        f"{label} MC price {mc_price:.6f} differs from reference {ref_price:.6f} "
        f"by more than tolerance {tolerance:.6f}"
    )


def test_mc_put_matches_analytical() -> None:
    """Test MC American put pricing against analytical approximation."""
    pricing_env = create_pricing_env(spot=100.0, vol=0.20, rate=0.05, div=0.02)
    option = AmericanOption(
        strike=100.0,
        option_type=OptionType.PUT,
        maturity=1.0,
    )

    mc_engine = create_mc_engine()
    mc_price = mc_engine.price(option, pricing_env)
    mc_result = mc_engine.get_last_result()
    assert mc_result is not None

    analytical_engine = AmericanOptionAnalyticalEngine(
        method=AmericanAnalyticalMethod.BS93
    )
    analytical_price = analytical_engine.price(option, pricing_env)

    _assert_mc_close_to_reference(
        mc_price=mc_price,
        ref_price=analytical_price,
        std_error=mc_result.std_error,
        label="Analytical",
    )


def test_mc_put_matches_pde() -> None:
    """Test MC American put pricing against PDE solver."""
    pricing_env = create_pricing_env(spot=90.0, vol=0.25, rate=0.05, div=0.02)
    option = AmericanOption(
        strike=100.0,
        option_type=OptionType.PUT,
        maturity=1.0,
    )

    mc_engine = create_mc_engine()
    mc_price = mc_engine.price(option, pricing_env)
    mc_result = mc_engine.get_last_result()
    assert mc_result is not None

    pde_params = PDEParams()
    pde_engine = AmericanPDESolver(params=pde_params)
    pde_price = pde_engine.price(option, pricing_env)

    _assert_mc_close_to_reference(
        mc_price=mc_price,
        ref_price=pde_price,
        std_error=mc_result.std_error,
        label="PDE",
        min_abs_tol=0.5,
    )
