"""
Greeks calculator for FX products.
"""

from typing import Dict, Optional

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine, FxEngineParams
from quantark.asset.fx.product.base_fx_product import BaseFxProduct
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import GreeksCalculationMode


class FxGreeksCalculator:
    """
    Orchestrates FX Greeks calculation across engines.

    Modes (quantark.util.enum.GreeksCalculationMode):
        - ENGINE: use the engine's own calculate_greeks (closed forms when
          the engine overrides them)
        - BUMP: force the universal finite-difference implementation from
          BaseFxEngine, regardless of the engine's overrides
        - AUTO: use the engine's method when it overrides the base
          implementation, otherwise bump (default)

    All modes return the same key set with legacy conventions: price, delta,
    delta_percentage, delta_premium, fwd_delta, fwd_delta_premium, gamma,
    gamma_percentage, vega (per 1% vol), theta (daily), rho_dom, rho_for
    (dV/dr / 100).
    """

    def __init__(self, params: Optional[FxEngineParams] = None):
        """
        Args:
            params: Bump sizes for the finite-difference path (optional)
        """
        self.params = params

    def calculate(
        self,
        product: BaseFxProduct,
        fx_env: FxPricingEnvironment,
        engine: BaseFxEngine,
        mode: GreeksCalculationMode = GreeksCalculationMode.AUTO,
    ) -> Dict[str, float]:
        """
        Calculate Greeks for an FX product.

        Args:
            product: The FX product
            fx_env: FX pricing environment
            engine: Pricing engine for the product
            mode: Greeks calculation mode

        Returns:
            Dictionary of Greeks (see class docstring for conventions)
        """
        if self.params is not None:
            engine = type(engine)(params=self.params)

        if mode == GreeksCalculationMode.BUMP:
            return BaseFxEngine.calculate_greeks(engine, product, fx_env)
        if mode == GreeksCalculationMode.ENGINE:
            return engine.calculate_greeks(product, fx_env)

        # AUTO: prefer the engine's own (analytical) Greeks when overridden
        if self._has_custom_greeks(engine):
            return engine.calculate_greeks(product, fx_env)
        return BaseFxEngine.calculate_greeks(engine, product, fx_env)

    @staticmethod
    def _has_custom_greeks(engine: BaseFxEngine) -> bool:
        """True when the engine overrides the base FDM implementation."""
        return (
            type(engine).calculate_greeks is not BaseFxEngine.calculate_greeks
        )

    def __repr__(self):
        return "FxGreeksCalculator()"
