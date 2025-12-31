"""
Base class for pricing engines.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from asset.equity.product.base_equity_product import BaseEquityProduct
from priceenv import PricingEnvironment
from asset.equity.param import EngineParams
from util.enum.engine_enums import EngineType


class BaseEngine(ABC):
    """
    Abstract base class for all pricing engines.

    Engines are responsible for computing prices and Greeks for derivatives.

    Attributes:
        engine_type: The type category of this engine (ANALYTICAL, MONTE_CARLO, PDE, etc.)
    """

    engine_type: EngineType = EngineType.ANALYTICAL

    def __init__(self, params: Optional[EngineParams] = None):
        """
        Initialize the engine.

        Args:
            params: Engine configuration parameters
        """
        self.params = params if params is not None else EngineParams()

    @abstractmethod
    def price(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> float:
        """
        Calculate the price of the product.

        Args:
            product: The derivative product to price
            pricing_env: Pricing environment with market data

        Returns:
            Product price
        """
        pass

    def calculate_greeks(
        self, product: BaseEquityProduct, pricing_env: PricingEnvironment
    ) -> Dict[str, float]:
        """
        Calculate Greeks using finite difference method.

        This default implementation uses bump-and-reprice.
        Subclasses can override to provide analytical Greeks.

        Args:
            product: The derivative product
            pricing_env: Pricing environment with market data

        Returns:
            Dictionary of Greeks
        """
        from copy import deepcopy

        base_price = self.price(product, pricing_env)
        greeks = {"price": base_price}

        # Delta: dV/dS
        env_up = deepcopy(pricing_env)
        env_up.spot_quote.spot *= 1 + self.params.bump_size
        price_up = self.price(product, env_up)

        env_down = deepcopy(pricing_env)
        env_down.spot_quote.spot *= 1 - self.params.bump_size
        price_down = self.price(product, env_down)

        delta = (price_up - price_down) / (2 * pricing_env.spot * self.params.bump_size)
        greeks["delta"] = delta

        # Gamma: d²V/dS²
        gamma = (price_up - 2 * base_price + price_down) / (
            pricing_env.spot * self.params.bump_size
        ) ** 2
        greeks["gamma"] = gamma

        return greeks

    def __repr__(self):
        return f"{self.__class__.__name__}()"
