"""
Base class for credit pricing engines.
"""
from abc import ABC, abstractmethod
from typing import Dict

from quantark.asset.credit.product.base_credit_product import BaseCreditProduct
from quantark.priceenv import CreditPricingEnvironment


class BaseCreditEngine(ABC):
    """
    Abstract base class for credit pricing engines.

    Engines compute present values and risk measures from a product
    specification and a :class:`CreditPricingEnvironment`.
    """

    @abstractmethod
    def price(
        self, product: BaseCreditProduct, env: CreditPricingEnvironment
    ) -> float:
        """Present value of the product from the position holder's perspective."""

    def calculate_greeks(
        self, product: BaseCreditProduct, env: CreditPricingEnvironment
    ) -> Dict[str, float]:
        """Risk measures for the product. Default: not implemented per engine."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement calculate_greeks"
        )
