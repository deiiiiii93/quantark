"""
Base class for credit derivative products.
"""
from abc import ABC, abstractmethod


class BaseCreditProduct(ABC):
    """
    Abstract base class for all credit derivative products.

    Credit products are pure instrument specifications on one or more
    reference entities: they carry no market data. Hazard rates and discount
    curves are supplied at pricing time via a
    :class:`~quantark.priceenv.CreditPricingEnvironment`.
    """

    @abstractmethod
    def validate(self) -> None:
        """Validate the product specification; raise ValidationError if invalid."""
