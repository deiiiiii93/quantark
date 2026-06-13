"""Credit products."""
from .base_credit_product import BaseCreditProduct
from .cds import CDS, ProtectionSide
from .basket_cds import BasketCDS, BasketType, CopulaType

__all__ = [
    "BaseCreditProduct",
    "CDS",
    "ProtectionSide",
    "BasketCDS",
    "BasketType",
    "CopulaType",
]
