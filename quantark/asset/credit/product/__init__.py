"""Credit products."""
from .base_credit_product import BaseCreditProduct
from .cds import CDS, ProtectionSide

__all__ = ["BaseCreditProduct", "CDS", "ProtectionSide"]
