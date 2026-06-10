"""Coupon bond products."""

from .fixed_bond import FixedBond, create_simple_fixed_bond
from .frn import FloatingRateBond, FloatingCashFlow, create_simple_frn

__all__ = [
    "FixedBond",
    "create_simple_fixed_bond",
    "FloatingRateBond",
    "FloatingCashFlow",
    "create_simple_frn",
]
