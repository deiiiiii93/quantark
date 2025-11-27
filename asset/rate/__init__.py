"""
Interest Rate derivatives module.

This module provides:
- Interest Rate Swap (IRS) products
- Basis Swaps
- Pricing engines for rate products
"""

from .product import InterestRateSwap, BasisSwap, FixedLeg, FloatingLeg
from .engine import IRSDiscountEngine

__all__ = [
    'InterestRateSwap',
    'BasisSwap',
    'FixedLeg',
    'FloatingLeg',
    'IRSDiscountEngine',
]

