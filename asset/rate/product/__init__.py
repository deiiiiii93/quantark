"""
Interest Rate Swap products.
"""

from .irs import (
    InterestRateSwap,
    BasisSwap,
    FixedLeg,
    FloatingLeg,
    SwapLeg,
    NotionalSchedule,
    SwapDirection,
)

__all__ = [
    'InterestRateSwap',
    'BasisSwap',
    'FixedLeg',
    'FloatingLeg',
    'SwapLeg',
    'NotionalSchedule',
    'SwapDirection',
]

