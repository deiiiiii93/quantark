"""
Interest Rate derivatives module.

This module provides:
- Interest Rate Swap (IRS) products
- Basis Swaps
- Forward Rate Agreements (FRA)
- Interest Rate Caps, Floors, and Collars
- Swaptions (options on swaps)
- Pricing engines for rate products
"""

from .product import (
    InterestRateSwap,
    BasisSwap,
    FixedLeg,
    FloatingLeg,
    ForwardRateAgreement,
    CapFloor,
    CapFloorType,
    Caplet,
    Collar,
    Swaption,
    SwaptionType,
    SwaptionExerciseStyle,
)
from .engine import (
    IRSDiscountEngine,
    FRAEngine,
    CapFloorEngine,
    SwaptionEngine,
    SwaptionModelType,
)

__all__ = [
    # IRS
    'InterestRateSwap',
    'BasisSwap',
    'FixedLeg',
    'FloatingLeg',
    # FRA
    'ForwardRateAgreement',
    # Cap/Floor
    'CapFloor',
    'CapFloorType',
    'Caplet',
    'Collar',
    # Swaption
    'Swaption',
    'SwaptionType',
    'SwaptionExerciseStyle',
    # Engines
    'IRSDiscountEngine',
    'FRAEngine',
    'CapFloorEngine',
    'SwaptionEngine',
    'SwaptionModelType',
]
