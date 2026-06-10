"""
Interest Rate derivative products.

Includes IRS, Basis Swaps, FRA, Cap/Floor, Collar, and Swaption.
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
from .fra import (
    ForwardRateAgreement,
    create_fra,
)
from .cap_floor import (
    CapFloor,
    CapFloorType,
    Caplet,
    Cap,
    Floor,
    Collar,
    create_cap,
    create_floor,
    create_collar,
)
from .swaption import (
    Swaption,
    SwaptionType,
    SwaptionExerciseStyle,
    create_payer_swaption,
    create_receiver_swaption,
)

__all__ = [
    # IRS
    'InterestRateSwap',
    'BasisSwap',
    'FixedLeg',
    'FloatingLeg',
    'SwapLeg',
    'NotionalSchedule',
    'SwapDirection',
    # FRA
    'ForwardRateAgreement',
    'create_fra',
    # Cap/Floor
    'CapFloor',
    'CapFloorType',
    'Caplet',
    'Cap',
    'Floor',
    'Collar',
    'create_cap',
    'create_floor',
    'create_collar',
    # Swaption
    'Swaption',
    'SwaptionType',
    'SwaptionExerciseStyle',
    'create_payer_swaption',
    'create_receiver_swaption',
]
