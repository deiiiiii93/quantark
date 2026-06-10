"""
Rate index definitions for floating rate instruments.
"""

from quantark.util.enum import ResetConvention
from .rate_index import (
    RateIndex,
    IndexFixing,
    IndexFixingStore,
    # Predefined indices - US Market
    SOFR,
    SOFR_1M,
    SOFR_3M,
    PRIME,
    # Predefined indices - Europe
    EURIBOR_3M,
    EURIBOR_6M,
    ESTR,
    # Predefined indices - China
    SHIBOR_3M,
    REPO_7D,
    # Predefined indices - Legacy
    LIBOR_3M,
    LIBOR_6M,
    # Factory function
    create_index,
)

__all__ = [
    "RateIndex",
    "IndexFixing",
    "IndexFixingStore",
    "ResetConvention",
    # US Market
    "SOFR",
    "SOFR_1M",
    "SOFR_3M",
    "PRIME",
    # Europe
    "EURIBOR_3M",
    "EURIBOR_6M",
    "ESTR",
    # China
    "SHIBOR_3M",
    "REPO_7D",
    # Legacy
    "LIBOR_3M",
    "LIBOR_6M",
    # Factory
    "create_index",
]

