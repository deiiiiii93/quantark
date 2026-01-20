"""
Greek name enums for common and asset-specific sensitivities.
"""

from enum import Enum


class CommonGreek(Enum):
    PRICE = "price"
    DELTA = "delta"
    GAMMA = "gamma"
    VEGA = "vega"
    THETA = "theta"
    RHO = "rho"


class EquityGreek(Enum):
    PRICE = "price"
    DELTA = "delta"
    GAMMA = "gamma"
    VEGA = "vega"
    THETA = "theta"
    RHO = "rho"
    DIVIDEND_RHO = "dividend_rho"
    VANNA = "vanna"
    VOLGA = "volga"
    DELTA_Q = "delta_q"
    CHARM = "charm"
    COLOR = "color"
