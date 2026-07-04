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


class EquityDividendInputMode(Enum):
    """How the option-pricing dividend/carry input is supplied."""

    FLAT_DIVIDEND = "flat_dividend"
    TERM_DIVIDEND = "term_dividend"


class FuturesCarryRiskMode(Enum):
    """Interpretation of index futures marks for pricing and carry risk.

    MARKET_PRICE: futures mark is exogenous; model rhoq = 0 by convention.
    THEORETICAL_CARRY: futures generated from S, r, q(T); rhoq non-zero.
    IMPLIED_FUTURES_CARRY: marks imply q(T) for option pricing; futures/rhoq
        buckets are portfolio risk coordinates.
    """

    MARKET_PRICE = "market_price"
    THEORETICAL_CARRY = "theoretical_carry"
    IMPLIED_FUTURES_CARRY = "implied_futures_carry"
