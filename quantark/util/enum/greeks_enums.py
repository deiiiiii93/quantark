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
    """Requestable scalar greeks for equity products (``GreeksCalculator``).

    Units: spot greeks are per unit spot (delta), per unit spot squared
    (gamma), per unit spot cubed (speed); vol greeks per 1 vol point (vega,
    vanna, volga, zomma); rate/carry greeks per 1 percentage point of r or q
    (rho, dividend_rho, delta_q, dividend_volga). Time greeks (theta family,
    charm, color, vega_theta, gamma_theta) are the change over one step as
    the valuation date advances, signed so that decay is negative; the step
    is one calendar or one trading day per ``BumpConfig.time_bump_mode``,
    and can be pinned per request with the ``_1d`` / ``_1td`` name suffixes
    (``"theta_1td"``, ``"charm_1d"``). All values scale with
    ``contract_multiplier``.

    Aliases accepted in requests: ``rhoq`` / ``div_rho`` / ``dividendrho``
    for DIVIDEND_RHO, ``deltaq`` / ``deltadq`` / ``d_delta_d_q`` for
    DELTA_Q, ``veta`` for VEGA_THETA. The default request is
    price, delta, gamma, vega, theta, rho, dividend_rho plus the theta
    components convexity_theta / r_theta / q_theta.

    Higher-order members: VANNA (dDelta/dVol), VOLGA (d2V/dVol2), DELTA_Q
    (dDelta/dq), CHARM (dDelta/dt), COLOR (dGamma/dt), SPEED (d3V/dS3),
    ZOMMA (dGamma/dVol), DIVIDEND_VOLGA (d2V/dq2), VEGA_THETA (dVega/dt,
    alias veta), GAMMA_THETA (the gamma bleed -1/2 sigma^2 S^2 Gamma per
    step, distinct from the convexity_theta residual on exotics).
    """

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
    SPEED = "speed"
    ZOMMA = "zomma"
    DIVIDEND_VOLGA = "dividend_volga"
    VEGA_THETA = "vega_theta"
    GAMMA_THETA = "gamma_theta"


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
