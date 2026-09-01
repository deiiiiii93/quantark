"""Closed-form Black-Scholes greeks for European vanillas.

The base set is the incumbent GreeksCalculator surface (moved verbatim in
R1c); EXTENDED_GREEKS adds closed forms for the higher-order greeks, which
are reachable only through the explicit ``greeks=`` parameter or auto
routing for names that never had an incumbent route.
"""

import math
from typing import Dict, Iterable, Optional, Sequence

from scipy import stats

from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero

#: Higher-order names with closed forms (beyond the incumbent base dict).
EXTENDED_GREEKS = frozenset(
    {
        "vanna",
        "volga",
        "charm",
        "color",
        "speed",
        "zomma",
        "vega_theta",
        "gamma_theta",
        "delta_q",
        "dividend_volga",
    }
)

#: Keys of the incumbent base dict (calculate_analytical_greeks(greeks=None)).
BASE_GREEKS = frozenset(
    {
        "price",
        "delta",
        "gamma",
        "vega",
        "theta",
        "convexity_theta",
        "r_theta",
        "q_theta",
        "rho",
        "dividend_rho",
    }
)


def calculate_analytical_greeks(
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    price: Optional[float] = None,
    greeks: Optional[Sequence[object]] = None,
) -> Dict[str, float]:
    """Closed-form BS greeks.

    With ``greeks=None`` this returns exactly the incumbent key set. An
    explicit ``greeks`` list may add the EXTENDED_GREEKS closed forms and
    returns only the requested keys; clock-qualified names are rejected
    (analytical time greeks are per calendar day).
    """
    if greeks is None:
        return _base_greeks(product, pricing_env, price)

    from quantark.asset.equity.riskmeasures.greeks.registry import (
        normalize_greeks,
    )

    requests = normalize_greeks(greeks)
    if requests is None or len(requests) == 0:
        return {}
    names = set()
    for req in requests:
        if req.clock is not None:
            raise ValidationError(
                "Analytical greeks do not support clock-qualified names "
                f"(got {req.key!r}); they are per calendar day"
            )
        if req.canonical not in BASE_GREEKS | EXTENDED_GREEKS:
            raise ValidationError(
                f"Analytical greeks do not support: {req.canonical!r}"
            )
        names.add(req.canonical)

    full = _base_greeks(product, pricing_env, price)
    extended_needed = names & EXTENDED_GREEKS
    if extended_needed:
        full.update(_extended_greeks(product, pricing_env, extended_needed))
    return {name: full[name] for name in names}


def _extended_greeks(
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    names: Iterable[str],
) -> Dict[str, float]:
    """Closed forms for the higher-order greeks (continuous dividend q).

    Units match the numerical conventions: charm/color per calendar day,
    vega_theta as (vega per 1 vol pt) per calendar day, vanna/volga/speed/
    zomma/delta_q/dividend_volga raw. All scaled by contract_multiplier.
    Signs are pinned by the FD oracle in test_analytical_higher_order.py.
    """
    if not isinstance(product, EuropeanVanillaOption):
        raise ValidationError(
            f"Analytical Greeks only support EuropeanVanillaOption, "
            f"got {type(product).__name__}"
        )
    S = pricing_env.spot
    K = product.strike
    T = product.get_maturity(pricing_env)
    if is_zero(T):
        return {name: 0.0 for name in names}
    r = pricing_env.get_rate(T)
    q = pricing_env.get_div_yield(T)
    sigma = pricing_env.get_vol(K, T)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    disc_div = math.exp(-q * T)
    phi_d1 = stats.norm.pdf(d1)
    is_call = product.is_call()
    N_d1 = stats.norm.cdf(d1)
    N_m_d1 = stats.norm.cdf(-d1)

    vega_raw = S * disc_div * phi_d1 * sqrt_T
    gamma = disc_div * phi_d1 / (S * sigma * sqrt_T)

    out: Dict[str, float] = {}
    out["vanna"] = -disc_div * phi_d1 * d2 / sigma
    out["volga"] = vega_raw * d1 * d2 / sigma
    out["speed"] = -(gamma / S) * (d1 / (sigma * sqrt_T) + 1.0)
    out["zomma"] = gamma * (d1 * d2 - 1.0) / sigma

    # charm: dDelta per calendar day as the valuation date advances
    charm_common = (
        -disc_div
        * phi_d1
        * (2.0 * (r - q) * T - d2 * sigma * sqrt_T)
        / (2.0 * T * sigma * sqrt_T)
    )
    if is_call:
        charm_annual = q * disc_div * N_d1 + charm_common
    else:
        charm_annual = -q * disc_div * N_m_d1 + charm_common
    out["charm"] = charm_annual / 365.0

    # color: dGamma per calendar day as the valuation date advances
    color_annual = (
        -disc_div
        * phi_d1
        / (2.0 * S * T * sigma * sqrt_T)
        * (
            2.0 * q * T
            + 1.0
            + d1 * (2.0 * (r - q) * T - d2 * sigma * sqrt_T) / (sigma * sqrt_T)
        )
    )
    out["color"] = -color_annual / 365.0

    # vega_theta (veta): d(vega per 1 vol pt) per calendar day as the
    # valuation date advances (sign pinned by the FD oracle: vega decays)
    veta_annual = vega_raw * (
        q
        + (r - q) * d1 / (sigma * sqrt_T)
        - (1.0 + d1 * d2) / (2.0 * T)
    )
    out["vega_theta"] = veta_annual / 365.0 / 100.0

    # gamma_theta: the BS-PDE identity term. The expression is written
    # exactly as _base_greeks writes convexity_theta (term1 / 365) so the
    # two are bitwise equal for vanillas, not merely algebraically equal.
    out["gamma_theta"] = (-S * disc_div * phi_d1 * sigma / (2 * sqrt_T)) / 365

    # q-axis greeks; dV/dq collapses to -/+ tau S e^{-q tau} Phi(+/-d1)
    # after the identity S e^{-q tau} phi(d1) = K e^{-r tau} phi(d2)
    if is_call:
        out["delta_q"] = -T * disc_div * N_d1 - disc_div * phi_d1 * sqrt_T / sigma
        out["dividend_volga"] = (
            T**2 * S * disc_div * N_d1 + T**1.5 * S * disc_div * phi_d1 / sigma
        )
    else:
        out["delta_q"] = T * disc_div * N_m_d1 - disc_div * phi_d1 * sqrt_T / sigma
        out["dividend_volga"] = (
            -(T**2) * S * disc_div * N_m_d1
            + T**1.5 * S * disc_div * phi_d1 / sigma
        )

    multiplier = product.contract_multiplier
    return {name: out[name] * multiplier for name in names}


def _base_greeks(
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    price: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate Greeks using analytical Black-Scholes formulas.

    Only works for European vanilla options under Black-Scholes model.

    Args:
        product: European vanilla option
        pricing_env: Pricing environment
        price: Pre-calculated price (optional, will calculate if not provided)

    Returns:
        Dictionary of Greeks: delta, gamma, vega, theta, rho

    Raises:
        ValidationError: If product is not a European vanilla option
    """
    if not isinstance(product, EuropeanVanillaOption):
        raise ValidationError(
            f"Analytical Greeks only support EuropeanVanillaOption, "
            f"got {type(product).__name__}"
        )

    # Extract parameters
    S = pricing_env.spot
    K = product.strike
    T = product.get_maturity(pricing_env)
    r = pricing_env.get_rate(T)
    q = pricing_env.get_div_yield(T)
    sigma = pricing_env.get_vol(K, T)

    # Handle edge case: option at expiry
    if is_zero(T):
        return greeks_at_expiry(product, S)

    # Calculate d1 and d2
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Calculate discount factors
    discount_div = math.exp(-q * T)
    discount_rf = math.exp(-r * T)

    # Standard normal PDF and CDF
    n_d1 = stats.norm.pdf(d1)  # phi(d1)
    N_d1 = stats.norm.cdf(d1)  # Phi(d1)
    N_d2 = stats.norm.cdf(d2)  # Phi(d2)

    greeks = {}

    multiplier = product.contract_multiplier

    # Calculate price if not provided (per-unit)
    if price is None:
        if product.is_call():
            price = S * discount_div * N_d1 - K * discount_rf * N_d2
        else:
            price = K * discount_rf * stats.norm.cdf(
                -d2
            ) - S * discount_div * stats.norm.cdf(-d1)
    else:
        price = price / multiplier
    greeks["price"] = price

    # Delta: ∂V/∂S
    if product.is_call():
        delta = discount_div * N_d1
    else:
        delta = -discount_div * stats.norm.cdf(-d1)
    greeks["delta"] = delta

    # Gamma: ∂²V/∂S²
    gamma = discount_div * n_d1 / (S * sigma * sqrt_T)
    greeks["gamma"] = gamma

    # Vega: ∂V/∂σ (divided by 100 for 1% change)
    vega = S * discount_div * n_d1 * sqrt_T / 100
    greeks["vega"] = vega

    # Theta: ∂V/∂t (per day, divided by 365)
    # Decomposed into three components:
    #   convexity_theta: time decay from gamma/convexity erosion (always negative)
    #   r_theta: time decay from interest rate cost of carry
    #   q_theta: time decay from dividend yield
    term1 = -S * discount_div * n_d1 * sigma / (2 * sqrt_T)
    if product.is_call():
        term2 = -r * K * discount_rf * N_d2
        term3 = q * S * discount_div * N_d1
    else:
        term2 = r * K * discount_rf * stats.norm.cdf(-d2)
        term3 = -q * S * discount_div * stats.norm.cdf(-d1)

    # Store decomposed components (per day)
    convexity_theta = term1 / 365
    r_theta = term2 / 365
    q_theta = term3 / 365
    theta = convexity_theta + r_theta + q_theta

    greeks["theta"] = theta
    greeks["convexity_theta"] = convexity_theta
    greeks["r_theta"] = r_theta
    greeks["q_theta"] = q_theta

    # Rho: ∂V/∂r (divided by 100 for 1% change)
    if product.is_call():
        rho = K * T * discount_rf * N_d2 / 100
    else:
        rho = -K * T * discount_rf * stats.norm.cdf(-d2) / 100
    greeks["rho"] = rho

    # Dividend Rho: ∂V/∂q (divided by 100 for 1% change)
    if product.is_call():
        dividend_rho = -S * T * discount_div * N_d1 / 100
    else:
        dividend_rho = S * T * discount_div * stats.norm.cdf(-d1) / 100
    greeks["dividend_rho"] = dividend_rho

    for key, value in greeks.items():
        greeks[key] = value * multiplier

    return greeks


def greeks_at_expiry(
    product: EuropeanVanillaOption, spot: float
) -> Dict[str, float]:
    """
    Calculate Greeks at expiry.

    At expiry:
    - Price = intrinsic value
    - Delta = 1 (ITM call), -1 (ITM put), 0 (OTM)
    - Gamma, Vega, Theta, Rho = 0

    Args:
        product: European vanilla option
        spot: Current spot price

    Returns:
        Dictionary of Greeks
    """
    multiplier = product.contract_multiplier
    price = product.get_payoff(spot) / multiplier

    # Delta at expiry
    if product.is_call():
        delta = 1.0 if spot > product.strike else 0.0
    else:
        delta = -1.0 if spot < product.strike else 0.0

    return {
        "price": price * multiplier,
        "delta": delta * multiplier,
        "gamma": 0.0,
        "vega": 0.0,
        "theta": 0.0,
        "convexity_theta": 0.0,
        "r_theta": 0.0,
        "q_theta": 0.0,
        "rho": 0.0,
        "dividend_rho": 0.0,
    }
