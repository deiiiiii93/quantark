"""Closed-form Black-Scholes greeks for European vanillas.

Bodies moved verbatim from GreeksCalculator (R1c pure code motion).
"""

import math
from typing import Dict, Optional

from scipy import stats

from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero


def calculate_analytical_greeks(
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
    }
