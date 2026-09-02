"""Closed-form Black-Scholes greeks for European vanillas.

The base set is the incumbent GreeksCalculator surface (moved verbatim in
R1c); EXTENDED_GREEKS adds closed forms for the higher-order greeks, which
are reachable only through the explicit ``greeks=`` parameter or auto
routing for names that never had an incumbent route.

Time units. The incumbent base theta family is per calendar day (``/365``)
whatever the environment's day count; that is frozen by the compatibility
contract. The new time greeks (TIME_GREEKS: charm, color, vega_theta,
gamma_theta) instead follow the same clock as their numerical
counterparts: a bare name uses the resolved theta bump mode (per trading
day on a BUSINESS_DAYS env with a calendar, per calendar day otherwise),
``<name>_1d`` pins one calendar day (``/365``) and ``<name>_1td`` one trading
day (``/bus_days_in_year``).
"""

import math
from typing import Dict, Iterable, Optional, Sequence

from scipy import stats

from quantark.asset.equity.product.base_equity_product import BaseEquityProduct
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures.greeks import bump_envs
from quantark.asset.equity.riskmeasures.greeks.registry import (
    ANALYTICAL_AUTO_SET,
    GreekRequest,
    normalize_greeks,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero

#: Keys of the incumbent base dict (calculate_analytical_greeks(greeks=None)),
#: in the dict's insertion order. Result-dict order is part of the contract
#: (downstream DataFrame builders copy it into column order), so requested
#: subsets are emitted in this order, never in set-iteration order.
BASE_ORDER = (
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
)
BASE_GREEKS = frozenset(BASE_ORDER)

#: Higher-order names with closed forms (beyond the incumbent base dict), in
#: definition order; they follow the base keys in the result dict.
EXTENDED_ORDER = (
    "vanna",
    "volga",
    "speed",
    "zomma",
    "charm",
    "color",
    "vega_theta",
    "gamma_theta",
    "delta_q",
    "dividend_volga",
)
EXTENDED_GREEKS = frozenset(EXTENDED_ORDER)

#: Extended names that are per-day time derivatives and therefore carry a
#: clock (bare / _1d / _1td) on the analytical route.
TIME_GREEKS = frozenset({"charm", "color", "vega_theta", "gamma_theta"})

_ORDER_INDEX = {name: i for i, name in enumerate(BASE_ORDER + EXTENDED_ORDER)}
_CLOCK_RANK = {None: 0, "1d": 1, "1td": 2}


def supports_request(req: GreekRequest) -> bool:
    """True if ``calculate()`` may auto-route this request entry to the
    closed forms: an analytical-auto name, clock-qualified only for the
    TIME_GREEKS (the base theta family stays per calendar day)."""
    if req.canonical not in ANALYTICAL_AUTO_SET:
        return False
    return req.clock is None or req.canonical in TIME_GREEKS


def calculate_analytical_greeks(
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    price: Optional[float] = None,
    greeks: Optional[Sequence[object]] = None,
    time_bump_mode: str = "auto",
) -> Dict[str, float]:
    """Closed-form BS greeks.

    With ``greeks=None`` this returns exactly the incumbent key set. An
    explicit ``greeks`` list may add the EXTENDED_GREEKS closed forms and
    returns only the requested keys, base names first in the incumbent dict
    order, then extended names in definition order.

    Clock qualifiers are accepted on the TIME_GREEKS only: ``charm_1d`` is
    per calendar day, ``charm_1td`` per trading day (``bus_days_in_year``),
    and bare ``charm`` follows ``time_bump_mode`` resolved against the env
    exactly like the numerical route. The base theta family keeps its
    incumbent per-calendar-day scaling and rejects qualifiers.
    """
    if greeks is None:
        return _base_greeks(product, pricing_env, price)

    requests = normalize_greeks(greeks)
    if requests is None or len(requests) == 0:
        return {}
    for req in requests:
        if req.canonical not in BASE_GREEKS | EXTENDED_GREEKS:
            raise ValidationError(
                f"Analytical greeks do not support: {req.canonical!r}"
            )
        if req.clock is not None and req.canonical not in TIME_GREEKS:
            raise ValidationError(
                "Analytical greeks accept clock qualifiers only on "
                f"{sorted(TIME_GREEKS)} (got {req.key!r}); the base theta "
                "family is per calendar day"
            )
    ordered = sorted(
        requests,
        key=lambda req: (_ORDER_INDEX[req.canonical], _CLOCK_RANK[req.clock]),
    )
    names = {req.canonical for req in requests}

    full = _base_greeks(product, pricing_env, price)
    extended_needed = names & EXTENDED_GREEKS
    raw_extended = (
        _extended_greeks(product, pricing_env, extended_needed)
        if extended_needed
        else {}
    )
    multiplier = product.contract_multiplier
    out: Dict[str, float] = {}
    for req in ordered:
        name = req.canonical
        if name in raw_extended:
            days = (
                bump_envs.time_days_per_year(pricing_env, req.clock, time_bump_mode)
                if name in TIME_GREEKS
                else None
            )
            out[req.key] = _scale_extended(name, raw_extended[name], days, multiplier)
        else:
            # At expiry the legacy base dict omits some zero-valued names
            # (dividend_rho); an explicit request still deserves its 0.0 key.
            out[req.key] = full.get(name, 0.0)
    return out


def _scale_extended(
    name: str, raw: float, days: Optional[float], multiplier: float
) -> float:
    """Apply the per-day clock and contract multiplier to one raw closed
    form, in the incumbent operation order (divide, then multiply)."""
    value = raw
    if days is not None:
        value = value / days
        if name == "vega_theta":
            value = value / 100.0  # vega per 1 vol point
    return value * multiplier


def _extended_greeks(
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    names: Iterable[str],
) -> Dict[str, float]:
    """Raw closed forms for the higher-order greeks (continuous dividend q).

    Returned unscaled: the TIME_GREEKS as annualised rates (the caller
    divides by the clock's days per year; vega_theta is additionally per
    1 vol point), vanna/volga/speed/zomma/delta_q/dividend_volga as raw
    derivatives, and no contract_multiplier. Signs are pinned by the FD
    oracle in test_analytical_higher_order.py.
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
    out["charm"] = charm_annual

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
    out["color"] = -color_annual

    # vega_theta (veta): d(vega per 1 vol pt) per calendar day as the
    # valuation date advances (sign pinned by the FD oracle: vega decays)
    veta_annual = vega_raw * (
        q
        + (r - q) * d1 / (sigma * sqrt_T)
        - (1.0 + d1 * d2) / (2.0 * T)
    )
    out["vega_theta"] = veta_annual

    # gamma_theta: the BS-PDE identity term. The expression is written
    # exactly as _base_greeks writes convexity_theta's term1, so after the
    # caller's /365 on a calendar-day env the two are bitwise equal for
    # vanillas, not merely algebraically equal.
    out["gamma_theta"] = -S * disc_div * phi_d1 * sigma / (2 * sqrt_T)

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

    return {name: out[name] for name in EXTENDED_ORDER if name in names}


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

    # Legacy default key set (no dividend_rho) — the greeks=None result at
    # expiry is frozen by the compatibility contract. An explicit request
    # for a missing base name is filled with 0.0 in
    # calculate_analytical_greeks instead.
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
