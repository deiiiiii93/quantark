"""
Developer B Independent Implementation
======================================
Purpose: Verify Developer A's double barrier option engine logic.
Principle: Clarity over performance. NO access to Developer A code.

Formula Source:
- Ikeda & Kuintomo (1992), as reproduced in Haug (2007) Table 4-15.
"""

import math
from scipy.stats import norm


def safe_pow(base, exp):
    """Safe power that returns inf on overflow instead of raising."""
    try:
        return math.pow(base, exp)
    except (OverflowError, ValueError):
        if base > 0:
            if exp > 0:
                return float("inf")
            else:
                return 0.0
        return float("nan")


def price_double_barrier_call_ko(
    S: float,
    K: float,
    L: float,
    U: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    delta1: float = 0.0,
    delta2: float = 0.0,
    max_terms: int = 10,
) -> float:
    """Independent Ikeda-Kuintomo implementation for call up-and-out-down-and-out."""
    b = r - q
    F = U * math.exp(delta1 * T)

    sqrt_t = math.sqrt(T)
    denom = sigma * sqrt_t
    drift = (b + 0.5 * sigma * sigma) * T
    sig2 = sigma * sigma

    asset_sum = 0.0
    strike_sum = 0.0

    for n in range(-max_terms, max_terms + 1):
        U_pow = math.pow(U, n)
        L_pow = math.pow(L, n)
        U2n = U_pow * U_pow
        L2n = L_pow * L_pow
        L2n2 = L2n * L * L

        mu1 = 2.0 * (b - delta2 - n * (delta1 - delta2)) / sig2 + 1.0
        mu2 = 2.0 * n * (delta1 - delta2) / sig2
        mu3 = 2.0 * (b - delta2 + n * (delta1 - delta2)) / sig2 + 1.0

        # Asset weights
        w1 = safe_pow(U_pow / L_pow, mu1) * safe_pow(L / S, mu2)
        w2 = safe_pow(safe_pow(L, n + 1) / (U_pow * S), mu3)

        # d arguments (call uses K and F)
        d1 = (math.log(S * U2n / (K * L2n)) + drift) / denom
        d2 = (math.log(S * U2n / (F * L2n)) + drift) / denom
        d3 = (math.log(L2n2 / (K * S * U2n)) + drift) / denom
        d4 = (math.log(L2n2 / (F * S * U2n)) + drift) / denom

        cdf_ab = norm.cdf(d1) - norm.cdf(d2)
        cdf_cd = norm.cdf(d3) - norm.cdf(d4)

        term_asset = 0.0
        if math.isfinite(w1) and math.isfinite(w2):
            term_asset = w1 * cdf_ab - w2 * cdf_cd
        elif math.isfinite(w1):
            term_asset = w1 * cdf_ab
        elif math.isfinite(w2):
            term_asset = -w2 * cdf_cd
        asset_sum += term_asset

        # Strike weights
        w1_strike = safe_pow(U_pow / L_pow, mu1 - 2.0) * safe_pow(L / S, mu2)
        w2_strike = safe_pow(safe_pow(L, n + 1) / (U_pow * S), mu3 - 2.0)

        cdf_ab_strike = norm.cdf(d1 - denom) - norm.cdf(d2 - denom)
        cdf_cd_strike = norm.cdf(d3 - denom) - norm.cdf(d4 - denom)

        term_strike = 0.0
        if math.isfinite(w1_strike) and math.isfinite(w2_strike):
            term_strike = w1_strike * cdf_ab_strike - w2_strike * cdf_cd_strike
        elif math.isfinite(w1_strike):
            term_strike = w1_strike * cdf_ab_strike
        elif math.isfinite(w2_strike):
            term_strike = -w2_strike * cdf_cd_strike
        strike_sum += term_strike

    df_carry = math.exp((b - r) * T)
    df_riskfree = math.exp(-r * T)

    price = S * df_carry * asset_sum - K * df_riskfree * strike_sum
    return float(price)


def price_double_barrier_put_ko(
    S: float,
    K: float,
    L: float,
    U: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    delta1: float = 0.0,
    delta2: float = 0.0,
    max_terms: int = 10,
) -> float:
    """Independent Ikeda-Kuintomo implementation for put up-and-out-down-and-out."""
    b = r - q
    E = L * math.exp(delta2 * T)

    sqrt_t = math.sqrt(T)
    denom = sigma * sqrt_t
    drift = (b + 0.5 * sigma * sigma) * T
    sig2 = sigma * sigma

    asset_sum = 0.0
    strike_sum = 0.0

    for n in range(-max_terms, max_terms + 1):
        U_pow = math.pow(U, n)
        L_pow = math.pow(L, n)
        U2n = U_pow * U_pow
        L2n = L_pow * L_pow
        L2n2 = L2n * L * L

        mu1 = 2.0 * (b - delta2 - n * (delta1 - delta2)) / sig2 + 1.0
        mu2 = 2.0 * n * (delta1 - delta2) / sig2
        mu3 = 2.0 * (b - delta2 + n * (delta1 - delta2)) / sig2 + 1.0

        # Asset weights
        w1 = safe_pow(U_pow / L_pow, mu1) * safe_pow(L / S, mu2)
        w2 = safe_pow(safe_pow(L, n + 1) / (U_pow * S), mu3)

        # y arguments (put uses E and K)
        y1 = (math.log(S * U2n / (E * L2n)) + drift) / denom
        y2 = (math.log(S * U2n / (K * L2n)) + drift) / denom
        y3 = (math.log(L2n2 / (E * S * U2n)) + drift) / denom
        y4 = (math.log(L2n2 / (K * S * U2n)) + drift) / denom

        cdf_ab = norm.cdf(y1) - norm.cdf(y2)
        cdf_cd = norm.cdf(y3) - norm.cdf(y4)

        term_asset = 0.0
        if math.isfinite(w1) and math.isfinite(w2):
            term_asset = w1 * cdf_ab - w2 * cdf_cd
        elif math.isfinite(w1):
            term_asset = w1 * cdf_ab
        elif math.isfinite(w2):
            term_asset = -w2 * cdf_cd
        asset_sum += term_asset

        # Strike weights
        w1_strike = safe_pow(U_pow / L_pow, mu1 - 2.0) * safe_pow(L / S, mu2)
        w2_strike = safe_pow(safe_pow(L, n + 1) / (U_pow * S), mu3 - 2.0)

        cdf_ab_strike = norm.cdf(y1 - denom) - norm.cdf(y2 - denom)
        cdf_cd_strike = norm.cdf(y3 - denom) - norm.cdf(y4 - denom)

        term_strike = 0.0
        if math.isfinite(w1_strike) and math.isfinite(w2_strike):
            term_strike = w1_strike * cdf_ab_strike - w2_strike * cdf_cd_strike
        elif math.isfinite(w1_strike):
            term_strike = w1_strike * cdf_ab_strike
        elif math.isfinite(w2_strike):
            term_strike = -w2_strike * cdf_cd_strike
        strike_sum += term_strike

    df_carry = math.exp((b - r) * T)
    df_riskfree = math.exp(-r * T)

    price = K * df_riskfree * strike_sum - S * df_carry * asset_sum
    return float(price)


def black_scholes_call(S, K, T, r, q, sigma):
    """Simple Black-Scholes call for parity checks."""
    if T <= 0:
        return max(S - K, 0.0)
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, q, sigma):
    """Simple Black-Scholes put for parity checks."""
    if T <= 0:
        return max(K - S, 0.0)
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
