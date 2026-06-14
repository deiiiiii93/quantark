"""Heston semi-analytical European pricers (Lewis, Gatheral, Weber).

Fourier/characteristic-function pricers for European calls; puts via put-call parity.
Asset-neutral: ``r`` discounts, ``carry`` is the continuous yield (dividend for equity,
foreign rate for FX); forward F = s0 * exp((r - carry) * T). Complex-domain CF algebra
uses numpy directly; real-domain scalars use quantark.util.numerical guards.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_exp, safe_log


def _validate(s0: float, strike: float, T: float, params) -> None:
    if s0 <= 0 or strike <= 0:
        raise ValidationError("s0 and strike must be positive")
    if T <= 0:
        raise ValidationError("maturity must be positive")


def price_european_lewis(s0, r, carry, params, strike, T) -> float:
    """European CALL via Lewis (2000) single-integral formulation."""
    _validate(s0, strike, T, params)
    k = float(strike)
    f = s0 * float(safe_exp((r - carry) * T))
    v = params.sigma ** 2

    def phi(k_in: float) -> complex:
        k_complex = k_in + 0.5j
        b = params.kappa + 1j * params.rho * params.sigma * k_complex
        d = np.sqrt(b ** 2 + v * k_complex * (k_complex - 1j))
        g = (b - d) / (b + d)
        q_exp = np.exp(-d * T)
        t_m = (b - d) / v
        tt = t_m * (1.0 - q_exp) / (1.0 - g * q_exp)
        w = params.kappa * params.theta * (T * t_m - 2.0 * np.log((1.0 - g * q_exp) / (1.0 - g)) / v)
        return np.exp(w + params.v0 * tt)

    def integrand(u: float) -> float:
        return 2.0 * np.real(np.exp(-1j * u * np.log(f / k)) * phi(u)) / (u ** 2 + 0.25)

    integral, _ = integrate.quad(integrand, 0.0, np.inf, limit=500)
    i1 = integral / (2.0 * np.pi)
    return float(f * np.exp(-r * T) - np.sqrt(k * f) * np.exp(-r * T) * i1)


def price_european_gatheral(s0, r, carry, params, strike, T) -> float:
    """European CALL via Gatheral (2006) two-probability formulation."""
    _validate(s0, strike, T, params)
    k = float(strike)
    f = s0 * float(safe_exp((r - carry) * T))
    x0 = float(safe_log(f / k))
    v = params.sigma ** 2

    def compute_prob(j: int) -> float:
        def integrand(u: float) -> float:
            a = -u * u / 2.0 - 1j * u / 2.0 + 1j * j * u
            b = params.kappa - params.rho * params.sigma * j - params.rho * params.sigma * 1j * u
            g = v / 2.0
            d = np.sqrt(b ** 2 - 4.0 * a * g)
            r_plus = (b + d) / (2.0 * g)
            r_minus = (b - d) / (2.0 * g)
            rr = r_minus / r_plus
            q_exp = np.exp(-d * T)
            dd = r_minus * (1.0 - q_exp) / (1.0 - rr * q_exp)
            cc = params.kappa * (r_minus * T - (2.0 / v) * np.log((1.0 - rr * q_exp) / (1.0 - rr)))
            phi = np.exp(cc * params.theta + dd * params.v0 + 1j * u * x0) / (1j * u)
            return np.real(phi)

        integral, _ = integrate.quad(integrand, 0.0, np.inf, limit=500)
        return 0.5 + integral / np.pi

    p1 = compute_prob(1)
    p2 = compute_prob(0)
    return float(s0 * np.exp(-carry * T) * p1 - k * np.exp(-r * T) * p2)


def price_european_weber(s0, r, carry, params, strike, T) -> float:
    """European CALL via the Weber formulation."""
    _validate(s0, strike, T, params)
    k = float(strike)
    v = params.sigma ** 2

    def compute_f(s: float, b0: float) -> float:
        def integrand(u: float) -> float:
            beta = b0 - 1j * params.rho * params.sigma * u
            d = np.sqrt(beta ** 2 - v * u * (s * 1j - u))
            g = (beta - d) / (beta + d)
            q_exp = np.exp(-d * T)
            bb = (beta - d) * (1.0 - q_exp) / (1.0 - g * q_exp) / v
            aa = params.kappa * ((beta - d) * T - 2.0 * np.log((1.0 - g * q_exp) / (1.0 - g))) / v
            phi = np.exp(
                aa * params.theta + bb * params.v0
                + 1j * u * np.log(s0 / (k * np.exp(-(r - carry) * T)))
            ) / (u * 1j)
            return np.real(phi)

        integral, _ = integrate.quad(integrand, 0.0, np.inf, limit=500)
        return 0.5 + integral / np.pi

    return float(
        s0 * np.exp(-carry * T) * compute_f(1.0, params.kappa - params.rho * params.sigma)
        - np.exp(-r * T) * k * compute_f(-1.0, params.kappa)
    )


_METHODS = {
    "lewis": price_european_lewis,
    "gatheral": price_european_gatheral,
    "weber": price_european_weber,
}


def heston_call_price(s0, strike, T, params, r, carry, method: str = "gatheral") -> float:
    """European call price using the named CF method (default Gatheral)."""
    fn = _METHODS.get(method)
    if fn is None:
        raise ValidationError(f"unknown analytical method: {method}")
    return fn(s0, r, carry, params, strike, T)


def heston_put_price(s0, strike, T, params, r, carry, method: str = "gatheral") -> float:
    """European put via put-call parity: P = C - s0 e^{-carry T} + K e^{-r T}."""
    call = heston_call_price(s0, strike, T, params, r, carry, method=method)
    return float(call - s0 * np.exp(-carry * T) + strike * np.exp(-r * T))


def heston_implied_vol(s0, strike, T, params, r, carry, method: str = "gatheral") -> float:
    """Black-Scholes implied vol of the Heston call price."""
    from quantark.volmodels.black_scholes import implied_vol_call

    price = heston_call_price(s0, strike, T, params, r, carry, method=method)
    return implied_vol_call(s0, strike, T, price, r, carry)
