"""Heston semi-analytical European pricers (Lewis, Gatheral, Weber).

Fourier/characteristic-function pricers for European calls; puts via put-call parity.
Asset-neutral: ``r`` discounts, ``carry`` is the continuous yield (dividend for equity,
foreign rate for FX); forward F = s0 * exp((r - carry) * T). Complex-domain CF algebra
uses numpy directly; real-domain scalars use quantark.util.numerical guards.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate

from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.numerical import safe_exp, safe_log, safe_sqrt

# Below this vol-of-vol the CF integrands are numerically unstable; the exact
# deterministic-variance (sigma -> 0) limit is used instead.
_SIGMA_MIN = 1e-4


def _validate(s0: float, strike: float, T: float, params) -> None:
    if s0 <= 0 or strike <= 0:
        raise ValidationError("s0 and strike must be positive")
    if T <= 0:
        raise ValidationError("maturity must be positive")


def _deterministic_limit_call(s0, r, carry, params, strike, T) -> float:
    """Exact sigma->0 Heston call: BS with the integrated CIR variance.

    v_t = theta + (v0 - theta) e^{-kappa t};  V(T) = int_0^T v_t dt.
    """
    from quantark.volmodels.black_scholes import bs_call_price

    if params.kappa > 0:
        # -expm1(-kT)/k is the stable form of (1 - e^{-kT})/k (-> T as k -> 0).
        integrated = params.theta * T + (params.v0 - params.theta) * (
            -np.expm1(-params.kappa * T) / params.kappa
        )
    else:
        integrated = params.v0 * T
    vol_eff = float(safe_sqrt(max(integrated, 0.0) / T))
    return bs_call_price(s0, strike, T, vol_eff, r, carry)


def _validate_call_noarb(price: float, s0, r, carry, strike, T) -> float:
    """Clamp tiny rounding violations; raise if the CF result is materially
    outside the European call no-arbitrage band (a sign quadrature failed)."""
    if not np.isfinite(price):
        raise NumericalError("Heston CF quadrature produced a non-finite price")
    lower = max(s0 * float(safe_exp(-carry * T)) - strike * float(safe_exp(-r * T)), 0.0)
    upper = s0 * float(safe_exp(-carry * T))
    tol = max(1e-7, 1e-6 * s0)
    if price < lower - tol or price > upper + tol:
        raise NumericalError(
            f"Heston CF price {price} outside no-arbitrage band [{lower}, {upper}] "
            "(extreme parameters or maturity too small for quadrature)"
        )
    return min(max(price, lower), upper)


def _cf_core(beta, d, T, params):
    """Shared Heston CF algebra (F13): given method-specific ``beta`` and ``d``,
    return coefficients ``(A, B)`` with CF exponent ``A * params.theta + B * params.v0``.

    F13 sketched ``_cf_core(u_complex, b0, ...)``, but the ``d**2`` discriminant is
    method-specific (Lewis ``d**2 = beta**2 + v*k_c*(k_c - i)``, Gatheral
    ``d**2 = beta**2 - 2 a v``, Weber ``d**2 = beta**2 - v*u*(s*i - u)``). So each method
    forms its own ``beta``/``d`` and passes them in; this owns exactly the shared
    ``g / e^{-dT} / log((1 - g e^{-dT})/(1 - g))`` block that was duplicated. One canonical
    operation order ⇒ each method's integrand is reproduced to <=1e-13 relative
    (Weber/Gatheral reorder the ``/v`` division), pinned in test_heston_cf_core.py.

    ``beta``, ``d`` may be complex scalars or ``np.ndarray`` (vectorized over the
    integration variable); all operations are elementwise.
    """
    v = params.sigma ** 2
    g = (beta - d) / (beta + d)
    q = np.exp(-d * T)
    t_m = (beta - d) / v
    one_minus_gq = 1.0 - g * q
    B = t_m * (1.0 - q) / one_minus_gq
    A = params.kappa * (T * t_m - (2.0 / v) * np.log(one_minus_gq / (1.0 - g)))
    return A, B


def price_european_lewis(s0, r, carry, params, strike, T) -> float:
    """European CALL via Lewis (2000) single-integral formulation."""
    _validate(s0, strike, T, params)
    if params.sigma < _SIGMA_MIN:
        return _deterministic_limit_call(s0, r, carry, params, strike, T)
    k = float(strike)
    f = s0 * float(safe_exp((r - carry) * T))
    v = params.sigma ** 2

    y_lm = float(safe_log(f / k))                    # hoisted: constant across u

    def phi(k_in: float) -> complex:
        k_complex = k_in + 0.5j
        b = params.kappa + 1j * params.rho * params.sigma * k_complex
        d = np.sqrt(b ** 2 + v * k_complex * (k_complex - 1j))
        A, B = _cf_core(b, d, T, params)
        return np.exp(A * params.theta + B * params.v0)

    def integrand(u: float) -> float:
        return 2.0 * np.real(np.exp(-1j * u * y_lm) * phi(u)) / (u ** 2 + 0.25)

    integral, _ = integrate.quad(integrand, 0.0, np.inf, limit=500)
    i1 = integral / (2.0 * np.pi)
    price = float(f * np.exp(-r * T) - np.sqrt(k * f) * np.exp(-r * T) * i1)
    return _validate_call_noarb(price, s0, r, carry, strike, T)


def price_european_gatheral(s0, r, carry, params, strike, T) -> float:
    """European CALL via Gatheral (2006) two-probability formulation."""
    _validate(s0, strike, T, params)
    if params.sigma < _SIGMA_MIN:
        return _deterministic_limit_call(s0, r, carry, params, strike, T)
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
            # Gatheral d**2 = b**2 - 2 a v; r_minus = (b-d)/v, rr = r_minus/r_plus = g_core.
            A, B = _cf_core(b, d, T, params)
            phi = np.exp(A * params.theta + B * params.v0 + 1j * u * x0) / (1j * u)
            return np.real(phi)

        integral, _ = integrate.quad(integrand, 0.0, np.inf, limit=500)
        return 0.5 + integral / np.pi

    p1 = compute_prob(1)
    p2 = compute_prob(0)
    price = float(s0 * np.exp(-carry * T) * p1 - k * np.exp(-r * T) * p2)
    return _validate_call_noarb(price, s0, r, carry, strike, T)


def price_european_weber(s0, r, carry, params, strike, T) -> float:
    """European CALL via the Weber formulation."""
    _validate(s0, strike, T, params)
    if params.sigma < _SIGMA_MIN:
        return _deterministic_limit_call(s0, r, carry, params, strike, T)
    k = float(strike)
    v = params.sigma ** 2

    y_w = float(safe_log(s0 / (k * np.exp(-(r - carry) * T))))    # hoisted: constant across u

    def compute_f(s: float, b0: float) -> float:
        def integrand(u: float) -> float:
            beta = b0 - 1j * params.rho * params.sigma * u
            d = np.sqrt(beta ** 2 - v * u * (s * 1j - u))
            A, B = _cf_core(beta, d, T, params)
            phi = np.exp(A * params.theta + B * params.v0 + 1j * u * y_w) / (u * 1j)
            return np.real(phi)

        integral, _ = integrate.quad(integrand, 0.0, np.inf, limit=500)
        return 0.5 + integral / np.pi

    price = float(
        s0 * np.exp(-carry * T) * compute_f(1.0, params.kappa - params.rho * params.sigma)
        - np.exp(-r * T) * k * compute_f(-1.0, params.kappa)
    )
    return _validate_call_noarb(price, s0, r, carry, strike, T)


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
