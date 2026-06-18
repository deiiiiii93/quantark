"""Black-Scholes-Merton closed-form helpers as free functions (forward F = S e^{(r-q)T})."""

from __future__ import annotations

from scipy.stats import norm

from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.numerical import safe_exp, safe_log, safe_sqrt
from quantark.util.numerical.constants import Tolerance


def _validate(s: float, k: float, t: float, sigma: float) -> None:
    if s <= 0 or k <= 0:
        raise ValidationError("spot and strike must be positive")
    if t < 0:
        raise ValidationError(f"maturity must be non-negative, got {t}")
    if sigma < 0:
        raise ValidationError(f"volatility must be non-negative, got {sigma}")


def _d1_d2(s: float, k: float, t: float, sigma: float, r: float, q: float):
    vol_sqrt_t = sigma * safe_sqrt(t)
    d1 = (safe_log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_call_price(s: float, k: float, t: float, sigma: float, r: float, q: float) -> float:
    """European call price under Black-Scholes with continuous yield q."""
    _validate(s, k, t, sigma)
    if t <= Tolerance.ZERO or sigma <= Tolerance.ZERO:
        return max(s * safe_exp(-q * t) - k * safe_exp(-r * t), 0.0)
    d1, d2 = _d1_d2(s, k, t, sigma, r, q)
    return float(s * safe_exp(-q * t) * norm.cdf(d1) - k * safe_exp(-r * t) * norm.cdf(d2))


def bs_put_price(s: float, k: float, t: float, sigma: float, r: float, q: float) -> float:
    """European put price via put-call parity."""
    return float(bs_call_price(s, k, t, sigma, r, q) - s * safe_exp(-q * t) + k * safe_exp(-r * t))


def bs_vega(s: float, k: float, t: float, sigma: float, r: float, q: float) -> float:
    """Black-Scholes vega dC/dsigma (per unit vol)."""
    _validate(s, k, t, sigma)
    if t <= Tolerance.ZERO or sigma <= Tolerance.ZERO:
        return 0.0
    d1, _ = _d1_d2(s, k, t, sigma, r, q)
    return float(s * safe_exp(-q * t) * norm.pdf(d1) * safe_sqrt(t))


def implied_vol_call(
    s: float, k: float, t: float, price: float, r: float, q: float,
    tol: float = Tolerance.PRECISION, max_iter: int = 100, max_vol: float = 10.0,
) -> float:
    """Invert the BS call price for implied vol via Brent's method.

    - A price at (or just below) the intrinsic lower bound implies zero vol -> returns 0.0.
    - A price at (or above) the upper bound S e^{-qT} has unbounded implied vol -> raises.
    - Zero/negative maturity is not invertible -> raises.

    Raises NumericalError outside the no-arbitrage band or if bracketing fails.
    """
    from scipy.optimize import brentq

    _validate(s, k, t, 0.0)
    if t <= Tolerance.ZERO:
        raise NumericalError("cannot invert implied vol at zero maturity")

    intrinsic = max(s * safe_exp(-q * t) - k * safe_exp(-r * t), 0.0)
    upper = s * safe_exp(-q * t)
    if price < intrinsic - tol or price > upper + tol:
        raise NumericalError(f"call price {price} outside no-arbitrage [{intrinsic}, {upper}]")
    if price <= intrinsic + tol:
        return 0.0
    if price >= upper - tol:
        raise NumericalError(f"call price {price} at upper bound {upper}: implied vol unbounded")

    def objective(sigma: float) -> float:
        return bs_call_price(s, k, t, sigma, r, q) - price

    lo, hi = 1e-9, float(max_vol)
    # Expand the upper bracket if the price implies a very high vol.
    while objective(hi) < 0.0 and hi < 1e3:
        hi *= 2.0
    try:
        return float(brentq(objective, lo, hi, xtol=tol, maxiter=max_iter))
    except ValueError as exc:  # pragma: no cover
        raise NumericalError(f"implied vol bracketing failed: {exc}") from exc
