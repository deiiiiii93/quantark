"""Heston parameter calibration to market option prices or implied vols."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares

from quantark.util.exceptions import ValidationError
from quantark.volmodels.black_scholes import bs_call_price, implied_vol_call
from quantark.volmodels.heston.analytical_kernel import (
    heston_call_price,
    heston_call_prices_vectorized,
    heston_implied_vol,
)
from quantark.volmodels.heston.params import HestonParams


@dataclass
class MarketOption:
    """A market call quote for calibration (price or iv must be set)."""

    K: float
    T: float
    price: Optional[float] = None
    iv: Optional[float] = None
    weight: float = 1.0


@dataclass
class CalibrationResult:
    params: HestonParams
    success: bool
    cost: float
    message: str
    nfev: int


def calibrate_heston(
    s0: float,
    options: Sequence[MarketOption],
    r: Union[float, Callable[[float], float]],
    carry: Union[float, Callable[[float], float]],
    initial: HestonParams,
    bounds: Tuple[
        Tuple[float, float, float, float, float],
        Tuple[float, float, float, float, float],
    ] = (
        (1e-8, 1e-6, 1e-6, 1e-6, -0.999),
        (5.0, 50.0, 5.0, 5.0, 0.999),
    ),
    target: str = "price",
    regularize_feller: float = 1e-4,
    method: str = "lewis",
    max_nfev: int = 200,
    xtol: float = 1e-6,
    ftol: float = 1e-6,
    gtol: float = 1e-6,
) -> CalibrationResult:
    """Calibrate (v0, kappa, theta, sigma, rho) to market options via least squares.

    Args:
        s0: spot. options: market call quotes. r: discount rate. carry: continuous yield
        (dividend for equity, foreign rate for FX). initial: starting guess.
        bounds: (lower, upper) parameter bounds. target: "price" or "iv".
        regularize_feller: weight of the max(0, sigma^2 - 2 kappa theta) penalty.
        method: CF pricer for the objective. "lewis" (default) uses the strike-vectorized
            fixed-quadrature pricer (heston_call_prices_vectorized) — one CF sweep per
            maturity, the fast path. "gatheral"/"weber" use the per-option adaptive pricer
            (the pre-WS-B1 objective), preserved for compatibility; they are ~10x+ slower
            but numerically agree with lewis to ~1e-10. The default changed from "gatheral"
            to "lewis" in WS-B1 to deliver the speedup by default.
        max_nfev, xtol, ftol, gtol: least-squares solver controls.

    Notes:
        A NumericalError raised by the CF pricer at extreme trial parameters propagates by
        design (there is no fallback residual). If calibration aborts this way, tighten the
        bounds so least_squares does not probe those parameters. The fixed-quadrature
        vectorized pricer with the no-arbitrage clamp makes this substantially rarer than
        the old adaptive-quad path (quad tail failures were the main source).

    Raises:
        ValidationError: on bad inputs, inverted bounds, or an initial guess outside
            [lower, upper].
    """
    if target not in ("price", "iv"):
        raise ValidationError("target must be 'price' or 'iv'")
    if method not in ("lewis", "gatheral", "weber"):
        raise ValidationError("method must be 'lewis', 'gatheral', or 'weber'")
    if not options:
        raise ValidationError("at least one market option is required")
    if not (np.isfinite(regularize_feller) and regularize_feller >= 0.0):
        raise ValidationError("regularize_feller must be finite and non-negative")
    if not (np.isfinite(s0) and s0 > 0):
        raise ValidationError("s0 must be finite and positive")
    def resolve(value, t: float, name: str) -> float:
        resolved = value(t) if callable(value) else value
        if not np.isfinite(resolved):
            raise ValidationError(f"{name} must return finite values")
        return float(resolved)

    if max_nfev <= 0:
        raise ValidationError("max_nfev must be positive")
    for name, value in (("xtol", xtol), ("ftol", ftol), ("gtol", gtol)):
        if not np.isfinite(value) or value <= 0:
            raise ValidationError(f"{name} must be finite and positive")
    for opt in options:
        if opt.price is None and opt.iv is None:
            raise ValidationError("each MarketOption must set price or iv")
        if opt.price is not None and not np.isfinite(opt.price):
            raise ValidationError("MarketOption price must be finite")
        if opt.iv is not None and not (np.isfinite(opt.iv) and opt.iv > 0):
            raise ValidationError("MarketOption iv must be finite and positive")
        if not (np.isfinite(opt.K) and opt.K > 0 and np.isfinite(opt.T) and opt.T > 0):
            raise ValidationError("MarketOption K and T must be finite and positive")
        if not np.isfinite(opt.weight) or opt.weight < 0:
            raise ValidationError("MarketOption weight must be finite and non-negative")

    Ks = np.array([opt.K for opt in options], dtype=float)
    Ts = np.array([opt.T for opt in options], dtype=float)
    w = np.array([opt.weight for opt in options], dtype=float)
    rates = np.array([resolve(r, opt.T, "r") for opt in options], dtype=float)
    carries = np.array([resolve(carry, opt.T, "carry") for opt in options], dtype=float)

    if target == "price":
        y = np.array([
            opt.price if opt.price is not None
            else bs_call_price(s0, opt.K, opt.T, opt.iv, rate_i, carry_i)
            for opt, rate_i, carry_i in zip(options, rates, carries)
        ])
    else:
        y = np.array([
            opt.iv if opt.iv is not None
            else implied_vol_call(s0, opt.K, opt.T, opt.price, rate_i, carry_i)
            for opt, rate_i, carry_i in zip(options, rates, carries)
        ])

    def unpack(x: np.ndarray) -> HestonParams:
        return HestonParams(v0=float(x[0]), kappa=float(x[1]), theta=float(x[2]),
                            sigma=float(x[3]), rho=float(x[4]))

    x0 = np.array([initial.v0, initial.kappa, initial.theta, initial.sigma, initial.rho], dtype=float)
    lower = np.array(bounds[0], dtype=float)
    upper = np.array(bounds[1], dtype=float)
    # Validate bounds/initial up front with a clear error (else scipy raises a raw
    # "x0 is infeasible" ValueError deep in least_squares).
    if np.any(lower > upper):
        raise ValidationError("each lower bound must not exceed its upper bound")
    if np.any(x0 < lower) or np.any(x0 > upper):
        raise ValidationError(
            "initial parameters must lie within [lower, upper]; tighten the initial "
            "guess or widen the bounds"
        )

    # Group options by maturity: the Heston CF is strike-independent, so all strikes at one
    # maturity share a single phi(u) sweep. resolve() is deterministic in T, so options with
    # equal T share the same resolved rate/carry.
    uniqueT, inv = np.unique(Ts, return_inverse=True)
    groups = []  # (Tg, rate_g, carry_g, member_indices)
    for g in range(uniqueT.size):
        idx = np.nonzero(inv == g)[0]
        groups.append((float(uniqueT[g]), float(rates[idx[0]]), float(carries[idx[0]]), idx))

    def residuals(x: np.ndarray) -> np.ndarray:
        p = unpack(x)
        model = np.empty(Ks.shape[0], dtype=float)
        if method == "lewis":
            # Fast path: one strike-vectorized Lewis CF sweep per maturity group.
            for Tg, rate_g, carry_g, idx in groups:
                call_prices = heston_call_prices_vectorized(s0, Ks[idx], Tg, p, rate_g, carry_g)
                if target == "price":
                    model[idx] = call_prices
                else:  # invert each group call to Black-Scholes implied vol
                    model[idx] = [
                        implied_vol_call(s0, float(k), Tg, float(cp), rate_g, carry_g)
                        for k, cp in zip(Ks[idx], call_prices)
                    ]
        else:
            # Compatibility path: per-option adaptive pricer with the requested CF method
            # (gatheral/weber) — exactly the pre-WS-B1 objective.
            if target == "price":
                model = np.array([
                    heston_call_price(s0, K, T, p, rate_i, carry_i, method=method)
                    for K, T, rate_i, carry_i in zip(Ks, Ts, rates, carries)
                ])
            else:
                model = np.array([
                    heston_implied_vol(s0, K, T, p, rate_i, carry_i, method=method)
                    for K, T, rate_i, carry_i in zip(Ks, Ts, rates, carries)
                ])
        res = (model - y) * np.sqrt(w)
        # Fixed-length residual: always append the penalty term when enabled (it is 0
        # when Feller is satisfied) so least_squares sees a constant dimension.
        if regularize_feller > 0.0:
            feller_violation = max(0.0, p.sigma * p.sigma - 2.0 * p.kappa * p.theta)
            res = np.concatenate([res, np.array([math.sqrt(regularize_feller) * feller_violation])])
        return res

    res = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        max_nfev=max_nfev,
        xtol=xtol,
        ftol=ftol,
        gtol=gtol,
        verbose=0,
    )
    return CalibrationResult(params=unpack(res.x), success=bool(res.success),
                             cost=float(res.cost), message=str(res.message), nfev=int(res.nfev))
