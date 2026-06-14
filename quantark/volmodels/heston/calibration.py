"""Heston parameter calibration to market option prices or implied vols."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from quantark.util.exceptions import ValidationError
from quantark.volmodels.black_scholes import bs_call_price, implied_vol_call
from quantark.volmodels.heston.analytical_kernel import heston_call_price, heston_implied_vol
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
    r: float,
    carry: float,
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
    method: str = "gatheral",
) -> CalibrationResult:
    """Calibrate (v0, kappa, theta, sigma, rho) to market options via least squares.

    Args:
        s0: spot. options: market call quotes. r: discount rate. carry: continuous yield
        (dividend for equity, foreign rate for FX). initial: starting guess.
        bounds: (lower, upper) parameter bounds. target: "price" or "iv".
        regularize_feller: weight of the max(0, sigma^2 - 2 kappa theta) penalty.
        method: analytical CF method used inside the objective.
    """
    if target not in ("price", "iv"):
        raise ValidationError("target must be 'price' or 'iv'")
    if not options:
        raise ValidationError("at least one market option is required")
    if regularize_feller < 0.0:
        raise ValidationError("regularize_feller must be non-negative")
    for opt in options:
        if opt.price is None and opt.iv is None:
            raise ValidationError("each MarketOption must set price or iv")
        if not (np.isfinite(opt.K) and opt.K > 0 and np.isfinite(opt.T) and opt.T > 0):
            raise ValidationError("MarketOption K and T must be finite and positive")
        if not np.isfinite(opt.weight) or opt.weight < 0:
            raise ValidationError("MarketOption weight must be finite and non-negative")

    Ks = np.array([opt.K for opt in options], dtype=float)
    Ts = np.array([opt.T for opt in options], dtype=float)
    w = np.array([opt.weight for opt in options], dtype=float)

    if target == "price":
        y = np.array([
            opt.price if opt.price is not None
            else bs_call_price(s0, opt.K, opt.T, opt.iv if opt.iv is not None else 0.2, r, carry)
            for opt in options
        ])
    else:
        y = np.array([
            opt.iv if opt.iv is not None
            else implied_vol_call(s0, opt.K, opt.T, opt.price if opt.price is not None else 0.0, r, carry)
            for opt in options
        ])

    def unpack(x: np.ndarray) -> HestonParams:
        return HestonParams(v0=float(x[0]), kappa=float(x[1]), theta=float(x[2]),
                            sigma=float(x[3]), rho=float(x[4]))

    x0 = np.array([initial.v0, initial.kappa, initial.theta, initial.sigma, initial.rho], dtype=float)
    lower = np.array(bounds[0], dtype=float)
    upper = np.array(bounds[1], dtype=float)

    def residuals(x: np.ndarray) -> np.ndarray:
        p = unpack(x)
        if target == "price":
            model = np.array([heston_call_price(s0, K, T, p, r, carry, method=method)
                              for K, T in zip(Ks, Ts)])
        else:
            model = np.array([heston_implied_vol(s0, K, T, p, r, carry, method=method)
                              for K, T in zip(Ks, Ts)])
        res = (model - y) * np.sqrt(w)
        # Fixed-length residual: always append the penalty term when enabled (it is 0
        # when Feller is satisfied) so least_squares sees a constant dimension.
        if regularize_feller > 0.0:
            feller_violation = max(0.0, p.sigma * p.sigma - 2.0 * p.kappa * p.theta)
            res = np.concatenate([res, np.array([math.sqrt(regularize_feller) * feller_violation])])
        return res

    res = least_squares(residuals, x0, bounds=(lower, upper), max_nfev=200,
                        xtol=1e-6, ftol=1e-6, gtol=1e-6, verbose=0)
    return CalibrationResult(params=unpack(res.x), success=bool(res.success),
                             cost=float(res.cost), message=str(res.message), nfev=int(res.nfev))
