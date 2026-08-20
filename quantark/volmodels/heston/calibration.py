"""Heston parameter calibration to market option prices or implied vols."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import least_squares, minimize

from quantark.util.exceptions import NumericalError, ValidationError
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
    data_cost: float = 0.0
    feller_penalty_cost: float = 0.0
    temporal_penalty_cost: float = 0.0
    feller_margin: float = 0.0
    enforce_feller: bool = False
    optimizer: str = "least_squares"


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
    enforce_feller: bool = False,
    temporal_reference: Optional[HestonParams] = None,
    temporal_regularization: float = 0.0,
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
        max_nfev, xtol, ftol, gtol: least-squares solver controls.  The hard
            Feller branch maps ``max_nfev`` and ``ftol`` to SLSQP's iteration
            and objective tolerances; ``xtol`` and ``gtol`` retain their
            validation and default-branch meanings.
        enforce_feller: if True, minimize the same half squared-residual
            objective under the nonlinear constraint
            ``2 * kappa * theta >= sigma**2``.  This is a genuine constrained
            solve; ``regularize_feller`` remains the independent soft-penalty
            control used by the objective.
        temporal_reference: optional prior Heston parameter set for temporal
            regularization.  Only the structural parameters ``kappa``,
            ``theta``, ``sigma``, and ``rho`` are penalized; ``v0`` remains a
            daily surface fit.
        temporal_regularization: non-negative weight on squared structural
            parameter changes, normalized by each parameter's calibration
            bound span.  Zero (default) preserves independent calibration.

    Notes:
        In the default least-squares branch, a NumericalError raised by the CF
        pricer at extreme trial parameters propagates by design.  The hard
        constraint branch instead assigns an invalid SLSQP line-search trial a
        large barrier objective; its final parameters are always repriced and
        validated, so an invalid fit is never returned.  Tightening broad
        custom bounds remains the preferred remedy for repeated invalid trials.

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
    if not isinstance(enforce_feller, bool):
        raise ValidationError("enforce_feller must be a bool")
    if not (
        np.isfinite(temporal_regularization)
        and temporal_regularization >= 0.0
    ):
        raise ValidationError(
            "temporal_regularization must be finite and non-negative"
        )
    if temporal_reference is not None and not isinstance(
        temporal_reference, HestonParams
    ):
        raise ValidationError("temporal_reference must be a HestonParams")
    if temporal_regularization > 0.0 and temporal_reference is None:
        raise ValidationError(
            "temporal_reference is required when temporal_regularization > 0"
        )
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
    temporal_reference_x: Optional[np.ndarray] = None
    if temporal_reference is not None:
        temporal_reference_x = np.array(
            [
                temporal_reference.v0,
                temporal_reference.kappa,
                temporal_reference.theta,
                temporal_reference.sigma,
                temporal_reference.rho,
            ],
            dtype=float,
        )
        if np.any(temporal_reference_x < lower) or np.any(
            temporal_reference_x > upper
        ):
            raise ValidationError(
                "temporal_reference parameters must lie within [lower, upper]"
            )
    if enforce_feller:
        max_box_margin = 2.0 * upper[1] * upper[2] - lower[3] * lower[3]
        if max_box_margin <= 0.0:
            raise ValidationError(
                "bounds contain no parameters satisfying the Feller constraint"
            )

    # Group options by maturity: the Heston CF is strike-independent, so all strikes at one
    # maturity share a single phi(u) sweep. resolve() is deterministic in T, so options with
    # equal T share the same resolved rate/carry.
    uniqueT, inv = np.unique(Ts, return_inverse=True)
    groups = []  # (Tg, rate_g, carry_g, member_indices)
    for g in range(uniqueT.size):
        idx = np.nonzero(inv == g)[0]
        groups.append((float(uniqueT[g]), float(rates[idx[0]]), float(carries[idx[0]]), idx))

    def data_residuals(x: np.ndarray) -> np.ndarray:
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
        return (model - y) * np.sqrt(w)

    def feller_margin(x: np.ndarray) -> float:
        return float(2.0 * x[1] * x[2] - x[3] * x[3])

    def residuals(x: np.ndarray) -> np.ndarray:
        res = data_residuals(x)
        # Fixed-length residual: always append the penalty term when enabled (it is 0
        # when Feller is satisfied) so least_squares sees a constant dimension.
        if regularize_feller > 0.0:
            feller_violation = max(0.0, -feller_margin(x))
            res = np.concatenate([res, np.array([math.sqrt(regularize_feller) * feller_violation])])
        if temporal_regularization > 0.0:
            # Normalize by the frozen calibration box so one unit of kappa is
            # not treated as the same move as one unit of theta.  v0 is
            # deliberately excluded: it remains today's variance anchor.
            structural_span = np.where(
                upper[1:] > lower[1:], upper[1:] - lower[1:], 1.0
            )
            temporal_residual = (
                math.sqrt(temporal_regularization)
                * (x[1:] - temporal_reference_x[1:])
                / structural_span
            )
            res = np.concatenate([res, temporal_residual])
        return res

    if enforce_feller:
        # A positive buffer absorbs SLSQP's constraint tolerance while ensuring
        # the returned HestonParams passes the strict public Feller predicate.
        # It is numerical slack, not an economic model margin.
        #
        # The buffer has to SCALE with ``ftol``: SLSQP satisfies its constraints
        # only to roughly its own accuracy tolerance, so a fixed value is large
        # enough for tight tolerances and silently too small for loose ones.
        # Measured slack at the boundary: ~1e-9 at ftol=1e-8 but ~1.6e-7 at
        # ftol=1e-6 — the latter is the frozen ``mo_frozen`` preset, whose fits
        # on real CFFEX MO settlement surfaces failed the strict check below
        # under the previous constant 1e-8.  The 10x factor is headroom over
        # the observed slack, and the check still fails closed if it is ever
        # insufficient.  Economically negligible: at MO scale
        # (2*kappa*theta ~ 0.16) a 1e-5 buffer perturbs the constraint by ~6e-5
        # relative, far under the optimizer's own convergence tolerance.
        hard_buffer = max(1e-8, 10.0 * float(ftol))
        parameter_span = np.maximum(upper - lower, 1.0)

        def objective(x: np.ndarray) -> float:
            try:
                values = residuals(x)
            except NumericalError:
                # SLSQP can probe an extreme yet box-feasible point during a
                # line search.  Such a point has no valid Heston objective;
                # use a smooth, overwhelmingly large barrier so the real
                # constrained optimizer can reject the trial.  The final
                # point is always repriced below, so an invalid result can
                # never be returned as a calibration.
                distance = (np.asarray(x, dtype=float) - x0) / parameter_span
                return float(1e12 + 1e6 * np.dot(distance, distance))
            return float(0.5 * np.dot(values, values))

        def constraint(x: np.ndarray) -> float:
            return feller_margin(x) - hard_buffer

        def constraint_jacobian(x: np.ndarray) -> np.ndarray:
            return np.array([
                0.0, 2.0 * x[2], 2.0 * x[1], -2.0 * x[3], 0.0
            ])

        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=list(zip(lower, upper)),
            constraints={
                "type": "ineq",
                "fun": constraint,
                "jac": constraint_jacobian,
            },
            options={"maxiter": max_nfev, "ftol": ftol, "disp": False},
        )
        fitted_x = np.asarray(res.x, dtype=float)
        margin = feller_margin(fitted_x)
        if not np.all(np.isfinite(fitted_x)) or not math.isfinite(margin):
            raise NumericalError(
                "hard-Feller calibration returned non-finite parameters"
            )
        bound_scale = np.maximum(
            1.0, np.maximum(np.abs(lower), np.abs(upper))
        )
        bound_tolerance = 1e-10 * bound_scale
        if (
            np.any(fitted_x < lower - bound_tolerance)
            or np.any(fitted_x > upper + bound_tolerance)
        ):
            raise NumericalError(
                "hard-Feller calibration returned parameters outside bounds"
            )
        if margin < 0.0:
            raise NumericalError(
                "hard-Feller calibration returned constraint-violating "
                f"parameters: 2*kappa*theta-sigma^2={margin:.6g}"
            )
        success = bool(res.success)
        message = str(res.message)
        nfev = int(res.nfev)
        optimizer = "SLSQP"
    else:
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
        fitted_x = np.asarray(res.x, dtype=float)
        margin = feller_margin(fitted_x)
        success = bool(res.success)
        message = str(res.message)
        nfev = int(res.nfev)
        optimizer = "least_squares"

    fitted_data_residuals = data_residuals(fitted_x)
    data_cost = float(
        0.5 * np.dot(fitted_data_residuals, fitted_data_residuals)
    )
    violation = max(0.0, -margin)
    penalty_cost = float(0.5 * regularize_feller * violation * violation)
    temporal_penalty_cost = 0.0
    if temporal_regularization > 0.0:
        structural_span = np.where(
            upper[1:] > lower[1:], upper[1:] - lower[1:], 1.0
        )
        normalized_change = (
            fitted_x[1:] - temporal_reference_x[1:]
        ) / structural_span
        temporal_penalty_cost = float(
            0.5
            * temporal_regularization
            * np.dot(normalized_change, normalized_change)
        )
    return CalibrationResult(
        params=unpack(fitted_x),
        success=success,
        cost=data_cost + penalty_cost + temporal_penalty_cost,
        message=message,
        nfev=nfev,
        data_cost=data_cost,
        feller_penalty_cost=penalty_cost,
        temporal_penalty_cost=temporal_penalty_cost,
        feller_margin=margin,
        enforce_feller=enforce_feller,
        optimizer=optimizer,
    )
