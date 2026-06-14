"""Dupire local volatility from a market implied-vol grid (implied-variance / Gatheral form).

With total implied variance w(y,T) = sigma_imp(K,T)^2 * T and log-moneyness
y = ln(K / F_T) (forward F_T from the per-maturity curve, so rates enter only via the
forward):

    sigma_LV^2(K,T) = (dw/dT|_y) / D
    D = 1 - (y/w) w_y + 0.25 (-0.25 - 1/w + y^2/w^2) w_y^2 + 0.5 w_yy
    dw/dT|_y = dw/dT|_K + b(T) w_y,   b(T) = d ln(F_T)/dT

w_y, w_yy are finite-differenced on the smooth w in log-strike (well conditioned, unlike
differentiating call prices). For a flat implied-vol surface (w_y = w_yy = 0, dw/dT = sigma^2)
this returns sigma to machine precision on any well-formed grid. Requires >= 3 strikes and
>= 3 maturities. Admissibility (default ON): calendar dw/dT|_y >= 0 (in moneyness,
rate-agnostic) and butterfly D > 0. Vol flooring is OPT-IN (off by default -> raise on
non-positive local variance).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.numerical import safe_exp, safe_log, safe_sqrt
from quantark.util.numerical.constants import Tolerance
from quantark.volmodels.localvol.surface import LocalVolSurface


def _fd1(arr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """First derivative along last axis (non-uniform central interior; first-order one-sided ends)."""
    d = np.zeros_like(arr)
    dxm = x[1:-1] - x[:-2]
    dxp = x[2:] - x[1:-1]
    d[..., 1:-1] = (
        -arr[..., :-2] * dxp / (dxm * (dxm + dxp))
        + arr[..., 1:-1] * (dxp - dxm) / (dxm * dxp)
        + arr[..., 2:] * dxm / (dxp * (dxm + dxp))
    )
    d[..., 0] = (arr[..., 1] - arr[..., 0]) / (x[1] - x[0])
    d[..., -1] = (arr[..., -1] - arr[..., -2]) / (x[-1] - x[-2])
    return d


def _fd2_interior(arr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Second derivative, non-uniform central on interior; boundary cols copy nearest interior."""
    d2 = np.zeros_like(arr)
    dxm = x[1:-1] - x[:-2]
    dxp = x[2:] - x[1:-1]
    denom = 0.5 * dxm * dxp * (dxm + dxp)
    d2[..., 1:-1] = (
        arr[..., :-2] * dxp - arr[..., 1:-1] * (dxm + dxp) + arr[..., 2:] * dxm
    ) / denom
    d2[..., 0] = d2[..., 1]
    d2[..., -1] = d2[..., -2]
    return d2


def build_dupire_local_vol(
    iv_surface,
    spot: float,
    rate_curve,
    div_yield: Callable[[float], float],
    vol_floor: Optional[float] = None,
    validate_arbitrage: bool = True,
) -> LocalVolSurface:
    """Build a LocalVolSurface from a market GridVolSurface via the implied-variance Dupire form.

    Args:
        iv_surface: GridVolSurface (provides .strikes, .maturities, .iv_grid).
        spot: current spot.
        rate_curve: RateCurve; per-maturity forward defines the moneyness y = ln(K/F_T).
        div_yield: callable T -> continuously-compounded dividend/foreign zero yield.
        vol_floor: OPT-IN floor on local vol (None => no flooring; raise on bad variance).
        validate_arbitrage: reject calendar/butterfly-arbitraging input (default True).

    Returns:
        LocalVolSurface on (time_grid=maturities, strike_grid=strikes).
    """
    K = np.asarray(iv_surface.strikes, dtype=float)
    T = np.asarray(iv_surface.maturities, dtype=float)
    iv = np.asarray(iv_surface.iv_grid, dtype=float)
    nT, nK = iv.shape
    if nK < 3 or nT < 3:
        raise ValidationError("Dupire requires >= 3 strikes and >= 3 maturities")
    if vol_floor is not None and vol_floor <= 0:
        raise ValidationError("vol_floor must be positive when provided")

    ln_k = safe_log(K)                                   # (nK,)
    r_zero = np.array([rate_curve.get_rate(t) for t in T])
    q_zero = np.array([div_yield(t) for t in T])
    carry_zero = r_zero - q_zero                         # zero cost-of-carry to T
    fwd = spot * safe_exp(carry_zero * T)                # forward F_T (nT,)
    ln_fwd = safe_log(fwd)                               # (nT,)

    w = iv ** 2 * T[:, None]                             # total variance (nT, nK)
    y = ln_k[None, :] - ln_fwd[:, None]                  # log-moneyness (nT, nK)

    w_y = _fd1(w, ln_k)                                  # dw/dy == dw/dln K at fixed T
    w_yy = _fd2_interior(w, ln_k)
    dw_dT_K = _fd1(w.T, T).T                             # dw/dT at fixed strike
    # d ln(F_T)/dT via the SAME _fd1 stencil as dw_dT_K, so dw/dT|_y is a consistent
    # discrete operator (b(T) = d ln F/dT); avoids backward/central stencil mismatch.
    dlnF_dT = _fd1(ln_fwd, T)                            # (nT,)
    dw_dT_y = dw_dT_K + dlnF_dT[:, None] * w_y           # dw/dT at fixed moneyness

    denom = (
        1.0
        - (y / w) * w_y
        + 0.25 * (-0.25 - 1.0 / w + (y * y) / (w * w)) * (w_y * w_y)
        + 0.5 * w_yy
    )

    if validate_arbitrage:
        if np.any(dw_dT_y < -Tolerance.PRECISION):
            idx = np.argwhere(dw_dT_y < -Tolerance.PRECISION)
            raise NumericalError(
                f"calendar arbitrage: dw/dT|_y < 0 (moneyness) at nodes {idx.tolist()}"
            )
        if np.any(denom <= Tolerance.PRECISION):
            idx = np.argwhere(denom <= Tolerance.PRECISION)
            raise NumericalError(
                f"butterfly arbitrage: Dupire denominator <= 0 at nodes {idx.tolist()}"
            )

    with np.errstate(divide="ignore", invalid="ignore"):
        lv2 = dw_dT_y / denom

    bad = ~np.isfinite(lv2) | (lv2 <= 0)
    if np.any(bad):
        if vol_floor is None:
            idx = np.argwhere(bad)
            raise NumericalError(
                "Dupire produced non-positive/instable local variance at nodes "
                f"{idx.tolist()}; supply a vol_floor to stabilize or fix the input surface."
            )
        lv2 = np.where(bad, float(vol_floor) ** 2, lv2)
    if vol_floor is not None:
        lv2 = np.maximum(lv2, float(vol_floor) ** 2)

    return LocalVolSurface(strike_grid=K, time_grid=T, lv_grid=safe_sqrt(lv2))
