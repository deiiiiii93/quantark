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
from quantark.util.numerical import fd1_nonuniform, fd2_nonuniform, safe_exp, safe_log, safe_sqrt
from quantark.util.numerical.constants import Tolerance
from quantark.volmodels.localvol.surface import LocalVolSurface

# Absolute floor on total implied variance w = iv^2 * T. Deliberately NOT derived from
# Tolerance.ZERO (1e-10): legitimate short-dated low-vol rows (iv=0.03, T=1e-4 -> w=9e-8)
# must pass, while genuinely degenerate rows must raise instead of silently bypassing
# the butterfly check (safe_divide's 0-fallback pointed the failure the wrong way).
_W_FLOOR = 1e-12

# Appended to an arbitrage-rejection message when a boundary node is implicated, so the
# user knows the one-sided edge stencil (not necessarily true arbitrage) may be the cause.
_EDGE_HINT = ("; boundary node — one-sided stencil, consider extending the input grid "
              "one strike/maturity")


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
    if not np.isfinite(spot) or spot <= 0:
        raise ValidationError(f"spot must be positive and finite, got {spot}")

    ln_k = np.asarray(safe_log(K), dtype=float)          # (nK,)
    r_zero = np.array([rate_curve.get_rate(t) for t in T])
    q_zero = np.array([div_yield(t) for t in T])
    if not (np.all(np.isfinite(r_zero)) and np.all(np.isfinite(q_zero))):
        raise ValidationError("rate_curve / div_yield returned non-finite values")
    carry_zero = r_zero - q_zero                         # zero cost-of-carry to T
    fwd = np.asarray(spot * safe_exp(carry_zero * T), dtype=float)   # forward F_T (nT,)
    ln_fwd = np.asarray(safe_log(fwd), dtype=float)      # (nT,)

    w = iv ** 2 * T[:, None]                             # total variance (nT, nK)
    y = ln_k[None, :] - ln_fwd[:, None]                  # log-moneyness (nT, nK)

    w_y = fd1_nonuniform(w, ln_k)                                  # dw/dy == dw/dln K at fixed T
    w_yy = fd2_nonuniform(w, ln_k)
    dw_dT_K = fd1_nonuniform(w.T, T).T                             # dw/dT at fixed strike
    # d ln(F_T)/dT via the SAME fd1_nonuniform stencil as dw_dT_K, so dw/dT|_y is a consistent
    # discrete operator (b(T) = d ln F/dT); avoids backward/central stencil mismatch.
    dlnF_dT = fd1_nonuniform(ln_fwd, T)                            # (nT,)
    dw_dT_y = dw_dT_K + dlnF_dT[:, None] * w_y           # dw/dT at fixed moneyness

    if np.any(w <= _W_FLOOR):
        idx = np.argwhere(w <= _W_FLOOR)
        raise NumericalError(
            f"total implied variance w <= {_W_FLOOR} at (T, K) nodes {idx.tolist()}; "
            "the input surface has a degenerate row — fix the input, do not floor"
        )
    inv_w = 1.0 / w
    denom = (
        1.0
        - y * inv_w * w_y
        + 0.25 * (-0.25 - inv_w + (y * y) * inv_w * inv_w) * (w_y * w_y)
        + 0.5 * w_yy
    )

    if validate_arbitrage:
        # Boundary nodes (first/last maturity row, first/last strike column) use one-sided
        # FD stencils, which can manufacture tiny spurious arbitrage violations on an
        # otherwise-admissible surface. Give those nodes 10x the interior tolerance (F3);
        # interior nodes keep the exact check. The rejection message hints at the cause.
        is_edge = np.zeros((nT, nK), dtype=bool)
        is_edge[0, :] = is_edge[-1, :] = True     # first/last maturity: one-sided dw/dT
        is_edge[:, 0] = is_edge[:, -1] = True      # first/last strike: one-sided w_y/w_yy

        cal_thresh = np.where(is_edge, 10.0 * Tolerance.PRECISION, Tolerance.PRECISION)
        cal_bad = dw_dT_y < -cal_thresh
        if np.any(cal_bad):
            idx = np.argwhere(cal_bad)
            hint = _EDGE_HINT if np.any(is_edge & cal_bad) else ""
            raise NumericalError(
                f"calendar arbitrage: dw/dT|_y < 0 (moneyness) at nodes {idx.tolist()}{hint}"
            )
        # Butterfly no-arbitrage requires a non-negative denominator. A strictly negative
        # denominator is arbitrage; a tiny non-negative denominator is a numerical-stability
        # concern handled by the lv2 check below, not an arbitrage rejection. Interior nodes
        # reject at denom < 0; boundary nodes get 10*PRECISION slack for one-sided stencils.
        bf_thresh = np.where(is_edge, 10.0 * Tolerance.PRECISION, 0.0)
        bf_bad = denom < -bf_thresh
        if np.any(bf_bad):
            idx = np.argwhere(bf_bad)
            hint = _EDGE_HINT if np.any(is_edge & bf_bad) else ""
            raise NumericalError(
                f"butterfly arbitrage: Dupire denominator < 0 at nodes {idx.tolist()}{hint}"
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
