from __future__ import annotations

import numpy as np
from scipy.stats import ncx2

from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston.params import HestonParams


def concentrated_grid(lo: float, hi: float, center: float, n: int, concentration: float) -> np.ndarray:
    """n nodes on [lo, hi] concentrated around ``center`` via a sinh map (Tavella-Randall style).

    Smaller ``concentration`` => tighter clustering near ``center``. Endpoints are exactly lo, hi.
    """
    if not (lo < hi):
        raise ValidationError("require lo < hi")
    if n < 3:
        raise ValidationError("require n >= 3")
    if concentration <= 0:
        raise ValidationError("concentration must be positive")
    c = min(max(center, lo), hi)
    a_lo = np.arcsinh((lo - c) / concentration)
    a_hi = np.arcsinh((hi - c) / concentration)
    xi = np.linspace(0.0, 1.0, n)
    g = c + concentration * np.sinh(a_lo + xi * (a_hi - a_lo))
    g[0], g[-1] = lo, hi          # pin endpoints exactly
    if np.any(np.diff(g) <= 0):
        raise ValidationError("non-monotone grid; widen concentration")
    return g


def trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    """Composite-trapezoid quadrature weights for a (possibly non-uniform) 1D grid."""
    g = np.asarray(grid, dtype=float)
    n = g.size
    if n < 2:
        raise ValidationError("need >= 2 nodes for quadrature weights")
    w = np.empty(n)
    w[0] = 0.5 * (g[1] - g[0])
    w[-1] = 0.5 * (g[-1] - g[-2])
    w[1:-1] = 0.5 * (g[2:] - g[:-2])
    return w


def z_extents(params: HestonParams, eta: float, t_nodes: np.ndarray,
              cir_quantile: float, v_floor: float):
    """[v_min, v_max] enveloping the CIR transition law over all t>0 nodes, unioned with {v0, theta}.

    CIR(non-central chi-square): v_t = c * X, X ~ ncx2(df=d, nc=lam),
        c   = sigma_eff^2 (1 - e^{-kappa t}) / (4 kappa)
        d   = 4 kappa theta / sigma_eff^2
        lam = (4 kappa e^{-kappa t}) / (sigma_eff^2 (1 - e^{-kappa t})) * v0
    """
    sig_eff2 = (eta * params.sigma) ** 2
    if sig_eff2 <= 0:
        raise ValidationError("z_extents requires eta*sigma > 0 (use the deterministic branch)")
    kappa, theta, v0 = params.kappa, params.theta, params.v0
    d = 4.0 * kappa * theta / sig_eff2
    lows, highs = [], []
    for t in np.asarray(t_nodes, dtype=float):
        if t <= 0.0:
            continue                                   # t=0 is a point mass at v0 (anchor covers it)
        c = sig_eff2 * (1.0 - np.exp(-kappa * t)) / (4.0 * kappa)
        lam = (4.0 * kappa * np.exp(-kappa * t)) / (sig_eff2 * (1.0 - np.exp(-kappa * t))) * v0
        lows.append(c * ncx2.ppf(cir_quantile, d, lam))
        highs.append(c * ncx2.ppf(1.0 - cir_quantile, d, lam))
    q_low = max(v_floor, min(lows)) if lows else v_floor   # floor applies to CIR quantile only
    v_min = min(v0, theta, q_low)
    v_max = max(v0, theta, max(highs)) if highs else max(v0, theta)
    if not (np.isfinite(v_min) and np.isfinite(v_max) and 0.0 < v_min < v_max):
        raise ValidationError("degenerate z-extents (check params / quantile)")
    return float(v_min), float(v_max)


def x_extents(s0: float, b_fwd: np.ndarray, step_dt: np.ndarray, vbar2: float, x_span_stds: float):
    """[x_min, x_max] = ln s0 + drift envelope +/- W_x, W_x = x_span_stds*sqrt(vbar2 * T).

    b_fwd[i] is the per-step forward log-drift (cost-of-carry minus 0.5*vbar2 contribution).
    """
    dt = np.asarray(step_dt, dtype=float)
    b = np.asarray(b_fwd, dtype=float)
    T = float(dt.sum())
    cum = np.concatenate([[0.0], np.cumsum(b * dt)])     # forward log-drift envelope at each node
    w_x = x_span_stds * np.sqrt(max(vbar2, 1e-12) * max(T, 1e-12))
    x0 = np.log(s0)
    x_min = x0 + float(cum.min()) - w_x
    x_max = x0 + float(cum.max()) + w_x
    return x_min, x_max
