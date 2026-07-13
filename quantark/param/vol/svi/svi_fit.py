"""Raw-SVI slice fit (spec WP4.2).

w(y) = a + b * (rho * (y - m) + sqrt((y - m)^2 + sigma^2)) in forward
log-moneyness y, per expiry. Constraints (all enforced in the optimizer):
b >= 0, |rho| < 1, sigma > 0, a + b*sigma*sqrt(1 - rho^2) >= 0 (slice
positivity), and the Lee wing bound b * (1 + |rho|) <= 2 as an explicit
penalty + post-fit verification (the base constraints alone do not imply
it). Butterfly no-arb via the Gatheral g(y) function; policy: one
constrained refit with a butterfly penalty, then fail loudly.

Deterministic: fixed multi-start initial points, no RNG.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from quantark.util.exceptions import NumericalError, ValidationError

LEE_WING_BOUND = 2.0          # max total-variance wing slope (Lee moment bound)
BUTTERFLY_TOL = -1e-8         # min allowed g(y) on the dense grid (spec WP4.2)
DENSE_Y = np.arange(-1.5, 1.5 + 1e-12, 0.01)
_POSITIVITY_PENALTY = 1e6
_LEE_PENALTY = 1e6
_BUTTERFLY_PENALTY = 1e5


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, y):
        y = np.asarray(y, dtype=float)
        d = y - self.m
        return self.a + self.b * (
            self.rho * d + np.sqrt(d * d + self.sigma * self.sigma)
        )

    def _w_derivatives(self, y):
        y = np.asarray(y, dtype=float)
        d = y - self.m
        root = np.sqrt(d * d + self.sigma * self.sigma)
        w = self.a + self.b * (self.rho * d + root)
        w1 = self.b * (self.rho + d / root)
        w2 = self.b * self.sigma * self.sigma / (root ** 3)
        return w, w1, w2

    def g(self, y):
        """Gatheral butterfly function; g(y) >= 0 <=> no butterfly arb."""
        w, w1, w2 = self._w_derivatives(y)
        y = np.asarray(y, dtype=float)
        term = 1.0 - y * w1 / (2.0 * w)
        return term * term - (w1 * w1 / 4.0) * (1.0 / w + 0.25) + w2 / 2.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class SVISliceFit:
    params: SVIParams
    expiry_t: float
    rmse: float
    n_quotes: int
    refit_applied: bool

    def to_dict(self) -> dict:
        return {
            "params": self.params.to_dict(),
            "expiry_t": self.expiry_t,
            "rmse": self.rmse,
            "n_quotes": self.n_quotes,
            "refit_applied": self.refit_applied,
        }


def _objective(x, y, w_target, weights, butterfly_penalty: bool):
    a, b, rho, m, sig = x
    p = SVIParams(a=a, b=b, rho=rho, m=m, sigma=sig)
    resid = p.total_variance(y) - w_target
    sse = float(np.sum(weights * resid * resid))
    # slice positivity: min w = a + b*sigma*sqrt(1-rho^2) must be >= 0
    min_w = a + b * sig * np.sqrt(max(1.0 - rho * rho, 0.0))
    sse += _POSITIVITY_PENALTY * max(0.0, -min_w) ** 2
    # Lee wing bound in total-variance units
    sse += _LEE_PENALTY * max(0.0, b * (1.0 + abs(rho)) - LEE_WING_BOUND) ** 2
    if butterfly_penalty:
        gv = p.g(DENSE_Y)
        neg = np.minimum(gv, 0.0)
        sse += _BUTTERFLY_PENALTY * float(np.sum(neg * neg))
    return sse


def fit_svi_slice(y, w, weights, expiry_t: float) -> SVISliceFit:
    """Weighted least-squares raw-SVI fit of one total-variance slice."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if y.size != w.size or y.size < 5:
        raise ValidationError("need >= 5 (y, w) points of equal length")
    if np.any(w <= 0.0):
        raise ValidationError("total variances must be positive")
    weights = (
        np.ones_like(w) if weights is None else np.asarray(weights, dtype=float)
    )
    weights = weights / float(weights.sum())

    w_min, w_max = float(w.min()), float(w.max())
    bounds = [
        (-1.0, 4.0 * w_max),          # a
        (0.0, LEE_WING_BOUND),        # b
        (-0.999, 0.999),              # rho
        (float(y.min()) - 1.0, float(y.max()) + 1.0),  # m
        (1e-4, 5.0),                  # sigma
    ]
    starts = [
        (0.8 * w_min, b0, rho0, float(y[np.argmin(w)]), 0.2)
        for b0 in (0.1, 0.4)
        for rho0 in (-0.5, 0.0)
    ]

    def _solve(butterfly_penalty: bool) -> SVIParams:
        best = None
        for x0 in starts:
            res = minimize(
                _objective,
                x0=np.asarray(x0, dtype=float),
                args=(y, w, weights, butterfly_penalty),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-16, "gtol": 1e-12},
            )
            if best is None or res.fun < best.fun:
                best = res
        return SVIParams(*[float(v) for v in best.x])

    params = _solve(butterfly_penalty=False)
    refit_applied = False
    if float(np.min(params.g(DENSE_Y))) < BUTTERFLY_TOL:
        params = _solve(butterfly_penalty=True)  # one constrained refit
        refit_applied = True
        min_g = float(np.min(params.g(DENSE_Y)))
        if min_g < BUTTERFLY_TOL:
            raise NumericalError(
                f"SVI slice at T={expiry_t:g} violates butterfly no-arb "
                f"after constrained refit (min g = {min_g:.3e})"
            )
    if params.b * (1.0 + abs(params.rho)) > LEE_WING_BOUND + 1e-9:
        raise NumericalError(
            f"SVI slice at T={expiry_t:g} violates the Lee wing bound: "
            f"b(1+|rho|) = {params.b * (1 + abs(params.rho)):.4f} > 2"
        )
    resid = params.total_variance(y) - w
    rmse = float(np.sqrt(np.mean(resid * resid)))
    return SVISliceFit(
        params=params,
        expiry_t=float(expiry_t),
        rmse=rmse,
        n_quotes=int(y.size),
        refit_applied=refit_applied,
    )
