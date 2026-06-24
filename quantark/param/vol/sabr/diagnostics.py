"""No-arbitrage diagnostics for SABR-parameterized implied-vol slices.

Butterfly (static, per maturity): the risk-neutral density g(K)=d2C/dK2 implied
by the Black call prices off the surface must be non-negative.
Calendar (across maturities): total variance w(K,T)=sigma(K,T)^2 * T must be
non-decreasing in T at every strike.

These checks are diagnostic. They do NOT change pricing. Interpolating raw Hagan
parameters across pillars is convenient but not guaranteed arbitrage-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.stats import norm

from quantark.util.numerical import safe_log, safe_sqrt


@dataclass(frozen=True)
class ArbitrageReport:
    butterfly_ok: bool
    calendar_ok: bool
    min_density: float
    min_total_variance_gap: float
    messages: List[str]


def _bs_call(F: float, K: float, T: float, sigma: float) -> float:
    """Undiscounted Black call on the forward (numeraire-free, for density only)."""
    if sigma <= 0.0 or T <= 0.0:
        return max(F - K, 0.0)
    v = float(sigma) * float(safe_sqrt(T))
    d1 = (float(safe_log(F / K)) + 0.5 * v * v) / v
    d2 = d1 - v
    return float(F * norm.cdf(d1) - K * norm.cdf(d2))


def butterfly_density(surface, T: float, strikes, spot: float) -> np.ndarray:
    """Risk-neutral density g(K)=d2C/dK2 by centred differences (>= 0 if arb-free)."""
    strikes = np.asarray(list(strikes), dtype=float)
    calls = np.array([
        _bs_call(spot, float(k), float(T), float(surface.get_vol(float(k), float(T), spot)))
        for k in strikes
    ])
    g = (calls[2:] - 2.0 * calls[1:-1] + calls[:-2]) / np.diff(strikes)[:-1] ** 2
    return g


def check_arbitrage(surface, strikes, maturities, spot: float) -> ArbitrageReport:
    strikes = np.asarray(list(strikes), dtype=float)
    maturities = sorted(float(t) for t in maturities)
    messages: List[str] = []

    min_density = float("inf")
    for T in maturities:
        g = butterfly_density(surface, T, strikes, spot)
        if g.size:
            min_density = min(min_density, float(np.min(g)))
    butterfly_ok = min_density >= -1e-8
    if not butterfly_ok:
        messages.append(f"Butterfly arbitrage: min density {min_density:.3e} < 0")

    if len(maturities) >= 2:
        min_gap = float("inf")
        for k in strikes:
            w = [surface.get_vol(float(k), T, spot) ** 2 * T for T in maturities]
            gaps = np.diff(w)
            min_gap = min(min_gap, float(np.min(gaps)))
        calendar_ok = min_gap >= -1e-10
        if not calendar_ok:
            messages.append(f"Calendar arbitrage: min total-variance gap {min_gap:.3e} < 0")
    else:
        calendar_ok = True
        min_gap = 0.0

    return ArbitrageReport(butterfly_ok, calendar_ok, min_density, min_gap, messages)
