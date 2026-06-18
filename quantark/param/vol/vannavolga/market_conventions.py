"""
FX market conventions for Vanna-Volga: delta conventions, ATM and 25-delta
strike solving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm

from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.numerical import Tolerance, is_zero


class DeltaConvention(str, Enum):
    """FX delta quoting conventions.

    Values are the legacy string codes so existing call sites / serialized data
    keep working (``DeltaConvention("spot") is DeltaConvention.SPOT``).
    """

    SPOT = "spot"  # Black-Scholes (Garman-Kohlhagen) spot delta
    SPOT_PREM = "spot_prem"  # premium-included spot delta
    FWD = "fwd"  # forward (driftless) delta
    FWD_PREM = "fwd_prem"  # premium-included forward delta


@dataclass(frozen=True)
class FXEnv:
    spot: float
    rd: float
    rf: float
    tau: float  # time to expiry in years

    @property
    def df_dom(self) -> float:
        return math.exp(-self.rd * self.tau)

    @property
    def df_for(self) -> float:
        return math.exp(-self.rf * self.tau)

    @property
    def forward(self) -> float:
        return self.spot * math.exp((self.rd - self.rf) * self.tau)


def _d1_d2(forward: float, strike: float, vol: float, tau: float) -> Tuple[float, float, float]:
    sqrt_tau = math.sqrt(max(tau, 0.0))
    if is_zero(sqrt_tau) or vol <= 0.0:
        # Deterministic limit: sign-aware infinities so deltas respect moneyness
        # (OTM call -> 0, ITM put -> negative), matching the GK helper.
        sign = float("inf") if forward >= strike else float("-inf")
        return (sign, sign, sqrt_tau)
    a1 = math.log(forward / strike) + 0.5 * (vol**2) * tau
    a2 = vol * sqrt_tau
    d1 = a1 / a2
    d2 = d1 - a2
    return d1, d2, sqrt_tau


def bs_delta(
    strike: float,
    is_call: bool,
    env: FXEnv,
    vol: float,
    conv: DeltaConvention = DeltaConvention.SPOT,
) -> float:
    """FX delta under various market conventions (quoted in Ccy1 units)."""
    conv = DeltaConvention(conv)
    F = env.forward
    d1, d2, _ = _d1_d2(F, strike, vol, env.tau)

    if conv is DeltaConvention.SPOT:
        base = env.df_for * norm.cdf(d1) if is_call else -env.df_for * norm.cdf(-d1)
        return float(base)
    if conv is DeltaConvention.SPOT_PREM:
        coef = (strike / env.spot) * env.df_dom
        base = coef * norm.cdf(d2) if is_call else -coef * norm.cdf(-d2)
        return float(base)
    if conv is DeltaConvention.FWD:
        base = norm.cdf(d1) if is_call else -norm.cdf(-d1)
        return float(base)
    if conv is DeltaConvention.FWD_PREM:
        coef = (strike / env.spot) * (env.df_dom / env.df_for)
        base = coef * norm.cdf(d2) if is_call else -coef * norm.cdf(-d2)
        return float(base)

    raise ValidationError(f"Unknown delta convention: {conv}")


def choose_delta_convention(tau: float) -> DeltaConvention:
    """Default baseline: spot for maturities <= 1y; forward for > 1y."""
    return DeltaConvention.SPOT if tau <= 1.0 + Tolerance.ZERO else DeltaConvention.FWD


def atm_strike(
    sigma_atm: float,
    env: FXEnv,
    premium_included: bool = False,
) -> float:
    """ATM delta-neutral strike.

    - premium-excluded (BS/forward delta): K_ATM = F * exp(+0.5 sigma^2 tau)
    - premium-included:                    K_ATM = F * exp(-0.5 sigma^2 tau)
    """
    F = env.forward
    sign = -1.0 if premium_included else 1.0
    return F * math.exp(sign * 0.5 * (sigma_atm**2) * env.tau)


def _solve_strike_for_delta_target(
    target_delta: float,
    is_call: bool,
    env: FXEnv,
    vol: float,
    conv: DeltaConvention,
    K_low: Optional[float] = None,
    K_high: Optional[float] = None,
    tol: float = Tolerance.LOG_MIN,
    max_iter: int = 100,
) -> float:
    """Robust bisection on strike to achieve a target delta under a convention.

    Raises:
        NumericalError: If a sign-changing bracket cannot be found (so the
            solver never silently returns the forward).
    """
    F = env.forward

    def f(K: float) -> float:
        return bs_delta(K, is_call, env, vol, conv) - target_delta

    # 25-delta strikes are out-of-the-money: OTM calls sit above the forward,
    # OTM puts below it. Anchoring one bracket endpoint at the forward and
    # expanding the other outward isolates the OTM root. This is essential for
    # premium-adjusted conventions, where call delta is hump-shaped and a
    # symmetric bracket would put both endpoints on the same (near-zero) side.
    span = 10.0 * max(vol * math.sqrt(max(env.tau, 0.0)), Tolerance.PRECISION)
    if K_low is None or K_high is None:
        if is_call:
            K_low, K_high = F, F * math.exp(span)
        else:
            K_low, K_high = F * math.exp(-span), F

    fl = f(K_low)
    fh = f(K_high)
    if np.sign(fl) == np.sign(fh):
        # Expand the OTM endpoint outward before giving up.
        for m in (3.0, 10.0, 30.0, 100.0):
            if is_call:
                K_low, K_high = F, F * m
            else:
                K_low, K_high = F / m, F
            fl = f(K_low)
            fh = f(K_high)
            if np.sign(fl) != np.sign(fh):
                break
        else:
            raise NumericalError(
                f"Failed to bracket a strike for target delta {target_delta} "
                f"(is_call={is_call}, conv={conv})"
            )

    K_mid = 0.5 * (K_low + K_high)
    for _ in range(max_iter):
        K_mid = 0.5 * (K_low + K_high)
        fm = f(K_mid)
        if abs(fm) < tol:
            return K_mid
        if np.sign(fm) == np.sign(fl):
            K_low, fl = K_mid, fm
        else:
            K_high, fh = K_mid, fm
        if abs(K_high - K_low) / max(K_mid, 1.0) < tol:
            return K_mid
    return K_mid


def strike_for_delta(
    target_delta: float,
    is_call: bool,
    env: FXEnv,
    vol: float,
    conv: DeltaConvention,
) -> float:
    """Solve the strike achieving ``target_delta`` at ``vol`` under ``conv``."""
    return _solve_strike_for_delta_target(
        target_delta, is_call, env, vol, DeltaConvention(conv)
    )


def strikes_25d(
    sigma_atm: float,
    env: FXEnv,
    conv: DeltaConvention,
) -> Tuple[float, float]:
    """Solve 25-delta put and call strikes at a single vol (Delta -0.25 / +0.25).

    Note: for smile calibration each wing strike should be solved with its own
    quoted wing vol; see ``strike_for_delta``.
    """
    conv = DeltaConvention(conv)
    k_put = _solve_strike_for_delta_target(-0.25, False, env, sigma_atm, conv)
    k_call = _solve_strike_for_delta_target(+0.25, True, env, sigma_atm, conv)
    return k_put, k_call


__all__ = [
    "DeltaConvention",
    "FXEnv",
    "bs_delta",
    "choose_delta_convention",
    "atm_strike",
    "strike_for_delta",
    "strikes_25d",
]
