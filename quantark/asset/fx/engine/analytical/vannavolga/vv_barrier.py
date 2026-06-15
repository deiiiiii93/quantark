"""
Vanna-Volga one-touch barrier pricing entrypoint.

Assembles the VV-corrected one-touch price from:
  1. the Black-Scholes baseline (BSTV) one-touch price at the ATM vol,
  2. numeric greeks (vega/vanna/volga) of that one-touch price,
  3. the vanilla Vanna-Volga Omega weights (``compute_omega``), and
  4. an attenuation factor (survival probability or first-exit-time) mapped to
     piecewise-linear vanna/volga weights.

VV price = BSTV + p_vanna * vanna * Omega[vanna] + p_volga * volga * Omega[volga]
(the vega term is dropped by construction).

Scope note: the legacy source implements VV pricing only for the one-touch
instrument; it provides no Black-Scholes vanilla knock-out pricer. Vanilla
knock-out VV pricing (strike + call/put, using the ``BarrierPrices`` arbitrage
clamps) is therefore intentionally deferred rather than fabricated here.
# TODO(vanilla-KO): add a Reiner-Rubinstein FX knock-out pricer + VV correction
# and wire enforce_single/double_barrier_arbitrage when that work is scheduled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np

from quantark.param.vol.vannavolga import (
    DeltaConvention,
    FXEnv,
    SmileQuotes,
    choose_delta_convention,
    compute_omega,
)
from quantark.util.exceptions import ValidationError

from .attenuation import gamma_fet, gamma_surv, p_vanna_p_volga_from_gamma
from .barrier_bs import one_touch_hit_prob

# Finite-difference bump sizes for one-touch greeks (model FD steps).
_H_SIGMA = 5e-4
_H_SPOT_REL = 1e-4


class BarrierGamma(str, Enum):
    """Attenuation measure for the VV barrier correction."""

    SURV = "surv"  # survival probability
    FET = "fet"  # expected first-exit time


# Piecewise-linear (a, b, c) presets per attenuation measure.
_GAMMA_PRESETS: Dict[BarrierGamma, tuple] = {
    BarrierGamma.SURV: (1.0, 0.5, 0.5),
    BarrierGamma.FET: (1.0, 0.0, 1.0),
}


@dataclass(frozen=True)
class VVBarrierResult:
    bstv: float
    vv: float
    gamma: float
    p_vanna: float
    p_volga: float
    omega: np.ndarray
    greeks: Dict[str, float]


def price_ot_bstv(env: FXEnv, sigma: float, barrier: float, is_up: bool) -> float:
    """Black-Scholes one-touch value: DF_dom * P_hit (expiry-pay one-touch)."""
    p_hit = one_touch_hit_prob(env.spot, barrier, sigma, env.tau, env.rd - env.rf, is_up=is_up)
    return float(math.exp(-env.rd * env.tau) * p_hit)


def numeric_greeks_ot(env: FXEnv, sigma: float, barrier: float, is_up: bool) -> Dict[str, float]:
    """Numeric vega/vanna/volga of the one-touch price via finite differences."""
    # Keep the central-difference bump strictly inside (0, sigma) so the lower
    # leg never crosses into the deterministic (vol<=0) branch, which would
    # corrupt the derivative estimate for very small vols.
    h_sig = min(_H_SIGMA, 0.5 * sigma) if sigma > 0.0 else _H_SIGMA

    def f_sig(s: float) -> float:
        return price_ot_bstv(env, s, barrier, is_up)

    vega = (f_sig(sigma + h_sig) - f_sig(sigma - h_sig)) / (2.0 * h_sig)

    h_S = max(1e-6, env.spot * _H_SPOT_REL)

    def vega_wrt_S(S: float) -> float:
        env2 = FXEnv(spot=S, rd=env.rd, rf=env.rf, tau=env.tau)
        return (
            price_ot_bstv(env2, sigma + h_sig, barrier, is_up)
            - price_ot_bstv(env2, sigma - h_sig, barrier, is_up)
        ) / (2.0 * h_sig)

    vanna = (vega_wrt_S(env.spot + h_S) - vega_wrt_S(env.spot - h_S)) / (2.0 * h_S)
    volga = (f_sig(sigma + h_sig) - 2.0 * f_sig(sigma) + f_sig(sigma - h_sig)) / (h_sig**2)

    return {"vega": float(vega), "vanna": float(vanna), "volga": float(volga)}


def price_vv_one_touch(
    env: FXEnv,
    quotes: SmileQuotes,
    barrier: float,
    is_up: bool,
    conv: Optional[DeltaConvention] = None,
    gamma_type: BarrierGamma = BarrierGamma.SURV,
    fet_method: str = "pde",
    gamma_star: float = 0.95,
    premium_included_atm: bool = False,
) -> VVBarrierResult:
    """Vanna-Volga corrected one-touch price.

    Args:
        env: FX market snapshot.
        quotes: ATM/RR/BF smile quotes.
        barrier: Absolute barrier level.
        is_up: True for an up-barrier, False for a down-barrier.
        conv: Delta convention (defaults to the maturity-based baseline).
        gamma_type: Attenuation measure (survival or first-exit-time).
        fet_method: 'pde' (default) or 'mc' when ``gamma_type`` is FET.
        gamma_star: Piecewise-linear transition threshold.
        premium_included_atm: Whether the ATM strike is premium-included.

    Raises:
        ValidationError: For non-positive barriers or unknown gamma types.
    """
    if barrier <= 0.0:
        raise ValidationError(f"barrier must be positive, got {barrier}")
    if env.tau < 0.0:
        raise ValidationError(f"time to expiry must be non-negative, got {env.tau}")
    gamma_type = BarrierGamma(gamma_type)
    sigma = quotes.sigma_atm

    # A matured one-touch settles its immediate expiry payoff; there is no smile
    # to calibrate against, so return before any 25-delta strike solving.
    if env.tau == 0.0:
        x_bs = price_ot_bstv(env, sigma, barrier, is_up)
        return VVBarrierResult(
            bstv=x_bs,
            vv=x_bs,
            gamma=0.0,
            p_vanna=0.0,
            p_volga=0.0,
            omega=np.zeros(3),
            greeks={"vega": 0.0, "vanna": 0.0, "volga": 0.0},
        )

    conv = DeltaConvention(conv) if conv is not None else choose_delta_convention(env.tau)
    x_bs = price_ot_bstv(env, sigma, barrier, is_up)
    gx = numeric_greeks_ot(env, sigma, barrier, is_up)
    omega, _ = compute_omega(env, quotes, conv, premium_included_atm=premium_included_atm)

    barrier_low = None if is_up else barrier
    barrier_high = barrier if is_up else None
    if gamma_type is BarrierGamma.SURV:
        g = gamma_surv(env, barrier_low, barrier_high, sigma)
    else:
        g = gamma_fet(env, barrier_low, barrier_high, sigma, method=fet_method)

    a, b, c = _GAMMA_PRESETS[gamma_type]
    p_vanna, p_volga = p_vanna_p_volga_from_gamma(g, a, b, c, gamma_star)

    adj = p_vanna * gx["vanna"] * float(omega[1]) + p_volga * gx["volga"] * float(omega[2])
    # A one-touch pays at most one unit at expiry, so its value is bounded by
    # [0, DF_dom]. Large VV corrections can push the raw approximation outside
    # this no-arbitrage range; clamp to the bound.
    df_dom = math.exp(-env.rd * env.tau)
    x_vv = min(max(x_bs + adj, 0.0), df_dom)

    return VVBarrierResult(
        bstv=x_bs,
        vv=x_vv,
        gamma=g,
        p_vanna=p_vanna,
        p_volga=p_volga,
        omega=omega,
        greeks=gx,
    )


__all__ = [
    "BarrierGamma",
    "VVBarrierResult",
    "price_ot_bstv",
    "numeric_greeks_ot",
    "price_vv_one_touch",
]
