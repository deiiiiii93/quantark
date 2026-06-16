"""
Vanna-Volga correction for vanilla single-barrier FX options.

VV price = BS_RR + p_vanna * vanna * Omega[vanna] + p_volga * volga * Omega[volga]
(vega term dropped by construction), survival-attenuated and clamped to the
no-arbitrage range [0, VV-vanilla] via enforce_single_barrier_arbitrage.

Reference: Castagna & Mercurio, "The Vanna-Volga method for implied
volatilities," Risk (2007); Wystup, FX Options and Structured Products.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from quantark.param.vol.vannavolga import (
    DeltaConvention,
    FXEnv,
    GKInput,
    SmileQuotes,
    choose_delta_convention,
    compute_omega,
    greeks_gk,
    price_gk,
    vv_adjustment_matrix,
)
from quantark.util.exceptions import ValidationError

from .arbitrage import BarrierPrices, enforce_single_barrier_arbitrage
from .attenuation import gamma_surv, p_vanna_p_volga_from_gamma
from .barrier_bs import reiner_rubinstein_barrier
from .vv_barrier import BarrierGamma, VVBarrierResult, _GAMMA_PRESETS

# FD bump sizes, consistent with the one-touch numeric greeks.
_H_SIGMA = 5e-4
_H_SPOT_REL = 1e-4


def numeric_greeks_barrier(
    env: FXEnv,
    sigma: float,
    strike: float,
    barrier: float,
    is_up: bool,
    is_call: bool,
    knock_in: bool,
    rebate: float,
    rebate_at_hit: bool,
) -> Dict[str, float]:
    """Numeric vega/vanna/volga of the RR barrier price via finite differences."""
    h_sig = min(_H_SIGMA, 0.5 * sigma) if sigma > 0.0 else _H_SIGMA

    def f(spot_: float, s_: float) -> float:
        return reiner_rubinstein_barrier(
            spot_, strike, barrier, s_, env.tau, env.rd, env.rf,
            is_up=is_up, is_call=is_call, knock_in=knock_in,
            rebate=rebate, rebate_at_hit=rebate_at_hit,
        )

    vega = (f(env.spot, sigma + h_sig) - f(env.spot, sigma - h_sig)) / (2.0 * h_sig)

    h_S = max(1e-6, env.spot * _H_SPOT_REL)

    def vega_wrt_S(spot_: float) -> float:
        return (f(spot_, sigma + h_sig) - f(spot_, sigma - h_sig)) / (2.0 * h_sig)

    vanna = (vega_wrt_S(env.spot + h_S) - vega_wrt_S(env.spot - h_S)) / (2.0 * h_S)
    volga = (
        f(env.spot, sigma + h_sig) - 2.0 * f(env.spot, sigma) + f(env.spot, sigma - h_sig)
    ) / (h_sig ** 2)
    return {"vega": float(vega), "vanna": float(vanna), "volga": float(volga)}


def _vv_vanilla(env: FXEnv, quotes: SmileQuotes, strike: float, is_call: bool,
                omega: np.ndarray) -> float:
    """Smile-consistent (VV-adjusted) vanilla price — the KO upper bound."""
    sigma = quotes.sigma_atm
    g = greeks_gk(is_call, GKInput(env.spot, strike, env.rd, env.rf, sigma, env.tau))
    base = price_gk(is_call, GKInput(env.spot, strike, env.rd, env.rf, sigma, env.tau))
    return base + vv_adjustment_matrix(g["vega"], g["vanna"], g["volga"], omega)


def price_vv_barrier(
    env: FXEnv,
    quotes: SmileQuotes,
    strike: float,
    barrier: float,
    is_up: bool,
    is_call: bool,
    knock_in: bool,
    rebate: float = 0.0,
    rebate_at_hit: bool = False,
    conv: Optional[DeltaConvention] = None,
    gamma_type: BarrierGamma = BarrierGamma.SURV,
    gamma_star: float = 0.95,
    premium_included_atm: bool = False,
) -> VVBarrierResult:
    """Vanna-Volga corrected vanilla single-barrier price."""
    if strike <= 0.0:
        raise ValidationError(f"strike must be positive, got {strike}")
    if barrier <= 0.0:
        raise ValidationError(f"barrier must be positive, got {barrier}")
    if env.tau < 0.0:
        raise ValidationError(f"time to expiry must be non-negative, got {env.tau}")
    gamma_type = BarrierGamma(gamma_type)
    sigma = quotes.sigma_atm

    bstv = reiner_rubinstein_barrier(
        env.spot, strike, barrier, sigma, env.tau, env.rd, env.rf,
        is_up=is_up, is_call=is_call, knock_in=knock_in,
        rebate=rebate, rebate_at_hit=rebate_at_hit,
    )

    if env.tau == 0.0:
        return VVBarrierResult(
            bstv=bstv, vv=bstv, gamma=0.0, p_vanna=0.0, p_volga=0.0,
            omega=np.zeros(3), greeks={"vega": 0.0, "vanna": 0.0, "volga": 0.0},
        )

    conv = DeltaConvention(conv) if conv is not None else choose_delta_convention(env.tau)
    omega, _ = compute_omega(env, quotes, conv, premium_included_atm=premium_included_atm)

    # Already-triggered states are plain instruments, not barriers: the
    # vanna/volga-only barrier correction (which drops the vanilla vega term) is
    # wrong for them. A touched knock-in is a vanilla -> return the FULL VV
    # vanilla price (vega included). A touched knock-out is dead -> only the
    # rebate remains, with no smile correction (bstv already holds it).
    breached = (is_up and env.spot >= barrier) or ((not is_up) and env.spot <= barrier)
    if breached:
        if knock_in:
            vanilla_vv = _vv_vanilla(env, quotes, strike, is_call, omega)
            # A touched KI is a live vanilla: report its actual vega/vanna/volga
            # (not zeros) so the diagnostics match the returned price, and full
            # vanna/volga weighting (gamma=1, survival certain) since the option
            # is fully knocked in.
            vg = greeks_gk(
                is_call, GKInput(env.spot, strike, env.rd, env.rf, sigma, env.tau)
            )
            result = VVBarrierResult(
                bstv=bstv, vv=float(vanilla_vv), gamma=1.0,
                p_vanna=1.0, p_volga=1.0, omega=omega,
                greeks={"vega": vg["vega"], "vanna": vg["vanna"], "volga": vg["volga"]},
            )
            object.__setattr__(result, "vanilla", float(vanilla_vv))
            return result
        # touched knock-out: dead, worth only the rebate already in bstv.
        result = VVBarrierResult(
            bstv=bstv, vv=float(bstv), gamma=0.0,
            p_vanna=0.0, p_volga=0.0, omega=omega,
            greeks={"vega": 0.0, "vanna": 0.0, "volga": 0.0},
        )
        object.__setattr__(result, "vanilla", float(bstv))
        return result

    gx = numeric_greeks_barrier(
        env, sigma, strike, barrier, is_up, is_call, knock_in, rebate, rebate_at_hit
    )

    barrier_low = None if is_up else barrier
    barrier_high = barrier if is_up else None
    g = gamma_surv(env, barrier_low, barrier_high, sigma)
    a, b, c = _GAMMA_PRESETS[gamma_type]
    p_vanna, p_volga = p_vanna_p_volga_from_gamma(g, a, b, c, gamma_star)

    adj = p_vanna * gx["vanna"] * float(omega[1]) + p_volga * gx["volga"] * float(omega[2])
    raw = bstv + adj

    vanilla_vv = _vv_vanilla(env, quotes, strike, is_call, omega)
    if rebate > 0.0:
        # A rebate is an extra cash leg on top of the option, so a barrier with
        # a rebate can legitimately be worth MORE than the plain vanilla. It is
        # still bounded: the rebate is paid at most once, so its PV is at most
        # rebate * max(1, DF_dom) (DF_dom can exceed 1 under negative rd). Clamp
        # to vanilla_vv plus that conservative max-rebate PV.
        dom_df = float(np.exp(-env.rd * env.tau))
        rebate_pv_max = rebate * max(1.0, dom_df)
        upper = vanilla_vv + rebate_pv_max
        vv = min(max(raw, 0.0), upper)
    else:
        clamped = enforce_single_barrier_arbitrage(
            BarrierPrices(vanilla=vanilla_vv, ko=raw)
        )
        vv = clamped.ko if clamped.ko is not None else max(raw, 0.0)

    result = VVBarrierResult(
        bstv=bstv, vv=float(vv), gamma=g, p_vanna=p_vanna, p_volga=p_volga,
        omega=omega, greeks=gx,
    )
    # Attach the vanilla bound for downstream tests/diagnostics.
    object.__setattr__(result, "vanilla", float(vanilla_vv))
    return result


__all__ = ["numeric_greeks_barrier", "price_vv_barrier"]
