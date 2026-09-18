"""Test-local controls around the independent Gaussian-transition solver.

The solver itself lives in ``quantark.modelvalidation.builders.intraday_gaussian`` (it is the
deterministic reference of the daily-KI intraday study). This module keeps the three-level
``reference_snowball`` wrapper the runtime tests compare against, and the closed forms that
check the solver and the analytical routes: none of them is a certification reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Tuple

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

from quantark.intraday.timestamp import calendar_year_fraction
from quantark.modelvalidation.builders.intraday_gaussian import (  # noqa: F401 - re-exported for the control tests
    PLJ, _SWEEP_CACHE, expect, moments, solve_snowball as _solve_snowball, splice, with_knots,
)


@dataclass(frozen=True)
class ReferenceResult:
    price: float
    delta: float
    gamma: float
    grid_points: int
    uncertainty_price: float
    uncertainty_delta: float
    uncertainty_gamma: float
    method: str = "dense_gaussian_backward"



def reference_snowball(ctx, *, points=(2001, 4001, 8001), width_std=8.0) -> ReferenceResult:
    """Price, delta, gamma of the remaining claim (excluding pending receivables) with Richardson uncertainty."""
    vals = [_solve_snowball(ctx, p, width_std) for p in points]
    (_, _, _), (p1, dl1, g1), (p2, dl2, g2) = vals

    def unc(a, b):
        return abs(b - a) / 3.0 if np.isfinite(a) and np.isfinite(b) else float("nan")

    return ReferenceResult(price=p2, delta=dl2, gamma=g2, grid_points=points[-1],
                           uncertainty_price=unc(p1, p2), uncertainty_delta=unc(dl1, dl2), uncertainty_gamma=unc(g1, g2))


# ---------------------------------------------------------------------------
# closed forms
def reference_digital(ctx) -> ReferenceResult:
    """Cash-or-nothing digital: N(d2) with W from the clock and R - Q from the curves; exact (no uncertainty)."""
    prod, env = ctx.numerical.product, ctx.pricing_env
    S, K, T = float(env.spot), float(prod.strike), float(ctx.numerical.maturity_tau)
    W = float(env.vol_surface.total_variance(K, T, S)) if T > 0.0 else 0.0
    R = -log(float(env.get_discount_factor(T))) if T > 0.0 else 0.0
    Q = float(env.get_div_yield(T)) * T if T > 0.0 else 0.0
    pay = calendar_year_fraction(ctx.valuation_timestamp, ctx.timeline.terminal().payment_timestamp)
    df_pay = float(env.get_discount_factor(pay)) if pay > 0.0 else 1.0
    payout = float(prod.payout) * float(getattr(prod, "contract_multiplier", 1.0))
    call = prod.option_type.name == "CALL"
    if W == 0.0:
        F = S * exp(R - Q)
        pays = F > K if call else F < K
        return ReferenceResult(payout * df_pay * (1.0 if pays else 0.0), float("nan"), float("nan"), 0, 0.0, 0.0, 0.0,
                               "closed_form_deterministic")
    sw = sqrt(W)
    d2 = (log(S / K) + (R - Q) - 0.5 * W) / sw
    sign = 1.0 if call else -1.0
    price = payout * df_pay * norm.cdf(sign * d2)
    delta = sign * payout * df_pay * norm.pdf(d2) / (S * sw)
    gamma = -delta * (1.0 + d2 / sw) / S
    return ReferenceResult(price, delta, gamma, 0, 0.0, 0.0, 0.0, "closed_form_bs_effective_variance")


def _band(s0: float, lo, hi, sw: float) -> Tuple[float, float]:
    """(P, E[S_T 1{.}]) for lo <= S_T < hi, ln S_T ~ N(ln s0 - sw^2/2, sw^2); None = unbounded."""
    def tail(level):
        if level is None:
            return 1.0, s0
        d = (log(s0 / level) - 0.5 * sw * sw) / sw
        return float(ndtr(d)), s0 * float(ndtr(d + sw))
    p_lo, e_lo = tail(lo)
    if hi is None:
        return p_lo, e_lo
    p_hi, e_hi = tail(hi)
    return p_lo - p_hi, e_lo - e_hi


def _payoff_band(s0: float, K: float, is_call: bool, lo, hi, sw: float) -> float:
    """E[vanilla payoff(S_T) 1{lo <= S_T < hi}] under zero carry."""
    if is_call:
        lo = K if lo is None else max(lo, K)
    else:
        hi = K if hi is None else min(hi, K)
    if lo is not None and hi is not None and hi <= lo:
        return 0.0
    p, e = _band(s0, lo, hi, sw)
    return e - K * p if is_call else K * p - e


def barrier_zero_carry(S: float, K: float, H: float, u: float, sigma: float, *, is_call: bool, is_up: bool,
                       is_knock_out: bool, rebate: float = 0.0) -> float:
    """Continuously monitored single barrier under r = q = 0, total variance sigma^2 u.

    Zero carry makes S a martingale with log drift -sigma^2/2 per unit variance
    time, and the reflection principle gives, for a payoff f supported beyond the
    barrier, E[f(S_T) 1{hit}] = (S/H) E^{S'}[f(S_T)] with S' = H^2/S (lambda = 1/2
    in Reiner-Rubinstein). Paths ending on the far side have hit for certain.
    A knock-out's rebate is paid on hit, a knock-in's at expiry if never hit;
    at r = 0 its timing is irrelevant.
    """
    if u <= 0.0:
        raise ValueError("variance time must be positive")
    sw = sigma * sqrt(u)
    vanilla = _payoff_band(S, K, is_call, None, None, sw)
    if (S >= H) if is_up else (S <= H):
        return rebate if is_knock_out else vanilla
    reflected = H * H / S
    if is_up:
        knock_in = _payoff_band(S, K, is_call, H, None, sw) + (S / H) * _payoff_band(reflected, K, is_call, None, H, sw)
        p_hit = _prob_hit_up(S, H, sw)
    else:
        knock_in = _payoff_band(S, K, is_call, None, H, sw) + (S / H) * _payoff_band(reflected, K, is_call, H, None, sw)
        p_hit = _prob_hit_down(S, H, sw)
    if is_knock_out:
        return vanilla - knock_in + rebate * p_hit
    return knock_in + rebate * (1.0 - p_hit)


def _prob_hit_up(S, H, sw):
    """P(max_t S_t >= H) for S < H under zero carry (log drift -sw^2/2 per unit variance time)."""
    return norm.cdf((log(S / H) - 0.5 * sw * sw) / sw) + (S / H) * norm.cdf((log(S / H) + 0.5 * sw * sw) / sw)


def _prob_hit_down(S, H, sw):
    """P(min_t S_t <= H) for S > H under zero carry."""
    return norm.cdf((log(H / S) + 0.5 * sw * sw) / sw) + (S / H) * norm.cdf((log(H / S) - 0.5 * sw * sw) / sw)
