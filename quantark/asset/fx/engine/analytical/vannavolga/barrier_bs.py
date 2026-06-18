"""
Black-Scholes barrier building blocks for Vanna-Volga: Reiner-Rubinstein
one-touch / no-touch probabilities and single-barrier survival probabilities.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def _mu_from_b(b: float, vol: float) -> float:
    return (b - 0.5 * vol * vol) / (vol * vol)


def one_touch_hit_prob(
    spot: float,
    barrier: float,
    vol: float,
    tau: float,
    b: float,
    is_up: bool,
) -> float:
    """Probability that a GBM with drift ``b`` hits the barrier before T.

    Reiner-Rubinstein expiry-pay cash-or-nothing one-touch probability.
    """
    if tau <= 0.0:
        return (
            1.0
            if (is_up and spot >= barrier) or ((not is_up) and spot <= barrier)
            else 0.0
        )
    if vol <= 0.0:
        # Deterministic drift path S(t) = spot * exp(b*t) is monotonic in t, so
        # its extreme over [0, tau] is at an endpoint. The barrier is touched
        # iff that extreme reaches it.
        s_T = spot * math.exp(b * tau)
        if is_up:
            hit = max(spot, s_T) >= barrier
        else:
            hit = min(spot, s_T) <= barrier
        return 1.0 if hit else 0.0

    mu = _mu_from_b(b, vol)
    sqrt_tau = math.sqrt(tau)
    log_H_over_S = math.log(barrier / spot)

    x2 = math.log(spot / barrier) / (vol * sqrt_tau) + (1.0 + mu) * vol * sqrt_tau
    y2 = math.log(barrier / spot) / (vol * sqrt_tau) + (1.0 + mu) * vol * sqrt_tau

    if is_up:
        phi = 1.0
        eta = -1.0
    else:
        phi = -1.0
        eta = 1.0

    term1 = norm.cdf(phi * x2 - phi * vol * sqrt_tau)
    # term2 = (H/S)^(2 mu) * N(...). For small vol with strong carry the power
    # overflows before the vanishing CDF can offset it, so evaluate the product
    # in log space (the CDF tail is handled stably by ``norm.logcdf``).
    log_term2 = 2.0 * mu * log_H_over_S + float(norm.logcdf(eta * y2 - eta * vol * sqrt_tau))
    term2 = math.exp(log_term2) if log_term2 < 700.0 else float("inf")
    p_hit = float(term1 + term2)
    return max(0.0, min(1.0, p_hit))


def no_touch_price(
    spot: float,
    barrier: float,
    vol: float,
    tau: float,
    r_discount: float,
    b_drift: float,
    is_up: bool,
) -> float:
    """Single-barrier no-touch paying 1 at expiry if the barrier is not hit."""
    df = math.exp(-r_discount * tau)
    p_hit = one_touch_hit_prob(spot, barrier, vol, tau, b_drift, is_up)
    return df * (1.0 - p_hit)


def survival_probability_single(
    spot: float,
    barrier: float,
    rd: float,
    rf: float,
    vol: float,
    tau: float,
    is_up: bool,
) -> float:
    """Single-barrier survival probability: average of domestic/foreign measures.

    p_surv^d = NT^d / DF_d with b = rd - rf
    p_surv^f = NT^f / DF_f with b = rd - rf + sigma^2
    """
    if tau <= 0.0:
        # At expiry there is no remaining time to hit: survival is 0 only if the
        # barrier is already breached, otherwise 1.
        breached = (is_up and spot >= barrier) or ((not is_up) and spot <= barrier)
        return 0.0 if breached else 1.0

    nt_d = no_touch_price(
        spot=spot, barrier=barrier, vol=vol, tau=tau,
        r_discount=rd, b_drift=rd - rf, is_up=is_up,
    )
    df_d = math.exp(-rd * tau)
    p_surv_d = nt_d / df_d

    nt_f = no_touch_price(
        spot=spot, barrier=barrier, vol=vol, tau=tau,
        r_discount=rf, b_drift=(rd - rf + vol * vol), is_up=is_up,
    )
    df_f = math.exp(-rf * tau)
    p_surv_f = nt_f / df_f

    return max(0.0, min(1.0, 0.5 * (p_surv_d + p_surv_f)))


def reiner_rubinstein_barrier(
    spot: float,
    strike: float,
    barrier: float,
    vol: float,
    tau: float,
    rd: float,
    rf: float,
    is_up: bool,
    is_call: bool,
    knock_in: bool,
    rebate: float = 0.0,
    rebate_at_hit: bool = False,
) -> float:
    """Reiner-Rubinstein continuously-monitored single-barrier option value.

    Black-Scholes/Garman-Kohlhagen baseline (cost of carry b = rd - rf,
    domestic discounting r = rd). Covers all 8 KO/KI types via the standard
    A-F term decomposition with sign parameters phi (call/put) and eta
    (barrier side). Rebate: for KO paid at hit (rebate_at_hit) or at expiry;
    for KI paid at expiry if never knocked in.

    Reference: Haug, The Complete Guide to Option Pricing Formulas, 2nd ed.,
    single-barrier chapter.
    """
    if tau < 0.0:
        raise ValueError(
            f"reiner_rubinstein_barrier requires tau >= 0, got {tau}."
        )
    if tau == 0.0:
        # No remaining time: knock-in cannot trigger; knock-out is the vanilla
        # unless already breached. Handle terminal value directly.
        intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
        breached = (is_up and spot >= barrier) or ((not is_up) and spot <= barrier)
        if knock_in:
            # Never knocked in over [0, T]: pay the expiry rebate; else the
            # option is alive and worth its intrinsic value.
            return intrinsic if breached else rebate
        # knock-out: if breached it is dead (rebate already due, at expiry now);
        # otherwise it survived and pays intrinsic.
        return rebate if breached else intrinsic
    if vol <= 0.0:
        raise ValueError(
            "reiner_rubinstein_barrier requires vol > 0; the zero-vol "
            "deterministic limit is not implemented (would need a separate "
            "monotonic-path treatment)."
        )

    phi = 1.0 if is_call else -1.0
    eta = 1.0 if not is_up else -1.0  # +1 down-barrier, -1 up-barrier

    b = rd - rf
    r = rd
    sqrt_t = math.sqrt(tau)
    vst = vol * sqrt_t
    mu = (b - 0.5 * vol * vol) / (vol * vol)

    S, X, H, K = spot, strike, barrier, rebate
    carry_df = math.exp((b - r) * tau)  # e^{(b-r)T}
    dom_df = math.exp(-r * tau)

    x1 = math.log(S / X) / vst + (1.0 + mu) * vst
    x2 = math.log(S / H) / vst + (1.0 + mu) * vst
    y1 = math.log(H * H / (S * X)) / vst + (1.0 + mu) * vst
    y2 = math.log(H / S) / vst + (1.0 + mu) * vst

    HS = H / S

    # In this parameterization x1 is the standard d1, so the A term is exactly
    # the plain (unbarriered) vanilla value.
    A = phi * S * carry_df * norm.cdf(phi * x1) - phi * X * dom_df * norm.cdf(phi * x1 - phi * vst)

    # Spot already on the dead/alive side of the barrier (continuous monitoring):
    # the closed-form A-F decomposition assumes spot has NOT yet touched, so the
    # already-triggered state must be handled before it.
    breached = (is_up and S >= H) or ((not is_up) and S <= H)
    if breached:
        if knock_in:
            # Already knocked in -> the option is a plain vanilla now.
            return float(A)
        # Already knocked out -> only the rebate remains (paid now at hit, or
        # discounted to expiry).
        return float(K if rebate_at_hit else K * dom_df)

    B = phi * S * carry_df * norm.cdf(phi * x2) - phi * X * dom_df * norm.cdf(phi * x2 - phi * vst)
    C = (
        phi * S * carry_df * (HS ** (2.0 * (mu + 1.0))) * norm.cdf(eta * y1)
        - phi * X * dom_df * (HS ** (2.0 * mu)) * norm.cdf(eta * y1 - eta * vst)
    )
    D = (
        phi * S * carry_df * (HS ** (2.0 * (mu + 1.0))) * norm.cdf(eta * y2)
        - phi * X * dom_df * (HS ** (2.0 * mu)) * norm.cdf(eta * y2 - eta * vst)
    )
    # Rebate paid at expiry (used by KI, paid if never knocked in):
    E = K * dom_df * (
        norm.cdf(eta * x2 - eta * vst) - (HS ** (2.0 * mu)) * norm.cdf(eta * y2 - eta * vst)
    )

    strike_above_barrier = X >= H

    if knock_in:
        # In-options: rebate E (paid at expiry if not knocked in).
        if is_call and not is_up:        # down-and-in call
            val = (C + E) if strike_above_barrier else (A - B + D + E)
        elif is_call and is_up:          # up-and-in call
            val = (A + E) if strike_above_barrier else (B - C + D + E)
        elif (not is_call) and not is_up:  # down-and-in put
            val = (B - C + D + E) if strike_above_barrier else (A + E)
        else:                            # up-and-in put
            val = (A - B + D + E) if strike_above_barrier else (C + E)
    else:
        # Out-options: the rebate is paid because the barrier IS knocked out.
        # At hit -> F (the touch term with lambda). At expiry -> the rebate
        # discounted times the touch probability. Do NOT reuse E here: E is the
        # knock-in "never touched" term and pays on the opposite states.
        if rebate_at_hit and K != 0.0:
            # lam/z/F are only well-defined (and only needed) for a nonzero
            # at-hit rebate; mu^2 + 2r/vol^2 can go negative for some valid
            # negative-rate configs, so compute them lazily here, not above.
            radicand = mu * mu + 2.0 * r / (vol * vol)
            if radicand < 0.0:
                # The closed-form at-hit rebate term F has no real-lambda
                # representation here (it would need an oscillatory/complex
                # treatment). Reject explicitly rather than raising an opaque
                # math-domain error or silently approximating.
                raise ValueError(
                    "at-hit rebate (rebate_at_hit=True) is not supported for "
                    "this rate/vol configuration: mu^2 + 2*rd/vol^2 < 0 "
                    f"(rd={rd}, rf={rf}, vol={vol}). Use rebate_at_hit=False "
                    "(expiry-paid rebate) for these rates."
                )
            lam = math.sqrt(radicand)
            z = math.log(H / S) / vst + lam * vst
            reb = K * (
                (HS ** (mu + lam)) * norm.cdf(eta * z)
                + (HS ** (mu - lam)) * norm.cdf(eta * z - 2.0 * eta * lam * vst)
            )
        elif K != 0.0:
            p_hit = one_touch_hit_prob(S, H, vol, tau, b, is_up)
            reb = K * dom_df * p_hit
        else:
            reb = 0.0
        if is_call and not is_up:        # down-and-out call
            val = (A - C + reb) if strike_above_barrier else (B - D + reb)
        elif is_call and is_up:          # up-and-out call
            val = reb if strike_above_barrier else (A - B + C - D + reb)
        elif (not is_call) and not is_up:  # down-and-out put
            val = (A - B + C - D + reb) if strike_above_barrier else reb
        else:                            # up-and-out put
            val = (B - D + reb) if strike_above_barrier else (A - C + reb)

    return float(val)


__all__ = [
    "one_touch_hit_prob",
    "no_touch_price",
    "survival_probability_single",
    "reiner_rubinstein_barrier",
]
